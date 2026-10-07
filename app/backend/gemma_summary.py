"""HPI's prose contract plus source annotation/review by the unadapted model.

The adapter never sees JSON, evidence IDs, timestamps or previous summaries.
The base model selects sources for immutable paragraphs; it cannot rewrite them.
"""
from copy import deepcopy
from pathlib import Path
import re

import durable_jobs as durable
from processing_mode import policy
from source_contract import reviewed
from source_contract import SourceCatalog
from summary_grounding import Workflow, SummaryValidationError, digest, obj, arr, TEXT, EVIDENCE, SCOPES, SECTIONS, positive
from llm_transport import complete, fits, cache_key, cache_read, cache_write, ContextBudgetError, structured_output_budget

PROMPT = Path(__file__).with_name('prompt_gemma.txt').read_text(encoding='utf-8-sig').strip()
VERSION = 'gemma4-protokoll-v2'
ANNOTATIONS = obj({'paragraphs': arr(obj({'id': TEXT, 'section': {'enum': list(SECTIONS)},
    'scope': {'enum': list(SCOPES)}, 'evidence': EVIDENCE})), 'considered_source_ids': arr(TEXT)})


def transcript_turns(lines):
    """Merge consecutive turns only; unresolved names remain unchanged."""
    turns = []
    for line in lines:
        speaker, sep, text = line.partition(':')
        speaker = speaker.strip() if sep else 'SPEAKER_UNKNOWN'
        text = ' '.join((text if sep else line).split())
        if not text:
            continue
        if turns and turns[-1][0] == speaker:
            turns[-1] = (speaker, turns[-1][1] + ' ' + text)
        else:
            turns.append((speaker or 'SPEAKER_UNKNOWN', text))
    return '\n'.join(f'{speaker}: {text}' for speaker, text in turns)


def protocol_messages(title, lines):
    user = (f'Erstelle eine Zusammenfassung für folgenden Tagesordnungspunkt:\n\n'
            f'TOP: {title}\n\nTranskript:\n{transcript_turns(lines)}\n\nZusammenfassung:')
    # Matches the production Unsloth framing: system and user in one user turn.
    return [{'role': 'user', 'content': PROMPT + '\n\n' + user}]


def paragraphs(text):
    """Remove only layout headings; preserve every substantive output line."""
    text = re.sub(r'^```(?:markdown)?\s*\n|\n```\s*$', '', text.strip())
    blocks, current = [], []
    for line in text.splitlines():
        heading = line.strip().strip('#* ').rstrip(':').strip()
        is_heading = (re.fullmatch(r'Zu TOP\s+[^:]+:?.*', heading, re.I) is not None
                      or heading.casefold() in {'aus der beratung', 'beschlüsse und festlegungen',
                          'beschlüsse/festlegungen', 'beschlüsse', 'festlegungen'})
        if not line.strip() or is_heading:
            if current:
                blocks.append('\n'.join(current).strip())
                current = []
            # Do not drop prose placed after the TOP heading's colon.
            if is_heading and re.match(r'Zu TOP\s', heading, re.I) and ':' in line:
                remainder = line.split(':', 1)[1].strip()
                if remainder:
                    current.append(remainder)
        else:
            current.append(line.strip())
    if current:
        blocks.append('\n'.join(current).strip())
    if not blocks:
        raise SummaryValidationError('Protocol adapter returned no paragraphs')
    return blocks


class GemmaWorkflow(Workflow):
    def __init__(self, client, config, title, context, usage):
        super().__init__(client, config, 'Annotiere und prüfe den Wortlaut; keine stilistische Neufassung.\nTOP: ' + title,
                         context, usage)
        self.title = title
        self.prose_config = config.for_protocol()
        self.output = config.output_budget(positive('SUMMARY_OUTPUT_TOKENS', 4096))
        self.reserve = structured_output_budget(config, self.output)
        self.policy.update(output=self.output, protocol_version=VERSION, adapter=self.prose_config.public_snapshot(),
                           prompt_sha256=digest(PROMPT), protocol_code=digest(Path(__file__).read_text(encoding='utf-8')))

    def source_text(self, rows):
        grouped = {}
        for row in rows:
            grouped.setdefault(row['line_index'], []).append(row)
        result = []
        for index, parts in grouped.items():
            text = ''.join(part['text'] for part in parts)
            if parts[0]['start_char']:
                speaker, sep, _ = self.lines[index].partition(':')
                text = (speaker if sep else 'SPEAKER_UNKNOWN') + ': ' + text
            result.append(text)
        return result

    def generate(self, rows):
        messages = protocol_messages(self.title, self.source_text(rows))
        planning = self.messages('protocol_sources', 'Ordne Absätze unverändert Quellen zu.', dict(source=rows))
        if (not fits(messages, self.output, self.prose_config)
                or not fits(planning, self.reserve + self.output + 1024, self.config, self.format(ANNOTATIONS))):
            if len(rows) < 2:
                raise ContextBudgetError('Protocol source unit exceeds context budget')
            middle = len(rows) // 2
            return self.generate(rows[:middle]) + self.generate(rows[middle:])
        key = cache_key(self.prose_config, messages, VERSION, self.policy)
        step = 'protocol:' + digest([messages, self.policy])
        def operation():
            cached = cache_read(key)
            if cached is not None:
                paragraphs(cached)
                self.usage['cached_calls'] = self.usage.get('cached_calls', 0) + 1
                return cached
            durable.progress({'phase': 'summary_protocol'})
            self.usage['attempted_calls'] = self.usage.get('attempted_calls', 0) + 1
            response = complete(self.client, self.prose_config, messages=messages,
                                model=self.prose_config.model, max_tokens=self.output)
            text = response.choices[0].message.content
            paragraphs(text)
            if hasattr(response, 'llm_provenance'):
                self.usage.setdefault('requests', []).append(response.llm_provenance)
            durable.artifact(step, 'protocol_draft', {'text': text})
            cache_write(key, text)
            return text
        text = durable.checkpoint(step, operation)
        blocks = paragraphs(text)
        allowed = {row['source_id']: row for row in rows}
        # Keep the prose reviewable even if source annotation fails technically.
        prior = deepcopy(self.latest_claims)
        _, missing = SourceCatalog(rows, 'source_id').inspect([])
        self.latest_claims = prior + [dict(text=t, section='discussion', scope='unclear',
            evidence=[], grounding=deepcopy(missing)) for t in blocks]
        self.partial_rows.update(allowed)
        paragraph_ids = [f'P:{i}' for i in range(len(blocks))]
        def validate(data):
            self.coverage(data['considered_source_ids'], allowed)
            self.coverage([p['id'] for p in data['paragraphs']], paragraph_ids)
            for item in data['paragraphs']:
                self.evidence(item['evidence'], allowed)
        answer = self.call('protocol_sources',
            'Ordne JEDEM vorgegebenen Absatz seine Originalquellen, Kategorie und zeitlichen Bezug zu. '
            'Die Absätze sind ungeprüfte Modellvorschläge. Keine Aussage hinzuerfinden, um sie zu belegen. '
            'Bei fehlenden Belegen evidence leer lassen. considered_source_ids enthält alle gelesenen IDs.',
            dict(source=rows, paragraphs=[dict(id=i, text=t) for i, t in zip(paragraph_ids, blocks)]),
            ANNOTATIONS, validate)
        by_id = {item['id']: item for item in answer['paragraphs']}
        claims = [dict(text=text, **{k: v for k, v in by_id[identity].items() if k != 'id'})
                  for identity, text in zip(paragraph_ids, blocks)]
        self.draft_validator(rows)({'claims': claims, 'considered_source_ids': list(allowed)})
        self.latest_claims = prior + deepcopy(claims)
        return [(rows, claims)]

    def run(self, lines):
        self.lines = lines
        primary = self.generate(self.sources(lines))
        self.rows = [r for rows, _ in primary for r in rows]
        claims = [c for _, group in primary for c in group]
        issues = []
        if not policy().fast:
            # Two independent full-source reviews retain
            # omissions/questions. The LoRA wording is never rewritten by the base.
            for phase in ('final_review', 'consolidated_review'):
                offset = 0
                for rows, group in primary:
                    # Each draft is checked against its complete input. Keeping
                    # unrelated chunks out avoids overflowing a small context
                    # with the accumulated minutes of a very long TOP.
                    for issue in self.review(group, rows, phase):
                        issue = deepcopy(issue)
                        issue['claim_ids'] = [f'C:{offset + int(identity[2:])}'
                                              for identity in issue['claim_ids']]
                        issues.append(issue)
                    offset += len(group)
            for index, claim in enumerate(claims):
                relevant = [q for q in issues if not q['claim_ids'] or f'C:{index}' in q['claim_ids']]
                claim['grounding'] = reviewed(claim['grounding'], supported=not relevant,
                    questions=[q['question'] for q in relevant],
                    contradicted=any(q['kind'] == 'contradiction' for q in relevant))
        self.usage.update(**policy().snapshot(), processing_complete=True,
            review_complete=not policy().fast, review_status='skipped' if policy().fast else 'completed',
            review_required=policy().fast or bool(issues) or any(c['grounding']['evidence_status'] != 'exact' for c in claims),
            grounding_incomplete=False, source_line_count=len(lines), source_sha256=digest(lines),
            considered_source_ids=[r['source_id'] for r in self.rows], prompt_version=VERSION,
            policy=self.policy, required_checks=['protocol', 'protocol_sources'] +
                ([] if policy().fast else ['final_review', 'consolidated_review']),
            reconciliation_rounds=0, summary_style='gemma4-lora', protocol_chunks=len(primary))
        return claims, issues, self.rows, len(primary)


def render_protocol(structured):
    sections = [('Beschlüsse und Festlegungen', structured.decisions + structured.votes + structured.action_items),
                ('Aus der Beratung', structured.discussion), ('Offene Punkte', structured.open_points),
                ('Unsicherheiten', structured.uncertainties)]
    return '\n\n'.join(title + ':\n' + '\n\n'.join(items) for title, items in sections if items)
