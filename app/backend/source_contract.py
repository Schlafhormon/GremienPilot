"""Lossless source selection and draft evidence states; never a semantic oracle.

Only a separate original-source review may promote an exact reference to an
exactly supported claim. Unknown IDs are retained as diagnostics, never guessed.
"""
from copy import deepcopy
import hashlib
import json
import re

VERSION = 'graded-sources-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


class SourceCatalog:
    def __init__(self, rows, id_key='line_id'):
        self.id_key = id_key
        self.rows = {r[id_key]: r for r in rows}
        if len(self.rows) != len(rows) or any(not isinstance(r.get('text'), str) for r in rows):
            raise ValueError('invalid_original_sources')
        self.aliases = {f'L{i+1}': r[id_key] for i, r in enumerate(rows)}
        self.reverse = {value: key for key, value in self.aliases.items()}
        self.sha256 = digest(rows)

    def manifest(self):
        return {'contract': VERSION, 'source_sha256': self.sha256,
                'references': [{'ref': alias, 'original_id': identity,
                    'original_sha256': digest(self.rows[identity])}
                    for alias, identity in self.aliases.items()]}

    def alias_schema(self):
        """Bound native source references without listing the entire transcript.

        Disjoint decimal prefixes cover L1..Ln. Grammar size depends on the
        number of digits, not rows; authoritative lookup still validates replies.
        Use only anchored groups, alternatives and digit classes supported by
        the native JSON-schema grammar converter.
        """
        if not self.aliases:
            raise ValueError('no_original_sources')
        last = str(len(self.aliases))
        parts = ['[1-9]' + '[0-9]' * (size-1) for size in range(1, len(last))]
        for i, digit in enumerate(last):
            low, high = (1 if i == 0 else 0), int(digit)-1
            if low <= high:
                choice = str(low) if low == high else f'[{low}-{high}]'
                parts.append(last[:i] + choice + '[0-9]' * (len(last)-i-1))
        parts.append(last)
        return {'type': 'string', 'pattern': '^L(' + '|'.join(parts) + ')$'}

    def translate(self, value, *, decode=False):
        mapping = self.aliases if decode else self.reverse
        keys = {self.id_key, 'start_line_id', 'end_line_id'}
        if isinstance(value, list):
            return [self.translate(v, decode=decode) for v in value]
        if isinstance(value, dict):
            return {k: mapping.get(v, v) if k in keys and isinstance(v, str)
                    else self.translate(v, decode=decode) for k, v in value.items()}
        return value

    def inspect(self, evidence):
        if not isinstance(evidence, list) or any(not isinstance(e, dict) for e in evidence):
            raise ValueError('unreadable_evidence')
        valid, ranges, diagnostics = [], [], []
        for index, item in enumerate(evidence):
            identity = item.get(self.id_key)
            row = self.rows.get(identity)
            if row is None:
                diagnostics.append({'index': index, 'code': 'unknown_source_id', 'reference': item})
                continue
            ranges.append(identity)
            quote = item.get('quote')
            if quote is None:  # New contract: application copies the selected original.
                quote = row['text']
            if isinstance(quote, str) and quote.strip() and quote in row['text'] and row['text'].count(quote) == 1:
                valid.append({self.id_key: identity, 'quote': quote})
                continue
            # Whitespace-only changes are safe only with a unique original span.
            parts = re.split(r'\s+', quote.strip()) if isinstance(quote, str) else []
            matches = list(re.finditer(r'\s+'.join(re.escape(p) for p in parts), row['text'])) if parts else []
            if len(matches) == 1:
                valid.append({self.id_key: identity, 'quote': matches[0].group()})
                diagnostics.append({'index': index, 'code': 'format_copy_error', 'reference': item})
            else:
                diagnostics.append({'index': index, 'code': 'invalid_source_quote', 'reference': item})
        unresolved = [d for d in diagnostics if d['code'] != 'format_copy_error']
        reference_status = ('exact' if valid and not unresolved else 'source_range' if ranges else 'unsupported')
        return valid, {'contract': VERSION, 'reference_status': reference_status,
            'evidence_status': 'source_range' if ranges else 'unsupported',
            'content_status': 'unreviewed', 'source_ids': list(dict.fromkeys(ranges)),
            'diagnostics': diagnostics, 'questions': [] if reference_status == 'exact' else [
                'Welche Originalstelle stützt diese Aussage einschließlich Zahlen und Verneinungen?']}

    def prepare(self, value):
        value = deepcopy(value)
        def walk(node):
            if isinstance(node, list):
                for item in node:
                    walk(item)
            elif isinstance(node, dict):
                if 'evidence' in node:
                    node.pop('grounding', None)
                    node['evidence'], node['grounding'] = self.inspect(node['evidence'])
                for key, item in list(node.items()):
                    if key not in {'grounding', 'evidence'}:
                        walk(item)
        walk(value)
        return value


def reviewed(grounding, *, supported=False, questions=(), contradicted=False):
    result = deepcopy(grounding)
    result['content_status'] = 'contradicted' if contradicted else 'supported' if supported else 'unclear'
    result['questions'] = list(dict.fromkeys([*result.get('questions', []), *questions]))
    result['evidence_status'] = ('exact' if supported and result['reference_status'] == 'exact'
                                 and not result['questions'] else
                                 'source_range' if result.get('source_ids') else 'unsupported')
    return result


def marked_text(text, grounding):
    if grounding.get('content_status') == 'contradicted':
        label = 'VERWORFEN – Quellenwiderspruch'
    elif grounding.get('evidence_status') == 'exact':
        return text
    else:
        label = 'UNBESTÄTIGT – ' + ('Quelle eingegrenzt' if grounding.get('evidence_status') == 'source_range' else 'Unbelegt')
    questions = ' '.join(grounding.get('questions') or ['Stützt die Originalquelle diese Aussage?'])
    return f'[{label}] {text} Prüffrage: {questions}'
