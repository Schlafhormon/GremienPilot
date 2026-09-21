"""Original transcript evidence, independent of any model assignment.

Events are conservative reading aids, not a deterministic agenda classifier.
No order/position-based TOP inference is performed. Unknown calls invalidate the
active TOP anchor; section and topic have separate lifetimes.
"""
import re
from agenda_labels import fold, parse_agenda_label, reference_sections, reference_targets
from assignment_suggestions import transition_kind, score_line_for_top


def model_agenda(tops):
    bases = []
    for title in tops:
        label = parse_agenda_label(title)
        bases.append(f'{label.section or "unspecified"}:{label.original_number or "unnumbered"}')
    counts = {}
    result = []
    for index, (title, base) in enumerate(zip(tops, bases)):
        counts[base] = counts.get(base, 0) + 1
        label = parse_agenda_label(title)
        # Suffix is explicitly an occurrence disambiguator, never an original number.
        identity = base if bases.count(base) == 1 else f'{base}~occurrence-{counts[base]}'
        result.append({'top_id': identity, 'title': label.title,
                       'number': label.original_number, 'number_key': label.number_key,
                       'section': label.section})
    return result


def closing_act(text):
    # Conventional chair formula: the condition refers to absence of further
    # interventions, followed by a present closing act (not future/negated).
    text = re.sub(r'^(?:Wenn|Falls)\s+(?:das\s+nicht\s+der\s+Fall\s+ist|'
                  r'keine\s+(?:weiteren\s+)?(?:Fragen|Wortmeldungen)\s+(?:vorliegen|bestehen)),\s*',
                  '', text, flags=re.I)
    value = fold(text)
    return transition_kind(text) not in {'mention', 'mixed'} and bool(re.search(
        r'\b(?:schliesse|schliessen|beende|beenden)\b.{0,80}\bsitzung\b|'
        r'\bsitzung\b.{0,60}\b(?:geschlossen|beendet)\b', value))


def section_act(text):
    value = fold(text)
    if transition_kind(text) not in {'call', 'heading'}:
        return None
    sections = reference_sections(text)
    # A call of minutes ABOUT a public meeting is not reopening that section.
    if len(sections) == 1 and (re.search(r'\b(?:nicht[\s-]*)?(?:o|oe)ffentlich\w*\s+teil\b', value)
            or re.search(r'\b(?:eroffne|eroffnen)\b.{0,60}\bsitzung\b', value)):
        return next(iter(sections))
    return None


class EvidenceContext:
    def __init__(self, transcript, tops, agenda):
        self.transcript, self.tops, self.agenda = transcript, tops, agenda
        self.states = []
        section = topic = continuation = None
        for i, line in enumerate(transcript):
            kind = transition_kind(line.text)
            scope = section_act(line.text)
            if scope:
                section = {'index': i, 'section': scope, 'kind': 'section_call'}
                topic = None
                continuation = None
            has_ref, targets = reference_targets(line.text, tops)
            if section:
                targets = {t for t in targets if agenda[t]['section'] in {None, section['section']}}
            event = None
            if closing_act(line.text):
                scopes = reference_sections(line.text) or ({section['section']} if section else set())
                matches = [j for j, t in enumerate(agenda) if re.search(r'schliessung|sitzungsende', fold(t['title']))
                           and (not scopes or t['section'] in scopes)]
                if len(matches) == 1:
                    event = {'index': i, 'top_id': agenda[matches[0]]['top_id'], 'kind': 'closing'}
            elif kind in {'call', 'heading', 'continuation'}:
                matches = targets if has_ref else {j for j, t in enumerate(tops)
                    if (not section or agenda[j]['section'] in {None, section['section']})
                    and score_line_for_top(line, t, j, tops)[0] >= 0.7}
                # A continuation can reaffirm an existing anchor; it cannot
                # manufacture a new topic from a protocol reference.
                if kind == 'continuation' and (not topic or topic['top_id'] not in {agenda[j]['top_id'] for j in matches}):
                    matches = set()
                if len(matches) == 1:
                    event = {'index': i, 'top_id': agenda[next(iter(matches))]['top_id'], 'kind': kind}
                elif has_ref and kind != 'continuation':
                    topic = None
                    continuation = None
            if event and event['kind'] == 'continuation':
                continuation = event
            elif event:
                topic = event
                continuation = None
                # A unique agenda call also provides section evidence. This is
                # independent of a model prediction, with its original quote.
                called = next(t for t in agenda if t['top_id'] == event['top_id'])
                if called['section'] and (not section or section['section'] != called['section']):
                    section = {'index': i, 'section': called['section'], 'kind': 'agenda_call_section'}
            self.states.append({'section': section, 'topic': topic, 'continuation': continuation})

    def at(self, index):
        return self.states[index] if index >= 0 else {'section': None, 'topic': None, 'continuation': None}

    def packet(self, start, end):
        prior = self.at(start - 1)
        events = []
        for key in ('section', 'topic', 'continuation'):
            event = prior[key]
            if event and event not in events:
                events.append(event)
        # Keep original evidence even when a repair child starts far from the call.
        indices = set()
        for event in events:
            i = event['index']
            # Section evidence needs its own original quote, not an entire old
            # topic discussion. Keep more original text for the actual TOP call.
            indices.update(range(max(0, i-1), min(start, i+(2 if event == prior['section'] else 5))))
        return {'section_anchor': prior['section'], 'topic_anchor': prior['topic'],
                'continuation_anchor': prior['continuation'],
                'original_evidence': [self.row(i) for i in sorted(indices)],
                'instruction': 'Belege aus Originaltext, keine bestätigten Modelllabels. '
                'Abschnitt und aktiver TOP sind getrennt. Weitere Sachthemen im offenen TOP bleiben '
                'Fortsetzungen; Niederschriftsrückblicke sind keine heutigen TOP-Wechsel. '
                'Die Anker sind konservative Erkennungshilfen; prüfe neue und indirekte Übergänge selbst.'}

    def row(self, i):
        return {'index': i, 'speaker': self.transcript[i].speaker, 'text': self.transcript[i].text}
