"""Original transcript evidence, independent of any model assignment.

Events are conservative reading aids, not a deterministic agenda classifier.
No order/position-based TOP inference is performed. Unknown calls invalidate the
active TOP anchor; section and topic have separate lifetimes.
"""
import re
from agenda_labels import fold, parse_agenda_label, reference_sections, reference_targets
from assignment_suggestions import transition_kind, score_line_for_top, NON_CURRENT


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


def closing_evidence_text(text, following=''):
    # Conventional chair formula: the condition refers to absence of further
    # interventions, followed by a present closing act (not future/negated).
    text = re.sub(r'^(?:Wenn|Falls)\s+(?:das\s+nicht\s+der\s+Fall\s+ist|'
                  r'keine\s+(?:weiteren\s+)?(?:Fragen|Wortmeldungen)\s+(?:vorliegen|bestehen)),\s*',
                  '', text, flags=re.I)
    value = fold(text)
    if '?' in text or re.search(r'[„“"«»]', text):
        return None
    target = r'\b(?:(?:(?:nicht[\s-]*)?(?:o|oe)ffentliche[nr]?\s+)?sitzung|(?:nicht[\s-]*)?(?:o|oe)ffentlichen?\s+teil)\b'
    # Scope is taken from the closing clause, not a subsequent announcement of
    # the other section. A numbered item closed in the same row is not a veto.
    match = re.search(r'\b(?:schliesse|schliessen|beende|beenden)\b(?!\s+(?:mich|uns)\b)[^.!?;]{0,160}' + target +
                      r'|' + target + r'[^.!?;]{0,60}\b(?:geschlossen|beendet)\b', value)
    if match and not NON_CURRENT.search(value):
        return match.group()
    # A polite present closing formula needs corroboration from the immediately
    # following farewell. It is not inferred from a preview of the closing TOP.
    polite = re.search(r'^(?:dann\s+)?(?:wurde\s+ich|ich\s+wurde)\s+jetzt\s+'
                       r'(?:die|den)\s+' + target + r'\s+(?:schliessen|beenden)\.?$', value)
    farewell = re.search(r'\b(?:nachhauseweg|heimweg|auf wiedersehen|guten abend|gute nacht)\b', fold(following))
    if polite and farewell and not NON_CURRENT.search(re.sub(r'\bwurde\b', '', value)):
        return polite.group()
    return None


def closing_act(text, following=''):
    return closing_evidence_text(text, following) is not None


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


def numbering_change_cue(text):
    """A proposal is enough to make literal printed-number matching unreliable.

    This is deliberately NOT an adopted amendment or an inferred offset. The
    model must read the original evidence, including any rejection/correction.
    """
    value = fold(text)
    return bool(re.search(r'\b(?:tagesordnung|tagesordnungspunkte?\w*|tops?)\b', value) and (
        re.search(r'\b(?:einfug\w*|hinzufug\w*|einschieb\w*|umnummerier\w*)\b', value)
        or re.search(r'\bfug\w*\b.{0,100}\b(?:ein|hinzu)\b', value)
        or re.search(r'\b(?:neu\w*|zusatzlich\w*)\b.{0,100}\b(?:aufnehm\w*|aufgenommen|erganz\w*)\b', value)
        or re.search(r'\b(?:rutsch\w*|verschieb\w*)\b.{0,100}\b(?:hinten|vorne|vorn)\b', value)))


class EvidenceContext:
    def __init__(self, transcript, tops, agenda):
        self.transcript, self.tops, self.agenda = transcript, tops, agenda
        self.states = []
        section = topic = continuation = numbering_changes = None
        for i, line in enumerate(transcript):
            kind = transition_kind(line.text)
            scope = section_act(line.text)
            if scope:
                if not section or section['section'] != scope:
                    numbering_changes = None
                section = {'index': i, 'section': scope, 'kind': 'section_call'}
                topic = None
                continuation = None
            if numbering_change_cue(line.text):
                numbering_changes = {'kind': 'possible_agenda_amendment',
                    'indices': list((numbering_changes or {}).get('indices', [])) + [i]}
            has_ref, targets = reference_targets(line.text, tops)
            if section:
                targets = {t for t in targets if agenda[t]['section'] in {None, section['section']}}
            event = None
            closing = closing_evidence_text(line.text, transcript[i+1].text if i+1 < len(transcript) else '')
            if closing:
                scopes = reference_sections(closing) or ({section['section']} if section else set())
                matches = [j for j, t in enumerate(agenda) if re.search(r'schliessung|sitzungsende', fold(t['title']))
                           and (not scopes or t['section'] in scopes)]
                if len(matches) == 1:
                    event = {'index': i, 'top_id': agenda[matches[0]]['top_id'], 'kind': 'closing'}
            elif kind in {'call', 'heading', 'continuation'}:
                matches = (set() if numbering_changes else targets) if has_ref else {j for j, t in enumerate(tops)
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
            self.states.append({'section': section, 'topic': topic, 'continuation': continuation,
                                'numbering_changes': numbering_changes})

    def at(self, index):
        return self.states[index] if index >= 0 else {'section': None, 'topic': None, 'continuation': None, 'numbering_changes': None}

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
        packet = {'section_anchor': prior['section'], 'topic_anchor': prior['topic'],
                'continuation_anchor': prior['continuation'],
                'original_evidence': [self.row(i) for i in sorted(indices)],
                'instruction': 'Belege aus Originaltext, keine bestätigten Modelllabels. '
                'Abschnitt und aktiver TOP sind getrennt. Weitere Sachthemen im offenen TOP bleiben '
                'Fortsetzungen; Niederschriftsrückblicke sind keine heutigen TOP-Wechsel. '
                'Die Anker sind konservative Erkennungshilfen; prüfe neue und indirekte Übergänge selbst.'}
        if prior['numbering_changes']:
            indices = {i for cue in prior['numbering_changes']['indices']
                       for i in range(max(0, cue-1), min(start, cue+7))}
            packet['numbering_changes'] = {**prior['numbering_changes'],
                'original_evidence': [self.row(i) for i in sorted(indices)],
                'instruction': 'Mögliche Änderung der Tagesordnung. Dies beweist weder Annahme noch Nummernversatz. '
                'Gesprochene Nummern können von PDF-Nummern abweichen. Identität anhand Originalinhalt prüfen, '
                'keinen pauschalen Versatz berechnen. Nicht in der Agenda enthaltene TOPs bleiben null mit Begründung.'}
        return packet

    def row(self, i):
        return {'index': i, 'speaker': self.transcript[i].speaker, 'text': self.transcript[i].text}
