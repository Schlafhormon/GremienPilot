"""Scripted model answers, not a semantic oracle. No model or network access."""
import json
from types import SimpleNamespace
import agenda_llm


class AgendaModel:
    def __init__(self, monkeypatch):
        self.calls = []
        self.labels = {}
        self.review_labels = {}
        self.resolve_labels = {}
        self.overrides = {}
        self.uncertain = set()
        self.inventory = None
        self.states = {}
        self.review_states = {}
        monkeypatch.setattr(agenda_llm, 'complete', self.complete)
        monkeypatch.setattr(agenda_llm, 'model_fingerprint', lambda c: {'model': c.model, 'digest': 'test'})

    def complete(self, client, config, **kwargs):
        body = json.loads(kwargs['messages'][1]['content'])
        self.calls.append((body, kwargs))
        phase = body['phase']
        if phase in self.overrides:
            value = self.overrides[phase]
            if isinstance(value, Exception):
                raise value
            data = value(body) if callable(value) else value
        else:
            data = self.answer(body)
        if phase.endswith(':discover') or phase.endswith(':reconstruct') or phase == 'resolve:states':
            data.setdefault('source_ranges', [])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(data)))])

    def evidence(self, row):
        return [{'line_id': row['line_id'], 'quote': row['text']}]

    def original(self, body):
        context = body['context']
        if 'original_transcript' in context:
            return context['original_transcript'][0]
        evidence = context['model_notes'][0]['evidence'][0]
        return {'line_id': evidence['line_id'], 'text': evidence['quote']}

    def answer(self, body):
        phase = body['phase']
        if phase.endswith(':context'):
            first = body['sources'][0]
            evidence = self.evidence(first) if 'text' in first else first['evidence']
            return {'narrative': 'Quellengebundener Sitzungsverlauf, einschließlich Wiederaufnahmen.', 'evidence': evidence}
        if phase.endswith(':discover'):
            if body.get('known_agenda') and self.inventory is None:
                return {'items': [], 'reason': 'Keine zusätzlichen Punkte.'}
            return {'items': self.inventory or [{'title': 'Haushalt', 'number': None, 'section': None,
                'evidence': self.evidence(self.original(body))}], 'reason': 'Im Transkript erkennbare Beratung.'}
        if phase.endswith(':reconstruct') or phase == 'resolve:states':
            return {'narrative': 'Gesamter Sitzungsverlauf mit indirekten Übergängen und Wiederaufnahmen.',
                    'episodes': [{'start_line_id': self.original(body)['line_id'], 'end_line_id': self.original(body)['line_id'],
                        'top_ids': [body['agenda'][0]['top_id']] if body['agenda'] else [], 'section': None,
                        'reason': 'Beratung', 'evidence': self.evidence(self.original(body))}],
                    'agenda_states': [{'top_id': t['top_id'],
                        'status': (self.review_states if phase.startswith('independent') else self.states).get(t['top_id'], 'treated'),
                        'reason': 'Anhand des vollständigen Verlaufs geprüft.',
                        'evidence': self.evidence(self.original(body))} for t in body['agenda']]}
        labels = self.review_labels if phase.startswith('independent') else self.resolve_labels if phase.startswith('resolve') else self.labels
        return {'source_ranges': [], 'lines': [{'line_id': row['line_id'],
            'top_ids': [body['agenda'][int(t.split(':')[1])]['top_id'] if t.startswith('detected:') else t
                        for t in labels.get(row['index'], self.labels.get(row['index'], [body['agenda'][0]['top_id']] if body['agenda'] else []))],
            'reason': 'Fachliche Bewertung im gesamten Sitzungskontext.',
            'evidence': self.evidence(row), 'uncertain': row['index'] in self.uncertain, 'confidence': 0.8}
            for row in body['target_lines']]}
