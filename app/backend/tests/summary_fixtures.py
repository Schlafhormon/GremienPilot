"""Scripted transport responses, never an empirical quality reference."""
import json


class SummaryModel:
    def __init__(self):
        self.calls = []
        self.overrides = {}

    def __call__(self, request):
        body = json.loads(request['messages'][1]['content'])
        self.calls.append((body, request))
        phase = body['phase']
        if phase in self.overrides:
            value = self.overrides[phase]
            if isinstance(value, BaseException):
                raise value
            return json.dumps(value(body) if callable(value) else value)
        rows = body.get('source', [])
        ids = [r['source_id'] for r in rows]
        if phase in {'generate', 'blind_inventory'}:
            row = next((r for r in rows if r['text'].strip()), None)
            claims = [] if row is None else [dict(section='discussion', text='Der Sachverhalt wurde beraten.',
                scope='current', evidence=[dict(source_id=row['source_id'], quote=row['text'])])]
            answer = dict(claims=claims, considered_source_ids=ids)
        elif phase in {'consolidate', 'reconcile'}:
            answer = dict(claims=body['candidate'], considered_source_ids=body['source_catalog'])
        else:
            answer = dict(checked_claim_ids=[c['claim_id'] for c in body['candidate']],
                          considered_source_ids=ids, issues=[])
        return json.dumps(answer)
