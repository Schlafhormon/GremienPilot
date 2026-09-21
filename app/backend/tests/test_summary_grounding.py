import json
from types import SimpleNamespace

import pytest
import summary_grounding as grounding


def part(**overrides):
    structured = dict(discussion=[], decisions=[], votes=[], action_items=[], open_points=[], uncertainties=[])
    structured.update(overrides)
    return {'text': 'Ich lese wortwörtlich vor. Die Verbandsversammlung beauftragt die Verwaltung.',
            'context': 'Frühere Sitzung.', 'structured': structured}


def run(monkeypatch, parts, answer, limit=3, year_conflict=False):
    monkeypatch.setenv('LLM_SUMMARY_GROUNDING_MAX_CALLS', str(limit))
    calls = []
    def complete(client, config, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(answer)))])
    monkeypatch.setattr(grounding, 'complete', complete)
    config = SimpleNamespace(uses_ollama=False, reasoning_effort='none', reasoning_options={},
                             base_url='https://example.invalid', model='test', timeout_seconds=10)
    usage = {}
    result = grounding.check_parts(parts, client=None, config=config,
                                   meeting_context='Heutiger Ausschuss.', usage=usage, year_conflict=year_conflict)
    return result, usage, calls


def test_reported_order_is_not_a_current_decision(monkeypatch):
    source = part(decisions=['Die Verwaltung wird beauftragt.'])
    result, usage, calls = run(monkeypatch, [source], {'0:decisions:0': {
        'status': 'reported', 'evidence_id': 'target:0', 'scope_id': 'target:0'}})
    assert not result[0]['structured']['decisions']
    assert result[0]['structured']['discussion'] == ['Bericht über einen anderweitigen Vorgang: Die Verwaltung wird beauftragt.']
    payload = json.loads(calls[0]['messages'][1]['content'])
    assert '\n'.join(payload['target_source'].values()) == source['text']
    assert payload['meeting_context'] == 'Heutiger Ausschuss.'
    assert usage['grounding_calls'] == 1


def test_invented_historical_claim_removed_but_auditable(monkeypatch):
    claim = '1929 wurde ein historischer Krieg ausgelöst.'
    result, usage, _ = run(monkeypatch, [part(discussion=[claim])], {'0:discussion:0': {
        'status': 'unsupported', 'evidence_id': None, 'scope_id': None}}, year_conflict=True)
    assert not result[0]['structured']['discussion']
    assert result[0]['structured']['uncertainties']
    assert usage['claim_checks'][0]['claim'] == claim


@pytest.mark.parametrize('answer', [{}, {'0:decisions:0': {
    'status': 'supported', 'evidence_id': 'target:99', 'scope_id': None}}])
def test_missing_or_unverifiable_verdict_cannot_certify_claim(monkeypatch, answer):
    result, usage, _ = run(monkeypatch, [part(decisions=['Ein Beschluss.'])], answer)
    assert not result[0]['structured']['decisions']
    assert result[0]['structured']['uncertainties']
    assert usage['grounding_unresolved_claims'] == 1


def test_call_budget_keeps_unchecked_claims_out_of_confirmed_results(monkeypatch):
    sources = [part(decisions=['Ein Beschluss.']), part(decisions=['Anderer Beschluss.'])]
    result, usage, calls = run(monkeypatch, sources, {'0:decisions:0': {
        'status': 'supported', 'evidence_id': 'target:0', 'scope_id': 'target:0'}}, limit=1)
    assert len(calls) == 1
    assert result[0]['structured']['decisions']
    assert not result[1]['structured']['decisions']
    assert usage['grounding_incomplete']


@pytest.mark.parametrize('same_evidence', [True, False])
def test_conflicting_temporal_checks_cannot_certify_current_action(monkeypatch, same_evidence):
    source = part(
        discussion=['Der Vorsitzende beauftragt die Verwaltung mit der Vorbereitung.'],
        action_items=['Der Vorsitzende beauftragt die Verwaltung mit der Vorbereitung.'])
    source['text'] += '\nDie Vorbereitung wird abgestimmt.'
    answers = iter([
        {'0:discussion:0': {'status': 'reported', 'evidence_id': 'target:0', 'scope_id': 'target:0'}},
        {'0:action_items:0': {'status': 'supported', 'evidence_id': 'target:0' if same_evidence else 'target:1', 'scope_id': 'target:0'}},
    ])
    monkeypatch.setenv('LLM_SUMMARY_GROUNDING_MAX_CALLS', '3')
    monkeypatch.setattr(grounding, 'complete', lambda *args, **kwargs: SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(next(answers))))]))
    config = SimpleNamespace(uses_ollama=False, reasoning_effort='none', reasoning_options={},
                             base_url='https://example.invalid', model='test', timeout_seconds=10)
    usage = {}
    result = grounding.check_parts([source], client=None, config=config,
                                  meeting_context='', usage=usage, year_conflict=False)
    assert not result[0]['structured']['action_items']
    assert result[0]['structured']['uncertainties']
    assert usage['claim_checks'][1]['validation_reason'] == 'conflicting_temporal_verdicts'
    assert usage['claim_checks'][1]['original_status'] == 'supported'
    assert not usage.get('grounding_incomplete')


def test_short_source_window_keeps_late_vote_after_minutes_discussion():
    lines = {f'target:{i}': f'Anmerkung {i}' for i in range(18)}
    lines['target:0'] = 'Entscheidung zur Niederschrift der Sitzung vom Januar.'
    lines['target:16'] = 'Handzeichen für die Zustimmung zum Protokoll.'
    lines['target:17'] = 'Danke, einstimmig.'
    window = grounding._source_window(lines, 'Genehmigung der Niederschrift der Sitzung vom Januar.')
    assert window == lines


def test_tentative_proposal_is_not_certified_as_accepted_action(monkeypatch):
    source = part(action_items=['Die Verwaltung hat den Text zu verteilen.'])
    source['text'] = 'Ich würde vorschlagen, dass wir den Text verteilen.'
    source['context'] = 'Bericht aus der Verbandsversammlung.'
    result, usage, _ = run(monkeypatch, [source], {'0:action_items:0': {
        'status': 'supported', 'evidence_id': 'target:0', 'scope_id': 'target:0'}})
    assert not result[0]['structured']['action_items']
    assert 'Die Verwaltung hat den Text zu verteilen.' in result[0]['structured']['uncertainties'][0]
    assert usage['claim_checks'][0]['validation_reason'] == 'proposal_without_confirmed_acceptance'


def test_corrected_authority_claim_is_checked_in_discussion(monkeypatch):
    source = part(discussion=['Das Satzungsrecht liegt beim Verband.'])
    source['text'] = 'Die Verbandsversammlung? Nein, das machen wir selber. Das ist unser Satzungsrecht.'
    result, usage, calls = run(monkeypatch, [source], {'0:discussion:0': {
        'status': 'unsupported', 'evidence_id': 'target:0', 'scope_id': 'target:0'}})
    assert not result[0]['structured']['discussion']
    assert 'Zuständigkeit' in calls[0]['messages'][0]['content']
    assert usage['claim_checks'][0]['status'] == 'unsupported'


def test_future_procedural_conditions_receive_source_check(monkeypatch):
    source = part(discussion=['Der Sitzungsteil kann erst entfallen, wenn drei Sitzungen ausgefallen sind.'])
    source['text'] = 'In der nächsten Sitzung geht es um das Protokoll. Danach kann der Sitzungsteil entfallen.'
    source['context'] = ''
    result, usage, calls = run(monkeypatch, [source], {'0:discussion:0': {
        'status': 'unsupported', 'evidence_id': 'target:0', 'scope_id': 'target:0'}})
    assert grounding.needs_grounding(source['text'])
    assert not result[0]['structured']['discussion']
    assert 'Wartefrist' in calls[0]['messages'][0]['content']


def test_disputed_minutes_quote_is_not_certified_as_speakers_current_position(monkeypatch):
    source = part(discussion=['Der Redner bestätigt die beanstandete Aussage.'])
    source['text'] = 'Im Protokoll steht diese Aussage. Das ist nicht richtig wiedergegeben. Die Aussage ist falsch.'
    source['context'] = ''
    result, usage, calls = run(monkeypatch, [source], {'0:discussion:0': {
        'status': 'unsupported', 'evidence_id': 'target:0', 'scope_id': 'target:0'}})
    assert grounding.needs_grounding(source['text'])
    assert not result[0]['structured']['discussion']
    assert 'heutige Position' in calls[0]['messages'][0]['content']


@pytest.mark.parametrize('claim,source,reason', [
    ('Das Satzungsrecht liegt beim Verband.',
     'Die Verbandsversammlung? Nein, nein, das machen wir selber.', 'disputed_authority'),
    ('Die Beratung darf erst stattfinden, wenn mehrere Sitzungen verstrichen sind.',
     'In der nächsten Sitzung können wir das Protokoll behandeln.', 'unverified_future_prerequisite'),
])
def test_reported_verdict_does_not_settle_disputed_authority_or_future_prerequisite(monkeypatch, claim, source, reason):
    item = part(discussion=[claim]); item['text'] = source
    result, usage, _ = run(monkeypatch, [item], {'0:discussion:0': {
        'status': 'reported', 'evidence_id': 'target:0', 'scope_id': 'target:0'}})
    assert not result[0]['structured']['discussion']
    assert usage['claim_checks'][0]['validation_reason'] == reason
    assert usage['claim_checks'][0]['original_status'] == 'reported'


@pytest.mark.parametrize('claim,source,reason', [
    ('Eine Umstellung von Bus auf Bahn ist geplant.', 'Bus ab 2028, Bahn ab 2030.', 'directional_change_not_explicit'),
    ('Es wird keine weiteren Preisanpassungen geben.', 'Genau das wird nicht passieren. Ein Preissprung bleibt möglich.', 'blanket_exclusion_not_explicit'),
])
def test_unstated_direction_or_blanket_exclusion_stays_uncertain(monkeypatch, claim, source, reason):
    item = part(discussion=[claim]); item['text'] = source
    result, usage, calls = run(monkeypatch, [item], {})
    assert not calls
    assert not result[0]['structured']['discussion']
    assert usage['claim_checks'][0]['validation_reason'] == reason
    assert claim in result[0]['structured']['uncertainties'][0]


def test_explicit_changes_and_exclusions_are_not_rejected_by_scope_guard():
    assert grounding._unverified_claim_scope('Umstellung von Bus auf Bahn.', 'Wir stellen von Bus auf Bahn um.') is None
    assert grounding._unverified_claim_scope('Keine Preisanpassungen.', 'Es wird keine Preisanpassungen geben.') is None


def test_incidental_protocol_mention_does_not_expand_information_top_checks(monkeypatch):
    monkeypatch.setenv('LLM_SUMMARY_GROUNDING_MAX_CALLS', '3')
    monkeypatch.setattr(grounding, 'complete', lambda *args, **kwargs: pytest.fail('Irrelevant minutes review'))
    source = part(discussion=['Ein Sachverhalt wurde erläutert.'])
    source['text'] = 'Eine Notiz für das Protokoll. Eine technische Angabe war falsch.'
    source['context'] = ''
    usage = {}
    config = SimpleNamespace(uses_ollama=False, reasoning_effort='none')
    result = grounding.check_parts([source], client=None, config=config,
        meeting_context='', usage=usage, year_conflict=False, top_title='Anfragen und Informationen')
    assert result[0]['structured']['discussion'] == ['Ein Sachverhalt wurde erläutert.']
    assert not usage.get('grounding_calls')
