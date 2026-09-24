"""Model-only decisions; tests validate orchestration, never claim model quality."""
import json
import pytest
import agenda_llm
from agenda_detection import segment_known_agenda, detect_agenda_from_transcript
from assignment_suggestions import TranscriptUtterance
from llm_transport import LLMCancelledError


def transcript(texts):
    return [TranscriptUtterance('M', text, f'line-{i}', i, i+1) for i, text in enumerate(texts)]


@pytest.mark.parametrize('mode', ['fast', 'slow'])
def test_context_packing_bounds_work_and_preserves_every_source(monkeypatch, mode):
    from processing_mode import processing_scope
    work = agenda_llm.Workflow.__new__(agenda_llm.Workflow)
    rows = [{'index': i, 'text': 'Source'} for i in range(1774)]
    checked = []
    def fits(phase, instruction, body, schema):
        checked.append(body['sources'])
        return len(body['sources']) <= 1400
    work.fits = fits
    with processing_scope(mode):
        groups = list(work.context_groups('context', '', [], rows, {}))
    assert [row for group in groups for row in group] == rows
    assert [len(group) for group in groups] == [1400, 374]
    assert len(checked) < 30  # Formerly 1773 complete tokenizations.
    assert all(group in checked for group in groups)


def test_context_packing_rejects_single_oversized_source_and_obeys_cancellation(monkeypatch):
    from llm_transport import ContextBudgetError
    work = agenda_llm.Workflow.__new__(agenda_llm.Workflow)
    work.fits = lambda *args: False
    with pytest.raises(ContextBudgetError, match='single_context_source'):
        list(work.context_groups('context', '', [], [{'index': 0}], {}))
    def cancelled():
        raise LLMCancelledError()
    monkeypatch.setattr(agenda_llm.durable, 'check', cancelled)
    with pytest.raises(LLMCancelledError):
        list(work.context_groups('context', '', [], [{'index': 0}], {}))


def run(texts, tops=None, **kwargs):
    return segment_known_agenda(transcript(texts), tops or ['1 Haushalt', '2 Schulbau'], use_llm=True, **kwargs)


@pytest.mark.parametrize('texts,labels', [
    (['Zum Haushalt.', 'Wie sieht der Bauplan aus?', 'Die Schule wird erweitert.'], [0, 1, 1]),
    (['Haushaltsberatung.', 'Später beraten wir TOP 2.', 'Weitere Haushaltsmittel.'], [0, 0, 0]),
    (['Zum Protokoll.', 'Damals wurde über TOP 2 gesprochen.', 'Die Niederschrift wird geändert.'], [0, 0, 0]),
    (['TOP 2 zuerst.', 'Danach Haushalt.', 'Zurück zum Bau.'], [1, 0, 1]),
    (['Öffentliche Beratung.', 'Die Öffentlichkeit wird ausgeschlossen.', 'Personalberatung.'], [0, 1, 1]),
])
def test_full_independent_review_of_model_decisions(agenda_model, texts, labels):
    agenda_model.labels = {i: [f'agenda:{label}'] for i, label in enumerate(labels)}
    result = run(texts)
    assert result.assignments == labels
    assert result.llm.processing_complete and result.llm.review_complete
    assert [r['review_status'] for r in result.llm.line_results] == ['agreed']*len(texts)
    reviewed = [r['index'] for b, _ in agenda_model.calls if b['phase'] == 'independent:detail' for r in b['target_lines']]
    assert reviewed == list(range(len(texts)))
    for body, _ in agenda_model.calls:
        if body['phase'].startswith('independent'):
            assert body.get('opinions') is None
            assert 'previous_top' not in body and 'evidence_context' not in body


def test_no_language_rule_vetoes_model_or_restricts_schema(agenda_model):
    agenda_model.labels = {0: ['top-public'], 1: ['top-private'], 2: ['top-public']}
    result = run(['TOP 99 nicht behandeln.', 'Schließung angekündigt.', 'Wir bleiben nichtöffentlich.'],
        ['[Öffentlich] 01 Haushalt', '[Nichtöffentlich] 01 Schließung'], top_ids=['top-public', 'top-private'])
    assert result.assignments == [0, 1, 0]
    assert not any(s.uncertain for s in result.segments)
    for body, call in agenda_model.calls:
        if body['phase'].endswith(':detail'):
            schema = call['response_format']['json_schema']['schema']
            assert schema['properties']['lines']['items']['properties']['top_ids']['items']['enum'] == ['top-public', 'top-private']


def test_joint_deliberation_has_multiple_ids_no_arbitrary_scalar(agenda_model):
    agenda_model.labels = {0: ['agenda:0', 'agenda:1']}
    result = run(['Wir beraten die beiden Punkte gemeinsam.'])
    assert result.assignments == [None] and result.segments == []
    assert result.llm.line_results[0]['status'] == 'assigned'
    assert result.llm.line_results[0]['top_ids'] == ['agenda:0', 'agenda:1']
    assert result.llm.gaps == []
    assert result.llm.review_complete


def test_all_semantic_gaps_reviewed_and_last_difference_resolved(agenda_model, monkeypatch):
    monkeypatch.setenv('AGENDA_DETECTION_CHUNK_LINES', '3')
    agenda_model.labels = {i: [] for i in range(50)}
    agenda_model.review_labels = {49: ['agenda:1']}
    agenda_model.resolve_labels = {49: ['agenda:1']}
    result = run(['Diskussion.']*50)
    assert result.assignments == [None]*49+[1]
    assert result.llm.processing_complete and result.llm.review_complete
    assert result.llm.line_results[49]['review_status'] == 'resolved'
    assert len(result.llm.gaps) == 49 and all(g['kind'] == 'semantic' for g in result.llm.gaps)
    assert sum(len(b['target_lines']) for b, _ in agenda_model.calls if b['phase'] == 'independent:detail') == 50


def test_unresolved_semantic_difference_is_not_technical_failure(agenda_model):
    agenda_model.uncertain = {0}
    result = run(['Mehrdeutige Beratung.'])
    assert result.llm.status == 'success'
    assert result.llm.processing_complete and result.llm.review_complete
    assert result.llm.review_required and result.llm.line_results[0]['review_status'] == 'unresolved'
    assert len([b for b, _ in agenda_model.calls if b['phase'] == 'resolve:detail']) == 1


def test_failed_review_keeps_initial_assignment_and_reports_technical_review_gap(agenda_model):
    agenda_model.overrides['independent:detail'] = TimeoutError('PRIVATE SECRET')
    result = run(['TOP 1 Haushalt.'])
    assert result.assignments == [0]
    assert result.llm.processing_complete and not result.llm.review_complete
    assert result.llm.line_results[0]['review_status'] == 'technical_pending'
    assert result.llm.gaps == [] and result.llm.status == 'partial_failure'
    assert 'PRIVATE' not in str(result)


def test_initial_failure_never_becomes_semantic_gap_or_heuristic(agenda_model, monkeypatch):
    monkeypatch.setenv('AGENDA_REPAIR_SPLIT_DEPTH', '0')
    agenda_model.overrides['primary:detail'] = TimeoutError('secret')
    result = run(['Kommen wir zu TOP 1 Haushalt.']*3)
    assert result.assignments == [None]*3
    assert result.llm.processed_lines == []
    assert all(g['kind'] == 'technical' for g in result.llm.gaps)


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'unknown_top', 'unknown_source', 'missing_reason'])
def test_strict_coverage_identity_and_evidence_validation(agenda_model, monkeypatch, mutation):
    monkeypatch.setenv('AGENDA_REPAIR_SPLIT_DEPTH', '0')
    def bad(body):
        data = agenda_model.answer(body)
        if mutation == 'missing': data['lines'] = []
        elif mutation == 'duplicate': data['lines'] *= 2
        elif mutation == 'unknown_top': data['lines'][0]['top_ids'] = ['foreign']
        elif mutation == 'unknown_source': data['lines'][0]['line_id'] = 'foreign'
        elif mutation == 'bad_quote': data['lines'][0]['evidence'][0]['quote'] = 'invented'
        else: data['lines'][0]['reason'] = ''
        return data
    agenda_model.overrides['primary:detail'] = bad
    result = run(['Beratung.'])
    assert result.llm.status == 'failed'
    assert result.llm.gaps[0]['kind'] == 'technical'
    assert result.llm.validation_reasons
    assert len([b for b, _ in agenda_model.calls if b['phase'] == 'primary:detail']) == 2


def test_duplicate_json_key_rejected():
    with pytest.raises(agenda_llm.AgendaValidationError):
        agenda_llm.parse_response('{"lines":[],"lines":[]}')


def test_cancel_is_not_swallowed_or_cached(agenda_model):
    agenda_model.overrides['primary:detail'] = LLMCancelledError()
    with pytest.raises(LLMCancelledError):
        run(['Beratung.'])


def test_resume_reuses_successful_calls_not_failed_review(agenda_model, monkeypatch, tmp_path):
    monkeypatch.setenv('LLM_CACHE_DIR', str(tmp_path))
    agenda_model.overrides['independent:detail'] = TimeoutError()
    first = run(['Beratung.'])
    agenda_model.calls.clear()
    agenda_model.overrides.clear()
    second = run(['Beratung.'])
    assert first.llm.status == 'partial_failure'
    assert second.llm.status == 'success'
    assert [b['phase'] for b, _ in agenda_model.calls] == ['independent:detail']


def test_long_transcript_all_originals_read_twice_and_context_retained(agenda_model, monkeypatch):
    monkeypatch.setenv('LLM_CONTEXT_TOKENS', '16384')
    texts = [f'Beitrag {i}: ' + 'Unterschiedliche Sitzungsthemen. '*6 for i in range(180)]
    result = run(texts)
    assert result.llm.processing_complete and result.llm.review_complete
    for role in ('primary', 'independent'):
        read = [r['index'] for body, _ in agenda_model.calls if body['phase'] == role+':context'
                for r in body['sources'] if 'index' in r]
        assert read == list(range(180))
        details = [b for b, _ in agenda_model.calls if b['phase'] == role+':detail']
        assert [r['index'] for b in details for r in b['target_lines']] == list(range(180))
        assert all(b['context']['coverage'] == [0, 179] for b in details)
        assert all('model_notes' in b['context'] for b in details)
    assert result.llm.provenance['context_archive']


def test_source_retrieval_is_model_requested_and_bounded(agenda_model, monkeypatch):
    monkeypatch.setenv('AGENDA_DETECTION_CHUNK_LINES', '1')
    def request(body):
        if body['target_start'] == 1 and 'requested_originals' not in body:
            return {'source_ranges': [{'start': 0, 'end': 0}], 'lines': []}
        return agenda_model.answer(body)
    agenda_model.overrides['independent:detail'] = request
    result = run(['Früherer Aufruf.', 'Fortsetzung.'])
    assert result.llm.review_complete
    reread = [b for b, _ in agenda_model.calls if 'requested_originals' in b]
    assert reread[0]['requested_originals'][0]['line_id'] == 'L1'
    agenda_model.overrides['independent:detail'] = {'source_ranges': [{'start': 0, 'end': 0}], 'lines': []}
    result = run(['Früherer Aufruf.'])
    assert result.llm.processing_complete and not result.llm.review_complete


def test_status_disagreements_resolved_by_model(agenda_model):
    agenda_model.states = {'agenda:0': 'deferred', 'agenda:1': 'not_evidenced'}
    agenda_model.review_states = {'agenda:0': 'removed', 'agenda:1': 'not_evidenced'}
    result = run(['Der Punkt entfällt.'])
    assert result.llm.agenda_states[0]['status'] == 'deferred'  # scripted adjudicator, no keyword override
    assert all(s['review_status'] == 'resolved' for s in result.llm.agenda_states)
    assert any(b['phase'] == 'resolve:states:states:v1' for b, _ in agenda_model.calls)


def test_discovery_precedes_common_reconstruction_without_invented_numbers(agenda_model):
    result = detect_agenda_from_transcript(transcript(['Wir beraten den Haushalt.']), use_llm=True)
    assert result.tops == ['Haushalt']
    assert result.llm.provenance['identities'][0]['number'] is None
    phases = [b['phase'] for b, _ in agenda_model.calls]
    assert phases[:2] == ['primary:discover', 'independent:discover']
    assert phases[-2:] == ['primary:detail', 'independent:detail']


@pytest.mark.parametrize('mode', ['fast','slow'])
@pytest.mark.parametrize('known', [True,False])
def test_discovery_request_distinguishes_additions_from_full_inventory(agenda_model, mode, known):
    result = segment_known_agenda(transcript(['Beratung.']), ['1 Haushalt'] if known else [],
        use_llm=True, processing_mode=mode)
    assert result.llm.processing_complete
    requests = [(b,r) for b,r in agenda_model.calls if b['phase'].endswith(':discover')]
    assert requests
    for body, request in requests:
        assert body['inventory_task'] == ('additional_topics_only' if known else 'full_inventory')
        assert bool(body['known_agenda']) == known
        properties = request['response_format']['json_schema']['schema']['properties']['response']['anyOf'][0]['properties']['result']['properties']
        assert list(properties)[:2] == ['reason','items']
    if known:
        assert result.tops == ['1 Haushalt']


def test_reconstruction_can_request_original_sources(agenda_model,monkeypatch):
    monkeypatch.setattr(agenda_llm.Workflow,'context',lambda self,*args:
        {'model_notes':[{'evidence':[{'line_id':self.rows[0]['line_id'],'quote':'Beratung'}]}], 'coverage':[0,0]})
    def reconstruct(body):
        if 'requested_originals' not in body:
            return {'source_ranges': [{'start': 0, 'end': 0}], 'narrative': 'Originalbeleg benötigt', 'episodes': [], 'agenda_states': []}
        return agenda_model.answer(body)
    agenda_model.overrides['independent:reconstruct:trajectory:v1'] = reconstruct
    result = run(['Beratung.'])
    assert result.llm.review_complete
    calls = [b for b, _ in agenda_model.calls if b['phase'] == 'independent:reconstruct:trajectory:v1']
    assert len(calls) == 2 and calls[1]['requested_originals'][0]['line_id'] == 'L1'


def test_oversized_single_source_never_silently_truncated(agenda_model):
    result = run(['Extrem langer Originaltext. '*2000])
    assert result.assignments == [None]
    assert not result.llm.processing_complete
    assert result.llm.gaps[0]['kind'] == 'technical'
    assert 'ContextBudgetError' in result.llm.failure_reasons


def test_source_change_invalidates_cache_even_with_same_model_notes(agenda_model, monkeypatch, tmp_path):
    monkeypatch.setenv('LLM_CACHE_DIR', str(tmp_path))
    first = run(['Unverändert.', 'Originalstand.'])
    agenda_model.calls.clear()
    second = run(['Unverändert.', 'Geänderter Quellenstand.'])
    assert first.llm.provenance['source_sha256'] != second.llm.provenance['source_sha256']
    assert second.llm.attempted_calls == 8


def test_model_confidence_not_replaced_by_speech_patterns(agenda_model):
    def confident(body):
        result = agenda_model.answer(body)
        for line in result['lines']:
            line['confidence'] = 0.61
        return result
    agenda_model.overrides.update({'primary:detail': confident, 'independent:detail': confident})
    result = run(['TOP 1 nicht beraten, später vielleicht.'])
    assert result.segments[0].confidence == 0.61


def test_output_budget_plans_detail_ownership(agenda_model, monkeypatch):
    monkeypatch.setenv('AGENDA_OUTPUT_TOKENS', '1024')
    monkeypatch.setenv('AGENDA_OUTPUT_TOKENS_PER_LINE', '256')
    monkeypatch.setenv('LLM_CHUNK_CHARS', '1')  # obsolete character limit must not select/omit source input
    result = run(['Beratung.']*9)
    assert result.llm.review_complete
    windows = [b for b, _ in agenda_model.calls if b['phase'] == 'primary:detail']
    assert [len(b['target_lines']) for b in windows] == [2, 2, 2, 2, 1]
    assert all(len(b['context']['original_transcript']) == 9 for b in windows)


def test_known_agenda_can_gain_model_decided_addition_without_reindexing(agenda_model):
    agenda_model.inventory = [{'title': 'Zusätzliche Beratung', 'number': None, 'section': None,
        'evidence': [{'line_id': 'line-0', 'quote': 'Ein zusätzlicher Punkt wird beraten.'}]}]
    def detail(body):
        data = agenda_model.answer(body)
        data['lines'][0]['top_ids'] = [body['agenda'][-1]['top_id']]
        return data
    agenda_model.overrides.update({'primary:detail': detail, 'independent:detail': detail})
    result = run(['Ein zusätzlicher Punkt wird beraten.'], top_ids=['stable-a', 'stable-b'])
    assert result.tops == ['1 Haushalt', '2 Schulbau', 'Zusätzliche Beratung']
    assert result.assignments == [2]
    assert [i['top_id'] for i in result.llm.provenance['identities'][:2]] == ['stable-a', 'stable-b']
    assert result.llm.review_complete
