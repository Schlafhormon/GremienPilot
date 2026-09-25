"""Offline contracts with scripted responses, isolated SQLite and no model I/O."""
from copy import deepcopy

import pytest
from fastapi.testclient import TestClient

import agenda_llm
from agenda_detection import segment_known_agenda
from agenda_order import enforce_order
from test_agenda_llm import transcript
from pdf_fixtures import agenda as pdf_agenda, item


TITLES = ['[Öffentlich] 01 Eröffnung', '[Öffentlich] 02 Haushalt',
          '[Öffentlich] 03 Schule', '[Nichtöffentlich] 01 Personal']
IDS = ['top-a', 'top-b', 'top-c', 'top-d']


def script(model, events):
    """Events are (first line, agenda position, boundary, evidence line)."""
    def answer(body):
        start, end = body['target_start'], body['target_end']
        active = max(event for event in events if event[0] <= start)

        def span(event, at):
            first, position, boundary, evidence = event
            if at != first:
                evidence = at  # Continuation refers only to an original in this window.
            identity = next((t['top_id'] for t in body['agenda'] if t['title'] == TITLES[position]), IDS[position])
            return dict(top_ids=[identity], boundary=boundary if at == first else 'continuation',
                        evidence=[dict(line_id=f'L{evidence+1}')] if evidence is not None else [],
                        reason='Beginnbeleg prüfen.', uncertain=boundary == 'unclear', confidence=0.9)

        return {'response': {'kind': 'assignments', 'initial': span(active, start),
            'changes': {f'L{event[0]+1}': span(event, event[0]) for event in events if start < event[0] <= end}}}
    for role in ('fast:detail', 'primary:detail', 'independent:detail', 'resolve:detail'):
        model.overrides[role] = answer
    return answer


def run(model, events, count=12, mode='fast'):
    script(model, events)
    return segment_known_agenda(transcript(['Originalberatung.']*count), TITLES,
        top_ids=IDS, processing_mode=mode, enforce_top_order=True)


@pytest.mark.parametrize('mode', ['fast', 'slow'])
def test_skipped_top_and_restarted_nonpublic_number_keep_full_agenda(agenda_model, mode):
    result = run(agenda_model, [(0, 0, 'confirmed', 0), (3, 1, 'confirmed', 3),
                                (8, 3, 'confirmed', 8)], mode=mode)
    assert result.assignments == [0]*3 + [1]*5 + [3]*4, (result.llm.failure_reasons, result.llm.line_results[0], len(agenda_model.calls))
    assert result.tops == TITLES
    assert result.llm.agenda_states[2]['status'] == 'not_evidenced'
    assert result.llm.agenda_states[2]['evidence'] == []
    assert result.llm.processing_complete
    if mode == 'slow':
        assert len(result.llm.reconstructions) == 2
        assert result.llm.review_complete
    else:
        assert {b['phase'] for b, _ in agenda_model.calls} == {'fast:detail'}


@pytest.mark.parametrize('mode', ['fast', 'slow'])
@pytest.mark.parametrize('boundary,evidence', [('unclear', 4), ('continuation', 4), ('confirmed', None), ('confirmed', 0)])
def test_premature_forward_jump_is_not_assigned_without_local_beginning(agenda_model, mode, boundary, evidence):
    result = run(agenda_model, [(0, 1, 'confirmed', 0), (4, 3, boundary, evidence)], mode=mode)
    assert result.assignments == [1]*4 + [None]*8
    assert result.llm.processing_complete and result.llm.review_required
    assert result.llm.status == 'review_draft'
    assert result.llm.agenda_states[3]['status'] == 'not_evidenced'
    assert any('Zeilen 5–12' in w for w in result.llm.warnings)
    raw = result.llm.provenance['original_line_results']
    assert raw[4]['top_ids'] == [IDS[3]]
    assert raw[4]['evidence'] == result.llm.line_results[4]['evidence']
    assert result.llm.line_results[4]['grounding']['content_status'] == 'unclear'
    assert result.llm.line_results[4]['grounding']['evidence_status'] != 'exact'


def test_uncertain_jump_cannot_be_laundered_by_next_block_continuation(agenda_model):
    result = run(agenda_model, [(0, 0, 'confirmed', 0), (76, 1, 'unclear', 76),
                                (120, 1, 'confirmed', 120)], count=170)
    assert result.assignments == [0]*76 + [None]*44 + [1]*50
    assert all(r['status'] == 'unassigned' for r in result.llm.line_results[76:120])


def test_backward_jump_across_block_boundary_reopens_conflicting_forward_jump(agenda_model):
    result = run(agenda_model, [(0, 0, 'confirmed', 0), (70, 3, 'confirmed', 70),
                                (80, 1, 'confirmed', 80), (90, 3, 'confirmed', 90)], count=100, mode='slow')
    assert result.assignments == [0]*70 + [None]*20 + [3]*10
    assert any('Zeilen 71–90' in w and 'Rücksprung' in w for w in result.llm.warnings)
    assert result.llm.provenance['primary_decisions'][80]['top_ids'] == [IDS[1]]
    assert result.llm.provenance['review_decisions'][80]['top_ids'] == [IDS[1]]


def test_fast_backward_proposal_at_block_edge_stays_open_without_repair(agenda_model, monkeypatch):
    monkeypatch.setenv('LLM_CONTEXT_TOKENS', '65536')
    result = run(agenda_model, [(0, 0, 'confirmed', 0), (70, 3, 'confirmed', 70),
                                (80, 1, 'confirmed', 80)], count=100)
    assert result.assignments == [0]*70 + [3]*10 + [None]*20
    assert result.llm.review_required and not result.llm.processing_complete
    assert len(agenda_model.calls) == 2
    assert result.llm.failure_reasons == ['invalid_top_identity']
    assert all(r['status'] == 'not_processed' for r in result.llm.line_results[80:])


def test_same_deterministic_guard_does_not_reset_at_gaps_or_block_edges():
    sources = [dict(line_id=f'l{i}', index=i) for i in range(165)]
    proposals = [dict(line_id=f'l{i}', index=i, top_ids=[IDS[0 if i < 79 else 3 if i < 160 else 1]],
        status='assigned', review_status='skipped', uncertain=False,
        boundary='confirmed', boundary_start=0 if i < 79 else 79 if i < 160 else 160,
        evidence=[{'line_id': f'l{0 if i < 79 else 79 if i < 160 else 160}'}]) for i in range(165)]
    proposals[159].update(top_ids=[], status='unassigned', uncertain=True)
    original = deepcopy(proposals)
    cleaned, issues, cursor = enforce_order(proposals, [dict(top_id=t) for t in IDS], sources)
    assert cursor == 3
    assert all(not r['top_ids'] for r in cleaned[79:])
    assert proposals == original
    assert any(i['start_index'] == 79 and i['end_index'] == 164 for i in issues)


def test_review_status_change_within_same_span_does_not_invent_a_new_boundary():
    sources = [dict(line_id=f'l{i}', index=i) for i in range(8)]
    proposals = [dict(line_id=f'l{i}', index=i, top_ids=[IDS[0]], status='assigned',
        review_status='agreed' if i < 4 else 'resolved', uncertain=False,
        boundary='confirmed', boundary_start=0, evidence=[{'line_id': 'l0'}]) for i in range(8)]
    cleaned, issues, _ = enforce_order(proposals, [dict(top_id=t) for t in IDS], sources)
    assert all(row['top_ids'] == [IDS[0]] for row in cleaned)
    assert issues == []


def test_fast_1774_lines_are_bounded_sequential_and_never_replay_completed_sources(agenda_model, monkeypatch):
    monkeypatch.setenv('LLM_CONTEXT_TOKENS', '65536')
    monkeypatch.setenv('LLM_FAST_REASONING_EFFORT', 'high')  # Ordered Fast still disables thinking.
    monkeypatch.setenv('AGENDA_DETECTION_CHUNK_LINES', '0')  # Still bounded in this mode.
    result = run(agenda_model, [(0, 0, 'confirmed', 0), (82, 1, 'confirmed', 82),
                                (900, 3, 'confirmed', 900)], count=1774)
    assert result.assignments == [0]*82 + [1]*818 + [3]*874
    calls = [b for b, _ in agenda_model.calls]
    assert len(calls) == 23
    assert all(b['phase'] == 'fast:detail' and b['source_windows'] == [] for b in calls)
    assert [r['index'] for b in calls for r in b['target_lines']] == list(range(1774))
    for b in calls:
        assert len(b['target_lines']) <= 80
        assert len(b['context']['boundary_overlap']) <= 2
        assert b['context']['previous_top_position'] <= min(t['agenda_position'] for t in b['agenda'])
        assert b['reconstruction'] == {'narrative': ''}
    assert not result.llm.reconstructions
    assert result.llm.provenance['configuration']['reasoning_effort'] == 'none'
    assert result.llm.provenance['configuration']['thinking_tokens'] == 0


def test_fast_context_splits_pass_fresh_cursor_to_each_smaller_window(agenda_model, monkeypatch):
    import json
    original_fits = agenda_llm.fits
    def fits(messages, *args):
        body = json.loads(messages[1]['content'])
        return len(body.get('target_lines', [])) <= 5 and original_fits(messages, *args)
    monkeypatch.setattr(agenda_llm, 'fits', fits)
    result = run(agenda_model, [(0, 0, 'confirmed', 0), (2, 1, 'confirmed', 2),
                                (8, 3, 'confirmed', 8)], count=20)
    assert result.assignments == [0]*2 + [1]*6 + [3]*12
    assert [b['context']['previous_top_position'] for b, _ in agenda_model.calls] == [-1, 1, 3, 3]
    assert [b['context']['continuation_allowed'] for b, _ in agenda_model.calls] == [False, True, True, True]
    assert [r['index'] for b, _ in agenda_model.calls for r in b['target_lines']] == list(range(20))


def test_slow_independent_boundary_disagreement_is_adjudicated(agenda_model, monkeypatch):
    monkeypatch.setenv('LLM_CONTEXT_TOKENS', '65536')
    answer = script(agenda_model, [(0, 0, 'confirmed', 0), (5, 1, 'confirmed', 5)])
    def disagree(body):
        result = answer(body)
        if body['target_start'] >= 5:
            result['response']['initial']['boundary'] = 'unclear'
        if 'L6' in result['response']['changes']:
            result['response']['changes']['L6']['boundary'] = 'unclear'
        return result
    agenda_model.overrides['independent:detail'] = disagree
    agenda_model.overrides['resolve:detail'] = disagree
    result = segment_known_agenda(transcript(['Original.']*10), TITLES, top_ids=IDS,
                                  processing_mode='slow', enforce_top_order=True)
    assert any(b['phase'] == 'resolve:detail' for b, _ in agenda_model.calls)
    assert result.llm.provenance['review_comparison']['disagreement_indices'] == list(range(5, 10))
    assert result.assignments[5:] == [None]*5


@pytest.mark.parametrize('mode', ['fast', 'slow'])
def test_default_off_and_no_agenda_keep_existing_workflow(agenda_model, mode):
    result = segment_known_agenda(transcript(['Original.']*2), [], processing_mode=mode, enforce_top_order=True)
    assert not result.llm.provenance['enforce_top_order']
    assert any(b['phase'].endswith(':discover') for b, _ in agenda_model.calls)
    assert result.assignments == [0, 0]
    agenda_model.labels = {0: ['agenda:1'], 1: ['agenda:0']}
    legacy = segment_known_agenda(transcript(['Original.']*2), TITLES, processing_mode=mode)
    assert legacy.assignments == [1, 0]


def test_pdf_artifacts_are_excluded_but_original_entries_and_sources_retained():
    import extract_tops as pdf
    data = pdf_agenda(items=[item('a', title='Einladungstext'), item('b', title='Unterschrift'),
                             item('c', title='Einführung der digitalen Unterschrift'), item('d', title='Haushalt'), item('e', number='05', title='Unterschrift')])
    result = pdf._result(data, verified=True)
    assert result.tops == ['Einführung der digitalen Unterschrift', 'Haushalt', '05 Unterschrift']
    assert len(result.items) == 5
    assert all(i['kind'] == 'heading' and i['original_kind'] == 'agenda' and i['sources'] for i in result.items[:2])
    assert all(i['kind'] == 'agenda' for i in data['items'])


def test_session_setting_migrates_persists_and_is_inherited_by_reassignment(monkeypatch):
    import main
    import persistence
    import durable_jobs
    client = TestClient(main.app)  # No lifespan worker, service or instance.
    old = client.post('/api/sessions', json={}).json()
    assert old['enforce_top_order'] is False
    sid = old['session_id']
    with persistence.connect() as db:
        db.execute('ALTER TABLE sessions DROP COLUMN enforce_top_order')
    persistence.init_db()
    assert persistence.load_session(sid)['enforce_top_order'] is False
    saved = client.put(f'/api/sessions/{sid}', json={'enforce_top_order': True}).json()
    assert saved['enforce_top_order'] is True
    assert client.put(f'/api/sessions/{sid}', json={}).json()['enforce_top_order'] is True
    assert client.get(f'/api/sessions/{sid}').json()['enforce_top_order'] is True
    response = client.post('/api/agenda-detection/jobs', json={'session_id': sid,
        'tops': TITLES, 'top_ids': IDS, 'transcript': [{'speaker': 'M', 'text': 'Original', 'line_id': 'l1', 'start': 0, 'end': 1}]})
    assert response.status_code == 202
    queued = durable_jobs.load(response.json()['job_id'])
    assert queued['payload']['request']['enforce_top_order'] is True
    assert durable_jobs.public(dict(queued, state='completed'))['source']['enforce_top_order'] is True
    assert queued['state'] == 'queued'  # Never execute this job.


def test_pipeline_submission_persists_order_without_running_worker(monkeypatch):
    from unittest.mock import AsyncMock
    from types import SimpleNamespace
    import main
    import persistence
    client = TestClient(main.app)
    monkeypatch.setattr(main.app.state, 'models_loaded', True, raising=False)
    monkeypatch.setattr(main.app.state, 'models', object(), raising=False)
    enqueue = AsyncMock()
    monkeypatch.setattr(main, 'get_or_create_pipeline_manager', AsyncMock(return_value=SimpleNamespace(enqueue=enqueue)))
    response = client.post('/api/pipeline/start', data={'enforce_top_order': 'true', 'processing_mode': 'fast'},
                           files={'audio': ('synthetic.wav', b'offline synthetic input', 'audio/wav')})
    assert response.status_code == 200, response.text
    data = response.json()
    assert persistence.load_session(data['session_id'])['enforce_top_order'] is True
    pipeline = persistence.load_pipeline_job(data['pipeline_id'])
    assert main._pipeline_refs(pipeline)['options']['enforce_top_order'] is True
    assert pipeline['status'] == 'pending'
    enqueue.assert_awaited_once_with(data['pipeline_id'])


def test_pipeline_passes_setting_and_summaries_use_only_cleaned_assignments(agenda_model, monkeypatch):
    import main
    result = run(agenda_model, [(0, 0, 'confirmed', 0), (4, 3, 'unclear', 4)])
    rows = [dict(line_id=f'line-{i}', speaker='M', text='Originalberatung.', start=i, end=i+1) for i in range(12)]
    captured = {}
    def detector(*args, **kwargs):
        captured.update(kwargs)
        return result
    monkeypatch.setattr(main, 'segment_known_agenda', detector)
    monkeypatch.setattr(main, 'append_pipeline_warning', lambda *args: None)
    tops, assignments, info, _ = main.detect_pipeline_agenda('offline', rows,
        known_tops=TITLES, pdf_path=None, options={'enforce_top_order': True, 'processing_mode': 'fast'})
    assert captured['enforce_top_order'] is True
    seen = []
    def summary(title, text, **kwargs):
        seen.append((title, text, kwargs))
        raise main.LLMCallError('offline sentinel')
    monkeypatch.setattr(main, 'summarize_segment', summary)
    summaries, reviews = main.summarize_pipeline_segments('offline', transcript=rows, tops=tops,
        assignments=assignments, options={'processing_mode': 'fast'}, source_session=dict(
            transcript=rows, tops=tops, top_ids=IDS, assignments=assignments,
            agenda_proposals={'source': {'top_ids': IDS, 'transcript': rows}, 'result': dict(info, assignments=assignments)}))
    assert [t for t, _, _ in seen] == [TITLES[0]]
    assert len(seen[0][2]['source_lines']) == 4
    assert all(summaries[i] == '' and reviews[i]['empty_source'] for i in (1, 2, 3))
