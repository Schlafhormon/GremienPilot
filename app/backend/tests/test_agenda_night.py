import asyncio
from dataclasses import replace, asdict
import json
from types import SimpleNamespace

import httpx
import pytest

import agenda_runtime as runtime
import agenda_timeline
from agenda_detection import AgendaLLMUsage
from assignment_suggestions import TranscriptUtterance
from extract_tops import parse_agenda_data_response
from summarize import get_llm_config


class Stream(httpx.AsyncByteStream):
    def __init__(self, events):
        self.events, self.closed = events, False

    async def __aiter__(self):
        for delay, data in self.events:
            await asyncio.sleep(delay)
            yield (json.dumps(data) + '\n').encode()

    async def aclose(self):
        self.closed = True


def stream_call(events, **options):
    config = replace(get_llm_config(), task='agenda', timeout_seconds=.05,
                     idle_timeout_seconds=.05, total_timeout_seconds=1, **options)
    stream = Stream(events)
    factory = lambda **kw: httpx.AsyncClient(**kw, transport=httpx.MockTransport(
        lambda req: httpx.Response(200, stream=stream)))
    return config, stream, factory


def test_active_generation_survives_first_response_limit_and_requires_final_json():
    config, stream, factory = stream_call([
        (.01, {'message': {'content': '{'}}), (.03, {'message': {'content': '"ok":'}}),
        (.03, {'message': {'content': 'true}'}}), (.03, {'done': True, 'eval_count': 4})])
    progress = []
    with runtime.observe(progress.append):
        data = asyncio.run(runtime.read_stream(config, {'messages': [{'content': json.dumps({
            'phase': 'timeline', 'target_start': 3, 'target_end': 9})}]}, factory))
    assert data['message']['content'] == '{"ok":true}'
    assert data['eval_count'] == 4 and stream.closed
    assert progress[-1]['phase'] == 'validating'
    assert progress[-1]['step'] == 'timeline' and progress[-1]['target_start'] == 3


@pytest.mark.parametrize('events,error', [
    ([(.2, {'done': True})], runtime.AgendaFirstResponseTimeout),
    ([(0, {'message': {'content': '{'}}), (.2, {'done': True})], runtime.AgendaInactivityTimeout),
    ([(0, {'message': {'content': '{'}})], runtime.AgendaStreamIncomplete),
    ([(0, {'error': 'private error'})], runtime.AgendaStreamIncomplete),
])
def test_stalled_and_incomplete_streams_are_closed(events, error):
    config, stream, factory = stream_call(events)
    with pytest.raises(error):
        asyncio.run(runtime.read_stream(config, {}, factory))
    assert stream.closed


def test_total_deadline_and_cancellation_close_stream():
    config, stream, factory = stream_call([(.01, {'message': {'content': 'x'}})] * 50)
    with pytest.raises(runtime.AgendaTotalTimeout):
        asyncio.run(runtime.read_stream(replace(config, total_timeout_seconds=.06), {}, factory))
    assert stream.closed
    config, stream, factory = stream_call([(0, {'message': {'content': 'x'}}), (.2, {'done': True})])
    def cancel(state):
        if state['response_chunks']:
            raise RuntimeError('cancelled')
    with runtime.observe(cancel), pytest.raises(RuntimeError, match='cancelled'):
        asyncio.run(runtime.read_stream(config, {}, factory))
    assert stream.closed


def test_pdf_reconciles_occurrences_artifacts_and_original_multiline_titles():
    source = '''Tagesordnung
Öffentlicher Teil
01. Beratung über den Ausbau
der Schule
_______________________________
2.1. Finanzierung
Seite 1 von 2
7. Anfragen
Nichtöffentlicher Teil
01. Grundstücke
01. Weitere Vergabe
gez. Erika Beispiel
1. stellv. Vorsitzende des Ausschusses
'''
    result = parse_agenda_data_response(json.dumps({'tops': [
        {'number': '01', 'section': 'public', 'title': 'Beratung über den Ausbau der Schule'},
        {'number': '1', 'title': 'stellv. Vorsitzende'}]}), source)
    assert len(result.tops) == 5
    assert result.tops[0] == '[Öffentlich] 01. Beratung über den Ausbau der Schule'
    assert result.tops[-2:] == ['[Nichtöffentlich] 01. Grundstücke', '[Nichtöffentlich] 01. Weitere Vergabe']
    assert all('___' not in t and 'gez.' not in t and 'Vorsitzende' not in t for t in result.tops)
    assert result.provenance['candidates'][0]['origin'] == 'source_and_model'
    assert result.provenance['requires_review']
    assert any(c['kind'] == 'model_item_not_confirmed_by_source' for c in result.provenance['conflicts'])


def test_timeline_resume_skips_failed_parent_and_successful_child(monkeypatch, tmp_path):
    monkeypatch.setenv('LLM_CACHE_DIR', str(tmp_path))
    config = replace(get_llm_config(), task='agenda', context_budget=32768)
    transcript = [TranscriptUtterance('M', f'Originalzeile {i}') for i in range(12)]
    seen = []
    broken = [True]
    def complete(*args, **kw):
        b = json.loads(kw['messages'][1]['content'])
        span = (b['target_start'], b['target_end'])
        seen.append(span)
        if span == (0, 11):
            raise runtime.AgendaInactivityTimeout()
        if span[0] and broken[0]:
            raise httpx.ConnectError('service stopped')
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"events":[]}'))])
    monkeypatch.setattr(agenda_timeline, 'complete', complete)
    with pytest.raises(httpx.ConnectError):
        agenda_timeline.analyze(None, config, transcript, [], AgendaLLMUsage(True, 'test'), {'digest': 'one'})
    assert (0, 5) in seen
    seen.clear()
    broken[0] = False
    result = agenda_timeline.analyze(None, config, transcript, [], AgendaLLMUsage(True, 'test'), {'digest': 'one'})
    assert (0, 11) not in seen and (0, 5) not in seen
    assert {i for a, b in result['coverage'] for i in range(a, b+1)} == set(range(12))
    assert result['seams']
    seen.clear()
    agenda_timeline.analyze(None, config, transcript, [], AgendaLLMUsage(True, 'test'), {'digest': 'two'})
    assert (0, 11) in seen


def test_technical_gate_separates_semantic_uncertainty_and_model_errors():
    from main import agenda_technical_check
    semantic = {'llm': {'status': 'success', 'gaps': [{'kind': 'semantic', 'reason': 'Mehrdeutig'}]}, 'uncertain_count': 2}
    assert agenda_technical_check(semantic, 10)['passed']
    for usage in ({'status': 'failed'}, {'status': 'partial_failure'},
                  {'status': 'success', 'gaps': [{'kind': 'technical'}]},
                  {'status': 'success', 'provenance': {'completion': {'unresolved_reviews': [1]}}}):
        assert not agenda_technical_check({'llm': usage}, 10)['passed']


def test_persisted_job_resumes_without_transcription_or_changing_source(tmp_path, monkeypatch):
    import main
    import persistence as p
    from fastapi.testclient import TestClient
    from test_main import configure_test_app, wait_until
    configure_test_app(tmp_path, monkeypatch)
    monkeypatch.setenv('AGENDA_MODEL_SETTINGS_PATH', str(tmp_path / 'agenda.json'))
    transcript = [{'line_id': 'stable-line', 'speaker': 'M', 'text': 'Haushalt.', 'start': 1, 'end': 2}]
    p.save_job('source-job', {'status': 'completed', 'progress': 100, 'transcript': transcript})
    p.save_session('source-session', {'job_id': 'source-job', 'transcript': transcript,
                                     'tops': ['1 Haushalt'], 'top_ids': ['stable-top'], 'assignments': [0]})
    before = p.load_session('source-session')
    monkeypatch.setattr(main, 'run_transcription', lambda *a: pytest.fail('Audio retranscribed'))
    monkeypatch.setattr(main, 'build_speaker_suggestion_responses', lambda *a: [])
    calls = []
    broken = [True]
    def detect(*args, **kw):
        calls.append(kw['options']['agenda_cache_namespace'])
        return ['1 Haushalt'], [None if broken[0] else 0], {
            'llm': asdict(AgendaLLMUsage(True, 'test', status='failed' if broken[0] else 'success')),
            'segments': [], 'strategy': 'test', 'uncertain_count': 0}, {}
    monkeypatch.setattr(main, 'detect_pipeline_agenda', detect)
    summaries = []
    monkeypatch.setattr(main, 'summarize_pipeline_segments', lambda *a, **kw: (summaries.append(kw) or ({0: 'Beschluss.'}, {})))
    with TestClient(main.app) as client:
        started = client.post('/api/sessions/source-session/agenda-jobs', json={})
        assert started.status_code == 200
        job = started.json()
        route = '/api/pipeline/' + job['pipeline_id']
        assert wait_until(lambda: client.get(route).json()['status'] == 'failed')
        assert not summaries
        assert client.get(route + '/result').status_code == 200
        assert wait_until(lambda: not p.load_pipeline_job(job['pipeline_id'])['result_refs'].get('worker_active'))
        broken[0] = False
        assert client.post(route + '/resume').status_code == 200
        assert wait_until(lambda: client.get(route).json()['status'] == 'completed')
        target = p.load_session(job['session_id'])
        assert target['top_ids'] == ['stable-top'] and target['transcript'] == before['transcript']
    assert len(summaries) == 1 and calls[0] == calls[1]
    assert p.load_session('source-session') == before


def test_single_mode_does_not_rewrite_persisted_dual_settings(tmp_path, monkeypatch):
    import agenda_model as model
    monkeypatch.setenv('AGENDA_MODEL_SETTINGS_PATH', str(tmp_path / 'settings.json'))
    model.save_settings(model.AgendaModelSettings(enabled=True))
    before = model.settings_path().read_bytes()
    monkeypatch.setenv('AGENDA_MODEL_FORCE_DISABLED', 'true')
    assert not model.load_settings().enabled
    with pytest.raises(model.AgendaModelError):
        model.save_settings(model.AgendaModelSettings(enabled=True))
    assert model.settings_path().read_bytes() == before


def test_invalid_json_is_not_reinterpreted_as_dozen_fake_title_lines():
    raw = '{"tops": [{"number":"1", "title":"Baugebiet "Nord""}]}'
    result = parse_agenda_data_response(raw, 'Tagesordnung\n1 Baugebiet „Nord“')
    assert result.tops == ['1 Baugebiet „Nord“']
    assert result.provenance['raw_model_response'] == raw
    assert result.provenance['model_tops'] == []
    assert result.provenance['conflicts'][0]['kind'] == 'invalid_model_json'


def test_line_assignment_resume_keeps_successful_split_children(monkeypatch, tmp_path, fake_openai_module):
    import agenda_llm
    from agenda_detection import segment_known_agenda
    monkeypatch.setenv('AGENDA_MODEL_SETTINGS_PATH', str(tmp_path / 'settings.json'))
    monkeypatch.setenv('LLM_CACHE_DIR', str(tmp_path / 'cache'))
    monkeypatch.setenv('AGENDA_DETECTION_BOUNDARY_REVIEW_MAX_CALLS', '0')
    monkeypatch.setenv('AGENDA_DETECTION_GAP_REVIEW_MAX_CALLS', '0')
    monkeypatch.setattr(agenda_llm, 'model_fingerprint', lambda c: {'model': c.model, 'digest': 'test', 'provider': 'ollama'})
    calls, broken = [], [True]
    def generate(*args, **kw):
        body = json.loads(kw['messages'][1]['content'])
        a, b = body['target_start'], body['target_end']
        calls.append((a, b))
        if (a, b) == (0, 3) or (broken[0] and a == 2):
            raise TimeoutError()
        data = {'assignments': {str(i): 'unspecified:1' for i in range(a, b+1)}, 'gaps': [], 'uncertain_lines': []}
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(data)))])
    monkeypatch.setattr(agenda_llm, 'complete', generate)
    transcript = [TranscriptUtterance('M', 'Wir beraten den Haushalt.') for _ in range(4)]
    first = segment_known_agenda(transcript, ['1 Haushalt'], use_llm=True)
    assert first.llm.status == 'partial_failure'
    assert first.assignments == [0, 0, None, None]
    calls.clear()
    broken[0] = False
    second = segment_known_agenda(transcript, ['1 Haushalt'], use_llm=True)
    assert second.llm.status == 'success' and second.assignments == [0] * 4
    assert calls == [(2, 3)]


def test_amendment_keeps_original_evidence_without_inventing_number_offset():
    from agenda_context import EvidenceContext, model_agenda
    tops = ['[Öffentlich] 1 Haushalt', '[Öffentlich] 2 Schule', '[Nichtöffentlich] 9 Vergabe']
    texts = ['Wir kommen zu TOP 1, Haushalt.',
             'Ich möchte einen neuen Tagesordnungspunkt einfügen.',
             'Der Antrag wird abgelehnt.',
             'Wir kommen zu TOP 2, Haushalt.'] + ['Beratung.'] * 30 + [
             'Ich eröffne den nichtöffentlichen Teil.', 'Wir kommen zu TOP 9, Vergabe.']
    context = EvidenceContext([TranscriptUtterance('M', t) for t in texts], tops, model_agenda(tops))
    assert context.at(0)['topic']['top_id'] == 'public:1'
    assert context.at(0)['numbering_changes'] is None
    assert context.at(3)['topic'] is None  # No wrong automatic public:2 anchor.
    packet = context.packet(30, 31)['numbering_changes']
    assert packet['kind'] == 'possible_agenda_amendment'
    assert any(r['text'] == 'Der Antrag wird abgelehnt.' for r in packet['original_evidence'])
    assert 'offset' not in packet
    assert context.at(len(texts)-1)['numbering_changes'] is None
    assert context.at(len(texts)-1)['topic']['top_id'] == 'nonpublic:9'


@pytest.mark.parametrize('amended', [True, False])
def test_correct_local_title_can_override_shifted_number_only_with_amendment_evidence(
        amended, fake_openai_module, monkeypatch, tmp_path):
    from test_agenda_llm import classify, row
    monkeypatch.setenv('AGENDA_MODEL_SETTINGS_PATH', str(tmp_path / 'agenda.json'))
    monkeypatch.setenv('LLM_CACHE_DIR', str(tmp_path / 'cache'))
    monkeypatch.setenv('LLM_REPAIR_SPLIT_DEPTH', '0')
    monkeypatch.setenv('AGENDA_DETECTION_BOUNDARY_REVIEW_MAX_CALLS', '0')
    monkeypatch.setenv('AGENDA_DETECTION_GAP_REVIEW_MAX_CALLS', '0')
    texts = ['Wir fügen einen neuen Tagesordnungspunkt ein.' if amended else 'Wir beginnen.',
             'Damit rutschen alle weiteren Tagesordnungspunkte nach hinten.' if amended else 'Beratung.',
             'Wir kommen zu TOP 3, Schulneubau.']
    # Separate proposal and renumbering evidence; neither invents a new ID.
    rows = [row(0, 1, 'unspecified:1', texts[0]), row(2, 2, 'unspecified:2', texts[2])]
    result = classify(fake_openai_module, rows, texts, ['1 Tagesordnung', '2 Schulneubau', '3 Abwasser'])
    if amended:
        assert result.assignments == [0, 0, 1]
        assert result.llm.status == 'success'
    else:
        assert result.llm.status == 'failed'
        assert 'contradictory_current_call' in result.llm.validation_reasons


def test_closing_clause_scope_is_not_overridden_by_next_section_announcement():
    from agenda_context import EvidenceContext, model_agenda
    tops = ['[Öffentlich] 8 Schließung', '[Nichtöffentlich] 6 Schließung']
    text = 'Dann schließe ich TOP 7 und schließe im Anschluss die öffentliche Sitzung und bitte die Nicht-Öffentlichkeit herzustellen.'
    context = EvidenceContext([TranscriptUtterance('M', text)], tops, model_agenda(tops))
    assert context.at(0)['topic']['top_id'] == 'public:8'


@pytest.mark.parametrize('text,following,expected', [
    ('Dann würde ich jetzt den nicht öffentlichen Teil schließen.', 'Danke, einen guten Nachhauseweg.', True),
    ('Ich würde jetzt die öffentliche Sitzung schließen.', 'Einen guten Heimweg.', True),
    ('Dann würde ich jetzt den nicht öffentlichen Teil schließen.', 'Es gibt noch eine Sachfrage.', False),
    ('Ich würde jetzt zu TOP 6 kommen, zur Schließung.', 'Einen guten Heimweg.', False),
    ('Ich werde die Sitzung schließen.', 'Einen guten Heimweg.', False),
    ('Ich schließe die Sitzung nicht.', 'Einen guten Heimweg.', False),
    ('Gestern sagte ich: Ich schließe die Sitzung.', 'Einen guten Heimweg.', False),
    ('Soll ich die Sitzung schließen?', 'Einen guten Heimweg.', False),
    ('Ich schließe mich der Kritik an der Sitzung an.', 'Einen guten Heimweg.', False),
])
def test_polite_closing_requires_original_farewell_and_preserves_negation(text, following, expected):
    from agenda_context import closing_act
    assert closing_act(text, following) is expected


def test_pdf_reconciliation_accepts_typography_but_never_truncated_titles_or_different_papers():
    source = '1 123/26 Ausbau „Nord“\n- Finanzierung\n2 124/26 Antrag und Batteriespeicher\n3 125/26 Haushalt'
    result = parse_agenda_data_response(json.dumps({'tops': [
        {'number': '1', 'title': 'Ausbau "Nord"'},
        {'number': '2', 'title': 'Antrag'},
        {'number': '3', 'title': '126/26 Haushalt'}]}), source)
    assert result.tops == ['1 123/26 Ausbau „Nord“', '2 124/26 Antrag und Batteriespeicher', '3 125/26 Haushalt']
    candidates = result.provenance['candidates']
    assert candidates[0]['origin'] == 'source_and_model'
    assert candidates[0]['subpoints'][0]['text'] == '- Finanzierung'
    assert candidates[1]['origin'] == candidates[2]['origin'] == 'numbered_source'
    assert len(result.provenance['conflicts']) == 4


def test_pipeline_progress_cannot_overwrite_concurrent_cancellation(monkeypatch):
    import copy
    import threading
    import main
    state = {'status': 'processing', 'result_refs': {'cancel_requested': False}}
    loaded, release, cancel_loaded = threading.Event(), threading.Event(), threading.Event()
    def read(_):
        snapshot = copy.deepcopy(state)
        if threading.current_thread().name == 'progress':
            loaded.set()
            assert release.wait(3)
        else:
            cancel_loaded.set()
        return snapshot
    def write(_, job):
        state.clear()
        state.update(copy.deepcopy(job))
        return job
    monkeypatch.setattr(main, 'load_pipeline_job', read)
    monkeypatch.setattr(main, 'save_pipeline_job', write)
    progress = threading.Thread(name='progress', target=lambda: main.save_pipeline_state('owned', result_refs={'agenda_progress': {'tokens': 3}}))
    cancel = threading.Thread(name='cancel', target=lambda: main.save_pipeline_state('owned', status='cancelled', result_refs={'cancel_requested': True}))
    progress.start()
    assert loaded.wait(3)
    cancel.start()
    try:
        assert not cancel_loaded.wait(.05)
    finally:
        release.set()
        progress.join(3)
        cancel.join(3)
    assert state['status'] == 'cancelled' and state['result_refs']['cancel_requested']
    assert state['result_refs']['agenda_progress']['tokens'] == 3


def test_user_cancellation_is_not_split_retried_or_reported_as_model_error(monkeypatch, tmp_path, fake_openai_module):
    import agenda_llm
    from agenda_detection import segment_known_agenda
    monkeypatch.setenv('AGENDA_MODEL_SETTINGS_PATH', str(tmp_path / 'agenda.json'))
    calls = []
    def cancel(*a, **kw):
        calls.append(1)
        raise runtime.OperationCancelled()
    monkeypatch.setattr(agenda_llm, 'complete', cancel)
    with pytest.raises(runtime.OperationCancelled):
        segment_known_agenda([TranscriptUtterance('M', 'Beratung.')] * 4, ['1 Haushalt'], use_llm=True)
    assert len(calls) == 1


def test_uncertain_boundaries_and_timeline_disagreement_receive_bounded_independent_review(
        monkeypatch, tmp_path, fake_openai_module):
    import agenda_llm
    monkeypatch.setenv('LLM_CACHE_DIR', str(tmp_path))
    config = replace(get_llm_config(), task='agenda', context_budget=32768)
    timeline = {'identity': 'overview', 'events': [{'index': 8, 'top_id': 'unspecified:2',
        'uncertain': False, 'contradictions': [], 'reason': 'Unbestätigte Hypothese'}]}
    monkeypatch.setattr(agenda_timeline, 'analyze', lambda *a, **k: timeline)
    calls = []
    def complete(*a, **kw):
        body = json.loads(kw['messages'][1]['content'])
        calls.append(body)
        start, end = body['target_start'], body['target_end']
        # Local evidence wins over the inconsistent global proposal again.
        response = {'assignments': {str(i): 'unspecified:1' for i in range(start, end+1)},
                    'gaps': [], 'uncertain_lines': [4] if len(calls) == 1 else []}
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(response)))])
    monkeypatch.setattr(agenda_llm, 'complete', complete)
    transcript = [TranscriptUtterance('M', 'Wir beraten den Haushalt.')] * 12
    usage = AgendaLLMUsage(True, 'test')
    segments = agenda_llm._classify(transcript, ['1 Haushalt', '2 Schule'], usage, config,
                                   fingerprint={'model': config.model, 'digest': 'test'})
    reviewed = {i for chunk in usage.chunks if chunk.get('phase') == 'boundary_review'
                for i in range(chunk['start_index'], chunk['end_index']+1)}
    assert {4, 8} <= reviewed
    assert len(calls) == 2  # Merged window, no recursive voting loop.
    assert all(segment.top_index == 0 for segment in segments)
    assert {t['kind'] for t in usage.provenance['review_triggers']} == {
        'uncertain_assignment_boundary', 'timeline_uncertainty_or_disagreement'}


@pytest.mark.parametrize('error,word', [
    (runtime.AgendaFirstResponseTimeout(), 'Erstantwortlimit'),
    (runtime.AgendaInactivityTimeout(), 'Inaktivitätslimit'),
    (runtime.AgendaTotalTimeout(), 'Gesamtzeitlimit'),
    (runtime.AgendaStreamIncomplete(), 'Teil-JSON'),
    (httpx.ConnectTimeout('private'), 'Verbindungsaufbau'),
    (httpx.ConnectError('private'), 'Verbindung'),
])
def test_runtime_failure_categories_have_distinct_content_free_user_messages(error, word):
    from agenda_model import report_error
    message = report_error(get_llm_config(), error)
    assert word in message and 'Kein Ersatzmodell' in message and 'private' not in message


@pytest.mark.parametrize('text,expected', [
    ('Kommen zum Tagesordnungspunkt 5, die Beschlusskontrolle.', 'call'),
    ('Und nun kommen wir zum Tagesordnungspunkt 5.', 'call'),
    ('Damit kommen wir zu TOP 5.', 'call'),
    ('Kommen morgen zu TOP 5.', 'mention'),
    ('Dann kommen wir auch hier zum Tagesordnungspunkt 3, Bericht.', 'call'),
    ('Somit komme ich auch schon zum zweiten Tagesordnungspunkt.', 'call'),
    ('Ist jetzt nicht der Fall, dann würde ich zum Tagesordnungspunkt 4 kommen und zwar zur Vergabe.', 'call'),
    ('Wenn das nicht der Fall wäre, würde ich zu TOP 4 kommen.', 'mention'),
    ('Ist jetzt nicht der Fall, dann würde ich morgen zu TOP 4 kommen.', 'mention'),
    ('Dann kommen wir auch hier nicht zu TOP 3.', 'mention'),
    ('Gestern sagte sie: Dann kommen wir auch hier zu TOP 3.', 'mention'),
])
def test_current_calls_with_adverbs_and_resolved_questions_are_not_hypothetical(text, expected):
    from assignment_suggestions import transition_kind
    assert transition_kind(text) == expected


@pytest.mark.parametrize('call', [
    'Ist jetzt nicht der Fall, dann würde ich zum Tagesordnungspunkt 2 kommen und zwar zur Schule.',
    'Kommen zum Tagesordnungspunkt 2, Schule.',
])
def test_resolved_polite_call_can_change_previous_topic(call, fake_openai_module, monkeypatch, tmp_path):
    from test_agenda_llm import classify, row
    monkeypatch.setenv('AGENDA_MODEL_SETTINGS_PATH', str(tmp_path / 'agenda.json'))
    texts = ['Wir kommen zu TOP 1, Haushalt.', call]
    result = classify(fake_openai_module, [row(0, 0, 'unspecified:1', texts[0]),
                      row(1, 1, 'unspecified:2', texts[1])], texts, ['1 Haushalt', '2 Schule'])
    assert result.assignments == [0, 1] and result.llm.status == 'success'


def test_original_closing_act_overrides_mention_guard_for_composite_numbered_row(fake_openai_module, monkeypatch, tmp_path):
    from test_agenda_llm import classify, row
    monkeypatch.setenv('AGENDA_MODEL_SETTINGS_PATH', str(tmp_path / 'agenda.json'))
    monkeypatch.setenv('AGENDA_DETECTION_BOUNDARY_REVIEW_MAX_CALLS', '0')
    texts = ['Wir kommen zu TOP 1, Einwohnerfragestunde.',
             'Dann schließe ich TOP 2 und schließe im Anschluss die öffentliche Sitzung und bitte die Nicht-Öffentlichkeit herzustellen.']
    result = classify(fake_openai_module, [row(0, 0, 'public:1', texts[0]),
        row(1, 1, 'public:2', texts[1])], texts, ['[Öffentlich] 1 Einwohnerfragestunde', '[Öffentlich] 2 Schließung'])
    assert result.assignments == [0, 1] and result.llm.status == 'success'
