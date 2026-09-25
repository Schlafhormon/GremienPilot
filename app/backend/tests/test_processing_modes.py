"""Mode contracts using synthetic documents and scripted models only."""
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from fastapi.testclient import TestClient

import agenda_detection
import durable_jobs
import extract_tops as pdf
import main
import persistence
import summary_grounding
import summarize
from conftest import FakeTranscriptionResult
from export_protocol import ProtocolMetadata, build_protocol_document
from pdf_fixtures import agenda, item, pdf_bytes
from processing_mode import agenda_complete, pdf_usable, policy, processing_scope
from test_main import configure_test_app, wait_until
from assignment_suggestions import TranscriptUtterance


def test_modes_are_isolated_and_restore_after_failure():
    barrier = Barrier(2)
    def run(mode):
        with processing_scope(mode):
            barrier.wait()
            assert policy().mode == mode
            with pytest.raises(RuntimeError), processing_scope('fast' if mode == 'slow' else 'slow'):
                raise RuntimeError()
            return policy().mode
    with ThreadPoolExecutor(2) as pool:
        assert list(pool.map(run, ['fast', 'slow'])) == ['fast', 'slow']
    assert policy().mode == 'slow'
    with pytest.raises(ValueError), processing_scope('turbo'):
        pass


@pytest.mark.parametrize('kinds', [('digital', 'digital'), ('scan',), ('digital', 'scan')])
def test_fast_pdf_covers_all_pages_without_audits(tmp_path, monkeypatch, kinds):
    monkeypatch.setenv('LLM_IMAGE_TOKENS', '1024')
    path = tmp_path / 'invitation.pdf'
    path.write_bytes(pdf_bytes(kinds))
    calls = []
    def request(config, prompt, content, schema):
        calls.append((content, schema))
        return json.dumps(agenda(range(1, len(kinds)+1), [item(f'p{n}', n) for n in range(1, len(kinds)+1)]))
    monkeypatch.setattr(pdf, '_request', request)
    if 'scan' not in kinds:
        monkeypatch.setattr(pdf, '_render_page', lambda *args: pytest.fail('Digital pages must not be rendered in Fast'))
    result = pdf.extract_agenda_data_from_pdf(path, processing_mode='fast')
    assert len(calls) == 1 and calls[0][1] is pdf.Agenda
    assert sum(c['type'] == 'image_url' for c in calls[0][0]) == kinds.count('scan')
    assert len(result.tops) == len(kinds)
    assert result.processing_complete and result.review_required
    assert result.review_status == 'skipped' and result.audits == []
    assert all(p['status'] == 'unreviewed' for p in result.pages)
    assert pdf_usable(result.to_dict(), 'fast')
    assert not pdf_usable(result.to_dict(), 'slow')


def test_fast_pdf_does_not_publish_blank_page_headers_as_tops(tmp_path, monkeypatch):
    path = tmp_path / 'invitation.pdf'
    path.write_bytes(pdf_bytes())
    real = item('p1-top', title='Haushalt')
    header = item('p1-header', number=None, title='Leere Seite / Seitenkopf')
    monkeypatch.setattr(pdf, '_request', lambda *args: json.dumps(agenda(items=[real, header])))
    result = pdf.extract_agenda_data_from_pdf(path, processing_mode='fast')
    assert result.tops == ['Haushalt']
    assert [entry['title'] for entry in result.items] == ['Haushalt']
    assert len(result.pages) == 1


def test_fast_pdf_invalid_answer_is_not_repaired(tmp_path, monkeypatch):
    path = tmp_path / 'invitation.pdf'
    path.write_bytes(pdf_bytes())
    calls = []
    monkeypatch.setattr(pdf, '_request', lambda *args: calls.append(args) or '{}')
    with pytest.raises(pdf.ExtractionError):
        pdf.extract_agenda_data_from_pdf(path, processing_mode='fast')
    assert len(calls) == 1


def test_fast_pdf_job_resumes_without_reextracting(tmp_path, monkeypatch):
    from test_durable_jobs import claimed
    path = tmp_path / 'invitation.pdf'
    path.write_bytes(pdf_bytes())
    calls = []
    monkeypatch.setattr(pdf, '_request', lambda *args: calls.append(args) or json.dumps(agenda()))
    job = durable_jobs.submit('pdf', {'path': str(path), 'processing_mode': 'fast'},
        documents=[durable_jobs.document(path)])
    with claimed(job) as (current, _):
        first, state = main.run_durable_job(current)
    assert state == 'review_required' and first['processing_complete']
    with claimed(job) as (current, _):
        second, state = main.run_durable_job(current)
    assert second == first and len(calls) == 1
    assert job['payload']['versions']['processing']['mode'] == 'fast'
    assert job['payload']['versions'] != durable_jobs.version_snapshot({'processing_mode': 'slow'})


@pytest.mark.parametrize('reuse_pdf', [False, True])
def test_fast_pipeline_accepts_unreviewed_pdf_but_slow_rejects_it(tmp_path, monkeypatch, agenda_model, summary_model, reuse_pdf):
    configure_test_app(tmp_path, monkeypatch)
    monkeypatch.setattr(pdf, '_request', lambda *args: json.dumps(agenda()))
    monkeypatch.setattr(main, 'transcribe_audio', lambda *args, **kwargs: FakeTranscriptionResult(
        transcript=[{'speaker': 'Rat', 'text': 'Der Haushalt wird beraten.', 'start': 0, 'end': 2}],
        audio_duration_seconds=2))
    with TestClient(main.app) as client:
        data = {'processing_mode': 'fast'}
        files = {'audio': ('meeting.mp3', b'audio', 'audio/mpeg')}
        if reuse_pdf:
            extracted = client.post('/api/extract-tops', data=data,
                files={'pdf': ('invitation.pdf', pdf_bytes(), 'application/pdf')})
            assert extracted.status_code == 200, extracted.text
            data['pdf_source_job_id'] = extracted.json()['document']['job_id']
            rejected = client.post('/api/pipeline/start', data={**data, 'processing_mode': 'slow'}, files=files)
            assert rejected.status_code == 422
        else:
            data['auto_detect_tops_from_pdf'] = 'true'
            files['pdf'] = ('invitation.pdf', pdf_bytes(), 'application/pdf')
        started = client.post('/api/pipeline/start', data=data, files=files)
        assert started.status_code == 200, started.text
        pid = started.json()['pipeline_id']
        assert wait_until(lambda: client.get(f'/api/pipeline/{pid}').json()['status'] in {'completed', 'failed'})
        result = client.get(f'/api/pipeline/{pid}/result').json()
        assert result['pipeline']['status'] == 'completed', result
        assert result['session']['pdf_extraction']['review_status'] == 'skipped'
        assert result['session']['summaries']['0']


@pytest.mark.parametrize('known', [True, False])
def test_fast_agenda_uses_one_unreviewed_compact_pass(agenda_model, known):
    rows = [TranscriptUtterance('Rat', 'Der Haushalt wird beraten.', start=0, end=2)]
    result = agenda_detection.segment_known_agenda(rows, ['Haushalt'] if known else [], processing_mode='fast')
    assert result.assignments == [0]
    assert result.llm.processing_complete and not result.llm.review_complete
    assert result.llm.status == 'success' and result.llm.review_status == 'skipped'
    assert agenda_complete(vars(result.llm))
    assert len(result.llm.reconstructions) == 1
    assert all(body['phase'].startswith('fast') for body, _ in agenda_model.calls)
    detail = [request for body, request in agenda_model.calls if body['phase'] == 'fast:detail']
    assert len(detail) == 1 and 'response' in detail[0]['response_format']['json_schema']['schema']['properties']
    assert result.llm.line_results[0]['grounding']['content_status'] == 'unreviewed'


def test_fast_agenda_technical_gaps_remain_failures(agenda_model):
    agenda_model.overrides['fast:detail'] = {'source_ranges': [], 'lines': []}
    result = agenda_detection.segment_known_agenda([TranscriptUtterance('Rat', 'Beratung')],
        ['Haushalt'], processing_mode='fast')
    assert result.llm.status == 'failed' and not result.llm.processing_complete
    assert not agenda_complete(vars(result.llm))
    assert len([b for b, _ in agenda_model.calls if b['phase'] == 'fast:detail']) == 1


@pytest.mark.parametrize('mode,fast_budget,override,expected', [
    ('fast', None, None, 8192),
    ('fast', '6144', None, 6144),
    ('slow', '8192', None, 4096),
    ('fast', '8192', '6000', 6000),
])
def test_agenda_mode_budget_reaches_planner_and_model(agenda_model, monkeypatch, mode, fast_budget, override, expected):
    monkeypatch.setenv('AGENDA_OUTPUT_TOKENS', '4096')
    for name, value in [('AGENDA_FAST_OUTPUT_TOKENS', fast_budget), ('LLM_OUTPUT_TOKENS', override)]:
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    result = agenda_detection.segment_known_agenda([TranscriptUtterance('Rat', 'Beratung.')],
        ['Haushalt'], processing_mode=mode)
    assert result.llm.processing_complete
    assert result.llm.provenance['planner']['output'] == expected
    assert all(request['max_tokens'] == expected for _, request in agenda_model.calls)
    if mode == 'fast':
        assert result.llm.provenance['planner']['attempts'] == 1
        assert result.llm.provenance['planner']['split_depth'] == 0
        assert not result.llm.review_complete


def test_fast_context_budget_preserves_every_source_without_review(agenda_model, monkeypatch):
    from llm_transport import fits, structured_output_budget
    from llm_config import get_llm_config
    monkeypatch.setenv('LLM_CONTEXT_TOKENS', '16384')
    monkeypatch.delenv('LLM_THINKING', raising=False)
    monkeypatch.setenv('AGENDA_FAST_OUTPUT_TOKENS', '8192')
    monkeypatch.delenv('LLM_OUTPUT_TOKENS', raising=False)
    rows = [TranscriptUtterance('Rat', f'Beitrag {i}: ' + 'Beratung zum Haushalt. '*12,
        line_id=f'line-{i}') for i in range(40)]
    result = agenda_detection.segment_known_agenda(rows, ['Haushalt'], processing_mode='fast')
    assert result.llm.processing_complete and result.assignments == [0]*len(rows)
    assert not result.llm.review_complete and result.llm.review_status == 'skipped'
    context_calls = [(body, request) for body, request in agenda_model.calls if body['phase'] == 'fast:context']
    assert len(context_calls) > 1  # Input must split when the larger output reserve is included.
    assert [r['index'] for body, _ in context_calls for r in body['sources'] if 'index' in r] == list(range(len(rows)))
    assert [r['index'] for body, _ in agenda_model.calls if body['phase'] == 'fast:detail'
            for r in body['target_lines']] == list(range(len(rows)))
    archive = result.llm.provenance['context_archive']
    assert [i for node in archive if node['level'] == 0
            for i in range(node['coverage'][0], node['coverage'][1]+1)] == list(range(len(rows)))
    assert len({json.dumps(body, sort_keys=True) for body, _ in context_calls}) == len(context_calls)
    for _, request in context_calls:
        schema = request['response_format']['json_schema']['schema']
        assert schema['properties']['evidence']['maxItems'] == 8
    config = get_llm_config()
    for body, request in agenda_model.calls:
        assert body['phase'].startswith('fast:')
        if body['phase'] == 'fast:detail':
            # Condensed notes must not override the endpoint/window protocol
            # with the numeric retrieval protocol used by preparation phases.
            assert 'source_ranges' not in ''.join(m['content'] for m in request['messages'])
        assert fits(request['messages'], structured_output_budget(config, request['max_tokens']),
                    config, request['response_format'])


@pytest.mark.parametrize('context,thinking,thinking_tokens,expected', [
    ('16384', '', '0', 8192),
    ('16384', 'true', '1024', 3072),
    ('32768', 'false', '0', 8192),
    ('12288', 'false', '0', 6144),
])
def test_fast_native_budget_leaves_room_for_sources(agenda_model, monkeypatch, context, thinking, thinking_tokens, expected):
    monkeypatch.setenv('LLM_PROVIDER', 'ollama')
    monkeypatch.setenv('LLM_FAST_REASONING_EFFORT', 'high' if thinking == 'true' else 'none')
    monkeypatch.setenv('LLM_THINKING_TOKENS', thinking_tokens)
    monkeypatch.setenv('LLM_CONTEXT_TOKENS', context)
    monkeypatch.setenv('AGENDA_FAST_OUTPUT_TOKENS', '8192')
    monkeypatch.delenv('LLM_OUTPUT_TOKENS', raising=False)
    result = agenda_detection.segment_known_agenda([TranscriptUtterance('Rat', 'Beratung.')],
        ['Haushalt'], processing_mode='fast')
    assert result.llm.processing_complete
    assert result.llm.provenance['planner']['output'] == expected
    assert all(c['reserved_output_tokens'] <= int(context)//2 for c in result.llm.chunks)
    assert all(request['max_tokens'] == expected for _, request in agenda_model.calls)
    assert all(body['phase'].startswith('fast:') for body, _ in agenda_model.calls)


@pytest.mark.parametrize('failure', ['length', 'too_many_anchors'])
def test_fast_context_failure_has_no_extra_attempt_or_review(agenda_model, monkeypatch, failure):
    from llm_transport import IncompleteResponseError
    monkeypatch.setenv('LLM_CONTEXT_TOKENS', '16384')
    monkeypatch.delenv('LLM_THINKING', raising=False)
    rows = [TranscriptUtterance('Rat', 'Beratung zum Haushalt. '*12, line_id=f'line-{i}') for i in range(40)]
    if failure == 'length':
        agenda_model.overrides['fast:context'] = IncompleteResponseError('LLM output incomplete (length)')
    else:
        agenda_model.overrides['fast:context'] = lambda body: {
            'narrative': 'Verlauf', 'evidence': [{'line_id': r['line_id']} for r in body['sources'][:9]]}
    result = agenda_detection.segment_known_agenda(rows, ['Haushalt'], processing_mode='fast')
    assert not result.llm.processing_complete
    assert result.assignments == [None]*len(rows)
    assert [body['phase'] for body, _ in agenda_model.calls] == ['fast:context']
    assert result.llm.failure_reasons == [
        'IncompleteResponseError' if failure == 'length' else 'context_evidence_limit_exceeded']


def test_fast_summary_is_one_call_and_export_is_labelled(summary_model):
    result = summarize.summarize_segment('Haushalt', 'Rat: Der Haushalt wird beraten.', processing_mode='fast')
    assert [body['phase'] for body, _ in summary_model.calls] == ['generate']
    assert result.llm_usage['processing_complete'] and result.llm_usage['review_status'] == 'skipped'
    assert not result.structured.verification['review_complete']
    assert result.structured.evidence[0]['grounding']['content_status'] == 'unreviewed'
    assert '[UNBESTÄTIGT' not in result.summary
    review = summarize.build_summary_review(structured=result.structured, summary=result.summary,
        lines=[{'speaker': 'Rat', 'text': 'Der Haushalt wird beraten.'}])
    assert any(w.kind == 'review_skipped' for w in review.warnings)
    assert not any(w.severity == 'error' for w in review.warnings)
    document = build_protocol_document(metadata=ProtocolMetadata(), tops=['Haushalt'],
        summaries={0: result.summary}, summary_reviews={0: {'structured': result.structured.to_dict()}})
    assert 'Fast – ohne automatische Inhaltsprüfung' in document.metadata.title


def test_fast_summary_preserves_long_sources_without_review(monkeypatch, summary_model):
    real_fits = summary_grounding.fits
    def fits(messages, *args):
        body = json.loads(messages[1]['content'])
        if len(body.get('source', [])) > 2:
            return False
        return real_fits(messages, *args)
    monkeypatch.setattr(summary_grounding, 'fits', fits)
    lines = ['Rat: Beratung ' + str(i) for i in range(9)]
    result = summarize.summarize_segment('Haushalt', '\n'.join(lines), source_lines=lines, processing_mode='fast')
    phases = [body['phase'] for body, _ in summary_model.calls]
    assert phases.count('generate') > 1 and phases.count('fast_consolidate') == 1
    assert set(phases) == {'generate', 'fast_consolidate'}
    assert [r['text'] for r in result.structured.verification['sources']] == lines


def test_mode_is_part_of_summary_cache_policy(tmp_path, monkeypatch, summary_model):
    monkeypatch.setenv('LLM_CACHE_DIR', str(tmp_path / 'cache'))
    fast = summarize.summarize_segment('Haushalt', 'Rat: Beratung.', processing_mode='fast')
    slow = summarize.summarize_segment('Haushalt', 'Rat: Beratung.', processing_mode='slow')
    phases = [body['phase'] for body, _ in summary_model.calls]
    assert phases.count('generate') == 2
    assert fast.llm_usage['policy']['processing_mode'] == 'fast'
    assert slow.llm_usage['review_complete'] and 'final_review' in phases


def test_session_mode_persists_validates_and_blocks_active_changes(tmp_path, monkeypatch):
    configure_test_app(tmp_path, monkeypatch)
    client = TestClient(main.app)
    created = client.post('/api/sessions', json={'processing_mode': 'fast'}).json()
    sid = created['session_id']
    assert client.get(f'/api/sessions/{sid}').json()['processing_mode'] == 'fast'
    assert client.get('/api/sessions').json()['items'][0]['processing_mode'] == 'fast'
    assert client.put(f'/api/sessions/{sid}', json={}).json()['processing_mode'] == 'fast'
    assert client.put(f'/api/sessions/{sid}', json={'processing_mode': 'turbo'}).status_code == 422
    assert client.post('/api/sessions', json={}).json()['processing_mode'] == 'slow'
    persistence.save_pipeline_job('active', {'session_id': sid, 'status': 'processing', 'stage': 'agenda_detect'})
    assert client.put(f'/api/sessions/{sid}', json={'processing_mode': 'slow'}).status_code == 409
    assert persistence.load_session(sid)['processing_mode'] == 'fast'


def test_existing_sessions_migrate_to_slow_without_losing_content():
    persistence.save_session('legacy', {'tops': ['Haushalt'], 'summaries': {0: 'Bestand'}})
    with persistence.connect() as db:
        db.execute('ALTER TABLE sessions DROP COLUMN processing_mode')
    persistence.init_db()
    restored = persistence.load_session('legacy')
    assert restored['processing_mode'] == 'slow'
    assert restored['summaries'] == {0: 'Bestand'}


@pytest.mark.parametrize('mode', ['fast', 'slow'])
@pytest.mark.parametrize('no_tops', [False, True])
def test_pipeline_and_selective_regeneration_keep_mode(tmp_path, monkeypatch, agenda_model, summary_model, mode, no_tops):
    configure_test_app(tmp_path, monkeypatch)
    monkeypatch.setattr(main, 'transcribe_audio', lambda *args, **kwargs: FakeTranscriptionResult(
        transcript=[{'speaker': 'Rat', 'text': 'Der Haushalt wird beraten.', 'start': 0, 'end': 2}],
        audio_duration_seconds=2))
    with TestClient(main.app) as client:
        started = client.post('/api/pipeline/start', data={'processing_mode': mode,
            'tops': json.dumps([] if no_tops else ['Haushalt']), 'skip_agenda_detection': str(no_tops).lower()},
            files={'audio': ('meeting.mp3', b'audio', 'audio/mpeg')})
        assert started.status_code == 200, started.text
        job = started.json()
        assert wait_until(lambda: client.get(f"/api/pipeline/{job['pipeline_id']}").json()['status'] in {'completed', 'failed'})
        result = client.get(f"/api/pipeline/{job['pipeline_id']}/result").json()
        assert result['pipeline']['status'] == 'completed', result
        session = result['session']
        assert session['processing_mode'] == mode and session['summaries']['0']
        assert session['summary_reviews']['0']['llm_usage']['processing_mode'] == mode
        assert durable_jobs.load(job['pipeline_id'])['payload']['versions']['processing']['mode'] == mode
        sid = session['session_id']
        top_id = session['top_ids'][0] if not no_tops else f'whole-session:{sid}'
        regenerated = client.post(f'/api/sessions/{sid}/summary-jobs', json={
            'top_ids': [top_id], 'revision': session['revision']})
        assert regenerated.status_code == 200, regenerated.text
        summary_id = regenerated.json()['summary_job_id']
        assert wait_until(lambda: client.get(f'/api/summary-jobs/{summary_id}').json()['status'] in {'completed', 'failed'})
        assert client.get(f'/api/summary-jobs/{summary_id}').json()['status'] == 'completed'
        assert persistence.load_summary_job(summary_id)['refs']['processing_mode'] == mode
        assert client.get(f'/api/sessions/{sid}').json()['summary_reviews']['0']['llm_usage']['processing_mode'] == mode


def test_fast_manual_edit_keeps_plain_text_and_exportable_unreviewed_provenance(summary_model):
    generated = summarize.summarize_segment('Haushalt', 'Rat: Beratung.', processing_mode='fast')
    initial = {'tops': ['Haushalt'], 'top_ids': ['a'],
        'transcript': [{'line_id': 'l1', 'speaker': 'Rat', 'text': 'Beratung.', 'start': 0, 'end': 2}],
        'assignments': [0], 'summaries': {0: generated.summary},
        'summary_reviews': {0: {'structured': generated.structured.to_dict(), 'llm_usage': generated.llm_usage}},
        'processing_mode': 'fast'}
    saved = persistence.save_session('edited', initial)
    changed = main.reconcile_session_summaries(saved, {**saved, 'summaries': {0: 'Manuell ergänzte Beratung.'}})
    assert changed['summaries'][0] == 'Manuell ergänzte Beratung.'
    review = changed['summary_reviews'][0]
    assert review['source_links'] == [] and review['structured']['evidence'] == []
    assert review['llm_usage']['processing_complete'] and review['llm_usage']['origin'] == 'manual_edit'
    assert review['structured']['verification']['review_status'] == 'skipped'
    persistence.save_session('edited', changed)
    client = TestClient(main.app)
    exported = client.post('/api/export', json={'session_id': 'edited', 'format': 'txt',
        'tops': changed['tops'], 'summaries': changed['summaries'], 'metadata': {}})
    assert exported.status_code == 200, exported.text
    assert 'Manuell ergänzte Beratung.' in exported.text
    assert 'Fast – ohne automatische Inhaltsprüfung' in exported.text
