"""Restart/crash, concurrency and long work tests: only temporary data and simulated models."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import durable_jobs as jobs
import llm_transport as transport
from llm_config import get_llm_config
import main
import persistence


def wait(predicate, seconds=5):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.01)
    assert predicate()


@contextmanager
def claimed(job, owner='worker'):
    with persistence.connect() as db:
        db.execute("UPDATE durable_jobs SET lease_until=0,available_at=0 WHERE job_id=?", (job['job_id'],))
    current = jobs.claim(owner, 60)
    assert current is not None
    ctx = jobs.Runtime(current, owner, threading.Event())
    token = jobs.CURRENT.set(ctx)
    try:
        yield current, ctx
    finally:
        jobs.CURRENT.reset(token)


def test_transactional_claim_and_stale_worker_fencing():
    job = jobs.submit('test', {'source': 'frozen'})
    barrier = threading.Barrier(2)
    def claim(owner):
        barrier.wait()
        return jobs.claim(owner, 60)
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(claim, ['a', 'b']))
    assert sum(r is not None for r in results) == 1
    old = next(r for r in results if r)
    old_ctx = jobs.Runtime(old, old['owner'], threading.Event())
    with persistence.connect() as db:
        db.execute('UPDATE durable_jobs SET lease_until=0')
    new = jobs.claim('replacement', 60)
    assert new['attempt'] == 2
    token = jobs.CURRENT.set(old_ctx)
    try:
        with pytest.raises(transport.LLMCancelledError):
            jobs.checkpoint('late', lambda: 'bad')
        with pytest.raises(transport.LLMCancelledError):
            persistence.save_session('s', {'tops': ['bad']})
    finally:
        jobs.CURRENT.reset(token)
    assert persistence.load_session('s') is None


def test_process_lock_rejects_second_process():
    script = "import durable_jobs; durable_jobs.ProcessLock().__enter__()"
    with jobs.ProcessLock():
        other = subprocess.run([sys.executable, '-c', script], cwd=Path(main.__file__).parent,
                               capture_output=True, text=True)
    assert other.returncode != 0
    assert 'Only one backend process' in other.stderr
    with jobs.ProcessLock():
        pass


def test_real_process_crash_resumes_only_unfinished_step(tmp_path):
    job = jobs.submit('test', {'input': 'immutable'})
    script = '''
import os, threading
import durable_jobs as j
with j.ProcessLock():
    job = j.claim('dead-process', 3600)
    j.CURRENT.set(j.Runtime(job, 'dead-process', threading.Event()))
    j.checkpoint('validated-first', lambda: {'checked': 'first result'})
    os._exit(17)
'''
    crashed = subprocess.run([sys.executable, '-c', script], cwd=Path(main.__file__).parent)
    assert crashed.returncode == 17
    calls = []
    def runner(_):
        first = jobs.checkpoint('validated-first', lambda: calls.append('repeated'))
        second = jobs.checkpoint('second', lambda: calls.append('second') or 'second result')
        return [first, second], 'completed'
    manager = jobs.Manager(runner)
    asyncio.run(manager.start())
    wait(lambda: jobs.load(job['job_id'])['state'] == 'completed')
    asyncio.run(manager.stop())
    assert calls == ['second']
    assert jobs.load(job['job_id'])['attempt'] == 2


def test_shutdown_cancels_silent_stream_and_resumes_without_repeating_checkpoint(monkeypatch):
    job = jobs.submit('test', {})
    streaming = threading.Event()
    closed = threading.Event()
    calls = []
    async def silent(*args, **kwargs):
        streaming.set()
        try:
            await asyncio.sleep(10000)
        finally:
            closed.set()
    monkeypatch.setattr(transport, '_generate', silent)
    def runner(_):
        jobs.checkpoint('first', lambda: calls.append('first') or {'valid': True})
        transport.complete(None, get_llm_config(), messages=[], max_tokens=10)
        return {}, 'completed'
    manager = jobs.Manager(runner)
    asyncio.run(manager.start())
    assert streaming.wait(5)
    started = time.monotonic()
    asyncio.run(manager.stop())
    assert time.monotonic() - started < 2
    assert closed.is_set()
    assert jobs.load(job['job_id'])['state'] == 'queued'
    def resumed(_):
        jobs.checkpoint('first', lambda: calls.append('duplicate'))
        return {'done': True}, 'completed'
    next_manager = jobs.Manager(resumed)
    asyncio.run(next_manager.start())
    wait(lambda: jobs.load(job['job_id'])['state'] == 'completed')
    asyncio.run(next_manager.stop())
    assert calls == ['first']


def test_cancel_waiting_resource_does_not_publish():
    job = jobs.submit('test', {})
    gate = threading.Lock()
    gate.acquire()
    entered = threading.Event()
    def runner(_):
        entered.set()
        with transport.work_slot(gate):
            persistence.save_session('late', {'tops': ['wrong']})
        return {}, 'completed'
    manager = jobs.Manager(runner)
    asyncio.run(manager.start())
    assert entered.wait(5)
    jobs.cancel(job['job_id'])
    asyncio.run(manager.stop())
    gate.release()
    assert jobs.load(job['job_id'])['state'] == 'cancelled'
    assert persistence.load_session('late') is None


def test_bounded_retries_and_changed_configuration(monkeypatch):
    job = jobs.submit('test', {})
    def fail(_):
        raise ConnectionError('private provider error')
    manager = jobs.Manager(fail)
    for _ in range(manager.max_attempts):
        with persistence.connect() as db:
            db.execute('UPDATE durable_jobs SET available_at=0')
        manager.execute(jobs.claim(manager.owner, 60))
    final = jobs.load(job['job_id'])
    assert final['state'] == 'failed'
    assert final['error'] == 'ConnectionError'
    assert jobs.claim('another', 60) is None
    new = jobs.submit('test', {})
    monkeypatch.setenv('LLM_MODEL', 'other-model')
    manager.execute(jobs.claim(manager.owner, 60))
    assert jobs.load(new['job_id'])['state'] == 'review_required'


def test_atomic_publication_and_revision_check():
    source = persistence.save_session('session', {'tops': ['Original']})
    job = jobs.submit('test', {'session_revision': source['revision']})
    with claimed(job):
        manual = persistence.save_session('session', {'tops': ['Manual']}, expected_revision=source['revision'])
        with jobs.publication('published'):
            with pytest.raises(persistence.SessionConflictError):
                persistence.save_session('session', {'tops': ['Automatic']}, expected_revision=source['revision'])
        assert not jobs.published('published')
        with jobs.publication('published'):
            persistence.save_session('session', manual, expected_revision=manual['revision'])
        assert jobs.published('published')
        jobs.cancel(job['job_id'])
        with pytest.raises(transport.LLMCancelledError):
            persistence.save_session('session', {'tops': ['Late']})
    assert persistence.load_session('session')['tops'] == ['Manual']


def test_pdf_start_response_short_and_document_survives_restart_and_cancel(monkeypatch):
    started = threading.Event()
    def extract(*args, **kwargs):
        started.set()
        while True:
            jobs.check()
            time.sleep(.01)
    monkeypatch.setattr(main, 'extract_agenda_data_from_pdf', extract)
    with TestClient(main.app) as client:
        before = time.monotonic()
        response = client.post('/api/extract-tops/jobs', files={'pdf': ('a.pdf', b'%PDF-fake', 'application/pdf')})
        assert response.status_code == 202
        assert time.monotonic() - before < 1
        job_id = response.json()['job_id']
        assert started.wait(5)
        for _ in range(3):
            status = client.get('/api/model-jobs/' + job_id)
            assert status.status_code == 200 and status.json()['state'] == 'running'
            assert client.get('/health').status_code == 200
        client.post('/api/model-jobs/' + job_id + '/cancel')
    stored = jobs.load(job_id)
    assert stored['state'] == 'cancelled'
    assert Path(stored['documents'][0]['path']).exists()
    assert 'path' not in status.json()['documents'][0]


def test_retention_default_and_review_holds(tmp_path, monkeypatch):
    paths = []
    for state in ['completed', 'review_required', 'failed', 'running']:
        pdf = tmp_path / (state + '.pdf')
        pdf.write_bytes(state.encode())
        job = jobs.submit('pdf', {}, documents=[jobs.document(pdf)])
        with persistence.connect() as db:
            db.execute('UPDATE durable_jobs SET state=?,updated_at=0 WHERE job_id=?', (state, job['job_id']))
        paths.append(pdf)
    assert jobs.cleanup_documents() == []
    monkeypatch.setenv('MODEL_DOCUMENT_RETENTION_DAYS', '1')
    candidates = jobs.cleanup_documents()
    assert len(candidates) == 1
    assert all(path.exists() for path in paths)  # dry-run is default
    jobs.cleanup_documents(dry_run=False)
    assert not paths[0].exists()
    assert all(path.exists() for path in paths[1:])


def test_token_progress_outlives_old_http_timeout(monkeypatch):
    ticks = [0.0]
    real_monotonic = time.monotonic
    # A dedicated transport clock simulates 30 hours without waiting or touching model services.
    monkeypatch.setattr(transport, 'time', SimpleNamespace(monotonic=lambda: ticks[0], time=time.time))
    async def long_generate(client, config, kwargs, report):
        for _ in range(30):
            ticks[0] += 3600
            report('generating')
            await asyncio.sleep(.001)
        return 'validated'
    monkeypatch.setattr(transport, '_generate', long_generate)
    updates = []
    result = asyncio.run(transport._run(None, get_llm_config(), {}, None, updates.append))
    assert result == 'validated'
    assert ticks[0] == 108000
    assert updates[-1]['last_delta_at'] is not None
    assert real_monotonic() > 0


def test_empty_stream_events_do_not_count_as_generation(monkeypatch):
    async def empty_stream(*args):
        while True:
            yield {'choices': [{'delta': {}}]}
            await asyncio.sleep(.005)
    monkeypatch.setattr(transport, '_openai_stream', empty_stream)
    config = replace(get_llm_config(), load_seconds=.03, max_retries=0)
    updates = []
    with pytest.raises(TimeoutError):
        transport._complete(None, config, messages=[{'role': 'user', 'content': 'test'}],
                            max_tokens=10, progress_callback=updates.append)
    assert all(item['last_delta_at'] is None for item in updates)


def test_summary_crash_after_publication_does_not_regenerate_successful_top(monkeypatch):
    from summarize import SummarizationResult
    source = persistence.save_session('s', main.reconcile_session_summaries(None, {
        'session_id': 's', 'tops': ['A', 'B'], 'top_ids': ['a', 'b'], 'assignments': [0, 1],
        'transcript': [{'line_id': key, 'speaker': 'S', 'text': key, 'start': i, 'end': i+1}
                       for i, key in enumerate(['a', 'b'])],
        'summaries': {0: 'old A', 1: 'old B'}, 'speaker_names': {},
    }))
    response = asyncio.run(main.create_summary_job('s', main.SummaryJobCreateRequest(
        top_ids=['a', 'b'], revision=source['revision'])))
    job = jobs.load(response.summary_job_id)
    calls = []
    def generate(title, *args, **kwargs):
        calls.append(title)
        return SummarizationResult(summary='New ' + title, duration_seconds=.1)
    monkeypatch.setattr(main, 'summarize_segment', generate)
    report = main.update_summary_job
    def crash_after_publish(*args, **kwargs):
        if jobs.published('summary:a'):
            raise KeyboardInterrupt('simulated power loss')
        return report(*args, **kwargs)
    monkeypatch.setattr(main, 'update_summary_job', crash_after_publish)
    with claimed(job):
        with pytest.raises(KeyboardInterrupt):
            main.run_summary_job(job['job_id'])
    assert persistence.load_session('s')['summaries'][0] == 'New A'
    monkeypatch.setattr(main, 'update_summary_job', report)
    with claimed(job, 'restarted'):
        main.run_summary_job(job['job_id'])
    assert calls == ['A', 'B']
    final = persistence.load_summary_job(job['job_id'])
    assert final['status'] == 'completed'
    assert set(final['refs']['outcomes']) == {'a', 'b'}


def test_pipeline_snapshot_rejects_manual_edits_and_retains_computed_result(monkeypatch, agenda_model):
    from conftest import FakeTranscriptionResult
    from summarize import SummarizationResult
    monkeypatch.setattr(main, 'transcribe_audio', lambda *args, **kwargs: FakeTranscriptionResult(
        transcript=[{'speaker': 'S', 'text': 'TOP 1 Haushalt.', 'start': 0, 'end': 1}],
        audio_duration_seconds=1))
    target = {}
    ready = threading.Event()
    def generate(*args, **kwargs):
        assert ready.wait(5)
        source = persistence.load_session(target['session_id'])
        persistence.save_session(target['session_id'], {**source, 'tops': ['Manual edit']},
                                 expected_revision=source['revision'])
        return SummarizationResult(summary='Automatic text', duration_seconds=.1)
    monkeypatch.setattr(main, 'summarize_segment', generate)
    with TestClient(main.app) as client:
        start = client.post('/api/pipeline/start', files={'audio': ('a.mp3', b'audio', 'audio/mpeg')},
                            data={'tops': '["1 Haushalt"]', 'agenda_use_llm': 'true'}).json()
        target.update(start)
        ready.set()
        wait(lambda: jobs.load(start['pipeline_id'])['state'] in jobs.TERMINAL)
        assert jobs.load(start['pipeline_id'])['state'] == 'superseded'
    assert persistence.load_session(start['session_id'])['tops'] == ['Manual edit']
    with persistence.connect() as db:
        row = db.execute("SELECT value FROM durable_steps WHERE job_id=? AND step_key='pipeline:summaries'",
                         (start['pipeline_id'],)).fetchone()
    assert 'Automatic text' in row[0]


def test_cancellation_fences_publication_even_after_model_return(monkeypatch):
    session = persistence.save_session('s', {'tops': ['Keep']})
    job = jobs.submit('test', {})
    with claimed(job):
        result = {'tops': ['Late model answer']}
        jobs.cancel(job['job_id'])
        with pytest.raises(transport.LLMCancelledError):
            persistence.save_session('s', result, expected_revision=session['revision'])
    assert persistence.load_session('s')['tops'] == ['Keep']


def test_user_cancel_closes_active_model_stream(monkeypatch):
    job = jobs.submit('test', {})
    streaming, closed = threading.Event(), threading.Event()
    async def generate(client, config, kwargs, report):
        report('generating')
        streaming.set()
        try:
            await asyncio.sleep(10000)
        finally:
            closed.set()
    monkeypatch.setattr(transport, '_generate', generate)
    def runner(_):
        transport.complete(None, get_llm_config(), messages=[], max_tokens=10)
        return {'late': 'must not be published'}, 'completed'
    manager = jobs.Manager(runner)
    asyncio.run(manager.start())
    assert streaming.wait(5)
    jobs.cancel(job['job_id'])
    assert closed.wait(2)
    asyncio.run(manager.stop())
    assert jobs.load(job['job_id'])['state'] == 'cancelled'
    assert jobs.load(job['job_id'])['result'] is None


def test_model_loading_has_separate_allowance_from_generation_silence(monkeypatch):
    async def generate(client, config, kwargs, report):
        await asyncio.sleep(.12)  # Longer than the read/stall allowance, shorter than loading allowance.
        report('generating')
        return 'result'
    monkeypatch.setattr(transport, '_generate', generate)
    config = replace(get_llm_config(), load_seconds=.4, timeout_seconds=.03)
    assert transport._complete(None, config, messages=[], max_tokens=10) == 'result'


def test_cancel_pipeline_keeps_pdf_for_review(tmp_path, monkeypatch):
    source = tmp_path / 'source.pdf'
    source.write_bytes(b'%PDF-original')
    persistence.save_session('s', {})
    persistence.save_pipeline_job('pipeline', {
        'session_id': 's', 'status': 'pending', 'stage': 'upload',
        'result_refs': {'pdf_path': str(source)},
    })
    main.submit_legacy_job('pipeline', 'pipeline')
    main.request_pipeline_cancellation('pipeline')
    assert jobs.load('pipeline')['state'] == 'cancelled'
    assert source.read_bytes() == b'%PDF-original'


def test_malformed_pdf_model_result_is_not_complete(monkeypatch):
    from extract_tops import parse_agenda_data_response
    with pytest.raises(ValueError):
        parse_agenda_data_response('not json', fallback_text='Tagesordnung\n1. Haushalt\n2. Anfragen')


def test_shared_document_reference_prevents_cleanup(tmp_path, monkeypatch):
    monkeypatch.setenv('MODEL_DOCUMENT_RETENTION_DAYS', '1')
    pdf = tmp_path / 'shared.pdf'
    pdf.write_bytes(b'PDF')
    first = jobs.submit('pdf', {}, documents=[jobs.document(pdf)])
    jobs.submit('pdf', {}, documents=[jobs.document(pdf)])
    with persistence.connect() as db:
        db.execute("UPDATE durable_jobs SET state='completed',updated_at=0 WHERE job_id=?", (first['job_id'],))
    assert jobs.cleanup_documents(dry_run=False) == []
    assert pdf.exists()


def test_expired_lease_is_retryable_interruption_not_user_cancel():
    job = jobs.submit('test', {})
    def loses_ownership(_):
        with persistence.connect() as db:
            db.execute('UPDATE durable_jobs SET lease_until=0 WHERE job_id=?', (job['job_id'],))
        jobs.check()
        return {}, 'completed'
    manager = jobs.Manager(loses_ownership)
    manager.execute(jobs.claim(manager.owner, 60))
    result = jobs.load(job['job_id'])
    assert result['state'] == 'retry_wait'
    assert result['error'] == 'LeaseLost'


def test_pipeline_restart_reuses_transcription_agenda_and_summaries(monkeypatch):
    from conftest import FakeTranscriptionResult
    from summarize import SummarizationResult
    calls = {'transcription': 0, 'agenda': 0, 'summary': 0}
    interrupted = [False]
    save = main.save_pipeline_session
    def transcribe(*args, **kwargs):
        calls['transcription'] += 1
        return FakeTranscriptionResult(
            transcript=[{'speaker': 'S', 'text': 'TOP 1 Haushalt.', 'start': 0, 'end': 1}],
            audio_duration_seconds=1)
    def agenda(*args, **kwargs):
        calls['agenda'] += 1
        return ['1 Haushalt'], [0], {'strategy': 'test', 'segments': [], 'uncertain_count': 0}, {}
    def summary(*args, **kwargs):
        calls['summary'] += 1
        return SummarizationResult(summary='Fertiges Ergebnis', duration_seconds=.1)
    def interrupted_publication(*args, **kwargs):
        if kwargs.get('summaries') is not None and not interrupted[0]:
            interrupted[0] = True
            raise jobs.WorkerStopped('simulated shutdown after validated results')
        return save(*args, **kwargs)
    monkeypatch.setattr(main, 'transcribe_audio', transcribe)
    monkeypatch.setattr(main, 'detect_pipeline_agenda', agenda)
    monkeypatch.setattr(main, 'summarize_segment', summary)
    monkeypatch.setattr(main, 'save_pipeline_session', interrupted_publication)
    with TestClient(main.app) as client:
        started = client.post('/api/pipeline/start', files={'audio': ('a.mp3', b'audio', 'audio/mpeg')}).json()
        job_id = started['pipeline_id']
        wait(lambda: interrupted[0] and jobs.load(job_id)['state'] == 'queued')
    with TestClient(main.app):
        wait(lambda: jobs.load(job_id)['state'] in {'completed', 'review_required'})
    assert calls == {'transcription': 1, 'agenda': 1, 'summary': 1}
    assert persistence.load_session(started['session_id'])['summaries'] == {0: 'Fertiges Ergebnis'}


def test_agenda_resume_keeps_split_source_ids_and_successful_model_steps(agenda_model):
    text = ('Dies ist ein ausführlicher Beitrag zur laufenden Beratung mit weiteren Einzelheiten. '
            'Danach folgt eine ausführliche Rückfrage zur Planung und Finanzierung des Projektes. '
            'Wir nehmen anschließend die ursprüngliche Beratung wieder auf.')
    request = main.AgendaDetectionRequest(tops=['Haushalt'], use_llm=True, transcript=[
        main.TranscriptLine(line_id='original', speaker='M', text=text, start=10, end=40)])
    job = jobs.submit('agenda', {'request': request.model_dump()})
    agenda_model.overrides['independent:detail'] = jobs.WorkerStopped()
    with claimed(job):
        with pytest.raises(jobs.WorkerStopped):
            main.calculate_agenda(request)
    original_ids = [r['line_id'] for b, _ in agenda_model.calls if b['phase'] == 'primary:detail' for r in b['target_lines']]
    assert len(original_ids) == 3 and original_ids[0] == 'original'
    agenda_model.calls.clear()
    agenda_model.overrides.clear()
    with claimed(job):
        result = main.calculate_agenda(request)
    assert result.llm.review_complete
    assert [line.line_id for line in result.transcript] == original_ids
    assert [b['phase'] for b, _ in agenda_model.calls] == ['independent:detail']
    assert result.transcript[0].start == 10 and result.transcript[-1].end == 40


def test_agenda_phase_and_coverage_survive_transport_progress():
    job = jobs.submit('agenda', {})
    with claimed(job):
        jobs.progress({'phase': 'independent:detail', 'agenda_phase': 'independent:detail',
                       'processed_lines': 12, 'total_lines': 24})
        jobs.progress({'phase': 'generating', 'elapsed_seconds': 10})
    progress = jobs.load(job['job_id'])['progress']
    assert progress['agenda_phase'] == 'independent:detail'
    assert progress['processed_lines'] == 12 and progress['total_lines'] == 24
