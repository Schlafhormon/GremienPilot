"""Session-bound recalculation uses retained sources and leaves edits untouched."""
import json

from fastapi.testclient import TestClient

import durable_jobs as jobs
import main
import persistence


def finish(job, result, state='completed'):
    with persistence.connect() as db:
        db.execute('UPDATE durable_jobs SET state=?,result=? WHERE job_id=?',
                   (state, json.dumps(result), job['job_id']))


def session_with_pdf(tmp_path):
    path = tmp_path / 'invitation.pdf'
    path.write_bytes(b'%PDF-original')
    doc = jobs.document(path)
    source = jobs.submit('pdf', {'path': str(path), 'processing_mode': 'fast'}, documents=[doc])
    result = {'tops': ['Haushalt'], 'metadata': {}, 'processing_complete': True,
              'processing_mode': 'fast', 'review_status': 'skipped', 'review_required': True,
              'document': {'job_id': source['job_id'], 'sha256': doc['sha256']}}
    finish(source, result, 'review_required')
    state = {'tops': ['Manuell bearbeitet'], 'top_ids': ['top-a'], 'processing_mode': 'fast',
             'transcript': [{'line_id': 'line-a', 'speaker': 'S1', 'text': 'Haushalt', 'start': 0, 'end': 1}],
             'assignments': [0], 'pdf_source_job_id': source['job_id']}
    persistence.save_session('session-1', state)
    return path, source, result, state


def test_restart_pdf_preserves_edits_and_restores_status_until_explicit_acceptance(tmp_path):
    path, source, old_result, state = session_with_pdf(tmp_path)
    client = TestClient(main.app)
    response = client.post('/api/sessions/session-1/pdf-jobs', json={})
    assert response.status_code == 202, response.text
    job = jobs.load(response.json()['job_id'])
    assert job['job_id'] != source['job_id']
    assert job['payload']['processing_mode'] == 'fast'
    assert job['payload']['path'] == str(path)
    assert 'path' not in response.json()['documents'][0]
    assert client.post('/api/sessions/session-1/pdf-jobs', json={}).status_code == 409
    assert client.put('/api/sessions/session-1', json={**state, 'processing_mode': 'slow'}).status_code == 409
    assert client.put('/api/sessions/session-1', json={**state, 'pdf_source_job_id': job['job_id']}).status_code == 422
    latest = client.get('/api/sessions/session-1').json()
    assert latest['latest_pdf_job']['job_id'] == job['job_id']
    assert latest['has_pdf_source'] is True
    assert latest['tops'] == state['tops']
    assert latest['pdf_extraction'] == old_result
    result = {**old_result, 'tops': ['Neue Tagesordnung'],
              'document': {**old_result['document'], 'job_id': job['job_id']}}
    finish(job, result, 'review_required')
    assert client.get('/api/sessions/session-1').json()['pdf_extraction'] == old_result
    accepted = {**state, 'tops': result['tops'], 'pdf_source_job_id': job['job_id']}
    saved = client.put('/api/sessions/session-1', json=accepted)
    assert saved.status_code == 200, saved.text
    assert saved.json()['pdf_extraction'] == result
    del accepted['pdf_source_job_id']
    assert client.put('/api/sessions/session-1', json=accepted).json()['pdf_source_job_id'] == job['job_id']


def test_missing_or_changed_pdf_is_reported_without_starting_work(tmp_path):
    path, _, _, _ = session_with_pdf(tmp_path)
    client = TestClient(main.app)
    path.write_bytes(b'changed')
    assert client.post('/api/sessions/session-1/pdf-jobs', json={}).status_code == 409
    path.unlink()
    assert client.post('/api/sessions/session-1/pdf-jobs', json={}).status_code == 410
    assert client.post('/api/sessions/missing/pdf-jobs', json={}).status_code == 404
    assert jobs.latest_for_session('session-1', 'pdf') is None


def test_agenda_job_belongs_to_session_and_exposes_frozen_source_for_reload(tmp_path):
    _, _, _, state = session_with_pdf(tmp_path)
    client = TestClient(main.app)
    request = {key: state[key] for key in ('tops', 'top_ids', 'transcript', 'processing_mode')}
    response = client.post('/api/agenda-detection/jobs', json={**request, 'session_id': 'session-1', 'fresh': True})
    assert response.status_code == 202, response.text
    job = jobs.load(response.json()['job_id'])
    assert job['payload']['request']['cache_namespace']
    assert 'source' not in response.json()
    assert client.post('/api/sessions/session-1/pdf-jobs', json={}).status_code == 409
    assert persistence.load_session('session-1')['assignments'] == [0]
    finish(job, {'tops': state['tops']})
    latest = client.get('/api/sessions/session-1').json()['latest_agenda_job']
    assert latest['source'] == request
    assert latest['job_id'] == job['job_id']
    assert client.post('/api/agenda-detection/jobs', json={**request, 'session_id': 'missing'}).status_code == 404
