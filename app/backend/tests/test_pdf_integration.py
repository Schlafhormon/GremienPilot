"""PDF input/source API invariants using isolated storage and synthetic sources."""
from pathlib import Path
import json

from fastapi.testclient import TestClient
import pytest

import main
import durable_jobs as jobs
from extract_tops import ExtractionError
from pdf_fixtures import pdf_bytes, agenda, audit


def test_missing_upload_rejected_before_audio_or_jobs_are_saved():
    with TestClient(main.app) as client:
        response = client.post('/api/pipeline/start', data={'auto_detect_tops_from_pdf': 'true'},
                               files={'audio': ('a.mp3', b'audio', 'audio/mpeg')})
        assert response.status_code == 422
        assert 'Einladung' in response.json()['detail']
        assert client.post('/api/extract-tops/jobs').status_code == 422
        assert not list(main.UPLOAD_DIR.glob('*'))


@pytest.mark.parametrize('missing', [True, False])
def test_legacy_pipeline_pdf_failure_never_calls_transcript_detector(tmp_path, monkeypatch, missing):
    source = tmp_path / 'source.pdf'
    if not missing: source.write_bytes(b'broken PDF')
    monkeypatch.setattr(main, 'detect_agenda_from_transcript', lambda *a, **kw: pytest.fail('Transcript fallback called'))
    with pytest.raises(Exception):
        main.detect_pipeline_agenda('legacy', [], known_tops=[], pdf_path=str(source),
                                    options={'auto_detect_tops_from_pdf': True})


def test_incomplete_pdf_result_never_calls_transcript_detector(tmp_path, monkeypatch):
    from extract_tops import PdfAgendaExtractionResult
    source = tmp_path / 'source.pdf'; source.write_bytes(pdf_bytes())
    monkeypatch.setattr(main, 'extract_agenda_data_from_pdf', lambda *a, **kw: PdfAgendaExtractionResult())
    monkeypatch.setattr(main, 'detect_agenda_from_transcript', lambda *a, **kw: pytest.fail('Transcript fallback called'))
    tops, assignments, info, _ = main.detect_pipeline_agenda('legacy', [], known_tops=[],
        pdf_path=str(source), options={'auto_detect_tops_from_pdf': True})
    assert tops == assignments == []
    assert info['pdf_incomplete'] and not info['llm']['processing_complete']
    assert info['pdf_extraction'] is not None and info['warnings']


def test_retained_sources_are_hash_bound_and_accessible_after_completion(tmp_path, monkeypatch, fake_openai_module):
    monkeypatch.setenv('LLM_IMAGE_TOKENS', '1024')
    fake_openai_module.responses = [json.dumps(v) for v in [agenda(), agenda(), audit(), audit(0)]]
    source = pdf_bytes()
    with TestClient(main.app) as client:
        response = client.post('/api/extract-tops', files={'pdf': ('source.pdf', source, 'application/pdf')})
        assert response.status_code == 200, response.text
        result = response.json()
        url = result['document']['url']
        assert client.get(url).content == source
        assert 'path' not in result['document']
        job = jobs.load(result['document']['job_id'])
        path = Path(job['documents'][0]['path'])
        path.write_bytes(b'changed')
        assert client.get(url).status_code == 409
        path.unlink()
        assert client.get(url).status_code == 410
        assert client.get(url.replace(result['document']['sha256'], 'unknown')).status_code == 404


def test_failed_pdf_job_retains_attempts_and_reports_safe_reason(monkeypatch):
    import extract_tops
    from test_main import wait_until
    monkeypatch.setenv('PDF_MODEL_ATTEMPTS', '2')
    monkeypatch.setattr(extract_tops, '_request', lambda *args: 'not JSON')
    with TestClient(main.app) as client:
        response = client.post('/api/extract-tops/jobs', files={'pdf': ('synthetic.pdf', pdf_bytes(), 'application/pdf')})
        job_id = response.json()['job_id']
        assert wait_until(lambda: jobs.load(job_id)['state'] == 'failed')
        job = jobs.load(job_id)
        assert 'Reparaturversuchen' in job['error']
        assert Path(job['documents'][0]['path']).exists()
        import persistence
        with persistence.connect() as db:
            steps = db.execute('SELECT step_key FROM durable_steps WHERE job_id=?', (job_id,)).fetchall()
        assert any('manifest' in step[0] for step in steps)
        assert len([step for step in steps if ':attempt:' in step[0]]) == 2


def test_attached_broken_pdf_is_not_ignored_when_auto_flag_is_false(tmp_path, monkeypatch):
    source = tmp_path / 'broken.pdf'; source.write_bytes(b'broken')
    monkeypatch.setattr(main, 'detect_agenda_from_transcript', lambda *a, **kw: pytest.fail('Transcript fallback called'))
    with pytest.raises(Exception):
        main.detect_pipeline_agenda('legacy', [], known_tops=[], pdf_path=str(source),
                                   options={'auto_detect_tops_from_pdf': False})


def test_legacy_completed_result_cannot_claim_new_visual_verification():
    import persistence
    with TestClient(main.app) as client:
        old = jobs.submit('pdf', {})
        with persistence.connect() as db:
            db.execute("UPDATE durable_jobs SET state='completed', result=? WHERE job_id=?", (
                json.dumps({'tops': ['Rat'], 'metadata': {}, 'processing_complete': True, 'review_required': False}), old['job_id']))
        response = client.post('/api/pipeline/start', files={'audio': ('a.mp3', b'audio', 'audio/mpeg')},
                               data={'pdf_source_job_id': old['job_id']})
        assert response.status_code == 422
        assert 'visuelle Quellenprüfung' in response.json()['detail']


def test_failed_pdf_draft_and_questions_are_visible_in_restored_session(tmp_path):
    import persistence
    from extract_tops import PdfAgendaExtractionResult
    draft = PdfAgendaExtractionResult(review_questions=[dict(kind='unclear', item_ids=[], pages=[1], description='Bitte Quelle prüfen')]).to_dict()
    persistence.save_session('draft-session', {})
    persistence.save_pipeline_job('draft-pipeline', dict(session_id='draft-session', status='failed', stage='agenda_detect',
        result_refs={'pdf_extraction': draft, 'processing_complete': False}))
    response = main.build_session_response(persistence.load_session('draft-session'))
    assert response.pdf_extraction['review_questions'][0]['pages'] == [1]
    import asyncio
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        asyncio.run(main.export_protocol_endpoint(main.ProtocolExportRequest(session_id='draft-session', tops=['Draft'])))
    assert exc.value.status_code == 409
