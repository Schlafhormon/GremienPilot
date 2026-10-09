"""Gemma text/review lifecycle with temporary storage and simulated model calls."""
import asyncio
from copy import deepcopy
from io import BytesIO
import json

from docx import Document
from fastapi.testclient import TestClient
import pdfplumber
import pytest

import durable_jobs as durable
import gemma_summary as gemma
import main
import persistence
import summarize
from export_protocol import ProtocolMetadata, build_protocol_document, render_protocol
from protocol_layout import render_text
from test_gemma_summary import DRAFT, ORIGINAL, FINDING, generate, setup_model


def review_lines():
    return [dict(line_id=f'line-{i}', speaker=line.split(': ', 1)[0], text=line.split(': ', 1)[1], start=i, end=i+1)
            for i, line in enumerate(ORIGINAL)]


def review_data(result):
    review = summarize.build_summary_review(structured=result.structured, summary=result.summary, lines=review_lines())
    return dict(structured=result.structured.to_dict(), source_links=[s.to_dict() for s in review.source_links],
                review_warnings=[w.to_dict() for w in review.warnings], llm_usage=result.llm_usage)


def session_state(result):
    return dict(session_id='gemma-session', tops=['Kosten'], top_ids=['costs'], transcript=review_lines(),
        assignments=[0, 0], summaries={0: result.summary}, summary_reviews={0: review_data(result)},
        processing_mode=result.llm_usage['processing_mode'], speaker_names={}, current_step=3)


@pytest.mark.parametrize('mode', ['fast', 'slow'])
@pytest.mark.parametrize('discard_metadata', [True, False])
@pytest.mark.parametrize('style', ['gemma4-lora', 'gemma4-custom'])
def test_save_reopen_edit_keeps_prose_and_all_review_findings(monkeypatch, style, mode, discard_metadata):
    setup_model(monkeypatch, mode, style=style)
    result = generate(mode)
    first = persistence.save_session('gemma-session', main.reconcile_session_summaries(None, session_state(result)))
    reopened = persistence.load_session('gemma-session')
    assert reopened['summaries'][0] == DRAFT
    assert reopened['summary_reviews'][0]['structured']['protocol_text'] == DRAFT
    assert reopened['summary_reviews'][0]['structured']['review_questions'] == result.structured.review_questions
    unchanged = main.reconcile_session_summaries(reopened, deepcopy(reopened))
    assert unchanged['summary_reviews'] == first['summary_reviews']
    edited = deepcopy(reopened)
    edited['summaries'][0] = DRAFT + '\nManuelle Ergänzung.'
    # Also simulate the frontend's removal of its old result metadata.
    edited['summary_reviews'] = {} if discard_metadata else {0: {}}
    saved = persistence.save_session('gemma-session', main.reconcile_session_summaries(reopened, edited))
    proof = saved['summary_reviews'][0]['structured']
    assert saved['summaries'][0] == edited['summaries'][0] == proof['protocol_text']
    assert proof['review_questions'] == result.structured.review_questions
    assert proof['verification']['review_complete'] is False
    assert proof['verification']['review_status'] == 'stale'
    assert saved['summary_reviews'][0]['source_links'] == []
    assert any(w['kind'] == 'verification_required' for w in saved['summary_reviews'][0]['review_warnings'])


@pytest.mark.parametrize('mode', ['fast', 'slow'])
@pytest.mark.parametrize('export_format', ['txt', 'docx', 'pdf'])
@pytest.mark.parametrize('style', ['gemma4-lora', 'gemma4-custom'])
def test_export_keeps_headings_paragraphs_and_separate_hints(monkeypatch, style, mode, export_format):
    setup_model(monkeypatch, mode, style=style)
    result = generate(mode)
    document = build_protocol_document(metadata=ProtocolMetadata(title='Sitzung'), tops=['Kosten'],
        summaries={0: DRAFT}, summary_reviews={0: review_data(result)})
    assert document.tops[0].protocol_text == DRAFT
    assert document.tops[0].discussion == document.tops[0].decisions == []
    artifact = render_protocol(document, export_format)
    if export_format == 'txt':
        text = artifact.decode('utf-8')
        assert render_text(DRAFT) in text
    elif export_format == 'docx':
        paragraphs = [p.text for p in Document(BytesIO(artifact)).paragraphs]
        text = '\n\n'.join(paragraphs)
        assert render_text(DRAFT) in text
    else:
        with pdfplumber.open(BytesIO(artifact)) as pdf:
            text = '\n'.join(page.extract_text() for page in pdf.pages)
        for line in DRAFT.splitlines():
            if line:
                assert line.replace('## ', '').replace('**', '') in text
    assert 'Zu TOP 2:' in text and 'Aus der Beratung' in text
    assert '## Zu TOP 2:' not in text and '**Aus der Beratung**' not in text
    assert document.tops[0].protocol_text == result.summary == DRAFT
    assert 'Quelle fehlt' not in text and '[UNBESTÄTIGT' not in text
    assert 'Diskussion:' not in text and 'Keine Angabe.' not in text
    if mode == 'slow':
        assert 'Prüfhinweise' in text and FINDING['question'] in text
        assert '9 Euro … 5 Euro' in text
        assert 'keine Garantie für Fehlerfreiheit' in text
    else:
        assert 'Ungeprüfter Entwurf' in text
        assert 'Inhaltsprüfungen abgeschlossen' not in text


def test_changed_sources_invalidate_review_but_keep_findings(monkeypatch):
    setup_model(monkeypatch)
    result = generate()
    first = main.reconcile_session_summaries(None, session_state(result))
    incoming = deepcopy(first)
    incoming['transcript'][0]['text'] = 'Die Kosten betragen 7 Euro.'
    changed = main.reconcile_session_summaries(first, incoming)
    assert changed['summaries'][0] == DRAFT
    review = changed['summary_reviews'][0]
    assert review['structured']['review_questions'] == result.structured.review_questions
    assert review['llm_usage']['review_complete'] is False
    assert review['llm_usage']['review_status'] == 'stale'


@pytest.mark.parametrize('failure', [False, True])
@pytest.mark.parametrize('style', ['gemma4-lora', 'gemma4-custom'])
def test_direct_api_preserves_draft_and_truthful_status(monkeypatch, style, failure):
    def review(body):
        if failure:
            raise TimeoutError()
        return dict(complete=True, issues=[FINDING])
    setup_model(monkeypatch, style=style, review=review)
    with TestClient(main.app) as client:
        response = client.post('/api/summarize', json={'top_title': 'Kosten', 'lines': review_lines(), 'processing_mode': 'slow'})
    assert response.status_code == 200
    body = response.json()
    assert body['summary'] == body['structured']['protocol_text'] == DRAFT
    assert body['source_links'] == []
    assert body['llm_usage']['review_complete'] is not failure
    assert body['llm_usage']['processing_complete'] is not failure
    assert any(w['kind'] == 'technical_incomplete' for w in body['review_warnings']) is failure


@pytest.mark.parametrize('style', ['gemma4-lora', 'gemma4-custom'])
def test_failed_regeneration_stores_new_draft_instead_of_losing_it(monkeypatch, style):
    setup_model(monkeypatch, style=style)
    result = generate()
    state = session_state(result)
    state['summaries'] = {0: 'Alter Entwurf'}
    session = persistence.save_session('gemma-session', main.reconcile_session_summaries(None, state))
    async def manager():
        return type('Manager', (), {'enqueue': staticmethod(async_enqueue)})()
    async def async_enqueue(_):
        pass
    monkeypatch.setattr(main, 'get_or_create_summary_job_manager', manager)
    job = asyncio.run(main.create_summary_job('gemma-session', main.SummaryJobCreateRequest(
        revision=session['revision'], top_ids=['costs'])))
    def fail(_):
        raise TimeoutError()
    setup_model(monkeypatch, style=style, review=fail)
    main.run_summary_job(job.summary_job_id)
    saved = persistence.load_session('gemma-session')
    assert saved['summaries'][0] == DRAFT
    assert saved['summary_states'][0]['status'] == 'failed'
    assert saved['summary_reviews'][0]['llm_usage']['review_complete'] is False
    assert persistence.load_summary_job(job.summary_job_id)['status'] == 'failed'


def test_partial_pipeline_draft_can_be_exported_with_incomplete_label(monkeypatch):
    def fail(_):
        raise TimeoutError()
    setup_model(monkeypatch, review=fail)
    with pytest.raises(summarize.LLMCallError) as caught:
        generate()
    state = session_state(caught.value.partial_result)
    persistence.save_session('gemma-session', main.reconcile_session_summaries(None, state))
    payload = dict(format='txt', session_id='gemma-session', metadata={'title': 'Sitzung'},
        tops=['Kosten'], summaries={0: DRAFT})
    with TestClient(main.app) as client:
        response = client.post('/api/export', json=payload)
    assert response.status_code == 200
    assert render_text(DRAFT) in response.text
    assert 'Inhaltsprüfung fehlgeschlagen oder unvollständig' in response.text
    assert 'Prüfentwurf' in response.text


def test_pipeline_progress_does_not_reuse_old_gemma_results(monkeypatch):
    calls = setup_model(monkeypatch, 'fast')
    monkeypatch.setattr(main, 'load_pipeline_job', lambda _: dict(result_refs={'summary_progress': {
        'summaries': {'0': 'Altes Ergebnis'}, 'summary_reviews': {'0': {'llm_usage': {'processing_complete': True}}}}}))
    monkeypatch.setattr(main, 'save_pipeline_state', lambda *a, **k: None)
    monkeypatch.setattr(summarize, 'get_llm_config', lambda model=None: __import__('test_gemma_summary').config('fast'))
    # main imports this resolver locally; it therefore sees the same Gemma config.
    summaries, reviews = main.summarize_pipeline_segments('pipeline', transcript=review_lines(), tops=['Kosten'],
        assignments=[0, 0], options={'processing_mode': 'fast'})
    assert summaries[0] == DRAFT
    assert len(calls) == 1
    assert reviews[0]['structured']['verification']['prompt_version'] == gemma.VERSION


def test_old_job_version_is_rejected_before_any_model_or_runner_call():
    job = durable.submit('test', {})
    with persistence.connect() as db:
        payload = json.loads(db.execute('SELECT payload FROM durable_jobs WHERE job_id=?', (job['job_id'],)).fetchone()[0])
        payload['versions']['code']['gemma_summary.py'] = 'old-annotated-workflow'
        db.execute('UPDATE durable_jobs SET payload=? WHERE job_id=?', (json.dumps(payload), job['job_id']))
    claimed = durable.claim('version-test', 60)
    def forbidden(_):
        pytest.fail('Old job must not replay downstream checkpoints')
    durable.Manager(forbidden).execute(claimed)
    saved = durable.load(job['job_id'])
    assert saved['state'] == 'failed'
    assert 'neue Verarbeitung erforderlich' in saved['error']
