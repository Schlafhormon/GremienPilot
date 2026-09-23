"""Technical/transport tests, explicitly NOT a real-model quality evaluation."""
import json
import threading
from pathlib import Path

import pytest
from pydantic import ValidationError

import durable_jobs as jobs
import extract_tops as pdf
from llm_transport import LLMCancelledError
from pdf_fixtures import agenda, item, audit, pdf_bytes


def test_schema_preserves_original_numbers_hierarchy_short_titles_and_sections():
    values = [item('h', title='Öffentlicher Teil', kind='heading'),
              item('a', number='01', title='Rat', section='public', parent_id='h'),
              item('b', number='01.2', title='Bau', section='public', parent_id='a'),
              item('c', number='01', title='Personal', section='nonpublic'),
              item('d', title='Ja')]
    result = pdf.parse_agenda_data_response(json.dumps(agenda(items=values)), '1 Längere\n2 Heuristik\n3 Liste')
    assert result.tops == ['[Öffentlich] 01 Rat', '[Öffentlich] 01.2 Bau', '[Nichtöffentlich] 01 Personal', 'Ja']
    assert result.items[2]['parent_id'] == 'a'
    assert not result.processing_complete  # text-only parsing cannot certify PDF pages


@pytest.mark.parametrize('response', ['', '1 Rat', '{"tops": []}', '{"tops":["1 Rat"]}'])
def test_no_legacy_heuristic_fallback(response):
    with pytest.raises((ValidationError, pdf.ExtractionError)):
        pdf.parse_agenda_data_response(response, 'Einladung Rat am 01.02.2026\n1 Rat\n2 Bau')


@pytest.mark.parametrize('change', ['duplicate', 'cycle', 'source', 'page', 'number', 'metadata', 'empty'])
def test_technical_rejections(change):
    data = agenda()
    if change == 'duplicate': data['items'] *= 2
    if change == 'cycle': data['items'][0]['parent_id'] = 'p1-a'
    if change == 'source': data['items'][0]['sources'][0]['page'] = 2
    if change == 'page': data['pages'] = []
    if change == 'number': data['items'][0]['number'] = 1
    if change == 'metadata': data['metadata']['date'] = '2026-09-22'
    if change == 'empty': data['items'] = []
    with pytest.raises((ValidationError, pdf.ExtractionError)):
        pdf._validate(data, [1])


@pytest.mark.parametrize('kinds', [('digital',), ('scan',), ('digital', 'scan', 'digital')])
def test_real_pdf_rendering_all_pages_and_independent_audit(tmp_path, monkeypatch, fake_openai_module, kinds):
    path = tmp_path / 'synthetic.pdf'
    path.write_bytes(pdf_bytes(kinds))
    monkeypatch.setenv('LLM_IMAGE_TOKENS', '1024')
    monkeypatch.setenv('LLM_CONTEXT_TOKENS', '65536')
    per_page = [agenda([p], [item(f'p{p}-a', p)]) for p in range(1, len(kinds) + 1)]
    merged = agenda(range(1, len(kinds) + 1), [i for page in per_page for i in page['items']])
    fake_openai_module.responses = [json.dumps(v) for v in per_page + [merged] + [audit(p) for p in merged['pages']] + [audit(0)]]
    result = pdf.extract_agenda_data_from_pdf(str(path))
    assert result.processing_complete and not result.review_required
    assert len(result.pages) == len(kinds)
    assert all(p['width'] >= 1000 and p['status'] == 'verified' for p in result.pages)
    for p, kind in zip(result.pages, kinds):
        assert (p['text_characters'] == 0) == (kind == 'scan')
    calls = [c for instance in fake_openai_module.instances for c in instance.calls]
    visual = [c for c in calls if isinstance(c['messages'][1]['content'], list) and any(p['type'] == 'image_url' for p in c['messages'][1]['content'])]
    assert len(visual) == len(kinds) * 2 + 1
    assert all(c['response_format']['type'] == 'json_schema' for c in calls)
    assert all(len(c['messages']) == 2 for c in calls)  # independent audit contexts


def test_broken_text_layer_never_overrules_visual_evidence(tmp_path, monkeypatch, fake_openai_module):
    import pdfplumber.page
    path = tmp_path / 'synthetic.pdf'; path.write_bytes(pdf_bytes())
    monkeypatch.setattr(pdfplumber.page.Page, 'extract_text', lambda *_: 'f�r\nWRONG DATE 1900-01-01')
    monkeypatch.setenv('LLM_IMAGE_TOKENS', '1024')
    fake_openai_module.responses = [json.dumps(v) for v in [agenda(), agenda(), audit(), audit(0)]]
    result = pdf.extract_agenda_data_from_pdf(str(path))
    assert result.tops == ['Haushalt']
    request = fake_openai_module.instances[0].calls[0]
    assert any('f�r' in part.get('text', '') for part in request['messages'][1]['content'])


def test_missing_and_unreadable_pdf_fail_without_model(tmp_path, fake_openai_module):
    with pytest.raises(pdf.ExtractionError, match='fehlt'):
        pdf.extract_agenda_data_from_pdf(str(tmp_path / 'absent.pdf'))
    path = tmp_path / 'broken.pdf'; path.write_bytes(b'not a PDF')
    with pytest.raises(Exception): pdf.extract_agenda_data_from_pdf(str(path))
    assert not fake_openai_module.instances


def test_repair_and_page_break_reconciliation(tmp_path, monkeypatch):
    path = tmp_path / 'synthetic.pdf'; path.write_bytes(pdf_bytes(('digital', 'scan')))
    first = agenda([1], [item('a', title='Plan')])
    second = agenda([2], [item('fragment', 2, title='Fortsetzung'), item('b', 2, title='Rat', parent_id='fragment')])
    merged = agenda([1, 2], [item('a', title='Plan Fortsetzung'), item('b', 2, title='Rat', parent_id='a')])
    corrected = json.loads(json.dumps(merged)); corrected['items'][0]['sources'].append({'page': 2, 'quote': 'Fortsetzung'})
    issue = dict(kind='continuation', item_ids=['a'], metadata_fields=[], description='Fortsetzungsquelle prüfen?',
                 pages=[1, 2], evidence=[{'page': 2, 'quote': 'Fortsetzung'}])
    patch = dict(upsert=[dict(item=corrected['items'][0], after_id=None)], delete_ids=[], metadata=[])
    calls = []
    responses = iter([first, second, merged, audit(1), audit(2), audit(0, [issue]), patch, audit(1), audit(2), audit(0)])
    def request(config, prompt, content, schema):
        calls.append((prompt, content))
        return json.dumps(next(responses))
    monkeypatch.setattr(pdf, '_request', request)
    result = pdf.extract_agenda_data_from_pdf(str(path))
    assert result.processing_complete
    assert len(result.audits) == 6
    assert len([p for p in calls[6][1] if p['type'] == 'image_url']) == 2
    assert result.items[0]['sources'][-1]['page'] == 2
    assert result.items[1]['parent_id'] == result.items[0]['id']


def test_malformed_and_empty_model_outputs_are_repaired(tmp_path, monkeypatch):
    path = tmp_path / 'synthetic.pdf'; path.write_bytes(pdf_bytes())
    responses = iter(['not JSON', json.dumps(agenda()), json.dumps(agenda(items=[])), json.dumps(agenda()), json.dumps(audit()), json.dumps(audit(0))])
    monkeypatch.setattr(pdf, '_request', lambda *args: next(responses))
    assert pdf.extract_agenda_data_from_pdf(str(path)).processing_complete


def test_unresolved_audit_cannot_publish(tmp_path, monkeypatch):
    path = tmp_path / 'synthetic.pdf'; path.write_bytes(pdf_bytes())
    monkeypatch.setenv('PDF_REVIEW_ROUNDS', '1')
    issue = dict(kind='unclear', item_ids=[], metadata_fields=[], description='Welche Zeile ist lesbar?',
                 pages=[1], evidence=[{'page': 1, 'quote': None}])
    responses = iter([agenda(), agenda(), audit(1, [issue]), audit(0)])
    monkeypatch.setattr(pdf, '_request', lambda *args: json.dumps(next(responses)))
    result = pdf.extract_agenda_data_from_pdf(str(path))
    assert not result.processing_complete and result.review_required
    assert result.review_questions == [issue]
    assert result.items and result.stop_reason == 'review_budget'


def test_resume_uses_completed_page_and_request_checkpoints(tmp_path, monkeypatch):
    path = tmp_path / 'synthetic.pdf'; path.write_bytes(pdf_bytes(('digital', 'scan')))
    job = jobs.submit('pdf', {}, documents=[jobs.document(path)])
    claimed = jobs.claim('worker', 60)
    ctx = jobs.Runtime(claimed, 'worker', threading.Event())
    token = jobs.CURRENT.set(ctx)
    calls = []
    def interrupted(*args):
        calls.append(args)
        if len(calls) == 2: raise LLMCancelledError()
        return json.dumps(agenda())
    monkeypatch.setattr(pdf, '_request', interrupted)
    try:
        with pytest.raises(LLMCancelledError): pdf.extract_agenda_data_from_pdf(str(path))
        assert len(calls) == 2
        merged = agenda([1, 2], [item('p1-a'), item('p2-a', 2)])
        responses = iter([agenda([2], [item('p2-a', 2)]), merged, audit(1), audit(2), audit(0)])
        monkeypatch.setattr(pdf, '_request', lambda *args: json.dumps(next(responses)))
        result = pdf.extract_agenda_data_from_pdf(str(path))
        monkeypatch.setattr(pdf, '_request', lambda *args: pytest.fail('Completed calls repeated'))
        repeated = pdf.extract_agenda_data_from_pdf(str(path))
        assert repeated.to_dict() == result.to_dict()
        assert result.document['job_id'] == job['job_id']
    finally:
        jobs.CURRENT.reset(token)


def test_page_limit_is_failure_not_truncation(tmp_path, monkeypatch):
    path = tmp_path / 'synthetic.pdf'; path.write_bytes(pdf_bytes(('digital', 'scan')))
    monkeypatch.setenv('PDF_MAX_PAGES', '1')
    with pytest.raises(pdf.ExtractionError, match='keine Seiten ausgelassen'):
        pdf.extract_agenda_data_from_pdf(str(path))


def test_text_extraction_error_is_recorded_but_image_still_processed(tmp_path, monkeypatch):
    import pdfplumber.page
    path = tmp_path / 'synthetic.pdf'; path.write_bytes(pdf_bytes())
    def broken(*args): raise RuntimeError('synthetic text failure')
    monkeypatch.setattr(pdfplumber.page.Page, 'extract_text', broken)
    answers = iter([agenda(), agenda(), audit(), audit(0)])
    monkeypatch.setattr(pdf, '_request', lambda *args: json.dumps(next(answers)))
    result = pdf.extract_agenda_data_from_pdf(str(path))
    assert result.processing_complete
    assert result.pages[0]['text_error'] == 'RuntimeError'


def test_metadata_is_model_sourced_not_guessed_from_letter_date():
    data = agenda()
    data['metadata']['date'] = '2026-10-01'
    data['metadata']['time'] = '18:00 Uhr'
    data['metadata_sources']['date'] = [{'page': 1, 'quote': 'Sitzung am 01.10.2026'}]
    data['metadata_sources']['time'] = [{'page': 1, 'quote': '18:00 Uhr'}]
    result = pdf.parse_agenda_data_response(json.dumps(data), 'Briefdatum 22.09.2026\nHauptausschuss')
    assert result.metadata.date == '2026-10-01'
    assert result.metadata.time == '18:00 Uhr'
    assert result.metadata.committee == ''
