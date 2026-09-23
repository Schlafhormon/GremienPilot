"""Synthetic source-contract regressions; no private documents or model services."""
import copy
import json

import pytest
import extract_tops as pdf
from pdf_fixtures import agenda, item, audit, pdf_bytes


def finding(kind='omission', ids=None):
    return dict(kind=kind, item_ids=ids or [], metadata_fields=[], pages=[2],
                evidence=[dict(page=2, quote='02.1 Bau')], description='Fehlt Bau auf Seite 2?')


def run(tmp_path, monkeypatch, answers):
    path = tmp_path / 'three-pages.pdf'
    path.write_bytes(pdf_bytes(('digital',) * 3))
    calls = []
    answers = iter(answers)
    def request(config, prompt, content, schema):
        calls.append((prompt, content, schema))
        return json.dumps(next(answers))
    monkeypatch.setattr(pdf, '_request', request)
    return pdf.extract_agenda_data_from_pdf(path), calls


def sources():
    inventories = [agenda([p], [item(f'p{p}', p)]) for p in [1, 2, 3]]
    return inventories, agenda([1, 2, 3], [i for a in inventories for i in a['items']])


def test_three_page_audits_never_receive_following_page_entries(tmp_path, monkeypatch):
    inventories, candidate = sources()
    result, calls = run(tmp_path, monkeypatch, inventories + [candidate, audit(1), audit(2), audit(3), audit(0)])
    assert result.processing_complete
    for number, (_, content, _) in enumerate(calls[4:7], 1):
        projection = json.loads(content[0]['text'].split('\n', 1)[1])
        assert [i['id'] for i in projection['items']] == [f'p{number}']
        assert all(s['page'] == number for i in projection['items'] for s in i['sources'])
        assert sum(c['type'] == 'image_url' for c in content) == 1
    assert sum(c['type'] == 'image_url' for c in calls[7][1]) == 3


def test_continuation_projection_contains_only_local_fragments():
    data = agenda([1, 2], [item('parent'), item('child', 2, parent_id='parent')])
    data['items'][0]['sources'] += [dict(page=2, quote='Fortsetzung')]
    first = pdf.page_projection(data, 1)
    assert first['items'] == [dict(id='parent', sources=[dict(page=1, quote=None)], scope='local_fragment')]
    second = pdf.page_projection(data, 2)
    assert second['items'][0]['sources'] == [dict(page=2, quote='Fortsetzung')]
    assert 'parent_id' not in second['items'][1]
    ids, pages, _, context_ids = pdf.repair_scope(data, [finding('parent', ['child'])])
    assert ids == {'child'} and context_ids == {'child', 'parent'} and pages == {1, 2}


def test_missing_top_on_page_two_is_added_without_rewriting_other_entries(tmp_path, monkeypatch):
    inventories, candidate = sources()
    issue = finding()
    patch = dict(upsert=[dict(item=item('missing', 2, title='Bau'), after_id='p2')], delete_ids=[], metadata=[])
    result, calls = run(tmp_path, monkeypatch, inventories + [candidate,
        audit(1), audit(2, [issue]), audit(3), audit(0), patch,
        audit(1), audit(2), audit(3), audit(0)])
    assert result.processing_complete
    assert [i['title'] for i in result.items] == ['Haushalt', 'Haushalt', 'Bau', 'Haushalt']
    repair = next(content for _, content, schema in calls if schema is pdf.Patch)
    assert sum(c['type'] == 'image_url' for c in repair) == 1


@pytest.mark.parametrize('change_candidate', [False, True])
def test_identical_candidate_or_findings_end_with_retained_review(tmp_path, monkeypatch, change_candidate):
    inventories, candidate = sources()
    issue = finding('unsupported', ['p2'])
    replacement = copy.deepcopy(candidate['items'][1])
    if change_candidate:
        replacement['title'] = 'Geänderter Kandidat'
    patch = dict(upsert=[dict(item=replacement, after_id=None)], delete_ids=[], metadata=[])
    answers = inventories + [candidate, audit(1), audit(2, [issue]), audit(3), audit(0), patch]
    if change_candidate:
        answers += [audit(1), audit(2, [issue]), audit(3), audit(0)]
    monkeypatch.setenv('PDF_REVIEW_ROUNDS', '4')
    result, calls = run(tmp_path, monkeypatch, answers)
    assert result.review_required and not result.processing_complete
    assert result.stop_reason == ('repeated_findings' if change_candidate else 'unchanged_candidate')
    assert result.review_questions[0]['item_ids'] == [result.items[1]['id']]
    assert sum(schema is pdf.Patch for _, _, schema in calls) == 1


def test_repair_cannot_change_unrelated_items_or_hide_wrong_sources():
    _, candidate = sources()
    patch = dict(upsert=[dict(item=item('p3', 3), after_id=None)], delete_ids=[], metadata=[])
    with pytest.raises(pdf.ExtractionError, match='unbeanstandete'):
        pdf.apply_patch(candidate, patch, [finding('unsupported', ['p2'])])
    patch['upsert'] = [dict(item=item('p2', 3), after_id=None)]
    with pytest.raises(pdf.ExtractionError, match='ungesehene'):
        pdf.apply_patch(candidate, patch, [finding('unsupported', ['p2'])])


def test_page_findings_cannot_accuse_an_entry_on_another_page():
    _, candidate = sources()
    value = audit(1, [finding('unsupported', ['p2'])])
    with pytest.raises(pdf.ExtractionError, match='unbekannte'):
        pdf.validate_audit(value, pdf.page_projection(candidate, 1), [1], 1)
