"""Synthetic scorer contract, not a released/human-rated meeting corpus."""
import importlib.util
from pathlib import Path
from copy import deepcopy

import pytest

spec = importlib.util.spec_from_file_location('quality', Path(__file__).resolve().parents[3] / 'scripts/verify_llm_snapshot.py')
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)


def data():
    source = [dict(path='synthetic.txt', sha256='synthetic-test-only')]
    reference = dict(origin='human', approved=True, approved_by='simulated-reviewer-for-unit-test',
        source_files=source, agenda=[dict(id='a', number='2.1', section='public'), dict(id='b', number='2', section='nonpublic')],
        transcript=[dict(id='l0', top_ids=['a']), dict(id='l1', top_ids=['b'])],
        facts=[dict(id='f1', kind='decision', top_id='b'), dict(id='f2', kind='vote', top_id='b')])
    candidate = dict(source_files=source, agenda=[dict(id='x', number='1', section='public')],
        assignments=dict(l0=['x'], l1=['x']), claims=[dict(id='c1', kind='decision', top_id='x'),
        dict(id='c2', kind='vote', top_id='x')], review_questions=['Konkrete Frage?'], processing_complete=True)
    review = dict(origin='human', approved=True, approved_by='simulated-reviewer-for-unit-test',
        reference_sha256=quality.sha(reference), candidate_sha256=quality.sha(candidate),
        top_matches=dict(x='b'), claim_matches=dict(c1=['f1'], c2=[]),
        claim_source_validity=dict(c1=True, c2=False), remaining_review_questions=['Offene Testfrage?'])
    return reference, candidate, review


def test_metrics_count_independent_reference_errors():
    report = quality.evaluate(*data())
    assert report['top_completeness']['found'] == 1
    assert report['top_completeness']['expected'] == 2
    assert report['original_number_errors'] == report['meeting_section_errors'] == 1
    assert report['assignment_errors'] == report['boundary_errors'] == 1
    assert report['unsupported_outcomes']['vote'] == 1
    assert report['missing_outcomes']['vote'] == 1
    assert report['missing_outcomes']['decision'] == 0
    assert report['remaining_manual_questions'] == 1
    assert report['source_attribution_errors'] == 1


def test_model_generated_reference_is_rejected():
    reference, candidate, review = data()
    reference['origin'] = 'model'
    with pytest.raises(ValueError, match='human'):
        quality.evaluate(reference, candidate, review)


def test_changed_candidate_cannot_reuse_human_judgment():
    reference, candidate, review = data()
    candidate['claims'][0]['text'] = 'Changed'
    with pytest.raises(ValueError, match='different immutable'):
        quality.evaluate(reference, candidate, review)


def test_unprocessed_transcript_is_not_hidden():
    reference, candidate, review = data()
    del candidate['assignments']['l1']
    review['candidate_sha256'] = quality.sha(candidate)
    assert quality.evaluate(reference, candidate, review)['unprocessed_lines'] == 1
