"""Compact model output must preserve coverage, joint assignments and blind review."""
import pytest
from test_agenda_llm import run


@pytest.fixture(autouse=True)
def compact(monkeypatch):
    monkeypatch.setenv('LLM_CONTEXT_TOKENS', '65536')
    monkeypatch.setenv('AGENDA_COMPACT_ASSIGNMENTS', 'true')
    monkeypatch.setenv('AGENDA_OUTPUT_TOKENS_PER_LINE', '40')
    monkeypatch.setenv('AGENDA_DETECTION_CHUNK_LINES', '80')


def test_compact_full_coverage_and_blind_review(agenda_model):
    agenda_model.labels = {i: ['agenda:0'] if i < 85 else ['agenda:1'] for i in range(177)}
    result = run(['Beratung.'] * 177)
    assert result.assignments == [0]*85 + [1]*92
    assert result.llm.processing_complete and result.llm.review_complete
    for role in ('primary', 'independent'):
        calls = [body for body, _ in agenda_model.calls if body['phase'] == role + ':detail']
        assert len(calls) == 3
        assert [r['index'] for b in calls for r in b['target_lines']] == list(range(177))
        assert all(b['opinions'] is None for b in calls)
    assert len(result.llm.line_results) == 177


def test_compact_joint_gap_and_disagreement(agenda_model):
    agenda_model.labels = {0: ['agenda:0', 'agenda:1'], 1: [], 2: ['agenda:0']}
    agenda_model.review_labels = {2: ['agenda:1']}
    agenda_model.resolve_labels = {2: ['agenda:1']}
    result = run(['Gemeinsame Beratung.', 'Pause.', 'Schulbau.'])
    assert result.assignments == [None, None, 1]
    assert result.llm.line_results[0]['top_ids'] == ['agenda:0', 'agenda:1']
    assert result.llm.line_results[2]['review_status'] == 'resolved'
    assert result.llm.processing_complete and result.llm.review_complete


@pytest.mark.parametrize('mutation', ['gap', 'overlap', 'outside', 'bool', 'reversed', 'unknown_top', 'quote', 'confidence'])
def test_invalid_compact_spans_are_technical_failure(agenda_model, monkeypatch, mutation):
    monkeypatch.setenv('AGENDA_REPAIR_SPLIT_DEPTH', '0')
    def bad(body):
        span = dict(start=0, end=2, top_ids=['agenda:0'], reason='Beratung',
                    evidence=[dict(line_id='line-0', quote='Beratung.')], uncertain=False, confidence=0.8)
        spans = [span]
        if mutation == 'gap': span['end'] = 1
        elif mutation == 'overlap': spans.append(dict(span))
        elif mutation == 'outside': span['end'] = 3
        elif mutation == 'bool': span['start'] = False
        elif mutation == 'reversed': span.update(start=2, end=0)
        elif mutation == 'unknown_top': span['top_ids'] = ['foreign']
        elif mutation == 'quote': span['evidence'][0]['quote'] = 'invented'
        else: span['confidence'] = float('nan')
        return dict(source_ranges=[], spans=spans)
    agenda_model.overrides['primary:detail'] = bad
    result = run(['Beratung.']*3)
    assert not result.llm.processing_complete
    assert all(g['kind'] == 'technical' for g in result.llm.gaps)


def test_compact_failed_large_response_splits_without_losing_lines(agenda_model):
    def limited(body):
        if len(body['target_lines']) > 2:
            return dict(source_ranges=[], spans=[])
        return agenda_model.answer(body)
    agenda_model.overrides['primary:detail'] = limited
    result = run(['Beratung.']*8)
    assert result.assignments == [0]*8
    assert result.llm.processing_complete and result.llm.review_complete


def test_compact_requests_original_sources_before_deciding(agenda_model):
    def retrieve(body):
        if not body.get('requested_originals'):
            return dict(source_ranges=[dict(start=0, end=2)], spans=[])
        return agenda_model.answer(body)
    agenda_model.overrides['primary:detail'] = retrieve
    result = run(['Beratung.']*3)
    assert result.llm.processing_complete and result.llm.review_complete
    assert any(b.get('requested_originals') for b, _ in agenda_model.calls)
