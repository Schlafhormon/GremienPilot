import pytest
from agenda_detection import segment_known_agenda, detect_agenda_from_transcript, _should_use_llm
from assignment_suggestions import TranscriptUtterance


def test_disabled_detection_is_technical_nonprocessing():
    lines = [TranscriptUtterance('M', 'Kommen wir zu TOP 1 Haushalt.')]
    known = segment_known_agenda(lines, ['1 Haushalt'], use_llm=False)
    unknown = detect_agenda_from_transcript(lines, use_llm=False)
    assert known.assignments == unknown.assignments == [None]
    assert unknown.tops == [] and known.tops == ['1 Haushalt']
    for result in (known, unknown):
        assert result.llm.status == 'disabled'
        assert result.llm.line_results[0]['status'] == 'not_processed'
        assert result.llm.gaps[0]['kind'] == 'technical'


@pytest.mark.parametrize('value', [1, 'true', 'false', [], {}])
def test_mode_is_strict_boolean(value):
    with pytest.raises(ValueError):
        _should_use_llm(value)


def test_disabled_and_empty_inputs_do_not_construct_client(monkeypatch):
    import agenda_llm
    monkeypatch.setattr(agenda_llm, 'Workflow', lambda *args: pytest.fail('No client expected'))
    assert segment_known_agenda([], ['Haushalt'], use_llm=True).assignments == []
    assert detect_agenda_from_transcript([], use_llm=False).tops == []


def test_legacy_suggestion_entry_uses_same_model_workflow(agenda_model):
    from assignment_suggestions import suggest_assignments
    result = suggest_assignments([TranscriptUtterance('M', 'Eine indirekte Frage.')], ['Haushalt'])
    assert result.suggested_assignments == [0]
    assert result.strategy == 'model_agenda_v1'
    assert any(b['phase'] == 'independent:detail' for b, _ in agenda_model.calls)
