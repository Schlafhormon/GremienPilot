import pytest
from agenda_context import source_rows, model_agenda
from assignment_suggestions import TranscriptUtterance


def test_identity_and_time_are_preserved_not_reconstructed():
    line = TranscriptUtterance('M', 'TOP 1 wird erwähnt.', 'permanent-line', 10.125, 21.875)
    assert source_rows([line]) == [dict(index=0, line_id='permanent-line', speaker='M',
        text=line.text, start=10.125, end=21.875)]
    assert model_agenda(['[Öffentlich] 01 Titel', '[Nichtöffentlich] 01 Titel'], ['a', 'b']) == [
        dict(top_id='a', title='[Öffentlich] 01 Titel', number='01', section='public'),
        dict(top_id='b', title='[Nichtöffentlich] 01 Titel', number='01', section='nonpublic')]


def test_legacy_source_ids_are_deterministic_and_content_bound():
    lines = [TranscriptUtterance('M', 'Beratung.')]*2
    assert source_rows(lines) == source_rows(lines)
    assert source_rows(lines)[0]['line_id'] != source_rows(lines)[1]['line_id']
    assert source_rows(lines)[0]['line_id'] != source_rows([TranscriptUtterance('M', 'Geändert.')])[0]['line_id']


@pytest.mark.parametrize('start,end', [(-1, 2), (3, 2), (float('nan'), 2), (0, float('inf')), (None, 2)])
def test_invalid_source_times_rejected(start, end):
    with pytest.raises(ValueError):
        source_rows([TranscriptUtterance('M', 'Text', 'line', start, end)])


def test_duplicate_source_and_top_ids_rejected():
    with pytest.raises(ValueError):
        source_rows([TranscriptUtterance('M', 'Text', 'same')]*2)
    with pytest.raises(ValueError):
        model_agenda(['A', 'B'], ['same', 'same'])
    with pytest.raises(ValueError):
        model_agenda(['A', 'B'], ['only-one'])
