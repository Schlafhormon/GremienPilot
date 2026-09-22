import pytest

from agenda_labels import parse_agenda_label, reference_targets
from assignment_suggestions import extract_agenda_number, references_top_number, suggest_assignments, TranscriptUtterance


@pytest.mark.parametrize('prefix,number', [
    ('2.', '2'), ('2)', '2'), ('02', '02'), ('TOP 2:', '2'), ('top 2', '2'),
    ('2.1', '2.1'), ('2.2.', '2.2'), ('3.1', '3.1'), ('3.1.2.', '3.1.2'),
    ('I.', 'I'), ('IV.', 'IV'), ('a)', 'a'),
])
def test_lossless_standard_list_formats(prefix, number):
    text = f'{prefix} Schulbau'
    parsed = parse_agenda_label(text)
    assert parsed.original_number == number
    assert parsed.title == 'Schulbau'


@pytest.mark.parametrize('text,key', [
    ('Tagesordnungspunkt zwei', '2'), ('TOP zwölf', '12'), ('Punkt fünf', '5'),
    ('TOP sechzehn', '16'), ('TOP siebzehn', '17'), ('TOP dreißig', '30'),
    ('TOP einundzwanzig', '21'), ('TOP neunundneunzig', '99'),
    ('TOP fuenf', '5'), ('TOP zwoelf', '12'), ('TOP eins', '1'),
    ('TOP 02', '2'), ('TOP 2.1', '2.1'), ('TOP 2.2', '2.2'), ('TOP 2.10', '2.10'), ('TOP 3.1.2', '3.1.2'), ('TOP 2.', '2'),
    ('TOP 2, Schulbau', '2'),
])
def test_exact_number_references(text, key):
    assert references_top_number(text, key)
    for other in {'1', '2', '2.1', '2.2', '12', '21'} - {key}:
        assert not references_top_number(text, other)


@pytest.mark.parametrize('text', [
    'TOP 2a', 'TOP 2/1', 'TOP 2,1', 'TOP 2 . 1', 'TOP zwei Punkt eins',
    'TOP 2-3', 'TOP 2 und 3', 'TOP 2 bis drei', 'TOP 2, 3', 'TOP 2 und TOP 2.1',
])
def test_unsupported_or_multiple_references_never_match_parent(text):
    assert not references_top_number(text, '2')


def test_no_position_fallback_or_title_number_guess():
    assert extract_agenda_number('Haushalt 2026', 2) is None
    assert extract_agenda_number('2.2 Schulbau', 2) == '2.2'
    assert not references_top_number('TOP 2.2', '2')


def test_duplicate_number_requires_unambiguous_section():
    tops = ['[Öffentlich] 2 Haushalt', '[Nichtöffentlich] 2 Personal', '2.1 Schule']
    assert reference_targets('TOP zwei', tops) == (True, {0, 1})
    assert reference_targets('TOP 2 im nichtöffentlichen Teil', tops) == (True, {1})
    assert reference_targets('TOP 2 im öffentlichen Teil', tops) == (True, {0})
    assert reference_targets('TOP 2 öffentlich und nichtöffentlich', tops) == (True, set())
    assert reference_targets('TOP 2', ['2 A', '2 B']) == (True, {0, 1})
    assert reference_targets('TOP 2 öffentlich', ['[Öffentlich] 2 A', '2 B']) == (True, {0, 1})


def test_only_exact_known_numbers_are_certain_even_with_keyword_overlap():
    transcript = [TranscriptUtterance('MOD', t) for t in [
        'Beginn', 'TOP 2.2 Schulbau', 'Diskussion', 'TOP sieben Anfragen', 'Ende',
    ]]
    result = suggest_assignments(transcript, ['2 Haushalt', '2.2 Schulbau', '7 Anfragen'])
    assert result.suggested_assignments == [None, 1, 1, 2, 2]
    assert all(not segment.uncertain for segment in result.segments[1:])
    for tops in [
        ['1 Beginn', '2 Schulbau', '2 Schulbau'],
        ['1 Beginn', '2 Schulbau', '7 Anfragen'],
    ]:
        result = suggest_assignments(transcript, tops)
        assert not any(s.start_index == 1 and not s.uncertain for s in result.segments)


@pytest.mark.parametrize('text', ['2a Schulbau', '2/1 Schulbau', '2,1 Schulbau', '2 . 1 Schulbau', '2.1a Schulbau'])
def test_unsupported_labels_are_preserved_without_a_partial_number(text):
    assert parse_agenda_label(text).original_number is None


@pytest.mark.parametrize('text', ['1.Titel', '2)Titel', '3.1.Titel', 'IV.Titel', 'a)Titel'])
def test_legacy_lists_without_space_after_delimiter(text):
    assert parse_agenda_label(text).title == 'Titel'


@pytest.mark.parametrize('spelling', ['nichtöffentlich', 'nicht öffentlich', 'nicht-öffentlich', 'nicht oeffentlich'])
def test_negative_scope_is_not_mistaken_for_public_scope(spelling):
    tops = ['[Öffentlich] 2 Haushalt', '[Nichtöffentlich] 2 Vergabe']
    assert reference_targets(f'TOP 2 {spelling}', tops) == (True, {1})
