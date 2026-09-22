import pytest

from agenda_labels import parse_agenda_label


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


@pytest.mark.parametrize('text', ['2a Schulbau', '2/1 Schulbau', '2,1 Schulbau', '2 . 1 Schulbau', '2.1a Schulbau'])
def test_unsupported_labels_are_preserved_without_a_partial_number(text):
    assert parse_agenda_label(text).original_number is None


@pytest.mark.parametrize('text', ['1.Titel', '2)Titel', '3.1.Titel', 'IV.Titel', 'a)Titel'])
def test_legacy_lists_without_space_after_delimiter(text):
    assert parse_agenda_label(text).title == 'Titel'
