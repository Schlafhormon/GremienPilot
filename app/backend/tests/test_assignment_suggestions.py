import pytest

from assignment_suggestions import TranscriptUtterance, suggest_assignments


def test_suggest_assignments_detects_explicit_moderator_transitions():
    transcript = [
        TranscriptUtterance("SPEAKER_00", "Ich eröffne die Sitzung und begrüße alle."),
        TranscriptUtterance("SPEAKER_01", "Vielen Dank."),
        TranscriptUtterance("SPEAKER_00", "Kommen wir zu TOP 2 Haushalt 2026."),
        TranscriptUtterance("SPEAKER_02", "Der Haushalt enthält Investitionen."),
        TranscriptUtterance("SPEAKER_00", "Als nächstes rufe ich TOP 3 Schulbau auf."),
        TranscriptUtterance("SPEAKER_03", "Beim Schulbau geht es um die Grundschule."),
    ]
    tops = ["1. Begrüßung", "2. Haushalt 2026", "3. Schulbau"]

    result = suggest_assignments(transcript, tops)

    assert result.suggested_assignments == [0, 0, 1, 1, 2, 2]
    assert [(segment.top_index, segment.start_index, segment.end_index) for segment in result.segments] == [
        (0, 0, 1),
        (1, 2, 3),
        (2, 4, 5),
    ]
    assert result.segments[1].confidence >= 0.7
    assert not result.segments[1].uncertain
    assert "TOP 2" in result.segments[1].reason


def test_suggest_assignments_marks_missing_boundaries_as_uncertain():
    transcript = [
        TranscriptUtterance("SPEAKER_00", "Ich eröffne die Sitzung."),
        TranscriptUtterance("SPEAKER_01", "Allgemeine Diskussion ohne klare Stichworte."),
        TranscriptUtterance("SPEAKER_02", "Weitere Wortmeldung."),
        TranscriptUtterance("SPEAKER_03", "Noch eine Wortmeldung."),
    ]
    tops = ["Begrüßung", "Haushalt", "Schulbau"]

    result = suggest_assignments(transcript, tops)

    assert result.segments == []
    assert result.uncertain_count == 0  # counts segments, not unassigned rows
    assert result.suggested_assignments == [None] * len(transcript)


def test_suggest_assignments_uses_topic_keywords_without_explicit_top_number():
    transcript = [
        TranscriptUtterance("MOD", "Begrüßung und Formalien."),
        TranscriptUtterance("MOD", "Dann kommen wir zum Haushalt und zur Finanzplanung."),
        TranscriptUtterance("A", "Die Finanzplanung ist nachvollziehbar."),
        TranscriptUtterance("MOD", "Weiter geht es mit dem Neubau der Grundschule."),
        TranscriptUtterance("B", "Der Neubau ist dringend."),
    ]
    tops = ["Begrüßung", "Haushalt und Finanzplanung", "Neubau Grundschule"]

    result = suggest_assignments(transcript, tops)

    assert result.suggested_assignments == [0, 1, 1, 2, 2]
    assert result.segments[1].transition_type in {"explicit", "keyword"}
    assert result.segments[2].confidence >= 0.55


def utterances(*texts):
    return [TranscriptUtterance('MOD', text) for text in texts]


@pytest.mark.parametrize('tops', [['Begrüßung', 'Haushalt'], ['1 Begrüßung', '2 Haushalt']])
def test_future_reference_does_not_beat_actual_call(tops):
    result = suggest_assignments(utterances(
        'Ich eröffne die Sitzung.',
        'Den Haushalt behandeln wir später unter TOP 2.',
        'Zunächst noch die Anwesenheitsliste.',
        'Kommen wir jetzt zu TOP 2 Haushalt.',
        'Die Einnahmen steigen.',
    ), tops)
    assert result.suggested_assignments == [None, None, None, 1, 1]
    assert len(result.segments) == 1
    assert result.segments[0].evidence_index == 3
    assert not result.segments[0].uncertain


@pytest.mark.parametrize('text', [
    'Kommen wir jetzt zu TOP 2 Haushalt.',
    'Ich rufe TOP 2 Haushalt auf.',
    'Nun kommen wir zum Haushalt.',
    'Wir behandeln jetzt den Haushalt.',
    'Kommen wir wieder zu TOP 2 Haushalt.',
    'Wir nehmen TOP 2 Haushalt wieder auf.',
    'Wir setzen die Beratung zu TOP 2 Haushalt fort.',
    'TOP 2 Haushalt.',
])
def test_current_calls_create_boundaries(text):
    result = suggest_assignments(utterances(text, 'Wortmeldung.'), ['1 Eröffnung', '2 Haushalt'])
    assert result.suggested_assignments == [1, 1]
    assert not result.segments[0].uncertain
    assert result.segments[0].confidence >= 0.7


@pytest.mark.parametrize('text', [
    'Den Haushalt behandeln wir später unter TOP 2.',
    'Kommen wir später zu TOP 2 Haushalt.',
    'Kommen wir noch nicht zu TOP 2 Haushalt.',
    'TOP 2 Haushalt wird vertagt.',
    'TOP 2 Haushalt ist abgesetzt.',
    'TOP 2 Haushalt enthält die Zahlen.',
    'Das gehört zu TOP 2 Haushalt.',
    'Unter TOP 2 Haushalt haben wir das bereits besprochen.',
    'Ich rufe TOP 2 Haushalt nicht auf.',
    'Ich werde TOP 2 Haushalt aufrufen.',
    'Wenn wir zu TOP 2 Haushalt kommen, klären wir das.',
    'Kommen wir jetzt zu TOP 2 Haushalt?',
    'Sie sagte: „Kommen wir zu TOP 2 Haushalt.“',
    'Kommen wir zu TOP 2 Haushalt, sobald die Unterlagen da sind.',
    'Als nächstes werden wir den Haushalt behandeln.',
])
def test_non_calls_never_start_a_segment(text):
    result = suggest_assignments(utterances(text, 'Wortmeldung.'), ['1 Eröffnung', '2 Haushalt'])
    assert result.suggested_assignments == [None, None]
    assert result.segments == []


def test_mentions_do_not_interrupt_an_active_topic():
    result = suggest_assignments(utterances(
        'Kommen wir zu TOP 1 Eröffnung.',
        'Den Haushalt behandeln wir später unter TOP 2.',
        'Weiter mit den Formalien.',
        'Kommen wir zu TOP 2 Haushalt.',
    ), ['1 Eröffnung', '2 Haushalt'])
    assert result.suggested_assignments == [0, 0, 0, 1]


def test_reordering_omission_and_resumption_use_original_identity():
    result = suggest_assignments(utterances(
        'Vorgespräch.',
        'Kommen wir zu TOP 3 Schulbau.',
        'Wortmeldung.',
        'Kommen wir zu TOP 2 Haushalt.',
        'Wortmeldung.',
        'Kommen wir wieder zu TOP 3 Schulbau.',
        'Weitere Wortmeldung.',
    ), ['1 Begrüßung', '2 Haushalt', '3 Schulbau', '4 Anfragen'])
    assert result.suggested_assignments == [None, 2, 2, 1, 1, 2, 2]
    assert [(s.top_index, s.start_index, s.end_index) for s in result.segments] == [(2, 1, 2), (1, 3, 4), (2, 5, 6)]


def test_keyword_evidence_stays_local_uncertain_and_cannot_steal_later_call():
    result = suggest_assignments(utterances(
        'Der Haushalt enthält Investitionen.',
        *(['Allgemeine Diskussion.'] * 30),
        'Kommen wir zum Haushalt.',
        'Die Einnahmen steigen.',
    ), ['Begrüßung', 'Haushalt'])
    assert result.segments[0].uncertain
    assert result.segments[0].confidence < 0.7
    assert result.segments[0].start_index == result.segments[0].end_index == 0
    assert result.suggested_assignments[1:31] == [None] * 30
    assert result.segments[-1].start_index == 31
    assert result.segments[-1].confidence == 0.8
    assert result.uncertain_count == 1


@pytest.mark.parametrize('interruption', [
    'Kommen wir zu TOP 99.',
    'Kommen wir zu TOP 1 und 2.',
    'Kommen wir zum Haushalt und zum Schulbau.',
    'Wir unterbrechen die Beratung.',
    'Ich schließe die Sitzung.',
])
def test_unknown_joint_calls_and_closing_leave_gaps(interruption):
    result = suggest_assignments(utterances(
        'Kommen wir zu TOP 1 Haushalt.', interruption, 'Wortmeldung.',
        'Kommen wir zu TOP 2 Schulbau.',
    ), ['1 Haushalt', '2 Schulbau'])
    assert result.suggested_assignments == [0, None, None, 1]


def test_more_topics_than_rows_and_empty_inputs_do_not_invent_segments():
    for texts, tops in [([], ['1 Haushalt']), (['Hallo.'], []), (['Hallo.'], ['Haushalt', 'Schulbau', 'Anfragen'])]:
        result = suggest_assignments(utterances(*texts), tops)
        assert result.suggested_assignments == [None] * len(texts)
        assert result.segments == []


def test_duplicate_titles_are_not_resolved_by_first_match_or_speaker_frequency():
    result = suggest_assignments(utterances(*(['Haushalt.'] * 20), 'Kommen wir zum Haushalt.'), ['Haushalt', 'Haushalt'])
    assert result.suggested_assignments == [None] * 21


@pytest.mark.parametrize('text', [
    'TOP 2 Haushalt wächst.',
    'Kommen wir zu TOP 2 Schulbau.',
    'Kommen wir zu TOP 2 Haushalt. TOP 3 Schulbau behandeln wir später.',
])
def test_conflicting_or_mixed_evidence_does_not_become_a_safe_boundary(text):
    result = suggest_assignments(utterances(
        'Kommen wir zu TOP 1 Eröffnung.', text, 'Wortmeldung.',
    ), ['1 Eröffnung', '2 Haushalt', '3 Schulbau'])
    assert result.suggested_assignments == [0, None, None]


@pytest.mark.parametrize('cancellation', [
    'Wir vertagen TOP 2 Haushalt.',
    'TOP 2 Haushalt wird vertagt.',
    'TOP 2 Haushalt ist abgesetzt.',
])
def test_current_topic_can_be_suspended_and_resumed(cancellation):
    result = suggest_assignments(utterances(
        'Kommen wir zu TOP 2 Haushalt.', cancellation, 'Pause.',
        'Kommen wir wieder zu TOP 2 Haushalt.',
    ), ['1 Eröffnung', '2 Haushalt'])
    assert result.suggested_assignments == [1, None, None, 1]


def test_unrelated_cancellation_does_not_invent_a_segment():
    result = suggest_assignments(utterances(
        'Kommen wir zu TOP 1 Eröffnung.',
        'TOP 2 Haushalt ist abgesetzt.',
        'Weitere Formalien.',
        'Kommen wir zu TOP 3 Schulbau.',
    ), ['1 Eröffnung', '2 Haushalt', '3 Schulbau'])
    assert result.suggested_assignments == [0, 0, 0, 2]
    assert [s.top_index for s in result.segments] == [0, 2]


def test_unlabelled_numbers_are_never_resolved_by_list_position():
    tops = ['Schulbau', 'Haushalt']
    assert suggest_assignments(utterances('Kommen wir zu TOP 2.'), tops).suggested_assignments == [None]
    result = suggest_assignments(utterances('Kommen wir zu TOP 99 Haushalt.'), tops)
    assert result.suggested_assignments == [1]
    assert result.segments[0].confidence == 0.75
    assert 'Nummer ist nicht hinterlegt' in result.segments[0].reason
