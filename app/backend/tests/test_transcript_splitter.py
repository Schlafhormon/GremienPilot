from copy import deepcopy
import math
import re

import pytest

from transcript_splitter import (
    split_transcript_at_agenda_transitions,
    split_transcript_for_agenda_detection,
)


FIRST_SENTENCE = (
    "Dies ist ein ausreichend langer erster Satz mit Erläuterungen zur bisherigen Beratung."
)
SECOND_SENTENCE = "Danach folgt ein weiterer längerer Satz zum gleichen Thema."


def normalized_text(text):
    return re.sub(r"\s+", " ", text).strip()


def test_split_transcript_for_agenda_detection_splits_long_sentence_chunks():
    transcript = [
        {
            "speaker": "SPEAKER_04",
            "text": (
                "Das ist einstimmig. Kommen wir zum Tagesordnungspunkt 3. "
                "Wir haben heute einen neuen sachkundigen Einwohner."
            ),
            "start": 10.0,
            "end": 20.0,
        }
    ]

    result = split_transcript_for_agenda_detection(transcript)

    assert [line["text"] for line in result] == [
        "Das ist einstimmig.",
        "Kommen wir zum Tagesordnungspunkt 3.",
        "Wir haben heute einen neuen sachkundigen Einwohner.",
    ]
    assert all(line["speaker"] == "SPEAKER_04" for line in result)
    assert result[0]["start"] == 10.0
    assert result[0]["end"] == result[1]["start"]
    assert result[1]["end"] == result[2]["start"]
    assert result[2]["end"] == 20.0
    assert len({line["line_id"] for line in result}) == 3


def test_split_transcript_for_agenda_detection_does_not_use_top_phrases():
    transcript = [
        {
            "speaker": "SPEAKER_04",
            "text": (
                "Das ist einstimmig. Danach beraten wir den nächsten Abschnitt. "
                "Die Verwaltung erläutert den Sachstand."
            ),
            "start": 10.0,
            "end": 20.0,
        }
    ]

    result = split_transcript_for_agenda_detection(transcript)

    assert [line["text"] for line in result] == [
        "Das ist einstimmig.",
        "Danach beraten wir den nächsten Abschnitt.",
        "Die Verwaltung erläutert den Sachstand.",
    ]


def test_split_transcript_for_agenda_detection_keeps_short_line():
    transcript = [
        {
            "speaker": "SPEAKER_04",
            "text": "Kommen wir zum Tagesordnungspunkt 3.",
            "start": 10.0,
            "end": 20.0,
        }
    ]

    assert split_transcript_for_agenda_detection(transcript) == transcript


@pytest.mark.parametrize("tail", ["TOP 2.", "Ja.", "X", "Ende?!", "Äh…"])
def test_short_tail_is_retained_after_multiple_long_chunks(tail):
    text = f"{FIRST_SENTENCE} {SECOND_SENTENCE} {tail}"
    result = split_transcript_for_agenda_detection([{"text": text}])

    assert [line["text"] for line in result] == [FIRST_SENTENCE, SECOND_SENTENCE, tail]
    assert " ".join(line["text"] for line in result) == text


def test_long_sentence_with_only_a_short_tail_is_stable_when_split_again():
    result = split_transcript_for_agenda_detection([
        {"text": f"{FIRST_SENTENCE} TOP 2.", "start": 1.0, "end": 9.0},
    ])

    assert [line["text"] for line in result] == [FIRST_SENTENCE, "TOP 2."]
    assert split_transcript_for_agenda_detection(result) == result


@pytest.mark.parametrize("separator", [" ", "\n", "\t", "  \n\t "])
@pytest.mark.parametrize("punctuation", [".", "!", "?", "?!", "...", ".“", "…"])
def test_content_order_and_punctuation_survive_splitting(separator, punctuation):
    # Also exercise punctuation the conservative boundary heuristic does not split.
    text = separator.join([
        "Ja.", FIRST_SENTENCE[:-1] + punctuation, "Oh!", "Hm?",
        SECOND_SENTENCE, "TOP 2.",
    ])
    source = [{"text": f"  {text}\n", "start": 3.0, "end": 17.0}]
    result = split_transcript_for_agenda_detection(source)

    assert normalized_text(" ".join(line["text"] for line in result)) == normalized_text(text)
    assert all(line["text"].strip() for line in result)
    assert split_transcript_for_agenda_detection(result) == result


@pytest.mark.parametrize("text", [
    "", " \t\n", "TOP 2.", "Wort " * 40,
    "Ein langer Satz ohne passende Trennstelle " * 4 + ".",
    "Ein langer Beitrag; ohne erkannte Satzgrenze: " * 4,
])
def test_lines_without_usable_boundaries_are_unchanged(text):
    source = [{"text": text, "speaker": "SPEAKER_01", "start": 2, "end": 9}]

    result = split_transcript_for_agenda_detection(source)

    assert result == source
    assert result[0] is not source[0]


def test_split_preserves_speaker_attributes_ids_and_source_without_mutation():
    source = [
        {
            "line_id": "original-line",
            "speaker": "SPEAKER_04",
            "speaker_name": "Erika Beispiel",
            "speaker_metadata": {"confidence": 0.75},
            "text": f"{FIRST_SENTENCE} {SECOND_SENTENCE} TOP 2.",
            "start": 10.0,
            "end": 20.0,
        },
        {"line_id": "next-line", "speaker": "SPEAKER_02", "text": "Ja.",
         "start": 22.0, "end": 23.0},
    ]
    before = deepcopy(source)

    result = split_transcript_for_agenda_detection(source)

    assert source == before
    assert len(result) == 4
    assert result[0]["line_id"] == "original-line"
    assert result[-1] == source[-1]
    assert len({line["line_id"] for line in result}) == len(result)
    for line in result[:-1]:
        for key in ("speaker", "speaker_name", "speaker_metadata"):
            assert line[key] == source[0][key]
    # Repeated endpoint calls must not accumulate chunks or alter IDs/times.
    for _ in range(3):
        assert split_transcript_for_agenda_detection(result) == result
        assert split_transcript_at_agenda_transitions(result) == result


@pytest.mark.parametrize("start,end", [(10.0, 20.0), (0.0, 0.0), (2.5, 2.5), (2.5, 2.501)])
def test_estimated_times_are_contiguous_and_within_source_bounds(start, end):
    text = f"  {FIRST_SENTENCE}\n\t{SECOND_SENTENCE}   TOP 2.  "

    result = split_transcript_for_agenda_detection([
        {"text": text, "start": start, "end": end},
    ])

    assert result[0]["start"] == start
    assert result[-1]["end"] == end
    assert all(start <= line["start"] <= line["end"] <= end for line in result)
    assert all(left["end"] == right["start"] for left, right in zip(result, result[1:]))
    # These assertions check interval consistency, not real speech alignment.
    assert split_transcript_for_agenda_detection(result) == result


@pytest.mark.parametrize("times,expected_bounds", [
    ({}, (0.0, 0.0)),
    ({"start": "10", "end": "20"}, (10.0, 20.0)),
    ({"start": 10, "end": 5}, (10.0, 10.0)),
    ({"start": 10, "end": None}, (10.0, 10.0)),
    ({"start": "invalid", "end": 5}, (0.0, 5.0)),
    ({"start": float("nan"), "end": float("inf")}, (0.0, 0.0)),
    ({"start": 10, "end": float("-inf")}, (10.0, 10.0)),
])
def test_invalid_times_have_finite_ordered_fallbacks(times, expected_bounds):
    source = [{"text": f"{FIRST_SENTENCE} {SECOND_SENTENCE} TOP 2.", **times}]

    result = split_transcript_for_agenda_detection(source)

    assert (result[0]["start"], result[-1]["end"]) == expected_bounds
    assert all(math.isfinite(line[key]) for line in result for key in ("start", "end"))
    assert all(line["start"] <= line["end"] for line in result)
    assert all(left["end"] == right["start"] for left, right in zip(result, result[1:]))
