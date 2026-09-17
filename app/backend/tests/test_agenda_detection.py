from agenda_detection import (
    DEFAULT_AGENDA_DETECTION_PROMPT,
    build_agenda_detection_system_prompt,
    detect_agenda_from_transcript,
    segment_known_agenda,
)
from assignment_suggestions import TranscriptUtterance
import agenda_detection
import pytest


def test_segment_known_agenda_detects_clear_top_announcements():
    transcript = [
        TranscriptUtterance("MOD", "Ich eröffne die Sitzung und begrüße alle."),
        TranscriptUtterance("A", "Vielen Dank."),
        TranscriptUtterance("MOD", "Kommen wir zu TOP 2 Haushalt 2026."),
        TranscriptUtterance("B", "Der Haushalt enthält Investitionen."),
        TranscriptUtterance("MOD", "Als nächstes rufe ich TOP 3 Schulbau auf."),
        TranscriptUtterance("C", "Beim Schulbau geht es um die Grundschule."),
    ]
    tops = ["Begrüßung", "Haushalt 2026", "Schulbau"]

    result = segment_known_agenda(transcript, tops)

    assert result.tops == tops
    assert result.assignments == [0, 0, 1, 1, 2, 2]
    assert [(segment.start_index, segment.end_index) for segment in result.segments] == [
        (0, 1),
        (2, 3),
        (4, 5),
    ]
    assert result.segments[1].confidence >= 0.7


def test_detect_agenda_from_transcript_without_known_tops():
    transcript = [
        TranscriptUtterance("MOD", "Kommen wir zu TOP 1 Haushalt 2026."),
        TranscriptUtterance("A", "Die Investitionen sind eingeplant."),
        TranscriptUtterance("MOD", "Als nächstes rufe ich TOP 2 Schulbau auf."),
        TranscriptUtterance("B", "Die Grundschule braucht mehr Räume."),
    ]

    result = detect_agenda_from_transcript(transcript)

    assert result.tops == ["Haushalt 2026", "Schulbau"]
    assert result.assignments == [0, 0, 1, 1]
    assert [(segment.top_title, segment.start_index, segment.end_index) for segment in result.segments] == [
        ("Haushalt 2026", 0, 1),
        ("Schulbau", 2, 3),
    ]
    assert result.strategy == "heuristic_transcript_fallback"


def test_segment_known_agenda_marks_uncertain_boundaries():
    transcript = [
        TranscriptUtterance("MOD", "Ich eröffne die Sitzung."),
        TranscriptUtterance("A", "Allgemeine Wortmeldung ohne Stichworte."),
        TranscriptUtterance("B", "Weitere Wortmeldung."),
        TranscriptUtterance("C", "Noch eine Wortmeldung."),
    ]

    result = segment_known_agenda(transcript, ["Begrüßung", "Haushalt", "Schulbau"])

    assert len(result.segments) == 3
    assert result.uncertain_count == 2
    assert all(segment.uncertain for segment in result.segments[1:])
    assert result.assignments.count(None) == 0


def test_llm_invalid_boundaries_are_repaired(fake_openai_module):
    fake_openai_module.content = """
    {
      "tops": [
        {
          "top_title": "Haushalt",
          "start_index": -5,
          "end_index": 99,
          "confidence": 0.91,
          "evidence_text": "Kommen wir zu TOP 1 Haushalt.",
          "uncertain": false
        },
        {
          "top_title": "Schulbau",
          "start_index": 0,
          "end_index": 1,
          "confidence": 0.88,
          "evidence_text": "TOP 2 Schulbau.",
          "uncertain": false
        }
      ]
    }
    """
    transcript = [
        TranscriptUtterance("MOD", "Kommen wir zu TOP 1 Haushalt."),
        TranscriptUtterance("A", "Der Haushalt wird beraten."),
        TranscriptUtterance("MOD", "TOP 2 Schulbau."),
    ]

    result = detect_agenda_from_transcript(transcript, model="test-model")

    assert result.strategy == "heuristic_transcript_llm_repaired"
    assert [(segment.start_index, segment.end_index) for segment in result.segments] == [
        (0, 1),
        (2, 2),
    ]
    assert result.assignments == [0, 0, 1]
    assert result.segments[0].uncertain
    assert result.segments[1].uncertain
    request = fake_openai_module.instances[0].calls[0]
    assert not request["messages"][0]["content"].startswith("/no_think")


def test_llm_reasoning_field_is_used_when_content_is_empty(fake_openai_module):
    fake_openai_module.content = ""
    fake_openai_module.reasoning = """
    {"tops": [
      {
        "top_title": "Haushalt",
        "start_index": 0,
        "end_index": 1,
        "confidence": 0.91,
        "evidence_text": "TOP 1 Haushalt.",
        "uncertain": false
      }
    ]}
    """
    transcript = [
        TranscriptUtterance("MOD", "TOP 1 Haushalt."),
        TranscriptUtterance("A", "Der Haushalt wird beraten."),
    ]

    result = detect_agenda_from_transcript(transcript, model="test-model")

    assert result.strategy == "heuristic_transcript_llm"
    assert result.tops == ["Haushalt"]
    assert result.assignments == [0, 0]


def test_fallback_without_llm_returns_reviewable_assignments():
    transcript = [
        TranscriptUtterance("MOD", "TOP 1 Genehmigung der Niederschrift."),
        TranscriptUtterance("A", "Keine Einwände."),
        TranscriptUtterance("MOD", "TOP 2 Verschiedenes."),
    ]

    result = detect_agenda_from_transcript(transcript)

    assert result.tops == ["Genehmigung der Niederschrift", "Verschiedenes"]
    assert result.assignments == [0, 0, 1]
    assert result.strategy == "heuristic_transcript_fallback"


def test_unknown_agenda_llm_detection_chunks_long_transcripts(
    fake_openai_module,
    monkeypatch,
    frontend_summary_prompt,
):
    monkeypatch.setattr(agenda_detection, "AGENDA_DETECTION_CHUNK_LINES", 2)
    monkeypatch.setattr(agenda_detection, "AGENDA_DETECTION_CHUNK_OVERLAP_LINES", 0)
    fake_openai_module.responses = [
        """
        {"tops": [
          {"top_title": "Haushalt", "start_index": 0, "end_index": 1, "confidence": 0.9}
        ]}
        """,
        """
        {"tops": [
          {"top_title": "Schulbau", "start_index": 0, "end_index": 1, "confidence": 0.88}
        ]}
        """,
    ]
    transcript = [
        TranscriptUtterance("MOD", "Kommen wir zu TOP 1 Haushalt."),
        TranscriptUtterance("A", "Der Haushalt wird beraten."),
        TranscriptUtterance("MOD", "Kommen wir zu TOP 2 Schulbau."),
        TranscriptUtterance("B", "Der Schulbau wird beraten."),
    ]

    result = detect_agenda_from_transcript(
        transcript, model="test-model", system_prompt=frontend_summary_prompt,
    )

    assert result.tops == ["Haushalt", "Schulbau"]
    assert result.assignments == [0, 0, 1, 1]
    assert [(segment.start_index, segment.end_index) for segment in result.segments] == [
        (0, 1),
        (2, 3),
    ]
    calls = [call for instance in fake_openai_module.instances for call in instance.calls]
    assert len(calls) == 2
    for call in calls:
        assert DEFAULT_AGENDA_DETECTION_PROMPT in call["messages"][0]["content"]
        assert frontend_summary_prompt in call["messages"][0]["content"]
    assert "0: MOD" in calls[0]["messages"][1]["content"]
    assert "2: MOD" not in calls[0]["messages"][1]["content"]


def test_build_agenda_detection_system_prompt_does_not_force_no_think():
    prompt = build_agenda_detection_system_prompt("Nur JSON")
    assert DEFAULT_AGENDA_DETECTION_PROMPT in prompt
    assert "Nur JSON" in prompt
    assert not prompt.startswith("/no_think")


@pytest.mark.parametrize("custom_prompt", [None, "", "   ", DEFAULT_AGENDA_DETECTION_PROMPT])
def test_default_detection_contract(custom_prompt):
    assert build_agenda_detection_system_prompt(custom_prompt) == DEFAULT_AGENDA_DETECTION_PROMPT


@pytest.mark.parametrize("known_tops", [False, True])
def test_frontend_summary_prompt_cannot_replace_detection_contract(
    fake_openai_module, frontend_summary_prompt, known_tops,
):
    fake_openai_module.content = '{"tops": [{"top_title": "Haushalt", "start_index": 0, "end_index": 0, "confidence": 0.9}]}'
    transcript = [TranscriptUtterance("MOD", "TOP 1 Haushalt.")]
    if known_tops:
        segment_known_agenda(transcript, ["Haushalt"], model="test-model", system_prompt=frontend_summary_prompt)
    else:
        detect_agenda_from_transcript(transcript, model="test-model", system_prompt=frontend_summary_prompt)
    request = fake_openai_module.instances[0].calls[0]
    prompt = request["messages"][0]["content"]
    assert DEFAULT_AGENDA_DETECTION_PROMPT in prompt
    assert frontend_summary_prompt in prompt
    assert "diese haben Vorrang" in prompt
    assert "0: MOD" in request["messages"][1]["content"]
