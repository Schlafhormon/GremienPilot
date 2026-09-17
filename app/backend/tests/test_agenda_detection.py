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
    tops = ["1. Begrüßung", "2. Haushalt 2026", "3. Schulbau"]

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

    assert result.tops == ["TOP 1 Haushalt 2026", "TOP 2 Schulbau"]
    assert result.assignments == [0, 0, 1, 1]
    assert [(segment.top_title, segment.start_index, segment.end_index) for segment in result.segments] == [
        ("TOP 1 Haushalt 2026", 0, 1),
        ("TOP 2 Schulbau", 2, 3),
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

    result = detect_agenda_from_transcript(transcript, model="test-model", use_llm=True)

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

    result = detect_agenda_from_transcript(transcript, model="test-model", use_llm=True)

    assert result.strategy == "heuristic_transcript_llm"
    assert result.tops == ["TOP 1 Haushalt"]
    assert result.assignments == [0, 0]


def test_fallback_without_llm_returns_reviewable_assignments():
    transcript = [
        TranscriptUtterance("MOD", "TOP 1 Genehmigung der Niederschrift."),
        TranscriptUtterance("A", "Keine Einwände."),
        TranscriptUtterance("MOD", "TOP 2 Verschiedenes."),
    ]

    result = detect_agenda_from_transcript(transcript)

    assert result.tops == ["TOP 1 Genehmigung der Niederschrift", "TOP 2 Verschiedenes"]
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
        transcript, model="test-model", use_llm=True, system_prompt=frontend_summary_prompt,
    )

    assert result.tops == ["TOP 1 Haushalt", "TOP 2 Schulbau"]
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
        segment_known_agenda(transcript, ["Haushalt"], model="test-model", use_llm=True, system_prompt=frontend_summary_prompt)
    else:
        detect_agenda_from_transcript(transcript, model="test-model", use_llm=True, system_prompt=frontend_summary_prompt)
    request = fake_openai_module.instances[0].calls[0]
    prompt = request["messages"][0]["content"]
    assert DEFAULT_AGENDA_DETECTION_PROMPT in prompt
    assert frontend_summary_prompt in prompt
    assert "diese haben Vorrang" in prompt
    assert "0: MOD" in request["messages"][1]["content"]


@pytest.mark.parametrize("known_tops", [False, True])
@pytest.mark.parametrize("server_default", [False, True])
@pytest.mark.parametrize("decision", [None, False, True])
@pytest.mark.parametrize("context", [{}, {"model": "chosen"}, {"system_prompt": "custom"}])
def test_explicit_decision_overrides_default_without_content_side_effects(
    monkeypatch, fake_openai_module, known_tops, server_default, decision, context,
):
    monkeypatch.setattr(agenda_detection, "AGENDA_DETECTION_USE_LLM", server_default)
    fake_openai_module.content = '{"tops":[{"top_title":"Haushalt","start_index":0,"end_index":0}]}'
    transcript = [TranscriptUtterance("MOD", "TOP 1 Haushalt.")]
    result = (segment_known_agenda(transcript, ["Haushalt"], use_llm=decision, **context)
              if known_tops else detect_agenda_from_transcript(transcript, use_llm=decision, **context))
    enabled = server_default if decision is None else decision
    assert len(fake_openai_module.instances) == int(enabled)
    assert result.llm.enabled is enabled
    assert result.llm.source == ("server_default" if decision is None else "request")
    assert result.llm.status == ("success" if enabled else "disabled")
    assert result.llm.warnings == []


def test_actual_client_timeout_and_retry_policy(monkeypatch, fake_openai_module):
    monkeypatch.setattr(agenda_detection, "AGENDA_DETECTION_TIMEOUT_SECONDS", 1.25)
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "999")
    detect_agenda_from_transcript([TranscriptUtterance("MOD", "TOP 1 Haushalt.")], use_llm=True)
    assert fake_openai_module.instances[0].kwargs["timeout"] == 1.25
    assert fake_openai_module.instances[0].kwargs["max_retries"] == 0


@pytest.mark.parametrize("known_tops", [False, True])
@pytest.mark.parametrize("response, reason", [
    (TimeoutError("SECRET_TRANSCRIPT token=SECRET_KEY"), "timeout"),
    (ConnectionError("SECRET_TRANSCRIPT token=SECRET_KEY"), "connection_error"),
    (RuntimeError("SECRET_TRANSCRIPT token=SECRET_KEY"), "request_error"),
    ("SECRET_TRANSCRIPT token=SECRET_KEY", "invalid_response"),
    ('{"tops":42}', "invalid_response"),
    ('{"tops":[]}', "empty_response"),
])
def test_safe_observable_fallback(fake_openai_module, caplog, known_tops, response, reason):
    fake_openai_module.responses = [response]
    transcript = [TranscriptUtterance("MOD", "TOP 1 Haushalt.")]
    result = (segment_known_agenda(transcript, ["Haushalt"], use_llm=True)
              if known_tops else detect_agenda_from_transcript(transcript, use_llm=True))
    assert result.assignments == [0]
    assert "llm_fallback" in result.strategy
    assert result.llm.status == "fallback"
    assert result.llm.attempted_calls == result.llm.failed_calls == 1
    assert result.llm.failure_reasons == [reason]
    assert result.llm.warnings
    assert reason in caplog.text
    assert "SECRET" not in caplog.text + str(result.llm)


@pytest.mark.parametrize("all_fail", [False, True])
def test_chunk_failures_remain_visible_and_reviewable(monkeypatch, fake_openai_module, all_fail):
    monkeypatch.setattr(agenda_detection, "AGENDA_DETECTION_CHUNK_LINES", 1)
    monkeypatch.setattr(agenda_detection, "AGENDA_DETECTION_CHUNK_OVERLAP_LINES", 0)
    fake_openai_module.responses = [
        TimeoutError("secret"),
        TimeoutError("secret") if all_fail else '{"tops":[{"top_title":"Schulbau","start_index":0,"end_index":0}]}',
    ]
    result = detect_agenda_from_transcript([
        TranscriptUtterance("MOD", "TOP 1 Haushalt."),
        TranscriptUtterance("MOD", "TOP 2 Schulbau."),
    ], use_llm=True)
    assert result.tops == ["TOP 1 Haushalt", "TOP 2 Schulbau"]
    assert result.assignments == [0, 1]
    assert result.llm.status == ("fallback" if all_fail else "partial_fallback")
    assert result.llm.attempted_calls == 2
    assert result.llm.failed_calls == (2 if all_fail else 1)
    assert result.llm.failure_reasons == ["timeout"]
    if not all_fail:
        assert result.segments[0].uncertain


def test_empty_transcript_does_not_attempt_llm(fake_openai_module):
    result = detect_agenda_from_transcript([], use_llm=True)
    assert result.llm.status == "skipped"
    assert result.llm.attempted_calls == 0
    assert not fake_openai_module.instances


@pytest.mark.parametrize('use_llm', [False, True])
def test_repeated_numbers_cannot_become_certain_via_llm(fake_openai_module, use_llm):
    tops = ['[Öffentlich] 2.1 Schulbau', '[Nichtöffentlich] 2.1 Vergabe']
    transcript = [TranscriptUtterance('MOD', text) for text in [
        'TOP 2.1 Schulbau', 'Beratung', 'TOP 2.1 Vergabe', 'Beratung',
    ]]
    fake_openai_module.content = '''{"tops":[
        {"top_title":"Schulbau","start_index":0,"end_index":1,"confidence":0.99},
        {"top_title":"Vergabe","start_index":2,"end_index":3,"confidence":0.99}
    ]}'''
    result = segment_known_agenda(transcript, tops, use_llm=use_llm)
    assert result.tops == tops
    assert all(segment.uncertain for segment in result.segments)
    assert all(segment.confidence <= 0.5 for segment in result.segments)


@pytest.mark.parametrize('announcement,label', [
    ('TOP 3.1 Schulbau', 'TOP 3.1 Schulbau'),
    ('Tagesordnungspunkt zwei Schulbau', 'TOP 2 Schulbau'),
    ('TOP 02.10 Schulbau', 'TOP 02.10 Schulbau'),
])
def test_transcript_detection_keeps_number_with_or_without_llm(fake_openai_module, announcement, label):
    transcript = [TranscriptUtterance('MOD', announcement)]
    fake_openai_module.content = '{"tops":[{"top_title":"Schulbau","start_index":0,"end_index":0,"confidence":0.9}]}'
    for use_llm in (False, True):
        result = detect_agenda_from_transcript(transcript, use_llm=use_llm)
        assert result.tops == [label]
        assert not result.segments[0].uncertain
