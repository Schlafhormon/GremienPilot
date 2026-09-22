import json

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

    assert result.segments == []
    assert result.uncertain_count == 0  # counts segments, not unassigned rows
    assert result.assignments == [None] * len(transcript)


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


def test_llm_reasoning_field_is_never_used_as_final_content(fake_openai_module):
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

    assert result.strategy == "heuristic_transcript_llm_fallback"
    assert result.llm.failed_calls == 1
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
    from agenda_llm import PROMPT
    assert (PROMPT if known_tops else DEFAULT_AGENDA_DETECTION_PROMPT) in prompt
    assert frontend_summary_prompt in prompt
    assert ("Schema bleibt verbindlich" if known_tops else "diese haben Vorrang") in prompt
    assert ('"index": 0' if known_tops else "0: MOD") in request["messages"][1]["content"]


@pytest.mark.parametrize("known_tops", [False, True])
@pytest.mark.parametrize("server_default", [False, True])
@pytest.mark.parametrize("decision", [None, False, True])
@pytest.mark.parametrize("context", [{}, {"model": "chosen"}, {"system_prompt": "custom"}])
def test_explicit_decision_overrides_default_without_content_side_effects(
    monkeypatch, fake_openai_module, known_tops, server_default, decision, context,
):
    monkeypatch.setattr(agenda_detection, "AGENDA_DETECTION_USE_LLM", server_default)
    fake_openai_module.content = '{"tops":[{"top_id":"unspecified:unnumbered","reason":"Haushalt aufgerufen","top_title":"Haushalt","start_index":0,"end_index":0,"evidence_index":0,"evidence_text":"TOP 1 Haushalt."}]}'
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
    assert fake_openai_module.instances[0].kwargs["timeout"].read == 999
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
    if known_tops:
        assert result.assignments == [None]
        assert result.llm.status == "failed"
        assert result.llm.gaps[0]["kind"] == "technical"
        assert "SECRET" not in str(result.llm)
        return
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
        # A model confidence alone does not supply the missing evidence quote.
        assert result.segments[0].uncertain


def test_unknown_agenda_number_only_revisit_keeps_original_topic():
    result = detect_agenda_from_transcript([
        TranscriptUtterance('MOD', text) for text in [
            'Kommen wir zu TOP 1 Haushalt.', 'Beratung.',
            'Kommen wir zu TOP 2 Schulbau.', 'Beratung.',
            'Kommen wir wieder zu TOP 1.', 'Beratung.',
        ]
    ], use_llm=False)
    assert result.tops == ['TOP 1 Haushalt', 'TOP 2 Schulbau']
    assert result.assignments == [0, 0, 1, 1, 0, 0]


@pytest.mark.parametrize('title', ['TOP 1', '1.'])
def test_known_number_only_title_is_a_valid_identity(fake_openai_module, title):
    result = known_llm_result(fake_openai_module, [title], ['Kommen wir zu TOP 1.'], [{
        'top_id': 'agenda:0', 'top_title': title, 'start_index': 0, 'end_index': 0,
        'confidence': 0.9, 'evidence_index': 0, 'evidence_text': 'Kommen wir zu TOP 1.',
    }])
    assert result.assignments == [0]
    assert not result.segments[0].uncertain
    assert not result.llm.validation_reasons


@pytest.mark.parametrize('interruption', [
    'Wir unterbrechen die Beratung.', 'Kommen wir zu TOP 99.',
    'Kommen wir zum unbekannten Thema.', 'Kommen wir zu TOP 1 und TOP 2.',
])
def test_interior_unresolved_call_or_stop_prevents_safe_llm_range(fake_openai_module, interruption):
    result = known_llm_result(fake_openai_module, ['1 Haushalt', '2 Schulbau'], [
        'Kommen wir zu TOP 1 Haushalt.', interruption, 'Weitere Wortmeldung.',
    ], [{'top_id': 'agenda:0', 'start_index': 0, 'end_index': 2, 'confidence': 0.95,
         'evidence_index': 0, 'evidence_text': 'Kommen wir zu TOP 1 Haushalt.'}])
    assert result.segments[0].uncertain
    assert result.segments[0].confidence <= 0.5


@pytest.mark.parametrize('bounds', [
    {'start_index': -1, 'end_index': 1}, {'start_index': 0, 'end_index': 100},
    {'start_index': 0}, {'start_index': True, 'end_index': 1},
])
def test_unknown_agenda_does_not_invent_or_clip_invalid_ranges(fake_openai_module, bounds):
    fake_openai_module.content = json.dumps({'tops': [{'top_title': 'Haushalt', **bounds, 'confidence': 0.99}]})
    result = detect_agenda_from_transcript([TranscriptUtterance('A', 'Diskussion.')] * 2, use_llm=True)
    assert result.assignments == [None, None]
    assert 'invalid_bounds' in result.llm.validation_reasons


def test_unknown_agenda_clears_hallucinated_evidence(fake_openai_module):
    fake_openai_module.content = json.dumps({'tops': [{
        'top_title': 'Haushalt', 'start_index': 0, 'end_index': 0,
        'confidence': 0.99, 'evidence_index': 0, 'evidence_text': 'Erfundener Aufruf.',
    }]})
    result = detect_agenda_from_transcript([TranscriptUtterance('A', 'Diskussion.')], use_llm=True)
    assert result.segments[0].uncertain
    assert result.segments[0].evidence_index is None
    assert result.segments[0].evidence_text is None
    assert result.llm.warnings


def test_chunk_evidence_keeps_global_index_and_bounds_stay_local(fake_openai_module, monkeypatch):
    monkeypatch.setattr(agenda_detection, 'AGENDA_DETECTION_CHUNK_LINES', 2)
    monkeypatch.setattr(agenda_detection, 'AGENDA_DETECTION_CHUNK_OVERLAP_LINES', 0)
    fake_openai_module.responses = [
        json.dumps({'tops': [{'top_title': 'Falscher Bereich', 'start_index': 0, 'end_index': 3}]}),
        json.dumps({'tops': [{'top_title': '2 Schulbau', 'start_index': 0, 'end_index': 1,
                             'confidence': 0.9, 'evidence_index': 0,
                             'evidence_text': 'Kommen wir zu TOP 2 Schulbau.'}]}),
    ]
    result = detect_agenda_from_transcript([TranscriptUtterance('M', text) for text in [
        'Diskussion.', 'Weitere Diskussion.', 'Kommen wir zu TOP 2 Schulbau.', 'Beratung.',
    ]], use_llm=True)
    assert result.assignments == [None, None, 0, 0]
    assert result.segments[0].evidence_index == 2
    assert result.segments[0].evidence_text == 'Kommen wir zu TOP 2 Schulbau.'
    assert not result.segments[0].uncertain
    assert 'invalid_bounds' in result.llm.validation_reasons


@pytest.mark.parametrize('failure', [None, TimeoutError('offline'), 'not json'])
def test_known_fallback_preserves_gaps_order_and_resumption(fake_openai_module, failure):
    if failure is not None:
        fake_openai_module.responses = [failure]
    tops = ['1 Begrüßung', '2 Haushalt', '3 Schulbau', '4 Anfragen']
    transcript = [TranscriptUtterance('MOD', text) for text in [
        'Ich eröffne die Sitzung.',
        'Den Haushalt behandeln wir später unter TOP 2.',
        'Kommen wir zu TOP 3 Schulbau.',
        'Kommen wir zu TOP 2 Haushalt.',
        'Wortmeldung.',
        'Kommen wir zu TOP 99.',
        'Unbekanntes Thema.',
        'Kommen wir wieder zu TOP 3 Schulbau.',
    ]]
    result = segment_known_agenda(transcript, tops, use_llm=failure is not None)
    assert result.tops == tops
    if failure is not None:
        assert result.assignments == [None] * len(transcript)
        assert result.llm.status == 'failed'
        return
    assert result.assignments == [None, None, 2, 1, 1, None, None, 2]
    assert [s.top_index for s in result.segments] == [2, 1, 2]
    assert result.segments[1].evidence_index == 3


def test_unknown_agenda_fallback_rejects_previews_and_reuses_exact_labels():
    result = detect_agenda_from_transcript([
        TranscriptUtterance('MOD', text) for text in [
            'Später kommen wir zu TOP 2 Haushalt.',
            'Kommen wir zu TOP 3 Schulbau.',
            'Kommen wir zu TOP 2 Haushalt.',
            'Kommen wir wieder zu TOP 3 Schulbau.',
        ]
    ], use_llm=False)
    assert result.tops == ['TOP 3 Schulbau', 'TOP 2 Haushalt']
    assert result.assignments == [None, 0, 1, 0]
    assert [s.top_index for s in result.segments] == [0, 1, 0]


def test_llm_known_labels_are_resolved_by_identity_not_position(fake_openai_module):
    fake_openai_module.content = '''{"tops":[
        {"top_title":"Schulbau","start_index":1,"end_index":1,"confidence":0.9},
        {"top_title":"Haushalt","start_index":2,"end_index":2,"confidence":0.9},
        {"top_title":"Schulbau","start_index":4,"end_index":4,"confidence":0.9}
    ]}'''
    transcript = [TranscriptUtterance('MOD', text) for text in [
        'Vorgespräch.', 'Kommen wir zu TOP 3 Schulbau.', 'Kommen wir zu TOP 2 Haushalt.',
        'Pause.', 'Kommen wir wieder zu TOP 3 Schulbau.',
    ]]
    result = segment_known_agenda(transcript, ['1 Begrüßung', '2 Haushalt', '3 Schulbau'], use_llm=True)
    assert result.assignments == [None] * 5
    assert result.llm.status == 'failed'


def test_llm_cannot_promote_preview_to_safe_boundary(fake_openai_module):
    fake_openai_module.content = '''{"tops":[
        {"top_title":"Haushalt","start_index":0,"end_index":1,"confidence":0.99}
    ]}'''
    result = segment_known_agenda([
        TranscriptUtterance('MOD', 'Den Haushalt behandeln wir später unter TOP 2.'),
        TranscriptUtterance('MOD', 'Anwesenheitsliste.'),
    ], ['1 Begrüßung', '2 Haushalt'], use_llm=True)
    assert result.segments == []
    assert result.llm.status == 'failed'


def test_llm_missing_known_boundary_is_not_interpolated(fake_openai_module):
    fake_openai_module.content = '''{"tops":[
        {"top_title":"Haushalt","confidence":0.99}
    ]}'''
    result = segment_known_agenda([TranscriptUtterance('A', 'Allgemeine Diskussion.')], ['Haushalt'], use_llm=True)
    assert result.assignments == [None]
    assert result.segments == []


def test_known_llm_prompt_allows_omissions_and_revisits(fake_openai_module):
    segment_known_agenda([TranscriptUtterance('A', 'Diskussion.')], ['Haushalt'], use_llm=True)
    user_prompt = fake_openai_module.instances[0].calls[0]['messages'][1]['content']
    assert 'genau einen Eintrag' not in user_prompt
    from agenda_llm import PROMPT
    assert 'beliebig oft' in PROMPT
    assert 'target_start' in user_prompt


@pytest.mark.parametrize('raw_segments', [
    [{'top_title': 'TOP 99 Haushalt', 'start_index': 0, 'confidence': 0.99}],
    [
        {'top_title': 'Haushalt', 'start_index': 0, 'confidence': 0.99},
        {'top_title': 'Schulbau', 'start_index': 0, 'confidence': 0.99},
    ],
])
def test_known_llm_conflicting_identity_does_not_select_arbitrary_topic(fake_openai_module, raw_segments):
    import json
    fake_openai_module.content = json.dumps({'tops': raw_segments})
    result = segment_known_agenda([TranscriptUtterance('A', 'Diskussion.')], ['2 Haushalt', '3 Schulbau'], use_llm=True)
    assert result.assignments == [None]
    assert result.segments == []


def known_llm_result(fake, tops, lines, segments):
    # Unit tests of the legacy validator, still used for historical segment
    # contracts. Full-coverage inference has its own tests in test_agenda_llm.py.
    transcript = [TranscriptUtterance('MOD', line) for line in lines]
    usage = agenda_detection._llm_usage(True)
    raw = agenda_detection._parse_llm_segments(json.dumps({'tops': segments}))
    from assignment_suggestions import suggest_assignments
    validated, _ = agenda_detection._validate_known_segments(
        transcript, tops, raw, list(suggest_assignments(transcript, tops).segments),
        issues=usage.validation_reasons)
    validated = agenda_detection._guard_number_evidence(transcript, tops, validated)
    return agenda_detection._result_from_segments(len(lines), validated, 'legacy_validator', tops=tops, usage=usage)


@pytest.mark.parametrize('with_ids', [False, True])
@pytest.mark.parametrize('reverse', [False, True])
def test_omitted_household_never_receives_school_boundaries(fake_openai_module, with_ids, reverse):
    segments = [
        {'top_title': 'Begrüßung', 'start_index': 0, 'end_index': 1},
        {'top_title': 'Schulbau', 'start_index': 4, 'end_index': 5},
    ]
    if with_ids:
        segments[0]['top_id'] = 'agenda:0'
        segments[1]['top_id'] = 'agenda:2'
    if reverse:
        segments.reverse()
    result = known_llm_result(fake_openai_module, ['Begrüßung', 'Haushalt', 'Schulbau'],
                              ['Allgemeine Diskussion.'] * 6, segments)
    assert result.assignments == [0, 0, None, None, 2, 2]
    assert [s.top_title for s in result.segments] == ['Begrüßung', 'Schulbau']
    assert all(s.uncertain for s in result.segments)


@pytest.mark.parametrize('identity', [
    {'top_id': 'agenda:99', 'top_title': 'Haushalt'},
    {'top_id': 'agenda:0', 'top_title': 'Schulbau'},
    {'top_id': None, 'top_title': 'Haushalt'},
    {'top_id': 0, 'top_title': 'Haushalt'},
    {'top_title': 'TOP 99 Haushalt'},
    {'top_title': '[Nichtöffentlich] Haushalt'},
    {'top_title': 'Haushaltsberatung'},
])
def test_invalid_identity_cannot_be_rescued_by_position_or_title(fake_openai_module, identity):
    result = known_llm_result(fake_openai_module, ['Haushalt', 'Schulbau'], ['Diskussion.'],
                              [{**identity, 'start_index': 0, 'end_index': 0}])
    assert result.assignments == [None]
    assert 'invalid_identity' in result.llm.validation_reasons


@pytest.mark.parametrize('tops', [
    ['Haushalt', 'Haushalt'], ['Haushalt', '2 Haushalt'],
    ['2.1 Haushalt', '2.2 Haushalt'],
    ['[Öffentlich] 2.1 Haushalt', '[Nichtöffentlich] 2.1 Haushalt'],
])
def test_legacy_title_must_not_hide_ambiguity(fake_openai_module, tops):
    result = known_llm_result(fake_openai_module, tops, ['Diskussion.'],
                              [{'top_title': 'Haushalt', 'start_index': 0, 'end_index': 0}])
    assert result.assignments == [None]
    assert 'invalid_identity' in result.llm.validation_reasons


def test_explicit_id_resolves_duplicate_titles_but_does_not_prove_content(fake_openai_module):
    result = known_llm_result(fake_openai_module, ['Haushalt', 'Haushalt'], ['Kommen wir zum Haushalt.'],
                              [{'top_id': 'agenda:1', 'top_title': 'Haushalt', 'start_index': 0,
                                'end_index': 0, 'confidence': 0.99, 'evidence_index': 0,
                                'evidence_text': 'Kommen wir zum Haushalt.'}])
    assert result.assignments == [1]
    assert result.segments[0].uncertain
    assert result.segments[0].confidence <= 0.5


@pytest.mark.parametrize('field', ['start_index', 'end_index'])
@pytest.mark.parametrize('value', [None, True, False, 0.5, 1.0, '0', -1, 99, float('inf')])
def test_invalid_bounds_are_rejected_without_clamping(fake_openai_module, field, value):
    raw = {'top_id': 'agenda:0', 'top_title': 'Haushalt', 'start_index': 0, 'end_index': 1}
    raw[field] = value
    result = known_llm_result(fake_openai_module, ['Haushalt'], ['Diskussion.'] * 2, [raw])
    assert result.assignments == [None, None]
    assert 'invalid_bounds' in result.llm.validation_reasons


@pytest.mark.parametrize('bounds', [{'start_index': 1, 'end_index': 0}, {'start_index': 0}, {'end_index': 1}])
def test_missing_or_reversed_bounds_are_not_interpolated(fake_openai_module, bounds):
    result = known_llm_result(fake_openai_module, ['Haushalt'], ['Diskussion.'] * 2,
                              [{'top_id': 'agenda:0', **bounds}])
    assert result.assignments == [None, None]
    assert 'invalid_bounds' in result.llm.validation_reasons


@pytest.mark.parametrize('right', [(0, 2), (1, 1), (2, 3)])
@pytest.mark.parametrize('same_top', [False, True])
def test_all_overlapping_proposals_are_rejected(fake_openai_module, right, same_top):
    result = known_llm_result(fake_openai_module, ['Haushalt', 'Schulbau'], ['Diskussion.'] * 4, [
        {'top_id': 'agenda:0', 'start_index': 0, 'end_index': 2},
        {'top_id': 'agenda:0' if same_top else 'agenda:1', 'start_index': right[0], 'end_index': right[1]},
    ])
    assert result.assignments == [None] * 4
    assert 'overlapping_segments' in result.llm.validation_reasons


def test_reordered_ids_and_disjoint_revisits_keep_gaps(fake_openai_module):
    result = known_llm_result(fake_openai_module, ['2.1 Haushalt', '3 Schulbau'], ['Diskussion.'] * 6, [
        {'top_id': 'agenda:1', 'start_index': 5, 'end_index': 5},
        {'top_id': 'agenda:0', 'start_index': 2, 'end_index': 3},
        {'top_id': 'agenda:1', 'start_index': 0, 'end_index': 0},
        {'top_id': 'agenda:99', 'start_index': 1, 'end_index': 4},
    ])
    assert result.assignments == [1, None, 0, 0, None, 1]


@pytest.mark.parametrize('evidence', [
    {}, {'evidence_text': 'Erfundener Beleg', 'evidence_index': 0},
    {'evidence_text': 'Kommen wir zum Haushalt.', 'evidence_index': 2},
    {'evidence_text': 'Kommen wir zum Haushalt.', 'evidence_index': True},
    {'evidence_text': 'Kommen wir zum Haushalt.', 'evidence_index': '0'},
    {'evidence_text': 'Andere Zeile.', 'evidence_index': 0},
    {'evidence_text': 'Andere Zeile.'},
])
def test_unverified_evidence_is_cleared_and_never_certain(fake_openai_module, evidence):
    result = known_llm_result(fake_openai_module, ['Haushalt'],
                              ['Kommen wir zum Haushalt.', 'Andere Zeile.'],
                              [{'top_id': 'agenda:0', 'start_index': 0, 'end_index': 0,
                                'confidence': 0.99, **evidence}])
    segment = result.segments[0]
    assert segment.evidence_text is None
    assert segment.evidence_index is None
    assert segment.uncertain and segment.confidence <= 0.5
    assert 'unverified_evidence' in result.llm.validation_reasons
    assert result.llm.warnings


@pytest.mark.parametrize('indexed', [False, True])
def test_grounded_unambiguous_call_can_be_accepted(fake_openai_module, indexed):
    raw = {'top_id': 'agenda:1', 'top_title': '[Öffentlich] 02.10 Schulbau',
           'start_index': 0, 'end_index': 0, 'confidence': 0.95,
           'evidence_text': 'Kommen wir zu TOP 2.10 Schulbau im öffentlichen Teil.'}
    if indexed:
        raw['evidence_index'] = 0
    result = known_llm_result(fake_openai_module,
                              ['1 Haushalt', '[Öffentlich] 02.10 Schulbau'], [raw['evidence_text']], [raw])
    assert result.assignments == [1]
    assert not result.segments[0].uncertain
    assert result.segments[0].evidence_index == 0
    assert result.segments[0].top_title == '[Öffentlich] 02.10 Schulbau'
    assert result.llm.validation_reasons == []


def test_quote_must_not_hide_preview_context(fake_openai_module):
    result = known_llm_result(fake_openai_module, ['2 Haushalt'],
                              ['Später kommen wir zu TOP 2 Haushalt.'],
                              [{'top_id': 'agenda:0', 'start_index': 0, 'end_index': 0,
                                'confidence': 0.99, 'evidence_index': 0,
                                'evidence_text': 'kommen wir zu TOP 2 Haushalt.'}])
    assert result.segments[0].uncertain
    assert result.segments[0].confidence <= 0.5


def test_conflicting_current_call_rejects_wrong_id_and_uses_independent_fallback(fake_openai_module):
    result = known_llm_result(fake_openai_module, ['2 Haushalt', '3 Schulbau'],
                              ['Kommen wir zu TOP 3 Schulbau.'],
                              [{'top_id': 'agenda:0', 'start_index': 0, 'end_index': 0,
                                'confidence': 0.99, 'evidence_index': 0,
                                'evidence_text': 'Kommen wir zu TOP 3 Schulbau.'}])
    assert result.assignments == [1]
    assert result.segments[0].uncertain
    assert 'contradictory_evidence' in result.llm.validation_reasons
    assert 'heuristic_supplement' in result.llm.validation_reasons


def test_missing_top_is_only_supplemented_with_independent_range(fake_openai_module):
    result = known_llm_result(fake_openai_module, ['Begrüßung', 'Haushalt', 'Schulbau'], [
        'Kommen wir zur Begrüßung.', 'Danke.', 'Kommen wir zum Haushalt.', 'Beratung.',
        'Kommen wir zum Schulbau.', 'Beratung.',
    ], [
        {'top_id': 'agenda:0', 'start_index': 0, 'end_index': 1},
        {'top_id': 'agenda:2', 'start_index': 4, 'end_index': 5},
    ])
    assert result.assignments == [0, 0, 1, 1, 2, 2]
    replacement = result.segments[1]
    assert replacement.evidence_index == 2
    assert replacement.uncertain and replacement.confidence <= 0.5
    assert 'heuristische Ergänzung' in replacement.reason


def test_prompt_uses_request_ids_independent_of_original_numbering(fake_openai_module):
    segment_known_agenda([TranscriptUtterance('A', 'Diskussion.')],
                         ['[Öffentlich] 02.10 Schule', '[Nichtöffentlich] 02.10 Schule'], use_llm=True)
    request = fake_openai_module.instances[0].calls[0]
    prompt = request['messages'][1]['content']
    assert '"top_id": "public:02.10", "title": "Schule"' in prompt
    assert '"top_id": "nonpublic:02.10", "title": "Schule"' in prompt
    assert 'Zeilennummer' in request['messages'][0]['content']


def test_fallback_does_not_clip_ranges_to_fill_gaps(fake_openai_module):
    result = known_llm_result(fake_openai_module, ['Haushalt', 'Schulbau'], [
        'Kommen wir zum Haushalt.', 'Diskussion.', 'Diskussion.',
    ], [{'top_id': 'agenda:1', 'start_index': 2, 'end_index': 2}])
    # Heuristic Haushalt spans 0–2 and conflicts with an accepted LLM proposal.
    assert result.assignments == [None, None, 1]
    assert 'heuristic_supplement' not in result.llm.validation_reasons


def test_interior_topic_change_prevents_certain_range(fake_openai_module):
    result = known_llm_result(fake_openai_module, ['2 Haushalt', '3 Schulbau'], [
        'Kommen wir zu TOP 2 Haushalt.', 'Kommen wir zu TOP 3 Schulbau.',
    ], [{'top_id': 'agenda:0', 'start_index': 0, 'end_index': 1, 'confidence': 0.99,
         'evidence_index': 0, 'evidence_text': 'Kommen wir zu TOP 2 Haushalt.'}])
    assert result.segments[0].uncertain
    assert result.segments[0].confidence <= 0.5
    assert 'weak_boundary_evidence' in result.llm.validation_reasons


@pytest.mark.parametrize('confidence', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_confidence_never_becomes_certain(fake_openai_module, confidence):
    result = known_llm_result(fake_openai_module, ['Haushalt'], ['Kommen wir zum Haushalt.'], [
        {'top_id': 'agenda:0', 'start_index': 0, 'end_index': 0, 'confidence': confidence,
         'evidence_index': 0, 'evidence_text': 'Kommen wir zum Haushalt.'},
    ])
    assert result.segments[0].uncertain
    assert result.segments[0].confidence == 0.55


def test_legacy_evidence_requires_unique_line_within_range(fake_openai_module):
    result = known_llm_result(fake_openai_module, ['Haushalt'], ['Kommen wir zum Haushalt.'] * 2, [
        {'top_id': 'agenda:0', 'start_index': 0, 'end_index': 1, 'confidence': 0.99,
         'evidence_text': 'Kommen wir zum Haushalt.'},
    ])
    assert result.segments[0].evidence_index is None
    assert result.segments[0].evidence_text is None
    assert result.segments[0].uncertain
