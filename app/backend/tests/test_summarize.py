import summarize
import pytest


def test_actual_frontend_prompt_preserves_summary_contract(fake_openai_module, frontend_summary_prompt):
    fake_openai_module.content = structured_response()
    result = summarize.summarize_segment(
        "Haushalt", "MOD: Der Haushalt wird beraten.",
        model="test-model", system_prompt=frontend_summary_prompt,
    )
    prompt = fake_openai_module.instances[0].calls[0]["messages"][0]["content"]
    assert summarize.DEFAULT_SYSTEM_PROMPT in prompt
    assert frontend_summary_prompt in prompt
    assert "das JSON-Ausgabeformat hat Vorrang" in prompt
    assert not result.fallback_used


def structured_response(**overrides):
    payload = {
        "discussion": ["Die Vorsitzende erlaeuterte den Sachverhalt."],
        "decisions": [],
        "votes": [],
        "action_items": [],
        "open_points": [],
        "uncertainties": [],
    }
    payload.update(overrides)
    import json

    return json.dumps(payload)


def test_resolve_llm_base_url_uses_local_default_outside_docker(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.setattr(summarize, "is_docker_runtime", lambda: False)

    base_url, source = summarize.resolve_llm_base_url()

    assert base_url == "http://localhost:11434/v1"
    assert source == "local_development_default"


def test_resolve_llm_base_url_uses_internal_default_in_docker(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setattr(summarize, "is_docker_runtime", lambda: True)

    base_url, source = summarize.resolve_llm_base_url()

    assert base_url == "http://ollama:11434/v1"
    assert source == "internal_docker_default_from_local_value"


def test_resolve_llm_base_url_keeps_explicit_external_url(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example.test/v1")
    monkeypatch.setattr(summarize, "is_docker_runtime", lambda: True)

    base_url, source = summarize.resolve_llm_base_url()

    assert base_url == "https://llm.example.test/v1"
    assert source == "external_configured"


def test_summarize_segment_uses_structured_output_and_returns_duration(
    fake_openai_module,
    monkeypatch,
):
    fake_openai_module.responses = [
        structured_response(
            decisions=["Der Ausschuss empfahl die Annahme der Vorlage."],
            votes=["Die Empfehlung erfolgte einstimmig."],
        )
    ]
    times = iter([100.0, 102.5])
    monkeypatch.setattr(summarize.time, "time", lambda: next(times))

    result = summarize.summarize_segment(
        "Haushalt",
        "SPEAKER_00: Wir beraten den Haushalt.",
        model="test-model",
        system_prompt="Formal als JSON zusammenfassen",
    )

    assert "Diskussion:" in result.summary
    assert "Die Vorsitzende erlaeuterte den Sachverhalt." in result.summary
    assert "Beschluss:" in result.summary
    assert "Abstimmung:" in result.summary
    assert result.duration_seconds == 2.5
    assert result.structured is not None
    assert result.structured.decisions == [
        "Der Ausschuss empfahl die Annahme der Vorlage."
    ]
    assert result.fallback_used is False
    assert result.chunks_processed == 1

    client = fake_openai_module.instances[0]
    assert client.kwargs["base_url"]
    assert client.kwargs["timeout"] == summarize.LLM_TIMEOUT_SECONDS
    request = client.calls[0]
    assert request["model"] == "test-model"
    assert request["temperature"] == 0.2
    assert request["max_tokens"] == 1400
    assert request["messages"][0]["role"] == "system"
    assert "Gib ausschließlich valides JSON" in request["messages"][0]["content"]
    assert "Formal als JSON zusammenfassen" in request["messages"][0]["content"]
    assert "TOP: Haushalt" in request["messages"][1]["content"]
    assert "SPEAKER_00: Wir beraten den Haushalt." in request["messages"][1]["content"]


def test_summarize_segment_uses_map_reduce_for_long_transcripts(
    fake_openai_module,
    monkeypatch,
):
    monkeypatch.setattr(summarize, "LLM_CHUNK_CHARS", 80)
    fake_openai_module.responses = [
        structured_response(discussion=["Teil 1 wurde beraten."]),
        structured_response(discussion=["Teil 2 wurde beraten."]),
        structured_response(
            discussion=["Die Beratung wurde zusammengefuehrt."],
            open_points=["Die Verwaltung liefert Zahlen nach."],
        ),
    ]

    transcript = "\n".join(
        [
            "SPEAKER_00: " + ("Haushaltsansatz und Begruendung. " * 2),
            "SPEAKER_01: " + ("Nachfrage zu Kosten und Fristen. " * 2),
        ]
    )

    result = summarize.summarize_segment("Haushalt", transcript, model="test-model")

    assert result.chunks_processed == 2
    assert "Die Beratung wurde zusammengefuehrt." in result.summary
    assert "Die Verwaltung liefert Zahlen nach." in result.summary

    calls = fake_openai_module.instances[0].calls
    assert len(calls) == 3
    assert "Teil 1 von 2" in calls[0]["messages"][1]["content"]
    assert "Teil 2 von 2" in calls[1]["messages"][1]["content"]
    assert "Teilnotizen" in calls[2]["messages"][1]["content"]


def test_summarize_segment_falls_back_to_freetext_on_malformed_structured_response(
    fake_openai_module,
):
    fake_openai_module.responses = [
        "Das ist kein JSON.",
        "Die Vorlage wurde beraten. Ein Beschluss wurde nicht gefasst.",
    ]

    result = summarize.summarize_segment(
        "Baugebiet",
        "SPEAKER_00: Die Vorlage wird beraten.",
        model="test-model",
    )

    assert result.summary == "Die Vorlage wurde beraten. Ein Beschluss wurde nicht gefasst."
    assert result.structured is None
    assert result.fallback_used is True
    assert result.chunks_processed == 1

    calls = fake_openai_module.instances[0].calls
    assert len(calls) == 2
    assert calls[1]["temperature"] == 0.3
    assert "Zusammenfassung:" in calls[1]["messages"][1]["content"]


def test_summarize_segment_retries_transient_llm_errors(
    fake_openai_module,
    monkeypatch,
):
    class ServerError(Exception):
        status_code = 503

    monkeypatch.setattr(summarize, "LLM_RETRY_BACKOFF_SECONDS", 0)
    fake_openai_module.responses = [
        ServerError("Service unavailable"),
        structured_response(discussion=["Die Beratung wurde fortgesetzt."]),
    ]

    result = summarize.summarize_segment(
        "Gebuehren",
        "SPEAKER_00: Wir beraten die Gebuehren.",
        model="test-model",
    )

    assert result.fallback_used is False
    assert "Die Beratung wurde fortgesetzt." in result.summary
    assert len(fake_openai_module.instances[0].calls) == 2


def test_summarize_segment_checks_model_before_chat(fake_openai_module):
    fake_openai_module.models_response = ["other-model"]

    with pytest.raises(summarize.LLMCallError) as error:
        summarize.summarize_segment(
            "Haushalt",
            "SPEAKER_00: Wir beraten den Haushalt.",
            model="missing-model",
        )

    assert error.value.category == "model_missing"
    assert "missing-model" in str(error.value)
    assert "LLM_MODEL=missing-model" in str(error.value)
    assert fake_openai_module.instances[0].calls == []


def test_summarize_segment_reports_unreachable_ollama_before_chat(
    fake_openai_module,
):
    fake_openai_module.models_response = ConnectionError("Connection error")

    with pytest.raises(summarize.LLMCallError) as error:
        summarize.summarize_segment(
            "Haushalt",
            "SPEAKER_00: Wir beraten den Haushalt.",
            model="test-model",
        )

    assert error.value.category == "network"
    assert "LLM_BASE_URL=" in str(error.value)
    assert "LLM_MODEL=test-model" in str(error.value)
    assert fake_openai_module.instances[0].calls == []


def test_summary_review_links_structured_items_to_transcript_lines():
    structured = summarize.StructuredSummary(
        discussion=["Die Vorlage zum Haushalt wurde beraten."],
        decisions=["Der Ausschuss beschloss die Vorlage zum Haushalt."],
        votes=["Die Abstimmung erfolgte einstimmig."],
    )
    lines = [
        {
            "speaker": "SPEAKER_00",
            "text": "Wir beraten die Vorlage zum Haushalt.",
            "start": 5,
            "end": 9,
        },
        {
            "speaker": "SPEAKER_01",
            "text": "Der Ausschuss beschloss die Vorlage einstimmig.",
            "start": 10,
            "end": 14,
        },
    ]

    review = summarize.build_summary_review(
        structured=structured,
        summary=summarize.render_structured_summary(structured),
        lines=lines,
    )

    decision_link = next(
        link
        for link in review.source_links
        if link.section == "decisions" and link.item_index == 0
    )
    vote_link = next(
        link
        for link in review.source_links
        if link.section == "votes" and link.item_index == 0
    )

    assert decision_link.missing_source is False
    assert 1 in decision_link.line_indices
    assert decision_link.start == 5
    assert "beschloss" in decision_link.excerpt
    assert vote_link.missing_source is False
    assert 1 in vote_link.line_indices


def test_summary_review_warns_when_decision_keyword_is_missing_from_summary():
    lines = [
        {
            "speaker": "SPEAKER_00",
            "text": "Der Antrag wurde abgelehnt.",
            "start": 20,
            "end": 24,
        }
    ]

    review = summarize.build_summary_review(
        structured=summarize.StructuredSummary(
            discussion=["Der Antrag wurde diskutiert."]
        ),
        summary="Der Antrag wurde diskutiert.",
        lines=lines,
    )

    warning = next(
        warning
        for warning in review.warnings
        if warning.kind == "missing_decision_signal"
    )
    assert warning.keyword == "abgelehnt"
    assert warning.line_indices == [0]
    assert warning.start == 20
    assert "abgelehnt" in warning.message
