"""Reasoning configuration must reach every task without breaking legacy APIs."""

import json

import pytest

import agenda_detection
import extract_tops
from pdf_fixtures import agenda
import summarize
from assignment_suggestions import TranscriptUtterance


SUMMARY = json.dumps({
    "discussion": ["Der Haushalt wurde beraten."],
    "decisions": [], "votes": [], "action_items": [],
    "open_points": [], "uncertainties": [],
})


@pytest.mark.parametrize("effort", [None, "", "  ", "none", "low", "medium", "high", "max", " HIGH "])
@pytest.mark.parametrize("task", ["summary", "pdf_tops", "pdf_metadata", "known_agenda", "unknown_agenda"])
def test_reasoning_reaches_all_task_requests(monkeypatch, fake_openai_module, effort, task):
    if effort is None:
        monkeypatch.delenv("LLM_REASONING_EFFORT", raising=False)
    else:
        monkeypatch.setenv("LLM_REASONING_EFFORT", effort)
    normalized = (effort or "").strip().lower()
    if task == "summary":
        fake_openai_module.content = SUMMARY
        summarize.summarize_segment("Haushalt", "MOD: Der Haushalt wurde beraten.")
    elif task.startswith("pdf"):
        fake_openai_module.content = json.dumps(agenda())
        extract = extract_tops.extract_agenda_data_from_text if task == "pdf_metadata" else extract_tops.extract_tops_from_text
        extract("Einladung: 1. Haushalt", system_prompt="/no_think\nFachlicher Kontext")
    else:
        fake_openai_module.content = json.dumps({"tops": [{
            "top_id": "unspecified:1", "top_title": "1. Haushalt",
            "start_index": 0, "end_index": 0, "confidence": 0.9,
            "evidence_index": 0, "evidence_text": "Ich rufe TOP 1 Haushalt auf.", "reason": "Aufruf Haushalt",
        }]})
        transcript = [TranscriptUtterance("MOD", "Ich rufe TOP 1 Haushalt auf.")]
        if task == "known_agenda":
            result = agenda_detection.segment_known_agenda(transcript, ["1. Haushalt"], use_llm=True)
        else:
            result = agenda_detection.detect_agenda_from_transcript(transcript, use_llm=True)
        assert result.llm.attempted_calls == 1
        assert result.llm.failed_calls == 0

    request = fake_openai_module.instances[0].calls[0]
    if normalized:
        assert request["reasoning_effort"] == normalized
    else:
        assert "reasoning_effort" not in request
    if task.startswith("pdf"):
        prompt = request["messages"][0]["content"]
        assert "/no_think" not in prompt
        assert "ohne Denken" not in prompt
        assert "Fachlicher Kontext" in prompt


@pytest.mark.parametrize("effort", ["false", "true", "off", "auto", "minimal", "hihg"])
def test_invalid_reasoning_setting_is_rejected(monkeypatch, effort):
    monkeypatch.setenv("LLM_REASONING_EFFORT", effort)
    with pytest.raises(ValueError, match="LLM_REASONING_EFFORT"):
        summarize.get_llm_config()


def test_reasoning_applies_to_summary_chunks_reduce_and_fallback(monkeypatch, fake_openai_module):
    monkeypatch.setenv("LLM_REASONING_EFFORT", "none")
    monkeypatch.setattr(summarize, "LLM_CHUNK_CHARS", 80)
    fake_openai_module.responses = [
        SUMMARY, SUMMARY, "invalid JSON",
        "Teilzusammenfassung A.", "Teilzusammenfassung B.", "Zusammenfassung als Freitext.",
    ]
    transcript = "\n".join([
        "A: " + "Haushaltsansatz und Begründung. " * 2,
        "B: " + "Nachfrage zu Kosten und Fristen. " * 2,
    ])
    result = summarize.summarize_segment("Haushalt", transcript)
    calls = fake_openai_module.instances[0].calls
    assert not result.fallback_used
    assert len(calls) == 2
    assert all(call["reasoning_effort"] == "none" for call in calls)


@pytest.mark.parametrize("extract", [extract_tops.extract_tops_from_text, extract_tops.extract_agenda_data_from_text])
def test_pdf_calls_honor_timeout_and_retry_configuration(monkeypatch, fake_openai_module, extract):
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "1800")
    monkeypatch.setattr(extract_tops, "LLM_MAX_RETRIES", 0)
    fake_openai_module.content = json.dumps(agenda())
    extract("1. Haushalt")
    client_options = fake_openai_module.instances[0].kwargs
    assert client_options["timeout"].read == 1800
    assert client_options["max_retries"] == 0


def test_zero_summary_retries_has_no_hidden_sdk_retries(monkeypatch, fake_openai_module):
    monkeypatch.setattr(summarize, "LLM_MAX_RETRIES", 0)
    fake_openai_module.responses = [TimeoutError("timed out")]
    with pytest.raises(summarize.LLMCallError):
        summarize.summarize_segment("Haushalt", "MOD: Der Haushalt wurde beraten.")
    client = fake_openai_module.instances[0]
    assert client.kwargs["max_retries"] == 0
    assert len(client.calls) == 1


@pytest.mark.parametrize("effort", ["none", "high", "max"])
def test_real_sdk_serializes_reasoning_without_network(monkeypatch, effort):
    import httpx
    import openai

    real_client = openai.OpenAI
    requests = []

    def respond(request):
        if request.method == "GET" and request.url.path == "/v1/models":
            return httpx.Response(200, json={
                "object": "list", "data": [{
                    "id": "qwen3.5:9b", "object": "model", "created": 0, "owned_by": "test",
                }],
            })
        assert request.method == "POST"
        assert request.url.path == "/v1/chat/completions"
        requests.append(json.loads(request.content))
        return httpx.Response(200, text='data: ' + json.dumps({
            'model': 'qwen3.5:9b', 'choices': [{'index': 0, 'finish_reason': 'stop',
                'delta': {'content': SUMMARY, 'reasoning': 'This is not the final answer.'}}]
        }) + '\n\ndata: [DONE]\n\n')

    monkeypatch.setenv("LLM_REASONING_EFFORT", effort)
    monkeypatch.setenv("LLM_MODEL", "qwen3.5:9b")
    monkeypatch.setenv("LLM_API_KEY", "test-only")
    monkeypatch.setenv("LLM_BASE_URL", "http://llm.example.test/v1")
    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kwargs: real_async_client(transport=httpx.MockTransport(respond), **kwargs))
    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: real_client(http_client=http_client, **kwargs))
        result = summarize.summarize_segment("Haushalt", "MOD: Der Haushalt wurde beraten.")
    assert not result.fallback_used
    assert "This is not the final answer" not in result.summary
    assert len(requests) == 1
    assert requests[0]["reasoning_effort"] == effort
