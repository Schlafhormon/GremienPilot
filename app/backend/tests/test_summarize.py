import summarize
import pytest

@pytest.mark.parametrize('content', [
    '{"discussion":["Erster Beitrag"],"discussion":["Zweiter Beitrag"]}',
    '{"discussion":["Erster Beitrag"],"Diskussion":["Zweiter Beitrag"]}',
])
def test_summary_rejects_fields_that_would_silently_overwrite_content(content):
    with pytest.raises(summarize.StructuredOutputError):
        summarize.parse_structured_summary(content)

def test_resolve_llm_base_url_uses_local_default_outside_docker(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.setattr(__import__("llm_config"), "is_docker_runtime", lambda: False)

    base_url, source = summarize.resolve_llm_base_url()

    assert base_url == "http://localhost:11434/v1"
    assert source == "local_development_default"

def test_resolve_llm_base_url_uses_internal_default_in_docker(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setattr(__import__("llm_config"), "is_docker_runtime", lambda: True)

    base_url, source = summarize.resolve_llm_base_url()

    assert base_url == "http://ollama:11434/v1"
    assert source == "internal_docker_default_from_local_value"

def test_resolve_llm_base_url_keeps_explicit_external_url(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example.test/v1")
    monkeypatch.setattr(__import__("llm_config"), "is_docker_runtime", lambda: True)

    base_url, source = summarize.resolve_llm_base_url()

    assert base_url == "https://llm.example.test/v1"
    assert source == "external_configured"

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

@pytest.mark.parametrize('status,category,transient', [(429, 'rate_limit', True), (503, 'server', True), (400, 'client', False)])
def test_native_http_status_has_same_retry_classification(status, category, transient):
    import httpx
    response = httpx.Response(status, request=httpx.Request('POST', 'http://example.test/api/chat'))
    error = httpx.HTTPStatusError('test', request=response.request, response=response)
    result = summarize.classify_llm_error(error)
    assert result.category == category
    assert result.transient is transient
