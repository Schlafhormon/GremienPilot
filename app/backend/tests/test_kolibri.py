"""Kolibri wire format, memory handover and text-only PDF regressions."""
from dataclasses import replace
import json

import httpx
import pytest

from llm_config import get_llm_config, ModelConfigurationError, resolve_llm_base_url
import llm_transport as transport
import gpu_resources as gpu
import extract_tops as pdf
from pdf_fixtures import agenda, audit, pdf_bytes

_REAL_STREAM = transport._openai_stream


@pytest.fixture
def kolibri(monkeypatch):
    monkeypatch.setenv('LLM_MODEL', 'Aleph-Alpha/Kolibri-1')
    monkeypatch.setenv('LLM_PROVIDER', 'llama-cpp')
    monkeypatch.setenv('LLM_BASE_URL', 'http://kolibri:8080/v1')
    monkeypatch.delenv('LLM_CONTEXT_TOKENS', raising=False)
    return get_llm_config()


@pytest.mark.parametrize('mode,effort,reserve', [('fast', 'none', 0), ('slow', 'medium', 4096)])
def test_kolibri_defaults_and_model_overrides(kolibri, mode, effort, reserve):
    config = get_llm_config(processing_mode=mode)
    assert config.context_tokens == 32768
    assert (config.temperature, config.top_p, config.top_k) == (1.0, 0.97, 128)
    assert (config.reasoning_effort, config.thinking_tokens) == (effort, reserve)
    assert config.image_tokens == 0 and config.uses_local_llama_cpp
    assert not config.uses_ollama
    assert not get_llm_config('custom-model').tokenizer_path
    with pytest.raises(ModelConfigurationError, match='none, low, medium and high'):
        replace(config, reasoning_effort='max')
    with pytest.raises(ModelConfigurationError, match='text-only'):
        replace(config, image_tokens=100)


@pytest.mark.parametrize('mode', ['fast', 'slow'])
def test_kolibri_stream_keeps_reasoning_out_of_json(monkeypatch, kolibri, mode):
    calls = []
    config = get_llm_config(processing_mode=mode)
    def handler(request):
        if request.url.path == '/props':
            return httpx.Response(200, json={'default_generation_settings': {'n_ctx': 32768}})
        calls.append(json.loads(request.content))
        return httpx.Response(200, text=''.join('data: ' + json.dumps(event) + '\n\n' for event in [
            {'choices': [{'delta': {'reasoning_content': 'private thoughts'}}]},
            {'choices': [{'delta': {'content': '{"ok":true}'}, 'finish_reason': 'stop'}]},
            {'choices': [], 'usage': {'total_tokens': 42}},
        ]))
    cls = httpx.AsyncClient
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kw: cls(transport=httpx.MockTransport(handler), **kw))
    # The suite's network guard is intentionally bypassed only with MockTransport.
    monkeypatch.setattr(transport, '_openai_stream', _REAL_STREAM)
    result = transport.complete(None, config, messages=[{'role': 'user', 'content': 'Prüfe.'}],
        max_tokens=256, response_format={'type': 'json_object'})
    assert result.choices[0].message.content == '{"ok":true}'
    assert result.llm_provenance['verified_context_tokens'] == 32768
    body = calls[0]
    assert body['chat_template_kwargs'] == {'reasoning_effort': 'none' if mode == 'fast' else 'medium',
                                          'enable_thinking': mode == 'slow'}
    assert 'reasoning_effort' not in body
    assert body['top_k'] == 128 and body['max_tokens'] == 256 + config.thinking_tokens


def test_llama_context_mismatch_fails_before_completion(monkeypatch, kolibri):
    cls = httpx.AsyncClient
    def handler(request):
        assert request.url.path == '/props'
        return httpx.Response(200, json={'default_generation_settings': {'n_ctx': 4096}})
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kw: cls(transport=httpx.MockTransport(handler), **kw))
    with pytest.raises(transport.ContextBudgetError, match='llama.cpp context'):
        transport.complete(None, kolibri, messages=[{'role': 'user', 'content': 'Prüfe.'}], max_tokens=256)


def test_llama_handover_requires_confirmed_sleep(monkeypatch, kolibri):
    states = iter([False, False, True])
    calls = []
    def get(url, **kwargs):
        calls.append(url)
        return httpx.Response(200, request=httpx.Request('GET', url), json={'is_sleeping': next(states)})
    monkeypatch.setattr(httpx, 'get', get)
    monkeypatch.setattr(gpu.time, 'sleep', lambda _: None)
    gpu.unload_local_ollama(kolibri)
    assert calls == ['http://kolibri:8080/props'] * 3
    monkeypatch.setattr(httpx, 'get', lambda url, **kw: httpx.Response(200,
        request=httpx.Request('GET', url), json={'is_sleeping': 'true'}))
    with pytest.raises(gpu.GPUResourceError, match='Speicherstatus'):
        gpu.unload_local_ollama(kolibri)


def test_kolibri_pdf_reviews_text_and_ocr_without_images(tmp_path, monkeypatch, kolibri):
    path = tmp_path / 'invitation.pdf'
    path.write_bytes(pdf_bytes())
    monkeypatch.setattr(pdf, '_ocr_image', lambda image: 'TOP 1 Haushalt')
    calls = []
    answers = iter([agenda(), agenda(), audit(1), audit(0)])
    def request(config, prompt, content, schema):
        calls.append(content)
        assert 'keine Bilder' in pdf._source_prompt(prompt, config)
        return json.dumps(next(answers))
    monkeypatch.setattr(pdf, '_request', request)
    result = pdf.extract_agenda_data_from_pdf(path)
    assert result.processing_complete and result.contract_version == pdf.TEXT_PDF_CONTRACT
    assert result.document['source_mode'] == 'text+ocr'
    assert all(part['type'] == 'text' for content in calls for part in content)
    assert any('Lokale OCR' in part['text'] for content in calls for part in content)


def test_ocr_failure_is_actionable(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError('tesseract')
    monkeypatch.setattr(pdf.subprocess, 'run', missing)
    with pytest.raises(pdf.ExtractionError, match='Tesseract'):
        pdf._ocr_image('')
