import json
from types import SimpleNamespace

import httpx
import pytest

import llm_transport as transport
from summarize import get_llm_config


def test_budget_counts_utf8_system_user_and_output(monkeypatch):
    monkeypatch.setenv('LLM_CONTEXT_TOKENS', '4096')
    messages = [{'role': 'system', 'content': 'ä' * 1000}, {'role': 'user', 'content': 'x' * 1000}]
    assert transport.input_bound(messages) == 3576
    assert transport.fits(messages, 500)
    assert not transport.fits(messages, 600)


@pytest.mark.parametrize('effort, expected', [('', None), ('none', False), ('low', 'low'), ('max', 'max')])
def test_native_ollama_receives_options_and_schema(monkeypatch, effort, expected):
    monkeypatch.setenv('LLM_OLLAMA_NATIVE', 'true')
    monkeypatch.setenv('LLM_CONTEXT_TOKENS', '16384')
    monkeypatch.setenv('LLM_CPU_THREADS', '16')
    monkeypatch.setenv('LLM_REASONING_EFFORT', effort)
    requests = []
    def post(url, **kwargs):
        requests.append((url, kwargs))
        return httpx.Response(200, request=httpx.Request('POST', url), json={
            'done': True, 'done_reason': 'stop', 'prompt_eval_count': 40, 'message': {'content': '{}'}})
    monkeypatch.setattr(httpx, 'post', post)
    monkeypatch.setattr(transport, '_verify_native_context', lambda *args: 16384)
    config = get_llm_config()
    transport.complete(None, config, model=config.model, messages=[{'role': 'user', 'content': 'test'}],
                       max_tokens=100, response_format={'type': 'json_schema', 'json_schema': {'schema': {'type': 'object'}}},
                       **config.reasoning_options)
    url, options = requests[0]
    assert url.endswith('/api/chat')
    assert options['json']['options']['num_ctx'] == 16384
    assert options['json']['options']['num_predict'] == 100
    assert options['json']['options']['num_thread'] == 16
    assert options['json']['truncate'] is False
    assert options['json']['shift'] is False
    assert options['json']['format'] == {'type': 'object'}
    if expected is None:
        assert 'think' not in options['json']
    else:
        assert options['json']['think'] == expected


def test_budget_failure_does_not_send_request(fake_openai_module, monkeypatch):
    monkeypatch.setenv('LLM_CONTEXT_TOKENS', '4096')
    client = fake_openai_module()
    with pytest.raises(transport.ContextBudgetError):
        transport.complete(client, get_llm_config(), model='test',
                           messages=[{'role': 'user', 'content': 'x' * 4096}], max_tokens=1)
    assert client.calls == []


@pytest.mark.parametrize('reason,done,count', [('length', True, 10), ('stop', False, 10), ('stop', True, 30000)])
def test_incomplete_native_response_is_not_success(monkeypatch, reason, done, count):
    monkeypatch.setenv('LLM_OLLAMA_NATIVE', 'true')
    def post(url, **kwargs):
        return httpx.Response(200, request=httpx.Request('POST', url), json={
            'done': done, 'done_reason': reason, 'prompt_eval_count': count, 'message': {'content': '{}'}})
    monkeypatch.setattr(httpx, 'post', post)
    monkeypatch.setattr(transport, '_verify_native_context', lambda *args: 16384)
    with pytest.raises(transport.ContextBudgetError):
        transport.complete(None, get_llm_config(), model='test',
                           messages=[{'role': 'user', 'content': 'test'}], max_tokens=100)


def test_cache_is_private_and_bound_to_input_model_prompt(monkeypatch, tmp_path):
    monkeypatch.setenv('LLM_CACHE_DIR', str(tmp_path))
    config = get_llm_config()
    key = transport.cache_key(config, [{'role': 'user', 'content': 'a'}], 'test')
    transport.cache_write(key, {'complete': True})
    assert transport.cache_read(key) == {'complete': True}
    assert transport.cache_read(key + 'changed') is None
    assert next(tmp_path.iterdir()).stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize('actual', [None, 4096, 16384])
def test_native_context_is_verified_against_running_model(monkeypatch, actual):
    monkeypatch.setenv('LLM_CONTEXT_TOKENS', '16384')
    def get(url, **kwargs):
        return httpx.Response(200, request=httpx.Request('GET', url), json={
            'models': [{'model': 'qwen3.5:9b', 'context_length': actual}]})
    monkeypatch.setattr(httpx, 'get', get)
    if actual == 16384:
        assert transport._verify_native_context(get_llm_config(), 'qwen3.5:9b') == actual
    else:
        with pytest.raises(transport.ContextBudgetError):
            transport._verify_native_context(get_llm_config(), 'qwen3.5:9b')


def test_thinking_structured_output_reserves_both_server_phases(monkeypatch):
    monkeypatch.setenv('LLM_OLLAMA_NATIVE', 'true')
    monkeypatch.setenv('LLM_CONTEXT_TOKENS', '8192')
    monkeypatch.setattr(httpx, 'post', lambda *a, **kw: pytest.fail('Over-budget request sent'))
    with pytest.raises(transport.ContextBudgetError, match='all provider generation phases'):
        transport.complete(None, get_llm_config(), model='qwen3.5:9b',
            messages=[{'role': 'user', 'content': 'a'*5000}], max_tokens=2048,
            reasoning_effort='none', ollama_think=True, response_format={'type': 'json_object'})


def test_review_thinking_override_does_not_change_openai_reasoning_field(fake_openai_module, monkeypatch):
    monkeypatch.setenv('LLM_OLLAMA_NATIVE', 'false')
    client = fake_openai_module()
    transport.complete(client, get_llm_config(), model='test',
        messages=[{'role': 'user', 'content': 'test'}], max_tokens=100,
        reasoning_effort='none', ollama_think=True)
    assert client.calls[0]['reasoning_effort'] == 'none'
    assert 'ollama_think' not in client.calls[0]
