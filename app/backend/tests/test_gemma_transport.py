from dataclasses import replace
from types import SimpleNamespace

import pytest
import llm_transport as transport
from llm_config import LLMConfig, ModelConfigurationError, get_llm_config


def config(**kw):
    return LLMConfig(base_url='http://llama:8080/v1', model='gemma-4-31b', api_key='local',
                     provider='llama-cpp', summary_style='gemma4-lora', reasoning_effort='none', **kw)


def test_protocol_configuration_is_isolated():
    base = config()
    prose = base.for_protocol()
    assert base.lora_scale == 0
    assert prose.lora_scale == 1
    assert (prose.temperature, prose.top_p, prose.top_k, prose.thinking_tokens) == (0.3, 0.9, 0, 0)
    assert prose.public_snapshot()['config_id'] != base.public_snapshot()['config_id']


@pytest.mark.parametrize('adapter', [False, True])
def test_request_local_adapter_and_thinking(monkeypatch, adapter):
    cfg = config()
    if adapter:
        cfg = cfg.for_protocol()
    payloads = []
    async def stream(client, settings, payload):
        payloads.append(payload)
        yield {'choices': [{'delta': {'content': 'Text'}, 'finish_reason': 'stop'}]}
    async def metadata(*args, **kwargs):
        if args[2].endswith('/lora-adapters'):
            return [{'id': 0, 'path': '/models/gemma-4-31b-protokoll-f16.gguf'}]
        return {'default_generation_settings': {'n_ctx': 131072}}
    monkeypatch.setattr(transport, '_openai_stream', stream)
    monkeypatch.setattr(transport, '_json', metadata)
    result = transport.complete(None, cfg, messages=[{'role': 'user', 'content': 'Quelle'}], max_tokens=128)
    request = payloads[0]
    assert request['lora'] == [{'id': 0, 'scale': float(adapter)}]
    assert request['chat_template_kwargs'] == {'enable_thinking': False}
    assert 'reasoning_effort' not in request
    assert request['repeat_penalty'] == 1.0
    assert '<turn|>' in request['stop']
    assert result.llm_provenance['verified_context_tokens'] == 131072


def test_adapter_rejects_json():
    with pytest.raises(ModelConfigurationError, match='structured output'):
        transport.complete(None, config().for_protocol(), messages=[{'role': 'user', 'content': 'x'}],
                           max_tokens=10, response_format={'type': 'json_object'})


def test_context_contract_checked(monkeypatch):
    async def metadata(*args, **kwargs):
        return {'default_generation_settings': {'n_ctx': 4096}}
    monkeypatch.setattr(transport, '_json', metadata)
    with pytest.raises(ModelConfigurationError, match='context'):
        transport.complete(None, config(), messages=[{'role': 'user', 'content': 'x'}], max_tokens=10)


def test_llama_handover_waits_for_sleep(monkeypatch):
    import gpu_resources as gpu
    states = iter([False, True])
    calls = []
    def get(url, **kwargs):
        calls.append(url)
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {'is_sleeping': next(states)})
    monkeypatch.setattr('httpx.get', get)
    monkeypatch.setattr(gpu.time, 'sleep', lambda _: None)
    gpu.unload_local_ollama(config())
    assert calls == ['http://llama:8080/props'] * 2


def test_llama_invalid_handover_refuses_whisper(monkeypatch):
    import gpu_resources as gpu
    monkeypatch.setattr('httpx.get', lambda *a, **k: SimpleNamespace(
        raise_for_status=lambda: None, json=lambda: {}))
    with pytest.raises(gpu.GPUResourceError, match='Speicherstatus'):
        gpu.unload_local_ollama(config())
