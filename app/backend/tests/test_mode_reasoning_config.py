"""Mode defaults must agree at submission, execution, caching and transport."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import pytest

from llm_config import get_llm_config, configured, ModelConfigurationError
from llm_transport import _think, structured_output_budget
from processing_mode import processing_scope
import durable_jobs


@pytest.mark.parametrize('model', ['gemma4:12b','qwen3.5:9b'])
@pytest.mark.parametrize('mode,expected', [('fast',False),('slow',True)])
@pytest.mark.parametrize('value', [None,'','  ','none','low','medium','high','max'])
def test_mode_reasoning_and_native_boolean_switch(monkeypatch, model, mode, expected, value):
    key=f'LLM_{mode.upper()}_REASONING_EFFORT'
    if value is None: monkeypatch.delenv(key, raising=False)
    else: monkeypatch.setenv(key,value)
    monkeypatch.setenv('LLM_MODEL',model)
    monkeypatch.setenv('LLM_PROVIDER','ollama')
    monkeypatch.setenv('LLM_THINKING','true' if mode=='fast' else 'false')
    monkeypatch.setenv('LLM_REASONING_EFFORT','high' if mode=='fast' else 'none')
    monkeypatch.setenv('LLM_THINKING_TOKENS','2048')
    effective=expected if not (value or '').strip() else value!='none'
    with processing_scope(mode):
        config=get_llm_config()
    assert _think(config) is effective
    assert config.processing_mode==mode
    assert config.thinking_tokens==(2048 if effective else 0)
    assert structured_output_budget(config,4096)==(12288 if effective else 4096)


def test_job_snapshot_resolves_payload_mode_not_calling_thread(monkeypatch):
    monkeypatch.setenv('LLM_FAST_REASONING_EFFORT','none')
    monkeypatch.setenv('LLM_SLOW_REASONING_EFFORT','high')
    with processing_scope('slow'):
        fast=durable_jobs.version_snapshot({'request':{'processing_mode':'fast'}})
    with processing_scope('fast'):
        slow=durable_jobs.version_snapshot({'processing_mode':'slow'})
    assert fast['model']['reasoning_effort']=='none'
    assert slow['model']['reasoning_effort']=='high'
    assert fast['model']['config_id']!=slow['model']['config_id']
    monkeypatch.setenv('LLM_SLOW_REASONING_EFFORT','low')
    assert fast==durable_jobs.version_snapshot({'request':{'processing_mode':'fast'}})
    assert slow!=durable_jobs.version_snapshot({'processing_mode':'slow'})


def test_config_freezes_and_isolates_modes_across_threads(monkeypatch):
    monkeypatch.setenv('LLM_FAST_REASONING_EFFORT','')
    monkeypatch.setenv('LLM_SLOW_REASONING_EFFORT','')
    barrier=Barrier(2)
    @configured
    def operation(processing_mode):
        barrier.wait()
        return get_llm_config().reasoning_effort
    with ThreadPoolExecutor(2) as pool:
        assert list(pool.map(operation,['fast','slow']))==['none','medium']


@pytest.mark.parametrize('mode',['fast','slow'])
def test_bad_mode_effort_rejected_before_inference(monkeypatch, mode):
    monkeypatch.setenv(f'LLM_{mode.upper()}_REASONING_EFFORT','hihg')
    with pytest.raises(ModelConfigurationError,match=f'LLM_{mode.upper()}_REASONING_EFFORT'):
        get_llm_config(processing_mode=mode)
