"""Request-local styles, immutable jobs and persisted editing preferences."""
import asyncio
from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
import threading

import pytest
from fastapi.testclient import TestClient

import durable_jobs as durable
import gemma_summary as gemma
import llm_config
import main
import persistence
import summarize
from test_gemma_summary import config, setup_model, generate, DRAFT, ORIGINAL
from test_gemma_integration import session_state, review_lines


@pytest.mark.parametrize('mode', ['fast', 'slow'])
@pytest.mark.parametrize('style,scale', [('gemma4-lora', 1), ('gemma4-custom', 0)])
def test_style_controls_only_generation(monkeypatch, mode, style, scale):
    calls = setup_model(monkeypatch, mode)
    prompt = 'STIL: Kurze Absätze und eine Beschlusstabelle.'
    result = generate(mode, summary_style=style, custom_summary_prompt=prompt)
    assert result.summary == result.structured.protocol_text == DRAFT
    assert len(calls) == (1 if mode == 'fast' else 3)
    settings, request = calls[0]
    assert settings.model == 'gemma-4-31b' and settings.lora_scale == scale
    assert 'response_format' not in request
    assert result.llm_usage['summary_style'] == style
    assert result.llm_usage['configuration']['custom_summary_prompt'] == prompt
    if scale:
        assert request['messages'] == gemma.protocol_messages('2. Kosten', ORIGINAL)
    else:
        assert request['messages'][0]['content'] == gemma.CUSTOM_SYSTEM
        assert prompt in request['messages'][1]['content']
    for settings, request in calls[1:]:
        assert settings.model == 'gemma-4-31b' and settings.lora_scale == 0
        assert prompt not in json.dumps(request['messages'])
        body = json.loads(request['messages'][1]['content'])
        assert body['draft'] == DRAFT and body['original_transcript'] == '\n'.join(ORIGINAL)
    with pytest.raises(FrozenInstanceError):
        calls[0][0].custom_summary_prompt = 'Changed'


def test_default_prompt_matches_editable_ui_default():
    from pathlib import Path
    import re
    source = (Path(__file__).resolve().parents[2] / 'frontend/src/components/LLMSettingsPanel.tsx').read_text(encoding='utf-8')
    assert re.search(r'export const DEFAULT_CUSTOM_SUMMARY_PROMPT = `([^`]+)`;', source)[1] == llm_config.DEFAULT_CUSTOM_SUMMARY_PROMPT


def test_cache_and_checkpoints_separate_styles_prompts_and_adapters(monkeypatch):
    calls = setup_model(monkeypatch, 'fast')
    cache, steps = {}, {}
    monkeypatch.setattr(gemma, 'cache_read', cache.get)
    monkeypatch.setattr(gemma, 'cache_write', lambda key, value: cache.update({key: value}))
    def checkpoint(key, operation):
        if key not in steps:
            steps[key] = operation()
        return steps[key]
    monkeypatch.setattr(durable, 'checkpoint', checkpoint)
    for style, prompt in [('gemma4-lora', 'A'), ('gemma4-custom', 'A'), ('gemma4-custom', 'B')]:
        generate('fast', summary_style=style, custom_summary_prompt=prompt)
    assert len(calls) == len(steps) == len(cache) == 3
    generate('fast', summary_style='gemma4-custom', custom_summary_prompt='A')
    assert len(calls) == 3
    from dataclasses import replace
    monkeypatch.setattr(summarize, 'get_llm_config', lambda model=None: replace(config('fast'), lora_id=1, adapter_revision='new'))
    generate('fast', summary_style='gemma4-custom', custom_summary_prompt='A')
    assert len(calls) == len(steps) == len(cache) == 4


@pytest.mark.parametrize('mode', ['fast', 'slow'])
def test_style_save_reopen_preserves_historical_review(monkeypatch, mode):
    setup_model(monkeypatch, mode)
    state = session_state(generate(mode))
    state.update(summary_style='gemma4-custom', custom_summary_prompt='Meine Tabelle')
    with TestClient(main.app) as client:
        first = client.post('/api/sessions', json=state).json()
        historical = deepcopy(first['summary_reviews']['0'])
        for style in ('gemma4-lora', 'gemma4-custom'):
            first['summary_style'] = style
            response = client.put('/api/sessions/gemma-session', json=first)
            assert response.status_code == 200, response.text
            first = response.json()
            reopened = client.get('/api/sessions/gemma-session').json()
            assert reopened['summary_style'] == style
            assert reopened['custom_summary_prompt'] == 'Meine Tabelle'
            assert reopened['summaries']['0'] == DRAFT
            assert reopened['summary_reviews']['0'] == historical
            assert reopened['summary_reviews']['0']['structured']['verification']['summary_style'] == 'gemma4-lora'
        legacy = deepcopy(first)
        del legacy['summary_style'], legacy['custom_summary_prompt']
        response = client.put('/api/sessions/gemma-session', json=legacy)
        assert response.json()['summary_style'] == 'gemma4-custom'
        assert response.json()['custom_summary_prompt'] == 'Meine Tabelle'


def test_summary_job_retains_submission_settings_after_session_change(monkeypatch):
    calls = setup_model(monkeypatch, 'fast')
    state = session_state(generate('fast'))
    session = persistence.save_session('gemma-session', main.reconcile_session_summaries(None, state))
    async def manager():
        return main.DurableSubmission('summary')
    monkeypatch.setattr(main, 'get_or_create_summary_job_manager', manager)
    job = asyncio.run(main.create_summary_job('gemma-session', main.SummaryJobCreateRequest(
        revision=session['revision'], top_ids=['costs'], summary_style='gemma4-custom', custom_summary_prompt='Originalstil')))
    submitted = durable.load(job.summary_job_id)
    latest = persistence.load_session('gemma-session')
    latest.update(summary_style='gemma4-lora', custom_summary_prompt='Spätere Vorgabe')
    persistence.save_session('gemma-session', main.reconcile_session_summaries(latest, latest))
    assert durable.load(job.summary_job_id)['payload'] == submitted['payload']
    current = durable.claim('style-worker', 60)
    token = durable.CURRENT.set(durable.Runtime(current, 'style-worker', threading.Event()))
    try:
        main.run_summary_job(job.summary_job_id)
    finally:
        durable.CURRENT.reset(token)
    assert calls[-1][0].summary_style == 'gemma4-custom'
    assert calls[-1][0].lora_scale == 0
    assert 'Originalstil' in calls[-1][1]['messages'][1]['content']
    saved = persistence.load_session('gemma-session')
    assert saved['summary_style'] == 'gemma4-lora'
    assert saved['custom_summary_prompt'] == 'Spätere Vorgabe'
    assert saved['summary_reviews'][0]['llm_usage']['summary_style'] == 'gemma4-custom'


def test_pipeline_progress_separates_styles_and_prompts(monkeypatch):
    calls = setup_model(monkeypatch, 'fast')
    state = {'result_refs': {}}
    monkeypatch.setattr(main, 'load_pipeline_job', lambda _: deepcopy(state))
    monkeypatch.setattr(main, 'save_pipeline_state', lambda _, result_refs: state['result_refs'].update(result_refs))
    for style, prompt in [('gemma4-lora', 'A'), ('gemma4-custom', 'A'), ('gemma4-custom', 'B'), ('gemma4-custom', 'B')]:
        summaries, reviews = main.summarize_pipeline_segments('pipeline', transcript=review_lines(), tops=['Kosten'],
            assignments=[0, 0], options=dict(processing_mode='fast', summary_style=style, custom_summary_prompt=prompt))
        assert summaries[0] == DRAFT
        assert reviews[0]['llm_usage']['summary_style'] == style
    assert len(calls) == 3


@pytest.mark.parametrize('mode', ['fast', 'slow'])
def test_pipeline_api_captures_and_reopens_custom_style(monkeypatch, tmp_path, mode):
    from conftest import FakeTranscriptionResult
    from test_main import configure_test_app, wait_until
    configure_test_app(tmp_path, monkeypatch)
    calls = setup_model(monkeypatch, mode)
    monkeypatch.setattr(main, 'transcribe_audio', lambda *a, **k: FakeTranscriptionResult(review_lines(), 2))
    monkeypatch.setattr('llm_transport.model_fingerprint', lambda _: {'digest': 'fixed'})
    with TestClient(main.app) as client:
        response = client.post('/api/pipeline/start', data=dict(processing_mode=mode, skip_agenda_detection='true',
            summary_style='gemma4-custom', custom_summary_prompt='PIPELINE_STIL'),
            files={'audio': ('test.mp3', b'fake', 'audio/mpeg')})
        assert response.status_code == 200, response.text
        job_id = response.json()['pipeline_id']
        assert wait_until(lambda: client.get(f'/api/pipeline/{job_id}').json()['status'] in {'completed', 'failed'})
        result = client.get(f'/api/pipeline/{job_id}/result')
        assert result.status_code == 200, result.text
        session = result.json()['session']
        assert session['summaries']['0'] == DRAFT
        assert session['summary_style'] == 'gemma4-custom'
        assert session['custom_summary_prompt'] == 'PIPELINE_STIL'
        assert len(calls) == (1 if mode == 'fast' else 3) and calls[0][0].lora_scale == 0
        assert session['summary_reviews']['0']['llm_usage']['review_complete'] is (mode == 'slow')
        assert 'PIPELINE_STIL' in calls[0][1]['messages'][1]['content']


def test_versions_include_effective_summary_style(monkeypatch):
    monkeypatch.setattr(llm_config, 'get_llm_config', lambda *a, **k: config('fast'))
    def snapshot(style, prompt):
        return durable.version_snapshot({'legacy_snapshot': {'refs': dict(summary_style=style, custom_summary_prompt=prompt)}})['summary']
    assert snapshot('gemma4-lora', 'A')['lora_scale'] == 1
    assert snapshot('gemma4-custom', 'A')['lora_scale'] == 0
    assert snapshot('gemma4-custom', 'A')['config_id'] != snapshot('gemma4-custom', 'B')['config_id']


def test_other_profiles_ignore_gemma_preferences():
    base = llm_config.LLMConfig(base_url='http://example.test/v1', model='other', api_key='test')
    assert base.with_summary_style('gemma4-custom', 'My style') is base
    assert config().with_summary_style().summary_style == 'gemma4-lora'
