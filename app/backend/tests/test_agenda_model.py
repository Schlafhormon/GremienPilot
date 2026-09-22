import json
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

import agenda_model as models
import agenda_llm
import agenda_timeline
import gpu_resources
import llm_transport
from agenda_detection import segment_known_agenda, AgendaLLMUsage
from assignment_suggestions import TranscriptUtterance
from summarize import get_llm_config


@pytest.fixture
def isolated_settings(monkeypatch, tmp_path):
    monkeypatch.setenv('AGENDA_MODEL_SETTINGS_PATH', str(tmp_path / 'settings.json'))
    monkeypatch.setenv('LLM_MODEL', 'qwen3.5:9b')
    monkeypatch.setattr(models, '_last_error', None)
    return tmp_path


def enable(monkeypatch, **options):
    monkeypatch.setenv('LLM_OLLAMA_NATIVE', 'true')
    settings = models.AgendaModelSettings(enabled=True, **options)
    models.save_settings(settings)
    monkeypatch.setattr(models, 'require_model', lambda c: {
        'model': c.model, 'digest': 'gemma-digest', 'provider': 'ollama'})
    monkeypatch.setattr(gpu_resources, 'unload_ollama_model', lambda c: None)
    return models.resolve_config()


def response(data):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(data)))])


def test_disabled_preserves_legacy_model_and_every_summary_parameter(isolated_settings, monkeypatch):
    before = get_llm_config()
    assert models.resolve_config() == before
    assert models.resolve_config('legacy-request-model').model == 'legacy-request-model'
    enable(monkeypatch, context_tokens=65536, output_tokens=6144, timeout_seconds=1234, cpu_threads=3)
    config = models.resolve_config('legacy-request-model')
    assert config.model == 'gemma4:31b-it-q4_K_M'
    assert (config.context_budget, config.output_budget, config.timeout_seconds, config.cpu_threads) == (65536, 6144, 1234, 3)
    assert get_llm_config() == before
    models.save_settings(models.AgendaModelSettings())
    assert models.resolve_config() == before


def test_settings_persist_and_api_does_not_change_sessions(isolated_settings, monkeypatch):
    import main
    monkeypatch.setattr(models, 'require_model', lambda c: {'model': c.model, 'digest': 'd'})
    monkeypatch.setenv('LLM_OLLAMA_NATIVE', 'true')
    client = TestClient(main.app)
    setting = models.AgendaModelSettings(enabled=True, context_tokens=24576)
    result = client.put('/api/settings/agenda-model', json=setting.model_dump())
    assert result.status_code == 200
    assert result.json()['settings'] == setting.model_dump()
    # A new reader has no dependency on process-local settings or browser storage.
    assert models.AgendaModelSettings.model_validate_json(models.settings_path().read_text()) == setting
    assert client.get('/api/settings/agenda-model').json()['effective_model'] == setting.model
    assert get_llm_config().model == 'qwen3.5:9b'
    for invalid in ({'context_tokens': 8192, 'output_tokens': 8192}, {'model': 'gemma4:31b-cloud'}, {'enabled': 'true'}):
        assert client.put('/api/settings/agenda-model', json={**setting.model_dump(), **invalid}).status_code == 422
    assert models.load_settings() == setting


def test_missing_model_has_technical_gaps_and_no_provider_fallback(isolated_settings, monkeypatch, fake_openai_module):
    monkeypatch.setenv('LLM_OLLAMA_NATIVE', 'true')
    models.save_settings(models.AgendaModelSettings(enabled=True))
    monkeypatch.setattr(llm_transport, 'model_fingerprint', lambda c: {'model': c.model, 'digest': None})
    result = segment_known_agenda([TranscriptUtterance('A', 'Kommen wir zu TOP 1.')], ['1 Haushalt'], use_llm=True)
    assert result.assignments == [None]
    assert result.llm.status == 'failed'
    assert result.llm.gaps[0]['kind'] == 'technical'
    assert result.llm.attempted_calls == 0
    assert 'nicht lokal installiert' in ' '.join(result.llm.warnings)
    assert not fake_openai_module.instances


def test_unload_after_failure_but_not_on_cache_only_pass(isolated_settings, monkeypatch):
    config = enable(monkeypatch)
    released = []
    monkeypatch.setattr(gpu_resources, 'unload_ollama_model', lambda c: released.append(c.model))
    usage = AgendaLLMUsage(True, 'test')
    with pytest.raises(RuntimeError):
        with models.model_session(config, usage):
            usage.attempted_calls += 1
            raise RuntimeError('load failed')
    assert released == [config.model]
    with models.model_session(config, usage):
        pass
    with models.model_session(get_llm_config(), usage):
        usage.attempted_calls += 1
    assert released == [config.model]


def test_timeline_wrong_hypothesis_cannot_veto_original_call_and_cache_is_separate(isolated_settings, monkeypatch, fake_openai_module):
    enable(monkeypatch)
    monkeypatch.setenv('LLM_CACHE_DIR', str(isolated_settings / 'cache'))
    monkeypatch.setenv('AGENDA_DETECTION_GAP_REVIEW_MAX_CALLS', '0')
    calls = []
    text = 'Wir rufen jetzt TOP 2 auf.'
    def generate(client, config, **kw):
        calls.append((config, kw))
        body = json.loads(kw['messages'][1]['content'])
        if 'phase' in body:
            return response({'events': [{'index': 0, 'quote': text, 'kind': 'call', 'section': None,
                'top_id': 'unspecified:1', 'uncertain': False, 'reason': 'Falsche globale Hypothese', 'contradictions': []}]})
        assert body['topic_timeline']['source'] == 'unverified_model_hypotheses'
        assert 'unspecified:2' in kw['response_format']['json_schema']['schema']['properties']['assignments']['properties']['0']['enum']
        return response({'assignments': {'0': 'unspecified:2'}, 'uncertain_lines': [], 'gaps': []})
    monkeypatch.setattr(agenda_timeline, 'complete', generate)
    monkeypatch.setattr(agenda_llm, 'complete', generate)
    transcript = [TranscriptUtterance('M', text)]
    one = segment_known_agenda(transcript, ['1 Haushalt', '2 Schule'], use_llm=True)
    assert one.assignments == [1]
    assert len(calls) == 3 and all(c.model.startswith('gemma4:') for c, _ in calls)
    assert one.llm.chunks[-1]['phase'] == 'boundary_review'
    two = segment_known_agenda(transcript, ['1 Haushalt', '2 Schule'], use_llm=True)
    assert two.assignments == one.assignments and two.llm.attempted_calls == 0
    assert two.llm.provenance['timeline']['identity'] == one.llm.provenance['timeline']['identity']
    assert all(c['status'] == 'cached' for c in two.llm.chunks)
    segment_known_agenda(transcript, ['1 Haushalt', '2 Schule'], use_llm=True, cache_namespace='fresh')
    assert len(calls) == 6


def test_long_timeline_covers_original_text_and_reviews_every_seam(isolated_settings, monkeypatch, fake_openai_module):
    config = enable(monkeypatch, context_tokens=12288, output_tokens=1024, timeline_output_tokens=1024)
    transcript = [TranscriptUtterance('S', f'Beitrag {i}. ' + 'Schulberatung. '*30) for i in range(70)]
    seen = []
    def generate(client, config, **kw):
        body = json.loads(kw['messages'][1]['content'])
        seen.append(body)
        assert llm_transport.fits(kw['messages'], kw['max_tokens'] + 768, config)
        return response({'events': []})
    monkeypatch.setattr(agenda_timeline, 'complete', generate)
    usage = AgendaLLMUsage(True, 'test')
    timeline = agenda_timeline.analyze(None, config, transcript, [], usage, {'digest': 'gemma'})
    windows = [b for b in seen if b['phase'] == 'timeline']
    assert len(windows) > 1
    assert set(range(70)) == {i for b in windows for i in range(b['target_start'], b['target_end']+1)}
    assert all(a['target_end'] >= b['target_start'] for a, b in zip(windows, windows[1:]))
    assert len(timeline['seams']) == len(windows)-1
    seam_rows = {i for b in seen if b['phase'] == 'timeline_seam'
                 for i in range(b['target_start'], b['target_end']+1)}
    assert all(set(range(a, b+1)) <= seam_rows for a, b in timeline['seams'])


def test_oversized_single_original_line_is_explicit_failure_not_truncated(isolated_settings, monkeypatch):
    config = enable(monkeypatch, context_tokens=12288, output_tokens=1024, timeline_output_tokens=1024)
    monkeypatch.setattr(agenda_timeline, 'complete', lambda *a, **k: pytest.fail('Oversized input sent'))
    with pytest.raises(llm_transport.ContextBudgetError):
        agenda_timeline.analyze(None, config, [TranscriptUtterance('M', 'x'*20000)], [], AgendaLLMUsage(True, 'test'), {})


def test_native_parameters_and_summary_are_isolated(isolated_settings, monkeypatch):
    config = enable(monkeypatch, context_tokens=24576, cpu_threads=3)
    requests = []
    monkeypatch.setattr(gpu_resources, 'unload_local_ollama', lambda *a, **kw: None)
    monkeypatch.setattr(llm_transport, '_verify_native_context', lambda *a: 24576)
    def post(url, **kw):
        requests.append(kw['json'])
        return httpx.Response(200, request=httpx.Request('POST', url), json={
            'model': kw['json']['model'], 'done': True, 'done_reason': 'stop', 'prompt_eval_count': 20,
            'eval_count': 10, 'message': {'content': '{}'}})
    monkeypatch.setattr(httpx, 'post', post)
    import agenda_runtime
    monkeypatch.setattr(agenda_runtime, 'complete_stream', lambda c, p: post(c.base_url, json=p).json())
    for current in [config, get_llm_config(), get_llm_config()]:
        llm_transport.complete(None, current, model=current.model, messages=[{'role': 'user', 'content': 'Test'}],
                               max_tokens=current.output_budget, **current.reasoning_options)
    assert [p['model'] for p in requests] == [config.model, 'qwen3.5:9b', 'qwen3.5:9b']
    assert requests[0]['options']['num_ctx'] == 24576
    assert requests[0]['options']['num_thread'] == 3 and requests[0]['think'] is False
    assert requests[1]['options'] == requests[2]['options']
    assert requests[1]['options']['num_ctx'] == llm_transport.context_tokens()
    message = [{'role': 'user', 'content': 'same'}]
    assert llm_transport.cache_key(config, message, 'agenda') != llm_transport.cache_key(get_llm_config(), message, 'agenda')
    assert llm_transport.cache_key(config, message, 'agenda') != llm_transport.cache_key(replace(config, context_budget=32768), message, 'agenda')


@pytest.mark.parametrize('failure', ['structure', 'split'])
def test_all_reviews_repairs_and_splits_use_selected_model(isolated_settings, monkeypatch, fake_openai_module, failure):
    config = enable(monkeypatch)
    texts = ['Kommen wir zu TOP 1.', 'Beratung.', 'Technikpause.', 'Ich schließe die öffentliche Sitzung.']
    calls = []
    failed = False
    def generate(client, current, **kw):
        nonlocal failed
        calls.append(current)
        body = json.loads(kw['messages'][1]['content'])
        if 'phase' in body:
            return response({'events': []})
        if 'repair' in body:
            return response({'gap_reasons': {'2': 'Technikpause ohne Sachbezug.'}})
        labels = {str(i): (None if i == 2 else 'public:2' if i == 3 else 'public:1')
                  for i in range(body['target_start'], body['target_end']+1)}
        gaps = [{'line_index': 2, 'reason': 'Technikpause ohne Sachbezug.'}] if '2' in labels else []
        if not failed:
            failed = True
            if failure == 'split':
                raise TimeoutError('synthetic provider failure')
            gaps = []
        return response({'assignments': labels, 'uncertain_lines': [], 'gaps': gaps})
    monkeypatch.setattr(agenda_timeline, 'complete', generate)
    monkeypatch.setattr(agenda_llm, 'complete', generate)
    result = segment_known_agenda([TranscriptUtterance('M', t) for t in texts],
                                 ['[Öffentlich] 1 Haushalt', '[Öffentlich] 2 Schließung'], use_llm=True)
    assert result.assignments == [0, 0, None, 1]
    assert result.llm.status == 'success'
    assert {'timeline', 'line_assignment', 'boundary_review', 'gap_review'} <= {c.get('phase') for c in result.llm.chunks}
    assert all(c == config for c in calls)
    if failure == 'split':
        assert any(c.get('parent_cache_key') for c in result.llm.chunks)
    else:
        assert any(c.get('repair_history') for c in result.llm.chunks)


def test_model_load_error_is_visible_without_qwen_fallback(isolated_settings, monkeypatch, fake_openai_module):
    enable(monkeypatch)
    def fail(*args, **kwargs):
        response = httpx.Response(500, request=httpx.Request('POST', 'http://ollama/api/chat'),
                                  json={'error': 'model requires more system memory'})
        response.raise_for_status()
    monkeypatch.setattr(agenda_timeline, 'complete', fail)
    result = segment_known_agenda([TranscriptUtterance('S', 'Haushalt.')], ['1 Haushalt'], use_llm=True)
    assert result.llm.status == 'failed'
    assert result.assignments == [None]
    assert result.llm.attempted_calls == 1
    assert 'RAM-/VRAM' in ' '.join(result.llm.warnings)
    assert not any(c.calls for c in fake_openai_module.instances)


def test_regular_pdf_pipeline_keeps_pdf_qwen_and_routes_top_calls_to_gemma(isolated_settings, monkeypatch, fake_openai_module):
    import main
    import extract_tops
    enable(monkeypatch)
    calls = []
    monkeypatch.setattr(extract_tops, 'extract_text_from_pdf', lambda _: 'Einladung: 1 Haushalt')
    def pdf(client, config, **kw):
        calls.append(('pdf', config.model))
        return response({'tops': [{'number': '1', 'title': 'Haushalt', 'section': None}], 'metadata': {}})
    def agenda(client, config, **kw):
        body = json.loads(kw['messages'][1]['content'])
        calls.append(('agenda', config.model))
        if 'phase' in body:
            return response({'events': []})
        return response({'assignments': {'0': 'unspecified:1'}, 'uncertain_lines': [], 'gaps': []})
    monkeypatch.setattr(extract_tops, 'complete', pdf)
    monkeypatch.setattr(agenda_timeline, 'complete', agenda)
    monkeypatch.setattr(agenda_llm, 'complete', agenda)
    monkeypatch.setattr(main, 'save_pipeline_state', lambda *a, **kw: None)
    monkeypatch.setattr(main, 'ensure_pipeline_not_cancelled', lambda *a: None)
    monkeypatch.setattr(main, 'append_pipeline_warning', lambda *a: None)
    tops, assignments, info, _ = main.detect_pipeline_agenda('test',
        [{'line_id': 'stable-line', 'speaker': 'M', 'text': 'Kommen wir zu TOP 1.', 'start': 1, 'end': 3}],
        known_tops=[], pdf_path='test.pdf', options={'auto_detect_tops_from_pdf': True, 'agenda_use_llm': True})
    assert tops == ['1. Haushalt'] and assignments == [0]
    assert calls == [('pdf', 'qwen3.5:9b'), ('agenda', 'gemma4:31b-it-q4_K_M'), ('agenda', 'gemma4:31b-it-q4_K_M')]
    assert info['llm']['provenance']['input_references'][0]['line_id'] == 'stable-line'


def test_local_tokenizer_budget_and_provider_truncation_guard(isolated_settings, monkeypatch):
    import hashlib
    from tokenizers import Tokenizer, models as tokenizer_models, pre_tokenizers
    tokenizer = Tokenizer(tokenizer_models.WordLevel({'[UNK]': 0, 'Wort': 1}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    path = isolated_settings / 'tokenizer.json'
    tokenizer.save(str(path))
    config = replace(get_llm_config(), task='agenda', context_budget=8192,
                     tokenizer_json=str(path), tokenizer_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    messages = [{'role': 'user', 'content': 'Wort '*200}]
    assert llm_transport.content_tokens(messages, config) == 200
    assert llm_transport.input_bound(messages, config=config) == 744
    assert llm_transport.input_bound(messages) > llm_transport.input_bound(messages, config=config)
    monkeypatch.setenv('LLM_OLLAMA_NATIVE', 'true')
    monkeypatch.setattr(gpu_resources, 'unload_local_ollama', lambda *a, **k: None)
    monkeypatch.setattr(llm_transport, '_verify_native_context', lambda *a: 8192)
    monkeypatch.setattr(httpx, 'post', lambda url, **kw: httpx.Response(200,
        request=httpx.Request('POST', url), json={'done': True, 'done_reason': 'stop',
            'prompt_eval_count': 20, 'eval_count': 1, 'message': {'content': '{}'}}))
    import agenda_runtime
    monkeypatch.setattr(agenda_runtime, 'complete_stream', lambda c, p: httpx.post(c.base_url, json=p).json())
    with pytest.raises(llm_transport.ContextBudgetError, match='below original input'):
        llm_transport.complete(None, config, model=config.model, messages=messages, max_tokens=100)


def test_separate_runner_unloads_primary_without_changing_summary_options(isolated_settings, monkeypatch):
    monkeypatch.setenv('AGENDA_LLM_BASE_URL', 'http://ollama-agenda:11434/v1')
    config = enable(monkeypatch)
    primary = get_llm_config()
    releases = []
    monkeypatch.setattr(gpu_resources, 'unload_local_ollama', lambda c, **kw: releases.append((c.base_url, kw)))
    monkeypatch.setattr(llm_transport, '_complete', lambda client, c, **kw: c.model)
    assert llm_transport.complete(None, config) == config.model
    assert releases == [(primary.base_url, {}), (config.base_url, {'except_model': config.model})]
    for _ in range(2):
        assert llm_transport.complete(None, get_llm_config()) == 'qwen3.5:9b'
    assert len(releases) == 2
    assert get_llm_config() == primary


def test_unconfirmed_release_blocks_subsequent_gpu_work(isolated_settings, monkeypatch):
    config = enable(monkeypatch)
    monkeypatch.setattr(gpu_resources, '_release_error', False)
    def fail(_): raise gpu_resources.GPUResourceError('unload failed')
    monkeypatch.setattr(gpu_resources, 'unload_ollama_model', fail)
    usage = AgendaLLMUsage(True, 'test')
    with pytest.raises(gpu_resources.GPUResourceError):
        with models.model_session(config, usage): usage.attempted_calls += 1
    with pytest.raises(gpu_resources.GPUResourceError):
        with gpu_resources.gpu_slot(): pytest.fail('GPU work started despite failed release')


def test_unknown_agenda_discovery_has_isolated_cache_and_provenance(isolated_settings, monkeypatch, fake_openai_module):
    from agenda_detection import detect_agenda_from_transcript
    enable(monkeypatch)
    monkeypatch.setenv('LLM_CACHE_DIR', str(isolated_settings / 'cache'))
    calls = []
    def generate(client, config, **kw):
        calls.append(config.model)
        body = kw['messages'][1]['content']
        if not body.startswith('{'):
            return response({'tops': [{'top_title': '1 Haushalt', 'start_index': 0, 'end_index': 0,
                                      'evidence_index': 0, 'evidence_text': 'Kommen wir zu TOP 1, Haushalt.'}]})
        if 'phase' in json.loads(body):
            return response({'events': []})
        return response({'assignments': {'0': 'unspecified:1'}, 'uncertain_lines': [], 'gaps': []})
    for module in (llm_transport, agenda_timeline, agenda_llm):
        monkeypatch.setattr(module, 'complete', generate)
    transcript = [TranscriptUtterance('M', 'Kommen wir zu TOP 1, Haushalt.')]
    first = detect_agenda_from_transcript(transcript, use_llm=True, cache_namespace='one')
    assert first.assignments == [0] and first.llm.attempted_calls == 3
    detail = first.llm.provenance['agenda_discovery'][0]
    assert detail['digest'] == 'gemma-digest' and detail['input_identity']
    assert detail['context_tokens'] == 32768 and detail['status'] == 'success'
    second = detect_agenda_from_transcript(transcript, use_llm=True, cache_namespace='one')
    assert second.assignments == [0] and second.llm.attempted_calls == 0
    assert second.llm.provenance['agenda_discovery'][0]['status'] == 'cached'
    third = detect_agenda_from_transcript(transcript, use_llm=True, cache_namespace='two')
    assert third.assignments == [0] and third.llm.attempted_calls == 3
    assert calls == ['gemma4:31b-it-q4_K_M'] * 6


@pytest.mark.parametrize('known', [True, False])
def test_invalid_separate_endpoint_is_visible_and_cannot_fall_back(isolated_settings, monkeypatch, fake_openai_module, known):
    from agenda_detection import detect_agenda_from_transcript
    models.save_settings(models.AgendaModelSettings(enabled=True))
    monkeypatch.setenv('AGENDA_LLM_BASE_URL', 'https://external.example/v1')
    transcript = [TranscriptUtterance('M', 'Kommen wir zu TOP 1, Haushalt.')]
    result = (segment_known_agenda(transcript, ['1 Haushalt'], use_llm=True) if known
              else detect_agenda_from_transcript(transcript, use_llm=True))
    assert result.llm.status == 'failed' and result.assignments == [None]
    assert result.llm.provenance['model'] == 'gemma4:31b-it-q4_K_M'
    assert 'lokalen Ollama-Dienst' in ' '.join(result.llm.failure_reasons)
    assert not fake_openai_module.instances


def test_timeline_repair_splits_also_review_their_join(isolated_settings, monkeypatch):
    config = enable(monkeypatch)
    transcript = [TranscriptUtterance('M', f'Originalbeitrag {i}.') for i in range(12)]
    requests = []
    def generate(client, config, **kw):
        body = json.loads(kw['messages'][1]['content'])
        requests.append(body)
        if len(requests) == 1:
            return response({'events': [{'index': 0, 'quote': 'Erfundener Beleg', 'kind': 'uncertain',
                'section': None, 'top_id': None, 'uncertain': True, 'reason': 'Unsicher', 'contradictions': []}]})
        return response({'events': []})
    monkeypatch.setattr(agenda_timeline, 'complete', generate)
    usage = AgendaLLMUsage(True, 'test')
    timeline = agenda_timeline.analyze(None, config, transcript, [], usage, {})
    assert set(range(12)) == {i for a,b in timeline['coverage'] for i in range(a,b+1)}
    assert len(timeline['seams']) == 1
    assert requests[-1]['phase'] == 'timeline_seam'
    assert requests[-1]['target_start'] <= 5 < requests[-1]['target_end']
    assert any(c.get('parent_cache_key') for c in usage.chunks)


def test_timeline_case_repair_preserves_raw_quote_and_does_not_accept_invented_words():
    transcript = [TranscriptUtterance('M', 'Nach der Pause beraten wir den Haushalt.')]
    event = {'index': 0, 'quote': 'Beraten wir den Haushalt.', 'kind': 'continuation',
             'section': None, 'top_id': 'budget', 'uncertain': False, 'reason': 'Fortsetzung', 'contradictions': []}
    repairs = []
    checked = agenda_timeline.validate({'events': [event]}, transcript, ['budget'], {0}, repairs)
    assert event['quote'] == 'Beraten wir den Haushalt.' and not event['uncertain']
    assert checked[0]['quote'] == 'beraten wir den Haushalt.'
    assert checked[0]['model_quote'] == event['quote'] and checked[0]['uncertain']
    assert repairs[0]['kind'] == 'unique_case_only_original_span'
    for quote in ['Beschließen wir den Haushalt.', 'Beraten wir den Haushalt .']:
        with pytest.raises(ValueError, match='invalid_original_evidence'):
            agenda_timeline.validate({'events': [dict(event, quote=quote)]}, transcript, ['budget'], {0})
    with pytest.raises(ValueError, match='invalid_original_evidence'):
        agenda_timeline.validate({'events': [dict(event, quote='Haushalt')]},
            [TranscriptUtterance('M', 'haushalt HAUSHALT')], ['budget'], {0})
