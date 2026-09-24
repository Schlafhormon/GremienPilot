import asyncio
import json
import os
import threading
import time
from dataclasses import replace

import httpx
import pytest

import llm_transport as transport
from llm_config import get_llm_config, ModelConfigurationError

MESSAGES = [{'role': 'user', 'content': 'Prüfe den Haushalt.'}]


class Bytes(httpx.AsyncByteStream):
    def __init__(self, chunks, delay=0, error=None):
        self.chunks, self.delay, self.error = chunks, delay, error
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            await asyncio.sleep(self.delay)
            yield chunk
        if self.error:
            raise self.error

    async def aclose(self):
        self.closed = True


def ndjson(*events):
    return [(json.dumps(e) + '\n').encode() for e in events]


def sse(*events):
    return [('data: ' + json.dumps(e) + '\n\n').encode() for e in events]


def terminal(**kwargs):
    return {'model': 'gemma4:31b-it-q8_0', 'done': True, 'done_reason': 'stop',
            'prompt_eval_count': 42, 'eval_count': 5, 'message': {'content': '{}'}, **kwargs}


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setenv('LLM_PROVIDER', 'ollama')
    monkeypatch.setenv('LLM_MODEL', 'gemma4:31b-it-q8_0')
    monkeypatch.setenv('LLM_BASE_URL', 'http://provider.test/v1')
    state = {'calls': [], 'actual_context': 16384, 'digest': 'sha256:test', 'stream': Bytes(ndjson(terminal()))}
    def handle(request):
        state['calls'].append(request)
        path = request.url.path
        if path == '/api/version':
            return httpx.Response(200, json={'version': state.get('version', '0.34.2')})
        if path == '/api/show':
            return httpx.Response(200, json={'model_info': {'gemma4.context_length': 262144}, 'capabilities': ['vision', 'thinking']})
        if path == '/api/tags':
            return httpx.Response(200, json={'models': [{'name': 'gemma4:31b-it-q8_0', 'digest': 'sha256:test'}]})
        if path == '/api/ps':
            return httpx.Response(200, json={'models': [{'model': 'gemma4:31b-it-q8_0', 'digest': state['digest'], 'context_length': state['actual_context']}]})
        if path.endswith('/chat') or path.endswith('/chat/completions'):
            if state.get('responses'):
                return state['responses'].pop(0)
            return httpx.Response(200, stream=state['stream'])
        pytest.fail('Unexpected path: ' + path)
    cls = httpx.AsyncClient
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kw: cls(transport=httpx.MockTransport(handle), **kw))
    return state


def call(config=None, **kwargs):
    config = config or get_llm_config()
    return transport.complete(None, config, model=config.model, messages=kwargs.pop('messages', MESSAGES),
                              max_tokens=kwargs.pop('max_tokens', 100), **kwargs)


def test_budget_counts_utf8_system_user_and_output(monkeypatch):
    monkeypatch.setenv('LLM_CONTEXT_TOKENS', '4096')
    messages = [{'role': 'system', 'content': 'ä' * 1000}, {'role': 'user', 'content': 'x' * 1000}]
    assert transport.input_bound(messages) == 3576
    assert transport.fits(messages, 500)
    assert not transport.fits(messages, 600)


@pytest.mark.parametrize('mode', ['fast', 'slow'])
def test_larger_shared_context_is_sent_to_and_verified_on_ollama(server, monkeypatch, mode):
    from processing_mode import processing_scope
    monkeypatch.delenv('LLM_CONTEXT_TOKENS', raising=False)
    server['actual_context'] = 131072
    with processing_scope(mode):
        result = call(messages=[{'role': 'user', 'content': 'x' * 42000}])
    payload = json.loads(next(r for r in server['calls'] if r.url.path == '/api/chat').content)
    assert payload['options']['num_ctx'] == 131072
    assert payload['truncate'] is False and payload['shift'] is False
    assert result.llm_provenance['verified_context_tokens'] == 131072


@pytest.mark.parametrize('effort, expected', [('', None), ('none', False), ('low', True), ('max', True)])
def test_native_options_schema_boolean_gemma_thinking(server, monkeypatch, effort, expected):
    monkeypatch.setenv('LLM_REASONING_EFFORT', effort)
    monkeypatch.setenv('LLM_CPU_THREADS', '16')
    monkeypatch.setenv('LLM_GPU_LAYERS', '0')
    monkeypatch.setenv('LLM_TOP_K', '64')
    monkeypatch.setenv('LLM_KEEP_ALIVE', '30m')
    result = call(response_format={'type': 'json_schema', 'json_schema': {'schema': {'type': 'object'}}})
    payload = json.loads(next(r for r in server['calls'] if r.url.path == '/api/chat').content)
    assert payload['stream'] is True
    assert payload['truncate'] is False and payload['shift'] is False
    assert payload['options'] == {'num_ctx': 16384, 'num_predict': 100, 'num_thread': 16, 'num_gpu': 0, 'top_k': 64, 'temperature': 0.1}
    assert payload.get('think') == expected
    assert payload['keep_alive'] == '30m'
    assert payload['format'] == {'type': 'object'}
    assert result.llm_provenance['digest'] == 'sha256:test'
    assert result.llm_provenance['verified_context_tokens'] == 16384
    assert 'api_key' not in result.llm_provenance


@pytest.mark.parametrize('image', ['aW1hZ2U=', 'data:image/png;base64,aW1hZ2U='])
def test_multimodal_native_budget_and_wire_format(server, monkeypatch, image):
    monkeypatch.setenv('LLM_IMAGE_TOKENS', '2048')
    messages = [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': image}}, {'type': 'text', 'text': 'OCR'}]}]
    result = call(messages=messages)
    payload = json.loads(next(r for r in server['calls'] if r.url.path == '/api/chat').content)
    assert payload['messages'] == [{'role': 'user', 'content': 'OCR', 'images': ['aW1hZ2U=']}]
    assert result.llm_provenance['input_token_bound'] == 512 + 32 + 3 + 2048


def test_unknown_image_budget_fails_before_network(server):
    with pytest.raises(ModelConfigurationError, match='LLM_IMAGE_TOKENS'):
        call(messages=[{'role': 'user', 'content': 'OCR', 'images': ['aW1hZ2U=']}])
    assert not server['calls']


def test_schema_and_two_generation_phases_cannot_overflow(server):
    config = replace(get_llm_config(), context_tokens=4096, thinking=True)
    with pytest.raises(transport.ContextBudgetError):
        call(config, max_tokens=1400, response_format={'type': 'json_schema', 'json_schema': {'schema': {'description': 'x'*1000}}})
    assert not server['calls']


@pytest.mark.parametrize('event', [terminal(done_reason='length'), terminal(done=False), terminal(eval_count=100),
                                  terminal(prompt_eval_count=16384), terminal(message={'thinking': 'not an answer'})])
def test_incomplete_native_output_never_succeeds(server, event):
    server['stream'] = Bytes(ndjson(event))
    with pytest.raises(transport.IncompleteResponseError):
        call()
    assert server['stream'].closed


@pytest.mark.parametrize('ended', [False, True])
def test_failed_stream_retains_private_fragment_and_observed_terminal_metrics(server, monkeypatch, ended, caplog):
    import durable_jobs
    artifacts, metrics = [], []
    monkeypatch.setattr(durable_jobs, 'artifact', lambda step, kind, value: artifacts.append((kind, value)))
    monkeypatch.setattr(durable_jobs, 'record_metric', metrics.append)
    events = [{'message': {'content': 'PRIVATE PARTIAL'}, 'done': False}]
    if ended:
        events.append(terminal(done_reason='length', eval_count=100, message={'content': ''}))
    server['stream'] = Bytes(ndjson(*events))
    with pytest.raises(transport.IncompleteResponseError):
        call()
    failure = next(v for kind, v in artifacts if kind == 'transport_failure')
    assert failure['partial_content'] == 'PRIVATE PARTIAL'
    assert failure['provenance']['terminal_received'] is ended
    assert failure['provenance']['finish_reason'] == ('length' if ended else None)
    assert metrics[0]['status'] == 'failed'
    assert ('generated_tokens' in metrics[0]) is ended
    assert 'PRIVATE' not in caplog.text


@pytest.mark.parametrize('actual', [None, 4096])
def test_effective_context_is_verified(server, actual):
    server['actual_context'] = actual
    with pytest.raises(transport.ContextBudgetError):
        call()


def test_digest_change_fails(server):
    server['digest'] = 'changed'
    with pytest.raises(ModelConfigurationError, match='digest'):
        call()


def test_old_ollama_is_rejected_without_generation(server):
    server['version'] = '0.20.0'
    with pytest.raises(ModelConfigurationError, match='0.34.2'):
        call()
    assert not any(r.url.path == '/api/chat' for r in server['calls'])


def test_thinking_stream_is_not_exposed_as_content(server):
    server['stream'] = Bytes(ndjson({'message': {'thinking': 'private reasoning'}}, {'message': {'content': 'final '}}, terminal()))
    result = call()
    assert result.choices[0].message.content == 'final {}'
    assert 'private reasoning' not in str(result.llm_provenance)


def test_long_stream_has_no_total_deadline(server):
    # Virtual clock: multiple hours of active generation without sleeping for hours.
    original = transport.time.monotonic
    elapsed = [0]
    monkey = pytest.MonkeyPatch()
    monkey.setattr(transport.time, 'monotonic', lambda: original() + elapsed[0])
    class Long(Bytes):
        async def __aiter__(self):
            for chunk in self.chunks:
                elapsed[0] += 3600
                yield chunk
    server['stream'] = Long(ndjson({'message': {'thinking': 'work'}}, terminal()))
    try:
        assert call().choices[0].message.content == '{}'
    finally:
        monkey.undo()


@pytest.mark.parametrize('cancel', [False, True])
def test_cancel_or_total_timeout_interrupts_silent_stream(server, cancel):
    server['stream'] = Bytes(ndjson(terminal()), delay=30)
    start = time.monotonic()
    def check():
        if time.monotonic()-start > .15:
            raise transport.LLMCancelledError()
    config = replace(get_llm_config(), total_seconds=0 if cancel else .15)
    with pytest.raises(transport.LLMCancelledError if cancel else transport.LLMTotalTimeout):
        call(config, check_cancel=check if cancel else None)
    assert time.monotonic()-start < 2
    assert server['stream'].closed


def test_cancel_while_waiting_for_inference_lock(server):
    transport._INFERENCE_LOCK.acquire()
    try:
        with pytest.raises(transport.LLMCancelledError):
            call(check_cancel=lambda: (_ for _ in ()).throw(transport.LLMCancelledError()))
    finally:
        transport._INFERENCE_LOCK.release()
    assert not server['calls']


@pytest.mark.parametrize('error', [httpx.ReadTimeout('idle'), httpx.ConnectError('connect'), httpx.RemoteProtocolError('broken')])
def test_transient_failure_restarts_complete_stream_without_duplicate_text(server, error):
    server['responses'] = [httpx.Response(200, stream=Bytes(ndjson({'message': {'content': 'discard'}}), error=error)),
                           httpx.Response(200, stream=Bytes(ndjson(terminal())))]
    result = call(replace(get_llm_config(), max_retries=1))
    assert result.choices[0].message.content == '{}'
    assert result.llm_provenance['transport_attempts'] == 2


@pytest.mark.parametrize('status,body,retry', [(503, 'busy', True), (429, 'rate limit', True),
    (500, 'model requires more system memory', False), (500, 'out of memory', False), (400, 'bad options', False), (404, 'model not found', False)])
def test_status_retry_policy(server, status, body, retry):
    server['responses'] = [httpx.Response(status, text=body), httpx.Response(200, stream=Bytes(ndjson(terminal())))]
    if retry:
        assert call(replace(get_llm_config(), max_retries=1)).llm_provenance['transport_attempts'] == 2
    else:
        with pytest.raises(transport.ProviderError):
            call(replace(get_llm_config(), max_retries=1))
        assert len(server['responses']) == 1


def test_openai_multimodal_reasoning_schema_and_completion_budget(server, monkeypatch):
    monkeypatch.setenv('LLM_PROVIDER', 'openai-compatible')
    monkeypatch.setenv('LLM_REASONING_EFFORT', 'high')
    monkeypatch.setenv('LLM_OUTPUT_PARAMETER', 'max_completion_tokens')
    monkeypatch.setenv('LLM_THINKING_TOKENS', '200')
    monkeypatch.setenv('LLM_IMAGE_TOKENS', '2048')
    server['stream'] = Bytes(sse({'choices': [{'delta': {'reasoning_content': 'private'}, 'finish_reason': None}]},
                                {'choices': [{'delta': {'content': '{}'}, 'finish_reason': 'stop'}]}))
    messages = [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'https://example.test/img.png', 'detail': 'high'}}, {'type': 'text', 'text': 'OCR'}]}]
    result = call(messages=messages, response_format={'type': 'json_object'}, ollama_think=False)
    payload = json.loads(server['calls'][0].content)
    assert payload['messages'] == messages
    assert payload['reasoning_effort'] == 'high'
    assert payload['max_completion_tokens'] == 300
    assert 'options' not in payload and 'think' not in payload
    assert result.choices[0].message.content == '{}'


@pytest.mark.parametrize('reason', ['length', None, 'content_filter', 'tool_calls'])
def test_openai_requires_complete_final_answer(server, monkeypatch, reason):
    monkeypatch.setenv('LLM_PROVIDER', 'openai-compatible')
    server['stream'] = Bytes(sse({'choices': [{'delta': {'content': 'partial'}, 'finish_reason': reason}]}))
    with pytest.raises(transport.IncompleteResponseError):
        call()


def test_cache_private_versioned_and_configuration_bound(monkeypatch, tmp_path):
    monkeypatch.setenv('LLM_CACHE_DIR', str(tmp_path))
    config = get_llm_config()
    key = transport.cache_key(config, MESSAGES, 'test')
    transport.cache_write(key, {'complete': True})
    assert transport.cache_read(key) == {'complete': True}
    for changed in [replace(config, temperature=1), replace(config, model='override'), replace(config, model_revision='new'), replace(config, thinking_tokens=5, output_parameter='max_completion_tokens')]:
        assert transport.cache_read(transport.cache_key(changed, MESSAGES, 'test')) is None
    if os.name == 'posix':
        assert next(tmp_path.iterdir()).stat().st_mode & 0o777 == 0o600
    monkeypatch.setenv('LLM_MODEL_REVISION', '')
    assert transport.cache_key(get_llm_config(), MESSAGES, 'test') is None


@pytest.mark.parametrize('key,value', [('LLM_CONTEXT_TOKENS', '1024'), ('LLM_READ_TIMEOUT_SECONDS', 'nan'),
 ('LLM_TOTAL_TIMEOUT_SECONDS', '-1'), ('LLM_MAX_RETRIES', '2.5'), ('LLM_THINKING', 'low'), ('LLM_TOP_P', '1.1'),
 ('LLM_PROVIDER', 'typo'), ('LLM_CPU_THREADS', '0'), ('LLM_TEMPERATURE', 'inf')])
def test_invalid_configuration(key, value, monkeypatch):
    monkeypatch.setenv(key, value)
    with pytest.raises(ModelConfigurationError):
        get_llm_config()


def test_override_is_preserved_and_tokenizer_mismatch_rejected(monkeypatch):
    assert get_llm_config('old-browser-model').model == 'old-browser-model'
    assert get_llm_config('old-browser-model').model_source == 'request'
    monkeypatch.setenv('LLM_TOKENIZER_PATH', '/local/tokenizer')
    monkeypatch.setenv('LLM_TOKENIZER_MODEL', 'qwen3:8b')
    with pytest.raises(ModelConfigurationError, match='effective model'):
        get_llm_config('override')


def test_local_model_tokenizer_and_schema_are_counted(monkeypatch):
    class Tokenizer:
        def encode(self, text):
            from types import SimpleNamespace
            return SimpleNamespace(ids=[1] * (50 if text == MESSAGES[0]["content"] else 10))
    monkeypatch.setenv('LLM_TOKENIZER_PATH', '/local/tokenizer')
    monkeypatch.setenv('LLM_TOKENIZER_MODEL', 'qwen3:8b')
    monkeypatch.setattr(transport, '_tokenizer', lambda p: Tokenizer())
    assert transport.input_bound(MESSAGES, response_format={'type': 'json_object'}) == 604


@pytest.fixture
def streaming_http_server():
    """Real sockets, no model: verify httpx's actual read timeout semantics."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            if self.path.startswith('/silent'):
                time.sleep(.5)
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            try:
                for _ in range(8):
                    self.wfile.write(sse({'choices': [{'delta': {'content': 'x'}, 'finish_reason': None}]})[0])
                    self.wfile.flush()
                    time.sleep(.025)
                self.wfile.write(sse({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})[0])
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
    http = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=http.serve_forever, daemon=True)
    worker.start()
    yield 'http://127.0.0.1:' + str(http.server_port)
    http.shutdown()
    http.server_close()
    worker.join(timeout=1)


def test_real_http_active_stream_outlives_read_timeout(streaming_http_server):
    config = replace(get_llm_config(), base_url=streaming_http_server, timeout_seconds=.1)
    start = time.monotonic()
    assert call(config).choices[0].message.content == 'x'*8
    assert time.monotonic()-start > .15


def test_real_http_read_timeout_before_first_token(streaming_http_server):
    config = replace(get_llm_config(), base_url=streaming_http_server+'/silent', timeout_seconds=.05, load_seconds=.05)
    # The transport watchdog and socket timeout race, especially on Windows.
    # Both must abort the silent stream; neither may produce a completion.
    with pytest.raises((httpx.ReadTimeout, TimeoutError)):
        call(config)


def test_progress_and_cancellation_during_model_metadata_load(server, monkeypatch):
    async def wait_for_load(*args):
        await asyncio.sleep(30)
    monkeypatch.setattr(transport, '_metadata', wait_for_load)
    updates = []
    start = time.monotonic()
    def check():
        if time.monotonic() - start > .15:
            raise transport.LLMCancelledError()
    with pytest.raises(transport.LLMCancelledError):
        call(check_cancel=check, progress_callback=updates.append)
    assert updates[0]['phase'] == 'loading'
    assert updates[0]['last_delta_at'] is None
    assert not server['calls']


def test_config_freezes_environment_and_model_source(monkeypatch):
    from llm_config import configured
    @configured
    def operation(model=None):
        initial = get_llm_config(model)
        monkeypatch.setenv('LLM_MODEL', 'changed-mid-operation')
        monkeypatch.setenv('LLM_CONTEXT_TOKENS', '32768')
        return initial, get_llm_config(initial.model)
    a, b = operation()
    assert a is b
    assert b.model_source == 'environment'
    assert b.context_tokens == 16384
    assert get_llm_config().model == 'changed-mid-operation'


def test_legacy_prompt_controls_removed_without_changing_transcript(server):
    messages = [{'role': 'system', 'content': '/no_think\nFachliche Vorgabe\n/think\n'},
                {'role': 'user', 'content': '/no_think\nOriginalzitat'}]
    call(messages=messages)
    payload = json.loads(next(r for r in server['calls'] if r.url.path == '/api/chat').content)
    assert payload['messages'][0]['content'] == 'Fachliche Vorgabe\n'
    assert payload['messages'][1]['content'] == messages[1]['content']


def test_audit_contains_no_credentials_or_reasoning(server, monkeypatch, tmp_path):
    monkeypatch.setenv('LLM_AUDIT_DIR', str(tmp_path))
    monkeypatch.setenv('LLM_API_KEY', 'private-key')
    server['stream'] = Bytes(ndjson({'message': {'thinking': 'secret-thought'}}, terminal()))
    call()
    text = next(tmp_path.iterdir()).read_text()
    assert 'private-key' not in text and 'secret-thought' not in text
    assert json.loads(text)['provenance']['digest'] == 'sha256:test'


def test_external_cache_without_revision_does_not_break_summary(fake_openai_module, monkeypatch, tmp_path):
    import summarize
    monkeypatch.setenv('LLM_CACHE_DIR', str(tmp_path))
    monkeypatch.delenv('LLM_MODEL_REVISION')
    monkeypatch.setenv('LLM_SUMMARY_GROUNDING_MAX_CALLS', '1')
    from summary_fixtures import SummaryModel
    fake_openai_module.content = SummaryModel()
    result = summarize.summarize_segment('TOP', 'Beratung.')
    assert result.summary
    assert not list(tmp_path.iterdir())


def test_permanent_memory_error_is_not_repaired_by_summary_pipeline(fake_openai_module, monkeypatch):
    import summarize
    fake_openai_module.responses = [transport.ProviderError('out of memory', 500)]
    with pytest.raises(summarize.LLMCallError) as error:
        summarize.summarize_segment('TOP', 'Langer Quelltext. '*40)
    assert error.value.category == 'memory'
    assert len(fake_openai_module.instances[0].calls) == 1


def test_model_tag_changed_since_cache_key_aborts_before_chat(server, monkeypatch, tmp_path):
    config = get_llm_config()
    monkeypatch.setenv('LLM_CACHE_DIR', str(tmp_path))
    monkeypatch.setattr(transport, 'model_fingerprint', lambda c: {'digest': 'old-digest'})
    transport.cache_key(config, MESSAGES, 'test')
    with pytest.raises(ModelConfigurationError, match='tag changed'):
        call(config)
    assert not any(r.url.path == '/api/chat' for r in server['calls'])


def test_context_failure_never_changes_requested_model(server):
    with pytest.raises(transport.ContextBudgetError):
        call(messages=[{'role': 'user', 'content': 'x'*16384}])
    assert not server['calls']


def test_timeout_configuration_is_separate_and_legacy_compatible(monkeypatch):
    monkeypatch.setenv('LLM_TIMEOUT_SECONDS', '1800')
    monkeypatch.setenv('LLM_CONNECT_TIMEOUT_SECONDS', '3')
    config = get_llm_config()
    assert config.http_timeout.connect == 3
    assert config.http_timeout.read == 1800
    assert config.total_seconds == 0
    monkeypatch.setenv('LLM_READ_TIMEOUT_SECONDS', '400')
    assert get_llm_config().http_timeout.read == 400
