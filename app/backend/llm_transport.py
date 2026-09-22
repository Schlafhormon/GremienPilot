"""Provider contracts, bounded context and cancellable internal streaming.

The synchronous facade is used by pipeline worker threads. Network I/O is async
so cancellation also interrupts model loading and a completely silent stream.
"""
import asyncio
import base64
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
import hashlib
import json
import logging
import os
import re
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import httpx
from gpu_resources import llm_gpu_slot
from llm_config import get_llm_config, ModelConfigurationError

logger = logging.getLogger(__name__)
CACHE_VERSION = 3


class ContextBudgetError(ValueError):
    pass


class IncompleteResponseError(ContextBudgetError):
    pass


class LLMCancelledError(Exception):
    pass


class LLMTotalTimeout(TimeoutError):
    pass


class ProviderError(RuntimeError):
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


_CONTROL = ContextVar('llm_control', default=(None, None))
_EXPECTED_DIGEST = ContextVar('llm_expected_digest', default=None)


@contextmanager
def request_control(check_cancel=None, progress=None):
    token = _CONTROL.set((check_cancel, progress))
    try:
        yield
    finally:
        _CONTROL.reset(token)


def context_tokens(config=None):
    return (config or get_llm_config()).context_tokens


def _parts(messages):
    text_messages, images = [], []
    for message in messages:
        if message.get('role') not in {'system', 'user', 'assistant'}:
            raise ModelConfigurationError('Unsupported message role')
        content = message.get('content', '')
        texts, pictures = [], list(message.get('images', []))
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for part in content:
                if part.get('type') == 'text':
                    texts.append(part['text'])
                elif part.get('type') == 'image_url':
                    pictures.append(part['image_url']['url'])
                else:
                    raise ModelConfigurationError('Unsupported content part')
        else:
            raise ModelConfigurationError('Message content must be text or content parts')
        text = '\n'.join(texts)
        if message['role'] == 'system':
            text = re.sub(r'(?m)^[ \t]*/(?:no_think|think)[ \t]*\n?', '', text)
        text_messages.append({'role': message['role'], 'content': text})
        images.append(pictures)
    return text_messages, images


@lru_cache(maxsize=4)
def _tokenizer(path):
    # Explicit local assets only: no downloads or remote Python code during jobs.
    from tokenizers import Tokenizer
    source = Path(path)
    if source.is_dir():
        source = source / 'tokenizer.json'
    try:
        return Tokenizer.from_file(str(source))
    except Exception as exc:
        raise ModelConfigurationError('Cannot load configured local tokenizer.json') from exc


def input_bound(messages, config=None, response_format=None):
    config = config or get_llm_config()
    texts, images = _parts(messages)
    count = sum(map(len, images))
    if count and not config.image_tokens:
        raise ModelConfigurationError('Set LLM_IMAGE_TOKENS for the effective vision model')
    schema = json.dumps(response_format, ensure_ascii=False) if response_format else ''
    if config.tokenizer_path:
        tokenizer = _tokenizer(config.tokenizer_path)
        # Provider template may differ; keep a reserve even with exact text tokens.
        tokens = sum(len(tokenizer.encode(m['content']).ids) + 32 for m in texts)
        tokens += len(tokenizer.encode(schema).ids) if schema else 0
        return 512 + tokens + count * config.image_tokens
    # Conservative UTF-8 byte fallback, documented in docs/llm-configuration.md.
    return 512 + sum(len(m['content'].encode('utf-8')) + 32 for m in texts) + len(schema.encode('utf-8')) + count * config.image_tokens


def fits(messages, output_tokens, config=None, response_format=None):
    return input_bound(messages, config, response_format) + output_tokens <= context_tokens(config)


def _think(config, override=None):
    if override is not None:
        if type(override) is not bool:
            raise ModelConfigurationError('ollama_think must be a boolean')
        return override
    if config.thinking is not None:
        return config.thinking
    effort = config.reasoning_effort
    if effort == 'none':
        return False
    # Gemma has an on/off switch, not named thinking levels.
    if effort and config.model.lower().startswith('gemma4'):
        return True
    return effort


def structured_output_budget(config, maximum, think_override=None):
    maximum = config.output_budget(maximum) + config.thinking_tokens
    return maximum * (2 if (config.provider == 'ollama') and _think(config, think_override) is not False else 1)


def retryable(error):
    if isinstance(error, (ModelConfigurationError, ContextBudgetError, LLMCancelledError, LLMTotalTimeout)):
        return False
    message = str(error).lower()
    if any(word in message for word in ('out of memory', 'requires more system memory', 'insufficient memory',
                                         'unable to allocate', 'failed to allocate', 'cuda error', 'model not found')):
        return False
    status = getattr(error, 'status_code', None) or getattr(getattr(error, 'response', None), 'status_code', None)
    return status in {408, 409, 425, 429, 500, 502, 503, 504} or isinstance(error, (
        httpx.TransportError, TimeoutError, ConnectionError))


def permanent_failure(error):
    if isinstance(error, (ModelConfigurationError, LLMTotalTimeout)):
        return True
    if isinstance(error, ProviderError):
        return not retryable(error)
    return False


def _normal(name):
    return name if ':' in name else name + ':latest'


def _headers(config):
    return {'Authorization': 'Bearer ' + config.api_key}


def _native_url(config):
    return config.base_url.removesuffix('/v1')


async def _json(client, method, url, **kwargs):
    response = await client.request(method, url, **kwargs)
    if response.is_error:
        raise ProviderError(response.text, response.status_code)
    return response.json()


async def _metadata(client, config):
    base = _native_url(config)
    version = await _json(client, 'GET', base + '/api/version')
    try:
        numbers = tuple(int(p) for p in version['version'].split('-')[0].split('.')[:3])
    except (KeyError, ValueError):
        raise ModelConfigurationError('Cannot verify Ollama version')
    if numbers < (0, 34, 2):
        raise ModelConfigurationError('Native transport requires Ollama >= 0.34.2 (truncate/shift contract)')
    show = await _json(client, 'POST', base + '/api/show', json={'model': config.model})
    lengths = [v for k, v in show.get('model_info', {}).items() if k.endswith('.context_length') and type(v) is int]
    if not lengths or config.context_tokens > max(lengths):
        raise ContextBudgetError('Requested context exceeds model context or model metadata is missing')
    tags = await _json(client, 'GET', base + '/api/tags')
    entry = next((m for m in tags.get('models', []) if _normal(m.get('name', '')) == _normal(config.model)), {})
    return {'digest': entry.get('digest'), 'ollama_version': version['version'],
            'model_context_tokens': max(lengths), 'capabilities': show.get('capabilities', [])}


async def _verify_context(client, config, digest):
    data = await _json(client, 'GET', _native_url(config) + '/api/ps')
    entry = next((m for m in data.get('models', []) if _normal(m.get('model', m.get('name', ''))) == _normal(config.model)), {})
    actual = entry.get('context_length')
    if not isinstance(actual, int) or actual < config.context_tokens:
        raise ContextBudgetError('Cannot verify requested effective Ollama context')
    if digest and entry.get('digest') != digest:
        raise ModelConfigurationError('Loaded model digest changed during request')
    return actual


def _native_messages(messages):
    texts, images = _parts(messages)
    for message, pictures in zip(texts, images):
        if pictures:
            encoded = []
            for picture in pictures:
                if picture.startswith('data:image/') and ';base64,' in picture:
                    picture = picture.split(';base64,', 1)[1]
                elif '://' in picture:
                    raise ModelConfigurationError('Native Ollama requires base64 images, not remote URLs')
                try:
                    if not base64.b64decode(picture, validate=True):
                        raise ValueError('empty image')
                except ValueError as exc:
                    raise ModelConfigurationError('Invalid base64 image') from exc
                encoded.append(picture)
            message['images'] = encoded
    return texts


def _openai_messages(messages):
    texts, images = _parts(messages)
    for original, message, pictures in zip(messages, texts, images):
        if pictures:
            parts = [{'type': 'image_url', 'image_url': {'url': p if p.startswith(('data:', 'https://', 'http://')) else 'data:image/png;base64,' + p}} for p in pictures]
            parts.append({'type': 'text', 'text': message['content']})
            # Preserve detail and part ordering for the standard wire format.
            message['content'] = original['content'] if isinstance(original['content'], list) and not original.get('images') else parts
    return texts


async def _openai_stream(client, config, payload):
    """Separate seam for SDK-free contract tests; caller's SDK remains used for diagnostics."""
    async with httpx.AsyncClient(timeout=config.http_timeout, headers=_headers(config)) as http:
        async with http.stream('POST', config.base_url + '/chat/completions', json=payload) as response:
            if response.is_error:
                await response.aread()
                raise ProviderError(response.text, response.status_code)
            async for line in response.aiter_lines():
                if line.startswith('data:'):
                    data = line[5:].strip()
                    if data == '[DONE]':
                        return
                    yield json.loads(data)


async def _generate(client, config, kwargs, progress):
    messages = kwargs['messages']
    maximum = config.output_budget(kwargs['max_tokens'])
    if type(maximum) is not int or maximum <= 0:
        raise ModelConfigurationError('Output budget must be a positive integer')
    if kwargs.get('model', config.model) != config.model:
        raise ModelConfigurationError('Request model differs from resolved configuration')
    override = kwargs.get('ollama_think')
    think = _think(config, override)
    response_format = kwargs.get('response_format')
    if response_format and response_format.get('type') not in {'json_schema', 'json_object', 'text'}:
        raise ModelConfigurationError('Unsupported structured output format')
    if response_format and response_format.get('type') == 'json_schema':
        if not isinstance(response_format.get('json_schema', {}).get('schema'), dict):
            raise ModelConfigurationError('JSON schema must be an object')
    cap = maximum + config.thinking_tokens
    reserved = cap * (2 if (config.provider == 'ollama') and response_format and think is not False else 1)
    bound = input_bound(messages, config, response_format)
    if bound + reserved > config.context_tokens:
        raise ContextBudgetError('Input, images, schema and all provider generation phases exceed context budget')
    snapshot = config.public_snapshot()
    snapshot.update(input_token_bound=bound, reserved_output_tokens=reserved,
                    requested_output_tokens=maximum, provider_generation_limit=cap, effective_thinking=think,
                    schema_sha256=hashlib.sha256(json.dumps(response_format, sort_keys=True).encode()).hexdigest() if response_format else None,
                    temperature=config.temperature if config.temperature is not None else kwargs.get('temperature', 0.1))
    content = []
    content_bytes = 0
    def append_content(value):
        nonlocal content_bytes
        content_bytes += len(value.encode('utf-8'))
        if content_bytes > reserved * 256:
            raise IncompleteResponseError('Provider exceeded response size guard')
        content.append(value)
    finish = None
    if (config.provider == 'ollama'):
        payload = {'model': config.model, 'messages': _native_messages(messages), 'stream': True,
                   'truncate': False, 'shift': False, 'keep_alive': float(config.keep_alive) if re.fullmatch(r'-?\d+(?:\.\d+)?', config.keep_alive) else config.keep_alive,
                   'options': {'num_ctx': config.context_tokens, 'num_predict': cap, 'temperature': snapshot['temperature']}}
        for key, value in [('top_p', config.top_p), ('top_k', config.top_k), ('seed', config.seed),
                           ('num_thread', config.cpu_threads), ('num_gpu', config.gpu_layers)]:
            if value is not None:
                payload['options'][key] = value
        if think is not None:
            payload['think'] = think
        if response_format and response_format['type'] != 'text':
            payload['format'] = 'json' if response_format['type'] == 'json_object' else response_format['json_schema']['schema']
        async with httpx.AsyncClient(timeout=config.http_timeout, headers=_headers(config)) as http:
            metadata = await _metadata(http, config)
            snapshot.update(metadata)
            expected = _EXPECTED_DIGEST.get()
            if expected and expected[0] is config and expected[1] != metadata['digest']:
                raise ModelConfigurationError('Model tag changed after cache lookup')
            if any(_parts(messages)[1]) and 'vision' not in metadata['capabilities']:
                raise ModelConfigurationError('Selected model does not advertise vision')
            if think not in (False, None) and 'thinking' not in metadata['capabilities']:
                raise ModelConfigurationError('Selected model does not advertise thinking')
            final = None
            async with http.stream('POST', _native_url(config) + '/api/chat', json=payload) as response:
                if response.is_error:
                    await response.aread()
                    raise ProviderError(response.text, response.status_code)
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if 'error' in data:
                        raise ProviderError(data['error'], data.get('status'))
                    if data.get('model') and _normal(data['model']) != _normal(config.model):
                        raise ModelConfigurationError('Provider changed the requested model')
                    message = data.get('message', {})
                    append_content(message.get('content', ''))
                    progress('thinking' if message.get('thinking') else 'generating')
                    if data.get('done'):
                        final = data
                        break
            if not final:
                raise IncompleteResponseError('Stream ended without terminal response')
            finish = final.get('done_reason')
            if finish != 'stop':
                raise IncompleteResponseError(f'LLM output incomplete ({finish})')
            count, generated = final.get('prompt_eval_count'), final.get('eval_count')
            # Structured thinking may be included in the second phase's prompt count.
            if type(count) is not int or type(generated) is not int or count < 0 or generated < 0 or count + generated > config.context_tokens or generated >= cap:
                raise IncompleteResponseError('Unexpected context usage or generation cap reached')
            snapshot.update(prompt_tokens=count, generated_tokens=generated)
            snapshot['verified_context_tokens'] = await _verify_context(http, config, metadata['digest'])
    else:
        payload = {'model': config.model, 'messages': _openai_messages(messages), 'stream': True,
                   config.output_parameter: cap, 'temperature': snapshot['temperature']}
        payload.update(config.reasoning_options)
        if response_format:
            payload['response_format'] = response_format
        for key, value in [('top_p', config.top_p), ('seed', config.seed)]:
            if value is not None:
                payload[key] = value
        async for data in _openai_stream(client, config, payload):
            if data.get('model'):
                snapshot['provider_model'] = data['model']
            if data.get('system_fingerprint'):
                snapshot['system_fingerprint'] = data['system_fingerprint']
            if 'error' in data:
                raise ProviderError(str(data['error']))
            for choice in data.get('choices', []):
                delta = choice.get('delta', {})
                append_content(delta.get('content') or '')
                progress('thinking' if delta.get('reasoning_content') or delta.get('reasoning') else 'generating')
                if choice.get('finish_reason') is not None:
                    finish = choice['finish_reason']
            if data.get('usage'):
                snapshot['usage'] = data['usage']
                if data['usage'].get('total_tokens', 0) > config.context_tokens:
                    raise ContextBudgetError('Provider context usage exceeds configured budget')
        if finish != 'stop':
            raise IncompleteResponseError(f'LLM output incomplete ({finish})')
        snapshot.update(digest=config.model_revision or None, verified_context_tokens=None)
    answer = ''.join(content)
    if not answer.strip():
        raise IncompleteResponseError('LLM returned no final content')
    _audit(snapshot, payload, answer)
    logger.info('LLM completion provenance: %s', json.dumps(snapshot, sort_keys=True))
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=answer), finish_reason=finish)],
                           llm_provenance=snapshot)


async def _run(client, config, kwargs, check_cancel, progress, operation=None):
    started = time.monotonic()
    def check():
        if check_cancel:
            check_cancel()
        if config.total_seconds and time.monotonic() - started >= config.total_seconds:
            raise LLMTotalTimeout('LLM total runtime exceeded')
    last_report = [0.0]
    def report(phase):
        now = time.monotonic()
        if progress and (now - last_report[0] >= 1 or phase == 'retrying'):
            progress({'phase': phase, 'elapsed_seconds': round(now - started, 1),
                      'model': config.model, 'config_id': config.public_snapshot()['config_id']})
            last_report[0] = now
    for attempt in range(config.max_retries + 1):
        check()
        task = asyncio.create_task(operation() if operation else _generate(client, config, kwargs, report))
        try:
            while not task.done():
                check()
                report('waiting')
                await asyncio.wait({task}, timeout=0.1)
            result = task.result()
            if hasattr(result, 'llm_provenance'):
                result.llm_provenance['transport_attempts'] = attempt + 1
            return result
        except BaseException as exc:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            if not isinstance(exc, Exception) or not retryable(exc) or attempt == config.max_retries:
                raise
            report('retrying')
            deadline = time.monotonic() + config.retry_backoff_seconds * 2**attempt
            while time.monotonic() < deadline:
                check()
                await asyncio.sleep(min(0.1, max(0, deadline - time.monotonic())))


@contextmanager
def work_slot(lock):
    """Keep existing whole-operation serialization cancellable as well."""
    check_cancel, _ = _CONTROL.get()
    while not lock.acquire(timeout=0.1):
        if check_cancel:
            check_cancel()
    try:
        if check_cancel:
            check_cancel()
        yield
    finally:
        lock.release()


_INFERENCE_LOCK = threading.Lock()


def complete(client, config, **kwargs):
    inherited_cancel, inherited_progress = _CONTROL.get()
    check_cancel = kwargs.pop('check_cancel', None) or inherited_cancel
    progress = kwargs.pop('progress_callback', None) or inherited_progress
    while not _INFERENCE_LOCK.acquire(timeout=0.1):
        if check_cancel:
            check_cancel()
    try:
        if check_cancel:
            check_cancel()
        with llm_gpu_slot(config, check_cancel):
            return _complete(client, config, check_cancel=check_cancel, progress_callback=progress, **kwargs)
    finally:
        _INFERENCE_LOCK.release()



def _complete(client, config, **kwargs):
    check_cancel = kwargs.pop('check_cancel', None)
    progress = kwargs.pop('progress_callback', None)
    return asyncio.run(_run(client, config, kwargs, check_cancel, progress))


def cache_read(key):
    if key is None:
        return None
    directory = os.environ.get('LLM_CACHE_DIR')
    if not directory:
        return None
    path = Path(directory) / (hashlib.sha256(key.encode()).hexdigest() + '.json')
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        return None


def cache_write(key, value):
    if key is None:
        return None
    directory = os.environ.get('LLM_CACHE_DIR')
    if not directory:
        return
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    target = path / (hashlib.sha256(key.encode()).hexdigest() + '.json')
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', dir=path, delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False)
        temporary = handle.name
    os.replace(temporary, target)


def model_fingerprint(config):
    if not (config.provider == 'ollama'):
        return {'model': config.model, 'digest': config.model_revision or None, 'provider': config.provider}
    async def read_tags():
        async with httpx.AsyncClient(timeout=config.http_timeout, headers=_headers(config)) as client:
            data = await _json(client, 'GET', _native_url(config) + '/api/tags')
            entry = next((m for m in data.get('models', []) if _normal(m.get('name', '')) == _normal(config.model)), {})
            return {'model': config.model, 'digest': entry.get('digest'), 'provider': config.provider}
    check_cancel, progress = _CONTROL.get()
    return asyncio.run(_run(None, config, {}, check_cancel, progress, operation=read_tags))


def cache_key(config, messages, purpose, provenance=None):
    # Without an immutable identity no persisted completion is trusted.
    identity = provenance if provenance and provenance.get('digest') else None
    if os.environ.get('LLM_CACHE_DIR') and identity is None:
        try:
            identity = model_fingerprint(config)
        except (httpx.HTTPError, ProviderError, ValueError):
            return None
        if not identity.get('digest'):
            return None
    if identity and identity.get('digest') and config.provider == 'ollama':
        _EXPECTED_DIGEST.set((config, identity['digest']))
    return json.dumps([CACHE_VERSION, purpose, config.public_snapshot(), messages, identity, provenance],
                      ensure_ascii=False, sort_keys=True)


def _audit(snapshot, payload, answer):
    directory = os.environ.get('LLM_AUDIT_DIR')
    if not directory:
        return
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    target = path / (str(time.time_ns()) + '.json')
    # Reasoning is never retained. Source text/answers remain private opt-in data.
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as handle:
        json.dump({'provenance': snapshot, 'request': payload, 'response': {'message': {'content': answer}}}, handle, ensure_ascii=False)
