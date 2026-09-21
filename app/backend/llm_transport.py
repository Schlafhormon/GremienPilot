"""Bounded text-only completions; native Ollama options are sent on every call."""
import hashlib
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace


class ContextBudgetError(ValueError):
    pass


def context_tokens():
    value = int(os.environ.get('LLM_CONTEXT_TOKENS', '16384'))
    if value < 4096:
        raise ValueError('LLM_CONTEXT_TOKENS must be at least 4096')
    return value


def input_bound(messages):
    # Byte-level tokenizer upper bound, plus chat template/special-token reserve.
    # Deliberately conservative; no characters/token average for German text.
    return 512 + sum(len(m['content'].encode('utf-8')) + 32 for m in messages)


def fits(messages, output_tokens):
    return input_bound(messages) + output_tokens <= context_tokens()


def structured_output_budget(config, maximum, think_override=None):
    native = os.environ.get('LLM_OLLAMA_NATIVE', 'true' if config.uses_ollama else 'false').lower() == 'true'
    thinking = think_override if think_override is not None else config.reasoning_effort != 'none'
    return maximum * (2 if native and thinking else 1)


_INFERENCE_LOCK = threading.Lock()


def complete(client, config, **kwargs):
    # One backend process: PDF, agenda, and summary workers share the same slot.
    with _INFERENCE_LOCK:
        return _complete(client, config, **kwargs)


def _verify_native_context(config, model):
    import httpx
    response = httpx.get(config.base_url.removesuffix('/v1') + '/api/ps', timeout=10,
                         headers={'Authorization': 'Bearer ' + config.api_key})
    response.raise_for_status()
    normalize = lambda name: name if ':' in name else name + ':latest'
    loaded = [entry for entry in response.json().get('models', [])
              if normalize(entry.get('model', entry.get('name', ''))) == normalize(model)]
    if not loaded or not isinstance(loaded[0].get('context_length'), int):
        raise ContextBudgetError('Cannot verify effective Ollama context')
    actual = loaded[0]['context_length']
    if actual < context_tokens():
        raise ContextBudgetError('Ollama context smaller than requested')
    return actual


def _complete(client, config, **kwargs):
    think_override = kwargs.pop('ollama_think', None)
    if think_override is not None and type(think_override) is not bool:
        raise ValueError('ollama_think must be a boolean')
    messages = kwargs['messages']
    maximum = kwargs['max_tokens']
    if not fits(messages, maximum):
        raise ContextBudgetError('Input plus output exceeds configured context budget')
    native = os.environ.get('LLM_OLLAMA_NATIVE', 'true' if config.uses_ollama else 'false').lower() == 'true'
    if not native:
        response = client.chat.completions.create(**kwargs)
        if getattr(response.choices[0], 'finish_reason', None) == 'length':
            raise ContextBudgetError('LLM output truncated')
        return response
    import httpx
    payload = {
        'model': kwargs['model'], 'messages': messages, 'stream': False,
        'truncate': False, 'shift': False,
        'options': {'num_ctx': context_tokens(), 'num_predict': maximum,
                    'num_thread': int(os.environ.get('LLM_CPU_THREADS', '16')),
                    'temperature': kwargs.get('temperature', 0.1)},
    }
    effort = kwargs.get('reasoning_effort')
    if effort is not None:
        payload['think'] = False if effort == 'none' else effort
    if think_override is not None:
        payload['think'] = think_override
    response_format = kwargs.get('response_format', {})
    if response_format.get('type') == 'json_object':
        payload['format'] = 'json'
    elif response_format.get('type') == 'json_schema':
        payload['format'] = response_format['json_schema']['schema']
    # Ollama 0.34.1 applies the schema in a second generation phase after
    # thinking. num_predict bounds each phase, not their combined generation.
    passes = 2 if payload.get('format') and payload.get('think', True) is not False else 1
    reserved_output = maximum * passes
    if not fits(messages, reserved_output):
        raise ContextBudgetError('Input plus all provider generation phases exceeds context budget')
    response = httpx.post(config.base_url.removesuffix('/v1') + '/api/chat',
                          json=payload, timeout=kwargs.get('timeout', config.timeout_seconds),
                          headers={'Authorization': 'Bearer ' + config.api_key})
    response.raise_for_status()
    data = response.json()
    actual_context = _verify_native_context(config, data.get('model') or kwargs['model'])
    # Optional private diagnostic artifacts, never application logs.
    audit_dir = os.environ.get('LLM_AUDIT_DIR')
    if audit_dir:
        import time
        path = Path(audit_dir)
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        record = {'request': payload, 'response': data, 'input_token_bound': input_bound(messages),
                  'reserved_output_tokens': reserved_output, 'max_generation_passes': passes,
                  'verified_context_tokens': actual_context}
        target = path / (str(time.time_ns()) + '.json')
        with target.open('x', encoding='utf-8') as handle:
            os.chmod(target, 0o600)
            json.dump(record, handle, ensure_ascii=False)
    if not data.get('done') or data.get('done_reason') == 'length':
        raise ContextBudgetError('LLM output incomplete')
    count = data.get('prompt_eval_count')
    generated = data.get('eval_count', 0)
    if (not isinstance(count, int) or count > input_bound(messages) or count + reserved_output > context_tokens()
            or not isinstance(generated, int) or generated > reserved_output):
        raise ContextBudgetError('Unexpected provider context usage')
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=data['message'].get('content', '')), finish_reason='stop')])


def cache_read(key):
    directory = os.environ.get('LLM_CACHE_DIR')
    if not directory:
        return None
    path = Path(directory) / (hashlib.sha256(key.encode()).hexdigest() + '.json')
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        return None


def cache_write(key, value):
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


def cache_key(config, messages, purpose):
    return json.dumps([purpose, config.base_url, config.model, config.reasoning_effort,
                       context_tokens(), messages], ensure_ascii=False, sort_keys=True)
