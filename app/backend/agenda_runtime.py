"""Interruptible native streaming for long local agenda calls only.

The first response includes loading AND prompt evaluation: Ollama does not expose
their boundary while a request is pending. Only final metrics separate them.
"""
import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
import json
import time

import httpx

_observer = ContextVar('agenda_observer', default=None)


class OperationCancelled(Exception):
    """A user stop is propagated, never treated as a retryable model failure."""


class AgendaDeadlineError(TimeoutError):
    pass


class AgendaFirstResponseTimeout(AgendaDeadlineError):
    pass


class AgendaInactivityTimeout(AgendaDeadlineError):
    pass


class AgendaTotalTimeout(AgendaDeadlineError):
    pass


class AgendaStreamIncomplete(RuntimeError):
    pass


@contextmanager
def observe(callback):
    token = _observer.set(callback)
    try:
        yield
    finally:
        _observer.reset(token)


async def read_stream(config, payload, client_factory=httpx.AsyncClient):
    began = last_activity = time.monotonic()
    content, thinking = [], []
    received = 0
    phase = 'loading_or_prompt'
    callback = _observer.get()
    scope = {}
    try:
        body = json.loads(payload.get('messages', [])[-1]['content'])
        if isinstance(body, dict):
            scope = {k: body[k] for k in ('target_start', 'target_end') if type(body.get(k)) is int}
            scope['step'] = body.get('phase', 'line_assignment_or_review')
    except (IndexError, KeyError, ValueError):
        pass

    def progress():
        if callback:
            callback({**scope, 'phase': phase, 'model': config.model,
                      'elapsed_seconds': round(time.monotonic() - began),
                      'idle_seconds': round(time.monotonic() - last_activity),
                      'response_chunks': received,
                      'output_characters': sum(map(len, content)),
                      'total_limit_seconds': config.total_timeout_seconds})

    async def wait_for_operation(awaitable):
        pending = asyncio.ensure_future(awaitable)
        try:
            while True:
                progress()  # Also checks cancellation, even during prefill.
                now = time.monotonic()
                if now - began >= config.total_timeout_seconds:
                    raise AgendaTotalTimeout('TOP-Gesamtzeitlimit pro Aufruf erreicht')
                limit = config.idle_timeout_seconds if received else config.timeout_seconds
                if now - last_activity >= limit:
                    error = AgendaInactivityTimeout if received else AgendaFirstResponseTimeout
                    raise error('TOP-Stream inaktiv' if received else 'TOP-Modellladen/Promptverarbeitung ohne erste Antwort')
                done, _ = await asyncio.wait({pending}, timeout=min(1, limit, config.total_timeout_seconds))
                if done:
                    return pending.result()
        finally:
            if not pending.done():
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)

    timeout = httpx.Timeout(connect=config.connect_timeout_seconds, read=None,
                            write=config.connect_timeout_seconds, pool=config.connect_timeout_seconds)
    async with client_factory(timeout=timeout) as client:
        request = client.build_request('POST', config.base_url.removesuffix('/v1') + '/api/chat',
                                       json=dict(payload, stream=True),
                                       headers={'Authorization': 'Bearer ' + config.api_key})
        response = await wait_for_operation(client.send(request, stream=True))
        try:
            if response.is_error:
                await wait_for_operation(response.aread())
            response.raise_for_status()
            lines = response.aiter_lines().__aiter__()
            while True:
                try:
                    line = await wait_for_operation(anext(lines))
                except StopAsyncIteration:
                    raise AgendaStreamIncomplete('TOP-Stream beendet ohne done=true') from None
                if not line.strip():
                    continue
                data = json.loads(line)
                if data.get('error'):
                    raise AgendaStreamIncomplete('Ollama meldet einen Streamfehler')
                message = data.get('message') or {}
                piece, thought = message.get('content', ''), message.get('thinking', '')
                if piece or thought or data.get('done'):
                    received += 1
                    last_activity = time.monotonic()
                content.append(piece)
                thinking.append(thought)
                phase = 'generating'
                if data.get('done') is True:
                    data['message'] = dict(message, content=''.join(content), thinking=''.join(thinking))
                    phase = 'validating'
                    progress()
                    return data
        finally:
            await response.aclose()


def complete_stream(config, payload):
    return asyncio.run(read_stream(config, payload))
