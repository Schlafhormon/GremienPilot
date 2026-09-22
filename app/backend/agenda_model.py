"""Persisted, local-only TOP model settings; never mutate summary environment."""
from contextlib import contextmanager
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import threading
import hashlib

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator


class AgendaModelSettings(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    enabled: StrictBool = False
    model: str = Field(default='gemma4:31b-it-q4_K_M', min_length=1, max_length=200,
                       pattern=r'^[a-zA-Z0-9][a-zA-Z0-9_.:/-]*$')
    context_tokens: int = Field(default=32768, ge=8192, le=262144, strict=True)
    output_tokens: int = Field(default=4096, ge=1024, le=16384, strict=True)
    timeline_output_tokens: int = Field(default=4096, ge=1024, le=16384, strict=True)
    timeout_seconds: float = Field(default=1800, ge=10, le=7200, allow_inf_nan=False)
    cpu_threads: int = Field(default=8, ge=1, le=128, strict=True)
    temperature: float = Field(default=0.1, ge=0, le=2, allow_inf_nan=False)
    seed: int | None = Field(default=42, ge=0, le=2147483647, strict=True)
    thinking: StrictBool = False

    @model_validator(mode='after')
    def validate_budget(self):
        if 'cloud' in self.model.lower():
            raise ValueError('Nur lokale Ollama-Modelle sind für Sitzungsinhalte zulässig.')
        reserve = max(self.output_tokens, self.timeline_output_tokens) * (2 if self.thinking else 1)
        if reserve + 4096 >= self.context_tokens:
            raise ValueError('Kontext muss Ausgabe/Thinking plus mindestens 4096 Eingabetokens aufnehmen.')
        return self


_SETTINGS_LOCK = threading.RLock()
_last_error = None


class AgendaModelError(RuntimeError):
    """Public, content-free diagnostic; no fallback is permitted."""


def settings_path():
    from persistence import get_db_path
    return Path(os.environ.get('AGENDA_MODEL_SETTINGS_PATH', str(get_db_path().with_name('agenda-model.json'))))


def load_settings():
    with _SETTINGS_LOCK:
        try:
            return AgendaModelSettings.model_validate_json(settings_path().read_text(encoding='utf-8'))
        except FileNotFoundError:
            return AgendaModelSettings()


def save_settings(settings):
    global _last_error
    path = settings_path()
    with _SETTINGS_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as handle:
            handle.write(settings.model_dump_json(indent=2))
            handle.flush()
            os.fsync(handle.fileno())
            temporary = handle.name
        os.replace(temporary, path)
        _last_error = None
    return settings


def resolve_config(model=None, settings=None):
    from summarize import get_llm_config
    config = get_llm_config(model)
    settings = settings or load_settings()
    if not settings.enabled:
        return config
    if not config.uses_ollama:
        raise AgendaModelError('Das separate TOP-Modell benötigt einen lokalen Ollama-Endpunkt.')
    endpoint = os.environ.get('AGENDA_LLM_BASE_URL', '').strip().rstrip('/')
    if endpoint:
        from urllib.parse import urlparse
        parsed = urlparse(endpoint)
        if (parsed.scheme not in {'http', 'https'} or parsed.hostname not in
                {'localhost', '127.0.0.1', '::1', 'ollama', 'ollama-agenda'} or parsed.username or parsed.password):
            raise AgendaModelError('AGENDA_LLM_BASE_URL muss einen lokalen Ollama-Dienst ohne URL-Zugangsdaten benennen.')
        config = replace(config, base_url=endpoint, base_url_source='agenda_local_configured')
    if os.environ.get('LLM_OLLAMA_NATIVE', 'true').lower() != 'true':
        raise AgendaModelError('Das separate TOP-Modell benötigt die native lokale Ollama-API.')
    tokenizer = {}
    directory = Path(os.environ.get('AGENDA_TOKENIZER_DIR', str(settings_path().parent / 'tokenizers' / 'gemma4-31b')))
    if (directory / 'metadata.json').exists():
        metadata = json.loads((directory / 'metadata.json').read_text(encoding='utf-8-sig'))
        if metadata.get('model') == settings.model:
            path = directory / 'tokenizer.json'
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != metadata.get('sha256'):
                raise AgendaModelError('Lokaler TOP-Tokenizer stimmt nicht mit seiner Prüfsumme überein.')
            tokenizer = {'tokenizer_json': str(path), 'tokenizer_sha256': digest,
                         'tokenizer_model_digest': metadata['model_digest']}
    return replace(config, model=settings.model, context_budget=settings.context_tokens,
                   output_budget=settings.output_tokens, timeline_output_budget=settings.timeline_output_tokens,
                   timeout_seconds=settings.timeout_seconds, cpu_threads=settings.cpu_threads,
                   temperature=settings.temperature, seed=settings.seed,
                   reasoning_effort='low' if settings.thinking else 'none', task='agenda', **tokenizer)


def require_model(config):
    from llm_transport import model_fingerprint
    try:
        fingerprint = model_fingerprint(config)
    except Exception as exc:
        raise AgendaModelError(f'TOP-Modell {config.model}: lokaler Ollama-Dienst nicht erreichbar ({type(exc).__name__}).') from exc
    if not fingerprint.get('digest'):
        raise AgendaModelError(f'TOP-Modell {config.model} ist nicht lokal installiert. Kein Ersatzmodell verwendet.')
    if config.tokenizer_model_digest and config.tokenizer_model_digest != fingerprint['digest']:
        raise AgendaModelError('TOP-Modell-Digest hat sich geändert. Tokenizer-Zuordnung vor der Verarbeitung erneut prüfen.')
    return fingerprint


def report_error(config, exc):
    global _last_error
    message = str(exc) if isinstance(exc, AgendaModelError) else (
        f'TOP-Modell {config.model}: {type(exc).__name__}. Modellladung, RAM/VRAM und Zeitlimit prüfen; kein Ersatzmodell verwendet.')
    import httpx
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            provider_error = str(exc.response.json().get('error', '')).lower()
        except (ValueError, AttributeError):
            provider_error = ''
        if any(term in provider_error for term in ('memory', 'allocate', 'cuda', 'out of memory')):
            message = f'TOP-Modell {config.model} konnte wegen fehlender RAM-/VRAM-Ressourcen nicht ausgeführt werden. Kontextbudget oder Docker-/WSL-Speicherlimit prüfen. Kein Ersatzmodell verwendet.'
    _last_error = {'model': config.model, 'message': message}
    return message


def diagnostics():
    settings = load_settings()
    from summarize import get_llm_config
    summary = get_llm_config()
    result = {'settings': settings.model_dump(), 'effective_model': summary.model,
              'summary_model': summary.model, 'available': None, 'digest': None,
              'message': 'Bisherige TOP-Modellwahl aktiv.', 'last_error': _last_error}
    try:
        config = resolve_config(settings=settings.model_copy(update={'enabled': True}))
        result['effective_model'] = config.model if settings.enabled else summary.model
        result['top_endpoint'] = config.base_url
        result.update(require_model(config), available=True,
                      message='Modell lokal vorhanden. Ladbarkeit und Kontextspeicher erst durch Verarbeitung geprüft.')
        result['token_count_method'] = 'local_model_tokenizer_with_reserve' if config.tokenizer_json else 'conservative_utf8_bytes'
    except AgendaModelError as exc:
        result.update(available=False, message=str(exc))
    return result


@contextmanager
def model_session(config, usage):
    """One complete TOP pass owns the slot; cache-only passes do not unload Qwen."""
    if config.task != 'agenda':
        yield
        return
    from llm_transport import _INFERENCE_LOCK
    from gpu_resources import llm_gpu_slot, unload_ollama_model
    with _INFERENCE_LOCK, llm_gpu_slot(config):
        before = usage.attempted_calls
        try:
            yield
        finally:
            if usage.attempted_calls > before:
                try:
                    unload_ollama_model(config)
                except Exception:
                    from gpu_resources import block_after_release_failure
                    block_after_release_failure()
                    raise
