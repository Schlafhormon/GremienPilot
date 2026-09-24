"""Validated, immutable per-request model settings. No machine-specific defaults."""
import hashlib
import json
import math
import os
import re
import inspect
from contextvars import ContextVar
from functools import wraps
from dataclasses import asdict, dataclass, field
from urllib.parse import urlparse
import httpx

LOCAL_OLLAMA_BASE_URL = "http://localhost:11434/v1"
DOCKER_OLLAMA_BASE_URL = "http://ollama:11434/v1"
LOCAL_LLM_HOSTS = {"localhost", "127.0.0.1", "::1"}
INTERNAL_LLM_HOSTS = {"ollama"}

def _is_truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def is_docker_runtime() -> bool:
    """Return whether the backend is running inside the Compose/container setup."""

    app_runtime = (os.environ.get("APP_RUNTIME") or "").strip().lower()
    return (
        app_runtime == "docker"
        or _is_truthy(os.environ.get("RUNNING_IN_DOCKER"))
        or os.path.exists("/.dockerenv")
    )


def _base_url_host(base_url: str) -> str:
    parsed = urlparse(base_url)
    return (parsed.hostname or "").lower()


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def resolve_llm_base_url(raw_base_url: str | None = None) -> tuple[str, str]:
    """
    Resolve the effective LLM endpoint and describe where it came from.

    A copied root .env used to contain LLM_BASE_URL=http://localhost:11434/v1,
    which is correct for local backend development but wrong inside Docker.
    In Docker, localhost points at the backend container, so local Ollama values
    are treated as the internal Compose default unless a non-local URL is set.
    """

    if raw_base_url is None:
        raw_base_url = os.environ.get("LLM_BASE_URL")

    configured = (raw_base_url or "").strip()
    docker_runtime = is_docker_runtime()

    if not configured:
        if docker_runtime:
            return DOCKER_OLLAMA_BASE_URL, "internal_docker_default"
        return LOCAL_OLLAMA_BASE_URL, "local_development_default"

    normalized = _normalize_base_url(configured)
    host = _base_url_host(normalized)
    if docker_runtime and host in LOCAL_LLM_HOSTS:
        return DOCKER_OLLAMA_BASE_URL, "internal_docker_default_from_local_value"
    if host in INTERNAL_LLM_HOSTS:
        return normalized, "internal_configured"
    if host in LOCAL_LLM_HOSTS:
        return normalized, "local_development_configured"
    return normalized, "external_configured"


class ModelConfigurationError(ValueError):
    pass


def _number(name, default, *, integer=False, minimum=0):
    raw = os.environ.get(name, '').strip() or str(default)
    try:
        value = int(raw) if integer else float(raw)
    except ValueError as exc:
        raise ModelConfigurationError(f'{name} must be numeric') from exc
    if not math.isfinite(value) or value < minimum:
        raise ModelConfigurationError(f'{name} must be finite and >= {minimum}')
    return value


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    model: str
    api_key: str = field(repr=False)
    timeout_seconds: float = 120  # Legacy alias: read/inactivity, never total.
    base_url_source: str = 'configured'
    reasoning_effort: str | None = None
    provider: str = 'openai-compatible'
    model_source: str = 'environment'
    ollama_endpoint: bool = False
    context_tokens: int = 16384
    thinking: bool | None = None
    thinking_tokens: int = 0  # Reservation; provider may only expose a combined cap.
    output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    cpu_threads: int | None = None
    gpu_layers: int | None = None
    keep_alive: str = '5m'
    connect_seconds: float = 10
    load_seconds: float = 1800
    total_seconds: float = 0
    max_retries: int = 2
    retry_backoff_seconds: float = 0.5
    image_tokens: int = 0  # Unknown vision tokenization must be configured explicitly.
    tokenizer_path: str = ''
    tokenizer_model: str = ''
    model_revision: str = ''
    output_parameter: str = 'max_tokens'

    def __post_init__(self):
        for name in ('context_tokens', 'timeout_seconds', 'connect_seconds', 'total_seconds',
                     'max_retries', 'retry_backoff_seconds', 'thinking_tokens', 'image_tokens', 'load_seconds'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ModelConfigurationError(f'{name} must be nonnegative and finite')
        url = urlparse(self.base_url)
        if url.scheme not in {'http', 'https'} or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ModelConfigurationError('LLM_BASE_URL must be an HTTP(S) URL without credentials/query')
        if not self.model.strip() or self.provider not in {'ollama', 'openai-compatible'}:
            raise ModelConfigurationError('Invalid LLM_MODEL or LLM_PROVIDER')
        if self.reasoning_effort not in {None, 'none', 'low', 'medium', 'high', 'max'}:
            raise ModelConfigurationError('Invalid LLM_REASONING_EFFORT')
        if self.context_tokens < 4096 or self.timeout_seconds <= 0 or self.connect_seconds <= 0:
            raise ModelConfigurationError('Context >= 4096 and positive connect/read timeouts required')
        if self.thinking is not None and self.reasoning_effort is not None:
            raise ModelConfigurationError('Set LLM_THINKING or LLM_REASONING_EFFORT, not both')
        if self.provider == 'openai-compatible' and self.thinking is not None:
            raise ModelConfigurationError('Use LLM_REASONING_EFFORT for OpenAI-compatible providers')
        if self.provider == 'openai-compatible' and self.top_k is not None:
            raise ModelConfigurationError('LLM_TOP_K requires native Ollama')
        if self.output_parameter not in {'max_tokens', 'max_completion_tokens'}:
            raise ModelConfigurationError('Invalid LLM_OUTPUT_PARAMETER')
        if self.thinking_tokens and self.provider == 'openai-compatible' and self.output_parameter != 'max_completion_tokens':
            raise ModelConfigurationError('Thinking reservation requires max_completion_tokens')
        if not re.fullmatch(r'-?\d+(?:\.\d+)?(?:ms|s|m|h)?', self.keep_alive):
            raise ModelConfigurationError('Invalid LLM_KEEP_ALIVE duration')
        if self.provider == 'ollama' and re.fullmatch(r'0+(?:\.0+)?(?:ms|s|m|h)?', self.keep_alive):
            raise ModelConfigurationError('Native keep_alive must allow post-generation context verification (not 0)')
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ModelConfigurationError('LLM_TOP_P must be in (0, 1]')
        if self.temperature is not None and not 0 <= self.temperature <= 2:
            raise ModelConfigurationError('LLM_TEMPERATURE must be in [0, 2]')
        if self.tokenizer_path and self.tokenizer_model != self.model:
            raise ModelConfigurationError('LLM_TOKENIZER_MODEL must match the effective model (including overrides)')

    @property
    def reasoning_options(self):
        return {} if self.reasoning_effort is None else {'reasoning_effort': self.reasoning_effort}

    @property
    def uses_ollama(self):
        return self.provider == 'ollama' or self.ollama_endpoint

    @property
    def uses_internal_ollama(self):
        return self.uses_ollama and _base_url_host(self.base_url) in INTERNAL_LLM_HOSTS

    @property
    def uses_local_ollama(self):
        return self.uses_ollama and _base_url_host(self.base_url) in LOCAL_LLM_HOSTS

    @property
    def http_timeout(self):
        return httpx.Timeout(self.timeout_seconds, connect=self.connect_seconds)

    def output_budget(self, requested):
        return self.output_tokens or requested

    def public_snapshot(self):
        data = asdict(self)
        data.pop('api_key')
        data['config_id'] = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        return data


_CURRENT = ContextVar('llm_config', default=None)


def configured(function):
    """Freeze environment resolution throughout one domain operation."""
    signature = inspect.signature(function)
    @wraps(function)
    def wrapped(*args, **kwargs):
        from processing_mode import processing_scope
        arguments = signature.bind(*args, **kwargs).arguments
        with processing_scope(arguments.get('processing_mode')):
            config = get_llm_config(arguments.get('model'))
            token = _CURRENT.set(config)
            try:
                return function(*args, **kwargs)
            finally:
                _CURRENT.reset(token)
    return wrapped


def get_llm_config(model=None, *, resolved=None):
    active = _CURRENT.get()
    if active is not None and (not model or model == active.model):
        return active
    base, source = resolved or resolve_llm_base_url()
    provider = os.environ.get('LLM_PROVIDER', '').strip()
    native = os.environ.get('LLM_OLLAMA_NATIVE', '').strip().lower()
    if native not in {'', 'true', 'false'}:
        raise ModelConfigurationError('LLM_OLLAMA_NATIVE must be true or false')
    if not provider:
        provider = ('ollama' if native == 'true' else 'openai-compatible') if native else (
            'ollama' if _base_url_host(base) in LOCAL_LLM_HOSTS | INTERNAL_LLM_HOSTS else 'openai-compatible')
    think = os.environ.get('LLM_THINKING', '').strip().lower()
    if think not in {'', 'true', 'false'}:
        raise ModelConfigurationError('LLM_THINKING must be empty, true or false')
    def optional(name, *, integer=False, minimum=0):
        return _number(name, 0, integer=integer, minimum=minimum) if os.environ.get(name, '').strip() else None
    return LLMConfig(
        base_url=base, base_url_source=source, provider=provider,
        model=model if model else os.environ.get('LLM_MODEL', 'gemma4:31b-it-q4_K_M'),
        model_source='request' if model else 'environment',
        ollama_endpoint=not os.environ.get("LLM_PROVIDER", "").strip() and _base_url_host(base) in LOCAL_LLM_HOSTS | INTERNAL_LLM_HOSTS,
        api_key=os.environ.get('LLM_API_KEY', 'ollama'),
        reasoning_effort=os.environ.get('LLM_REASONING_EFFORT', '').strip().lower() or None,
        thinking={'true': True, 'false': False}.get(think),
        context_tokens=_number('LLM_CONTEXT_TOKENS', 16384, integer=True, minimum=4096),
        output_tokens=optional('LLM_OUTPUT_TOKENS', integer=True, minimum=1),
        thinking_tokens=_number('LLM_THINKING_TOKENS', 0, integer=True),
        temperature=optional('LLM_TEMPERATURE'), top_p=optional('LLM_TOP_P'),
        top_k=optional('LLM_TOP_K', integer=True, minimum=1), seed=optional('LLM_SEED', integer=True),
        cpu_threads=optional('LLM_CPU_THREADS', integer=True, minimum=1),
        gpu_layers=optional('LLM_GPU_LAYERS', integer=True),
        keep_alive=os.environ.get('LLM_KEEP_ALIVE') or os.environ.get('OLLAMA_KEEP_ALIVE', '5m'),
        connect_seconds=_number('LLM_CONNECT_TIMEOUT_SECONDS', 10, minimum=0.001),
        timeout_seconds=_number('LLM_READ_TIMEOUT_SECONDS', os.environ.get('LLM_TIMEOUT_SECONDS') or 120, minimum=0.001),
        load_seconds=_number('LLM_LOAD_TIMEOUT_SECONDS', 1800, minimum=0.001),
        total_seconds=_number('LLM_TOTAL_TIMEOUT_SECONDS', 0),
        max_retries=_number('LLM_MAX_RETRIES', 2, integer=True),
        retry_backoff_seconds=_number('LLM_RETRY_BACKOFF_SECONDS', 0.5),
        image_tokens=_number('LLM_IMAGE_TOKENS', 0, integer=True),
        tokenizer_path=os.environ.get('LLM_TOKENIZER_PATH', ''),
        tokenizer_model=os.environ.get('LLM_TOKENIZER_MODEL', ''),
        model_revision=os.environ.get('LLM_MODEL_REVISION', ''),
        output_parameter=os.environ.get('LLM_OUTPUT_PARAMETER') or 'max_tokens',
    )
