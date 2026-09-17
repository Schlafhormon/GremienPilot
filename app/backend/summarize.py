"""
Summarization module for generating meeting minutes per TOP.

Uses an OpenAI-compatible API, typically Ollama, for local German
summarization. The public API still exposes an editable text summary, while
the backend internally works with structured minutes fields.

Configuration via environment variables:
- LLM_BASE_URL: API endpoint (local default: http://localhost:11434/v1,
  Docker default: http://ollama:11434/v1)
- LLM_MODEL: Model name (default: qwen3:8b)
- LLM_REASONING_EFFORT: empty for server default, none to disable, or low/medium/high/max
- LLM_TIMEOUT_SECONDS: request timeout per LLM call (default: 120)
- LLM_MAX_RETRIES: retry count for transient LLM errors (default: 2)
- LLM_CHUNK_CHARS: target chunk size for long TOP transcripts (default: 12000)
- LLM_STRUCTURED_FALLBACK: free-text fallback on structured failure (default: true)
"""

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional
from urllib.parse import urlparse

# LLM server configuration (Ollama)
LOCAL_OLLAMA_BASE_URL = "http://localhost:11434/v1"
DOCKER_OLLAMA_BASE_URL = "http://ollama:11434/v1"
LOCAL_LLM_HOSTS = {"localhost", "127.0.0.1", "::1"}
INTERNAL_LLM_HOSTS = {"ollama"}

LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3:8b")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "ollama")
LLM_TIMEOUT_SECONDS = float(os.environ.get("LLM_TIMEOUT_SECONDS", "120"))
LLM_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "2"))
LLM_RETRY_BACKOFF_SECONDS = float(os.environ.get("LLM_RETRY_BACKOFF_SECONDS", "0.5"))
LLM_CHUNK_CHARS = int(os.environ.get("LLM_CHUNK_CHARS", "12000"))
LLM_STRUCTURED_FALLBACK = (
    os.environ.get("LLM_STRUCTURED_FALLBACK", "true").lower() != "false"
)


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


LLM_BASE_URL, LLM_BASE_URL_SOURCE = resolve_llm_base_url()


@dataclass(frozen=True)
class LLMConfig:
    """Effective LLM configuration for one request."""

    base_url: str
    model: str
    api_key: str
    timeout_seconds: float
    base_url_source: str
    reasoning_effort: str | None = None

    @property
    def reasoning_options(self) -> dict[str, str]:
        """Omit the API field entirely for servers without reasoning support."""
        if self.reasoning_effort is None:
            return {}
        return {"reasoning_effort": self.reasoning_effort}

    @property
    def uses_internal_ollama(self) -> bool:
        return _base_url_host(self.base_url) in INTERNAL_LLM_HOSTS

    @property
    def uses_local_ollama(self) -> bool:
        return _base_url_host(self.base_url) in LOCAL_LLM_HOSTS

    @property
    def uses_ollama(self) -> bool:
        return self.uses_internal_ollama or self.uses_local_ollama


@dataclass
class LLMAvailability:
    """Diagnostics result for the configured LLM endpoint and model."""

    ok: bool
    base_url: str
    model: str
    base_url_source: str
    service_reachable: bool
    model_available: bool
    available_models: list[str] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_llm_config(model: str | None = None) -> LLMConfig:
    base_url, source = resolve_llm_base_url()
    reasoning_effort = os.environ.get("LLM_REASONING_EFFORT", "").strip().lower()
    if reasoning_effort not in {"", "none", "low", "medium", "high", "max"}:
        raise ValueError(
            "LLM_REASONING_EFFORT must be empty, none, low, medium, high or max"
        )
    return LLMConfig(
        base_url=base_url,
        model=model or os.environ.get("LLM_MODEL", LLM_MODEL),
        api_key=os.environ.get("LLM_API_KEY", LLM_API_KEY),
        timeout_seconds=float(os.environ.get("LLM_TIMEOUT_SECONDS", str(LLM_TIMEOUT_SECONDS))),
        base_url_source=source,
        reasoning_effort=reasoning_effort or None,
    )


@dataclass
class StructuredSummary:
    """Structured internal representation of one TOP summary."""

    discussion: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    votes: list[str] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)
    open_points: list[str] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, list[str]]:
        return asdict(self)


@dataclass
class SummarizationResult:
    """Result from summarization including timing and internal structure."""

    summary: str
    duration_seconds: float
    structured: StructuredSummary | None = None
    fallback_used: bool = False
    chunks_processed: int = 1


@dataclass
class SummarySourceLink:
    """Reviewable link from one structured summary item to transcript evidence."""

    section: str
    item_index: int
    item_text: str
    line_indices: list[int] = field(default_factory=list)
    start: float | None = None
    end: float | None = None
    excerpt: str = ""
    confidence: float = 0.0
    missing_source: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SummaryReviewWarning:
    """Warning shown when a summary item or transcript signal needs review."""

    kind: str
    message: str
    severity: str = "warning"
    keyword: str | None = None
    section: str | None = None
    item_index: int | None = None
    line_indices: list[int] = field(default_factory=list)
    start: float | None = None
    end: float | None = None
    excerpt: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SummaryReview:
    """Source links and warning signals for a generated summary."""

    source_links: list[SummarySourceLink] = field(default_factory=list)
    warnings: list[SummaryReviewWarning] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_links": [link.to_dict() for link in self.source_links],
            "warnings": [warning.to_dict() for warning in self.warnings],
        }


@dataclass
class LLMErrorInfo:
    """Classified LLM error metadata for retries and API diagnostics."""

    category: str
    transient: bool


class LLMCallError(RuntimeError):
    """Raised when an LLM call failed after classification/retries."""

    def __init__(self, message: str, *, category: str, transient: bool) -> None:
        super().__init__(message)
        self.category = category
        self.transient = transient


class StructuredOutputError(ValueError):
    """Raised when the model did not return usable structured JSON."""


# Default system prompt for structured municipal meeting summarization.
DEFAULT_SYSTEM_PROMPT = """Du bist ein Experte für Niederschriften deutscher kommunaler Gremien
(Rat, Ausschuss, Bezirksvertretung, Ortsbeirat).

Arbeite protokollarisch, sachlich und verwaltungsnah:
- keine wörtlichen Zitate, sondern präzise Paraphrasen
- dritte Person und formale Verwaltungssprache
- keine Ausschmückungen, keine rechtliche Bewertung über das Transkript hinaus
- Beschlüsse, Abstimmungen und Aufträge nur aufnehmen, wenn sie aus dem Transkript hervorgehen
- fehlende oder unklare Informationen ausdrücklich unter "uncertainties" markieren
- Geschäftsordnungs- und Technikdetails nur aufnehmen, wenn sie für den TOP relevant sind

Gib ausschließlich valides JSON mit genau diesen Schlüsseln zurück:
{
  "discussion": ["wesentliche Diskussionspunkte, Sachverhalte, Argumente und Positionen"],
  "decisions": ["Beschlüsse oder Einigungen"],
  "votes": ["Abstimmungsergebnisse mit Stimmenzahlen, Enthaltungen oder Einstimmigkeit"],
  "action_items": ["vereinbarte Maßnahmen, Prüfaufträge, Zuständigkeiten oder Fristen"],
  "open_points": ["offene Fragen, weiterer Beratungsbedarf oder Vertagungen"],
  "uncertainties": ["fachlich relevante Unsicherheiten der Auswertung"]
}

Jeder Wert ist eine Liste kurzer, vollständiger deutscher Sätze. Wenn eine
Kategorie im Transkript nicht vorkommt, nutze eine leere Liste. Keine Markdown-
Formatierung und kein Text außerhalb des JSON-Objekts."""


FREETEXT_SYSTEM_PROMPT = """Du bist ein Experte für die Erstellung von Sitzungsprotokollen
für deutsche Kommunalverwaltungen.

Erstelle aus dem Transkript eines Tagesordnungspunktes eine fachlich präzise
Zusammenfassung im Stil einer offiziellen Niederschrift.

STIL:
- Formale Verwaltungssprache, dritte Person
- Paraphrasieren statt wörtlich zitieren
- Direkt mit Inhalt beginnen, keine Einleitung

INHALT:
- Wesentliche Diskussionspunkte und Argumente
- Getroffene Beschlüsse und erkennbare Abstimmungsergebnisse
- Wichtige Positionen der Teilnehmenden
- Vereinbarte Maßnahmen, Prüfaufträge, offene Punkte und Unsicherheiten

IGNORIEREN:
- Füllwörter, Versprecher, triviale Zwischenbemerkungen
- Mikrofon-, Redezeit- und Technikdetails ohne fachliche Relevanz

FORMAT:
- 2 bis 5 knappe Absätze
- NUR Fließtext, KEINE Markdown-Formatierung"""


STRUCTURED_KEYS = {
    "discussion",
    "decisions",
    "votes",
    "action_items",
    "open_points",
    "uncertainties",
}

KEY_ALIASES = {
    "diskussion": "discussion",
    "beschluss": "decisions",
    "beschluesse": "decisions",
    "beschlüsse": "decisions",
    "abstimmung": "votes",
    "abstimmungen": "votes",
    "massnahmen": "action_items",
    "maßnahmen": "action_items",
    "offene_punkte": "open_points",
    "unsicherheiten": "uncertainties",
}


SECTION_ITEM_ACCESSORS = {
    "discussion": lambda structured: structured.discussion,
    "decisions": lambda structured: structured.decisions,
    "votes": lambda structured: structured.votes,
    "action_items": lambda structured: structured.action_items,
    "open_points": lambda structured: structured.open_points,
    "uncertainties": lambda structured: structured.uncertainties,
}

SOURCE_STOPWORDS = {
    "aber",
    "alle",
    "als",
    "auch",
    "auf",
    "aus",
    "bei",
    "das",
    "dem",
    "den",
    "der",
    "des",
    "die",
    "ein",
    "eine",
    "einem",
    "einen",
    "einer",
    "es",
    "fuer",
    "für",
    "hat",
    "im",
    "in",
    "ist",
    "mit",
    "nicht",
    "oder",
    "sich",
    "sie",
    "und",
    "von",
    "wird",
    "wurde",
    "zu",
    "zum",
    "zur",
}

DECISION_SIGNAL_TERMS = {
    "beschlossen",
    "beschluss",
    "beschließen",
    "beschliessen",
    "einstimmig",
    "enthaltung",
    "enthaltungen",
    "abgelehnt",
}

SECTION_KEYWORD_BOOSTS = {
    "decisions": {"beschluss", "beschlossen", "beschließen", "beschliessen"},
    "votes": {"abstimmung", "einstimmig", "stimmen", "enthaltung", "enthaltungen"},
    "action_items": {"auftrag", "prüfen", "pruefen", "maßnahme", "massnahme"},
    "open_points": {"offen", "vertagt", "nachreichen", "klären", "klaeren"},
}


def build_structured_system_prompt(system_prompt: str | None) -> str:
    """Keep the structured JSON contract even when the UI sends legacy prompts."""

    custom_prompt = (system_prompt or "").strip()
    if not custom_prompt or custom_prompt == DEFAULT_SYSTEM_PROMPT.strip():
        return DEFAULT_SYSTEM_PROMPT

    return (
        DEFAULT_SYSTEM_PROMPT
        + "\n\nZusätzliche fachliche Vorgaben des Nutzers. Diese Vorgaben nur "
        "anwenden, soweit sie dem JSON-Schema und der strukturierten Ausgabe "
        "oben nicht widersprechen; das JSON-Ausgabeformat hat Vorrang:\n"
        + custom_prompt
    )


def _load_openai_client(config: LLMConfig | None = None) -> Any:
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError(
            "OpenAI client nicht installiert. Installieren Sie mit: uv add openai"
        )

    config = config or get_llm_config()
    return OpenAI(
        base_url=config.base_url,
        api_key=config.api_key,
        timeout=config.timeout_seconds,
        # The summary retry loop owns retries; do not multiply them in the SDK.
        max_retries=0,
    )


def classify_llm_error(error: Exception) -> LLMErrorInfo:
    """Classify common OpenAI-compatible client errors."""

    status_code = getattr(error, "status_code", None)
    name = error.__class__.__name__.lower()
    message = str(error).lower()

    if "timeout" in name or "timed out" in message or "timeout" in message:
        return LLMErrorInfo("timeout", True)
    if "rate" in name or status_code == 429:
        return LLMErrorInfo("rate_limit", True)
    if "connection" in name or "network" in name or "connect" in message:
        return LLMErrorInfo("network", True)
    if status_code in {408, 409, 425} or (
        isinstance(status_code, int) and status_code >= 500
    ):
        return LLMErrorInfo("server", True)
    if isinstance(status_code, int) and 400 <= status_code < 500:
        return LLMErrorInfo("client", False)
    return LLMErrorInfo("unknown", False)


def _extract_model_ids(models_response: Any) -> list[str]:
    raw_models = getattr(models_response, "data", models_response)
    if raw_models is None:
        return []

    model_ids = []
    for item in raw_models:
        if isinstance(item, dict):
            model_id = item.get("id") or item.get("name")
        else:
            model_id = getattr(item, "id", None) or getattr(item, "name", None)
        if model_id:
            model_ids.append(str(model_id))
    return sorted(set(model_ids))


def _model_matches_configured(available_model: str, configured_model: str) -> bool:
    if available_model == configured_model:
        return True
    # Accept untagged configuration only when the endpoint returns a tagged ID.
    if ":" not in configured_model and available_model.split(":", 1)[0] == configured_model:
        return True
    return False


def _llm_hint(config: LLMConfig, *, model_missing: bool = False) -> str:
    if config.uses_internal_ollama:
        pull_hint = (
            f" Starten Sie Ollama mit Docker Compose und laden Sie das Modell: "
            f"docker compose exec ollama ollama pull {config.model}."
        )
    elif config.uses_local_ollama:
        pull_hint = (
            f" Starten Sie lokal Ollama und laden Sie das Modell: "
            f"ollama pull {config.model}."
        )
    else:
        pull_hint = " Prüfen Sie die externe OpenAI-kompatible LLM-Konfiguration."

    model_hint = (
        f" Prüfen Sie LLM_MODEL={config.model}."
        if model_missing
        else f" Prüfen Sie LLM_BASE_URL={config.base_url} und LLM_MODEL={config.model}."
    )
    return pull_hint + model_hint


def check_llm_availability(
    *,
    client: Any | None = None,
    model: str | None = None,
) -> LLMAvailability:
    """Check whether the configured LLM endpoint is reachable and has the model."""

    config = get_llm_config(model)
    client = client or _load_openai_client(config)

    try:
        models_response = client.models.list()
    except Exception as error:
        info = classify_llm_error(error)
        message = (
            f"Der konfigurierte LLM-Dienst ist nicht erreichbar "
            f"(LLM_BASE_URL={config.base_url}, LLM_MODEL={config.model})."
            f"{_llm_hint(config)}"
        )
        raise LLMCallError(
            message,
            category=info.category if info.category != "unknown" else "network",
            transient=True,
        ) from error

    available_models = _extract_model_ids(models_response)
    model_available = any(
        _model_matches_configured(available_model, config.model)
        for available_model in available_models
    )
    if not model_available:
        message = (
            f"Das konfigurierte LLM-Modell '{config.model}' ist unter "
            f"LLM_BASE_URL={config.base_url} nicht verfügbar."
            f"{_llm_hint(config, model_missing=True)}"
        )
        raise LLMCallError(
            message,
            category="model_missing",
            transient=False,
        )

    return LLMAvailability(
        ok=True,
        base_url=config.base_url,
        model=config.model,
        base_url_source=config.base_url_source,
        service_reachable=True,
        model_available=True,
        available_models=available_models,
        message="LLM-Dienst ist erreichbar und das Modell ist verfügbar.",
    )


def llm_diagnostics(model: str | None = None) -> LLMAvailability:
    """Return diagnostics without raising for API endpoints and tests."""

    config = get_llm_config(model)
    try:
        return check_llm_availability(model=model)
    except LLMCallError as error:
        return LLMAvailability(
            ok=False,
            base_url=config.base_url,
            model=config.model,
            base_url_source=config.base_url_source,
            service_reachable=error.category == "model_missing",
            model_available=False,
            message=str(error),
        )


def _chat_completion_content(
    client: Any,
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
) -> str:
    reasoning_options = get_llm_config(model).reasoning_options
    last_error: Exception | None = None
    last_info = LLMErrorInfo("unknown", False)

    for attempt in range(LLM_MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=LLM_TIMEOUT_SECONDS,
                **reasoning_options,
            )
            content = response.choices[0].message.content or ""
            if not content.strip():
                raise LLMCallError(
                    "Leere Antwort des LLM",
                    category="empty_response",
                    transient=False,
                )
            return content.strip()
        except LLMCallError:
            raise
        except Exception as error:
            last_error = error
            last_info = classify_llm_error(error)
            if not last_info.transient or attempt >= LLM_MAX_RETRIES:
                break
            time.sleep(LLM_RETRY_BACKOFF_SECONDS * (2**attempt))

    hint = ""
    if last_info.category == "network":
        hint = (
            f" Prüfen Sie, ob der LLM-Dienst erreichbar ist "
            f"(LLM_BASE_URL={get_llm_config(model).base_url})."
        )
    raise LLMCallError(
        f"LLM-Aufruf fehlgeschlagen ({last_info.category}): {last_error}.{hint}",
        category=last_info.category,
        transient=last_info.transient,
    )


def _extract_json_object(content: str) -> dict[str, Any]:
    stripped = content.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", stripped, re.DOTALL | re.I)
    if fence_match:
        stripped = fence_match.group(1).strip()

    if not stripped.startswith("{"):
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            stripped = stripped[start : end + 1]

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as error:
        raise StructuredOutputError(f"Keine valide JSON-Antwort: {error}") from error

    if not isinstance(parsed, dict):
        raise StructuredOutputError("Strukturierte Antwort ist kein JSON-Objekt")
    return parsed


def _normalize_items(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []

    items = []
    for item in value:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            items.append(text)
    return items


def parse_structured_summary(content: str) -> StructuredSummary:
    parsed = _extract_json_object(content)
    normalized: dict[str, list[str]] = {key: [] for key in STRUCTURED_KEYS}

    for raw_key, raw_value in parsed.items():
        key = str(raw_key).strip()
        normalized_key = KEY_ALIASES.get(key.lower(), key)
        if normalized_key in STRUCTURED_KEYS:
            normalized[normalized_key] = _normalize_items(raw_value)

    if not any(normalized.values()):
        raise StructuredOutputError("Strukturierte Antwort enthält keine Inhalte")

    return StructuredSummary(**normalized)


def render_structured_summary(structured: StructuredSummary) -> str:
    """Render structured minutes into editable text for existing users."""

    sections = [
        ("Diskussion", structured.discussion),
        ("Beschluss", structured.decisions),
        ("Abstimmung", structured.votes),
        ("Maßnahmen/offene Punkte", structured.action_items + structured.open_points),
        ("Unsicherheiten", structured.uncertainties),
    ]
    rendered_sections = []
    for title, items in sections:
        clean_items = [item.strip() for item in items if item.strip()]
        if not clean_items:
            continue
        rendered_sections.append(f"{title}:\n" + "\n".join(clean_items))
    return "\n\n".join(rendered_sections).strip()


def _line_value(line: Any, key: str, default: Any = None) -> Any:
    if isinstance(line, dict):
        return line.get(key, default)
    return getattr(line, key, default)


def _normalize_for_review(text: str) -> str:
    normalized = text.lower()
    normalized = normalized.replace("ß", "ss")
    normalized = normalized.replace("ä", "ae")
    normalized = normalized.replace("ö", "oe")
    normalized = normalized.replace("ü", "ue")
    return normalized


def _review_tokens(text: str) -> set[str]:
    normalized = _normalize_for_review(text)
    tokens = set(re.findall(r"[a-z0-9_]{4,}", normalized))
    return {token for token in tokens if token not in SOURCE_STOPWORDS}


def _line_text(line: Any) -> str:
    speaker = str(_line_value(line, "speaker", "") or "").strip()
    text = str(_line_value(line, "text", "") or "").strip()
    return f"{speaker}: {text}" if speaker else text


def _line_time(line: Any, key: str) -> float | None:
    value = _line_value(line, key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _match_summary_item_to_lines(
    item_text: str,
    lines: list[Any],
    *,
    section: str,
) -> tuple[list[int], float]:
    item_tokens = _review_tokens(item_text)
    if not item_tokens or not lines:
        return [], 0.0

    boosts = SECTION_KEYWORD_BOOSTS.get(section, set())
    scored_lines: list[tuple[float, int]] = []
    for index, line in enumerate(lines):
        line_tokens = _review_tokens(_line_text(line))
        if not line_tokens:
            continue

        overlap = item_tokens & line_tokens
        if not overlap:
            score = 0.0
        else:
            score = len(overlap) / max(len(item_tokens), 1)

        boost_overlap = boosts & line_tokens
        if boost_overlap:
            score += min(0.25, 0.08 * len(boost_overlap))

        if score > 0:
            scored_lines.append((score, index))

    scored_lines.sort(reverse=True)
    if not scored_lines:
        return [], 0.0

    best_score, best_index = scored_lines[0]
    selected = [best_index]

    # Add a neighboring line when it is likely part of the same utterance/evidence.
    for neighbor in (best_index - 1, best_index + 1):
        if 0 <= neighbor < len(lines):
            neighbor_tokens = _review_tokens(_line_text(lines[neighbor]))
            if item_tokens & neighbor_tokens:
                selected.append(neighbor)

    selected = sorted(set(selected))
    confidence = min(1.0, best_score)
    if confidence < 0.12:
        return [], confidence
    return selected, confidence


def _source_excerpt(lines: list[Any], line_indices: list[int]) -> str:
    parts = []
    for index in line_indices[:3]:
        if 0 <= index < len(lines):
            parts.append(_line_text(lines[index]))
    excerpt = " ".join(parts).strip()
    return excerpt[:320]


def _source_time_range(
    lines: list[Any],
    line_indices: list[int],
) -> tuple[float | None, float | None]:
    starts = [
        start
        for index in line_indices
        if 0 <= index < len(lines)
        if (start := _line_time(lines[index], "start")) is not None
    ]
    ends = [
        end
        for index in line_indices
        if 0 <= index < len(lines)
        if (end := _line_time(lines[index], "end")) is not None
    ]
    return (min(starts) if starts else None, max(ends) if ends else None)


def _keyword_present(text: str, keyword: str) -> bool:
    normalized_text = _normalize_for_review(text)
    normalized_keyword = _normalize_for_review(keyword)
    return re.search(rf"\b{re.escape(normalized_keyword)}\w*\b", normalized_text) is not None


def build_summary_review(
    *,
    structured: StructuredSummary | None,
    summary: str,
    lines: list[Any],
) -> SummaryReview:
    """Build review metadata for source navigation and omission warnings."""

    review = SummaryReview()

    if structured is not None:
        for section, accessor in SECTION_ITEM_ACCESSORS.items():
            for item_index, item_text in enumerate(accessor(structured)):
                line_indices, confidence = _match_summary_item_to_lines(
                    item_text,
                    lines,
                    section=section,
                )
                start, end = _source_time_range(lines, line_indices)
                missing_source = not line_indices
                link = SummarySourceLink(
                    section=section,
                    item_index=item_index,
                    item_text=item_text,
                    line_indices=line_indices,
                    start=start,
                    end=end,
                    excerpt=_source_excerpt(lines, line_indices),
                    confidence=confidence,
                    missing_source=missing_source,
                )
                review.source_links.append(link)

                if missing_source and section != "uncertainties":
                    review.warnings.append(
                        SummaryReviewWarning(
                            kind="missing_source",
                            severity="warning",
                            section=section,
                            item_index=item_index,
                            message=(
                                "Für einen Zusammenfassungspunkt wurde keine "
                                "klare Transkriptstelle gefunden."
                            ),
                        )
                    )

    transcript_text = "\n".join(_line_text(line) for line in lines)
    combined_summary_text = summary
    if structured is not None:
        combined_summary_text += "\n" + "\n".join(
            item
            for accessor in SECTION_ITEM_ACCESSORS.values()
            for item in accessor(structured)
        )

    for keyword in sorted(DECISION_SIGNAL_TERMS):
        if not _keyword_present(transcript_text, keyword):
            continue
        if _keyword_present(combined_summary_text, keyword):
            continue

        matching_indices = [
            index
            for index, line in enumerate(lines)
            if _keyword_present(_line_text(line), keyword)
        ]
        start, end = _source_time_range(lines, matching_indices)
        review.warnings.append(
            SummaryReviewWarning(
                kind="missing_decision_signal",
                severity="warning",
                keyword=keyword,
                line_indices=matching_indices[:3],
                start=start,
                end=end,
                excerpt=_source_excerpt(lines, matching_indices),
                message=(
                    f'Im Transkript kommt "{keyword}" vor, in der '
                    "Zusammenfassung aber nicht."
                ),
            )
        )

    return review


def split_transcript_into_chunks(
    transcript_text: str,
    *,
    max_chars: int | None = None,
) -> list[str]:
    """Split long transcripts on line boundaries for map-reduce summarization."""

    max_chars = max_chars or LLM_CHUNK_CHARS
    text = transcript_text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    current_chars = 0

    for line in text.splitlines():
        line = line.rstrip()
        line_len = len(line) + 1
        if current and current_chars + line_len > max_chars:
            chunks.append("\n".join(current).strip())
            current = []
            current_chars = 0

        if line_len > max_chars:
            for start in range(0, len(line), max_chars):
                part = line[start : start + max_chars].strip()
                if part:
                    chunks.append(part)
            continue

        current.append(line)
        current_chars += line_len

    if current:
        chunks.append("\n".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def _structured_user_prompt(
    top_title: str,
    transcript_text: str,
    *,
    chunk_index: int | None = None,
    chunk_count: int | None = None,
) -> str:
    chunk_note = ""
    if chunk_index is not None and chunk_count is not None:
        chunk_note = (
            f"\nDies ist Teil {chunk_index + 1} von {chunk_count}. "
            "Extrahiere nur Informationen, die in diesem Teil vorkommen."
        )

    return f"""Erstelle strukturierte Protokollnotizen für folgenden Tagesordnungspunkt.{chunk_note}

TOP: {top_title}

Transkript:
{transcript_text}

JSON:"""


def _reduce_user_prompt(top_title: str, partials: list[StructuredSummary]) -> str:
    partial_json = json.dumps(
        [partial.to_dict() for partial in partials],
        ensure_ascii=False,
        indent=2,
    )
    return f"""Führe die folgenden strukturierten Teilnotizen zu einer konsolidierten,
dublettenfreien Protokollzusammenfassung zusammen. Erhalte fachlich relevante
Unschärfen, erfinde keine Beschlüsse und keine Abstimmungsergebnisse.

TOP: {top_title}

Teilnotizen:
{partial_json}

JSON:"""


def _summarize_structured(
    client: Any,
    *,
    top_title: str,
    transcript_text: str,
    model: str,
    system_prompt: str,
) -> tuple[StructuredSummary, int]:
    chunks = split_transcript_into_chunks(transcript_text)
    if not chunks:
        raise StructuredOutputError("Kein Transkripttext vorhanden")

    if len(chunks) == 1:
        content = _chat_completion_content(
            client,
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": _structured_user_prompt(top_title, chunks[0]),
                },
            ],
            max_tokens=1400,
            temperature=0.2,
        )
        return parse_structured_summary(content), 1

    partials = []
    for chunk_index, chunk in enumerate(chunks):
        content = _chat_completion_content(
            client,
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": _structured_user_prompt(
                        top_title,
                        chunk,
                        chunk_index=chunk_index,
                        chunk_count=len(chunks),
                    ),
                },
            ],
            max_tokens=1200,
            temperature=0.2,
        )
        partials.append(parse_structured_summary(content))

    content = _chat_completion_content(
        client,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _reduce_user_prompt(top_title, partials)},
        ],
        max_tokens=1800,
        temperature=0.2,
    )
    return parse_structured_summary(content), len(chunks)


def _freetext_user_prompt(top_title: str, transcript_text: str) -> str:
    return f"""Erstelle eine Zusammenfassung für folgenden Tagesordnungspunkt:

TOP: {top_title}

Transkript:
{transcript_text}

Zusammenfassung:"""


def _reduce_freetext_prompt(top_title: str, partials: list[str]) -> str:
    joined = "\n\n".join(
        f"Teilzusammenfassung {index + 1}:\n{partial}"
        for index, partial in enumerate(partials)
    )
    return f"""Führe die folgenden Teilzusammenfassungen zu einer konsolidierten
Niederschrift für den Tagesordnungspunkt zusammen. Entferne Dopplungen und
erfinde keine Beschlüsse, Abstimmungen oder Zuständigkeiten.

TOP: {top_title}

{joined}

Zusammenfassung:"""


def _summarize_freetext(
    client: Any,
    *,
    top_title: str,
    transcript_text: str,
    model: str,
) -> tuple[str, int]:
    chunks = split_transcript_into_chunks(transcript_text)
    if not chunks:
        return "", 0

    if len(chunks) == 1:
        content = _chat_completion_content(
            client,
            model=model,
            messages=[
                {"role": "system", "content": FREETEXT_SYSTEM_PROMPT},
                {"role": "user", "content": _freetext_user_prompt(top_title, chunks[0])},
            ],
            max_tokens=1024,
            temperature=0.3,
        )
        return content, 1

    partials = []
    for chunk in chunks:
        partials.append(
            _chat_completion_content(
                client,
                model=model,
                messages=[
                    {"role": "system", "content": FREETEXT_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": _freetext_user_prompt(top_title, chunk),
                    },
                ],
                max_tokens=900,
                temperature=0.3,
            )
        )

    content = _chat_completion_content(
        client,
        model=model,
        messages=[
            {"role": "system", "content": FREETEXT_SYSTEM_PROMPT},
            {"role": "user", "content": _reduce_freetext_prompt(top_title, partials)},
        ],
        max_tokens=1400,
        temperature=0.3,
    )
    return content, len(chunks)


def summarize_segment(
    top_title: str,
    transcript_text: str,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> SummarizationResult:
    """
    Generate a summary for a meeting segment (TOP).

    The primary path asks the LLM for structured JSON and renders it into the
    existing editable text summary. Long transcripts are processed via
    map-reduce over transcript chunks. If structured output fails and fallback
    is enabled, a chunked free-text summarization path is used.
    """

    config = get_llm_config(model)
    client = _load_openai_client(config)
    check_llm_availability(client=client, model=config.model)
    actual_model = config.model
    actual_system_prompt = build_structured_system_prompt(system_prompt)

    start_time = time.time()
    try:
        structured, chunks_processed = _summarize_structured(
            client,
            top_title=top_title,
            transcript_text=transcript_text,
            model=actual_model,
            system_prompt=actual_system_prompt,
        )
        summary = render_structured_summary(structured)
        if not summary:
            raise StructuredOutputError(
                "Strukturierte Antwort konnte nicht gerendert werden"
            )
        duration_seconds = time.time() - start_time
        return SummarizationResult(
            summary=summary,
            duration_seconds=duration_seconds,
            structured=structured,
            fallback_used=False,
            chunks_processed=chunks_processed,
        )
    except LLMCallError:
        raise
    except StructuredOutputError:
        if not LLM_STRUCTURED_FALLBACK:
            raise

    summary, chunks_processed = _summarize_freetext(
        client,
        top_title=top_title,
        transcript_text=transcript_text,
        model=actual_model,
    )
    duration_seconds = time.time() - start_time
    return SummarizationResult(
        summary=summary,
        duration_seconds=duration_seconds,
        structured=None,
        fallback_used=True,
        chunks_processed=chunks_processed,
    )


def summarize_all_segments(
    tops: list[str],
    segments: dict[int, str],
) -> dict[int, str]:
    """
    Generate summaries for all TOPs.

    Returns:
        Dict mapping TOP index to editable summary text.
    """
    summaries = {}
    for top_idx, transcript_text in segments.items():
        if transcript_text.strip():
            top_title = tops[top_idx] if top_idx < len(tops) else f"TOP {top_idx + 1}"
            summaries[top_idx] = summarize_segment(top_title, transcript_text).summary
    return summaries
