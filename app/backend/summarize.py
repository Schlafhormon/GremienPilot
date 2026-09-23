"""
Summarization module for generating meeting minutes per TOP.

Uses an OpenAI-compatible API, typically Ollama, for local German
summarization. The public API still exposes an editable text summary, while
the backend internally works with structured minutes fields.

Configuration via environment variables:
- LLM_BASE_URL: API endpoint (local default: http://localhost:11434/v1,
  Docker default: http://ollama:11434/v1)
- LLM_MODEL: Model name (default: gemma4:31b-it-q4_K_M)
- LLM_REASONING_EFFORT: empty for server default, none to disable, or low/medium/high/max
- LLM_TIMEOUT_SECONDS: request timeout per LLM call (default: 120)
- LLM_MAX_RETRIES: retry count for transient LLM errors (default: 2)
- LLM_CHUNK_CHARS: target chunk size for long TOP transcripts (default: 12000)
- SUMMARY_OUTPUT_TOKENS, SUMMARY_MODEL_ATTEMPTS, SUMMARY_RECONCILIATION_ROUNDS: mandatory verification policy
"""
from llm_config import configured
from llm_transport import LLMCancelledError


import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional
from urllib.parse import urlparse
from llm_transport import complete, fits, structured_output_budget, ContextBudgetError, cache_key, cache_read, cache_write

# Compatibility exports for integrations importing configuration from summarize.
from llm_config import (LLMConfig, get_llm_config as _get_llm_config,
                        resolve_llm_base_url, is_docker_runtime)
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma4:31b-it-q4_K_M")
LLM_BASE_URL, LLM_BASE_URL_SOURCE = resolve_llm_base_url()
LLM_API_KEY = os.environ.get("LLM_API_KEY", "ollama")
LLM_TIMEOUT_SECONDS = float(os.environ.get("LLM_TIMEOUT_SECONDS") or "120")
LLM_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "2"))
LLM_RETRY_BACKOFF_SECONDS = float(os.environ.get("LLM_RETRY_BACKOFF_SECONDS", "0.5"))
LLM_CHUNK_CHARS = int(os.environ.get("LLM_CHUNK_CHARS", "12000"))



def get_llm_config(model=None):
    return _get_llm_config(model)


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

    configuration: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StructuredSummary:
    """Structured internal representation of one TOP summary."""

    discussion: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    votes: list[str] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)
    open_points: list[str] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)

    evidence: list[dict] = field(default_factory=list)
    review_questions: list[dict] = field(default_factory=list)
    verification: dict = field(default_factory=dict)
    rejected_candidates: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SummarizationResult:
    """Result from summarization including timing and internal structure."""

    summary: str
    duration_seconds: float
    structured: StructuredSummary | None = None
    fallback_used: bool = False
    chunks_processed: int = 1
    llm_usage: dict[str, Any] = field(default_factory=dict)


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
    source_ids: list[str] = field(default_factory=list)
    scope: str | None = None
    grounding: dict = field(default_factory=dict)

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
- Unter decisions/votes nur Ergebnisse dieser Sitzung; Berichte über frühere Sitzungen oder andere Gremien gehören ausdrücklich als Rückblick unter discussion
- Vorschläge und Absichten nicht als bereits getroffene Beschlüsse ausgeben
- Widersprüchliche Namen, Zahlen und Termine nicht still korrigieren, sondern als Unsicherheit kennzeichnen
- fehlende oder unklare Informationen ausdrücklich unter "uncertainties" markieren
- Geschäftsordnungs- und Technikdetails nur aufnehmen, wenn sie für den TOP relevant sind

Gib ausschließlich valides JSON entsprechend dem angeforderten Schema zurück.
Jede fachliche Notiz enthält section, text, scope und evidence (source_id, quote).
section ist discussion, decisions, votes, action_items, open_points oder uncertainties.
scope ist current, proposal, retrospective, quoted_prior oder unclear.
Nur heutige Ergebnisse gehören unter decisions/votes/action_items.
Das im jeweiligen Aufruf angegebene JSON-Schema ist verbindlich."""


STRUCTURED_KEYS = (
    "discussion",
    "decisions",
    "votes",
    "action_items",
    "open_points",
    "uncertainties",
)

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
        timeout=config.http_timeout,
        # The shared transport owns retries; diagnostics have no SDK retries.
        max_retries=0,
    )


def classify_llm_error(error: Exception) -> LLMErrorInfo:
    """Classify common OpenAI-compatible client errors."""

    from llm_transport import retryable, LLMCancelledError, LLMTotalTimeout
    from llm_config import ModelConfigurationError
    if isinstance(error, (ModelConfigurationError, ContextBudgetError, LLMCancelledError, LLMTotalTimeout)):
        return LLMErrorInfo("configuration" if isinstance(error, ModelConfigurationError) else "incomplete", False)
    if any(term in str(error).lower() for term in ("out of memory", "requires more system memory", "insufficient memory", "failed to allocate")):
        return LLMErrorInfo("memory", False)
    status_code = getattr(error, "status_code", None) or getattr(getattr(error, "response", None), "status_code", None)
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
    except LLMCancelledError:
        raise
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
        configuration=config.public_snapshot(),
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
            configuration=config.public_snapshot(),
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

    def unique_fields(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise StructuredOutputError("Strukturierte Antwort enthält doppelte JSON-Felder")
            result[key] = value
        return result

    try:
        parsed = json.loads(stripped, object_pairs_hook=unique_fields)
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
    seen = set()

    for raw_key, raw_value in parsed.items():
        key = str(raw_key).strip()
        normalized_key = KEY_ALIASES.get(key.lower(), key)
        if normalized_key in STRUCTURED_KEYS:
            if normalized_key in seen:
                raise StructuredOutputError("Strukturierte Antwort enthält mehrdeutige Kategorien")
            seen.add(normalized_key)
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
        ("Maßnahmen", structured.action_items),
        ("Offene Punkte", structured.open_points),
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


def _line_text(line: Any) -> str:
    speaker = str(_line_value(line, "speaker", "") or "")
    text = str(_line_value(line, "text", "") or "")
    return f"{speaker}: {text}"


def _line_time(line: Any, key: str) -> float | None:
    value = _line_value(line, key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def build_summary_review(
    *, structured: StructuredSummary | None, summary: str, lines: list[Any],
) -> SummaryReview:
    """Project verified model references; never infer evidence from similar words.

    Legacy/manual summaries remain editable, but cannot inherit a model certificate
    for different text or sources. Accepting a manual edit is not model verification.
    """
    from summary_grounding import digest
    review = SummaryReview()
    verification = structured.verification if structured else {}
    valid = bool(verification.get('source_sha256') == digest([_line_text(line) for line in lines])
                 and verification.get('summary_sha256') == digest(summary))
    if not valid:
        review.warnings.append(SummaryReviewWarning(kind='verification_required',
            message='Für diese Text- und Quellenfassung liegt keine vollständige automatische Prüfung vor. '
                    'Bitte neu generieren oder die manuelle Fassung fachlich prüfen.'))
        return review
    if not verification.get('processing_complete'):
        review.warnings.append(SummaryReviewWarning(kind='technical_incomplete',
            message='Entwurf erhalten; erforderliche unabhängige Prüfungen sind technisch unvollständig.', severity='error'))
    sources = {row['source_id']: row for row in verification.get('sources', [])}
    def indices(evidence):
        return sorted({sources[item['source_id']]['line_index'] for item in evidence
                       if item.get('source_id') in sources})
    for item in structured.evidence:
        refs = item['sources']
        accessible = list(dict.fromkeys([ref['source_id'] for ref in refs] + item.get('grounding', {}).get('source_ids', [])))
        line_indices = indices([{'source_id': identity} for identity in accessible])
        start, end = _source_time_range(lines, line_indices)
        review.source_links.append(SummarySourceLink(
            section=item['section'], item_index=item['item_index'], item_text=item['item_text'],
            line_indices=line_indices, start=start, end=end,
            excerpt=_source_excerpt(lines, line_indices), missing_source=not line_indices,
            source_ids=accessible, scope=item['scope'], grounding=item.get('grounding', {})))
        for question in item.get('grounding', {}).get('questions', []):
            review.warnings.append(SummaryReviewWarning(kind='open_evidence', message=question,
                section=item['section'], item_index=item['item_index'], line_indices=line_indices,
                start=start, end=end, excerpt=_source_excerpt(lines, line_indices)))
    for issue in structured.review_questions:
        line_indices = indices(issue['evidence'])
        start, end = _source_time_range(lines, line_indices)
        review.warnings.append(SummaryReviewWarning(kind=issue['kind'], message=issue['question'],
            line_indices=line_indices, start=start, end=end, excerpt=_source_excerpt(lines, line_indices)))
    for item in structured.evidence:
        if item['section'] == 'uncertainties':
            line_indices = indices(item['sources'])
            start, end = _source_time_range(lines, line_indices)
            review.warnings.append(SummaryReviewWarning(kind='unclear', message=item['item_text'],
                line_indices=line_indices, start=start, end=end, excerpt=_source_excerpt(lines, line_indices)))
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


def meeting_context_from_transcript(transcript) -> str:
    """Small verbatim opening excerpt, only to identify the current meeting."""
    return '\n'.join(str(_line_value(line, 'text', '')) for line in transcript[:5])[:800]


@configured
def summarize_segment(
    top_title: str,
    transcript_text: str,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    meeting_context: Optional[str] = None,
    source_lines: list[str] | None = None,
) -> SummarizationResult:
    """Generate, independently verify, reconcile and verify the final minutes.

    Existing editable text/list fields remain compatible. Source evidence and
    verification are additive. There is no unverified free-text fallback.
    """
    from summary_grounding import Workflow, digest
    config = get_llm_config(model)
    client = _load_openai_client(config)
    check_llm_availability(client=client, model=config.model)
    lines = source_lines if source_lines is not None else transcript_text.splitlines()
    if not lines or not transcript_text.strip():
        raise StructuredOutputError("Kein Transkripttext vorhanden")
    if source_lines is not None and "\n".join(source_lines) != transcript_text:
        raise StructuredOutputError("Quellzeilen stimmen nicht mit Transkript überein")
    start = time.monotonic()
    usage = {'configuration': config.public_snapshot()}
    workflow = Workflow(client, config, build_structured_system_prompt(system_prompt)
                        + "\nTOP: " + top_title, meeting_context, usage)
    def attach_partial(error):
        if not workflow.latest_claims:
            return error
        from source_contract import reviewed, marked_text
        partial = StructuredSummary()
        for claim in workflow.latest_claims:
            g = reviewed(claim['grounding'], questions=['Die unabhängige Prüfung ist unvollständig. Stützt die Quelle diese Aussage?'])
            items = getattr(partial, claim['section'])
            text = marked_text(claim['text'], g)
            partial.evidence.append(dict(section=claim['section'],item_index=len(items),item_text=text,
                original_text=claim['text'],scope=claim['scope'],sources=claim['evidence'],grounding=g))
            items.append(text)
        text = render_structured_summary(partial)
        partial.verification = dict(processing_complete=False,source_contract='graded-sources-v1',
            source_sha256=digest(lines),summary_sha256=digest(text),sources=list(workflow.partial_rows.values()))
        error.partial_result = SummarizationResult(summary=text,structured=partial,duration_seconds=time.monotonic()-start,
            llm_usage={**usage,'processing_complete':False,'grounding_incomplete':True,'review_required':True})
        return error
    try:
        claims, issues, rows, count = workflow.run(lines)
    except LLMCancelledError:
        raise
    except ContextBudgetError as exc:
        raise attach_partial(exc)
    except ValueError as exc:
        raise attach_partial(StructuredOutputError("Automatische Quellenprüfung technisch unvollständig")) from exc
    except Exception as exc:
        info = classify_llm_error(exc)
        raise attach_partial(LLMCallError("Automatische Quellenprüfung fehlgeschlagen (" + info.category + ")",
                           category=info.category, transient=info.transient)) from exc
    structured = StructuredSummary()
    from source_contract import marked_text
    for claim in claims:
        section = claim['section']
        items = getattr(structured, section)
        structured.evidence.append(dict(section=section, item_index=len(items),
            item_text=marked_text(claim['text'], claim['grounding']), scope=claim['scope'], sources=claim['evidence'],
            grounding=claim['grounding'], original_text=claim['text']))
        items.append(marked_text(claim['text'], claim['grounding']))
    structured.rejected_candidates = usage.get('rejected_candidates', [])
    for claim in structured.rejected_candidates:
        structured.uncertainties.append(marked_text(claim['text'], claim['grounding']))
    structured.review_questions = issues
    structured.verification = dict(processing_complete=True, source_sha256=digest(lines),
        sources=rows, checks=usage['required_checks'], prompt_version=usage['prompt_version'],
        source_contract='graded-sources-v1')
    summary = render_structured_summary(structured)
    if not summary:
        # No semantic filler. Absence is a model result, with evidence and completed checks.
        summary = "Keine protokollrelevanten Inhalte festgestellt."
    structured.verification['summary_sha256'] = digest(summary)
    return SummarizationResult(summary=summary, structured=structured,
        duration_seconds=time.monotonic()-start, chunks_processed=count, llm_usage=usage)


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
