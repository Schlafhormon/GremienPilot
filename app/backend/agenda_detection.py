"""
Agenda detection and segmentation for meeting transcripts.

This module builds on the existing deterministic assignment suggestions and
adds an optional LLM pass for reviewable TOP detection when no PDF/manual agenda
is available, or for refined boundaries when an agenda is already known.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from dataclasses import dataclass, field, replace
from typing import Any

from assignment_suggestions import (
    AssignmentSegment,
    TranscriptUtterance,
    assignments_from_segments,
    has_transition_phrase,
    likely_moderator_speakers,
    normalize_text,
    suggest_assignments,
)
from summarize import get_llm_config


logger = logging.getLogger(__name__)

LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3:8b")
LLM_BASE_URL = get_llm_config().base_url
LLM_API_KEY = get_llm_config().api_key
_agenda_llm_default = os.environ.get("AGENDA_DETECTION_USE_LLM", "false").strip().lower()
if _agenda_llm_default not in {"true", "false"}:
    raise ValueError("AGENDA_DETECTION_USE_LLM must be true or false")
AGENDA_DETECTION_USE_LLM = _agenda_llm_default == "true"
AGENDA_DETECTION_TIMEOUT_SECONDS = float(
    os.environ.get("AGENDA_DETECTION_TIMEOUT_SECONDS", "8")
)
if not math.isfinite(AGENDA_DETECTION_TIMEOUT_SECONDS) or AGENDA_DETECTION_TIMEOUT_SECONDS <= 0:
    raise ValueError("AGENDA_DETECTION_TIMEOUT_SECONDS must be finite and positive")
AGENDA_DETECTION_CHUNK_LINES = int(
    os.environ.get("AGENDA_DETECTION_CHUNK_LINES", "160")
)
AGENDA_DETECTION_CHUNK_OVERLAP_LINES = int(
    os.environ.get("AGENDA_DETECTION_CHUNK_OVERLAP_LINES", "12")
)
AGENDA_DETECTION_CONTEXT_WINDOW_BEFORE = int(
    os.environ.get("AGENDA_DETECTION_CONTEXT_WINDOW_BEFORE", "4")
)
AGENDA_DETECTION_CONTEXT_WINDOW_AFTER = int(
    os.environ.get("AGENDA_DETECTION_CONTEXT_WINDOW_AFTER", "8")
)

DEFAULT_AGENDA_DETECTION_PROMPT = """Du erkennst Tagesordnungspunkte (TOPs) und Segmentgrenzen in deutschen Sitzungstranskripten.

Gib ausschliesslich valides JSON im folgenden Format zurueck:
{
  "tops": [
    {
      "top_title": "Titel ohne TOP-Nummer",
      "start_index": 0,
      "end_index": 4,
      "confidence": 0.0,
      "evidence_text": "kurzer Beleg aus dem Transkript",
      "uncertain": false
    }
  ]
}

Regeln:
- Indizes sind 0-basiert und beziehen sich auf die Transkriptzeilen.
- TOPs muessen in Transkriptreihenfolge stehen.
- Segmente duerfen sich nicht ueberlappen.
- Markiere geschaetzte oder schwache Grenzen mit uncertain=true.
- Nutze Moderationssignale wie "kommen wir zu", "rufe ich auf", "naechster Punkt" und explizite TOP-Zahlen."""


def build_agenda_detection_system_prompt(system_prompt: str | None = None) -> str:
    """Caller context may refine detection, but never replace its output contract."""
    custom_prompt = (system_prompt or "").strip()
    if not custom_prompt or custom_prompt == DEFAULT_AGENDA_DETECTION_PROMPT.strip():
        return DEFAULT_AGENDA_DETECTION_PROMPT
    return (
        DEFAULT_AGENDA_DETECTION_PROMPT
        + "\n\nZusätzliche fachliche Vorgaben des Nutzers. Diese nur anwenden, "
        "soweit sie der TOP-Erkennung, dem JSON-Schema und den Regeln für "
        "Segmentgrenzen oben nicht widersprechen; diese haben Vorrang:\n"
        + custom_prompt
    )


@dataclass
class AgendaLLMUsage:
    enabled: bool
    source: str
    timeout_seconds: float = AGENDA_DETECTION_TIMEOUT_SECONDS
    status: str = "skipped"
    attempted_calls: int = 0
    failed_calls: int = 0
    failure_reasons: list[str] = field(default_factory=list)

    @property
    def warnings(self) -> list[str]:
        if not self.failed_calls:
            return []
        return [
            f"TOP-Erkennung: {self.failed_calls} von {self.attempted_calls} LLM-Aufrufen "
            f"fehlgeschlagen ({', '.join(self.failure_reasons)}). "
            "Heuristische Ersatzverarbeitung verwendet; bitte TOP-Zuordnung prüfen."
        ]


def _llm_usage(use_llm: bool | None) -> AgendaLLMUsage:
    enabled = _should_use_llm(use_llm)
    return AgendaLLMUsage(
        enabled=enabled,
        source="server_default" if use_llm is None else "request",
        timeout_seconds=AGENDA_DETECTION_TIMEOUT_SECONDS,
        status="skipped" if enabled else "disabled",
    )


class _InvalidLLMResponse(ValueError):
    pass


class _EmptyLLMResponse(ValueError):
    pass


@dataclass(frozen=True)
class AgendaDetectionResult:
    tops: list[str]
    assignments: list[int | None]
    segments: list[AssignmentSegment]
    uncertain_count: int
    strategy: str
    llm: AgendaLLMUsage | None = None


@dataclass(frozen=True)
class _RawSegment:
    top_title: str
    start_index: int | None
    end_index: int | None
    confidence: float
    evidence_text: str | None
    uncertain: bool
    reason: str
    transition_type: str


def detect_agenda_from_transcript(
    transcript: list[TranscriptUtterance],
    model: str | None = None,
    system_prompt: str | None = None,
    *,
    use_llm: bool | None = None,
) -> AgendaDetectionResult:
    """Detect TOP titles and line boundaries without a known agenda list."""
    usage = _llm_usage(use_llm)
    if not transcript:
        return AgendaDetectionResult([], [], [], 0, "heuristic_transcript_empty", usage)

    heuristic_segments = _heuristic_detect_unknown_agenda(transcript)
    llm_segments = _maybe_detect_with_llm(
        transcript,
        tops=None,
        heuristic_segments=heuristic_segments,
        model=model,
        system_prompt=system_prompt,
        usage=usage,
    )

    if llm_segments:
        segments, repaired = _validate_unknown_segments(
            len(transcript),
            llm_segments,
            fallback_segments=heuristic_segments,
        )
        strategy = "heuristic_transcript_llm_repaired" if repaired else "heuristic_transcript_llm"
    else:
        segments, repaired = _validate_unknown_segments(
            len(transcript),
            [_segment_to_raw(segment) for segment in heuristic_segments],
            fallback_segments=[],
        )
        strategy = "heuristic_transcript_fallback" if not usage.enabled else "heuristic_transcript_llm_fallback"
        if repaired:
            strategy += "_repaired"

    return _result_from_segments(len(transcript), segments, strategy, usage=usage)


def segment_known_agenda(
    transcript: list[TranscriptUtterance],
    tops: list[str],
    model: str | None = None,
    system_prompt: str | None = None,
    *,
    use_llm: bool | None = None,
) -> AgendaDetectionResult:
    """Detect start/end lines for an already known TOP list."""
    usage = _llm_usage(use_llm)
    valid_tops = [top.strip() for top in tops if top.strip()]
    if not transcript or not valid_tops:
        return AgendaDetectionResult(valid_tops, [None] * len(transcript), [], 0, "known_agenda_empty", usage)

    heuristic_result = suggest_assignments(transcript, valid_tops)
    heuristic_segments = list(heuristic_result.segments)
    llm_segments = _maybe_detect_with_llm(
        transcript,
        tops=valid_tops,
        heuristic_segments=heuristic_segments,
        model=model,
        system_prompt=system_prompt,
        usage=usage,
    )

    if llm_segments:
        segments, repaired = _validate_known_segments(
            len(transcript),
            valid_tops,
            llm_segments,
            heuristic_segments,
        )
        strategy = "known_agenda_heuristic_llm_repaired" if repaired else "known_agenda_heuristic_llm"
    else:
        segments, repaired = _validate_known_segments(
            len(transcript),
            valid_tops,
            [_segment_to_raw(segment) for segment in heuristic_segments],
            heuristic_segments,
        )
        strategy = "known_agenda_heuristic"
        if usage.enabled:
            strategy += "_llm_fallback"
        if repaired:
            strategy += "_repaired"

    return _result_from_segments(len(transcript), segments, strategy, tops=valid_tops, usage=usage)


def _result_from_segments(
    transcript_length: int,
    segments: list[AssignmentSegment],
    strategy: str,
    *,
    tops: list[str] | None = None,
    usage: AgendaLLMUsage,
) -> AgendaDetectionResult:
    assignments = assignments_from_segments(transcript_length, segments)
    return AgendaDetectionResult(
        tops=tops if tops is not None else [segment.top_title for segment in segments],
        assignments=assignments,
        segments=segments,
        uncertain_count=sum(1 for segment in segments if segment.uncertain),
        strategy=strategy,
        llm=usage,
    )


def _should_use_llm(use_llm: bool | None = None) -> bool:
    """An explicit request overrides the server default, never model/prompt text."""
    if use_llm is not None and not isinstance(use_llm, bool):
        raise ValueError("use_llm must be a boolean or None")
    return AGENDA_DETECTION_USE_LLM if use_llm is None else use_llm


def _attempt_llm_detection(*, usage: AgendaLLMUsage, **kwargs: Any) -> list[_RawSegment]:
    usage.attempted_calls += 1
    try:
        segments = _detect_with_llm(**kwargs)
        if not segments:
            raise _EmptyLLMResponse()
        return segments
    except Exception as exc:
        # Fixed reason codes only: exception messages may contain transcript,
        # provider response bodies, URLs or credentials.
        names = {cls.__name__ for cls in type(exc).__mro__}
        if isinstance(exc, _InvalidLLMResponse):
            reason = "invalid_response"
        elif isinstance(exc, _EmptyLLMResponse):
            reason = "empty_response"
        elif names & {"TimeoutError", "APITimeoutError", "TimeoutException"}:
            reason = "timeout"
        elif names & {"ConnectionError", "APIConnectionError", "ConnectError"}:
            reason = "connection_error"
        else:
            reason = "request_error"
        usage.failed_calls += 1
        if reason not in usage.failure_reasons:
            usage.failure_reasons.append(reason)
        logger.warning("Agenda LLM fallback: reason=%s call=%d", reason, usage.attempted_calls)
        return []


def _maybe_detect_with_llm(
    transcript: list[TranscriptUtterance],
    *,
    tops: list[str] | None,
    heuristic_segments: list[AssignmentSegment],
    model: str | None,
    system_prompt: str | None,
    usage: AgendaLLMUsage,
) -> list[_RawSegment]:
    if not usage.enabled:
        return []
    if tops is None and len(transcript) > AGENDA_DETECTION_CHUNK_LINES:
        segments = _detect_unknown_agenda_with_llm_chunks(
            transcript, heuristic_segments=heuristic_segments,
            model=model, system_prompt=system_prompt, usage=usage,
        )
    else:
        segments = _attempt_llm_detection(
            usage=usage, transcript=transcript, tops=tops,
            heuristic_segments=heuristic_segments, model=model, system_prompt=system_prompt,
        )
    usage.status = (
        "success" if not usage.failed_calls else
        "fallback" if usage.failed_calls == usage.attempted_calls else "partial_fallback"
    )
    return segments


def _iter_transcript_chunks(
    transcript: list[TranscriptUtterance],
) -> list[tuple[int, list[TranscriptUtterance]]]:
    max_lines = max(1, AGENDA_DETECTION_CHUNK_LINES)
    overlap = max(0, min(AGENDA_DETECTION_CHUNK_OVERLAP_LINES, max_lines - 1))
    step = max(1, max_lines - overlap)
    chunks: list[tuple[int, list[TranscriptUtterance]]] = []

    for start in range(0, len(transcript), step):
        chunk = transcript[start : start + max_lines]
        if chunk:
            chunks.append((start, chunk))
        if start + max_lines >= len(transcript):
            break
    return chunks


def _segments_for_chunk(
    segments: list[AssignmentSegment],
    *,
    chunk_start: int,
    chunk_length: int,
) -> list[AssignmentSegment]:
    chunk_end = chunk_start + chunk_length - 1
    adjusted = []
    for segment in segments:
        if segment.start_index > chunk_end or segment.end_index < chunk_start:
            continue
        adjusted.append(
            AssignmentSegment(
                top_index=segment.top_index,
                top_title=segment.top_title,
                start_index=max(0, segment.start_index - chunk_start),
                end_index=min(chunk_length - 1, segment.end_index - chunk_start),
                confidence=segment.confidence,
                uncertain=segment.uncertain,
                transition_type=segment.transition_type,
                reason=segment.reason,
                evidence_index=(
                    segment.evidence_index - chunk_start
                    if segment.evidence_index is not None
                    else None
                ),
                evidence_text=segment.evidence_text,
            )
        )
    return adjusted


def _offset_raw_segments(
    segments: list[_RawSegment],
    *,
    offset: int,
) -> list[_RawSegment]:
    return [
        _RawSegment(
            top_title=segment.top_title,
            start_index=(
                segment.start_index + offset
                if segment.start_index is not None
                else None
            ),
            end_index=(
                segment.end_index + offset if segment.end_index is not None else None
            ),
            confidence=segment.confidence,
            evidence_text=segment.evidence_text,
            uncertain=segment.uncertain,
            reason=segment.reason,
            transition_type=segment.transition_type,
        )
        for segment in segments
    ]


def _detect_unknown_agenda_with_llm_chunks(
    transcript: list[TranscriptUtterance],
    *,
    heuristic_segments: list[AssignmentSegment],
    model: str | None,
    system_prompt: str | None,
    usage: AgendaLLMUsage,
) -> list[_RawSegment]:
    detected: list[_RawSegment] = []
    for chunk_start, chunk in _iter_transcript_chunks(transcript):
        segments = _attempt_llm_detection(
            usage=usage, transcript=chunk, tops=None,
            heuristic_segments=_segments_for_chunk(
                heuristic_segments, chunk_start=chunk_start, chunk_length=len(chunk),
            ),
            model=model, system_prompt=system_prompt,
        )
        if not segments:
            segments = [
                replace(
                    _segment_to_raw(segment), uncertain=True,
                    reason="Heuristische Ersatzverarbeitung nach LLM-Ausfall.",
                )
                for segment in _segments_for_chunk(
                    heuristic_segments, chunk_start=chunk_start, chunk_length=len(chunk),
                )
            ]
        detected.extend(_offset_raw_segments(segments, offset=chunk_start))
    return [] if usage.failed_calls == usage.attempted_calls else detected


def _detect_with_llm(
    transcript: list[TranscriptUtterance],
    *,
    tops: list[str] | None,
    heuristic_segments: list[AssignmentSegment],
    model: str | None,
    system_prompt: str | None,
) -> list[_RawSegment]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("OpenAI client nicht installiert") from exc

    config = get_llm_config(model)
    actual_model = config.model
    actual_system_prompt = build_agenda_detection_system_prompt(system_prompt)
    client = OpenAI(
        base_url=config.base_url,
        api_key=config.api_key,
        timeout=AGENDA_DETECTION_TIMEOUT_SECONDS,
        max_retries=0,
    )

    response = client.chat.completions.create(
        model=actual_model,
        messages=[
            {"role": "system", "content": actual_system_prompt},
            {"role": "user", "content": _build_llm_user_prompt(transcript, tops, heuristic_segments)},
        ],
        temperature=0.1,
        max_tokens=2048,
    )
    try:
        raw_response = _llm_message_text(response.choices[0].message)
        return _parse_llm_segments(raw_response)
    except Exception as exc:
        raise _InvalidLLMResponse() from exc


def _llm_message_text(message: Any) -> str:
    content = str(getattr(message, "content", "") or "").strip()
    if content:
        return content
    for attribute in ("reasoning", "reasoning_content"):
        value = str(getattr(message, attribute, "") or "").strip()
        if value:
            return value
    return ""


def _build_llm_user_prompt(
    transcript: list[TranscriptUtterance],
    tops: list[str] | None,
    heuristic_segments: list[AssignmentSegment],
) -> str:
    indexed_transcript, compacted = _indexed_transcript_for_llm(
        transcript,
        tops=tops,
        heuristic_segments=heuristic_segments,
    )
    heuristic_json = [
        {
            "top_title": segment.top_title,
            "start_index": segment.start_index,
            "end_index": segment.end_index,
            "confidence": segment.confidence,
            "uncertain": segment.uncertain,
            "evidence_text": segment.evidence_text,
        }
        for segment in heuristic_segments
    ]

    if tops:
        agenda = "\n".join(f"{index}: {top}" for index, top in enumerate(tops))
        compact_note = (
            "Das Transkript ist auf relevante Kontextfenster gekuerzt; "
            "die angezeigten Indizes bleiben die originalen Transkriptindizes. "
            if compacted
            else ""
        )
        task = (
            "Bekannte TOP-Liste. Pruefe und verbessere die Segmentgrenzen. "
            f"{compact_note}"
            "Gib genau einen Eintrag pro TOP in derselben Reihenfolge zurueck.\n\n"
            f"TOPs:\n{agenda}"
        )
    else:
        task = (
            "Keine TOP-Liste vorhanden. Erkenne TOP-Titel und Segmentgrenzen aus "
            "dem Transkript. Gib nur TOPs zurueck, die im Transkript belegbar sind."
        )

    return (
        f"{task}\n\n"
        f"Heuristische Voranalyse:\n{json.dumps(heuristic_json, ensure_ascii=False)}\n\n"
        f"Transkript:\n{indexed_transcript}"
    )


def _indexed_transcript_for_llm(
    transcript: list[TranscriptUtterance],
    *,
    tops: list[str] | None,
    heuristic_segments: list[AssignmentSegment],
) -> tuple[str, bool]:
    if not tops or len(transcript) <= AGENDA_DETECTION_CHUNK_LINES:
        return (
            "\n".join(
                f"{index}: {line.speaker}: {line.text}"
                for index, line in enumerate(transcript)
            ),
            False,
        )

    selected = _compact_known_agenda_indices(transcript, heuristic_segments)
    return (
        "\n".join(
            f"{index}: {transcript[index].speaker}: {transcript[index].text}"
            for index in selected
        ),
        True,
    )


def _compact_known_agenda_indices(
    transcript: list[TranscriptUtterance],
    heuristic_segments: list[AssignmentSegment],
) -> list[int]:
    selected: set[int] = set()
    last_index = len(transcript) - 1
    if last_index < 0:
        return []

    selected.update(range(0, min(last_index + 1, AGENDA_DETECTION_CONTEXT_WINDOW_AFTER)))
    selected.update(
        range(
            max(0, last_index - AGENDA_DETECTION_CONTEXT_WINDOW_BEFORE + 1),
            last_index + 1,
        )
    )

    for segment in heuristic_segments:
        for anchor in {segment.start_index, segment.end_index, segment.evidence_index}:
            if anchor is None:
                continue
            start = max(0, anchor - AGENDA_DETECTION_CONTEXT_WINDOW_BEFORE)
            end = min(last_index, anchor + AGENDA_DETECTION_CONTEXT_WINDOW_AFTER)
            selected.update(range(start, end + 1))

    for index, line in enumerate(transcript):
        if has_transition_phrase(line.text):
            start = max(0, index - AGENDA_DETECTION_CONTEXT_WINDOW_BEFORE)
            end = min(last_index, index + AGENDA_DETECTION_CONTEXT_WINDOW_AFTER)
            selected.update(range(start, end + 1))

    return sorted(selected)


def _parse_llm_segments(response_text: str) -> list[_RawSegment]:
    payload = _extract_json_payload(response_text)
    if isinstance(payload, dict):
        raw_items = payload.get("tops") or payload.get("segments") or []
    elif isinstance(payload, list):
        raw_items = payload
    else:
        raw_items = []

    segments: list[_RawSegment] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        title = str(item.get("top_title") or item.get("title") or "").strip()
        if not title:
            continue
        segments.append(
            _RawSegment(
                top_title=title,
                start_index=_coerce_int(item.get("start_index")),
                end_index=_coerce_int(item.get("end_index")),
                confidence=_coerce_confidence(item.get("confidence"), default=0.55),
                evidence_text=_coerce_optional_text(item.get("evidence_text")),
                uncertain=bool(item.get("uncertain", False)),
                reason="LLM-Erkennung mit strukturierter Ausgabe.",
                transition_type="llm",
            )
        )
    return segments


def _extract_json_payload(response_text: str) -> Any:
    text = response_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    object_start = text.find("{")
    object_end = text.rfind("}")
    array_start = text.find("[")
    array_end = text.rfind("]")
    candidates = []
    if object_start != -1 and object_end > object_start:
        candidates.append(text[object_start : object_end + 1])
    if array_start != -1 and array_end > array_start:
        candidates.append(text[array_start : array_end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError("LLM-Antwort enthaelt kein valides JSON")


def _heuristic_detect_unknown_agenda(
    transcript: list[TranscriptUtterance],
) -> list[AssignmentSegment]:
    moderator_speakers = likely_moderator_speakers(transcript)
    raw_segments: list[_RawSegment] = []

    for index, line in enumerate(transcript):
        parsed = _parse_heuristic_top_announcement(line.text)
        if parsed is None:
            continue
        title, explicit_number = parsed
        if line.speaker not in moderator_speakers and not explicit_number:
            continue
        confidence = 0.86 if explicit_number else 0.62
        if line.speaker in moderator_speakers:
            confidence = min(0.95, confidence + 0.08)
        raw_segments.append(
            _RawSegment(
                top_title=title,
                start_index=index,
                end_index=None,
                confidence=round(confidence, 2),
                evidence_text=line.text,
                uncertain=confidence < 0.7,
                reason=(
                    "Explizite TOP-Ankuendigung im Transkript."
                    if explicit_number
                    else "Moderationsformulierung als TOP-Wechsel erkannt."
                ),
                transition_type="explicit" if explicit_number else "heuristic",
            )
        )

    segments, _ = _validate_unknown_segments(len(transcript), raw_segments, fallback_segments=[])
    return segments


def _parse_heuristic_top_announcement(text: str) -> tuple[str, bool] | None:
    if not has_transition_phrase(text):
        return None

    explicit_match = re.search(
        r"\b(?:top|tagesordnungspunkt|punkt)\s*(\d+(?:\.\d+)*)\b[\s).:-]*(?P<title>.*)$",
        text,
        flags=re.IGNORECASE,
    )
    if explicit_match:
        title = _clean_detected_title(explicit_match.group("title"))
        number = explicit_match.group(1)
        return (title or f"TOP {number}", True)

    transition_match = re.search(
        r"(?:kommen\s+wir\s+(?:zu|zum|zur)|komme\s+ich\s+(?:zu|zum|zur)|"
        r"weiter\s+geht\s+es\s+(?:mit|um)|als\s+n(?:ä|ae)chstes|"
        r"n(?:ä|ae)chste(?:r|n|s)?\s+punkt|dann\s+haben\s+wir)\s+(?P<title>.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if not transition_match:
        return None
    title = _clean_detected_title(transition_match.group("title"))
    if len(title) < 4:
        return None
    return title, False


def _clean_detected_title(value: str) -> str:
    title = value.strip(" \t\n\r.:;-")
    title = re.sub(r"\b(?:rufe\s+ich|rufen\s+wir|ich\s+rufe)\b", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\b(?:auf|an)\s*$", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\b(?:bitte|dazu)\s*$", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s+", " ", title).strip(" \t\n\r.:;-")
    if len(title) > 120:
        title = title[:117].rstrip() + "..."
    return title


def _validate_unknown_segments(
    transcript_length: int,
    raw_segments: list[_RawSegment],
    *,
    fallback_segments: list[AssignmentSegment],
) -> tuple[list[AssignmentSegment], bool]:
    if transcript_length <= 0:
        return [], False

    candidates = list(raw_segments)
    if not candidates:
        candidates = [_segment_to_raw(segment) for segment in fallback_segments]

    candidates = [
        candidate for candidate in candidates if candidate.top_title.strip()
    ]
    candidates.sort(key=lambda item: item.start_index if item.start_index is not None else transcript_length)

    deduped: list[_RawSegment] = []
    seen_starts: set[int] = set()
    for candidate in candidates:
        start = candidate.start_index
        if start is not None and start in seen_starts:
            continue
        if start is not None:
            seen_starts.add(start)
        deduped.append(candidate)

    return _materialize_ordered_segments(
        transcript_length,
        deduped,
        tops=[candidate.top_title for candidate in deduped],
        heuristic_fallbacks=_match_fallback_segments(deduped, fallback_segments),
    )


def _validate_known_segments(
    transcript_length: int,
    tops: list[str],
    raw_segments: list[_RawSegment],
    heuristic_segments: list[AssignmentSegment],
) -> tuple[list[AssignmentSegment], bool]:
    if transcript_length <= 0:
        return [], False

    ordered_raw: list[_RawSegment] = []
    for index, top in enumerate(tops):
        if index < len(raw_segments):
            raw = raw_segments[index]
            ordered_raw.append(
                _RawSegment(
                    top_title=top,
                    start_index=raw.start_index,
                    end_index=raw.end_index,
                    confidence=raw.confidence,
                    evidence_text=raw.evidence_text,
                    uncertain=raw.uncertain,
                    reason=raw.reason,
                    transition_type=raw.transition_type,
                )
            )
        elif index < len(heuristic_segments):
            ordered_raw.append(_segment_to_raw(heuristic_segments[index], top_title=top))
        else:
            ordered_raw.append(
                _RawSegment(
                    top_title=top,
                    start_index=None,
                    end_index=None,
                    confidence=0.35,
                    evidence_text=None,
                    uncertain=True,
                    reason="Keine Grenze gefunden; Segment wurde aus der Reihenfolge geschaetzt.",
                    transition_type="inferred",
                )
            )

    return _materialize_ordered_segments(
        transcript_length,
        ordered_raw,
        tops=tops,
        heuristic_fallbacks=heuristic_segments,
    )


def _materialize_ordered_segments(
    transcript_length: int,
    raw_segments: list[_RawSegment],
    *,
    tops: list[str],
    heuristic_fallbacks: list[AssignmentSegment],
) -> tuple[list[AssignmentSegment], bool]:
    repaired = False
    starts: list[int] = []
    prepared: list[_RawSegment] = []

    for index, raw in enumerate(raw_segments):
        if len(starts) >= transcript_length:
            repaired = True
            break

        fallback = heuristic_fallbacks[index] if index < len(heuristic_fallbacks) else None
        fallback_start = fallback.start_index if fallback else None
        start = raw.start_index if raw.start_index is not None else fallback_start
        if start is None:
            remaining_segments = max(1, len(raw_segments) - index)
            remaining_lines = max(1, transcript_length - (starts[-1] + 1 if starts else 0))
            start = (starts[-1] + 1 if starts else 0) + max(0, remaining_lines // remaining_segments - 1)
            repaired = True

        minimum_start = starts[-1] + 1 if starts else 0
        if fallback_start is not None and (
            start < minimum_start or start < 0 or start >= transcript_length
        ):
            if fallback_start >= minimum_start:
                start = fallback_start

        original_start = start
        start = max(0, min(int(start), transcript_length - 1))
        if start < minimum_start:
            start = minimum_start
        if start != original_start:
            repaired = True
        if start >= transcript_length:
            repaired = True
            break

        starts.append(start)
        prepared.append(raw)

    segments: list[AssignmentSegment] = []
    for index, (raw, start) in enumerate(zip(prepared, starts)):
        next_start = starts[index + 1] if index + 1 < len(starts) else transcript_length
        max_end = max(start, next_start - 1)
        fallback = heuristic_fallbacks[index] if index < len(heuristic_fallbacks) else None
        raw_end = raw.end_index if raw.end_index is not None else (fallback.end_index if fallback else None)
        original_end = raw_end
        if raw_end is None:
            end = max_end
        else:
            end = max(start, min(int(raw_end), transcript_length - 1, max_end))
        if original_end is not None and end != original_end:
            repaired = True

        title = tops[index] if index < len(tops) and tops[index].strip() else raw.top_title
        segment_repaired = (
            raw.start_index != start
            or (raw.end_index is not None and raw.end_index != end)
            or raw.start_index is None
        )
        confidence = max(0.0, min(1.0, raw.confidence))
        if segment_repaired:
            confidence = min(confidence, 0.5)

        segments.append(
            AssignmentSegment(
                top_index=index,
                top_title=title,
                start_index=start,
                end_index=end,
                confidence=round(confidence, 2),
                uncertain=raw.uncertain or segment_repaired or confidence < 0.55,
                transition_type=raw.transition_type,
                reason=(
                    raw.reason
                    if not segment_repaired
                    else f"{raw.reason} Grenzen wurden validiert oder repariert."
                ),
                evidence_index=start,
                evidence_text=raw.evidence_text,
            )
        )

    return segments, repaired


def _match_fallback_segments(
    raw_segments: list[_RawSegment],
    fallback_segments: list[AssignmentSegment],
) -> list[AssignmentSegment]:
    if not fallback_segments:
        return []

    by_title = {
        normalize_text(segment.top_title): segment
        for segment in fallback_segments
        if segment.top_title.strip()
    }
    matched: list[AssignmentSegment] = []
    for index, raw in enumerate(raw_segments):
        fallback = by_title.get(normalize_text(raw.top_title))
        if fallback is None and index < len(fallback_segments):
            fallback = fallback_segments[index]
        if fallback is not None:
            matched.append(fallback)
    return matched


def _segment_to_raw(
    segment: AssignmentSegment,
    *,
    top_title: str | None = None,
) -> _RawSegment:
    return _RawSegment(
        top_title=top_title or segment.top_title,
        start_index=segment.start_index,
        end_index=segment.end_index,
        confidence=segment.confidence,
        evidence_text=segment.evidence_text,
        uncertain=segment.uncertain,
        reason=segment.reason,
        transition_type=segment.transition_type,
    )


def _coerce_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_confidence(value: Any, *, default: float) -> float:
    if value is None or isinstance(value, bool):
        return default
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _coerce_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
