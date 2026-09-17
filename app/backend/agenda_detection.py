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
    normalize_text,
    score_line_for_top,
    transition_kind,
    suggest_assignments,
)
from agenda_labels import parse_agenda_label, agenda_references, reference_sections, reference_targets, with_section
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
      "top_id": "agenda:0",
      "top_title": "Originalnummer und Titel, z.B. 2.1. Schulbau",
      "start_index": 0,
      "end_index": 4,
      "confidence": 0.0,
      "evidence_index": 0,
      "evidence_text": "wörtlicher Beleg aus dieser Transkriptzeile",
      "uncertain": false
    }
  ]
}

Regeln:
- Indizes sind 0-basiert und beziehen sich auf die Transkriptzeilen.
- TOPs muessen in Transkriptreihenfolge stehen.
- Erhalte explizite Originalnummern einschließlich Unterpunkten. Erfinde keine Nummern.
- Bei bekanntem Abschnitt Titel mit [Öffentlich] oder [Nichtöffentlich] beginnen.
- Listenindizes sind keine TOP-Nummern. Wiederholte Nummern sind ohne eindeutigen Abschnitt mehrdeutig.
- Mehrdeutige Nummernverweise und mehrere TOP-Verweise in einer Zeile sind uncertain=true.
- Bei bekannter Agenda top_id exakt aus der TOP-Liste kopieren; sie ist keine TOP-Nummer.
- top_title muss zum Originaltitel dieser ID passen; niemals über die Antwortposition zuordnen.
- Ohne bekannte Agenda top_id weglassen.
- start_index und end_index sind inklusive, beide als ganze JSON-Zahlen erforderlich.
- evidence_index muss innerhalb des Segments liegen; evidence_text wörtlich aus dieser Zeile kopieren.
- Eine ID belegt nur die Identität, nicht die inhaltliche Richtigkeit der Grenzen.
- Vorgezogene oder wiederaufgenommene TOPs dürfen in anderer Reihenfolge und mehrfach vorkommen.
- Ohne Evidenz TOPs auslassen und Transkriptzeilen unzugeordnet lassen; keine Grenzen interpolieren.
- Vorschauen, Rückverweise, Negationen und zitierte Aufrufe sind keine aktuellen Aufrufe.
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
    validation_reasons: list[str] = field(default_factory=list)

    @property
    def warnings(self) -> list[str]:
        warnings = []
        if self.failed_calls:
            warnings.append(
                f"TOP-Erkennung: {self.failed_calls} von {self.attempted_calls} LLM-Aufrufen "
                f"fehlgeschlagen ({', '.join(self.failure_reasons)}). "
                "Heuristische Ersatzverarbeitung verwendet; bitte TOP-Zuordnung prüfen."
            )
        if self.validation_reasons:
            warnings.append(
                "TOP-Erkennung: LLM-Segmente nur eingeschränkt übernommen "
                f"({', '.join(self.validation_reasons)}). Bitte TOP-Zuordnung prüfen."
            )
        return warnings


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
    top_id: str | None = None
    id_provided: bool = False
    evidence_index: int | None = None
    evidence_index_provided: bool = False


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
            transcript,
            llm_segments,
            fallback_segments=heuristic_segments,
            issues=usage.validation_reasons,
        )
        strategy = "heuristic_transcript_llm_repaired" if repaired else "heuristic_transcript_llm"
    else:
        segments, repaired = heuristic_segments, False
        strategy = "heuristic_transcript_fallback" if not usage.enabled else "heuristic_transcript_llm_fallback"
        if repaired:
            strategy += "_repaired"

    # Repeated announcements of the same detected label share an identity.
    detected_tops = list(dict.fromkeys(segment.top_title for segment in segments))
    segments = [replace(segment, top_index=detected_tops.index(segment.top_title)) for segment in segments]
    segments = _guard_number_evidence(transcript, detected_tops, segments)
    return _result_from_segments(len(transcript), segments, strategy, tops=detected_tops, usage=usage)


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
            transcript,
            valid_tops,
            llm_segments,
            heuristic_segments,
            issues=usage.validation_reasons,
        )
        strategy = "known_agenda_heuristic_llm_repaired" if repaired else "known_agenda_heuristic_llm"
    else:
        # Deterministic segments already preserve identity, gaps and revisits.
        segments, repaired = heuristic_segments, False
        strategy = "known_agenda_heuristic"
        if usage.enabled:
            strategy += "_llm_fallback"
        if repaired:
            strategy += "_repaired"

    segments = _guard_number_evidence(transcript, valid_tops, segments)
    return _result_from_segments(len(transcript), segments, strategy, tops=valid_tops, usage=usage)


def _guard_number_evidence(
    transcript: list[TranscriptUtterance], tops: list[str], segments: list[AssignmentSegment],
) -> list[AssignmentSegment]:
    """The optional LLM cannot promote ambiguous references to certain hits."""
    guarded = []
    for segment in segments:
        evidence = [transcript[segment.start_index].text, segment.evidence_text or ""]
        contradictory_number = any(
            reference_targets(text, tops)[0]
            and score_line_for_top(
                TranscriptUtterance("", text), tops[segment.top_index], segment.top_index, tops,
            )[0] < 0.7
            for text in evidence if text
        )
        blocked_act = any(transition_kind(text) in {"mention", "stop", "mixed"} for text in evidence if text)
        if contradictory_number or blocked_act:
            segment = replace(
                segment, uncertain=True, confidence=min(segment.confidence, 0.5),
                transition_type="inferred",
                reason="Kein eindeutiger aktueller TOP-Aufruf; Zuordnung prüfen.",
            )
        guarded.append(segment)
    return guarded


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
        replace(
            segment,
            start_index=(
                segment.start_index + offset
                if segment.start_index is not None
                else None
            ),
            end_index=(
                segment.end_index + offset if segment.end_index is not None else None
            ),
            evidence_index=(segment.evidence_index + offset if segment.evidence_index is not None else None),
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
        valid_segments = [segment for segment in segments
                          if segment.start_index is not None and segment.end_index is not None
                          and 0 <= segment.start_index <= segment.end_index < len(chunk)]
        if len(valid_segments) != len(segments) and "invalid_bounds" not in usage.validation_reasons:
            usage.validation_reasons.append("invalid_bounds")
        segments = valid_segments
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
    # Identical proposals from overlapping context windows are one observation.
    return [] if usage.failed_calls == usage.attempted_calls else list(dict.fromkeys(detected))


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
        segments = _parse_llm_segments(raw_response)
        # Unknown agendas still require a title; ID-only entries are meaningful
        # solely against the supplied known agenda.
        return segments if tops is not None else [segment for segment in segments if segment.top_title]
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
            **({"top_id": f"agenda:{segment.top_index}"} if tops else {}),
            "top_title": segment.top_title,
            "evidence_index": segment.evidence_index,
            "start_index": segment.start_index,
            "end_index": segment.end_index,
            "confidence": segment.confidence,
            "uncertain": segment.uncertain,
            "evidence_text": segment.evidence_text,
        }
        for segment in heuristic_segments
    ]

    if tops:
        # Request-local references, independent of persisted session IDs and TOP numbers.
        agenda = json.dumps([
            {"top_id": f"agenda:{index}", "top_title": top}
            for index, top in enumerate(tops)
        ], ensure_ascii=False)
        compact_note = (
            "Das Transkript ist auf relevante Kontextfenster gekuerzt; "
            "die angezeigten Indizes bleiben die originalen Transkriptindizes. "
            if compacted
            else ""
        )
        task = (
            "Bekannte TOP-Liste. Pruefe und verbessere die Segmentgrenzen. "
            f"{compact_note}"
            "Gib belegte Segmente in Transkriptreihenfolge mit top_id und dem Originaltitel zurück. "
            "TOPs dürfen fehlen oder mehrfach auftreten; Lücken sind erlaubt.\n\n"
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
        raw_items = payload.get("tops", payload.get("segments", []))
    elif isinstance(payload, list):
        raw_items = payload
    else:
        raise ValueError("Expected segment list")
    if not isinstance(raw_items, list):
        raise ValueError("Expected segment list")

    segments: list[_RawSegment] = []
    for item in raw_items:
        if not isinstance(item, dict):
            raise ValueError("Expected segment object")
        title = item.get("top_title", item.get("title", ""))
        title = title.strip() if isinstance(title, str) else ""
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
                top_id=item.get("top_id") if isinstance(item.get("top_id"), str) else None,
                id_provided="top_id" in item,
                evidence_index=_coerce_int(item.get("evidence_index")),
                evidence_index_provided="evidence_index" in item,
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
    labels = [
        parsed[0] for line in transcript
        if (parsed := _parse_heuristic_top_announcement(line.text)) is not None
    ]
    titles = list(dict.fromkeys(_canonical_unknown_labels(labels).values()))
    segments = suggest_assignments(transcript, titles).segments
    return [
        replace(segment, confidence=min(segment.confidence, 0.65), uncertain=True,
                reason="Mögliche TOP-Überschrift ohne bekannte Agenda; Titel und Bereich prüfen.")
        if transition_kind(transcript[segment.start_index].text) == "heading" else segment
        for segment in segments
    ]


def _parse_heuristic_top_announcement(text: str) -> tuple[str, bool] | None:
    if not has_transition_phrase(text):
        return None

    refs = agenda_references(text)
    if refs:
        if len(refs) != 1 or refs[0].number is None:
            return None
        ref = refs[0]
        title = _clean_detected_title(text[ref.end:])
        label = f"TOP {ref.original_number}" + (f" {title}" if title else "")
        sections = reference_sections(text)
        if len(sections) == 1:
            label = with_section(label, next(iter(sections)))
        if reference_targets(text, [label])[1] != {0}:
            return None
        return label, True

    transition_match = re.search(
        r"(?:kommen\s+wir\s+(?:(?:jetzt|nun|wieder|zurück)\s+)*(?:zu|zum|zur)|komme\s+ich\s+(?:zu|zum|zur)|"
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


def _canonical_unknown_labels(labels: list[str]) -> dict[str, str]:
    """Reconcile spelling and omitted metadata, never conflicting identities.

    A later "TOP 1" can refer back to "TOP 1 Haushalt". Different named
    topics sharing a number remain separate and therefore ambiguous.
    """
    unique = list(dict.fromkeys(labels))
    parsed = {value: parse_agenda_label(value) for value in unique}
    mapping: dict[str, str] = {}
    for value, label in parsed.items():
        candidates = [other for other, known in parsed.items()
                      if known.number_key == label.number_key and known.section == label.section
                      and (not label.title or normalize_text(known.title) == normalize_text(label.title))]
        named = [other for other in candidates if parsed[other].title]
        if len({normalize_text(parsed[other].title) for other in named}) == 1:
            mapping[value] = named[0]
        else:
            mapping[value] = candidates[0] if not named and candidates else value
    # A unique fully labelled title may enrich an unnumbered model response.
    for value, label in parsed.items():
        if label.number_key is not None or not label.title:
            continue
        candidates = list(dict.fromkeys(mapping[other] for other, known in parsed.items()
                          if known.number_key is not None
                          and normalize_text(known.title) == normalize_text(label.title)
                          and (label.section is None or label.section == known.section)))
        if len(candidates) == 1:
            mapping[value] = candidates[0]
    return mapping


def _validate_unknown_segments(
    transcript: list[TranscriptUtterance],
    raw_segments: list[_RawSegment],
    *,
    fallback_segments: list[AssignmentSegment],
    issues: list[str],
) -> tuple[list[AssignmentSegment], bool]:
    """Free detection obeys the same bounds and evidence rules as known TOPs.

    Temporary identities only associate the returned labels. They do not prove
    that a topic occurred. Heuristic evidence is validated independently.
    """
    mapping = _canonical_unknown_labels(
        [segment.top_title for segment in fallback_segments]
        + [raw.top_title for raw in raw_segments if raw.top_title]
    )
    tops = list(dict.fromkeys(mapping.values()))
    raw_with_ids = [replace(raw, top_title=mapping[raw.top_title],
                           top_id=f"agenda:{tops.index(mapping[raw.top_title])}", id_provided=True)
                    for raw in raw_segments if raw.top_title]
    fallbacks = [replace(segment, top_title=mapping[segment.top_title],
                         top_index=tops.index(mapping[segment.top_title]))
                 for segment in fallback_segments]
    return _validate_known_segments(transcript, tops, raw_with_ids, fallbacks, issues=issues)


def _known_title_matches(title: str, tops: list[str]) -> list[int]:
    """Legacy labels must match exactly after normalization, never fuzzily.

    Supplied numbers and sections must match; even an exact full label cannot
    hide another candidate with the same title and unspecified metadata.
    """
    label = parse_agenda_label(title)
    if not label.title and label.number_key is None:
        return []
    return [
        i for i, top in enumerate(tops)
        if normalize_text((known := parse_agenda_label(top)).title) == normalize_text(label.title)
        and (label.number_key is None or label.number_key == known.number_key)
        and (label.section is None or label.section == known.section)
    ]


def _validate_known_segments(
    transcript: list[TranscriptUtterance],
    tops: list[str],
    raw_segments: list[_RawSegment],
    heuristic_segments: list[AssignmentSegment],
    *,
    issues: list[str] | None = None,
) -> tuple[list[AssignmentSegment], bool]:
    """Validate identities before boundaries; never interpolate or clip LLM ranges.

    All overlapping proposals are rejected, with no response-order winner.
    Independent heuristic supplements retain their identity and complete range.
    A grounded quotation proves provenance, not semantic correctness.
    """
    issues = issues if issues is not None else []

    def note(code: str) -> None:
        if code not in issues:
            issues.append(code)

    mapped: list[AssignmentSegment] = []
    identities = {f"agenda:{i}": i for i in range(len(tops))}
    for raw in raw_segments:
        matches = _known_title_matches(raw.top_title, tops)
        if raw.id_provided:
            top_index = identities.get(raw.top_id)
            # Never rescue a bad ID using its title. Supplied titles must agree.
            if top_index is None or (raw.top_title and top_index not in matches):
                note("invalid_identity")
                continue
        elif len(matches) == 1:
            top_index = matches[0]
        else:
            note("invalid_identity")
            continue
        start, end = raw.start_index, raw.end_index
        if start is None or end is None or not 0 <= start <= end < len(transcript):
            note("invalid_bounds")
            continue

        # Only a literal excerpt from a single line inside the range is grounded.
        evidence_index = raw.evidence_index
        quote = raw.evidence_text
        if not raw.evidence_index_provided and quote:
            hits = [i for i in range(start, end + 1) if quote in transcript[i].text]
            evidence_index = hits[0] if len(hits) == 1 else None
        grounded = bool(
            quote and evidence_index is not None and start <= evidence_index <= end
            and quote in transcript[evidence_index].text
        )
        if not grounded:
            note("unverified_evidence")
            evidence_index, quote = None, None
        # Use full lines so an excerpt cannot hide negation or a conflicting TOP.
        support = [i for i in {start, evidence_index} if i is not None]
        supported_topics = {
            line: [i for i, top in enumerate(tops)
                   if score_line_for_top(transcript[line], top, i, tops)[0] >= 0.7]
            for line in range(start, end + 1)
        }
        if any(len(supported_topics[line]) == 1 and top_index not in supported_topics[line]
               for line in support):
            note("contradictory_evidence")
            continue
        strong = (
            grounded
            and all(supported_topics[line] == [top_index] for line in support)
            and not any(targets and targets != [top_index] for targets in supported_topics.values())
            and not any(
                (transition_kind(transcript[line].text) in {"call", "heading", "mixed"}
                 and supported_topics[line] != [top_index])
                or (transition_kind(transcript[line].text) == "stop"
                    and not (reference_targets(transcript[line].text, tops)[1]
                             and top_index not in reference_targets(transcript[line].text, tops)[1]))
                for line in range(start, end + 1)
            )
        )
        if not strong:
            note("weak_boundary_evidence")
        mapped.append(AssignmentSegment(
            top_index=top_index, top_title=tops[top_index], start_index=start, end_index=end,
            confidence=min(raw.confidence, 0.5) if not strong else raw.confidence,
            uncertain=raw.uncertain or not strong or raw.confidence < 0.7,
            transition_type="llm" if strong else "inferred",
            reason=raw.reason + (" Grenzen inhaltlich nicht eindeutig belegt; Zuordnung prüfen." if not strong else ""),
            evidence_index=evidence_index, evidence_text=quote,
        ))

    # Reject every member of an overlap, including duplicate ranges for one TOP.
    # Disjoint revisits of the same identity are valid.
    conflicts: set[int] = set()
    for i, left in enumerate(mapped):
        for j in range(i + 1, len(mapped)):
            right = mapped[j]
            if left.start_index <= right.end_index and right.start_index <= left.end_index:
                conflicts.update((i, j))
    if conflicts:
        note("overlapping_segments")
    segments = [segment for i, segment in enumerate(mapped) if i not in conflicts]

    # Only supplement identities absent from accepted LLM segments. Never extend
    # an accepted range into a gap, or clip a fallback to make it fit.
    accepted_ids = {segment.top_index for segment in segments}
    for fallback in heuristic_segments:
        if fallback.top_index in accepted_ids or any(
            fallback.start_index <= segment.end_index and segment.start_index <= fallback.end_index
            for segment in segments
        ):
            continue
        segments.append(replace(
            fallback, uncertain=True, confidence=min(fallback.confidence, 0.5),
            reason="Unabhängige heuristische Ergänzung; LLM-Zuordnung fehlt oder wurde verworfen.",
        ))
        note("heuristic_supplement")
    segments.sort(key=lambda segment: segment.start_index)
    return segments, bool(issues)


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
        evidence_index=segment.evidence_index,
        evidence_index_provided=segment.evidence_index is not None,
    )


def _coerce_int(value: Any) -> int | None:
    # Never truncate fractions, parse strings, or treat booleans as indices.
    return value if type(value) is int else None


def _coerce_confidence(value: Any, *, default: float) -> float:
    if value is None or isinstance(value, bool):
        return default
    try:
        number = float(value)
        return max(0.0, min(1.0, number)) if math.isfinite(number) else default
    except (TypeError, ValueError, OverflowError):
        return default


def _coerce_optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None
