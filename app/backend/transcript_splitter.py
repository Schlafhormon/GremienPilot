"""Utilities for making transcript lines granular enough for agenda detection."""

from __future__ import annotations

import math
import re
import uuid
from typing import Any


SENTENCE_BOUNDARY_PATTERN = re.compile(
    r"(?<=[.!?])\s+(?=[A-ZÄÖÜ])",
    flags=re.UNICODE,
)

MIN_SPLIT_LINE_CHARS = 80
MIN_SENTENCE_CHARS = 8


def split_transcript_for_agenda_detection(
    transcript: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Split long transcript lines into sentence-like chunks before TOP detection.

    This deliberately does not look for TOP phrases. It only gives the downstream
    agenda detector smaller units so a TOP boundary can fall between chunks
    instead of being forced to cover one whole speaker turn.

    All non-whitespace content is retained, including short trailing fragments.
    Interior times are estimates proportional to character offsets, not measured
    sentence/word boundaries; their decimal places do not imply audio precision.
    Valid source start/end times are retained as the outer bounds. Missing or
    invalid times fall back to a finite, non-negative-duration interval.
    Splitting the result again preserves its text, IDs and estimated times.
    """
    split_lines: list[dict[str, Any]] = []

    for line in transcript:
        text = str(line.get("text", "")).strip()
        chunks = _sentence_chunks(text)
        if len(chunks) <= 1:
            split_lines.append(dict(line))
            continue

        start_time = _coerce_float(line.get("start"), 0.0)
        end_time = max(start_time, _coerce_float(line.get("end"), start_time))
        duration = end_time - start_time
        text_length = max(1, len(text))

        for index, (start_offset, end_offset, chunk) in enumerate(chunks):
            part_line = dict(line)
            if index > 0 or not part_line.get("line_id"):
                part_line["line_id"] = str(uuid.uuid4())
            part_line["text"] = chunk
            part_line["start"] = (
                start_time
                if index == 0
                else start_time + duration * (start_offset / text_length)
            )
            part_line["end"] = (
                end_time
                if index == len(chunks) - 1
                else start_time + duration * (end_offset / text_length)
            )
            split_lines.append(part_line)

    return split_lines


def split_transcript_at_agenda_transitions(
    transcript: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Backward-compatible wrapper for the generic agenda granularity splitter."""
    return split_transcript_for_agenda_detection(transcript)


def _sentence_chunks(text: str) -> list[tuple[int, int, str]]:
    if len(text) < MIN_SPLIT_LINE_CHARS:
        return [(0, len(text), text)] if text else []

    chunks: list[tuple[int, int, str]] = []
    start = 0
    for match in SENTENCE_BOUNDARY_PATTERN.finditer(text):
        end = match.end()
        chunk = text[start:end].strip()
        if len(chunk) >= MIN_SENTENCE_CHARS:
            chunks.append((start, end, chunk))
            start = match.end()

    tail = text[start:].strip()
    # The minimum size controls granularity, never whether text is retained.
    # Keep a short tail separately so a long preceding chunk cannot be split
    # again on the next pass, changing its ID and interpolating its times again.
    if tail:
        chunks.append((start, len(text), tail))

    return chunks if len(chunks) > 1 else [(0, len(text), text)]


def _coerce_float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default
