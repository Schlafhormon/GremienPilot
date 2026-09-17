"""
Heuristic TOP assignment suggestions for meeting transcripts.

The implementation intentionally stays deterministic and explainable. It uses
moderator transition phrases and lightweight keyword overlap from the agenda
titles, then marks weak or inferred boundaries as uncertain.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

from agenda_labels import parse_agenda_label, reference_targets


TRANSITION_PATTERNS = [
    r"\btagesordnungspunkt\b",
    r"\btop\b",
    r"\bpunkt\b",
    r"\brufe\b.+\bauf\b",
    r"\bkomme(?:n)?\s+wir\s+zu\b",
    r"\bkommen\s+wir\s+zum\b",
    r"\bals\s+nachstes\b",
    r"\bnachste(?:n|r|s)?\b",
    r"\bnaechste(?:n|r|s)?\b",
    r"\bdann\s+haben\s+wir\b",
    r"\bweiter\s+geht\b",
    r"\babschliessend\b",
]

STOPWORDS = {
    "aber",
    "alle",
    "als",
    "am",
    "an",
    "auch",
    "auf",
    "aus",
    "bei",
    "beschluss",
    "beratungen",
    "berichten",
    "bericht",
    "bis",
    "das",
    "dem",
    "den",
    "der",
    "des",
    "die",
    "dies",
    "diese",
    "dieser",
    "dieses",
    "ein",
    "eine",
    "einer",
    "eines",
    "fur",
    "gegen",
    "haben",
    "im",
    "in",
    "ist",
    "mit",
    "nach",
    "nicht",
    "oder",
    "punkt",
    "sowie",
    "tagesordnungspunkt",
    "top",
    "und",
    "uber",
    "um",
    "von",
    "vorlage",
    "wir",
    "zu",
    "zum",
    "zur",
}


@dataclass(frozen=True)
class TranscriptUtterance:
    speaker: str
    text: str


@dataclass(frozen=True)
class BoundaryCandidate:
    top_index: int
    start_index: int
    confidence: float
    uncertain: bool
    transition_type: str
    reason: str
    evidence_index: int | None = None
    evidence_text: str | None = None


@dataclass(frozen=True)
class AssignmentSegment:
    top_index: int
    top_title: str
    start_index: int
    end_index: int
    confidence: float
    uncertain: bool
    transition_type: str
    reason: str
    evidence_index: int | None = None
    evidence_text: str | None = None


@dataclass(frozen=True)
class AssignmentSuggestionResult:
    suggested_assignments: list[int | None]
    segments: list[AssignmentSegment]
    strategy: str
    uncertain_count: int


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.lower().replace("ß", "ss"))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", ascii_text).strip()


def tokenize(value: str) -> list[str]:
    normalized = normalize_text(value)
    tokens = re.findall(r"[a-z0-9]{3,}", normalized)
    return [token for token in tokens if token not in STOPWORDS and not token.isdigit()]


def compact_token(token: str) -> str:
    for suffix in ("ungen", "ung", "lichkeit", "keiten", "ischen", "ische", "iger", "ige", "en", "er", "es", "e", "n", "s"):
        if len(token) > len(suffix) + 4 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def token_set(value: str) -> set[str]:
    return {compact_token(token) for token in tokenize(value)}


def extract_agenda_number(top: str, fallback: int | None = None) -> str | None:
    """Original number key; the legacy fallback argument is intentionally ignored."""
    return parse_agenda_label(top).number_key


def has_transition_phrase(text: str) -> bool:
    normalized = normalize_text(text)
    return any(re.search(pattern, normalized) for pattern in TRANSITION_PATTERNS)


def references_top_number(text: str, number: str | int | None) -> bool:
    if number is None:
        return False
    has_reference, targets = reference_targets(text, [f"TOP {number}"])
    return has_reference and targets == {0}


def speaker_transition_counts(transcript: list[TranscriptUtterance]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in transcript:
        if has_transition_phrase(line.text):
            counts[line.speaker] = counts.get(line.speaker, 0) + 1
    return counts


def likely_moderator_speakers(transcript: list[TranscriptUtterance]) -> set[str]:
    counts = speaker_transition_counts(transcript)
    if not counts:
        return set()
    maximum = max(counts.values())
    return {speaker for speaker, count in counts.items() if count == maximum and count > 0}


def keyword_overlap_score(line_tokens: set[str], top_tokens: set[str]) -> float:
    if not line_tokens or not top_tokens:
        return 0.0
    overlap = len(line_tokens & top_tokens)
    if overlap == 0:
        return 0.0
    return min(1.0, overlap / math.sqrt(len(top_tokens)))


def score_line_for_top(
    line: TranscriptUtterance,
    top: str,
    top_index: int,
    moderator_speakers: set[str],
    tops: list[str] | None = None,
) -> tuple[float, str, str]:
    top_number = extract_agenda_number(top)
    line_tokens = token_set(line.text)
    top_tokens = token_set(parse_agenda_label(top).title)
    overlap = keyword_overlap_score(line_tokens, top_tokens)
    transition = has_transition_phrase(line.text)
    moderator_bonus = 0.08 if line.speaker in moderator_speakers else 0.0

    has_reference, targets = reference_targets(line.text, tops if tops is not None else [top])
    target_index = top_index if tops is not None else 0
    if has_reference and targets != {target_index}:
        # An ambiguous or contradictory number must not become a strong keyword hit.
        return 0.0, "none", "TOP-Verweis ist mehrdeutig oder passt nicht zum TOP."

    if has_reference:
        confidence = min(0.98, 0.82 + moderator_bonus + (0.08 if transition else 0.0))
        return (
            confidence,
            "explicit",
            f"Expliziter Verweis auf TOP {top_number}.",
        )

    if transition and overlap > 0:
        confidence = min(0.9, 0.5 + (overlap * 0.25) + moderator_bonus)
        return (
            confidence,
            "explicit",
            "Moderations- oder Übergangsformulierung mit Begriffen aus dem TOP.",
        )

    if overlap >= 0.7:
        confidence = min(0.78, 0.45 + overlap * 0.25 + moderator_bonus)
        return confidence, "keyword", "Starker Begriffsabgleich mit dem TOP-Titel."

    if overlap >= 0.35:
        confidence = min(0.62, 0.34 + overlap * 0.25 + moderator_bonus)
        return confidence, "keyword", "Teilweiser Begriffsabgleich mit dem TOP-Titel."

    return 0.0, "none", "Keine belastbare Evidenz."


def find_boundary_for_top(
    transcript: list[TranscriptUtterance],
    tops: list[str],
    top_index: int,
    search_start: int,
    moderator_speakers: set[str],
) -> BoundaryCandidate | None:
    remaining_after = len(tops) - top_index - 1
    search_end = max(search_start, len(transcript) - remaining_after)
    best: tuple[float, int, str, str] | None = None

    for line_index in range(search_start, search_end):
        confidence, transition_type, reason = score_line_for_top(
            transcript[line_index],
            tops[top_index],
            top_index,
            moderator_speakers,
            tops,
        )
        if confidence <= 0:
            continue

        # Prefer earlier plausible boundaries. TOPs usually occur in agenda order,
        # and late keyword mentions inside an old TOP are common in debate.
        distance_penalty = min(0.18, (line_index - search_start) * 0.01)
        ranked_confidence = confidence - distance_penalty
        if best is None or ranked_confidence > best[0]:
            best = (ranked_confidence, line_index, transition_type, reason)

    if best is None:
        return None

    ranked_confidence, line_index, transition_type, reason = best
    confidence = max(0.0, min(1.0, ranked_confidence))
    return BoundaryCandidate(
        top_index=top_index,
        start_index=line_index,
        confidence=round(confidence, 2),
        uncertain=confidence < 0.55,
        transition_type=transition_type,
        reason=reason,
        evidence_index=line_index,
        evidence_text=transcript[line_index].text,
    )


def inferred_boundary(
    top_index: int,
    start_index: int,
    reason: str,
) -> BoundaryCandidate:
    return BoundaryCandidate(
        top_index=top_index,
        start_index=start_index,
        confidence=0.35,
        uncertain=True,
        transition_type="inferred",
        reason=reason,
    )


def build_segments(
    transcript: list[TranscriptUtterance],
    tops: list[str],
    boundaries: list[BoundaryCandidate],
) -> list[AssignmentSegment]:
    segments: list[AssignmentSegment] = []
    for index, boundary in enumerate(boundaries):
        next_start = boundaries[index + 1].start_index if index + 1 < len(boundaries) else len(transcript)
        end_index = max(boundary.start_index, next_start - 1)
        if boundary.start_index >= len(transcript):
            continue
        segments.append(
            AssignmentSegment(
                top_index=boundary.top_index,
                top_title=tops[boundary.top_index],
                start_index=boundary.start_index,
                end_index=min(end_index, len(transcript) - 1),
                confidence=boundary.confidence,
                uncertain=boundary.uncertain,
                transition_type=boundary.transition_type,
                reason=boundary.reason,
                evidence_index=boundary.evidence_index,
                evidence_text=boundary.evidence_text,
            )
        )
    return segments


def assignments_from_segments(
    transcript_length: int, segments: Iterable[AssignmentSegment]
) -> list[int | None]:
    assignments: list[int | None] = [None] * transcript_length
    for segment in segments:
        for index in range(segment.start_index, segment.end_index + 1):
            assignments[index] = segment.top_index
    return assignments


def suggest_assignments(
    transcript: list[TranscriptUtterance],
    tops: list[str],
) -> AssignmentSuggestionResult:
    valid_tops = [top.strip() for top in tops if top.strip()]
    if not transcript or not valid_tops:
        return AssignmentSuggestionResult(
            suggested_assignments=[None] * len(transcript),
            segments=[],
            strategy="heuristic_moderator_keyword",
            uncertain_count=0,
        )

    moderator_speakers = likely_moderator_speakers(transcript)
    first_has_reference, first_targets = reference_targets(transcript[0].text, valid_tops)
    first_ambiguous = first_has_reference and first_targets != {0}
    boundaries: list[BoundaryCandidate] = [
        BoundaryCandidate(
            top_index=0,
            start_index=0,
            confidence=0.35 if first_ambiguous else 0.6,
            uncertain=first_ambiguous,
            transition_type="inferred",
            reason=("Mehrdeutiger oder unbekannter TOP-Verweis am Transkriptanfang; Zuordnung prüfen."
                    if first_ambiguous else "Erster TOP beginnt am Anfang des Transkripts."),
            evidence_index=0,
            evidence_text=transcript[0].text,
        )
    ]

    search_start = 1
    for top_index in range(1, len(valid_tops)):
        candidate = find_boundary_for_top(
            transcript,
            valid_tops,
            top_index,
            search_start,
            moderator_speakers,
        )
        if candidate is None:
            remaining_topics = len(valid_tops) - top_index
            remaining_lines = max(1, len(transcript) - search_start)
            fallback_start = min(
                len(transcript) - 1,
                search_start + max(1, remaining_lines // (remaining_topics + 1)),
            )
            candidate = inferred_boundary(
                top_index,
                fallback_start,
                "Keine klare Ankündigung gefunden; Grenze wurde aus der Reihenfolge der TOPs geschätzt.",
            )
        if candidate.start_index <= boundaries[-1].start_index:
            candidate = inferred_boundary(
                top_index,
                min(len(transcript) - 1, boundaries[-1].start_index + 1),
                "Grenze wurde angepasst, damit TOPs ohne Überlappung in Reihenfolge bleiben.",
            )
        boundaries.append(candidate)
        search_start = min(len(transcript), candidate.start_index + 1)

    segments = build_segments(transcript, valid_tops, boundaries)
    assignments = assignments_from_segments(len(transcript), segments)
    uncertain_count = sum(1 for segment in segments if segment.uncertain)

    return AssignmentSuggestionResult(
        suggested_assignments=assignments,
        segments=segments,
        strategy="heuristic_moderator_keyword",
        uncertain_count=uncertain_count,
    )
