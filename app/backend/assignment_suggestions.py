"""Compatible assignment types; all automatic decisions belong to models."""
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class TranscriptUtterance:
    speaker: str
    text: str
    line_id: str | None = None
    start: float | None = None
    end: float | None = None


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


def assignments_from_segments(transcript_length: int, segments: Iterable[AssignmentSegment]) -> list[int | None]:
    groups = [set() for _ in range(transcript_length)]
    for segment in segments:
        if not 0 <= segment.start_index <= segment.end_index < transcript_length:
            raise ValueError('invalid_segment_range')
        for index in range(segment.start_index, segment.end_index + 1):
            groups[index].add(segment.top_index)
    # The legacy scalar API cannot represent joint deliberations.
    return [next(iter(group)) if len(group) == 1 else None for group in groups]


def suggest_assignments(transcript, tops):
    from agenda_detection import segment_known_agenda
    result = segment_known_agenda(transcript, tops, use_llm=True)
    return AssignmentSuggestionResult(result.assignments, result.segments, result.strategy, result.uncertain_count)
