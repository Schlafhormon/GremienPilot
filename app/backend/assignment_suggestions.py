"""
Heuristic TOP assignment suggestions for meeting transcripts.

The implementation intentionally stays deterministic and explainable. It uses
moderator transition phrases and lightweight keyword overlap from the agenda
titles. Missing evidence leaves gaps; weak title matches remain local and uncertain.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace
from typing import Iterable

from agenda_labels import agenda_references, parse_agenda_label, reference_targets


# Speech acts, not occurrences of "TOP", determine transitions. These patterns
# express present performatives; tense/modality/polarity are checked separately.
CALL = re.compile(
    r"^(?:(?:so|gut|also|und|damit|somit|dann|nun|jetzt|abschliessend|als n(?:a|ae)chstes)[,:]?\s+)*"
    r"(?:komme(?:n)?\s+(?:(?:wir|ich)\s+)?(?:(?:jetzt|nun|auch|hier|schon|wieder|zuruck)\s+)*(?:zu|zum|zur)\b"
    r"|(?:wir\s+kommen|ich\s+komme)\s+(?:(?:jetzt|nun|auch|hier|schon|wieder|zuruck)\s+)*(?:zu|zum|zur)\b"
    r"|(?:ich\s+rufe|rufe\s+ich|wir\s+rufen|rufen\s+wir)\b.+\bauf\b"
    r"|(?:ich\s+eroffne|wir\s+eroffnen)\b"
    r"|weiter\s+geht\s+es\s+(?:mit|um)\b"
    r"|(?:wir\s+)?(?:behandeln|beraten|besprechen)\s+(?:wir\s+)?(?:jetzt|nun)\b"
    r"|(?:wir\s+)?(?:nehmen|setzen)\b.+\b(?:wieder\s+auf|fort)\b"
    r"|n(?:a|ae)chste(?:r|s)?\s+(?:punkt|top|tagesordnungspunkt)\b)"
)
NON_CURRENT = re.compile(
    r"\b(?:nicht(?![-\s]*(?:o|oe)ffentlich)|kein\w*|spater|spaeter|nachher|morgen|"
    r"anschliessend|danach|zuvor|vorhin|bereits|damals|gestern|"
    r"hatten|haben|wurde\w*|war|waren|warst|wuerde\w*|"
    r"konnten|koennten|soll\w*|wollen|mochte\w*|moechte\w*|"
    r"bevor|wenn|falls|sobald|"
    r"erwahnt\w*|erwaehnt\w*|zitiert\w*|"
    r"n(?:a|ae)chste[nr]?\s+(?:sitzung|woche|monat))\b"
)
STOP = re.compile(
    r"^(?:damit\s+)?(?:ich\s+schliesse|wir\s+(?:beenden|unterbrechen|vertagen))\b"
    r"|\b(?:top|tagesordnungspunkt|beratung|sitzung)\b.*\b(?:abgeschlossen|beendet|unterbrochen)\b"
    r"|^(?:top|tagesordnungspunkt)\b.*\b(?:wird\s+(?:vertagt|verschoben|abgesetzt)|ist\s+abgesetzt|entfallt|entfaellt)\b"
)
# A bare agenda heading is useful evidence, but a sentence about a TOP is not.
HEADING_PREDICATE = re.compile(
    r"\b(?:ist|sind|hat|enthalt|enthaelt|betrifft|kostet|zeigt|steht|geht|"
    r"behandelt|bespricht|kommt|bleibt|fehlt|steigt|sinkt|braucht|sagt)\b"
)


def transition_kind(text: str) -> str:
    """Classify a whole assignable row conservatively.

    Mixed/negated/future/reported utterances are not positive boundary evidence.
    We intentionally do not try to assign two clauses within a single row.
    """
    normalized = normalize_text(text)
    # Hedges before a present performative do not turn it into reported speech.
    normalized = re.sub(r'^ich (?:denke|glaube|meine),?\s*(?:dass\s+)?', '', normalized)
    # An explicitly resolved invitation for further remarks, followed by a
    # polite current call, is not a merely hypothetical future agenda change.
    resolved = re.match(r'^(?:das\s+)?ist\s+(?:jetzt\s+)?nicht\s+der\s+fall[,.:]?\s+', normalized)
    if resolved:
        remainder = normalized[resolved.end():]
        polite = re.match(r'^(?:dann\s+)?wurde\s+ich\s+((?:(?:jetzt|nun|auch|hier|schon)\s+)*'
                          r'(?:zu|zum|zur)\b.+?)\s+kommen\b', remainder)
        if polite:
            normalized = 'ich komme ' + polite[1] + remainder[polite.end():]
    if (agenda_references(text)
            and re.search(r'\b(?:weitere|weiteren|noch)\s+(?:anfragen|fragen|informationen|wortmeldungen)\b', normalized)
            and not NON_CURRENT.search(normalized) and not re.search(r'[„“"«»]', text)):
        return 'continuation'
    # A conditional invitation for further remarks is a current continuation,
    # not a new call and not a prohibition on correcting a previous model label.
    if (agenda_references(text) and re.search(r'\b(?:habe|hat|gibt)\b.{0,40}\b(?:hinweis|frage|anmerkung|wortmeldung)', normalized)
            and not re.search(r'\b(?:gestern|damals|vorhin|protokoll|niederschrift|nachste\w* sitzung)\b', normalized)
            and not re.search(r'[„“"«»]', text)):
        return 'continuation'
    blocked = bool(NON_CURRENT.search(normalized) or "?" in text or re.search(r'[„“"«»]', text))
    if not blocked and STOP.search(normalized):
        return "stop"
    deferred = r"\b(?:werde\w*|wird|verschieb\w*|vertag\w*|abgesetzt|entfall\w*)\b"
    if blocked or re.search(deferred, normalized):
        # A row containing both an actual call and a separate preview cannot be
        # split by the assignment model. Do not carry the old topic across it.
        clauses = re.split(r"[.!?;]\s+(?!\d)", normalized)
        if len(clauses) > 1 and any(
            CALL.search(clause) and not NON_CURRENT.search(clause)
            and not re.search(deferred, clause) for clause in clauses
        ):
            return "mixed"
        return "mention"
    if CALL.search(normalized):
        return "call"
    refs = agenda_references(text)
    if refs and refs[0].start == 0 and not HEADING_PREDICATE.search(normalized):
        return "heading"
    if refs:
        return 'mention'
    return "none"


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
    return transition_kind(text) in {"call", "heading"}


def references_top_number(text: str, number: str | int | None) -> bool:
    if number is None:
        return False
    has_reference, targets = reference_targets(text, [f"TOP {number}"])
    return has_reference and targets == {0}


def keyword_overlap_score(line_tokens: set[str], top_tokens: set[str]) -> float:
    if not line_tokens or not top_tokens:
        return 0.0
    overlap = len(line_tokens & top_tokens)
    if overlap == 0:
        return 0.0
    return overlap / len(top_tokens)


def score_line_for_top(
    line: TranscriptUtterance,
    top: str,
    top_index: int,
    tops: list[str] | None = None,
) -> tuple[float, str, str]:
    # Speaker frequency is not independent evidence: repeated mentions used to
    # manufacture a moderator bonus and turn a weak hit into a certain one.
    kind = transition_kind(line.text)
    if kind in {"mention", "stop", "mixed"}:
        return 0.0, "none", "Kein aktueller TOP-Aufruf."
    agenda = tops if tops is not None else [top]
    target_index = top_index if tops is not None else 0
    label = parse_agenda_label(top)
    top_tokens = token_set(label.title)
    overlap = keyword_overlap_score(token_set(line.text), top_tokens)
    has_reference, targets = reference_targets(line.text, agenda)
    if has_reference:
        # A heading must contain only its title (plus section metadata).
        # Sentences beginning with a TOP reference remain mentions.
        if kind == "heading":
            tail = line.text[agenda_references(line.text)[0].end:]
            tail = re.sub(r"\b(?:nicht[- ]*)?(?:ö|oe|o)ffentlich\w*\b", "", tail, flags=re.I)
            if token_set(tail) - top_tokens:
                return 0.0, "none", "Satz über einen TOP, kein eindeutiger Aufruf."
        if targets != {target_index}:
            # Unnumbered agendas can match a title, never an assumed list number.
            # Unknown/ambiguous numbers in a numbered agenda cannot be rescued by keywords.
            refs = agenda_references(line.text)
            if (targets or label.number_key is not None or len(refs) != 1
                    or refs[0].number is None or overlap < 0.7
                    or reference_targets(line.text, [f"TOP {refs[0].number}"])[1] != {0}):
                return 0.0, "none", "TOP-Verweis ist mehrdeutig oder passt nicht zum TOP."
            if kind in {"call", "heading"}:
                return 0.75, "explicit", "Aktueller Aufruf mit passendem Titel; TOP-Nummer ist nicht hinterlegt."
        elif kind in {"call", "heading"}:
            if overlap < 0.7 and any(
                index != target_index
                and keyword_overlap_score(token_set(line.text), token_set(parse_agenda_label(other).title)) >= 0.7
                for index, other in enumerate(agenda)
            ):
                return 0.0, "none", "TOP-Nummer und Titel widersprechen sich."
            return (0.9 if kind == "call" else 0.8), "explicit", f"Aktueller Aufruf von TOP {label.original_number}."
        # A reference in running speech is not even a local title suggestion.
        return 0.0, "none", "Bloße Erwähnung einer TOP-Nummer."
    if kind == "call" and overlap >= 0.7:
        return 0.8, "explicit", "Aktueller Aufruf mit passendem TOP-Titel."
    if overlap >= 0.7:
        return 0.5, "keyword", "Nur Titelähnlichkeit in dieser Zeile; kein belegter Segmentbeginn."
    return 0.0, "none", "Keine belastbare Evidenz."


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
    """Follow observed calls in transcript order, allowing gaps and revisits.

    Confidence is an evidence score for the boundary, not a calibrated probability
    for every line. Explicit calls persist until the next call/stop/conflict;
    keyword-only suggestions cover one row and are always uncertain.
    """
    segments: list[AssignmentSegment] = []
    active: int | None = None  # index into segments, not agenda order
    for line_index, line in enumerate(transcript):
        kind = transition_kind(line.text)
        if kind == "stop":
            has_ref, targets = reference_targets(line.text, tops)
            if active is not None and has_ref and targets and segments[active].top_index not in targets:
                segments[active] = replace(segments[active], end_index=line_index)
            else:
                active = None
            continue
        scores = sorted(
            [(score_line_for_top(line, top, index, tops), index)
             for index, top in enumerate(tops) if top.strip()],
            key=lambda item: item[0][0], reverse=True,
        )
        best, top_index = scores[0] if scores else ((0.0, "none", ""), -1)
        confidence, transition_type, reason = best
        ambiguous = len(scores) > 1 and scores[1][0][0] > 0 and confidence - scores[1][0][0] < 0.15
        if kind in {"call", "heading", "mixed"}:
            # An unresolvable call interrupts the previous topic too. It must not
            # inherit its confident assignment or an agenda-order guess.
            active = None
        if confidence > 0 and not ambiguous:
            if active is not None and segments[active].top_index == top_index:
                segments[active] = replace(segments[active], end_index=line_index)
                continue
            active = None
            segments.append(AssignmentSegment(
                top_index=top_index, top_title=tops[top_index].strip(),
                start_index=line_index, end_index=line_index,
                confidence=confidence, uncertain=transition_type != "explicit",
                transition_type=transition_type, reason=reason,
                evidence_index=line_index, evidence_text=line.text,
            ))
            if transition_type == "explicit":
                active = len(segments) - 1
        elif ambiguous:
            active = None
        elif active is not None:
            segments[active] = replace(segments[active], end_index=line_index)

    return AssignmentSuggestionResult(
        suggested_assignments=assignments_from_segments(len(transcript), segments),
        segments=segments,
        strategy="heuristic_moderator_keyword",
        uncertain_count=sum(segment.uncertain for segment in segments),
    )
