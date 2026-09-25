"""One model-only workflow for supplied and transcript-derived agendas."""
from dataclasses import dataclass, field
import math
import os
from assignment_suggestions import AssignmentSegment, assignments_from_segments
from llm_config import configured, get_llm_config
from processing_mode import policy, FAST_NOTICE

_mode = os.environ.get('AGENDA_DETECTION_USE_LLM', 'true').strip().lower()
if _mode not in {'true', 'false'}:
    raise ValueError('AGENDA_DETECTION_USE_LLM must be true or false')
AGENDA_DETECTION_USE_LLM = _mode == 'true'
AGENDA_DETECTION_TIMEOUT_SECONDS = float(os.environ.get('AGENDA_DETECTION_TIMEOUT_SECONDS', '8'))
if not math.isfinite(AGENDA_DETECTION_TIMEOUT_SECONDS) or AGENDA_DETECTION_TIMEOUT_SECONDS <= 0:
    raise ValueError('AGENDA_DETECTION_TIMEOUT_SECONDS must be finite and positive')
# Compatibility constants only. They no longer truncate model context.
AGENDA_DETECTION_CHUNK_LINES = int(os.environ.get('AGENDA_DETECTION_CHUNK_LINES', '80'))
AGENDA_DETECTION_CHUNK_OVERLAP_LINES = int(os.environ.get('AGENDA_DETECTION_CHUNK_OVERLAP_LINES', '12'))
AGENDA_DETECTION_CONTEXT_WINDOW_BEFORE = int(os.environ.get('AGENDA_DETECTION_CONTEXT_WINDOW_BEFORE', '4'))
AGENDA_DETECTION_CONTEXT_WINDOW_AFTER = int(os.environ.get('AGENDA_DETECTION_CONTEXT_WINDOW_AFTER', '8'))
DEFAULT_AGENDA_DETECTION_PROMPT = 'Rekonstruiere Agenda und tatsächlichen Sitzungsverlauf quellengetreu. Erfinde keine Originalnummern.'


@dataclass
class AgendaLLMUsage:
    enabled: bool
    source: str
    timeout_seconds: float = AGENDA_DETECTION_TIMEOUT_SECONDS
    status: str = 'skipped'
    attempted_calls: int = 0
    failed_calls: int = 0
    failure_reasons: list = field(default_factory=list)
    validation_reasons: list = field(default_factory=list)
    processed_lines: list = field(default_factory=list)
    gaps: list = field(default_factory=list)
    chunks: list = field(default_factory=list)
    provenance: dict = field(default_factory=dict)
    line_results: list = field(default_factory=list)
    agenda_states: list = field(default_factory=list)
    reconstructions: list = field(default_factory=list)
    processing_complete: bool = False
    processing_mode: str = 'slow'
    review_status: str = 'pending'
    review_complete: bool = False
    review_required: bool = True

    @property
    def warnings(self):
        warnings = []
        technical = sum(r['status'] == 'not_processed' for r in self.line_results)
        semantic = sum(r['status'] == 'unassigned' for r in self.line_results)
        joint = sum(len(r['top_ids']) > 1 for r in self.line_results)
        if joint:
            warnings.append(f'{joint} Zeilen gemeinsam mehreren TOPs zugeordnet. Die Zusammenfassungen berücksichtigen die gemeinsamen Quellen; Zuordnung prüfen.')
        if technical:
            warnings.append(f'TOP-Zuordnung technisch unvollständig: {technical} Zeilen nicht verarbeitet.')
        if semantic:
            warnings.append(f'TOP-Zuordnung: {semantic} fachlich begründet unzugeordnete Zeilen; Gründe prüfen.')
        if self.processing_mode == 'fast' and self.review_status == 'skipped':
            warnings.append(FAST_NOTICE)
        elif not self.review_complete:
            warnings.append('Unabhängige Modellprüfung nicht vollständig abgeschlossen; keine nachgewiesene fachliche Qualität.')
        if any(r['review_status'] == 'unresolved' for r in self.line_results):
            warnings.append('Zuordnungsprüfung: fachliche Grenzen oder Abweichungen bleiben ungeklärt.')
        if self.failed_calls:
            warnings.append(f'{self.failed_calls} technische Teilaufrufe fehlgeschlagen; erfolgreiche Schritte bleiben erhalten.')
        for issue in self.provenance.get('order_review_ranges', []):
            warnings.append(f"Zeilen {issue['start_index']+1}–{issue['end_index']+1}: {issue['reason']}")
        return warnings


@dataclass(frozen=True)
class AgendaDetectionResult:
    tops: list[str]
    assignments: list[int | None]
    segments: list[AssignmentSegment]
    uncertain_count: int
    strategy: str
    llm: AgendaLLMUsage | None = None


def _should_use_llm(use_llm=None):
    if use_llm is not None and type(use_llm) is not bool:
        raise ValueError('use_llm must be a boolean or None')
    return AGENDA_DETECTION_USE_LLM if use_llm is None else use_llm


@configured
def segment_known_agenda(transcript, tops, model=None, system_prompt=None, *, use_llm=None,
                         progress_callback=None, cache_namespace='', top_ids=None, processing_mode=None, enforce_top_order=False):
    from agenda_llm import classify
    usage = AgendaLLMUsage(_should_use_llm(use_llm), 'server_default' if use_llm is None else 'request',
                           timeout_seconds=get_llm_config(model).timeout_seconds, processing_mode=policy().mode)
    if any(not isinstance(t, str) or not t.strip() for t in tops):
        raise ValueError('invalid_agenda_title')
    titles, segments = classify(transcript, list(tops), usage, model, system_prompt, progress_callback,
                                cache_namespace=cache_namespace, top_ids=top_ids, enforce_top_order=enforce_top_order)
    return AgendaDetectionResult(titles, assignments_from_segments(len(transcript), segments), segments,
                                 sum(s.uncertain for s in segments), 'model_agenda_v1', usage)


@configured
def detect_agenda_from_transcript(transcript, model=None, system_prompt=None, *, use_llm=None,
                                   progress_callback=None, cache_namespace='', processing_mode=None):
    return segment_known_agenda(transcript, [], model, system_prompt, use_llm=use_llm,
                                 progress_callback=progress_callback, cache_namespace=cache_namespace)
