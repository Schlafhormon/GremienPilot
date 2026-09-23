"""
FastAPI Backend for Meeting Minutes Generator
"""

import os
import re
import json
import uuid
import time
import logging
import mimetypes
import asyncio
import threading
import unicodedata
import hashlib
from collections import OrderedDict
from dataclasses import asdict
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
    Header,
    Query,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, StrictBool, model_serializer

from transcribe import (
    transcribe_audio,
    load_models,
    transcription_model_session,
    TranscriptionModels,
    _cleanup_memory,
)
from summarize import (
    LLMCallError,
    StructuredOutputError,
    build_summary_review,
    llm_diagnostics,
    meeting_context_from_transcript,
    summarize_segment,
)
from extract_tops import extract_agenda_data_from_pdf, PdfReviewRequired
from assignment_suggestions import TranscriptUtterance, suggest_assignments
from agenda_detection import detect_agenda_from_transcript, segment_known_agenda
from export_protocol import (
    ProtocolAppendix,
    ProtocolMetadata,
    TranscriptLine as ExportTranscriptLine,
    build_protocol_document,
    render_protocol,
)
from persistence import (
    SessionConflictError,
    anonymize_speaker_observations_for_profile,
    archive_speaker_profile,
    confirm_speaker_observation,
    count_all_speaker_embeddings,
    count_job_speaker_embeddings,
    count_speaker_embeddings,
    create_speaker_profile,
    delete_speaker_embeddings,
    delete_speaker_embeddings_for_source,
    init_db,
    load_job,
    load_job_speaker_embedding,
    load_job_speaker_embeddings,
    load_jobs,
    list_sessions,
    load_latest_pipeline_job_for_session,
    load_latest_summary_job_for_session,
    load_session,
    load_summary_job,
    load_speaker_embeddings,
    load_speaker_observation,
    load_speaker_observations,
    load_speaker_profile,
    load_speaker_profiles,
    prune_speaker_embeddings,
    reject_speaker_observation,
    load_pipeline_job,
    load_pipeline_jobs,
    save_pipeline_job,
    save_summary_job,
    save_job,
    save_job_speaker_embedding,
    save_session,
    save_speaker_embedding,
    save_speaker_observation,
    update_speaker_profile,
)
from speaker_recognition import (
    LocalSpeakerEmbedding,
    build_profile_references,
    diagnose_speaker_matches,
    extract_local_speaker_embeddings,
    match_speaker_embeddings,
    speaker_embedding_config_from_env,
)
from transcript_splitter import split_transcript_for_agenda_detection

# Configure logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


import durable_jobs as durable
from llm_transport import request_control, work_slot, LLMCancelledError


class CancellationRequested(LLMCancelledError):
    """Raised inside a transcription worker when a job has been cancelled."""


@asynccontextmanager
async def lifespan(app: FastAPI):
    manager = durable.Manager(run_durable_job, mirror_durable_job)
    manager.lock = durable.ProcessLock().__enter__()
    app.state.durable_manager = manager
    app.state.models = None
    app.state.models_loaded = False
    try:
        init_db()
        jobs.clear()
        jobs.update(load_jobs())
        try:
            app.state.models = load_models()
            app.state.models_loaded = True
        except Exception:
            logger.error("Could not prepare transcription models", exc_info=True)
        recover_legacy_jobs()
        await manager.start()
        yield
    finally:
        if manager.started:
            await manager.stop()
        else:
            manager.lock.__exit__()
        if app.state.models is not None:
            _cleanup_memory(app.state.models.device)
        app.state.models = None
        app.state.models_loaded = False


app = FastAPI(
    title="GremienPilot API",
    description="API für die automatische Erstellung von Sitzungsprotokollen",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS configuration - allow configurable origins via environment
CORS_ORIGINS = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:5173,http://localhost:5174,http://localhost:5175,http://localhost:3000",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Job cleanup configuration
JOB_MAX_AGE_SECONDS = int(os.environ.get("JOB_MAX_AGE_SECONDS", "7200"))  # 2 hours
JOB_MAX_COUNT = int(os.environ.get("JOB_MAX_COUNT", "100"))
DELETE_UPLOADS_ON_JOB_CLEANUP = (
    os.environ.get("DELETE_UPLOADS_ON_JOB_CLEANUP", "false").lower() == "true"
)
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(500 * 1024 * 1024)))
UPLOAD_CHUNK_SIZE = int(os.environ.get("UPLOAD_CHUNK_SIZE", str(1024 * 1024)))
TRANSCRIPTION_CONCURRENCY = int(os.environ.get("TRANSCRIPTION_CONCURRENCY", "1"))
PIPELINE_CONCURRENCY = int(os.environ.get("PIPELINE_CONCURRENCY", "1"))
SUMMARY_CONCURRENCY = int(os.environ.get("SUMMARY_CONCURRENCY", "1"))
DELETE_UPLOADS_ON_CANCEL_OR_FAILURE = (
    os.environ.get("DELETE_UPLOADS_ON_CANCEL_OR_FAILURE", "true").lower() == "true"
)

JOB_STATUS_PENDING = "pending"
JOB_STATUS_PROCESSING = "processing"
JOB_STATUS_COMPLETED = "completed"
JOB_STATUS_FAILED = "failed"
JOB_STATUS_CANCELLED = "cancelled"
TERMINAL_JOB_STATUSES = {
    JOB_STATUS_COMPLETED,
    JOB_STATUS_FAILED,
    JOB_STATUS_CANCELLED,
}
PIPELINE_STATUS_PENDING = "pending"
PIPELINE_STATUS_PROCESSING = "processing"
PIPELINE_STATUS_COMPLETED = "completed"
PIPELINE_STATUS_FAILED = "failed"
PIPELINE_STATUS_CANCELLED = "cancelled"
TERMINAL_PIPELINE_STATUSES = {
    PIPELINE_STATUS_COMPLETED,
    PIPELINE_STATUS_FAILED,
    PIPELINE_STATUS_CANCELLED,
}
PIPELINE_STAGE_UPLOAD = "upload"
PIPELINE_STAGE_TRANSCRIBE = "transcribe"
PIPELINE_STAGE_SPEAKER_MATCH = "speaker_match"
PIPELINE_STAGE_AGENDA_DETECT = "agenda_detect"
PIPELINE_STAGE_SUMMARIZE = "summarize"
PIPELINE_STAGE_READY_FOR_REVIEW = "ready_for_review"

# Every local summary call shares one resource gate. The pipeline and manual
# regeneration workers therefore cannot saturate Ollama concurrently.
LLM_WORK_LOCK = threading.Lock()

# In-memory cache for jobs. SQLite remains the durable source for polling after
# restarts; the lock keeps the cache coherent across API and worker threads.
jobs: OrderedDict = OrderedDict()
JOB_LOCK = threading.RLock()

# Temporary upload directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_AUDIO_CONTENT_TYPES = {
    "audio/mpeg",
    "audio/wav",
    "audio/mp4",
    "audio/x-m4a",
    "audio/mp3",
}
ALLOWED_AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a")

CONTENT_TYPE_EXTENSIONS = {
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/wav": ".wav",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "application/pdf": ".pdf",
}


def is_allowed_audio_file(filename: str | None, content_type: str | None) -> bool:
    """Return whether an uploaded file is an accepted audio format."""
    normalized_filename = (filename or "").lower()
    return (content_type in ALLOWED_AUDIO_CONTENT_TYPES) or normalized_filename.endswith(
        ALLOWED_AUDIO_EXTENSIONS
    )


def is_allowed_pdf_file(filename: str | None, content_type: str | None) -> bool:
    """Return whether an uploaded file is a PDF."""
    return content_type == "application/pdf" or (filename or "").lower().endswith(".pdf")


def normalize_upload_filename(
    filename: str | None,
    *,
    default_stem: str,
    allowed_extensions: tuple[str, ...],
    content_type: str | None = None,
) -> str:
    """Return a safe display filename derived from an uploaded filename."""
    raw_filename = (filename or "").replace("\\", "/").split("/")[-1]
    normalized = unicodedata.normalize("NFKD", raw_filename)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_name = re.sub(r"[^A-Za-z0-9._-]+", "_", ascii_name).strip("._-")

    extension = Path(ascii_name).suffix.lower()
    if extension not in allowed_extensions:
        extension = CONTENT_TYPE_EXTENSIONS.get(content_type or "", "")
        if extension not in allowed_extensions:
            extension = ""

    stem = Path(ascii_name).stem if ascii_name else ""
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_-") or default_stem
    stem = stem[:80]
    return f"{stem}{extension}"


def upload_path_for(job_id: str, safe_filename: str) -> Path:
    """Build an upload path from trusted components only."""
    suffix = Path(safe_filename).suffix.lower()
    return UPLOAD_DIR / f"{job_id}{suffix}"


async def save_upload_with_size_limit(
    upload: UploadFile,
    destination: Path,
    *,
    max_bytes: int | None = None,
) -> int:
    """Stream an upload to disk while enforcing the backend upload limit."""
    max_allowed = max_bytes if max_bytes is not None else MAX_UPLOAD_BYTES
    destination.parent.mkdir(parents=True, exist_ok=True)
    bytes_written = 0

    try:
        with open(destination, "wb") as output:
            while True:
                chunk = await upload.read(UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > max_allowed:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "Datei ist zu groß. Maximal erlaubt sind "
                            f"{max_allowed // 1024 // 1024} MB."
                        ),
                    )
                output.write(chunk)
    except Exception:
        remove_upload_file(str(destination))
        raise

    return bytes_written


def remove_upload_file(file_path: str | None) -> None:
    if not file_path:
        return
    try:
        path = Path(file_path)
        if path.exists() and path.is_file():
            path.unlink()
    except Exception as e:
        logger.warning("Failed to remove upload file for job cleanup: %s", e)


def persist_job_state(job_id: str) -> None:
    """Persist a cached job without interrupting the request flow on DB errors."""
    with JOB_LOCK:
        job = jobs.get(job_id)
        if not job:
            return
        job["updated_at"] = time.time()
        job_snapshot = dict(job)
    try:
        save_job(job_id, job_snapshot)
    except Exception as e:
        logger.warning(f"Failed to persist job {job_id}: {e}")


def update_job_state(job_id: str, **changes: Any) -> dict[str, Any] | None:
    durable.check()
    with JOB_LOCK:
        job = jobs.get(job_id)
        if job is None:
            return None
        job.update(changes)
        job["updated_at"] = time.time()
    persist_job_state(job_id)
    return get_job_from_cache_or_db(job_id)


def get_job_from_cache_or_db(job_id: str) -> dict[str, Any] | None:
    with JOB_LOCK:
        job = jobs.get(job_id)
        if job is not None:
            return job

    try:
        job = load_job(job_id)
    except Exception as e:
        logger.warning(f"Failed to load job {job_id} from persistence: {e}")
        return None

    if job is not None:
        with JOB_LOCK:
            jobs[job_id] = job
    return job


def is_job_cancelled(job_id: str) -> bool:
    durable.check()
    job = get_job_from_cache_or_db(job_id)
    if not job:
        return True
    return bool(job.get("cancellation_requested")) or job.get("status") == JOB_STATUS_CANCELLED


def cleanup_job_uploads(job_id: str, job_data: dict[str, Any]) -> None:
    file_paths = {
        path for path in (job_data.get("audio_path"), job_data.get("file_path")) if path
    }
    for file_path in file_paths:
        remove_upload_file(file_path)
    if file_paths:
        with JOB_LOCK:
            job = jobs.get(job_id)
            if job:
                job["file_path"] = None
                job["audio_path"] = None
        persist_job_state(job_id)


async def get_or_create_job_manager():
    return DurableSubmission("transcription")


def _pipeline_refs(job: dict[str, Any] | None) -> dict[str, Any]:
    return dict((job or {}).get("result_refs") or {})


def save_pipeline_state(pipeline_id: str, **changes: Any) -> dict[str, Any] | None:
    durable.check()
    job = load_pipeline_job(pipeline_id)
    if job is None:
        return None
    result_refs = _pipeline_refs(job)
    if "result_refs" in changes:
        result_refs.update(changes.pop("result_refs") or {})
    job.update(changes)
    job["result_refs"] = result_refs
    job["updated_at"] = time.time()
    return save_pipeline_job(pipeline_id, job)


def append_pipeline_warning(pipeline_id: str, message: str) -> None:
    job = load_pipeline_job(pipeline_id)
    if job is None:
        return
    refs = _pipeline_refs(job)
    warnings = list(refs.get("warnings") or [])
    warnings.append(message)
    save_pipeline_state(pipeline_id, result_refs={"warnings": warnings})


def safe_exception_label(exc: Exception) -> str:
    """Return a non-content-bearing exception label for logs and review warnings."""
    return exc.__class__.__name__


def is_pipeline_cancelled(pipeline_id: str) -> bool:
    job = load_pipeline_job(pipeline_id)
    if job is None:
        return True
    refs = _pipeline_refs(job)
    return bool(refs.get("cancel_requested")) or job.get("status") == PIPELINE_STATUS_CANCELLED


def ensure_pipeline_not_cancelled(pipeline_id: str) -> None:
    durable.check()
    if is_pipeline_cancelled(pipeline_id):
        raise CancellationRequested()


async def get_or_create_pipeline_manager():
    return DurableSubmission("pipeline")


async def get_or_create_summary_job_manager():
    return DurableSubmission("summary")


def request_job_cancellation(job_id: str) -> dict[str, Any] | None:
    if durable.load(job_id):
        durable.cancel(job_id)
    job = get_job_from_cache_or_db(job_id)
    if job is None:
        return None

    status = job.get("status")
    if status == JOB_STATUS_CANCELLED:
        return job
    if status in {JOB_STATUS_COMPLETED, JOB_STATUS_FAILED}:
        raise HTTPException(
            status_code=409,
            detail="Job kann in diesem Status nicht abgebrochen werden",
        )

    message = (
        "Transkription wird abgebrochen..."
        if status == JOB_STATUS_PROCESSING
        else "Transkription abgebrochen"
    )
    updated = update_job_state(
        job_id,
        status=JOB_STATUS_CANCELLED,
        cancellation_requested=True,
        message=message,
        error=None,
    )

    if status == JOB_STATUS_PENDING and updated:
        cleanup_job_uploads(job_id, updated)
        updated = get_job_from_cache_or_db(job_id)

    return updated


def get_audio_path_for_job(job: dict[str, Any]) -> str | None:
    """Return the persisted upload path used for playback, if available."""
    return job.get("audio_path") or job.get("file_path")


def audio_url_for_job(job_id: str, job: dict[str, Any]) -> str | None:
    audio_path = get_audio_path_for_job(job)
    if audio_path and os.path.exists(audio_path):
        return f"/api/audio/{job_id}"
    return None


def cleanup_old_jobs() -> int:
    """
    Remove old or excess jobs from memory.
    Upload files are retained by default so persisted sessions can be restored.
    Returns number of jobs removed.
    """
    now = time.time()
    removed = 0

    def cleanup_job_audio(job_id: str, job_data: dict) -> None:
        """Clean up audio file associated with a legacy job only."""
        if durable.load(job_id) or load_latest_pipeline_job_for_session(job_data.get("session_id") or ""):
            return
        if not DELETE_UPLOADS_ON_JOB_CLEANUP:
            return
        file_paths = {
            path
            for path in (job_data.get("audio_path"), job_data.get("file_path"))
            if path
        }
        for audio_path in file_paths:
            if not os.path.exists(audio_path):
                continue
            try:
                os.remove(audio_path)
                logger.info(f"Cleaned up audio file for job {job_id}")
            except Exception as e:
                logger.warning(f"Failed to clean up audio file for job {job_id}: {e}")

    with JOB_LOCK:
        # Remove jobs older than MAX_AGE
        jobs_to_remove = []
        for job_id, job_data in jobs.items():
            if now - job_data.get("created_at", now) > JOB_MAX_AGE_SECONDS:
                jobs_to_remove.append(job_id)

        for job_id in jobs_to_remove:
            cleanup_job_audio(job_id, jobs[job_id])
            del jobs[job_id]
            removed += 1

        # Remove oldest jobs if count exceeds MAX_COUNT
        while len(jobs) > JOB_MAX_COUNT:
            oldest_job_id = next(iter(jobs))
            cleanup_job_audio(oldest_job_id, jobs[oldest_job_id])
            del jobs[oldest_job_id]
            removed += 1

    if removed > 0:
        logger.info(f"Cleaned up {removed} old jobs, {len(jobs)} remaining")

    return removed


# ----- Pydantic Models -----


class TranscriptLine(BaseModel):
    line_id: Optional[str] = None
    speaker: str
    text: str
    # Seconds; sentence chunks may use character-based estimates, not measured
    # speech boundaries. Decimal places do not indicate timestamp accuracy.
    start: float
    end: float
    timing: Optional[Dict[str, Any]] = None

    @model_serializer(mode='wrap')
    def serialize_line(self, handler):
        value = handler(self)
        if value.get('timing') is None:
            value.pop('timing', None)
        return value


class AudioMetadata(BaseModel):
    filename: Optional[str] = None
    content_type: Optional[str] = None
    size_bytes: Optional[int] = None


class SpeakerSuggestionResponse(BaseModel):
    observation_id: int
    local_speaker_id: str
    profile_id: str
    profile_display_name: str
    confidence: float
    status: str


class TranscriptionJob(BaseModel):
    job_id: str
    status: str  # "pending", "processing", "completed", "failed"
    progress: int
    message: str
    transcript: Optional[List[TranscriptLine]] = None
    speaker_suggestions: Optional[List[SpeakerSuggestionResponse]] = None
    audio_url: Optional[str] = None  # URL to stream audio for playback
    audio_metadata: Optional[AudioMetadata] = None
    error: Optional[str] = None


class PipelineStartResponse(BaseModel):
    pipeline_id: str
    session_id: str
    transcription_job_id: str
    status: str
    stage: str
    progress: int
    warnings: List[str] = Field(default_factory=list)


class PipelineStatusResponse(BaseModel):
    execution: Optional[Dict[str, Any]] = None
    pipeline_id: str
    session_id: Optional[str] = None
    transcription_job_id: Optional[str] = None
    status: str
    stage: str
    progress: int
    warnings: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    created_at: Optional[float] = None
    updated_at: Optional[float] = None


class SummaryJobResponse(BaseModel):
    execution: Optional[Dict[str, Any]] = None
    summary_job_id: str
    session_id: str
    status: str
    progress: int
    current_top: int = 0
    total_tops: int = 0
    top_ids: List[str] = Field(default_factory=list)
    completed_tops: int = 0
    processed_tops: int = 0
    current_top_id: Optional[str] = None
    outcomes: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    created_at: Optional[float] = None
    updated_at: Optional[float] = None


class SessionSaveRequest(BaseModel):
    agenda_proposals: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    revision: Optional[int] = None
    job_id: Optional[str] = None
    current_step: Optional[int] = None
    tops: List[str] = Field(default_factory=list)
    top_ids: List[str] = Field(default_factory=list)
    transcript: Optional[List[TranscriptLine]] = None
    assignments: List[Optional[int]] = Field(default_factory=list)
    speaker_names: Dict[str, str] = Field(default_factory=dict)
    summaries: Dict[int, str] = Field(default_factory=dict)
    summary_reviews: Dict[int, Any] = Field(default_factory=dict)
    summary_states: Dict[int, Any] = Field(default_factory=dict)
    export_metadata: Dict[str, Any] = Field(default_factory=dict)
    skipped_assignment: bool = False


class SessionResponse(BaseModel):
    agenda_proposals: Optional[Dict[str, Any]] = None
    session_id: str
    revision: int = 1
    created_at: Optional[float] = None
    updated_at: Optional[float] = None
    job_id: Optional[str] = None
    current_step: Optional[int] = None
    tops: List[str] = Field(default_factory=list)
    top_ids: List[str] = Field(default_factory=list)
    assignments: List[Optional[int]] = Field(default_factory=list)
    speaker_names: Dict[str, str] = Field(default_factory=dict)
    summaries: Dict[int, str] = Field(default_factory=dict)
    summary_reviews: Dict[int, Any] = Field(default_factory=dict)
    summary_states: Dict[int, Any] = Field(default_factory=dict)
    export_metadata: Dict[str, Any] = Field(default_factory=dict)
    skipped_assignment: bool = False
    transcript: Optional[List[TranscriptLine]] = None
    audio_url: Optional[str] = None
    audio_metadata: Optional[AudioMetadata] = None
    job: Optional[TranscriptionJob] = None
    latest_pipeline: Optional[PipelineStatusResponse] = None
    pdf_extraction: Optional[Dict[str, Any]] = None
    latest_summary_job: Optional[SummaryJobResponse] = None


class SessionListItem(BaseModel):
    session_id: str
    title: str
    committee: str = ""
    meeting_date: str = ""
    status: str
    current_step: Optional[int] = None
    revision: int = 1
    created_at: float
    updated_at: float
    top_count: int = 0
    transcript_line_count: int = 0
    summary_count: int = 0
    audio_available: bool = False
    job_id: Optional[str] = None
    job_status: Optional[str] = None
    pipeline_job_id: Optional[str] = None
    pipeline_status: Optional[str] = None
    pipeline_stage: Optional[str] = None
    pipeline_error: Optional[str] = None
    pipeline_progress: Optional[int] = None


class SessionListResponse(BaseModel):
    items: List[SessionListItem] = Field(default_factory=list)
    total: int
    limit: int
    offset: int


class SpeakerProfileCreateRequest(BaseModel):
    display_name: str = Field(..., min_length=1)
    scope: Optional[str] = None


class SpeakerProfileUpdateRequest(BaseModel):
    display_name: Optional[str] = Field(None, min_length=1)
    scope: Optional[str] = None


class SpeakerProfileResponse(BaseModel):
    profile_id: str
    display_name: str
    scope: Optional[str] = None
    created_at: float
    updated_at: float
    archived_at: Optional[float] = None
    archived: bool = False
    embedding_count: int = 0


class SpeakerObservationResponse(BaseModel):
    observation_id: int
    job_id: str
    session_id: str
    local_speaker_id: str
    local_display_name: str
    profile_id: Optional[str] = None
    profile_display_name: Optional[str] = None
    profile: Optional[SpeakerProfileResponse] = None
    confidence: Optional[float] = None
    status: str
    display_name: str
    embedding_warning: Optional[str] = None
    created_at: float
    updated_at: float


class SpeakerEmbeddingProfileDiagnosticResponse(BaseModel):
    profile_id: str
    display_name: str
    embedding_count: int


class LocalSpeakerEmbeddingDiagnosticResponse(BaseModel):
    local_speaker_id: str
    embedding_available: bool
    model_name: Optional[str] = None
    quality: Optional[float] = None
    total_seconds: Optional[float] = None
    selected_segment_count: int = 0
    candidate_segment_count: int = 0
    excluded_short_count: int = 0
    excluded_overlap_count: int = 0


class SpeakerEmbeddingDiagnosticsResponse(BaseModel):
    enabled: bool
    loaded: bool
    on_demand: bool = False
    model_name: Optional[str] = None
    attempted_model_names: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    torch_force_no_weights_only_load: Optional[str] = None
    job_speaker_embedding_count: int = 0
    profile_embedding_count: int = 0
    profiles: List[SpeakerEmbeddingProfileDiagnosticResponse] = Field(default_factory=list)
    session_id: Optional[str] = None
    job_id: Optional[str] = None
    local_speakers: List[LocalSpeakerEmbeddingDiagnosticResponse] = Field(default_factory=list)


class SpeakerEmbeddingBackfillResponse(BaseModel):
    scanned_observation_count: int
    processed_job_count: int
    saved_embedding_count: int
    skipped_count: int
    errors: List[str] = Field(default_factory=list)


class SpeakerMatchDiagnosticResponse(BaseModel):
    local_speaker_id: str
    reason_code: str
    reason: str
    best_profile_id: Optional[str] = None
    best_profile_display_name: Optional[str] = None
    best_score: Optional[float] = None
    suggest_threshold: Optional[float] = None
    local_audio_seconds: Optional[float] = None
    local_embedding_available: bool = False
    profile_embedding_count: int = 0


class SpeakerObservationConfirmRequest(BaseModel):
    profile_id: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)


class SpeakerObservationManualRequest(BaseModel):
    local_speaker_id: str = Field(..., min_length=1)
    profile_id: Optional[str] = None
    display_name: Optional[str] = Field(None, min_length=1)
    scope: Optional[str] = None
    confidence: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    observation_id: Optional[int] = None


class SummarizeRequest(BaseModel):
    top_title: str
    lines: List[TranscriptLine]
    model: Optional[str] = None  # LLM model to use (e.g., "qwen3:8b")
    system_prompt: Optional[str] = None  # Custom system prompt


class StructuredSummaryResponse(BaseModel):
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    review_questions: List[Dict[str, Any]] = Field(default_factory=list)
    verification: Dict[str, Any] = Field(default_factory=dict)
    discussion: List[str] = Field(default_factory=list)
    decisions: List[str] = Field(default_factory=list)
    votes: List[str] = Field(default_factory=list)
    action_items: List[str] = Field(default_factory=list)
    open_points: List[str] = Field(default_factory=list)
    uncertainties: List[str] = Field(default_factory=list)


class SummarySourceLinkResponse(BaseModel):
    source_ids: List[str] = Field(default_factory=list)
    scope: Optional[str] = None
    section: str
    item_index: int
    item_text: str
    line_indices: List[int] = Field(default_factory=list)
    start: Optional[float] = None
    end: Optional[float] = None
    excerpt: str = ""
    confidence: float = 0.0
    missing_source: bool = False


class SummaryReviewWarningResponse(BaseModel):
    kind: str
    message: str
    severity: str = "warning"
    keyword: Optional[str] = None
    section: Optional[str] = None
    item_index: Optional[int] = None
    line_indices: List[int] = Field(default_factory=list)
    start: Optional[float] = None
    end: Optional[float] = None
    excerpt: str = ""


class SummarizeResponse(BaseModel):
    summary: str
    duration_seconds: float
    structured: Optional[StructuredSummaryResponse] = None
    source_links: List[SummarySourceLinkResponse] = Field(default_factory=list)
    review_warnings: List[SummaryReviewWarningResponse] = Field(default_factory=list)
    fallback_used: bool = False
    chunks_processed: int = 1
    llm_usage: Dict[str, Any] = Field(default_factory=dict)


class SummaryJobCreateRequest(BaseModel):
    revision: Optional[int] = None
    top_ids: List[str] = Field(default_factory=list, min_length=1)
    model: Optional[str] = None
    system_prompt: Optional[str] = None


class SummaryAcceptRequest(BaseModel):
    revision: Optional[int] = None


class LLMDiagnosticsResponse(BaseModel):
    ok: bool
    base_url: str
    model: str
    base_url_source: str
    service_reachable: bool
    model_available: bool
    available_models: List[str] = Field(default_factory=list)
    configuration: Dict[str, Any] = Field(default_factory=dict)
    message: str


class ExtractTOPsResponse(BaseModel):
    tops: List[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    processing_complete: bool = False
    review_required: bool = True
    items: List[Dict[str, Any]] = Field(default_factory=list)
    metadata_sources: Dict[str, Any] = Field(default_factory=dict)
    document: Dict[str, Any] = Field(default_factory=dict)
    pages: List[Dict[str, Any]] = Field(default_factory=list)
    audits: List[Dict[str, Any]] = Field(default_factory=list)
    contract_version: Optional[str] = None
    review_questions: List[Dict[str, Any]] = Field(default_factory=list)
    stop_reason: Optional[str] = None


class AssignmentSuggestionsRequest(BaseModel):
    transcript: List[TranscriptLine]
    tops: List[str]


class AgendaDetectionRequest(BaseModel):
    fresh: StrictBool = False
    cache_namespace: str = Field(default='', max_length=128)
    top_ids: List[str] = Field(default_factory=list)
    preserve_transcript_structure: StrictBool = False
    use_llm: Optional[StrictBool] = None
    transcript: List[TranscriptLine]
    tops: List[str] = Field(default_factory=list)
    model: Optional[str] = None
    system_prompt: Optional[str] = None


class AssignmentSuggestionSegmentResponse(BaseModel):
    top_index: int
    top_title: str
    start_index: int
    end_index: int
    confidence: float
    uncertain: bool
    transition_type: str
    reason: str
    evidence_index: Optional[int] = None
    evidence_text: Optional[str] = None


class AssignmentSuggestionsResponse(BaseModel):
    llm: Optional[Dict[str, Any]] = None
    warnings: List[str] = Field(default_factory=list)
    suggested_assignments: List[Optional[int]]
    segments: List[AssignmentSuggestionSegmentResponse]
    strategy: str
    uncertain_count: int


class AgendaLLMUsageResponse(BaseModel):
    line_results: List[Dict[str, Any]] = Field(default_factory=list)
    agenda_states: List[Dict[str, Any]] = Field(default_factory=list)
    reconstructions: List[Dict[str, Any]] = Field(default_factory=list)
    processing_complete: bool = False
    review_complete: bool = False
    review_required: bool = True
    provenance: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool
    source: str
    timeout_seconds: float
    status: str
    attempted_calls: int
    failed_calls: int
    failure_reasons: List[str] = Field(default_factory=list)
    validation_reasons: List[str] = Field(default_factory=list)
    processed_lines: List[int] = Field(default_factory=list)
    gaps: List[Dict[str, Any]] = Field(default_factory=list)
    chunks: List[Dict[str, Any]] = Field(default_factory=list)


class AgendaDetectionResponse(BaseModel):
    llm: Optional[AgendaLLMUsageResponse] = None
    warnings: List[str] = Field(default_factory=list)
    tops: List[str]
    transcript: List[TranscriptLine] = Field(default_factory=list)
    assignments: List[Optional[int]]
    segments: List[AssignmentSuggestionSegmentResponse]
    uncertain_count: int
    strategy: str


class PipelineResultResponse(BaseModel):
    pipeline: PipelineStatusResponse
    session: SessionResponse
    job: Optional[TranscriptionJob] = None
    speaker_observations: List[SpeakerObservationResponse] = Field(default_factory=list)
    summary_reviews: Dict[int, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    agenda_detection: Optional[AgendaDetectionResponse] = None


class ExportMetadataRequest(BaseModel):
    committee: str = ""
    date: str = ""
    location: str = ""
    title: str = ""
    participants: List[str] = Field(default_factory=list)


class ExportAppendixRequest(BaseModel):
    include_speaker_list: bool = True
    include_transcript: Optional[bool] = None
    include_transcript_excerpt: bool = False
    group_transcript_by_top: bool = False
    include_generation_note: bool = True
    transcript_excerpt_limit: int = Field(default=20, ge=1, le=200)


class ProtocolExportRequest(BaseModel):
    session_id: Optional[str] = None
    format: str = "docx"
    metadata: ExportMetadataRequest = Field(default_factory=ExportMetadataRequest)
    appendix: ExportAppendixRequest = Field(default_factory=ExportAppendixRequest)
    tops: List[str] = Field(default_factory=list)
    transcript: List[TranscriptLine] = Field(default_factory=list)
    assignments: List[Optional[int]] = Field(default_factory=list)
    speaker_names: Dict[str, str] = Field(default_factory=dict)
    summaries: Dict[int, str] = Field(default_factory=dict)
    summary_reviews: Dict[int, Any] = Field(default_factory=dict)


def build_speaker_suggestion_responses(
    job_id: str,
    session_id: str | None,
) -> list[SpeakerSuggestionResponse] | None:
    if not session_id:
        return None

    observations = load_speaker_observations(
        job_id=job_id,
        session_id=session_id,
        status="suggested",
    )
    suggestions: list[SpeakerSuggestionResponse] = []
    for observation in observations:
        profile_id = observation.get("profile_id")
        confidence = observation.get("confidence")
        if not profile_id or confidence is None:
            continue
        profile = load_speaker_profile(profile_id, include_archived=True)
        if profile is None:
            continue
        suggestions.append(
            SpeakerSuggestionResponse(
                observation_id=observation["observation_id"],
                local_speaker_id=observation["local_speaker_id"],
                profile_id=profile_id,
                profile_display_name=profile["display_name"],
                confidence=confidence,
                status=observation["status"],
            )
        )

    return suggestions or None


def build_transcription_job_response(
    job_id: str, job: dict[str, Any]
) -> TranscriptionJob:
    audio_metadata = None
    if any(
        job.get(key)
        for key in ("audio_filename", "audio_content_type", "audio_size_bytes")
    ):
        audio_metadata = AudioMetadata(
            filename=job.get("audio_filename"),
            content_type=job.get("audio_content_type"),
            size_bytes=job.get("audio_size_bytes"),
        )

    return TranscriptionJob(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        message=job["message"],
        transcript=(
            [TranscriptLine(**line) for line in job["transcript"]]
            if job.get("transcript")
            else None
        ),
        speaker_suggestions=build_speaker_suggestion_responses(
            job_id,
            job.get("session_id"),
        ),
        audio_url=audio_url_for_job(job_id, job),
        audio_metadata=audio_metadata,
        error=job.get("error"),
    )


def build_session_response(session: dict[str, Any]) -> SessionResponse:
    job_id = session.get("job_id")
    job = get_job_from_cache_or_db(job_id) if job_id else None
    job_response = build_transcription_job_response(job_id, job) if job else None
    transcript = session.get("transcript")
    if transcript is None and job_response:
        transcript = job_response.transcript
    latest_pipeline = load_latest_pipeline_job_for_session(session["session_id"])
    latest_summary_job = load_latest_summary_job_for_session(session["session_id"])
    summary_states = dict(session.get("summary_states") or {})
    if session.get("summaries") or {}:
        fallback_states = build_generated_summary_states(
            session_id=session["session_id"],
            transcript=[line_to_dict(line) for line in (transcript or [])],
            tops=list(session.get("tops") or []),
            top_ids=list(session.get("top_ids") or []),
            assignments=list(session.get("assignments") or []),
            summaries=dict(session.get("summaries") or {}),
            summary_reviews=dict(session.get("summary_reviews") or {}),
            origin="legacy",
        )
        for top_index, fallback_state in fallback_states.items():
            summary_states.setdefault(top_index, fallback_state)

    return SessionResponse(
        session_id=session["session_id"],
        revision=int(session.get("revision") or 1),
        created_at=session.get("created_at"),
        updated_at=session.get("updated_at"),
        job_id=job_id,
        current_step=session.get("current_step"),
        tops=session.get("tops") or [],
        top_ids=session.get("top_ids") or [],
        assignments=session.get("assignments") or [],
        agenda_proposals=session_agenda_proposals(session, latest_pipeline),
        speaker_names=session.get("speaker_names") or {},
        summaries=session.get("summaries") or {},
        summary_reviews=session.get("summary_reviews") or {},
        summary_states=summary_states,
        export_metadata=session.get("export_metadata") or {},
        skipped_assignment=bool(session.get("skipped_assignment")),
        transcript=transcript,
        audio_url=job_response.audio_url if job_response else None,
        audio_metadata=job_response.audio_metadata if job_response else None,
        job=job_response,
        pdf_extraction=_pipeline_refs(latest_pipeline).get("pdf_extraction") if latest_pipeline else None,
        latest_pipeline=(
            build_pipeline_status_response(latest_pipeline)
            if latest_pipeline is not None
            else None
        ),
        latest_summary_job=(
            build_summary_job_response(latest_summary_job)
            if latest_summary_job is not None
            else None
        ),
    )


def build_summary_job_response(job: dict[str, Any]) -> SummaryJobResponse:
    refs = dict(job.get("refs") or {})
    return SummaryJobResponse(
        execution=durable.public(work) if (work := durable.load(job["summary_job_id"])) else None,
        summary_job_id=job["summary_job_id"],
        session_id=job["session_id"],
        status=job["status"],
        progress=int(job.get("progress") or 0),
        current_top=int(job.get("current_top") or 0),
        total_tops=int(job.get("total_tops") or 0),
        top_ids=list(refs.get("top_ids") or []),
        completed_tops=sum(item.get("status") == "completed" for item in refs.get("outcomes", {}).values()),
        processed_tops=len(refs.get("outcomes", {})),
        current_top_id=refs.get("current_top_id"),
        outcomes=dict(refs.get("outcomes") or {}),
        error=job.get("error"),
        created_at=job.get("created_at"),
        updated_at=job.get("updated_at"),
    )


def build_speaker_profile_response(profile: dict[str, Any]) -> SpeakerProfileResponse:
    return SpeakerProfileResponse(
        profile_id=profile["profile_id"],
        display_name=profile["display_name"],
        scope=profile.get("scope"),
        created_at=profile["created_at"],
        updated_at=profile["updated_at"],
        archived_at=profile.get("archived_at"),
        archived=profile.get("archived_at") is not None,
        embedding_count=count_speaker_embeddings(profile["profile_id"]),
    )


def build_speaker_observation_response(
    observation: dict[str, Any],
    session: dict[str, Any],
    *,
    embedding_warning: str | None = None,
) -> SpeakerObservationResponse:
    profile = None
    if observation.get("profile_id"):
        profile = load_speaker_profile(
            observation["profile_id"],
            include_archived=True,
        )
    profile_response = build_speaker_profile_response(profile) if profile else None
    profile_display_name = profile.get("display_name") if profile else None
    local_display_name = (session.get("speaker_names") or {}).get(
        observation["local_speaker_id"],
        observation["local_speaker_id"],
    )
    display_name = profile_display_name or local_display_name

    return SpeakerObservationResponse(
        observation_id=observation["observation_id"],
        job_id=observation["job_id"],
        session_id=observation["session_id"],
        local_speaker_id=observation["local_speaker_id"],
        local_display_name=local_display_name,
        profile_id=observation.get("profile_id"),
        profile_display_name=profile_display_name,
        profile=profile_response,
        confidence=observation.get("confidence"),
        status=observation["status"],
        display_name=display_name,
        embedding_warning=embedding_warning,
        created_at=observation["created_at"],
        updated_at=observation["updated_at"],
    )


def get_required_session(session_id: str) -> dict[str, Any]:
    session = load_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session nicht gefunden")
    return session


def get_required_active_profile(profile_id: str) -> dict[str, Any]:
    profile = load_speaker_profile(profile_id, include_archived=True)
    if profile is None:
        raise HTTPException(status_code=404, detail="Profil nicht gefunden")
    if profile.get("archived_at") is not None:
        raise HTTPException(status_code=409, detail="Profil ist archiviert")
    return profile


def validate_display_name(display_name: str) -> str:
    cleaned = display_name.strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="Anzeigename darf nicht leer sein")
    return cleaned


def known_local_speaker_ids(session: dict[str, Any]) -> set[str]:
    speaker_ids = set((session.get("speaker_names") or {}).keys())
    for line in session.get("transcript") or []:
        if line.get("speaker"):
            speaker_ids.add(str(line["speaker"]))
    return speaker_ids


def ensure_local_speaker_exists(session: dict[str, Any], local_speaker_id: str) -> None:
    if local_speaker_id not in known_local_speaker_ids(session):
        raise HTTPException(status_code=404, detail="Lokaler Sprecher nicht gefunden")


def get_observation_for_session(
    session_id: str,
    observation_id: int,
) -> dict[str, Any]:
    observation = load_speaker_observation(observation_id)
    if observation is None or observation.get("session_id") != session_id:
        raise HTTPException(status_code=404, detail="Observation nicht gefunden")
    return observation


def ensure_no_accepted_local_mapping(
    session_id: str,
    local_speaker_id: str,
    *,
    exclude_observation_id: int | None = None,
) -> None:
    observations = load_speaker_observations(session_id=session_id)
    for observation in observations:
        if observation["observation_id"] == exclude_observation_id:
            continue
        if observation.get("local_speaker_id") != local_speaker_id:
            continue
        if observation.get("status") in {"confirmed", "manual"}:
            raise HTTPException(
                status_code=409,
                detail="Lokaler Sprecher ist bereits einem Profil zugeordnet",
            )


def apply_profile_display_name_to_session(
    session: dict[str, Any],
    local_speaker_id: str,
    profile: dict[str, Any],
) -> dict[str, Any]:
    updated_session = dict(session)
    speaker_names = dict(updated_session.get("speaker_names") or {})
    speaker_names[local_speaker_id] = profile["display_name"]
    updated_session["speaker_names"] = speaker_names
    updated_session = reconcile_session_summaries(session, updated_session)
    return save_session(
        session["session_id"],
        updated_session,
        bump_revision=False,
    )


def persist_job_speaker_embeddings(
    job_id: str,
    embeddings: list[LocalSpeakerEmbedding] | None,
) -> None:
    for embedding in embeddings or []:
        quality_metadata = dict(embedding.quality_metadata or {})
        if embedding.reference_embeddings:
            quality_metadata["reference_embeddings"] = embedding.reference_embeddings
        save_job_speaker_embedding(
            job_id=job_id,
            local_speaker_id=embedding.local_speaker_id,
            embedding=embedding.embedding,
            model_name=embedding.model_name,
            quality=embedding.quality,
            quality_metadata=quality_metadata,
        )


def load_local_speaker_embeddings_from_job(
    job_id: str,
    *,
    model_name: str | None = None,
) -> list[LocalSpeakerEmbedding]:
    embeddings = []
    for item in load_job_speaker_embeddings(job_id, model_name=model_name):
        embeddings.append(
            LocalSpeakerEmbedding(
                local_speaker_id=item["local_speaker_id"],
                embedding=item["embedding"],
                model_name=item["model_name"],
                quality=float(item.get("quality") or 0.0),
                reference_embeddings=(
                    item.get("quality_metadata", {}).get("reference_embeddings")
                    or []
                ),
                quality_metadata=item.get("quality_metadata") or {},
            )
        )
    return embeddings


def speaker_audio_seconds_from_transcript(
    transcript: list[dict[str, Any]] | None,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    for line in transcript or []:
        speaker = str(line.get("speaker") or "")
        if not speaker:
            continue
        duration = max(0.0, float(line.get("end", 0)) - float(line.get("start", 0)))
        totals[speaker] = totals.get(speaker, 0.0) + duration
    return totals


def create_speaker_suggestion_observations(
    *,
    job_id: str,
    session_id: str | None,
    local_embeddings: list[LocalSpeakerEmbedding] | None,
    speaker_memory_opt_in: bool = False,
) -> None:
    if not speaker_memory_opt_in or not session_id or not local_embeddings:
        return

    config = speaker_embedding_config_from_env()
    profiles = load_speaker_profiles()
    matches = []
    model_names = sorted({embedding.model_name for embedding in local_embeddings})
    for model_name in model_names:
        model_local_embeddings = [
            embedding
            for embedding in local_embeddings
            if embedding.model_name == model_name
        ]
        embeddings_by_profile = {
            profile["profile_id"]: load_speaker_embeddings(
                profile["profile_id"],
                model_name=model_name,
            )
            for profile in profiles
        }
        references = build_profile_references(profiles, embeddings_by_profile)
        matches.extend(
            match_speaker_embeddings(
                model_local_embeddings,
                references,
                auto_threshold=config.auto_threshold,
                suggest_threshold=config.suggest_threshold,
                top_k=config.match_top_k,
            )
        )
    if not matches:
        return

    existing_observations = load_speaker_observations(
        job_id=job_id,
        session_id=session_id,
    )
    existing_keys = {
        (
            observation.get("local_speaker_id"),
            observation.get("profile_id"),
            observation.get("status"),
        )
        for observation in existing_observations
    }
    accepted_local_speakers = {
        observation.get("local_speaker_id")
        for observation in load_speaker_observations(session_id=session_id)
        if observation.get("status") in {"confirmed", "manual"}
    }

    for match in matches:
        if match.local_speaker_id in accepted_local_speakers:
            continue
        key = (match.local_speaker_id, match.profile_id, "suggested")
        if key in existing_keys:
            continue
        save_speaker_observation(
            job_id=job_id,
            session_id=session_id,
            local_speaker_id=match.local_speaker_id,
            profile_id=match.profile_id,
            confidence=match.confidence,
            status="suggested",
        )


def add_job_embedding_to_profile(
    *,
    job_id: str,
    local_speaker_id: str,
    profile_id: str,
    observation_id: int | None = None,
    storage_reason: str | None = None,
) -> dict[str, Any]:
    if storage_reason not in {"opt_in", "confirm", "manual"}:
        raise HTTPException(
            status_code=403,
            detail=(
                "Globale Sprecher-Embeddings werden nur nach Opt-in oder "
                "expliziter Sprecheraktion gespeichert"
            ),
        )
    local_embedding = load_job_speaker_embedding(job_id, local_speaker_id)
    if local_embedding is None:
        return {
            "saved_count": 0,
            "skipped_reason": "no_job_embedding",
        }

    metadata = dict(local_embedding.get("quality_metadata") or {})
    reference_embeddings = metadata.pop("reference_embeddings", None) or []
    if not reference_embeddings:
        reference_embeddings = [local_embedding["embedding"]]
    selected_segments = metadata.get("selected_segments") or []
    metadata.update(
        {
            "source_job_id": job_id,
            "source_local_speaker_id": local_speaker_id,
            "source_observation_id": observation_id,
            "storage_reason": storage_reason,
        }
    )

    for existing in load_speaker_embeddings(
        profile_id,
        model_name=local_embedding["model_name"],
    ):
        existing_metadata = existing.get("metadata") or {}
        if (
            existing_metadata.get("source_job_id") == job_id
            and existing_metadata.get("source_local_speaker_id") == local_speaker_id
        ):
            return {
                "saved_count": 0,
                "skipped_reason": "already_stored",
            }

    saved_count = 0
    for reference_index, reference_embedding in enumerate(reference_embeddings):
        reference_metadata = dict(metadata)
        reference_metadata["source_reference_index"] = reference_index
        if reference_index < len(selected_segments):
            reference_metadata["source_segment"] = selected_segments[reference_index]
        save_speaker_embedding(
            profile_id,
            reference_embedding,
            model_name=local_embedding["model_name"],
            quality=local_embedding.get("quality"),
            metadata=reference_metadata,
        )
        saved_count += 1

    config = speaker_embedding_config_from_env()
    prune_speaker_embeddings(
        profile_id,
        model_name=local_embedding["model_name"],
        max_count=config.max_profile_embeddings_per_model,
    )
    return {
        "saved_count": saved_count,
        "skipped_reason": None,
    }


def speaker_embedding_storage_warning(result: dict[str, Any] | None) -> str | None:
    if not result:
        return None
    reason = result.get("skipped_reason")
    if reason == "no_job_embedding":
        return (
            "Profil wurde zugeordnet, aber für diese Sitzung ist kein "
            "Sprecher-Embedding verfügbar. Automatische Wiedererkennung wird "
            "erst nach erfolgreicher Embedding-Erzeugung funktionieren."
        )
    return None


class TranscriptDiarization:
    def __init__(self, transcript: list[dict[str, Any]]):
        self.transcript = transcript

    def itertracks(self, yield_label: bool = False):
        for line in self.transcript:
            speaker = str(line.get("speaker") or "")
            if not speaker:
                continue
            segment = SimpleNamespace(
                start=float(line.get("start", 0)),
                end=float(line.get("end", 0)),
            )
            if yield_label:
                yield segment, None, speaker
            else:
                yield segment, None


def extract_job_speaker_embeddings_from_transcript(
    job: dict[str, Any],
    *,
    models: Any,
) -> list[LocalSpeakerEmbedding]:
    with transcription_model_session(models):
        return _extract_job_speaker_embeddings_from_transcript(job, models=models)


def _extract_job_speaker_embeddings_from_transcript(
    job: dict[str, Any], *, models: Any,
) -> list[LocalSpeakerEmbedding]:
    embedding_inference = getattr(models, "speaker_embedding_inference", None)
    if embedding_inference is None:
        raise RuntimeError("Embedding-Modell nicht verfügbar")
    transcript = job.get("transcript") or []
    if not transcript:
        raise RuntimeError("Job hat kein Transkript")
    audio_path = job.get("audio_path") or job.get("file_path")
    if not audio_path or not Path(audio_path).exists():
        raise RuntimeError("Audiodatei für Backfill nicht verfügbar")

    try:
        import whisperx
    except Exception as exc:
        raise RuntimeError(f"WhisperX zum Laden der Audiodatei nicht verfügbar: {exc}") from exc

    audio = whisperx.load_audio(audio_path)
    return extract_local_speaker_embeddings(
        audio=audio,
        diarize_segments=TranscriptDiarization(transcript),
        embedding_inference=embedding_inference,
    )


def backfill_speaker_profile_embeddings(
    *,
    profile_id: str | None = None,
    session_id: str | None = None,
) -> SpeakerEmbeddingBackfillResponse:
    diagnostics = speaker_embedding_diagnostics()
    if not diagnostics.loaded and not diagnostics.on_demand:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Embedding-Modell nicht verfügbar",
                "diagnostics": model_to_dict(diagnostics),
            },
        )

    observations = [
        observation
        for observation in load_speaker_observations(session_id=session_id)
        if observation.get("status") in {"confirmed", "manual"}
        and observation.get("profile_id")
        and (profile_id is None or observation.get("profile_id") == profile_id)
    ]
    models = getattr(app.state, "models", None)
    processed_jobs: set[str] = set()
    saved_count = 0
    skipped_count = 0
    errors: list[str] = []

    for observation in observations:
        job_id = observation["job_id"]
        try:
            if not load_job_speaker_embeddings(job_id) and job_id not in processed_jobs:
                job = load_job(job_id)
                if job is None:
                    raise RuntimeError("Job nicht gefunden")
                embeddings = extract_job_speaker_embeddings_from_transcript(
                    job,
                    models=models,
                )
                persist_job_speaker_embeddings(job_id, embeddings)
                processed_jobs.add(job_id)

            result = add_job_embedding_to_profile(
                job_id=job_id,
                local_speaker_id=observation["local_speaker_id"],
                profile_id=observation["profile_id"],
                observation_id=observation["observation_id"],
                storage_reason="manual",
            )
            saved_count += int(result.get("saved_count") or 0)
            if result.get("skipped_reason"):
                skipped_count += 1
        except Exception as exc:
            skipped_count += 1
            errors.append(
                f"{job_id}/{observation['local_speaker_id']}: {safe_exception_label(exc)}"
            )

    return SpeakerEmbeddingBackfillResponse(
        scanned_observation_count=len(observations),
        processed_job_count=len(processed_jobs),
        saved_embedding_count=saved_count,
        skipped_count=skipped_count,
        errors=errors,
    )


def refresh_speaker_suggestions_for_session(
    *,
    job_id: str,
    session_id: str,
) -> None:
    local_embeddings = load_local_speaker_embeddings_from_job(job_id)
    create_speaker_suggestion_observations(
        job_id=job_id,
        session_id=session_id,
        local_embeddings=local_embeddings,
        speaker_memory_opt_in=True,
    )


def build_speaker_match_diagnostic_responses(
    session: dict[str, Any],
) -> list[SpeakerMatchDiagnosticResponse]:
    job_id = session.get("job_id")
    if not job_id:
        return []

    config = speaker_embedding_config_from_env()
    job = get_job_from_cache_or_db(job_id) or {}
    local_speaker_ids = known_local_speaker_ids(session)
    local_embeddings = load_local_speaker_embeddings_from_job(job_id)
    model_names = sorted({embedding.model_name for embedding in local_embeddings})
    if not model_names:
        model_names = [config.model_name]

    profiles = load_speaker_profiles()
    diagnostics_by_speaker: dict[str, SpeakerMatchDiagnosticResponse] = {}
    local_audio_seconds = speaker_audio_seconds_from_transcript(
        session.get("transcript") or job.get("transcript")
    )
    embedding_model_available = speaker_embedding_diagnostics().loaded
    for model_name in model_names:
        model_local_embeddings = [
            embedding
            for embedding in local_embeddings
            if embedding.model_name == model_name
        ]
        embeddings_by_profile = {
            profile["profile_id"]: load_speaker_embeddings(
                profile["profile_id"],
                model_name=model_name,
            )
            for profile in profiles
        }
        references = build_profile_references(profiles, embeddings_by_profile)
        for diagnostic in diagnose_speaker_matches(
            local_speaker_ids=local_speaker_ids,
            local_embeddings=model_local_embeddings,
            profile_references=references,
            config=config,
            local_audio_seconds=local_audio_seconds,
            embedding_model_available=embedding_model_available,
        ):
            diagnostics_by_speaker.setdefault(
                diagnostic.local_speaker_id,
                SpeakerMatchDiagnosticResponse(**diagnostic.__dict__),
            )

    return list(diagnostics_by_speaker.values())


def model_to_dict(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def speaker_embedding_diagnostics(
    session_id: str | None = None,
) -> SpeakerEmbeddingDiagnosticsResponse:
    config = speaker_embedding_config_from_env()
    models = getattr(app.state, "models", None)
    attempted = list(getattr(models, "speaker_embedding_attempted_models", ()) or ())
    model_name = getattr(models, "speaker_embedding_model_name", None)
    error = getattr(models, "speaker_embedding_error", None)
    loaded = bool(getattr(models, "speaker_embedding_inference", None))
    if not attempted and config.enabled:
        attempted = [config.model_name, *config.fallback_model_names]

    profiles = [
        SpeakerEmbeddingProfileDiagnosticResponse(
            profile_id=profile["profile_id"],
            display_name=profile["display_name"],
            embedding_count=count_speaker_embeddings(profile["profile_id"]),
        )
        for profile in load_speaker_profiles()
    ]

    job_id = None
    local_speakers: list[LocalSpeakerEmbeddingDiagnosticResponse] = []
    session = load_session(session_id) if session_id else None
    if session is not None:
        job_id = session.get("job_id")
        if job_id:
            local_embeddings_by_speaker = {
                embedding["local_speaker_id"]: embedding
                for embedding in load_job_speaker_embeddings(job_id)
            }
            for local_speaker_id in sorted(known_local_speaker_ids(session)):
                embedding = local_embeddings_by_speaker.get(local_speaker_id)
                metadata = (embedding or {}).get("quality_metadata") or {}
                local_speakers.append(
                    LocalSpeakerEmbeddingDiagnosticResponse(
                        local_speaker_id=local_speaker_id,
                        embedding_available=embedding is not None,
                        model_name=(embedding or {}).get("model_name"),
                        quality=(embedding or {}).get("quality"),
                        total_seconds=metadata.get("total_seconds"),
                        selected_segment_count=int(
                            metadata.get("selected_segment_count") or 0
                        ),
                        candidate_segment_count=int(
                            metadata.get("candidate_segment_count") or 0
                        ),
                        excluded_short_count=int(
                            metadata.get("excluded_short_count") or 0
                        ),
                        excluded_overlap_count=int(
                            metadata.get("excluded_overlap_count") or 0
                        ),
                    )
                )

    return SpeakerEmbeddingDiagnosticsResponse(
        enabled=config.enabled,
        loaded=loaded,
        on_demand=config.enabled and bool(getattr(models, "gpu_managed", False)),
        model_name=model_name,
        attempted_model_names=attempted,
        error=error,
        torch_force_no_weights_only_load=os.environ.get(
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
        ),
        job_speaker_embedding_count=count_job_speaker_embeddings(job_id),
        profile_embedding_count=count_all_speaker_embeddings(),
        profiles=profiles,
        session_id=session_id if session is not None else None,
        job_id=job_id,
        local_speakers=local_speakers,
    )


def parse_pipeline_tops(raw_tops: str | None) -> list[str]:
    if not raw_tops:
        return []
    text = raw_tops.strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except json.JSONDecodeError:
        pass
    separators = "\n" if "\n" in text else ","
    return [item.strip() for item in text.split(separators) if item.strip()]


def parse_pipeline_options(raw_options: str | None) -> dict[str, Any]:
    if not raw_options or not raw_options.strip():
        return {}
    try:
        parsed = json.loads(raw_options)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Optionen müssen valides JSON sein: {exc}",
        ) from exc
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail="Optionen müssen ein JSON-Objekt sein")
    if parsed.get("agenda_use_llm") is not None and type(parsed["agenda_use_llm"]) is not bool:
        raise HTTPException(status_code=400, detail="agenda_use_llm muss boolesch oder null sein")
    return parsed


def build_pipeline_status_response(job: dict[str, Any]) -> PipelineStatusResponse:
    refs = _pipeline_refs(job)
    return PipelineStatusResponse(
        execution=durable.public(work) if (work := durable.load(job["pipeline_job_id"])) else None,
        pipeline_id=job["pipeline_job_id"],
        session_id=job.get("session_id"),
        transcription_job_id=job.get("transcription_job_id"),
        status=job["status"],
        stage=job["stage"],
        progress=job["progress"],
        warnings=list(refs.get("warnings") or []),
        error=job.get("error"),
        created_at=job.get("created_at"),
        updated_at=job.get("updated_at"),
    )


def session_agenda_proposals(
    session: dict[str, Any], latest_pipeline: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if session.get("agenda_proposals") is not None:
        return session["agenda_proposals"]
    # Old pipeline artifacts have no immutable input snapshot. Preserve warnings,
    # but never rebind their positional indices to today's edited session.
    info = _pipeline_refs(latest_pipeline).get("agenda") if latest_pipeline else None
    if not isinstance(info, dict):
        return None
    return {
        "version": 1,
        "source": None,
        "result": {
            "tops": [], "transcript": [], "assignments": [],
            "segments": info.get("segments") or [],
            "uncertain_count": info.get("uncertain_count") or 0,
            "strategy": info.get("strategy") or "legacy",
            "warnings": info.get("warnings") or [],
            "llm": info.get("llm"),
        },
    }


def build_pipeline_agenda_detection_response(
    agenda_info: Any,
    session: SessionResponse,
) -> AgendaDetectionResponse | None:
    # The session owns the immutable result, including its original assignments.
    # Never combine historical segments with current manual assignments.
    proposals = session.agenda_proposals
    if not proposals:
        return None
    return AgendaDetectionResponse(**proposals["result"])


def line_to_dict(line: Any) -> dict[str, Any]:
    if isinstance(line, dict):
        return {
            "line_id": str(line.get("line_id") or uuid.uuid4()),
            "speaker": str(line.get("speaker", "")),
            "text": str(line.get("text", "")),
            "start": float(line.get("start", 0)),
            "end": float(line.get("end", 0)),
            **({'timing': line['timing']} if line.get('timing') else {}),
        }
    return {
        "line_id": str(getattr(line, "line_id", None) or uuid.uuid4()),
        "speaker": str(getattr(line, "speaker", "")),
        "text": str(getattr(line, "text", "")),
        "start": float(getattr(line, "start", 0)),
        "end": float(getattr(line, "end", 0)),
        **({'timing': line.timing} if getattr(line, 'timing', None) else {}),
    }


def transcript_utterances(transcript: list[dict[str, Any]]) -> list[TranscriptUtterance]:
    return [
        TranscriptUtterance(speaker=line.get("speaker", ""), text=line.get("text", ""),
                            line_id=line.get("line_id"), start=line.get("start"), end=line.get("end"))
        for line in transcript
    ]


def _normalized_summary_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def summary_source_snapshot(
    transcript: list[dict[str, Any]],
    assignments: list[int | None],
    top_index: int,
    *,
    no_top_mode: bool = False,
) -> list[dict[str, str]]:
    snapshot: list[dict[str, str]] = []
    for line_index, line in enumerate(transcript):
        if not no_top_mode and (
            line_index >= len(assignments) or assignments[line_index] != top_index
        ):
            continue
        speaker = str(line.get("speaker") or "")
        text = _normalized_summary_text(line.get("text"))
        if snapshot and snapshot[-1]["speaker"] == speaker:
            snapshot[-1]["text"] = _normalized_summary_text(
                f"{snapshot[-1]['text']} {text}"
            )
        else:
            snapshot.append(
                {
                    "line_id": str(line.get("line_id") or f"legacy:{line_index}"),
                    "speaker": speaker,
                    "text": text,
                }
            )
    return snapshot


def summary_snapshot_hash(snapshot: list[dict[str, str]]) -> str:
    semantic_snapshot = _normalized_summary_text(
        " ".join(item.get("text", "") for item in snapshot)
    )
    payload = json.dumps(
        semantic_snapshot,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def summary_line_indices(session: dict[str, Any], top_index: int) -> list[int]:
    """Include model-proven joint deliberations while honoring manual assignments."""
    transcript = session.get('transcript') or []
    assignments = session.get('assignments') or []
    if session.get('skipped_assignment') or not session.get('tops'):
        return list(range(len(transcript)))
    selected = {i for i, value in enumerate(assignments) if value == top_index and i < len(transcript)}
    proposals = session.get('agenda_proposals') or {}
    source = proposals.get('source') or {}
    result = proposals.get('result') or {}
    usage = result.get('llm') or {}
    ids = session.get('top_ids') or []
    original_ids = source.get('top_ids') or []
    if top_index >= len(ids) or ids[top_index] not in original_ids:
        return sorted(selected)
    original_index = original_ids.index(ids[top_index])
    identities = (usage.get('provenance') or {}).get('identities') or []
    model_ids = {item['top_id'] for item in identities if item.get('top_index') == original_index}
    original_lines = source.get('transcript') or []
    original_assignments = result.get('assignments') or []
    for row in usage.get('line_results') or []:
        i = row.get('index')
        if (type(i) is not int or i < 0 or i >= min(len(transcript), len(original_lines), len(assignments), len(original_assignments))
                or assignments[i] is not None or original_assignments[i] is not None
                or len(row.get('top_ids') or []) < 2 or not model_ids.intersection(row['top_ids'])):
            continue
        current, original = line_to_dict(transcript[i]), line_to_dict(original_lines[i])
        if all(current.get(key) == original.get(key) for key in ('line_id', 'speaker', 'text', 'start', 'end')):
            selected.add(i)
    return sorted(selected)


def current_summary_input(
    session: dict[str, Any], top_index: int
) -> tuple[list[dict[str, str]], str]:
    transcript = [line_to_dict(line) for line in (session.get("transcript") or [])]
    tops = list(session.get("tops") or [])
    no_top_mode = bool(session.get("skipped_assignment")) or not tops
    selected = set(summary_line_indices(session, top_index))
    snapshot = summary_source_snapshot(
        transcript,
        [top_index if i in selected else None for i in range(len(transcript))],
        top_index,
        no_top_mode=no_top_mode,
    )
    return snapshot, summary_snapshot_hash(snapshot)


def _replace_exact_labels(value: Any, replacements: dict[str, str]) -> Any:
    effective = {
        old: new
        for old, new in replacements.items()
        if old and new and old != new
    }
    if not effective:
        return value
    pattern = re.compile(
        r"(?<!\w)(" + "|".join(
            re.escape(item) for item in sorted(effective, key=len, reverse=True)
        ) + r")(?!\w)"
    )

    def replace_string(text: str) -> str:
        return pattern.sub(lambda match: effective[match.group(1)], text)

    if isinstance(value, str):
        return replace_string(value)
    if isinstance(value, list):
        return [_replace_exact_labels(item, effective) for item in value]
    if isinstance(value, dict):
        return {
            key: _replace_exact_labels(item, effective)
            for key, item in value.items()
        }
    return value


def _summary_change_reasons(
    old_snapshot: list[dict[str, str]],
    new_snapshot: list[dict[str, str]],
) -> list[str]:
    old_by_id = {item["line_id"]: item for item in old_snapshot}
    new_by_id = {item["line_id"]: item for item in new_snapshot}
    reasons: list[str] = []
    if old_by_id.keys() - new_by_id.keys():
        reasons.append("lines_removed")
    if new_by_id.keys() - old_by_id.keys():
        reasons.append("lines_added")
    if any(
        old_by_id[line_id] != new_by_id[line_id]
        for line_id in old_by_id.keys() & new_by_id.keys()
    ):
        reasons.append("transcript_changed")
    if not reasons and old_snapshot != new_snapshot:
        reasons.append("line_order_changed")
    return reasons or ["summary_input_changed"]


def ensure_session_top_ids(
    incoming: dict[str, Any], existing: dict[str, Any] | None = None
) -> list[str]:
    tops = list(incoming.get("tops") or [])
    supplied = [str(item) for item in (incoming.get("top_ids") or [])]
    if len(supplied) == len(tops) and len(set(supplied)) == len(supplied):
        return supplied
    previous = list((existing or {}).get("top_ids") or [])
    if len(previous) == len(tops):
        return previous
    return [str(uuid.uuid4()) for _ in tops]


def reconcile_session_summaries(
    existing: dict[str, Any] | None,
    incoming: dict[str, Any],
) -> dict[str, Any]:
    """Preserve summaries by stable TOP id and classify only semantic changes."""
    state = dict(incoming)
    previous_lines = list((existing or {}).get("transcript") or [])
    if state.get("transcript") is None:
        state["transcript"] = previous_lines
    normalized_lines: list[dict[str, Any]] = []
    previous_by_id = {line.get('line_id'): line for line in previous_lines if line.get('line_id')}
    for index, raw_line in enumerate(state.get("transcript") or []):
        normalized = line_to_dict(raw_line)
        supplied_id = (
            raw_line.get("line_id")
            if isinstance(raw_line, dict)
            else getattr(raw_line, "line_id", None)
        )
        if not supplied_id and index < len(previous_lines):
            previous = previous_lines[index]
            if (
                str(previous.get("speaker", "")) == normalized["speaker"]
                and str(previous.get("text", "")) == normalized["text"]
            ):
                normalized["line_id"] = str(
                    previous.get("line_id") or normalized["line_id"]
                )
        prior = previous_by_id.get(normalized['line_id'])
        if prior and prior.get('timing'):
            unchanged = all(prior.get(k) == normalized.get(k) for k in ('text', 'start', 'end'))
            if unchanged and not normalized.get('timing'):
                normalized['timing'] = prior['timing']  # Older clients omit this optional field.
            elif not unchanged and normalized.get('timing') == prior['timing']:
                normalized['timing'] = {'source': 'manual_estimate', 'words': [],
                                        'segments': prior['timing'].get('segments', [])}
        normalized_lines.append(normalized)
    state["transcript"] = normalized_lines
    state["top_ids"] = ensure_session_top_ids(state, existing)
    tops = list(state.get("tops") or [])
    no_top_mode = bool(state.get("skipped_assignment")) or not tops
    effective_ids = (
        [f"whole-session:{state.get('session_id', '')}"]
        if no_top_mode
        else state["top_ids"]
    )

    old_top_ids = list((existing or {}).get("top_ids") or [])
    if existing and (existing.get("skipped_assignment") or not existing.get("tops")):
        old_top_ids = [f"whole-session:{existing.get('session_id', '')}"]
    old_index_by_id = {top_id: index for index, top_id in enumerate(old_top_ids)}
    old_summaries = dict((existing or {}).get("summaries") or {})
    old_reviews = dict((existing or {}).get("summary_reviews") or {})
    old_states = dict((existing or {}).get("summary_states") or {})
    requested_summaries = dict(state.get("summaries") or {})
    requested_reviews = dict(state.get("summary_reviews") or {})
    requested_states = dict(state.get("summary_states") or {})

    old_names = dict((existing or {}).get("speaker_names") or {})
    new_names = dict(state.get("speaker_names") or {})
    speaker_replacements: dict[str, str] = {}
    for speaker_id in set(old_names) | set(new_names):
        old_label = _normalized_summary_text(old_names.get(speaker_id) or speaker_id)
        new_label = _normalized_summary_text(new_names.get(speaker_id) or speaker_id)
        if old_label != new_label:
            speaker_replacements[old_label] = new_label

    next_summaries: dict[int, str] = {}
    next_reviews: dict[int, Any] = {}
    next_states: dict[int, Any] = {}
    for index, top_id in enumerate(effective_ids):
        old_index = old_index_by_id.get(top_id)
        old_summary = old_summaries.get(old_index, "") if old_index is not None else ""
        old_review = old_reviews.get(old_index, {}) if old_index is not None else {}
        previous_state = (
            dict(old_states.get(old_index) or {}) if old_index is not None else {}
        )

        local_replacements = dict(speaker_replacements)
        if old_index is not None and old_index < len((existing or {}).get("tops") or []):
            old_title = str((existing or {}).get("tops", [])[old_index])
            new_title = str(tops[index]) if index < len(tops) else old_title
            if old_title != new_title:
                local_replacements[old_title] = new_title

        # The frontend mirrors deterministic label replacements immediately so
        # the UI does not flicker while autosaving. Treat that exact result as
        # reconciliation, not as a manual content edit that resets the baseline.
        expected_summary = str(
            _replace_exact_labels(str(old_summary or ""), local_replacements)
        )
        expected_review = dict(
            _replace_exact_labels(dict(old_review or {}), local_replacements)
        )
        has_requested_summary = index in requested_summaries
        has_requested_review = index in requested_reviews
        requested_summary = str(requested_summaries.get(index) or "")
        requested_review = dict(requested_reviews.get(index) or {})
        summary_was_edited = has_requested_summary and requested_summary not in {
            str(old_summary or ""),
            expected_summary,
        }
        review_was_edited = (
            has_requested_review
            and requested_review != dict(old_review or {})
            and requested_review != expected_review
        )
        manually_edited = summary_was_edited or review_was_edited
        summary = (
            str(_replace_exact_labels(requested_summary, local_replacements))
            if summary_was_edited
            else expected_summary or requested_summary
        )
        review = (
            dict(_replace_exact_labels(requested_review, local_replacements))
            if review_was_edited
            else expected_review or requested_review
        )
        if summary_was_edited and not review_was_edited:
            # Generated claims and source links no longer describe manual text.
            review = {}

        snapshot, input_hash = current_summary_input(state, index)
        proof = (review.get('structured') or {}).get('verification') or {}
        if proof:
            from summary_grounding import digest
            names = state.get('speaker_names') or {}
            source_lines = [format_line_for_summary(line_to_dict(state['transcript'][i]), names)
                            for i in summary_line_indices(state, index)]
            if proof.get('summary_sha256') != digest(summary) or proof.get('source_sha256') != digest(source_lines):
                review['source_links'] = []
                review['structured']['verification']['processing_complete'] = False
                review['llm_usage'] = {**review.get('llm_usage', {}), 'processing_complete': False}
                review['llm_usage'].pop('original_line_indices', None)
                review['review_warnings'] = [*[w for w in review.get('review_warnings', []) if w.get('kind') != 'verification_required'], {
                    'kind': 'verification_required', 'severity': 'warning', 'line_indices': [], 'excerpt': '',
                    'message': 'Text oder Quellen wurden geändert; die automatische Prüfung gilt für die frühere Fassung.'}]
        baseline_snapshot = list(previous_state.get("source_snapshot") or [])
        baseline_hash = str(previous_state.get("input_hash") or "")
        status = str(previous_state.get("status") or "")
        change_reasons: list[str] = []

        if not summary:
            has_error = any(
                str(item.get("severity", "")).lower() == "error"
                for item in review.get("review_warnings", [])
                if isinstance(item, dict)
            )
            status = "failed" if has_error else "missing"
        elif manually_edited:
            status = "ready"
            baseline_snapshot = snapshot
            baseline_hash = input_hash
        elif not baseline_hash:
            # Existing installations did not persist a generation snapshot.
            # Preserve their summaries and establish a baseline without an LLM call.
            status = "ready"
            baseline_snapshot = snapshot
            baseline_hash = input_hash
        elif baseline_hash != input_hash:
            status = "review_required"
            change_reasons = _summary_change_reasons(baseline_snapshot, snapshot)
        elif status not in {"queued", "running", "failed"}:
            status = "ready"

        next_summaries[index] = summary
        if review:
            next_reviews[index] = review
        next_states[index] = {
            **previous_state,
            **dict(requested_states.get(index) or {}),
            "top_id": top_id,
            "status": status or "missing",
            "input_hash": baseline_hash,
            "source_snapshot": baseline_snapshot,
            "current_input_hash": input_hash,
            "change_reasons": change_reasons,
            "origin": "manual" if manually_edited else previous_state.get("origin", "pipeline"),
            "updated_at": time.time(),
        }

    state["summaries"] = next_summaries
    state["summary_reviews"] = next_reviews
    state["summary_states"] = next_states
    return state


def build_generated_summary_states(
    *,
    session_id: str,
    transcript: list[dict[str, Any]],
    tops: list[str],
    top_ids: list[str],
    assignments: list[int | None],
    summaries: dict[int, str],
    summary_reviews: dict[int, Any],
    origin: str,
) -> dict[int, Any]:
    effective_ids = top_ids if tops else [f"whole-session:{session_id}"]
    states: dict[int, Any] = {}
    for index, top_id in enumerate(effective_ids):
        snapshot = summary_source_snapshot(
            transcript,
            assignments,
            index,
            no_top_mode=not bool(tops),
        )
        review = summary_reviews.get(index) or {}
        has_error = bool(review.get("error") or (review.get("llm_usage") or {}).get("grounding_incomplete")) or any(
            str(item.get("severity", "")).lower() == "error"
            for item in (summary_reviews.get(index) or {}).get("review_warnings", [])
            if isinstance(item, dict)
        )
        states[index] = {
            "top_id": top_id,
            "status": (
                "failed" if has_error
                else "ready" if str(summaries.get(index) or "").strip()
                else "missing"
            ),
            "input_hash": summary_snapshot_hash(snapshot),
            "current_input_hash": summary_snapshot_hash(snapshot),
            "source_snapshot": snapshot,
            "change_reasons": [],
            "origin": origin,
            "generated_at": time.time(),
            "updated_at": time.time(),
        }
    return states


def update_summary_job(summary_job_id: str, **changes: Any) -> dict[str, Any] | None:
    durable.check()
    job = load_summary_job(summary_job_id)
    if job is None:
        return None
    refs = dict(job.get("refs") or {})
    if "refs" in changes:
        refs.update(changes.pop("refs") or {})
    job.update(changes)
    job["refs"] = refs
    job["updated_at"] = time.time()
    return save_summary_job(summary_job_id, job)


def summary_job_cancelled(summary_job_id: str) -> bool:
    durable.check()
    job = load_summary_job(summary_job_id)
    if job is None:
        return True
    return bool((job.get("refs") or {}).get("cancel_requested")) or job.get(
        "status"
    ) == "cancelled"


class SummaryJobInputChanged(ValueError):
    """A safe, user-facing explanation for a TOP that cannot be regenerated."""


def summary_top_ids(session: dict[str, Any]) -> list[str]:
    if not session.get("tops") or session.get("skipped_assignment"):
        return [f"whole-session:{session['session_id']}"]
    return list(session.get("top_ids") or [])


def summary_edit_fingerprint(session: dict[str, Any], index: int) -> str:
    """Protect both source edits and manual output edits, including label changes."""
    _, input_hash = current_summary_input(session, index)
    return hashlib.sha256(json.dumps({
        "input": input_hash,
        "source_details": [line_to_dict(session['transcript'][i]) for i in summary_line_indices(session, index)],
        "title": "Gesamtes Gespräch" if session.get("skipped_assignment") else (session.get("tops") or ["Gesamtes Gespräch"])[index],
        "speakers": session.get("speaker_names") or {},
        "summary": (session.get("summaries") or {}).get(index),
        "review": (session.get("summary_reviews") or {}).get(index),
    }, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def mutate_summary_session(
    session_id: str, mutate: Callable[[dict[str, Any]], bool | None]
) -> dict[str, Any]:
    # Each retry rereads the session and reapplies only the intended TOP patch.
    # Bumping the revision also protects concurrent browser autosaves.
    for _ in range(10):
        session = load_session(session_id)
        if session is None:
            raise RuntimeError("Session nicht gefunden")
        if mutate(session) is False:
            return session
        try:
            return save_session(session_id, session, expected_revision=session["revision"])
        except SessionConflictError:
            continue
    raise RuntimeError("Sitzung wurde wiederholt zwischenzeitlich geändert")


def finalize_summary_job_cancellation(summary_job_id: str, job: dict[str, Any]) -> None:
    refs = dict(job.get("refs") or {})
    def restore(session):
        ids = summary_top_ids(session)
        for top_id in refs.get("top_ids", []):
            if top_id not in ids:
                continue
            index = ids.index(top_id)
            state = (session.get("summary_states") or {}).get(index, {})
            if state.get("status") in {"queued", "running"} and state.get("generation_job_id", summary_job_id) == summary_job_id:
                state["status"] = refs.get("previous_statuses", {}).get(top_id, "missing")
                state["updated_at"] = time.time()
    mutate_summary_session(job["session_id"], restore)
    current = load_summary_job(summary_job_id) or job
    update_summary_job(summary_job_id, status="cancelled",
                       refs={**current.get("refs", {}), "current_top_id": None})


def run_summary_job(summary_job_id: str) -> None:
    def check_cancel():
        if summary_job_cancelled(summary_job_id):
            raise CancellationRequested()
    with request_control(check_cancel, lambda state: update_summary_job(
            summary_job_id, refs={"llm_progress": state})):
        return _run_summary_job(summary_job_id)


def _run_summary_job(summary_job_id: str) -> None:
    job = load_summary_job(summary_job_id)
    if job is None:
        return
    refs = dict(job.get("refs") or {})
    top_ids = list(refs.get("top_ids") or [])
    outcomes: dict[str, Any] = dict(refs.get("outcomes") or {})
    total = len(top_ids)

    def report(**changes):
        update_summary_job(summary_job_id, **changes, refs={
            "outcomes": outcomes, "current_top_id": current_top_id,
        })

    current_top_id: str | None = None
    report(status="processing", progress=0, current_top=0, total_tops=total, error=None)
    for position, top_id in enumerate(top_ids):
        if outcomes.get(top_id, {}).get("status") == "completed" or durable.published("summary:" + top_id):
            outcomes[top_id] = {"status": "completed"}
            continue
        if summary_job_cancelled(summary_job_id):
            finalize_summary_job_cancellation(summary_job_id, job)
            return
        current_top_id = top_id
        report(current_top=position + 1)
        try:
            def check_and_mark(session):
                ids = summary_top_ids(session)
                if top_id not in ids:
                    raise SummaryJobInputChanged("TOP wurde entfernt")
                index = ids.index(top_id)
                expected = refs.get("edit_fingerprints", {}).get(top_id)
                if expected and summary_edit_fingerprint(session, index) != expected:
                    raise SummaryJobInputChanged("TOP wurde zwischenzeitlich bearbeitet; Ergebnis nicht übernommen")
                if current_summary_input(session, index)[1] != refs.get("input_hashes", {}).get(top_id):
                    raise SummaryJobInputChanged("TOP-Eingabe wurde zwischenzeitlich geändert")
                session.setdefault("summary_states", {}).setdefault(index, {}).update(
                    status="running", updated_at=time.time())

            session = mutate_summary_session(job["session_id"], check_and_mark)
            index = summary_top_ids(session).index(top_id)
            snapshot, input_hash = current_summary_input(session, index)
            transcript = [line_to_dict(line) for line in session.get("transcript", [])]
            assignments = session.get("assignments") or []
            no_top_mode = not session.get("tops") or session.get("skipped_assignment")
            source_indices = summary_line_indices(session, index)
            lines = [transcript[i] for i in source_indices]
            if not lines:
                raise SummaryJobInputChanged("Für den ausgewählten TOP sind keine Zeilen vorhanden")
            title = "Gesamtes Gespräch" if no_top_mode else session["tops"][index]
            names = session.get("speaker_names") or {}
            text = "\n".join(format_line_for_summary(line, names) for line in lines)
            with work_slot(LLM_WORK_LOCK):
                result = summarize_segment(title, text, model=refs.get("model"),
                    system_prompt=refs.get("system_prompt"),
                    meeting_context=meeting_context_from_transcript(transcript),
                    source_lines=[format_line_for_summary(line, names) for line in lines])
            result.llm_usage["original_line_indices"] = source_indices
            if summary_job_cancelled(summary_job_id):
                finalize_summary_job_cancellation(summary_job_id, job)
                return
            review = build_summary_review(structured=result.structured, summary=result.summary,
                lines=[{**line, "speaker": names.get(line["speaker"], line["speaker"])} for line in lines])
            if not getattr(result, "llm_usage", {}).get("processing_complete"):
                raise RuntimeError("Zusammenfassung technisch unvollständig")
            if not result.summary.strip():
                raise SummaryJobInputChanged("Die Generierung lieferte keine Zusammenfassung")

            def publish(latest):
                check_and_mark(latest)
                target = summary_top_ids(latest).index(top_id)
                latest.setdefault("summaries", {})[target] = result.summary
                latest.setdefault("summary_reviews", {})[target] = {
                    "structured": result.structured.to_dict() if result.structured else None,
                    "source_links": [link.to_dict() for link in review.source_links],
                    "review_warnings": [warning.to_dict() for warning in review.warnings],
                    "fallback_used": result.fallback_used,
                    "chunks_processed": result.chunks_processed,
                    "llm_usage": getattr(result, "llm_usage", {}),
                    "duration_seconds": result.duration_seconds,
                }
                latest["summary_states"][target].update(
                    top_id=top_id, status="ready", input_hash=input_hash,
                    current_input_hash=input_hash, source_snapshot=snapshot,
                    change_reasons=[], origin="manual_regeneration",
                    generated_at=time.time(), updated_at=time.time())
            with durable.publication("summary:" + top_id):
                mutate_summary_session(job["session_id"], publish)
            outcomes[top_id] = {"status": "completed"}
        except LLMCancelledError:
            if durable.CURRENT.get():
                raise
            finalize_summary_job_cancellation(summary_job_id, job)
            return
        except Exception as exc:
            durable.raise_if_transient(exc)
            # Retain successful TOPs and continue the same serial job after a failure.
            message = str(exc) if isinstance(exc, SummaryJobInputChanged) else safe_exception_label(exc)
            outcomes[top_id] = {"status": "failed", "error": message}
            def mark_failed(latest):
                ids = summary_top_ids(latest)
                if top_id not in ids:
                    return False
                target = ids.index(top_id)
                state = latest.setdefault("summary_states", {}).setdefault(target, {})
                if state.get("status") not in {"queued", "running"}:
                    return False  # A manual edit already established a newer state.
                state.update(status="failed", updated_at=time.time())
                review = latest.setdefault("summary_reviews", {}).setdefault(target, {})
                review["review_warnings"] = [*review.get("review_warnings", []), {
                    "kind": "summary_failed", "message": message, "severity": "error",
                    "line_indices": [], "excerpt": "",
                }]
            try:
                mutate_summary_session(job["session_id"], mark_failed)
            except Exception:
                logger.warning("Could not persist failure state for summary job %s", summary_job_id)
        current_top_id = None
        report(progress=int(len(outcomes) / max(total, 1) * 100))

    failed = sum(item["status"] == "failed" for item in outcomes.values())
    report(status="failed" if failed else "completed", progress=100,
           error=f"{failed} von {total} TOPs fehlgeschlagen" if failed else None)


def save_pipeline_session(
    session_id: str,
    *,
    job_id: str | None = None,
    transcript: list[dict[str, Any]] | None = None,
    tops: list[str] | None = None,
    top_ids: list[str] | None = None,
    assignments: list[int | None] | None = None,
    summaries: dict[int, str] | None = None,
    summary_reviews: dict[int, Any] | None = None,
    summary_states: dict[int, Any] | None = None,
    agenda_proposals: dict[str, Any] | None = None,
    export_metadata: dict[str, Any] | None = None,
    current_step: int | None = None,
    skipped_assignment: bool | None = None,
) -> dict[str, Any]:
    session = load_session(session_id) or {"session_id": session_id}
    ctx = durable.CURRENT.get()
    if ctx and summaries is None:
        return session
    if ctx and durable.published("pipeline:published"):
        return session
    state = dict(session)
    if job_id is not None:
        state["job_id"] = job_id
    if transcript is not None:
        state["transcript"] = transcript
        speaker_names = dict(state.get("speaker_names") or {})
        for line in transcript:
            speaker = str(line.get("speaker", "")).strip()
            if speaker:
                speaker_names.setdefault(speaker, speaker)
        state["speaker_names"] = speaker_names
    if tops is not None:
        state["tops"] = tops
        existing_ids = list(state.get("top_ids") or [])
        state["top_ids"] = (
            list(top_ids)
            if top_ids is not None and len(top_ids) == len(tops)
            else existing_ids
            if len(existing_ids) == len(tops)
            else [str(uuid.uuid4()) for _ in tops]
        )
    if assignments is not None:
        state["assignments"] = assignments
    if summaries is not None:
        state["summaries"] = summaries
    if summary_reviews is not None:
        state["summary_reviews"] = summary_reviews
    if summary_states is not None:
        state["summary_states"] = summary_states
    if agenda_proposals is not None:
        state["agenda_proposals"] = agenda_proposals
    if export_metadata is not None:
        current_metadata = dict(state.get("export_metadata") or {})
        for key, value in export_metadata.items():
            if value is not None and str(value).strip():
                current_metadata[key] = value
        state["export_metadata"] = current_metadata
    if current_step is not None:
        state["current_step"] = current_step
    if skipped_assignment is not None:
        state["skipped_assignment"] = skipped_assignment
    state.setdefault("speaker_names", {})
    state.setdefault("tops", [])
    state.setdefault("top_ids", [])
    state.setdefault("assignments", [])
    state.setdefault("summaries", {})
    state.setdefault("summary_reviews", {})
    state.setdefault("summary_states", {})
    state.setdefault("export_metadata", {})
    state.setdefault("skipped_assignment", False)
    with durable.publication("pipeline:published"):
        return save_session(session_id, state,
            expected_revision=ctx.payload.get("session_revision") if ctx else session.get("revision"),
            bump_revision=True)


def fallback_agenda(transcript, known_tops=None):
    """Legacy entry point: technical nonprocessing, never an invented agenda."""
    tops = list(known_tops or [])
    return tops, [None] * len(transcript), {
        "strategy": "agenda_not_processed", "segments": [], "uncertain_count": 0,
        "llm": {"enabled": True, "source": "pipeline", "status": "failed",
                "timeout_seconds": 0, "attempted_calls": 0, "failed_calls": 0,
                "processing_complete": False, "review_complete": False,
                "processed_lines": [], "gaps": [{"start_index": 0,
                    "end_index": len(transcript)-1, "kind": "technical", "reason": "model_unavailable"}] if transcript else []},
    }


def detect_pipeline_agenda(
    pipeline_id: str,
    transcript: list[dict[str, Any]],
    *,
    known_tops: list[str],
    pdf_path: str | None,
    options: dict[str, Any],
) -> tuple[list[str], list[int | None], dict[str, Any], dict[str, Any]]:
    agenda_tops = list(known_tops)
    pdf_metadata: dict[str, Any] = {}
    pdf_incomplete = False
    model = options.get("agenda_model") or options.get("model")
    # The legacy system_prompt belongs only to summarization.
    system_prompt = options.get("agenda_system_prompt")
    if options.get("skip_agenda_detection"):
        return [], [None for _ in transcript], {
            "strategy": "no_agenda_requested",
            "segments": [],
            "uncertain_count": 0,
        }, pdf_metadata

    pdf_extraction = options.get("pdf_source_extraction")
    if pdf_extraction:
        pdf_metadata = pdf_extraction["metadata"]
        if not agenda_tops:
            agenda_tops = pdf_extraction["tops"]
    if not agenda_tops and (pdf_path or options.get("auto_detect_tops_from_pdf")):
        if not pdf_path or not Path(pdf_path).is_file():
            raise ValueError("Vorgesehenes PDF fehlt; keine Ersatzagenda aus dem Transkript")
        extracted = durable.checkpoint("pipeline:pdf:page-evidence-v3", lambda:
            extract_agenda_data_from_pdf(pdf_path, model=model,
                system_prompt=options.get("pdf_system_prompt")).to_dict())
        save_pipeline_state(pipeline_id, result_refs={"pdf_extraction": extracted, "processing_complete": False})
        if not extracted["processing_complete"] or extracted["review_required"] or not extracted["tops"]:
            raise PdfReviewRequired(extracted)
        agenda_tops = extracted["tops"]
        pdf_metadata = extracted["metadata"]
        pdf_extraction = extracted
        save_pipeline_state(pipeline_id, result_refs={"pdf_extraction": extracted})

    detection_details: dict[str, Any] = {"pdf_incomplete": pdf_incomplete, "pdf_extraction": pdf_extraction}
    try:
        utterances = transcript_utterances(transcript)
        if agenda_tops:
            result = segment_known_agenda(
                utterances,
                agenda_tops,
                model=model,
                system_prompt=system_prompt,
                use_llm=options.get("agenda_use_llm"),
                cache_namespace=options.get('agenda_cache_namespace', ''),
                progress_callback=lambda usage: save_pipeline_state(
                    pipeline_id, result_refs={"agenda_progress": asdict(usage)}),
            )
        else:
            result = detect_agenda_from_transcript(
                utterances,
                model=model,
                system_prompt=system_prompt,
                use_llm=options.get("agenda_use_llm"),
                cache_namespace=options.get('agenda_cache_namespace', ''),
                progress_callback=lambda usage: save_pipeline_state(
                    pipeline_id, result_refs={"agenda_progress": asdict(usage)}),
            )
        usage = result.llm
        warnings = usage.warnings if usage else []
        detection_details = {"pdf_incomplete": pdf_incomplete, "pdf_extraction": pdf_extraction, "llm": asdict(usage) if usage else None, "warnings": warnings}
        for warning in warnings:
            append_pipeline_warning(pipeline_id, warning)
        return result.tops, result.assignments, {
            **detection_details, "strategy": result.strategy,
            "segments": [segment.__dict__ for segment in result.segments],
            "uncertain_count": result.uncertain_count,
        }, pdf_metadata
    except LLMCancelledError:
        raise
    except Exception as exc:
        durable.raise_if_transient(exc)
        message = f"TOP-Erkennung technisch fehlgeschlagen ({safe_exception_label(exc)}); Zuordnung prüfen."
        append_pipeline_warning(pipeline_id, message)
        return agenda_tops, [None] * len(transcript), {
            "strategy": "known_agenda_failed", "segments": [], "uncertain_count": 0,
            "pdf_extraction": pdf_extraction, "warnings": [message], "llm": {
                "enabled": True, "source": "pipeline", "status": "failed",
                "timeout_seconds": 0, "attempted_calls": 0, "failed_calls": 0,
                "failure_reasons": [safe_exception_label(exc)], "processed_lines": [],
                "processing_complete": False, "review_complete": False,
                "line_results": [{"line_id": row['line_id'], "index": i, "top_ids": [],
                    "status": "not_processed", "review_status": "technical_pending",
                    "reason": safe_exception_label(exc), "evidence": []} for i, row in enumerate(transcript)],
                "gaps": [{"start_index": 0, "end_index": len(transcript)-1,
                          "kind": "technical", "reason": safe_exception_label(exc)}] if transcript else [],
            },
        }, pdf_metadata



def format_line_for_summary(
    line: dict[str, Any],
    speaker_names: dict[str, str] | None = None,
) -> str:
    speaker = str(line.get("speaker", ""))
    display_name = (speaker_names or {}).get(speaker, speaker)
    return f"{display_name}: {line.get('text', '')}"


def summarize_pipeline_segments(
    pipeline_id: str,
    *,
    transcript: list[dict[str, Any]],
    tops: list[str],
    assignments: list[int | None],
    options: dict[str, Any],
    speaker_names: dict[str, str] | None = None,
    source_session: dict[str, Any] | None = None,
) -> tuple[dict[int, str], dict[int, Any]]:
    prior = _pipeline_refs(load_pipeline_job(pipeline_id)).get("summary_progress") or {}
    summaries = {int(k): v for k, v in (prior.get("summaries") or {}).items()}
    summary_reviews = {int(k): v for k, v in (prior.get("summary_reviews") or {}).items()}
    model = options.get("summary_model") or options.get("model")
    system_prompt = options.get("summary_system_prompt")
    if system_prompt is None:
        system_prompt = options.get("system_prompt")
    if not tops:
        try:
            transcript_text = "\n".join(
                format_line_for_summary(line, speaker_names) for line in transcript
            )
            with work_slot(LLM_WORK_LOCK):
                result = summarize_segment(
                    "Gesamtes Gespräch",
                    transcript_text,
                    source_lines=[format_line_for_summary(line, speaker_names) for line in transcript],
                    model=model,
                    system_prompt=system_prompt,
                )
            review = build_summary_review(
                structured=result.structured,
                summary=result.summary,
                lines=[{**line, "speaker": (speaker_names or {}).get(line["speaker"], line["speaker"])} for line in transcript],
            )
            return {
                0: result.summary,
            }, {
                0: {
                    "structured": (
                        result.structured.to_dict() if result.structured is not None else None
                    ),
                    "source_links": [link.to_dict() for link in review.source_links],
                    "review_warnings": [warning.to_dict() for warning in review.warnings],
                    "fallback_used": result.fallback_used,
                    "chunks_processed": result.chunks_processed,
                    "llm_usage": getattr(result, "llm_usage", {}),
                    "duration_seconds": result.duration_seconds,
                }
            }
        except LLMCancelledError:
            raise
        except Exception as exc:
            durable.raise_if_transient(exc)
            if isinstance(exc, LLMCallError):
                message = str(exc)
            else:
                message = (
                    "Zusammenfassung für das Gesamtgespräch fehlgeschlagen "
                    f"({safe_exception_label(exc)})."
                )
            append_pipeline_warning(pipeline_id, message)
            return {}, {
                0: {
                    "structured": None,
                    "source_links": [],
                    "review_warnings": [
                        {
                            "kind": "summary_failed",
                            "message": message,
                            "severity": "error",
                            "line_indices": [],
                            "excerpt": "",
                        }
                    ],
                    "error": safe_exception_label(exc),
                }
            }

    for top_index, top_title in enumerate(tops):
        if top_index in summaries and not summary_reviews.get(top_index, {}).get("error"):
            continue
        source_indices = (summary_line_indices(source_session, top_index) if source_session else
                          [i for i in range(len(transcript)) if i < len(assignments) and assignments[i] == top_index])
        lines = [transcript[i] for i in source_indices]
        if not lines:
            summaries[top_index] = ""
            summary_reviews[top_index] = {
                "structured": None,
                "source_links": [],
                "review_warnings": [
                    {
                        "kind": "empty_top_segment",
                        "message": "Für diesen TOP wurden keine Transkriptzeilen zugeordnet.",
                        "severity": "warning",
                        "line_indices": [],
                        "excerpt": "",
                    }
                ],
            }
            continue

        transcript_text = "\n".join(
            format_line_for_summary(line, speaker_names) for line in lines
        )
        try:
            with work_slot(LLM_WORK_LOCK):
                result = summarize_segment(
                    top_title,
                    transcript_text,
                    source_lines=[format_line_for_summary(line, speaker_names) for line in lines],
                    model=model,
                    system_prompt=system_prompt,
                    meeting_context=meeting_context_from_transcript(transcript),
                )
            review = build_summary_review(
                structured=result.structured,
                summary=result.summary,
                lines=[{**line, "speaker": (speaker_names or {}).get(line["speaker"], line["speaker"])} for line in lines],
            )
            result.llm_usage["original_line_indices"] = source_indices
            summaries[top_index] = result.summary
            summary_reviews[top_index] = {
                "structured": (
                    result.structured.to_dict() if result.structured is not None else None
                ),
                "source_links": [link.to_dict() for link in review.source_links],
                "review_warnings": [warning.to_dict() for warning in review.warnings],
                "fallback_used": result.fallback_used,
                "chunks_processed": result.chunks_processed,
                "llm_usage": getattr(result, "llm_usage", {}),
                "duration_seconds": result.duration_seconds,
            }
        except LLMCancelledError:
            raise
        except Exception as exc:
            durable.raise_if_transient(exc)
            if isinstance(exc, LLMCallError):
                message = str(exc)
            else:
                message = (
                    f"Zusammenfassung für TOP {top_index + 1} fehlgeschlagen "
                    f"({safe_exception_label(exc)})."
                )
            append_pipeline_warning(pipeline_id, message)
            summaries[top_index] = ""
            summary_reviews[top_index] = {
                "structured": None,
                "source_links": [],
                "review_warnings": [
                    {
                        "kind": "summary_failed",
                        "message": message,
                        "severity": "error",
                        "line_indices": [],
                        "excerpt": "",
                    }
                ],
                "error": safe_exception_label(exc),
            }

        save_pipeline_state(pipeline_id, result_refs={
            "summary_progress": {"completed_tops": top_index + 1, "total_tops": len(tops),
                                 "summaries": summaries, "summary_reviews": summary_reviews}})

    return summaries, summary_reviews


def run_pipeline_job(pipeline_id: str, models: TranscriptionModels) -> None:
    with request_control(lambda: ensure_pipeline_not_cancelled(pipeline_id),
                         lambda state: save_pipeline_state(pipeline_id, result_refs={"llm_progress": state})):
        return _run_pipeline_job(pipeline_id, models)


def _run_pipeline_job(
    pipeline_id: str,
    models: TranscriptionModels,
) -> None:
    job = load_pipeline_job(pipeline_id)
    if job is None:
        return
    refs = _pipeline_refs(job)
    session_id = job.get("session_id")
    transcription_job_id = job.get("transcription_job_id")
    audio_path = refs.get("audio_path")
    pdf_path = refs.get("pdf_path")
    options = dict(refs.get("options") or {})
    known_tops = list(refs.get("known_tops") or [])

    if not session_id or not transcription_job_id:
        save_pipeline_state(
            pipeline_id,
            status=PIPELINE_STATUS_FAILED,
            error="Pipeline ist unvollständig initialisiert",
            progress=0,
        )
        return

    try:
        ensure_pipeline_not_cancelled(pipeline_id)
        save_pipeline_state(
            pipeline_id,
            status=PIPELINE_STATUS_PROCESSING,
            stage=PIPELINE_STAGE_TRANSCRIBE,
            progress=15,
            error=None,
        )
        def obtain_transcript():
            if (load_job(transcription_job_id) or {}).get("status") != JOB_STATUS_COMPLETED:
                run_transcription(transcription_job_id, audio_path, models)
            ensure_pipeline_not_cancelled(pipeline_id)
            transcription_job = load_job(transcription_job_id)
            if transcription_job is None or transcription_job.get("status") != JOB_STATUS_COMPLETED:
                error = transcription_job.get("error") if transcription_job else "Transkriptionsjob nicht gefunden"
                raise RuntimeError(error or "Transkription fehlgeschlagen")
            return [line_to_dict(line) for line in (transcription_job.get("transcript") or [])]

        # A committed transcript is sufficient to resume, even if the transient
        # transcription job has since been cleaned up. Never reload Whisper here.
        transcript = durable.checkpoint("pipeline:transcript", obtain_transcript)
        save_pipeline_session(
            session_id,
            job_id=transcription_job_id,
            transcript=transcript,
            current_step=1,
        )
        save_pipeline_state(
            pipeline_id,
            stage=PIPELINE_STAGE_SPEAKER_MATCH,
            progress=62,
            result_refs={"transcript_line_count": len(transcript)},
        )

        try:
            suggestions = build_speaker_suggestion_responses(
                transcription_job_id,
                session_id,
            )
            save_pipeline_state(
                pipeline_id,
                result_refs={"speaker_suggestion_count": len(suggestions or [])},
            )
        except Exception as exc:
            append_pipeline_warning(
                pipeline_id,
                f"Sprecher-Matching konnte nicht ausgewertet werden: {exc}",
            )

        ensure_pipeline_not_cancelled(pipeline_id)
        save_pipeline_state(
            pipeline_id,
            stage=PIPELINE_STAGE_AGENDA_DETECT,
            progress=72,
        )
        transcript = durable.checkpoint("pipeline:agenda-transcript:v1",
            lambda: split_transcript_for_agenda_detection(transcript))
        tops, assignments, agenda_info, pdf_metadata = durable.checkpoint("pipeline:agenda", lambda: detect_pipeline_agenda(
            pipeline_id, transcript, known_tops=known_tops, pdf_path=pdf_path, options=options))
        pdf_extraction = agenda_info.get("pdf_extraction")
        exact_pdf_agenda = bool(pdf_extraction and tops[:len(pdf_extraction["tops"])] == pdf_extraction["tops"])
        pdf_ids = [item["id"] for item in (pdf_extraction or {}).get("items", []) if item["kind"] == "agenda"] if exact_pdf_agenda else []
        if options.get("auto_detect_tops_from_pdf") and not known_tops and pdf_extraction and not exact_pdf_agenda:
            raise ValueError("Transkriptzuordnung hat die geprüfte PDF-Agenda verändert")
        top_ids = durable.checkpoint("pipeline:top_ids", lambda: pdf_ids + [str(uuid.uuid4()) for _ in tops[len(pdf_ids):]])
        for identity in ((agenda_info.get('llm') or {}).get('provenance') or {}).get('identities', []):
            identity['top_uid'] = top_ids[identity['top_index']]
        # Freeze the exact detector input and result before manual editing begins.
        agenda_proposals = {
            "version": 1,
            "source": {"tops": tops, "top_ids": top_ids, "transcript": transcript, **({"pdf_extraction": pdf_extraction} if pdf_extraction else {})},
            "result": {
                **agenda_info, "tops": tops, "transcript": transcript,
                "assignments": assignments,
            },
        }
        save_pipeline_session(
            session_id,
            job_id=transcription_job_id,
            transcript=transcript,
            tops=tops,
            top_ids=top_ids,
            assignments=assignments,
            export_metadata=pdf_metadata,
            agenda_proposals=agenda_proposals,
            skipped_assignment=not bool(tops),
            current_step=2,
        )
        save_pipeline_state(
            pipeline_id,
            result_refs={"agenda": agenda_info, "top_count": len(tops)},
        )

        ensure_pipeline_not_cancelled(pipeline_id)
        save_pipeline_state(
            pipeline_id,
            stage=PIPELINE_STAGE_SUMMARIZE,
            progress=82,
        )
        if not tops and not options.get('skip_agenda_detection'):
            summaries, summary_reviews = {}, {}
            append_pipeline_warning(pipeline_id, 'Keine belegte Agenda; keine automatische Ersatz-Zusammenfassung.')
        else:
            summaries, summary_reviews = durable.checkpoint("pipeline:summaries", lambda: summarize_pipeline_segments(
                pipeline_id, transcript=transcript, tops=tops, assignments=assignments, options=options,
                source_session=dict(transcript=transcript, tops=tops, top_ids=top_ids,
                                    assignments=assignments, agenda_proposals=agenda_proposals)))
        summaries = {int(k): v for k, v in summaries.items()}
        summary_reviews = {int(k): v for k, v in summary_reviews.items()}
        summary_states = build_generated_summary_states(
            session_id=session_id,
            transcript=transcript,
            tops=tops,
            top_ids=top_ids,
            assignments=assignments,
            summaries=summaries,
            summary_reviews=summary_reviews,
            origin="pipeline",
        )
        for index, state in summary_states.items():
            snapshot, source_hash = current_summary_input(dict(transcript=transcript, tops=tops,
                top_ids=top_ids, assignments=assignments, agenda_proposals=agenda_proposals), index)
            state.update(source_snapshot=snapshot, input_hash=source_hash, current_input_hash=source_hash)
        save_pipeline_session(
            session_id,
            job_id=transcription_job_id,
            transcript=transcript,
            tops=tops,
            top_ids=top_ids,
            assignments=assignments,
            summaries=summaries,
            summary_reviews=summary_reviews,
            summary_states=summary_states,
            agenda_proposals=agenda_proposals,
            export_metadata=pdf_metadata,
            skipped_assignment=not bool(tops),
            current_step=2 if not tops else 3,
        )

        ensure_pipeline_not_cancelled(pipeline_id)
        agenda_usage = agenda_info.get("llm") or {}
        expected = {i for i in range(len(tops)) if summary_line_indices(dict(
            transcript=transcript, tops=tops, top_ids=top_ids, assignments=assignments,
            agenda_proposals=agenda_proposals), i)} if tops else {0}
        complete = (not agenda_info.get("pdf_incomplete")
                    and (options.get('skip_agenda_detection') or
                         (agenda_usage.get('processing_complete') and agenda_usage.get('review_complete')))
                    and expected <= set(summary_reviews)
                    and all(not summary_reviews[i].get("error")
                            and (summary_reviews[i].get("llm_usage") or {}).get("processing_complete")
                            for i in expected)
                    and (not tops or len(assignments) == len(transcript)))
        save_pipeline_state(
            pipeline_id,
            status=PIPELINE_STATUS_COMPLETED if complete else PIPELINE_STATUS_FAILED,
            stage=PIPELINE_STAGE_READY_FOR_REVIEW,
            progress=100,
            error=None,
            result_refs={"ready_for_review": True,
                         "processing_complete": complete,
                         "unassigned_line_count": assignments.count(None)},
        )
    except LLMCancelledError:
        if durable.CURRENT.get():
            raise
        save_pipeline_state(
            pipeline_id,
            status=PIPELINE_STATUS_CANCELLED,
            error=None,
            result_refs={"cancel_requested": True},
        )
    except Exception as exc:
        if durable.CURRENT.get():
            raise
        logger.error(
            "[Pipeline %s] failed (%s)",
            pipeline_id,
            safe_exception_label(exc),
            exc_info=True,
        )
        save_pipeline_state(
            pipeline_id,
            status=PIPELINE_STATUS_FAILED,
            error=safe_exception_label(exc),
        )


def request_pipeline_cancellation(pipeline_id: str) -> dict[str, Any] | None:
    if durable.load(pipeline_id):
        durable.cancel(pipeline_id)
    job = load_pipeline_job(pipeline_id)
    if job is None:
        return None
    if job.get("status") == PIPELINE_STATUS_CANCELLED:
        return job
    if job.get("status") in {PIPELINE_STATUS_COMPLETED, PIPELINE_STATUS_FAILED}:
        raise HTTPException(
            status_code=409,
            detail="Pipeline kann in diesem Status nicht abgebrochen werden",
        )

    refs = _pipeline_refs(job)
    refs["cancel_requested"] = True
    updated = save_pipeline_state(
        pipeline_id,
        status=PIPELINE_STATUS_CANCELLED,
        result_refs=refs,
        error=None,
    )

    transcription_job_id = job.get("transcription_job_id")
    if transcription_job_id:
        try:
            request_job_cancellation(transcription_job_id)
        except HTTPException as exc:
            if exc.status_code != 409:
                raise

    return updated


# ----- Endpoints -----


@app.get("/")
async def root():
    return {"message": "GremienPilot API", "version": "0.1.0"}


@app.get("/health")
async def health_check():
    """
    Health check endpoint for Docker/Kubernetes.
    Returns 200 when the model holder is ready (including on-demand mode).
    """
    if not getattr(app.state, "models_loaded", False):
        raise HTTPException(
            status_code=503, detail="Models not loaded yet - server starting up"
        )
    manager = getattr(app.state, "durable_manager", None)
    if manager and manager.started and not manager.thread.is_alive():
        raise HTTPException(503, 'Hintergrund-Worker nicht verfügbar')
    if manager and manager.storage_fault:
        raise HTTPException(503, 'Datenbankintegrität gestört; gesicherte Wiederherstellung erforderlich'
                            if manager.storage_fault == 'DatabaseError' else 'Datenbankspeicher vorübergehend nicht verfügbar')
    models = getattr(app.state, "models", None)
    on_demand = bool(getattr(models, "gpu_managed", False))
    return {
        "status": "healthy",
        "models_loaded": bool(getattr(models, "whisper_model", None)) if on_demand else True,
        "models_on_demand": on_demand,
        "version": "0.1.0",
        "speaker_embeddings": model_to_dict(speaker_embedding_diagnostics()),
    }


@app.get(
    "/api/speaker-embeddings/diagnostics",
    response_model=SpeakerEmbeddingDiagnosticsResponse,
)
async def speaker_embedding_diagnostics_endpoint(session_id: Optional[str] = None):
    """Report whether persistent speaker recognition can create embeddings."""
    if session_id is not None and load_session(session_id) is None:
        raise HTTPException(status_code=404, detail="Session nicht gefunden")
    return speaker_embedding_diagnostics(session_id=session_id)


@app.get("/api/llm/diagnostics", response_model=LLMDiagnosticsResponse)
async def llm_diagnostics_endpoint(model: Optional[str] = None):
    """Check the configured OpenAI-compatible LLM endpoint and model."""

    diagnostics = llm_diagnostics(model=model)
    status_code = 200 if diagnostics.ok else 503
    if diagnostics.ok:
        return LLMDiagnosticsResponse(**diagnostics.to_dict())
    raise HTTPException(status_code=status_code, detail=diagnostics.to_dict())


@app.post("/api/pipeline/start", response_model=PipelineStartResponse)
async def start_pipeline(
    request: Request,
    audio: UploadFile = File(...),
    pdf: Optional[UploadFile] = File(None),
    pdf_source_job_id: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
    tops: Optional[str] = Form(None),
    options: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    system_prompt: Optional[str] = Form(None),
    summary_system_prompt: Optional[str] = Form(None),
    agenda_system_prompt: Optional[str] = Form(None),
    agenda_use_llm: Optional[bool] = Form(None),
    agenda_fresh: bool = Form(False),
    pdf_system_prompt: Optional[str] = Form(None),
    remember_speakers: bool = Form(False),
    skip_agenda_detection: bool = Form(False),
    auto_detect_tops_from_pdf: bool = Form(False),
):
    """Start an unattended upload-to-review pipeline job."""
    if (
        not getattr(app.state, "models_loaded", False)
        or getattr(app.state, "models", None) is None
    ):
        raise HTTPException(
            status_code=503,
            detail="Server startet noch - Modelle werden geladen. Bitte warten.",
        )
    if not is_allowed_audio_file(audio.filename, audio.content_type):
        raise HTTPException(
            status_code=400,
            detail="Ungültiger Dateityp. Erlaubt: MP3, WAV, M4A",
        )
    if pdf is not None and pdf.filename:
        if not is_allowed_pdf_file(pdf.filename, pdf.content_type):
            raise HTTPException(status_code=400, detail="Nur PDF-Dateien sind erlaubt")

    source_extraction = None
    if pdf_source_job_id:
        source_job = durable.load(pdf_source_job_id)
        if not source_job or source_job['kind'] != 'pdf' or source_job['state'] != 'completed':
            raise HTTPException(422, 'PDF-Quelljob ist nicht vollständig abgeschlossen')
        source_extraction = source_job.get('result')
        if not source_extraction or source_extraction.get('contract_version') != 'page-evidence-v3' or not source_extraction.get('processing_complete') or source_extraction.get('review_required'):
            raise HTTPException(422, 'PDF-Quelljob hat keine vollständige visuelle Quellenprüfung nach aktuellem Vertrag; neue Auswertung erforderlich')
        source_hash = (source_extraction.get('document') or {}).get('sha256')
        if not source_hash or not source_extraction.get('items') or not source_extraction.get('audits') or not any(
            doc['sha256'] == source_hash for doc in source_job.get('documents') or []
        ):
            raise HTTPException(422, 'PDF-Quelljob hat keine vollständige visuelle Quellenprüfung; neue Auswertung erforderlich')
        for document in source_job.get('documents') or []:
            if not Path(document['path']).is_file() or durable.document(document['path'])['sha256'] != document['sha256']:
                raise HTTPException(409, 'PDF-Quelle fehlt oder wurde verändert')

    pipeline_id = str(uuid.uuid4())
    transcription_job_id = str(uuid.uuid4())
    effective_session_id = session_id or str(uuid.uuid4())
    parsed_options = parse_pipeline_options(options)
    parsed_options.pop("pdf_source_extraction", None)  # only verified server-owned provenance
    if agenda_fresh:
        parsed_options['agenda_cache_namespace'] = str(uuid.uuid4())
    if agenda_use_llm is not None:
        parsed_options["agenda_use_llm"] = agenda_use_llm
    if model:
        parsed_options["model"] = model
    if system_prompt:
        parsed_options["system_prompt"] = system_prompt
    form = await request.form()
    for key, prompt in (
        ("summary_system_prompt", summary_system_prompt),
        ("agenda_system_prompt", agenda_system_prompt),
        ("pdf_system_prompt", pdf_system_prompt),
    ):
        if prompt is not None:
            parsed_options[key] = prompt
        elif form.get(key) == "":
            # FastAPI normalizes empty optional form strings to None. Preserve
            # explicit resets so neither JSON options nor the legacy alias win.
            parsed_options[key] = ""
    if source_extraction:
        parsed_options["pdf_source_extraction"] = source_extraction
    parsed_options["skip_agenda_detection"] = skip_agenda_detection
    parsed_options["auto_detect_tops_from_pdf"] = auto_detect_tops_from_pdf
    known_tops = parse_pipeline_tops(tops)
    if source_extraction and not known_tops:
        known_tops = source_extraction['tops']
    if pdf is not None and pdf.filename and not known_tops:
        parsed_options["auto_detect_tops_from_pdf"] = True
    if skip_agenda_detection:
        known_tops = []
        pdf = None
        parsed_options["auto_detect_tops_from_pdf"] = False

    if parsed_options["auto_detect_tops_from_pdf"] and not known_tops and not (pdf and pdf.filename):
        raise HTTPException(422, "Automatische PDF-Erkennung benötigt eine hochgeladene Einladung")

    safe_audio_filename = normalize_upload_filename(
        audio.filename,
        default_stem="audio",
        allowed_extensions=ALLOWED_AUDIO_EXTENSIONS,
        content_type=audio.content_type,
    )
    audio_path = upload_path_for(transcription_job_id, safe_audio_filename)
    audio_size_bytes = await save_upload_with_size_limit(audio, audio_path)

    pdf_path: str | None = None
    if pdf is not None and pdf.filename:
        safe_pdf_filename = normalize_upload_filename(
            pdf.filename,
            default_stem="agenda",
            allowed_extensions=(".pdf",),
            content_type=pdf.content_type,
        )
        pdf_destination = UPLOAD_DIR / f"{pipeline_id}-{safe_pdf_filename}"
        await save_upload_with_size_limit(pdf, pdf_destination)
        pdf_path = str(pdf_destination)
        if source_extraction and durable.document(pdf_path)['sha256'] != source_extraction['document']['sha256']:
            raise HTTPException(409, 'Hochgeladenes PDF stimmt nicht mit dem Quelljob überein')

    now = time.time()
    with JOB_LOCK:
        jobs[transcription_job_id] = {
            "created_at": now,
            "updated_at": now,
            "session_id": effective_session_id,
            "status": JOB_STATUS_PENDING,
            "progress": 0,
            "message": "Audio hochgeladen, Pipeline wartet auf Verarbeitung",
            "file_path": str(audio_path),
            "audio_path": str(audio_path),
            "audio_filename": safe_audio_filename,
            "audio_content_type": audio.content_type,
            "audio_size_bytes": audio_size_bytes,
            "remember_speakers": remember_speakers,
            "transcript": None,
            "error": None,
            "cancellation_requested": False,
        }
    persist_job_state(transcription_job_id)
    save_pipeline_session(
        effective_session_id,
        job_id=transcription_job_id,
        tops=known_tops,
        skipped_assignment=skip_agenda_detection,
        current_step=0,
    )

    pipeline_job = save_pipeline_job(
        pipeline_id,
        {
            "session_id": effective_session_id,
            "transcription_job_id": transcription_job_id,
            "status": PIPELINE_STATUS_PENDING,
            "stage": PIPELINE_STAGE_UPLOAD,
            "progress": 5,
            "error": None,
            "created_at": now,
            "updated_at": now,
            "result_refs": {
                "audio_path": str(audio_path),
                "pdf_path": pdf_path,
                "known_tops": known_tops,
                "options": parsed_options,
                "remember_speakers": remember_speakers,
                "warnings": [],
            },
        },
    )

    manager = await get_or_create_pipeline_manager()
    await manager.enqueue(pipeline_id)
    return PipelineStartResponse(
        pipeline_id=pipeline_id,
        session_id=effective_session_id,
        transcription_job_id=transcription_job_id,
        status=pipeline_job["status"],
        stage=pipeline_job["stage"],
        progress=pipeline_job["progress"],
        warnings=[],
    )


@app.get("/api/pipeline/{pipeline_id}", response_model=PipelineStatusResponse)
async def get_pipeline_status(pipeline_id: str):
    job = load_pipeline_job(pipeline_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Pipeline nicht gefunden")
    return build_pipeline_status_response(job)


@app.post("/api/pipeline/{pipeline_id}/cancel", response_model=PipelineStatusResponse)
async def cancel_pipeline(pipeline_id: str):
    job = request_pipeline_cancellation(pipeline_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Pipeline nicht gefunden")
    return build_pipeline_status_response(job)


@app.get("/api/pipeline/{pipeline_id}/result", response_model=PipelineResultResponse)
async def get_pipeline_result(pipeline_id: str):
    pipeline_job = load_pipeline_job(pipeline_id)
    if pipeline_job is None:
        raise HTTPException(status_code=404, detail="Pipeline nicht gefunden")
    if (
        pipeline_job.get("status") not in {PIPELINE_STATUS_COMPLETED, PIPELINE_STATUS_FAILED}
        or pipeline_job.get("stage") != PIPELINE_STAGE_READY_FOR_REVIEW
    ):
        raise HTTPException(status_code=409, detail="Pipeline ist noch nicht reviewbar")

    session_id = pipeline_job.get("session_id")
    if not session_id:
        raise HTTPException(status_code=404, detail="Pipeline hat keine Session")
    session = load_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session nicht gefunden")

    session_response = build_session_response(session)
    observations = [
        build_speaker_observation_response(observation, session)
        for observation in load_speaker_observations(session_id=session_id)
    ]
    status = build_pipeline_status_response(pipeline_job)
    agenda_detection = build_pipeline_agenda_detection_response(
        _pipeline_refs(pipeline_job).get("agenda"),
        session_response,
    )
    return PipelineResultResponse(
        pipeline=status,
        session=session_response,
        job=session_response.job,
        speaker_observations=observations,
        summary_reviews=session_response.summary_reviews,
        warnings=status.warnings,
        agenda_detection=agenda_detection,
    )


@app.get("/api/sessions", response_model=SessionListResponse)
async def list_sessions_endpoint(
    limit: int = Query(default=25, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    query: Optional[str] = Query(default=None, max_length=200),
    status: Optional[str] = Query(default=None),
):
    """List the shared, server-side session history for all visitors."""
    allowed_statuses = {
        "draft",
        "processing",
        "review",
        "ready",
        "failed",
        "cancelled",
    }
    if status and status not in allowed_statuses:
        raise HTTPException(status_code=400, detail="Ungültiger Sitzungsstatus")
    items, total = list_sessions(
        query=query,
        status=status,
        limit=limit,
        offset=offset,
    )
    return SessionListResponse(
        items=[SessionListItem(**item) for item in items],
        total=total,
        limit=limit,
        offset=offset,
    )


def save_session_or_conflict(
    session_id: str,
    state: dict[str, Any],
    expected_revision: int | None,
) -> dict[str, Any]:
    try:
        return save_session(
            session_id,
            state,
            expected_revision=expected_revision,
        )
    except SessionConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "message": (
                    "Diese Sitzung wurde zwischenzeitlich geändert. "
                    "Bitte laden Sie den aktuellen Stand neu."
                ),
                "expected_revision": exc.expected_revision,
                "actual_revision": exc.actual_revision,
            },
        ) from exc


@app.post("/api/sessions", response_model=SessionResponse)
async def create_or_save_session(request: SessionSaveRequest):
    """
    Create or save a persisted editing session.

    This stores user-editable state only: TOPs, corrected transcript lines,
    line assignments, speaker display names, summaries and the linked
    transcription job. Audio bytes are not copied.
    """
    session_id = request.session_id or str(uuid.uuid4())
    state = model_to_dict(request)
    state["session_id"] = session_id
    existing = load_session(session_id)
    if "agenda_proposals" not in request.model_fields_set and existing:
        state["agenda_proposals"] = session_agenda_proposals(
            existing, load_latest_pipeline_job_for_session(session_id)
        )
    state = reconcile_session_summaries(existing, state)
    session = save_session_or_conflict(
        session_id,
        state,
        request.revision,
    )
    return build_session_response(session)


@app.put("/api/sessions/{session_id}", response_model=SessionResponse)
async def save_existing_session(session_id: str, request: SessionSaveRequest):
    """Save a persisted editing session under a known session ID."""
    state = model_to_dict(request)
    state["session_id"] = session_id
    existing = load_session(session_id)
    if "agenda_proposals" not in request.model_fields_set and existing:
        state["agenda_proposals"] = session_agenda_proposals(
            existing, load_latest_pipeline_job_for_session(session_id)
        )
    state = reconcile_session_summaries(existing, state)
    session = save_session_or_conflict(session_id, state, request.revision)
    return build_session_response(session)


@app.get("/api/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """Load a persisted editing session and its linked transcription job."""
    session = load_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session nicht gefunden")
    return build_session_response(session)


@app.post(
    "/api/sessions/{session_id}/summaries/regenerate",
    response_model=SessionResponse,
)
async def regenerate_session_summaries(
    session_id: str,
):
    """Deprecated synchronous all-TOP endpoint kept as an explicit guard."""
    raise HTTPException(
        status_code=410,
        detail=(
            "Die synchrone Gesamtregenerierung wurde entfernt. "
            "Bitte ausgewählte TOPs über /summary-jobs neu generieren."
        ),
    )


@app.post(
    "/api/sessions/{session_id}/summary-jobs",
    response_model=SummaryJobResponse,
)
async def create_summary_job(session_id: str, request: SummaryJobCreateRequest):
    session = load_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session nicht gefunden")
    if request.revision is not None and int(session.get("revision") or 1) != request.revision:
        raise HTTPException(status_code=409, detail="Sitzung wurde zwischenzeitlich geändert")
    active_pipeline = load_latest_pipeline_job_for_session(session_id)
    if active_pipeline and active_pipeline.get("status") in {
        PIPELINE_STATUS_PENDING,
        PIPELINE_STATUS_PROCESSING,
    }:
        raise HTTPException(
            status_code=409,
            detail=(
                "Die automatische Pipeline dieser Sitzung erzeugt die "
                "Zusammenfassungen bereits."
            ),
        )

    effective_ids = list(session.get("top_ids") or [])
    if not session.get("tops") or session.get("skipped_assignment"):
        effective_ids = [f"whole-session:{session_id}"]
    selected = list(dict.fromkeys(request.top_ids))
    if not selected:
        raise HTTPException(status_code=400, detail="Mindestens ein TOP muss ausgewählt sein")
    invalid = [top_id for top_id in selected if top_id not in effective_ids]
    if invalid:
        raise HTTPException(status_code=400, detail="Unbekannter TOP ausgewählt")
    latest_job = load_latest_summary_job_for_session(session_id)
    if latest_job and latest_job.get("status") in {"pending", "processing", "cancelling"}:
        raise HTTPException(
            status_code=409,
            detail="Für diese Sitzung läuft bereits eine selektive Neugenerierung",
        )

    input_hashes: dict[str, str] = {}
    previous_statuses: dict[str, str] = {}
    edit_fingerprints: dict[str, str] = {}
    states = dict(session.get("summary_states") or {})
    fallback_states = build_generated_summary_states(
        session_id=session_id,
        transcript=[line_to_dict(line) for line in (session.get("transcript") or [])],
        tops=list(session.get("tops") or []),
        top_ids=list(session.get("top_ids") or []),
        assignments=list(session.get("assignments") or []),
        summaries=dict(session.get("summaries") or {}),
        summary_reviews=dict(session.get("summary_reviews") or {}),
        origin="legacy",
    )
    for top_index, fallback_state in fallback_states.items():
        states.setdefault(top_index, fallback_state)
    summary_job_id = str(uuid.uuid4())
    for top_id in selected:
        top_index = effective_ids.index(top_id)
        _, input_hash = current_summary_input(session, top_index)
        input_hashes[top_id] = input_hash
        edit_fingerprints[top_id] = summary_edit_fingerprint(session, top_index)
        previous_statuses[top_id] = str(
            (states.get(top_index) or {}).get("status") or "missing"
        )
        states[top_index] = {
            **dict(states.get(top_index) or {}),
            "top_id": top_id,
            "status": "queued",
            "generation_job_id": summary_job_id,
            "current_input_hash": input_hash,
            "updated_at": time.time(),
        }
    session["summary_states"] = states
    saved = save_session_or_conflict(session_id, session, request.revision)

    now = time.time()
    job = save_summary_job(
        summary_job_id,
        {
            "session_id": session_id,
            "status": "pending",
            "progress": 0,
            "current_top": 0,
            "total_tops": len(selected),
            "created_at": now,
            "updated_at": now,
            "refs": {
                "top_ids": selected,
                "input_hashes": input_hashes,
                "edit_fingerprints": edit_fingerprints,
                "previous_statuses": previous_statuses,
                "model": request.model,
                "system_prompt": request.system_prompt,
                "session_revision": saved.get("revision"),
                "input_snapshot": saved,
            },
        },
    )
    manager = await get_or_create_summary_job_manager()
    await manager.enqueue(summary_job_id)
    return build_summary_job_response(job)


@app.get("/api/summary-jobs/{summary_job_id}", response_model=SummaryJobResponse)
async def get_summary_job(summary_job_id: str):
    job = load_summary_job(summary_job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Zusammenfassungsjob nicht gefunden")
    return build_summary_job_response(job)


@app.post("/api/summary-jobs/{summary_job_id}/cancel", response_model=SummaryJobResponse)
async def cancel_summary_job(summary_job_id: str):
    if durable.load(summary_job_id):
        cancelled = durable.cancel(summary_job_id)
        mirror_durable_job(cancelled)
    job = load_summary_job(summary_job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Zusammenfassungsjob nicht gefunden")
    if job.get("status") in {"completed", "failed", "cancelled"}:
        return build_summary_job_response(job)
    updated = update_summary_job(
        summary_job_id,
        status="cancelling",
        refs={"cancel_requested": True},
    )
    return build_summary_job_response(updated or job)


@app.post(
    "/api/sessions/{session_id}/summaries/{top_id}/accept",
    response_model=SessionResponse,
)
async def accept_existing_summary(
    session_id: str,
    top_id: str,
    request: SummaryAcceptRequest,
):
    session = load_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session nicht gefunden")
    effective_ids = list(session.get("top_ids") or [])
    no_top_mode = not session.get("tops") or bool(session.get("skipped_assignment"))
    if no_top_mode:
        effective_ids = [f"whole-session:{session_id}"]
    if top_id not in effective_ids:
        raise HTTPException(status_code=404, detail="TOP nicht gefunden")
    top_index = effective_ids.index(top_id)
    summary = str((session.get("summaries") or {}).get(top_index) or "").strip()
    if not summary:
        raise HTTPException(status_code=400, detail="Keine Zusammenfassung zum Übernehmen")

    snapshot, input_hash = current_summary_input(session, top_index)
    transcript = [line_to_dict(line) for line in (session.get("transcript") or [])]
    assignments = list(session.get("assignments") or [])
    lines = [transcript[i] for i in summary_line_indices(session, top_index)]
    review = dict((session.get("summary_reviews") or {}).get(top_index) or {})
    structured_data = review.get("structured")
    structured = None
    if isinstance(structured_data, dict):
        from summarize import StructuredSummary

        structured = StructuredSummary(**structured_data)
    speaker_names = dict(session.get("speaker_names") or {})
    named_lines = [
        {**line, "speaker": speaker_names.get(line.get("speaker", ""), line.get("speaker", ""))}
        for line in lines
    ]
    rebuilt = build_summary_review(structured=structured, summary=summary, lines=named_lines)
    reviews = dict(session.get("summary_reviews") or {})
    reviews[top_index] = {
        **review,
        "source_links": [link.to_dict() for link in rebuilt.source_links],
        "review_warnings": [warning.to_dict() for warning in rebuilt.warnings],
    }
    states = dict(session.get("summary_states") or {})
    states[top_index] = {
        **dict(states.get(top_index) or {}),
        "top_id": top_id,
        "status": "ready",
        "input_hash": input_hash,
        "current_input_hash": input_hash,
        "source_snapshot": snapshot,
        "change_reasons": [],
        "accepted_at": time.time(),
        "updated_at": time.time(),
    }
    session["summary_reviews"] = reviews
    session["summary_states"] = states
    saved = save_session_or_conflict(session_id, session, request.revision)
    return build_session_response(saved)


@app.get("/api/speaker-profiles", response_model=List[SpeakerProfileResponse])
async def list_speaker_profiles(
    scope: Optional[str] = None,
    include_archived: bool = False,
):
    """List global speaker profiles. Archived profiles are hidden by default."""
    return [
        build_speaker_profile_response(profile)
        for profile in load_speaker_profiles(
            scope=scope,
            include_archived=include_archived,
        )
    ]


@app.post("/api/speaker-profiles", response_model=SpeakerProfileResponse)
async def create_speaker_profile_endpoint(request: SpeakerProfileCreateRequest):
    """Create a speaker profile only after explicit user action."""
    profile = create_speaker_profile(
        validate_display_name(request.display_name),
        scope=request.scope,
    )
    return build_speaker_profile_response(profile)


@app.put("/api/speaker-profiles/{profile_id}", response_model=SpeakerProfileResponse)
async def update_speaker_profile_endpoint(
    profile_id: str,
    request: SpeakerProfileUpdateRequest,
):
    """Rename or rescope an active speaker profile."""
    existing = get_required_active_profile(profile_id)
    fields_set = (
        request.model_fields_set
        if hasattr(request, "model_fields_set")
        else request.__fields_set__
    )
    display_name = (
        validate_display_name(request.display_name)
        if request.display_name is not None
        else existing["display_name"]
    )
    scope = request.scope if "scope" in fields_set else existing.get("scope")
    updated = update_speaker_profile(
        profile_id,
        display_name=display_name,
        scope=scope,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Profil nicht gefunden")
    if updated.get("archived_at") is not None:
        raise HTTPException(status_code=409, detail="Profil ist archiviert")
    return build_speaker_profile_response(updated)


@app.delete("/api/speaker-profiles/{profile_id}", response_model=SpeakerProfileResponse)
async def archive_speaker_profile_endpoint(profile_id: str):
    """Archive a speaker profile instead of hard-deleting it."""
    get_required_active_profile(profile_id)
    archived = archive_speaker_profile(profile_id)
    if archived is None:
        raise HTTPException(status_code=404, detail="Profil nicht gefunden")
    anonymize_speaker_observations_for_profile(profile_id)
    return build_speaker_profile_response(archived)


@app.delete("/api/speaker-profiles/{profile_id}/embeddings")
async def delete_speaker_profile_embeddings_endpoint(profile_id: str):
    """Delete persisted biometric embeddings for a speaker profile."""
    profile = load_speaker_profile(profile_id, include_archived=True)
    if profile is None:
        raise HTTPException(status_code=404, detail="Profil nicht gefunden")
    deleted_count = delete_speaker_embeddings(profile_id)
    return {"profile_id": profile_id, "deleted_count": deleted_count}


@app.post(
    "/api/speaker-embeddings/backfill",
    response_model=SpeakerEmbeddingBackfillResponse,
)
def backfill_speaker_embeddings_endpoint(
    profile_id: Optional[str] = None,
    session_id: Optional[str] = None,
):
    """Run GPU backfill in the thread pool so waiting never blocks the event loop."""
    if profile_id is not None:
        get_required_active_profile(profile_id)
    if session_id is not None:
        get_required_session(session_id)
    return backfill_speaker_profile_embeddings(
        profile_id=profile_id,
        session_id=session_id,
    )


@app.get(
    "/api/sessions/{session_id}/speaker-observations",
    response_model=List[SpeakerObservationResponse],
)
async def list_session_speaker_observations(session_id: str):
    """Return reviewable speaker observations for a session."""
    session = get_required_session(session_id)
    observations = load_speaker_observations(session_id=session_id)
    return [
        build_speaker_observation_response(observation, session)
        for observation in observations
    ]


@app.post(
    "/api/sessions/{session_id}/speaker-observations/{observation_id}/confirm",
    response_model=SpeakerObservationResponse,
)
async def confirm_session_speaker_observation(
    session_id: str,
    observation_id: int,
    request: SpeakerObservationConfirmRequest | None = None,
):
    """Confirm a suggested speaker-profile match after user review."""
    session = get_required_session(session_id)
    observation = get_observation_for_session(session_id, observation_id)
    if observation.get("status") == "rejected":
        raise HTTPException(
            status_code=409,
            detail="Abgelehnte Observation kann nicht bestätigt werden",
        )
    ensure_local_speaker_exists(session, observation["local_speaker_id"])
    ensure_no_accepted_local_mapping(
        session_id,
        observation["local_speaker_id"],
        exclude_observation_id=observation_id,
    )

    requested_profile_id = request.profile_id if request else None
    profile_id = requested_profile_id or observation.get("profile_id")
    if profile_id is None:
        raise HTTPException(status_code=400, detail="Profil fehlt für Bestätigung")
    profile = get_required_active_profile(profile_id)

    confirmed = confirm_speaker_observation(
        observation_id,
        profile_id=profile_id,
        confidence=request.confidence if request else None,
    )
    if confirmed is None:
        raise HTTPException(status_code=404, detail="Observation nicht gefunden")
    updated_session = apply_profile_display_name_to_session(
        session,
        observation["local_speaker_id"],
        profile,
    )
    embedding_result = add_job_embedding_to_profile(
        job_id=observation["job_id"],
        local_speaker_id=observation["local_speaker_id"],
        profile_id=profile_id,
        observation_id=observation_id,
        storage_reason="confirm",
    )
    refresh_speaker_suggestions_for_session(
        job_id=observation["job_id"],
        session_id=session_id,
    )
    return build_speaker_observation_response(
        confirmed,
        updated_session,
        embedding_warning=speaker_embedding_storage_warning(embedding_result),
    )


@app.post(
    "/api/sessions/{session_id}/speaker-observations/{observation_id}/reject",
    response_model=SpeakerObservationResponse,
)
async def reject_session_speaker_observation(session_id: str, observation_id: int):
    """Reject a suggested speaker-profile match after user review."""
    session = get_required_session(session_id)
    observation = get_observation_for_session(session_id, observation_id)
    if observation.get("status") in {"confirmed", "manual"}:
        raise HTTPException(
            status_code=409,
            detail="Bestätigte Zuordnung kann nicht abgelehnt werden",
        )
    rejected = reject_speaker_observation(observation_id)
    if rejected is None:
        raise HTTPException(status_code=404, detail="Observation nicht gefunden")
    return build_speaker_observation_response(rejected, session)


@app.post(
    "/api/sessions/{session_id}/speaker-observations/{observation_id}/unassign",
    response_model=SpeakerObservationResponse,
)
async def unassign_session_speaker_observation(session_id: str, observation_id: int):
    """Undo an accepted persistent speaker mapping so it can be corrected."""
    session = get_required_session(session_id)
    observation = get_observation_for_session(session_id, observation_id)
    if observation.get("status") not in {"confirmed", "manual"}:
        raise HTTPException(
            status_code=409,
            detail="Nur bestätigte Zuordnungen können gelöst werden",
        )

    profile_id = observation.get("profile_id")
    if profile_id:
        delete_speaker_embeddings_for_source(
            profile_id,
            job_id=observation["job_id"],
            local_speaker_id=observation["local_speaker_id"],
            observation_id=observation_id,
        )

    rejected = reject_speaker_observation(observation_id)
    if rejected is None:
        raise HTTPException(status_code=404, detail="Observation nicht gefunden")
    refresh_speaker_suggestions_for_session(
        job_id=observation["job_id"],
        session_id=session_id,
    )
    return build_speaker_observation_response(rejected, session)


@app.post(
    "/api/sessions/{session_id}/speaker-observations/manual",
    response_model=SpeakerObservationResponse,
)
async def create_manual_speaker_observation(
    session_id: str,
    request: SpeakerObservationManualRequest,
):
    """Manually link a local speaker to an active or newly created profile."""
    session = get_required_session(session_id)
    local_speaker_id = request.local_speaker_id.strip()
    ensure_local_speaker_exists(session, local_speaker_id)
    ensure_no_accepted_local_mapping(session_id, local_speaker_id)

    if bool(request.profile_id) == bool(request.display_name):
        raise HTTPException(
            status_code=400,
            detail="Genau eines von profile_id oder display_name ist erforderlich",
        )

    profile = (
        get_required_active_profile(request.profile_id)
        if request.profile_id
        else create_speaker_profile(
            validate_display_name(request.display_name or ""),
            scope=request.scope,
        )
    )

    observation_id = request.observation_id
    if observation_id is not None:
        existing = get_observation_for_session(session_id, observation_id)
        if existing["local_speaker_id"] != local_speaker_id:
            raise HTTPException(
                status_code=409,
                detail="Observation gehört zu einem anderen lokalen Sprecher",
            )
        job_id = existing["job_id"]
    else:
        job_id = session.get("job_id")
        if not job_id or load_job(job_id) is None:
            raise HTTPException(
                status_code=409,
                detail="Session hat keinen gültigen Transkriptionsjob",
            )
        for existing in load_speaker_observations(session_id=session_id):
            if (
                existing["local_speaker_id"] == local_speaker_id
                and existing.get("profile_id") == profile["profile_id"]
                and existing.get("status") == "suggested"
            ):
                observation_id = existing["observation_id"]
                job_id = existing["job_id"]
                break

    manual = save_speaker_observation(
        job_id=job_id,
        session_id=session_id,
        local_speaker_id=local_speaker_id,
        profile_id=profile["profile_id"],
        confidence=request.confidence,
        status="manual",
        observation_id=observation_id,
    )
    updated_session = apply_profile_display_name_to_session(
        session,
        local_speaker_id,
        profile,
    )
    embedding_result = add_job_embedding_to_profile(
        job_id=job_id,
        local_speaker_id=local_speaker_id,
        profile_id=profile["profile_id"],
        observation_id=manual["observation_id"],
        storage_reason="manual",
    )
    refresh_speaker_suggestions_for_session(
        job_id=job_id,
        session_id=session_id,
    )
    return build_speaker_observation_response(
        manual,
        updated_session,
        embedding_warning=speaker_embedding_storage_warning(embedding_result),
    )


@app.get(
    "/api/sessions/{session_id}/speaker-match-diagnostics",
    response_model=List[SpeakerMatchDiagnosticResponse],
)
async def list_session_speaker_match_diagnostics(session_id: str):
    """Explain why local speakers did not receive automatic profile suggestions."""
    session = get_required_session(session_id)
    return build_speaker_match_diagnostic_responses(session)


@app.post("/api/export")
async def export_protocol_endpoint(request: ProtocolExportRequest):
    """Render the completed protocol as TXT, DOCX or PDF."""
    if request.session_id:
        pipeline = load_latest_pipeline_job_for_session(request.session_id)
        if pipeline and (pipeline['status'] != 'completed' or _pipeline_refs(pipeline).get('processing_complete') is not True):
            raise HTTPException(409, 'Pipeline technisch unvollständig; erhaltene Ergebnisse sind ein prüfbarer Entwurf')
    if any((review.get('llm_usage') or {}).get('processing_complete') is False
           for review in request.summary_reviews.values() if isinstance(review, dict)):
        raise HTTPException(409, 'Technisch unvollständige Zusammenfassungen sind nicht exportierbar')
    export_format = request.format.lower().strip()
    if export_format not in {"txt", "docx", "pdf"}:
        raise HTTPException(status_code=400, detail="Exportformat nicht unterstützt")
    if not request.tops:
        raise HTTPException(status_code=400, detail="Keine TOPs vorhanden")

    filtered_tops = [top.strip() for top in request.tops if top.strip()]
    if not filtered_tops:
        raise HTTPException(status_code=400, detail="Keine TOPs vorhanden")

    metadata = ProtocolMetadata(
        committee=request.metadata.committee.strip(),
        date=request.metadata.date.strip(),
        location=request.metadata.location.strip(),
        title=request.metadata.title.strip() or "Sitzungsprotokoll",
        participants=[
            participant.strip()
            for participant in request.metadata.participants
            if participant.strip()
        ],
    )
    appendix = ProtocolAppendix(
        include_speaker_list=request.appendix.include_speaker_list,
        include_transcript=(
            request.appendix.include_transcript
            if request.appendix.include_transcript is not None
            else request.appendix.include_transcript_excerpt
        ),
        group_transcript_by_top=request.appendix.group_transcript_by_top,
        include_generation_note=request.appendix.include_generation_note,
    )
    export_transcript = [
        ExportTranscriptLine(
            speaker=request.speaker_names.get(line.speaker, line.speaker),
            text=line.text,
            start=line.start,
            end=line.end,
        )
        for line in request.transcript
    ]
    document = build_protocol_document(
        metadata=metadata,
        tops=filtered_tops,
        summaries=request.summaries,
        summary_reviews=request.summary_reviews,
        transcript=export_transcript,
        assignments=request.assignments,
        speaker_names=request.speaker_names,
        appendix=appendix,
    )
    content = render_protocol(document, export_format)  # type: ignore[arg-type]
    media_types = {
        "txt": "text/plain; charset=utf-8",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "pdf": "application/pdf",
    }
    title_stem = normalize_upload_filename(
        metadata.title or "protokoll",
        default_stem="protokoll",
        allowed_extensions=(),
    ).removesuffix(".")
    filename = f"{title_stem or 'protokoll'}.{export_format}"
    return Response(
        content=content,
        media_type=media_types[export_format],
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/transcribe", response_model=TranscriptionJob)
async def start_transcription(
    audio: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    remember_speakers: bool = Form(False),
):
    """
    Upload audio file and start transcription job.
    Returns job_id to poll for status.
    """
    logger.info("Received transcription request (%s)", audio.content_type)

    # Check if models are loaded
    if (
        not getattr(app.state, "models_loaded", False)
        or getattr(app.state, "models", None) is None
    ):
        logger.error("Transcription request rejected - models not loaded")
        raise HTTPException(
            status_code=503,
            detail="Server startet noch - Modelle werden geladen. Bitte warten.",
        )

    # Validate file type
    if not is_allowed_audio_file(audio.filename, audio.content_type):
        logger.warning(f"Rejected file with invalid type: {audio.content_type}")
        raise HTTPException(
            status_code=400, detail=f"Ungültiger Dateityp. Erlaubt: MP3, WAV, M4A"
        )

    job_id = str(uuid.uuid4())
    logger.info(f"Created job: {job_id}")

    safe_filename = normalize_upload_filename(
        audio.filename,
        default_stem="audio",
        allowed_extensions=ALLOWED_AUDIO_EXTENSIONS,
        content_type=audio.content_type,
    )
    file_path = upload_path_for(job_id, safe_filename)
    size_bytes = await save_upload_with_size_limit(audio, file_path)
    logger.info("Saved uploaded audio for job %s (%s bytes)", job_id, size_bytes)

    with JOB_LOCK:
        jobs[job_id] = {
            "created_at": time.time(),
            "updated_at": time.time(),
            "session_id": session_id,
            "status": JOB_STATUS_PENDING,
            "progress": 0,
            "message": "Audio hochgeladen, Job wartet auf Verarbeitung",
            "file_path": str(file_path),
            "audio_path": str(file_path),
            "audio_filename": safe_filename,
            "audio_content_type": audio.content_type,
            "audio_size_bytes": size_bytes,
            "remember_speakers": remember_speakers,
            "transcript": None,
            "error": None,
            "cancellation_requested": False,
        }
    persist_job_state(job_id)

    # Cleanup old jobs to prevent memory buildup
    cleanup_old_jobs()

    manager = await get_or_create_job_manager()
    await manager.enqueue(job_id)
    logger.info(f"Queued transcription job: {job_id}")

    return TranscriptionJob(
        job_id=job_id,
        status=JOB_STATUS_PENDING,
        progress=0,
        message="Transkription gestartet",
    )


@app.get("/api/transcribe/{job_id}", response_model=TranscriptionJob)
async def get_transcription_status(job_id: str):
    """
    Get status of transcription job.
    """
    job = get_job_from_cache_or_db(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job nicht gefunden")

    return build_transcription_job_response(job_id, job)


@app.post("/api/transcribe/{job_id}/cancel", response_model=TranscriptionJob)
async def cancel_transcription(job_id: str):
    """Cancel a pending job or request cooperative cancellation for an active job."""
    job = request_job_cancellation(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job nicht gefunden")
    return build_transcription_job_response(job_id, job)


@app.get("/api/audio/{job_id}")
async def stream_audio(
    job_id: str,
    range: Optional[str] = Header(None, alias="Range"),
):
    """
    Stream audio file for a transcription job.
    Supports HTTP Range requests for efficient seeking.
    """
    job = get_job_from_cache_or_db(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job nicht gefunden")

    audio_path = get_audio_path_for_job(job)

    if not audio_path or not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio nicht mehr verfügbar")

    file_size = os.path.getsize(audio_path)

    # Determine content type
    content_type, _ = mimetypes.guess_type(audio_path)
    if not content_type:
        content_type = "audio/mpeg"

    # Handle Range requests for seeking
    if range:
        # Parse range header: "bytes=start-end"
        range_match = re.match(r"bytes=(\d+)-(\d*)", range)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2)) if range_match.group(2) else file_size - 1

            if start >= file_size:
                raise HTTPException(status_code=416, detail="Range Not Satisfiable")

            chunk_size = end - start + 1

            with open(audio_path, "rb") as f:
                f.seek(start)
                data = f.read(chunk_size)

            return Response(
                content=data,
                status_code=206,
                headers={
                    "Content-Range": f"bytes {start}-{end}/{file_size}",
                    "Accept-Ranges": "bytes",
                    "Content-Length": str(chunk_size),
                    "Content-Type": content_type,
                },
            )

    # Return full file if no range requested
    with open(audio_path, "rb") as f:
        data = f.read()

    return Response(
        content=data,
        status_code=200,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
            "Content-Type": content_type,
        },
    )


@app.post("/api/summarize", response_model=SummarizeResponse)
async def generate_summary(request: SummarizeRequest):
    """
    Generate summary for a TOP segment.
    """
    if not request.lines:
        raise HTTPException(status_code=400, detail="Keine Zeilen zum Zusammenfassen")

    # Combine lines into text
    text = "\n".join([f"{line.speaker}: {line.text}" for line in request.lines])

    try:
        loop = asyncio.get_running_loop()

        def run_guarded_summary():
            with work_slot(LLM_WORK_LOCK):
                return summarize_segment(
                    request.top_title,
                    text,
                    source_lines=[f"{line.speaker}: {line.text}" for line in request.lines],
                    model=request.model,
                    system_prompt=request.system_prompt,
                )

        result = await loop.run_in_executor(None, run_guarded_summary)
        review = build_summary_review(
            structured=result.structured,
            summary=result.summary,
            lines=request.lines,
        )
        return SummarizeResponse(
            summary=result.summary,
            duration_seconds=result.duration_seconds,
            structured=(
                StructuredSummaryResponse(**result.structured.to_dict())
                if result.structured
                else None
            ),
            source_links=[
                SummarySourceLinkResponse(**link.to_dict())
                for link in review.source_links
            ],
            review_warnings=[
                SummaryReviewWarningResponse(**warning.to_dict())
                for warning in review.warnings
            ],
            fallback_used=result.fallback_used,
            chunks_processed=result.chunks_processed,
            llm_usage=getattr(result, "llm_usage", {}),
        )
    except LLMCallError as e:
        status_code = 504 if e.category == "timeout" else 503 if e.transient else 500
        raise HTTPException(
            status_code=status_code,
            detail=f"Fehler bei der Zusammenfassung ({e.category}): {str(e)}",
        )
    except StructuredOutputError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Fehler bei der strukturierten Zusammenfassung: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Fehler bei der Zusammenfassung: {str(e)}"
        )


@app.post("/api/extract-tops", response_model=ExtractTOPsResponse)
@app.post("/api/extract-tops/jobs", status_code=202)
async def extract_tops_endpoint(
    request: Request,
    pdf: UploadFile = File(...),
    model: Optional[str] = Form(None),
    system_prompt: Optional[str] = Form(None),
):
    """
    Extract TOPs (agenda items) from a German municipal meeting invitation PDF.
    Uses LLM to intelligently parse the document structure.
    """
    respond_async = request.url.path.endswith("/jobs") or request.headers.get("prefer") == "respond-async"
    logger.info("Received PDF for TOP extraction (%s)", pdf.content_type)

    # Validate file type
    if not is_allowed_pdf_file(pdf.filename, pdf.content_type):
        logger.warning(f"Rejected non-PDF file: {pdf.content_type}")
        raise HTTPException(status_code=400, detail="Nur PDF-Dateien sind erlaubt")

    # Save uploaded file temporarily
    file_id = str(uuid.uuid4())
    safe_filename = normalize_upload_filename(
        pdf.filename,
        default_stem="document",
        allowed_extensions=(".pdf",),
        content_type=pdf.content_type,
    )
    file_path = upload_path_for(file_id, safe_filename)

    try:
        size_bytes = await save_upload_with_size_limit(pdf, file_path)
        logger.info("Saved uploaded PDF for TOP extraction (%s bytes)", size_bytes)

        job = durable.submit("pdf", {"path": str(file_path.resolve()), "model": model,
            "system_prompt": system_prompt}, documents=[durable.document(file_path)])
        if respond_async:
            return Response(content=json.dumps(durable.public(job)), status_code=202,
                media_type="application/json", headers={"Location": f"/api/model-jobs/{job['job_id']}"})
        result = await await_durable_result(job["job_id"])
        return ExtractTOPsResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "TOP extraction failed (%s)",
            safe_exception_label(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail="PDF-Auswertung fehlgeschlagen; Job und Quellen bleiben zur Prüfung gespeichert"
        )



@app.post("/api/assignment-suggestions", response_model=AssignmentSuggestionsResponse)
async def assignment_suggestions_endpoint(request: AssignmentSuggestionsRequest):
    """Legacy shape, same durable model-only workflow as agenda detection."""
    if not request.transcript or not request.tops:
        raise HTTPException(400, 'Transkript und TOPs erforderlich')
    job = await start_agenda_job(AgendaDetectionRequest(
        transcript=request.transcript, tops=request.tops, use_llm=True,
        preserve_transcript_structure=True))
    result = await await_durable_result(job['job_id'])
    return AssignmentSuggestionsResponse(
        suggested_assignments=result['assignments'], segments=result['segments'],
        llm=result.get('llm'), warnings=result.get('warnings', []),
        strategy=result['strategy'], uncertain_count=result['uncertain_count'])


def calculate_agenda(request: AgendaDetectionRequest):
    """
    Detect reviewable TOPs and transcript segments.

    This synchronous endpoint runs in FastAPI's thread pool: model requests and
    CPU-bound detection must not block autosaves, health checks or cancellation.

    Known and unknown agendas use the same model-only reconstruction and
    independent complete review. Without TOPs, models first establish the agenda.
    """
    if not request.transcript:
        raise HTTPException(status_code=400, detail="Kein Transkript vorhanden")

    input_transcript = [line_to_dict(line) for line in request.transcript]
    split_transcript = durable.checkpoint('agenda:source:v1', lambda: (
        input_transcript if request.preserve_transcript_structure
        else split_transcript_for_agenda_detection(input_transcript)
    ))
    transcript = transcript_utterances(split_transcript)
    valid_tops = list(request.tops)
    if any(not top.strip() for top in valid_tops):
        raise HTTPException(400, "Leere TOP-Titel sind nicht zulässig")
    if request.top_ids and (len(request.top_ids) != len(valid_tops) or
                           len(set(request.top_ids)) != len(valid_tops) or not all(request.top_ids)):
        raise HTTPException(status_code=400, detail='TOP-IDs müssen vollständig und eindeutig sein')
    if valid_tops:
        result = segment_known_agenda(
            transcript,
            valid_tops,
            model=request.model,
            system_prompt=request.system_prompt,
            use_llm=request.use_llm,
            cache_namespace=str(uuid.uuid4()) if request.fresh else request.cache_namespace,
            top_ids=request.top_ids,
        )
    else:
        result = detect_agenda_from_transcript(
            transcript,
            model=request.model,
            system_prompt=request.system_prompt,
            use_llm=request.use_llm,
            cache_namespace=str(uuid.uuid4()) if request.fresh else request.cache_namespace,
        )

    if result.llm and request.top_ids:
        for identity in result.llm.provenance.get('identities', []):
            identity['top_uid'] = (request.top_ids[identity['top_index']] if identity['top_index'] < len(request.top_ids)
                                   else identity['top_id'])
    return AgendaDetectionResponse(
        llm=asdict(result.llm) if result.llm else None,
        warnings=result.llm.warnings if result.llm else [],
        tops=result.tops,
        transcript=[TranscriptLine(**line) for line in split_transcript],
        assignments=result.assignments,
        segments=[
            AssignmentSuggestionSegmentResponse(
                top_index=segment.top_index,
                top_title=segment.top_title,
                start_index=segment.start_index,
                end_index=segment.end_index,
                confidence=segment.confidence,
                uncertain=segment.uncertain,
                transition_type=segment.transition_type,
                reason=segment.reason,
                evidence_index=segment.evidence_index,
                evidence_text=segment.evidence_text,
            )
            for segment in result.segments
        ],
        uncertain_count=result.uncertain_count,
        strategy=result.strategy,
    )


# ----- Transcription Worker -----


def run_transcription(
    job_id: str,
    file_path: str | None,
    models: TranscriptionModels,
) -> None:
    """
    Run transcription in a managed worker using pre-loaded models.

    Transcription itself is not retried automatically: long GPU jobs can be
    expensive and failure modes are often input/model related. Cancellation is
    cooperative via the progress callback and checked again before persisting
    a completed result.
    """
    logger.info(f"[Job {job_id}] Worker task started")
    try:
        if get_job_from_cache_or_db(job_id) is None:
            persisted_job = load_job(job_id)
            if persisted_job:
                with JOB_LOCK:
                    jobs[job_id] = persisted_job
        job = get_job_from_cache_or_db(job_id)
        if job is None:
            raise RuntimeError("Job nicht gefunden")
        if is_job_cancelled(job_id):
            raise CancellationRequested()
        if not file_path:
            raise RuntimeError("Upload-Datei nicht gefunden")
        if not os.path.exists(file_path):
            raise RuntimeError("Upload-Datei ist nicht mehr verfügbar")

        update_job_state(
            job_id,
            status=JOB_STATUS_PROCESSING,
            progress=10,
            message="Transkription wird vorbereitet...",
        )
        logger.info(f"[Job {job_id}] Status: processing, preparing transcription...")

        # Run transcription with pre-loaded models
        def progress_callback(progress: int, message: str):
            if is_job_cancelled(job_id):
                raise CancellationRequested()
            update_job_state(
                job_id,
                progress=progress,
                message=message,
            )
            logger.info(f"[Job {job_id}] Progress: {progress}% - {message}")

        result = transcribe_audio(file_path, models, progress_callback)

        if is_job_cancelled(job_id):
            raise CancellationRequested()

        transcript = result.transcript
        transcript_line_count = len(transcript)
        speaker_embeddings = getattr(result, "speaker_embeddings", None) or []

        try:
            persist_job_speaker_embeddings(job_id, speaker_embeddings)
            create_speaker_suggestion_observations(
                job_id=job_id,
                session_id=job.get("session_id"),
                local_embeddings=speaker_embeddings,
                speaker_memory_opt_in=bool(job.get("remember_speakers")),
            )
        except Exception as e:
            logger.warning(
                "[Job %s] Speaker embedding matching failed without failing transcription: %s",
                job_id,
                e,
                exc_info=True,
            )

        update_job_state(
            job_id,
            status=JOB_STATUS_COMPLETED,
            progress=100,
            message="Transkription abgeschlossen",
            transcript=transcript,
            audio_path=file_path,
            error=None,
        )

        logger.info(
            f"[Job {job_id}] Transcription completed successfully with {transcript_line_count} lines"
        )

    except LLMCancelledError:
        if durable.CURRENT.get():
            raise
        logger.info(f"[Job {job_id}] Transcription cancelled")
        job = update_job_state(
            job_id,
            status=JOB_STATUS_CANCELLED,
            message="Transkription abgebrochen",
            error=None,
            cancellation_requested=True,
        )
        if DELETE_UPLOADS_ON_CANCEL_OR_FAILURE and job and durable.CURRENT.get() is None:
            cleanup_job_uploads(job_id, job)
    except Exception as e:
        if durable.CURRENT.get():
            raise
        logger.error(
            "[Job %s] Transcription failed (%s)",
            job_id,
            safe_exception_label(e),
            exc_info=True,
        )
        job = update_job_state(
            job_id,
            status=JOB_STATUS_FAILED,
            error=str(e),
            message=f"Fehler: {str(e)}",
        )
        if DELETE_UPLOADS_ON_CANCEL_OR_FAILURE and job and durable.CURRENT.get() is None:
            cleanup_job_uploads(job_id, job)
    finally:
        # Clean up GPU memory after every terminal worker run.
        try:
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info(f"[Job {job_id}] GPU memory cleared")
        except Exception:
            pass




class DurableSubmission:
    def __init__(self, kind):
        self.kind = kind

    async def enqueue(self, job_id):
        submit_legacy_job(self.kind, job_id)


def submit_legacy_job(kind, job_id):
    if durable.load(job_id):
        return
    loaders = {"pipeline": load_pipeline_job, "summary": load_summary_job, "transcription": load_job}
    old = loaders[kind](job_id)
    if old is None:
        return
    session = load_session(old.get("session_id")) if old.get("session_id") else None
    refs = _pipeline_refs(old) if kind == "pipeline" else old.get("refs") or {}
    pdf = refs.get("pdf_path")
    audio = refs.get("audio_path") or old.get("file_path")
    documents = [durable.document(path) for path in (pdf, audio) if path and Path(path).is_file()]
    source_id = (((refs.get('options') or {}).get('pdf_source_extraction') or {}).get('document') or {}).get('job_id')
    if source_id:
        source_job = durable.load(source_id)
        if source_job:
            documents.extend(doc for doc in source_job['documents'] if doc['path'] not in {d['path'] for d in documents})
    durable.submit(kind, {"legacy_snapshot": old, "session_snapshot": session,
        "session_revision": session.get("revision") if session else None}, job_id,
        documents=documents)


def recover_legacy_jobs():
    with durable.persistence.connect() as db:
        for kind, table, key in [("pipeline", "pipeline_jobs", "pipeline_job_id"),
                                  ("summary", "summary_jobs", "summary_job_id"),
                                  ("transcription", "transcription_jobs", "job_id")]:
            rows = db.execute(f"SELECT {key} FROM {table} WHERE status IN ('pending','processing','cancelling')").fetchall()
            for row in rows:
                if kind == "transcription" and db.execute(
                    "SELECT 1 FROM pipeline_jobs WHERE transcription_job_id=? AND status IN ('pending','processing')", (row[0],)).fetchone():
                    continue
                submit_legacy_job(kind, row[0])


def mirror_durable_job(job):
    if not job:
        return
    state = job["state"]
    status = {"queued": "pending", "running": "processing", "retry_wait": "pending",
              "review_required": "completed", "superseded": "failed"}.get(state, state)
    if state == 'review_required' and job.get('result') is None:
        status = 'failed'
    if job['kind'] == 'pipeline' and state == 'review_required' and (job.get('result') or {}).get('processing_complete') is False:
        status = 'failed'
    if job['kind'] == 'pipeline':
        # Technical failures never turn into a successful legacy completion.
        old = save_pipeline_state(job['job_id'], status=status, error=job.get('error'),
            result_refs={"execution_state": state})
        transcription_id = (old or {}).get('transcription_job_id')
        if transcription_id and state in {'failed', 'cancelled'}:
            transcription = load_job(transcription_id)
            if transcription and transcription['status'] in {'pending', 'processing'}:
                update_job_state(transcription_id, status=status, error=job.get('error'))
    elif job['kind'] == 'summary':
        old = load_summary_job(job['job_id'])
        if old is None or (old['status'] == status and old.get('refs', {}).get('execution_state') == state):
            return
        if state == 'cancelled':
            finalize_summary_job_cancellation(job['job_id'], old)
        update_summary_job(job['job_id'], status=status, error=job.get('error'), refs={"execution_state": state})
    elif job['kind'] == 'transcription':
        update_job_state(job['job_id'], status=status, error=job.get('error'))


def run_durable_job(job):
    payload = job['payload']
    for doc in job.get('documents') or []:
        if durable.document(doc['path'])['sha256'] != doc['sha256']:
            raise ValueError('Source PDF changed')
    if job['kind'] == 'pdf':
        def extract():
            value = extract_agenda_data_from_pdf(payload['path'], model=payload.get('model'),
                                                system_prompt=payload.get('system_prompt'))
            return value.to_dict()
        result = durable.checkpoint('pdf:validated:page-evidence-v3', extract)
        state = 'review_required' if result['review_required'] or not result['tops'] else 'completed'
        return result, state if result['processing_complete'] or result.get('review_questions') else 'failed'
    if job['kind'] == 'agenda':
        result = durable.checkpoint('agenda:validated', lambda: calculate_agenda(
            AgendaDetectionRequest(**payload['request'])).model_dump())
        state = (result.get('llm') or {}).get('status')
        if state in {'disabled', 'failed', 'partial_failure', 'fallback', 'partial_fallback'}:
            return result, 'failed'
        return result, 'review_required' if (result.get('llm') or {}).get('review_required', True) or result.get('uncertain_count') or None in result.get('assignments', []) else 'completed'
    if job['kind'] == 'pipeline':
        run_pipeline_job(job['job_id'], app.state.models)
        old = load_pipeline_job(job['job_id'])
        refs = _pipeline_refs(old)
        state = old['status']
        if state == 'completed':
            session = load_session(old['session_id']) or {}
            needs_review = refs.get('warnings') or refs.get('unassigned_line_count') or any(
                (review or {}).get('review_warnings') for review in (session.get('summary_reviews') or {}).values())
            state = 'review_required' if refs.get('processing_complete') and needs_review else 'completed'
            if refs.get('processing_complete') is False:
                state = 'failed'
        return {'session_id': old['session_id']}, state
    if job['kind'] == 'summary':
        run_summary_job(job['job_id'])
        old = load_summary_job(job['job_id'])
        state = old['status']
        session = load_session(old['session_id'])
        if state == 'completed' and any((review or {}).get('review_warnings') for review in (session.get('summary_reviews') or {}).values()):
            state = 'review_required'
        return old.get('refs', {}).get('outcomes'), state
    run_transcription(job['job_id'], payload['legacy_snapshot'].get('file_path'), app.state.models)
    return None, load_job(job['job_id'])['status']


async def await_durable_result(job_id):
    manager = getattr(app.state, "durable_manager", None)
    if manager is None or not manager.started:
        raise HTTPException(503, "Job ist gespeichert; Worker ist noch nicht gestartet")
    while True:
        job = durable.load(job_id)
        if job['state'] in {'completed', 'review_required'} or (job['state'] == 'failed' and job['result'] is not None):
            if job['result'] is None:
                raise HTTPException(409, job.get('error') or 'Job benötigt Prüfung')
            return job['result']
        if job['state'] in durable.TERMINAL:
            raise HTTPException(409 if job['state'] in {'cancelled', 'superseded'} else 500,
                                job.get('error') or 'Verarbeitung unvollständig')
        await asyncio.sleep(0.5)


@app.get('/api/model-jobs/{job_id}')
async def model_job_status(job_id: str):
    job = durable.load(job_id)
    if job is None:
        raise HTTPException(404, 'Job nicht gefunden')
    return durable.public(job)


@app.post('/api/model-jobs/{job_id}/cancel')
async def cancel_model_job(job_id: str):
    job = durable.cancel(job_id)
    if job is None:
        raise HTTPException(404, 'Job nicht gefunden')
    mirror_durable_job(job)
    return durable.public(job)


@app.post('/api/agenda-detection/jobs', status_code=202)
async def start_agenda_job(request: AgendaDetectionRequest):
    if not request.transcript:
        raise HTTPException(400, 'Kein Transkript vorhanden')
    data = request.model_dump()
    data['transcript'] = [line_to_dict(line) for line in request.transcript]
    from agenda_context import source_rows, model_agenda
    try:
        source_rows(transcript_utterances(data['transcript']))
        model_agenda(request.tops, request.top_ids)
        if any(not title.strip() for title in request.tops):
            raise ValueError('invalid_agenda_title')
    except (ValueError, TypeError):
        raise HTTPException(400, 'Ungültige Quellenidentitäten, TOP-Titel oder Audiozeiten')

    if request.fresh:
        data['cache_namespace'] = str(uuid.uuid4())
        data['fresh'] = False
    return durable.public(durable.submit('agenda', {'request': data}))


@app.post('/api/agenda-detection', response_model=AgendaDetectionResponse)
async def agenda_detection_endpoint(request: AgendaDetectionRequest):
    job = await start_agenda_job(request)
    return await await_durable_result(job['job_id'])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8010)


@app.get('/api/model-jobs/{job_id}/documents/{sha256}')
async def model_job_document(job_id: str, sha256: str):
    job = durable.load(job_id)
    if job is None:
        raise HTTPException(404, 'Job nicht gefunden')
    document = next((doc for doc in job.get('documents') or [] if doc['sha256'] == sha256), None)
    if document is None:
        raise HTTPException(404, 'Dokument nicht gefunden')
    path = Path(document['path'])
    if document.get('deleted_at') or not path.is_file():
        raise HTTPException(410, 'Quelldokument nicht mehr vorhanden')
    if durable.document(path)['sha256'] != sha256:
        raise HTTPException(409, 'Quelldokument wurde verändert')
    return FileResponse(path, media_type='application/pdf', headers={
        'Content-Disposition': 'inline; filename="Einladung.pdf"', 'Cache-Control': 'private, no-store'})
