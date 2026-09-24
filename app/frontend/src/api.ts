import type { ProcessingMode } from './types';
/**
 * API client for the GremienPilot backend
 */

import type {
  AgendaDetectionRequest,
  AgendaDetectionResponse,
  AssignmentSuggestionsResponse,
  ExportFormat,
  ExportMetadata,
  PdfAgendaExtractionResult,
  PipelineJob,
  PipelineResultResponse,
  PipelineStartOptions,
  SessionResponse,
  SessionHistoryResponse,
  SessionSavePayload,
  SummaryJob,
  SpeakerEmbeddingBackfillResult,
  SpeakerMatchDiagnostic,
  SpeakerObservation,
  SpeakerProfile,
  SummaryReviewWarning,
  SummarySourceLink,
  StructuredSummary,
  TranscriptLine,
  TranscriptionJob,
} from "./types";

export class SessionConflictError extends Error {
  actualRevision?: number;

  constructor(message: string, actualRevision?: number) {
    super(message);
    this.name = "SessionConflictError";
    this.actualRevision = actualRevision;
  }
}

export const API_BASE = import.meta.env.VITE_API_URL || "";
const configuredClientLlmTextChars = Number(
  import.meta.env.VITE_MAX_CLIENT_LLM_TEXT_CHARS
);
const MAX_CLIENT_LLM_TEXT_CHARS =
  Number.isFinite(configuredClientLlmTextChars) && configuredClientLlmTextChars > 0
    ? configuredClientLlmTextChars
    : 120000;

async function readApiError(response: Response, fallback: string): Promise<Error> {
  try {
    const error = await response.json();
    if (typeof error.detail === "string") {
      return new Error(error.detail);
    }
    if (error.detail?.message) {
      return new Error(error.detail.message);
    }
    return new Error(fallback);
  } catch {
    return new Error(fallback);
  }
}

function normalizePipelineJob(data: PipelineJob): PipelineJob {
  return {
    ...data,
    warnings: data.warnings ?? [],
  };
}

function normalizeAgendaDetection(
  data: PipelineResultResponse["agenda_detection"]
): PipelineResultResponse["agenda_detection"] {
  if (!data) {
    return null;
  }
  return {
    tops: data.tops ?? [],
    transcript: data.transcript ?? [],
    assignments: data.assignments ?? [],
    segments: data.segments ?? [],
    strategy: data.strategy ?? "unknown",
    uncertain_count: data.uncertain_count ?? 0,
    llm: data.llm ?? null,
    warnings: data.warnings ?? [],
  };
}

function estimateTranscriptChars(lines: TranscriptLine[]): number {
  return lines.reduce(
    (sum, line) => sum + line.speaker.length + line.text.length + 3,
    0
  );
}

function assertClientLlmPayloadFits(lines: TranscriptLine[], label: string): void {
  if (estimateTranscriptChars(lines) <= MAX_CLIENT_LLM_TEXT_CHARS) {
    return;
  }
  throw new Error(
    `${label} ist für den klassischen Browser-Workflow zu groß. ` +
      "Bitte die Backend-Pipeline verwenden oder die Sitzung in kleinere Abschnitte teilen."
  );
}

/**
 * Start a transcription job by uploading an audio file.
 */
export async function startTranscription(
  audioFile: File,
  sessionId?: string | null,
  rememberSpeakers = false
): Promise<TranscriptionJob> {
  const formData = new FormData();
  formData.append("audio", audioFile);
  if (sessionId) {
    formData.append("session_id", sessionId);
  }
  formData.append("remember_speakers", String(rememberSpeakers));

  const response = await fetch(`${API_BASE}/api/transcribe`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Fehler beim Starten der Transkription");
  }

  return response.json();
}

/**
 * Start the backend-controlled end-to-end pipeline.
 */
export async function startPipeline(
  audioFile: File,
  options: PipelineStartOptions = {}
): Promise<PipelineJob> {
  const formData = new FormData();
  formData.append("audio", audioFile);
  formData.append("processing_mode", options.processingMode ?? "slow");
  if (options.autoDetectTopsFromPdf && !options.pdfFile && !options.tops?.length && !options.skipAgendaDetection) {
    throw new Error('PDF-Erkennung ist aktiviert, aber keine Einladung hochgeladen. PDF auswählen oder PDF-Erkennung ausschalten.');
  }
  if (options.pdfSourceJobId) formData.append('pdf_source_job_id', options.pdfSourceJobId);
  if (options.pdfFile) {
    formData.append("pdf", options.pdfFile);
  }
  if (options.sessionId) {
    formData.append("session_id", options.sessionId);
  }

  const knownTops = (options.tops ?? []).map((top) => top.trim()).filter(Boolean);
  if (knownTops.length > 0) {
    formData.append("tops", JSON.stringify(knownTops));
  }
  if (options.model) {
    formData.append("model", options.model);
  }
  if (options.systemPrompt) {
    formData.append("system_prompt", options.systemPrompt);
  }
  for (const [field, prompt] of [
    ["summary_system_prompt", options.summarySystemPrompt],
    ["agenda_system_prompt", options.agendaSystemPrompt],
    ["pdf_system_prompt", options.pdfSystemPrompt],
  ] as const) {
    if (prompt !== undefined) formData.append(field, prompt);
  }
  formData.append("remember_speakers", String(Boolean(options.rememberSpeakers)));
  if (options.agendaFresh) formData.append('agenda_fresh', 'true');
  if (options.agendaUseLlm != null) {
    formData.append("agenda_use_llm", String(options.agendaUseLlm));
  }
  formData.append("skip_agenda_detection", String(Boolean(options.skipAgendaDetection)));
  formData.append(
    "auto_detect_tops_from_pdf",
    String(Boolean(options.autoDetectTopsFromPdf))
  );

  const response = await fetch(`${API_BASE}/api/pipeline/start`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw await readApiError(response, "Fehler beim Starten der Pipeline");
  }

  return normalizePipelineJob(await response.json());
}

/**
 * Get the current status of an end-to-end pipeline job.
 */
export async function getPipelineStatus(pipelineId: string): Promise<PipelineJob> {
  const response = await fetch(`${API_BASE}/api/pipeline/${pipelineId}`, { signal: AbortSignal.timeout(15000) });

  if (!response.ok) {
    throw new ModelJobStatusError(response.status);
  }

  return normalizePipelineJob(await response.json());
}

/**
 * Cancel a pending or running pipeline job.
 */
export async function cancelPipeline(pipelineId: string): Promise<PipelineJob> {
  const response = await fetch(`${API_BASE}/api/pipeline/${pipelineId}/cancel`, {
    method: "POST",
  });

  if (!response.ok) {
    throw await readApiError(response, "Fehler beim Abbrechen der Pipeline");
  }

  return normalizePipelineJob(await response.json());
}

/**
 * Load the reviewable pipeline result once the pipeline has completed.
 */
export async function getPipelineResult(
  pipelineId: string
): Promise<PipelineResultResponse> {
  const response = await fetch(`${API_BASE}/api/pipeline/${pipelineId}/result`);

  if (!response.ok) {
    throw await readApiError(response, "Fehler beim Laden des Pipeline-Ergebnisses");
  }

  const result = await response.json();
  return {
    ...result,
    pipeline: normalizePipelineJob(result.pipeline),
    warnings: result.warnings ?? [],
    agenda_detection: normalizeAgendaDetection(result.agenda_detection),
  };
}

/**
 * Poll for pipeline completion.
 */
export async function pollPipeline(
  pipelineId: string,
  onStatus?: (status: PipelineJob) => void,
  intervalMs = 1000
): Promise<PipelineJob> {
  while (true) {
    let status: PipelineJob;
    try {
      status = await getPipelineStatus(pipelineId);
    } catch (error) {
      if (error instanceof ModelJobStatusError && error.status >= 400 && error.status < 500 && error.status !== 429) throw error;
      await new Promise(resolve => setTimeout(resolve, intervalMs));
      continue;
    }
    onStatus?.(status);

    if (status.status === "completed") {
      return status;
    }

    if (status.status === "failed") {
      throw new Error(status.error || "Pipeline fehlgeschlagen");
    }

    if (status.status === "cancelled") {
      throw new Error("Pipeline abgebrochen");
    }

    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}

/**
 * Get the status of a transcription job.
 */
export async function getTranscriptionStatus(
  jobId: string
): Promise<TranscriptionJob> {
  const response = await fetch(`${API_BASE}/api/transcribe/${jobId}`);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Fehler beim Abrufen des Status");
  }

  return response.json();
}

/**
 * Create or update an editing session.
 */
export async function saveSession(
  payload: SessionSavePayload
): Promise<SessionResponse> {
  const sessionId = payload.session_id?.trim();
  const response = await fetch(
    sessionId ? `${API_BASE}/api/sessions/${sessionId}` : `${API_BASE}/api/sessions`,
    {
      method: sessionId ? "PUT" : "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    }
  );

  if (response.status === 409) {
    const error = await response.json().catch(() => ({}));
    throw new SessionConflictError(
      error.detail?.message || "Diese Sitzung wurde zwischenzeitlich geändert.",
      error.detail?.actual_revision
    );
  }

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Fehler beim Speichern der Sitzung");
  }

  return response.json();
}

export interface ListSessionsOptions {
  limit?: number;
  offset?: number;
  query?: string;
  status?: string;
}

/**
 * Load the shared server-side session history.
 */
export async function listSessions(
  options: ListSessionsOptions = {}
): Promise<SessionHistoryResponse> {
  const params = new URLSearchParams();
  params.set("limit", String(options.limit ?? 25));
  params.set("offset", String(options.offset ?? 0));
  if (options.query?.trim()) {
    params.set("query", options.query.trim());
  }
  if (options.status?.trim()) {
    params.set("status", options.status.trim());
  }
  const response = await fetch(`${API_BASE}/api/sessions?${params.toString()}`);
  if (!response.ok) {
    throw await readApiError(response, "Sitzungsverlauf konnte nicht geladen werden");
  }
  return response.json();
}

/**
 * Load a persisted editing session.
 */
export async function loadSession(sessionId: string): Promise<SessionResponse> {
  const response = await fetch(`${API_BASE}/api/sessions/${sessionId}`);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Fehler beim Laden der Sitzung");
  }

  return response.json();
}

export async function startSummaryJob(
  sessionId: string,
  payload: {
    revision?: number | null;
    topIds: string[];
    model?: string;
    systemPrompt?: string;
  }
): Promise<SummaryJob> {
  const response = await fetch(`${API_BASE}/api/sessions/${sessionId}/summary-jobs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      revision: payload.revision,
      top_ids: payload.topIds,
      model: payload.model,
      system_prompt: payload.systemPrompt,
    }),
  });
  if (response.status === 409) {
    const error = await readApiError(response, 'Zusammenfassungsjob konnte nicht gestartet werden');
    if (/zwischenzeitlich geändert/i.test(error.message)) {
      throw new SessionConflictError(error.message);
    }
    throw error;
  }
  if (!response.ok) {
    throw await readApiError(response, 'Zusammenfassungsjob konnte nicht gestartet werden');
  }
  return response.json();
}

export async function getSummaryJob(summaryJobId: string): Promise<SummaryJob> {
  const response = await fetch(`${API_BASE}/api/summary-jobs/${summaryJobId}`);
  if (!response.ok) {
    throw await readApiError(response, 'Zusammenfassungsjob konnte nicht geladen werden');
  }
  return response.json();
}

export async function pollSummaryJob(
  summaryJobId: string,
  onStatus?: (job: SummaryJob) => void | Promise<void>,
  intervalMs = 1500,
  signal?: AbortSignal,
): Promise<SummaryJob> {
  while (true) {
    signal?.throwIfAborted();
    const job = await getSummaryJob(summaryJobId);
    signal?.throwIfAborted();
    await onStatus?.(job);
    if (['completed', 'failed', 'cancelled'].includes(job.status)) {
      return job;
    }
    await new Promise((resolve) => window.setTimeout(resolve, intervalMs));
  }
}

export async function cancelSummaryJob(summaryJobId: string): Promise<SummaryJob> {
  const response = await fetch(`${API_BASE}/api/summary-jobs/${summaryJobId}/cancel`, {
    method: 'POST',
  });
  if (!response.ok) {
    throw await readApiError(response, 'Zusammenfassungsjob konnte nicht abgebrochen werden');
  }
  return response.json();
}

export async function acceptExistingSummary(
  sessionId: string,
  topId: string,
  revision?: number | null
): Promise<SessionResponse> {
  const response = await fetch(
    `${API_BASE}/api/sessions/${sessionId}/summaries/${encodeURIComponent(topId)}/accept`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ revision }),
    }
  );
  if (response.status === 409) {
    throw new SessionConflictError('Diese Sitzung wurde zwischenzeitlich geändert.');
  }
  if (!response.ok) {
    throw await readApiError(response, 'Zusammenfassung konnte nicht übernommen werden');
  }
  return response.json();
}

/**
 * Cancel a pending or running transcription job.
 */
export async function cancelTranscription(
  jobId: string
): Promise<TranscriptionJob> {
  const response = await fetch(`${API_BASE}/api/transcribe/${jobId}/cancel`, {
    method: "POST",
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Fehler beim Abbrechen der Transkription");
  }

  return response.json();
}

/**
 * Poll for transcription completion.
 * Returns the full TranscriptionJob including audio_url for playback.
 */
export async function pollTranscription(
  jobId: string,
  onProgress?: (progress: number, message: string) => void
): Promise<TranscriptionJob> {
  while (true) {
    const status = await getTranscriptionStatus(jobId);

    if (onProgress) {
      onProgress(status.progress, status.message);
    }

    if (status.status === "completed") {
      return status;
    }

    if (status.status === "failed") {
      throw new Error(status.error || "Transkription fehlgeschlagen");
    }

    if (status.status === "cancelled") {
      throw new Error(status.message || "Transkription abgebrochen");
    }

    // Wait before polling again
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
}

/**
 * Options for summary generation.
 */
export interface SummarizeOptions {
  processingMode?: ProcessingMode;
  model?: string;
  systemPrompt?: string;
}

/**
 * Result from summary generation including timing.
 */
export interface SummarizeResult {
  summary: string;
  durationSeconds: number;
  structured?: StructuredSummary | null;
  sourceLinks: SummarySourceLink[];
  reviewWarnings: SummaryReviewWarning[];
  fallbackUsed: boolean;
  chunksProcessed: number;
}

/**
 * Generate a summary for a TOP segment.
 */
export async function generateSummary(
  topTitle: string,
  lines: TranscriptLine[],
  options?: SummarizeOptions
): Promise<SummarizeResult> {
  assertClientLlmPayloadFits(lines, "Der Transkriptabschnitt");

  const response = await fetch(`${API_BASE}/api/summarize`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      top_title: topTitle,
      lines: lines,
      model: options?.model,
      system_prompt: options?.systemPrompt,
      processing_mode: options?.processingMode ?? "slow",
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Fehler bei der Zusammenfassung");
  }

  const data = await response.json();
  return {
    summary: data.summary,
    durationSeconds: data.duration_seconds,
    structured: data.structured ?? null,
    sourceLinks: data.source_links ?? [],
    reviewWarnings: data.review_warnings ?? [],
    fallbackUsed: Boolean(data.fallback_used),
    chunksProcessed: data.chunks_processed ?? 1,
  };
}

/**
 * Suggest reviewable TOP assignments for a transcript.
 */
export async function generateAssignmentSuggestions(
  tops: string[],
  transcript: TranscriptLine[]
): Promise<AssignmentSuggestionsResponse> {
  const response = await fetch(`${API_BASE}/api/assignment-suggestions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      tops,
      transcript,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Fehler beim Erzeugen der Zuordnungsvorschläge");
  }

  return response.json();
}

/**
 * Detect agenda titles and reviewable TOP assignments for a transcript.
 */
export interface ModelJob {
  job_id: string;
  kind: string;
  state: 'queued' | 'running' | 'retry_wait' | 'review_required' | 'failed' | 'completed' | 'cancelled' | 'superseded';
  created_at?: number;
  updated_at?: number;
  progress?: { phase?: string; agenda_phase?: string; pdf_phase?: string; page?: number; total_pages?: number; round?: number; last_delta_at?: number | null; silence_seconds?: number; processed_lines?: number; total_lines?: number; model_calls?: number; elapsed_seconds?: number };
  source?: { tops: string[]; top_ids: string[]; transcript: TranscriptLine[]; processing_mode?: ProcessingMode };
  error?: string | null;
  result?: unknown;
  documents?: { sha256: string; deleted_at?: number | null }[];
}

class ModelJobStatusError extends Error {
  constructor(public status: number) { super(`Jobstatus nicht verfügbar (${status})`); }
}

export async function getModelJob(jobId: string): Promise<ModelJob> {
  const response = await fetch(`${API_BASE}/api/model-jobs/${jobId}`, { signal: AbortSignal.timeout(15000) });
  if (!response.ok) throw new ModelJobStatusError(response.status);
  return response.json();
}

export async function cancelModelJob(jobId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/api/model-jobs/${jobId}/cancel`, { method: 'POST', signal: AbortSignal.timeout(15000) });
  if (!response.ok) throw new Error('Abbruch konnte nicht bestätigt werden');
}

export async function pollModelJob<T>(jobId: string, signal?: AbortSignal,
  onStatus?: (job: ModelJob) => void): Promise<T> {
  let cancellation: Promise<Error | null> | undefined;
  const requestCancel = () => {
    cancellation ??= cancelModelJob(jobId).then(() => null, error => error as Error);
  };
  signal?.addEventListener('abort', requestCancel, { once: true });
  try {
    while (true) {
      if (signal?.aborted) {
        requestCancel();
        const error = await cancellation;
        throw error || new Error('Verarbeitung abgebrochen');
      }
      let job: ModelJob;
      try {
        job = await getModelJob(jobId);
      } catch (error) {
        if (error instanceof ModelJobStatusError && error.status >= 400 && error.status < 500 && error.status !== 429) throw error;
        // Reconnect to the existing job; never resubmit model work on a polling failure.
        await new Promise(resolve => setTimeout(resolve, 2000));
        continue;
      }
      onStatus?.(job);
      if (signal?.aborted) continue;
      if (['completed', 'review_required'].includes(job.state) && job.result != null) return job.result as T;
      if (['failed', 'cancelled', 'superseded', 'review_required'].includes(job.state)) {
          const result = job.result as { llm?: { failure_reasons?: string[] }; warnings?: string[] } | null;
          throw new Error(job.error || result?.llm?.failure_reasons?.join(', ') ||
            (job.state === 'cancelled' ? 'Verarbeitung abgebrochen' : 'Verarbeitung fehlgeschlagen. Bitte erneut starten.'));
      }
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
  } finally {
    signal?.removeEventListener('abort', requestCancel);
  }
}

export async function detectAgenda(
  request: AgendaDetectionRequest
): Promise<AgendaDetectionResponse> {
  // This durable backend job plans bounded model inputs itself; a browser text
  // limit must not reject a full meeting before the job can even be started.

  const response = await fetch(`${API_BASE}/api/agenda-detection/jobs`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      tops: request.tops ?? [],
      transcript: request.transcript,
      model: request.model,
      system_prompt: request.systemPrompt,
      processing_mode: request.processingMode ?? "slow",
      use_llm: request.useLlm,
      preserve_transcript_structure: request.preserveTranscriptStructure,
      fresh: request.fresh,
      cache_namespace: request.cacheNamespace,
      top_ids: request.topIds,
      session_id: request.sessionId,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Fehler bei der automatischen TOP-Erkennung");
  }

  const started = await response.json();
  if (started.job_id) request.onStatus?.(started);
  const data = started.job_id
    ? await pollModelJob<AgendaDetectionResponse>(started.job_id, request.signal, request.onStatus)
    : started;
  return {
    tops: data.tops ?? [],
    transcript: data.transcript ?? request.transcript,
    assignments: data.assignments ?? [],
    segments: data.segments ?? [],
    strategy: data.strategy ?? "unknown",
    uncertain_count: data.uncertain_count ?? 0,
    llm: data.llm ?? null,
    warnings: data.warnings ?? [],
  };
}

export interface ListSpeakerProfilesOptions {
  scope?: string | null;
  includeArchived?: boolean;
}

export interface SpeakerProfilePayload {
  displayName: string;
  scope?: string | null;
}

export async function listSpeakerProfiles(
  options: ListSpeakerProfilesOptions = {}
): Promise<SpeakerProfile[]> {
  const params = new URLSearchParams();
  if (options.scope) {
    params.set("scope", options.scope);
  }
  if (options.includeArchived) {
    params.set("include_archived", "true");
  }

  const query = params.toString();
  const response = await fetch(
    `${API_BASE}/api/speaker-profiles${query ? `?${query}` : ""}`
  );

  if (!response.ok) {
    throw await readApiError(response, "Fehler beim Laden der Sprecherprofile");
  }

  return response.json();
}

export async function createSpeakerProfile(
  payload: SpeakerProfilePayload
): Promise<SpeakerProfile> {
  const response = await fetch(`${API_BASE}/api/speaker-profiles`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      display_name: payload.displayName,
      scope: payload.scope,
    }),
  });

  if (!response.ok) {
    throw await readApiError(response, "Fehler beim Anlegen des Sprecherprofils");
  }

  return response.json();
}

export async function updateSpeakerProfile(
  profileId: string,
  payload: Partial<SpeakerProfilePayload>
): Promise<SpeakerProfile> {
  const body: Record<string, string | null> = {};
  if (payload.displayName !== undefined) {
    body.display_name = payload.displayName;
  }
  if (payload.scope !== undefined) {
    body.scope = payload.scope;
  }

  const response = await fetch(`${API_BASE}/api/speaker-profiles/${profileId}`, {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw await readApiError(response, "Fehler beim Aktualisieren des Sprecherprofils");
  }

  return response.json();
}

export async function archiveSpeakerProfile(profileId: string): Promise<SpeakerProfile> {
  const response = await fetch(`${API_BASE}/api/speaker-profiles/${profileId}`, {
    method: "DELETE",
  });

  if (!response.ok) {
    throw await readApiError(response, "Fehler beim Archivieren des Sprecherprofils");
  }

  return response.json();
}

export async function deleteSpeakerProfileEmbeddings(
  profileId: string
): Promise<{ profile_id: string; deleted_count: number }> {
  const response = await fetch(
    `${API_BASE}/api/speaker-profiles/${profileId}/embeddings`,
    {
      method: "DELETE",
    }
  );

  if (!response.ok) {
    throw await readApiError(response, "Fehler beim Löschen der Sprecher-Embeddings");
  }

  return response.json();
}

export async function backfillSpeakerEmbeddings(
  profileId?: string | null
): Promise<SpeakerEmbeddingBackfillResult> {
  const params = new URLSearchParams();
  if (profileId) {
    params.set("profile_id", profileId);
  }
  const query = params.toString();
  const response = await fetch(
    `${API_BASE}/api/speaker-embeddings/backfill${query ? `?${query}` : ""}`,
    {
      method: "POST",
    }
  );

  if (!response.ok) {
    throw await readApiError(
      response,
      "Fehler beim Nachholen der Sprecher-Embeddings"
    );
  }

  return response.json();
}

export async function listSpeakerObservations(
  sessionId: string
): Promise<SpeakerObservation[]> {
  const response = await fetch(
    `${API_BASE}/api/sessions/${sessionId}/speaker-observations`
  );

  if (!response.ok) {
    throw await readApiError(response, "Fehler beim Laden der Sprecher-Erkennungen");
  }

  return response.json();
}

export async function listSpeakerMatchDiagnostics(
  sessionId: string
): Promise<SpeakerMatchDiagnostic[]> {
  const response = await fetch(
    `${API_BASE}/api/sessions/${sessionId}/speaker-match-diagnostics`
  );

  if (!response.ok) {
    throw await readApiError(
      response,
      "Fehler beim Laden der Sprecher-Erkennungsdiagnose"
    );
  }

  return response.json();
}

export interface ConfirmSpeakerObservationOptions {
  profileId?: string | null;
  confidence?: number | null;
}

export async function confirmSpeakerObservation(
  sessionId: string,
  observationId: number,
  options: ConfirmSpeakerObservationOptions = {}
): Promise<SpeakerObservation> {
  const response = await fetch(
    `${API_BASE}/api/sessions/${sessionId}/speaker-observations/${observationId}/confirm`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        profile_id: options.profileId,
        confidence: options.confidence,
      }),
    }
  );

  if (!response.ok) {
    throw await readApiError(response, "Fehler beim Bestätigen der Sprecher-Erkennung");
  }

  return response.json();
}

export async function rejectSpeakerObservation(
  sessionId: string,
  observationId: number
): Promise<SpeakerObservation> {
  const response = await fetch(
    `${API_BASE}/api/sessions/${sessionId}/speaker-observations/${observationId}/reject`,
    {
      method: "POST",
    }
  );

  if (!response.ok) {
    throw await readApiError(response, "Fehler beim Ablehnen der Sprecher-Erkennung");
  }

  return response.json();
}

export async function unassignSpeakerObservation(
  sessionId: string,
  observationId: number
): Promise<SpeakerObservation> {
  const response = await fetch(
    `${API_BASE}/api/sessions/${sessionId}/speaker-observations/${observationId}/unassign`,
    {
      method: "POST",
    }
  );

  if (!response.ok) {
    throw await readApiError(response, "Fehler beim Lösen der Sprecher-Zuordnung");
  }

  return response.json();
}

export interface ManualSpeakerObservationPayload {
  localSpeakerId: string;
  profileId?: string | null;
  displayName?: string | null;
  scope?: string | null;
  confidence?: number | null;
  observationId?: number | null;
}

export async function createManualSpeakerObservation(
  sessionId: string,
  payload: ManualSpeakerObservationPayload
): Promise<SpeakerObservation> {
  const response = await fetch(
    `${API_BASE}/api/sessions/${sessionId}/speaker-observations/manual`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        local_speaker_id: payload.localSpeakerId,
        profile_id: payload.profileId,
        display_name: payload.displayName,
        scope: payload.scope,
        confidence: payload.confidence,
        observation_id: payload.observationId,
      }),
    }
  );

  if (!response.ok) {
    throw await readApiError(response, "Fehler beim Speichern der Sprecher-Zuordnung");
  }

  return response.json();
}

/**
 * Options for TOP extraction from PDF.
 */
export interface ExtractTOPsOptions {
  processingMode?: ProcessingMode;
  signal?: AbortSignal;
  onStatus?: (job: ModelJob) => void;
  model?: string;
  systemPrompt?: string;
}

export async function reextractSessionPDF(sessionId: string, options: ExtractTOPsOptions): Promise<PdfAgendaExtractionResult> {
  const response = await fetch(`${API_BASE}/api/sessions/${encodeURIComponent(sessionId)}/pdf-jobs`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: options.model, system_prompt: options.systemPrompt,
      processing_mode: options.processingMode }),
  });
  if (!response.ok) throw await readApiError(response, 'PDF-Extraktion konnte nicht gestartet werden');
  const job: ModelJob = await response.json();
  options.onStatus?.(job);
  return pollModelJob<PdfAgendaExtractionResult>(job.job_id, options.signal, options.onStatus);
}

/**
 * Extract TOPs and session metadata from a PDF meeting invitation.
 */
export function pdfResultUsable(data: PdfAgendaExtractionResult, mode: ProcessingMode): boolean {
  if (data.processing_complete !== true) return false;
  return data.processing_mode === 'fast'
    ? mode === 'fast' && data.review_status === 'skipped'
    : !data.review_required;
}

export async function extractAgendaDataFromPDF(
  pdfFile: File,
  options?: ExtractTOPsOptions
): Promise<PdfAgendaExtractionResult> {
  const formData = new FormData();
  formData.append("pdf", pdfFile);
  formData.append("processing_mode", options?.processingMode ?? "slow");

  // Add optional parameters as form fields
  if (options?.model) {
    formData.append("model", options.model);
  }
  if (options?.systemPrompt) {
    formData.append("system_prompt", options.systemPrompt);
  }

  const response = await fetch(`${API_BASE}/api/extract-tops/jobs`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Fehler beim Extrahieren der TOPs");
  }

  const started = await response.json();
  if (started.job_id) localStorage.setItem('gremienpilot-pdf-job', started.job_id);
  const data = started.job_id
    ? await pollModelJob<PdfAgendaExtractionResult>(started.job_id, options?.signal, options?.onStatus)
    : started;
  if (!pdfResultUsable(data, options?.processingMode ?? "slow")) {
    throw new Error('PDF-Auswertung ist nicht vollständig geprüft. Ergebnis wurde nicht übernommen.');
  }
  return { ...data, tops: data.tops ?? [], metadata: data.metadata ?? {} };
}

/**
 * Extract only TOPs from a PDF meeting invitation.
 */
export async function extractTOPsFromPDF(
  pdfFile: File,
  options?: ExtractTOPsOptions
): Promise<string[]> {
  const data = await extractAgendaDataFromPDF(pdfFile, options);
  return data.tops;
}

export interface ProtocolExportPayload {
  sessionId?: string | null;
  format: ExportFormat;
  metadata: ExportMetadata;
  tops: string[];
  transcript: TranscriptLine[];
  assignments: (number | null)[];
  speakerNames: Record<string, string>;
  summaries: Record<number, string>;
  summaryReviews?: Record<number, unknown>;
}

export async function exportProtocol(payload: ProtocolExportPayload): Promise<Blob> {
  const response = await fetch(`${API_BASE}/api/export`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      format: payload.format,
      session_id: payload.sessionId,
      metadata: {
        committee: payload.metadata.committee,
        date: payload.metadata.date,
        location: payload.metadata.location,
        title: payload.metadata.title,
        participants: payload.metadata.participants,
      },
      appendix: {
        include_speaker_list: payload.metadata.includeSpeakerList,
        include_transcript: payload.metadata.includeTranscript,
        include_transcript_excerpt: payload.metadata.includeTranscript,
        group_transcript_by_top: payload.metadata.groupTranscriptByTop,
        include_generation_note: payload.metadata.includeGenerationNote,
      },
      tops: payload.tops,
      transcript: payload.transcript,
      assignments: payload.assignments,
      speaker_names: payload.speakerNames,
      summaries: payload.summaries,
      summary_reviews: payload.summaryReviews ?? {},
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || "Fehler beim Exportieren des Protokolls");
  }

  return response.blob();
}

/**
 * Check if the backend is available.
 */
export async function checkBackendHealth(): Promise<boolean> {
  try {
    console.log("Checking backend health at", API_BASE || "(relative)");
    const response = await fetch(`${API_BASE}/health`);
    return response.ok;
  } catch {
    return false;
  }
}
