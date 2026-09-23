import { mergeSummarySession } from './summarySync';
import { useCallback, useEffect, useMemo, useRef, useState, type Dispatch, type SetStateAction } from "react";
import Layout from "./components/Layout";
import StepIndicator from "./components/StepIndicator";
import PdfSources from "./components/PdfSources";
import UploadStep from "./components/UploadStep";
import ProcessingStep from "./components/ProcessingStep";
import AssignmentStep from "./components/AssignmentStep";
import SummaryStep from "./components/SummaryStep";
import SessionHistory from "./components/SessionHistory";
import LLMSettingsPanel, {
  DEFAULT_LLM_SETTINGS,
  type LLMSettings,
} from "./components/LLMSettingsPanel";
import {
  startPipeline as apiStartPipeline,
  pollPipeline,
  getPipelineStatus,
  getPipelineResult,
  cancelPipeline,
  startSummaryJob,
  pollSummaryJob,
  cancelSummaryJob,
  acceptExistingSummary,
  checkBackendHealth,
  detectAgenda,
  saveSession,
  loadSession,
  SessionConflictError,
} from "./api";
import type {
  PdfAgendaExtractionResult,
  AgendaDetectionResponse,
  AgendaProposals,
  ExportMetadata,
  PipelineJob,
  PipelineResultResponse,
  SessionResponse,
  SessionSavePayload,
  SpeakerObservation,
  SummaryReview,
  SummaryState,
  SummaryJob,
  TranscriptLine,
} from "./types";

import { agendaSource, proposalsAreValid } from "./agendaProposals";

// LocalStorage key for LLM settings
const LLM_SETTINGS_KEY = "llm-settings";
const ACTIVE_SESSION_KEY = "active-session-id";
const ACTIVE_PIPELINE_KEY = "active-pipeline-id";
const SESSION_DRAFT_KEY = "active-session-draft";
const SPEAKER_MEMORY_OPT_IN_KEY = "speaker-memory-opt-in";

// Initial editable TOP fields before a session is processed
const EMPTY_TOPS = ["", "", ""];
const DEFAULT_SESSION_DATE = new Date().toISOString().slice(0, 10);
const DEFAULT_EXPORT_METADATA: ExportMetadata = {
  committee: "",
  date: DEFAULT_SESSION_DATE,
  location: "",
  title: "Sitzungsprotokoll",
  participants: [],
  includeSpeakerList: true,
  includeTranscript: false,
  groupTranscriptByTop: false,
  includeGenerationNote: true,
};

// Generic system prompt for conversations without TOPs
const GENERIC_SUMMARY_PROMPT = `Du bist ein Experte für die Zusammenfassung von Gesprächen und Audioaufnahmen.

Deine Aufgabe ist es, aus einem Transkript eine klare und strukturierte Zusammenfassung zu erstellen.

STIL:
- Sachlich und gut lesbar
- Dritte Person
- Paraphrasieren statt wörtlich zitieren

INHALT:
- Wesentliche Themen und Diskussionspunkte
- Wichtige Aussagen und Positionen der Teilnehmer
- Getroffene Entscheidungen oder Vereinbarungen
- Offene Punkte oder nächste Schritte

FORMAT:
- Kurze Gespräche (< 10 Äußerungen): 1-2 Absätze
- Mittlere Gespräche (10-50 Äußerungen): 2-3 Absätze
- Lange Gespräche (> 50 Äußerungen): 3-5 Absätze
- Direkt mit Inhalt beginnen, keine Einleitung
- NUR Fließtext, KEINE Markdown-Formatierung (keine **, keine #)
`;

interface SessionDraft extends SessionSavePayload {
  audio_url?: string | null;
  pipeline_id?: string | null;
}

interface AppRoute {
  view: "editor" | "history";
  sessionId: string | null;
}

function readRoute(): AppRoute {
  const sessionMatch = window.location.pathname.match(/^\/sessions\/([^/]+)\/?$/);
  if (sessionMatch) {
    return { view: "editor", sessionId: decodeURIComponent(sessionMatch[1]!) };
  }
  if (/^\/sessions\/?$/.test(window.location.pathname)) {
    return { view: "history", sessionId: null };
  }
  return { view: "editor", sessionId: null };
}

function serializeSessionPayload(payload: SessionSavePayload): string {
  const contentPayload = { ...payload };
  delete contentPayload.revision;
  return JSON.stringify(contentPayload);
}

function withApiBase(url?: string | null): string | null {
  if (!url) return null;
  if (/^https?:\/\//i.test(url)) return url;
  const baseUrl = import.meta.env.VITE_API_URL || "";
  return `${baseUrl}${url}`;
}

function normalizeSummaries(
  summaries: Record<number, string> | Record<string, string> | undefined
): Record<number, string> {
  const normalized: Record<number, string> = {};
  Object.entries(summaries ?? {}).forEach(([key, value]) => {
    const index = Number(key);
    if (Number.isFinite(index)) {
      normalized[index] = value;
    }
  });
  return normalized;
}

function normalizeSummaryReviews(
  reviews:
    | Record<number, SummaryReview>
    | Record<string, SummaryReview>
    | undefined
): Record<number, SummaryReview> {
  const normalized: Record<number, SummaryReview> = {};
  Object.entries(reviews ?? {}).forEach(([key, value]) => {
    const index = Number(key);
    if (Number.isFinite(index) && value) {
      normalized[index] = {
        ...value,
        structured: value.structured ?? null,
        source_links: value.source_links ?? [],
        review_warnings: value.review_warnings ?? [],
        fallback_used: value.fallback_used,
        chunks_processed: value.chunks_processed,
      };
    }
  });
  return normalized;
}

function normalizeSummaryStates(
  states: Record<number, SummaryState> | Record<string, SummaryState> | undefined
): Record<number, SummaryState> {
  const normalized: Record<number, SummaryState> = {};
  Object.entries(states ?? {}).forEach(([key, value]) => {
    const index = Number(key);
    if (Number.isFinite(index) && value) normalized[index] = value;
  });
  return normalized;
}

function normalizeLineText(text: string): string {
  return text.trim().replace(/\s+/g, ' ');
}

function currentSourceSnapshot(
  transcript: TranscriptLine[],
  assignments: (number | null)[],
  topIndex: number,
  noTopMode: boolean
) {
  const snapshot: Array<{ line_id: string; speaker: string; text: string }> = [];
  transcript.forEach((line, lineIndex) => {
    if (!noTopMode && assignments[lineIndex] !== topIndex) return;
    const previous = snapshot[snapshot.length - 1];
    if (previous?.speaker === line.speaker) {
      previous.text = normalizeLineText(`${previous.text} ${line.text}`);
    } else {
      snapshot.push({
        line_id: line.line_id ?? `legacy:${lineIndex}`,
        speaker: line.speaker,
        text: normalizeLineText(line.text),
      });
    }
  });
  return snapshot;
}

function replaceExactLabels<T>(value: T, replacements: Record<string, string>): T {
  const entries = Object.entries(replacements)
    .filter(([oldLabel, newLabel]) => oldLabel && newLabel && oldLabel !== newLabel)
    .sort(([left], [right]) => right.length - left.length);
  if (entries.length === 0) return value;
  const escaped = entries.map(([label]) => label.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
  const pattern = new RegExp(`(^|[^\\p{L}\\p{N}_])(${escaped.join('|')})(?=$|[^\\p{L}\\p{N}_])`, 'gu');
  const replacementMap = Object.fromEntries(entries);
  const visit = (item: unknown): unknown => {
    if (typeof item === 'string') {
      return item.replace(pattern, (_match, prefix: string, label: string) => `${prefix}${replacementMap[label]}`);
    }
    if (Array.isArray(item)) return item.map(visit);
    if (item && typeof item === 'object') {
      return Object.fromEntries(Object.entries(item).map(([key, child]) => [key, visit(child)]));
    }
    return item;
  };
  return visit(value) as T;
}

function hasFreshSessionSummaries(session: SessionResponse | SessionDraft): boolean {
  const states = normalizeSummaryStates(session.summary_states);
  if (Object.keys(states).length > 0) {
    return hasAnySummaryArtifact(session) && Object.values(states).every(
      (state) => state.status === 'ready'
    );
  }
  return hasAnySummaryArtifact(session);
}

function hasAgendaUncertainty(
  agendaDetection?: AgendaDetectionResponse | null
): boolean {
  if (!agendaDetection) {
    return false;
  }
  return (
    (agendaDetection.uncertain_count ?? 0) > 0 ||
    (agendaDetection.segments ?? []).some((segment) => segment.uncertain)
  );
}

function hasTechnicalProcessingFailure(session: SessionResponse, detection?: AgendaDetectionResponse | null): boolean {
  const usage = (detection ?? session.agenda_proposals?.result)?.llm;
  return Boolean(usage?.enabled && ['failed', 'partial_failure', 'fallback', 'partial_fallback'].includes(usage.status)) ||
    Object.values(normalizeSummaryReviews(session.summary_reviews)).some(review =>
      review.review_warnings?.some(warning => warning.kind === 'summary_failed'));
}

const INCOMPLETE_PIPELINE_NOTICE = "Automatische Verarbeitung technisch unvollständig. Bitte prüfen Sie die Fehlermeldungen und fehlenden Zuordnungen oder Zusammenfassungen.";

function hasReviewUncertainty(
  session: SessionResponse,
  warnings: string[] = [],
  agendaDetection?: AgendaDetectionResponse | null
): boolean {
  const transcriptLength = session.transcript?.length ?? 0;
  const assignments = session.assignments ?? [];
  if (transcriptLength > 0 && session.skipped_assignment) {
    return true;
  }
  if (transcriptLength > 0 && !session.skipped_assignment) {
    if (assignments.length !== transcriptLength) {
      return true;
    }
    if (assignments.some((assignment) => assignment === null || assignment === undefined)) {
      return true;
    }
  }

  if (warnings.length > 0) {
    return true;
  }

  const proposals = session.agenda_proposals;
  const detection = agendaDetection ?? proposals?.result;
  if (transcriptLength > 0 && session.tops.length > 0 && !session.skipped_assignment &&
      !proposalsAreValid(proposals, session.tops, session.top_ids ?? [], session.transcript ?? [])) {
    return true;
  }
  if (hasAgendaUncertainty(detection) || (detection?.warnings?.length ?? 0) > 0) {
    return true;
  }

  const reviews = normalizeSummaryReviews(session.summary_reviews);
  return Object.values(reviews).some((review) => {
    const warningsForTop = review.review_warnings ?? [];
    const hasBlockingWarning = warningsForTop.some((warning) =>
      ["warning", "error"].includes(String(warning.severity ?? "").toLowerCase())
    );
    const hasStructuredUncertainty = Boolean(
      review.structured?.uncertainties?.some((item) => item.trim())
    );
    return hasBlockingWarning || hasStructuredUncertainty;
  });
}

function hasAnySummary(session: SessionResponse | SessionDraft): boolean {
  return Object.values(session.summaries ?? {}).some(
    (summary) => typeof summary === "string" && summary.trim() !== ""
  );
}

function hasAnySummaryArtifact(session: SessionResponse | SessionDraft): boolean {
  if (hasAnySummary(session)) {
    return true;
  }
  return Object.keys(session.summary_reviews ?? {}).length > 0;
}

function getTranscriptSpeakers(transcript: TranscriptLine[] = []): string[] {
  return Array.from(new Set(transcript.map((line) => line.speaker).filter(Boolean)));
}

function hasUnverifiedSpeakers(
  session: SessionResponse,
  observations: SpeakerObservation[] = []
): boolean {
  const speakers = getTranscriptSpeakers(session.transcript ?? []);
  if (speakers.length === 0) {
    return false;
  }
  if (observations.some((observation) => observation.status === "suggested")) {
    return true;
  }
  return speakers.some((speaker) => {
    const displayName = (session.speaker_names ?? {})[speaker]?.trim();
    return !displayName || displayName.toLowerCase() === speaker.toLowerCase();
  });
}

function applySuggestedSpeakerNames(
  session: SessionResponse,
  observations: SpeakerObservation[] = []
): Record<string, string> {
  const speakerNames = { ...(session.speaker_names ?? {}) };
  observations.forEach((observation) => {
    if (observation.status !== "suggested") {
      return;
    }
    const suggestedName =
      observation.profile_display_name?.trim() || observation.display_name?.trim();
    if (!suggestedName) {
      return;
    }
    const currentName = speakerNames[observation.local_speaker_id]?.trim();
    if (
      !currentName ||
      currentName.toLowerCase() === observation.local_speaker_id.toLowerCase()
    ) {
      speakerNames[observation.local_speaker_id] = suggestedName;
    }
  });
  return speakerNames;
}

export default function App() {
  const [route, setRoute] = useState<AppRoute>(() => readRoute());

  // Current step in wizard
  const [currentStep, setCurrentStep] = useState(1);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingProgress, setProcessingProgress] = useState(0);
  const [processingStatus, setProcessingStatus] = useState("");
  const [processingError, setProcessingError] = useState<string | null>(null);

  // Backend status
  const [backendAvailable, setBackendAvailable] = useState<boolean | null>(
    null
  );

  // Data state
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [pdfFile, setPdfFile] = useState<File | null>(null);
  const [pdfExtraction, setPdfExtraction] = useState<PdfAgendaExtractionResult | null>(null);
  const [tops, setTops] = useState<string[]>(EMPTY_TOPS);
  const [topIds, setTopIds] = useState<string[]>([]);
  const [transcript, setTranscript] = useState<TranscriptLine[]>([]);
  const [assignments, setAssignments] = useState<(number | null)[]>([]);
  const [agendaProposals, setAgendaProposals] = useState<AgendaProposals | null>(null);
  const agendaDetection = agendaProposals?.result ?? null;
  const agendaDetectionStale = Boolean(agendaProposals) &&
    !proposalsAreValid(agendaProposals, tops, topIds, transcript);
  const [isDetectingAgenda, setIsDetectingAgenda] = useState(false);
  const [agendaJobPhase, setAgendaJobPhase] = useState('');
  const agendaRequestRef = useRef(0);
  const agendaAbortRef = useRef<AbortController | null>(null);
  const agendaInputEpochRef = useRef(0);
  const agendaInputKey = JSON.stringify(agendaSource(tops, topIds, transcript));
  useEffect(() => {
    agendaInputEpochRef.current += 1;
    agendaAbortRef.current?.abort();
  }, [agendaInputKey, route.view, route.sessionId]);
  const [agendaDetectionError, setAgendaDetectionError] = useState<string | null>(null);
  const [summaries, setSummaries] = useState<Record<number, string>>({});
  const [summaryReviews, setSummaryReviews] = useState<Record<number, SummaryReview>>({});
  const [summaryStates, setSummaryStates] = useState<Record<number, SummaryState>>({});
  const [exportMetadata, setExportMetadata] = useState<ExportMetadata>(DEFAULT_EXPORT_METADATA);
  const [isGeneratingSummary, setIsGeneratingSummary] = useState(false);
  const [summaryJob, setSummaryJob] = useState<SummaryJob | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [speakerNames, setSpeakerNames] = useState<Record<string, string>>({});
  const [skippedAssignment, setSkippedAssignment] = useState(false);
  const [skipAgendaDetection, setSkipAgendaDetection] = useState(false);
  const [autoDetectTopsFromPdf, setAutoDetectTopsFromPdf] = useState(true);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sessionRevision, setSessionRevision] = useState<number | null>(null);
  const [pipelineId, setPipelineId] = useState<string | null>(null);
  const [pipelineJob, setPipelineJob] = useState<PipelineJob | null>(null);
  const [pipelineNotice, setPipelineNotice] = useState<string | null>(null);
  const [directProtocolAvailable, setDirectProtocolAvailable] = useState(false);
  const [restoreCandidate, setRestoreCandidate] = useState<{
    sessionId: string | null;
    pipelineId: string | null;
    draft: SessionDraft | null;
  } | null>(null);
  const [isRestoringSession, setIsRestoringSession] = useState(false);
  const [sessionMessage, setSessionMessage] = useState<string | null>(null);
  const [sessionConflict, setSessionConflict] = useState<string | null>(null);
  const [isLoadingRouteSession, setIsLoadingRouteSession] = useState(
    () => readRoute().sessionId !== null
  );
  const lastSavedPayloadRef = useRef<string | null>(null);
  const saveTimerRef = useRef<number | null>(null);
  const sessionRevisionRef = useRef<number | null>(null);
  const activeSessionIdRef = useRef<string | null>(null);
  const saveChainRef = useRef<Promise<void>>(Promise.resolve());
  const saveConflictRef = useRef(false);

  const effectiveSummaryStates = useMemo(() => {
    const noTopMode = tops.filter((top) => top.trim()).length === 0;
    const count = noTopMode ? 1 : tops.filter((top) => top.trim()).length;
    const next: Record<number, SummaryState> = {};
    for (let index = 0; index < count; index++) {
      const persisted = summaryStates[index];
      const snapshot = currentSourceSnapshot(transcript, assignments, index, noTopMode);
      const baseline = persisted?.source_snapshot ?? snapshot;
      const semantic = (items: typeof snapshot) => normalizeLineText(items.map(({ text }) => text).join(' '));
      const semanticChange = JSON.stringify(semantic(snapshot)) !== JSON.stringify(semantic(baseline));
      const summaryExists = Boolean(summaries[index]?.trim());
      next[index] = {
        ...(persisted ?? {
          top_id: noTopMode ? `whole-session:${sessionId ?? ''}` : topIds[index] ?? `top:${index}`,
          status: summaryExists ? 'ready' : 'missing',
          source_snapshot: baseline,
        }),
        status:
          persisted?.status === 'queued' || persisted?.status === 'running'
            ? persisted.status
            : !summaryExists
              ? persisted?.status === 'failed' ? 'failed' : 'missing'
              : semanticChange
                ? 'review_required'
                : persisted?.status ?? 'ready',
        change_reasons: semanticChange
          ? persisted?.change_reasons?.length ? persisted.change_reasons : ['summary_input_changed']
          : persisted?.change_reasons ?? [],
      };
    }
    return next;
  }, [assignments, sessionId, summaries, summaryStates, topIds, tops, transcript]);
  const summariesAreFresh = Object.values(effectiveSummaryStates).every(
    (state) => state.status === 'ready'
  );
  const hasFreshSummariesInState = hasAnySummaryArtifact({
    tops,
    transcript,
    assignments,
    speaker_names: speakerNames,
    summaries,
    summary_reviews: summaryReviews,
    skipped_assignment: skippedAssignment,
  }) && summariesAreFresh;
  const hasSummaryArtifactsInState = hasAnySummaryArtifact({
    tops,
    transcript,
    assignments,
    speaker_names: speakerNames,
    summaries,
    summary_reviews: summaryReviews,
    skipped_assignment: skippedAssignment,
  });

  const [jobId, setJobId] = useState<string | null>(null);
  const [rememberSpeakers, setRememberSpeakers] = useState<boolean>(() => {
    try {
      return localStorage.getItem(SPEAKER_MEMORY_OPT_IN_KEY) === "true";
    } catch {
      return false;
    }
  });

  // LLM Settings state
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [llmSettings, setLlmSettings] = useState<LLMSettings>(() => {
    // Load from localStorage on initial render
    try {
      const saved = localStorage.getItem(LLM_SETTINGS_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        return { model: typeof parsed.model === 'string' ? parsed.model.trim() : '',
          systemPrompt: typeof parsed.systemPrompt === 'string' ? parsed.systemPrompt : DEFAULT_LLM_SETTINGS.systemPrompt };
      }
    } catch (e) {
      console.error("Failed to load LLM settings from localStorage:", e);
    }
    return DEFAULT_LLM_SETTINGS;
  });

  const navigate = useCallback((path: string, replace = false) => {
    if (replace) {
      window.history.replaceState(null, "", path);
    } else {
      window.history.pushState(null, "", path);
    }
    setRoute(readRoute());
  }, []);

  useEffect(() => {
    const handlePopState = () => setRoute(readRoute());
    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  const buildSessionPayload = useCallback(
    (overrides: Partial<SessionSavePayload> = {}): SessionSavePayload => ({
      session_id: overrides.session_id ?? sessionId,
      revision: overrides.revision ?? sessionRevision,
      job_id: overrides.job_id ?? jobId,
      current_step: overrides.current_step ?? currentStep,
      tops: overrides.tops ?? tops,
      top_ids: overrides.top_ids ?? topIds,
      transcript: overrides.transcript ?? transcript,
      assignments: overrides.assignments ?? assignments,
      agenda_proposals: overrides.agenda_proposals === undefined ? agendaProposals : overrides.agenda_proposals,
      speaker_names: overrides.speaker_names ?? speakerNames,
      summaries: overrides.summaries ?? summaries,
      summary_reviews: overrides.summary_reviews ?? summaryReviews,
      summary_states: overrides.summary_states ?? effectiveSummaryStates,
      export_metadata: overrides.export_metadata ?? exportMetadata,
      skipped_assignment: overrides.skipped_assignment ?? skippedAssignment,
    }),
    [
      agendaProposals,
      assignments,
      currentStep,
      exportMetadata,
      jobId,
      sessionRevision,
      sessionId,
      skippedAssignment,
      speakerNames,
      summaryReviews,
      summaries,
      effectiveSummaryStates,
      topIds,
      tops,
      transcript,
    ]
  );
  const currentPayloadRef = useRef('');
  const latestPayloadRef = useRef(buildSessionPayload());
  const summaryBaselineRef = useRef<SessionSavePayload>(buildSessionPayload());
  useEffect(() => {
    latestPayloadRef.current = buildSessionPayload();
    currentPayloadRef.current = serializeSessionPayload(latestPayloadRef.current);
  }, [buildSessionPayload]);

  const applySession = useCallback((session: SessionResponse | SessionDraft) => {
    summaryBaselineRef.current = session;
    const nextSessionId = session.session_id ?? null;
    const nextRevision = session.revision ?? null;
    setSessionId(nextSessionId);
    activeSessionIdRef.current = nextSessionId;
    setSessionRevision(nextRevision);
    sessionRevisionRef.current = nextRevision;
    setSessionConflict(null);
    saveConflictRef.current = false;
    setJobId(session.job_id ?? null);
    setCurrentStep(session.current_step ?? 1);
    setTops(session.tops ?? EMPTY_TOPS);
    setTopIds(
      session.top_ids?.length === (session.tops?.length ?? 0)
        ? session.top_ids
        : (session.tops ?? []).map((_, index) =>
            normalizeSummaryStates(session.summary_states)[index]?.top_id ?? `top-${index}`
          )
    );
    setTranscript(session.transcript ?? []);
    setAssignments(session.assignments ?? []);
    setAgendaProposals(session.agenda_proposals ?? null);
    agendaRequestRef.current += 1;
    setIsDetectingAgenda(false);
    setAgendaDetectionError(null);
    setSpeakerNames(session.speaker_names ?? {});
    setSkipAgendaDetection(Boolean(session.skipped_assignment));
    setSummaries(normalizeSummaries(session.summaries));
    setSummaryReviews(normalizeSummaryReviews(session.summary_reviews));
    setSummaryStates(normalizeSummaryStates(session.summary_states));
    setExportMetadata({
      ...DEFAULT_EXPORT_METADATA,
      ...(session.export_metadata ?? {}),
      includeTranscript:
        session.export_metadata?.includeTranscript ??
        session.export_metadata?.includeTranscriptExcerpt ??
        DEFAULT_EXPORT_METADATA.includeTranscript,
      groupTranscriptByTop:
        session.export_metadata?.groupTranscriptByTop ??
        DEFAULT_EXPORT_METADATA.groupTranscriptByTop,
    });
    setSkippedAssignment(Boolean(session.skipped_assignment));
    setAudioUrl(withApiBase(session.audio_url));
    setAudioFile(null);
    setPdfFile(null);
    setPdfExtraction(null);
    setAutoDetectTopsFromPdf(false);
    setPipelineId((session as SessionDraft).pipeline_id ?? null);
    setPipelineJob("latest_pipeline" in session ? session.latest_pipeline ?? null : null);
    setPdfExtraction("pdf_extraction" in session ? session.pdf_extraction ?? null : null);
    setSummaryJob(
      "latest_summary_job" in session ? session.latest_summary_job ?? null : null
    );
    setPipelineNotice("latest_pipeline" in session && session.latest_pipeline?.status === 'failed'
      ? `Verarbeitung fehlgeschlagen (${session.latest_pipeline.stage}). ${session.latest_pipeline.error ?? ''} Gespeicherte Ergebnisse bleiben erhalten.` : null);
    setDirectProtocolAvailable(false);
    setIsProcessing(false);
    setIsGeneratingSummary(false);
    setProcessingError(null);
    setProcessingStatus("");
    setProcessingProgress(0);
  }, []);

  const applyPipelineResult = useCallback((result: PipelineResultResponse) => {
    const speakerObservations = result.speaker_observations ?? [];
    const suggestedSpeakerNames = applySuggestedSpeakerNames(result.session, speakerObservations);
    const speakerReplacements: Record<string, string> = {};
    Object.keys(suggestedSpeakerNames).forEach((speakerId) => {
      const previous = result.session.speaker_names?.[speakerId]?.trim() || speakerId;
      const next = suggestedSpeakerNames[speakerId]?.trim() || speakerId;
      if (previous !== next) speakerReplacements[previous] = next;
    });
    const sessionWithSuggestedSpeakers: SessionResponse = {
      ...result.session,
      speaker_names: suggestedSpeakerNames,
      summaries: replaceExactLabels(result.session.summaries, speakerReplacements),
      summary_reviews: replaceExactLabels(
        result.summary_reviews ?? result.session.summary_reviews,
        speakerReplacements
      ),
    };
    const completedPipeline = result.pipeline;
    const warnings = result.warnings ?? completedPipeline.warnings ?? [];
    const agendaDetectionResult = result.agenda_detection ?? null;
    const needsReview = hasReviewUncertainty(
      sessionWithSuggestedSpeakers,
      warnings,
      agendaDetectionResult
    );
    const needsSpeakerReview = hasUnverifiedSpeakers(
      sessionWithSuggestedSpeakers,
      speakerObservations
    );
    const shouldReviewBeforeProtocol = needsReview || needsSpeakerReview;

    applySession(sessionWithSuggestedSpeakers);

    setPipelineJob(completedPipeline);
    setPipelineId(null);
    setJobId(sessionWithSuggestedSpeakers.job_id ?? result.job?.job_id ?? completedPipeline.transcription_job_id ?? null);
    setAgendaProposals(sessionWithSuggestedSpeakers.agenda_proposals ??
      (agendaDetectionResult ? { version: 1, source: null, result: agendaDetectionResult } : null));
    setAgendaDetectionError(null);
    setIsProcessing(false);
    setProcessingProgress(100);
    setProcessingStatus("Pipeline abgeschlossen");
    setProcessingError(null);
    setDirectProtocolAvailable(false);
    setPipelineNotice(
      hasTechnicalProcessingFailure(sessionWithSuggestedSpeakers, agendaDetectionResult)
        ? INCOMPLETE_PIPELINE_NOTICE
        : shouldReviewBeforeProtocol
        ? "Automatische Verarbeitung beendet. Prüfen Sie Sprecher, Zuordnungen und Zusammenfassungen an den markierten Stellen."
        : "Automatische Verarbeitung abgeschlossen. Das Protokoll ist vorbereitet."
    );
    setCurrentStep(shouldReviewBeforeProtocol ? 2 : 3);

    try {
      localStorage.removeItem(ACTIVE_PIPELINE_KEY);
    } catch (e) {
      console.error("Failed to clear active pipeline id:", e);
    }
  }, [applySession]);

  const resetSession = useCallback(() => {
    if (saveTimerRef.current !== null) {
      window.clearTimeout(saveTimerRef.current);
      saveTimerRef.current = null;
    }
    agendaRequestRef.current += 1;
    setIsDetectingAgenda(false);
    setSessionId(null);
    activeSessionIdRef.current = null;
    setSessionRevision(null);
    sessionRevisionRef.current = null;
    setSessionConflict(null);
    saveConflictRef.current = false;
    setJobId(null);
    setPipelineId(null);
    setPipelineJob(null);
    setPipelineNotice(null);
    setDirectProtocolAvailable(false);
    setCurrentStep(1);
    setAudioFile(null);
    setPdfFile(null);
    setPdfExtraction(null);
    setTops(EMPTY_TOPS);
    setTopIds([]);
    setTranscript([]);
    setAssignments([]);
    setAgendaProposals(null);
    setAgendaDetectionError(null);
    setSummaries({});
    setSummaryReviews({});
    setSummaryStates({});
    setExportMetadata(DEFAULT_EXPORT_METADATA);
    setAudioUrl(null);
    setSpeakerNames({});
    setSkippedAssignment(false);
    setSkipAgendaDetection(false);
    setAutoDetectTopsFromPdf(true);
    setIsGeneratingSummary(false);
    setSummaryJob(null);
    setIsProcessing(false);
    setProcessingError(null);
    setProcessingStatus("");
    setProcessingProgress(0);
    setSessionMessage(null);
    setRestoreCandidate(null);
    lastSavedPayloadRef.current = null;
    try {
      localStorage.removeItem(ACTIVE_SESSION_KEY);
      localStorage.removeItem(ACTIVE_PIPELINE_KEY);
      localStorage.removeItem(SESSION_DRAFT_KEY);
    } catch (e) {
      console.error("Failed to clear session draft:", e);
    }
  }, []);

  // Check backend availability on mount
  useEffect(() => {
    checkBackendHealth().then(setBackendAvailable);
  }, []);

  // Detect restorable session on mount. Backend remains the source of truth.
  useEffect(() => {
    if (route.view !== "editor" || route.sessionId) {
      return;
    }
    try {
      const storedSessionId = localStorage.getItem(ACTIVE_SESSION_KEY);
      const storedPipelineId = localStorage.getItem(ACTIVE_PIPELINE_KEY);
      const storedDraft = localStorage.getItem(SESSION_DRAFT_KEY);
      const draft = storedDraft ? (JSON.parse(storedDraft) as SessionDraft) : null;

      if (storedSessionId || storedPipelineId || draft?.session_id || draft?.pipeline_id) {
        setRestoreCandidate({
          sessionId: storedSessionId || draft?.session_id || null,
          pipelineId: storedPipelineId || draft?.pipeline_id || null,
          draft,
        });
      }
    } catch (e) {
      console.error("Failed to read session draft:", e);
    }
  }, [route.sessionId, route.view]);

  // Save LLM settings to localStorage when they change
  useEffect(() => {
    try {
      localStorage.setItem(LLM_SETTINGS_KEY, JSON.stringify(llmSettings));
    } catch (e) {
      console.error("Failed to save LLM settings to localStorage:", e);
    }
  }, [llmSettings]);

  useEffect(() => {
    try {
      localStorage.setItem(SPEAKER_MEMORY_OPT_IN_KEY, String(rememberSpeakers));
    } catch (e) {
      console.error("Failed to save speaker memory setting:", e);
    }
  }, [rememberSpeakers]);

  useEffect(() => {
    if (saveTimerRef.current !== null) {
      window.clearTimeout(saveTimerRef.current);
    }

    if (
      !sessionId ||
      route.view === "history" ||
      isProcessing ||
      isGeneratingSummary ||
      (summaryJob && ['pending', 'processing', 'cancelling'].includes(summaryJob.status)) ||
      saveConflictRef.current
    ) {
      return;
    }

    const payload = buildSessionPayload();
    const draft: SessionDraft = {
      ...payload,
      session_id: sessionId,
      audio_url: audioUrl,
      pipeline_id: pipelineId,
    };

    try {
      localStorage.setItem(ACTIVE_SESSION_KEY, sessionId);
      if (pipelineId) {
        localStorage.setItem(ACTIVE_PIPELINE_KEY, pipelineId);
      } else {
        localStorage.removeItem(ACTIVE_PIPELINE_KEY);
      }
      localStorage.setItem(SESSION_DRAFT_KEY, JSON.stringify(draft));
    } catch (e) {
      console.error("Failed to save local session draft:", e);
    }

    const serializedPayload = serializeSessionPayload({
      ...payload,
      session_id: sessionId,
    });
    if (serializedPayload === lastSavedPayloadRef.current) {
      return;
    }

    saveTimerRef.current = window.setTimeout(() => {
      const targetSessionId = sessionId;
      saveChainRef.current = saveChainRef.current
        .then(async () => {
          if (saveConflictRef.current) return;
          const savedSession = await saveSession({
            ...payload,
            session_id: targetSessionId,
            revision: sessionRevisionRef.current,
          });
          if (activeSessionIdRef.current !== targetSessionId) return;
          summaryBaselineRef.current = savedSession;
          const nextRevision = savedSession.revision ?? sessionRevisionRef.current;
          sessionRevisionRef.current = nextRevision;
          setSessionRevision(nextRevision);
          setTopIds((current) => JSON.stringify(current) === JSON.stringify(payload.top_ids)
            ? savedSession.top_ids ?? current : current);
          // A delayed save must not put old positional summaries back after
          // TOP insertion/deletion, speaker renaming or further manual edits.
          if (currentPayloadRef.current === serializedPayload) {
            setSummaries(normalizeSummaries(savedSession.summaries));
            setSummaryReviews(normalizeSummaryReviews(savedSession.summary_reviews));
            setSummaryStates(normalizeSummaryStates(savedSession.summary_states));
          }
          lastSavedPayloadRef.current = serializedPayload;
          setSessionId(savedSession.session_id);
          activeSessionIdRef.current = savedSession.session_id;
          setSessionMessage(null);
        })
        .catch((error) => {
          if (activeSessionIdRef.current !== targetSessionId) return;
          const message =
            error instanceof Error ? error.message : "Sitzung konnte nicht gespeichert werden";
          if (error instanceof SessionConflictError) {
            saveConflictRef.current = true;
            setSessionConflict(message);
          }
          setSessionMessage(message);
          console.error("Failed to save session:", error);
        });
    }, 500);

    return () => {
      if (saveTimerRef.current !== null) {
        window.clearTimeout(saveTimerRef.current);
      }
    };
  }, [
    audioUrl,
    buildSessionPayload,
    isProcessing,
    isGeneratingSummary,
    summaryJob,
    pipelineId,
    route.view,
    sessionId,
    topIds,
  ]);

  const updatePipelineProcessingState = useCallback((status: PipelineJob) => {
    setPipelineJob(status);
    setPipelineId(status.pipeline_id);
    setJobId(status.transcription_job_id ?? null);
    if (status.session_id) {
      setSessionId(status.session_id);
      activeSessionIdRef.current = status.session_id;
    }
    setProcessingProgress(status.progress);
    setProcessingStatus("");
  }, []);

  const pollPipelineToResult = useCallback(
    async (activePipelineId: string, initialStatus?: PipelineJob) => {
      setIsProcessing(true);
      setProcessingError(null);
      if (initialStatus) {
        updatePipelineProcessingState(initialStatus);
      }

      const completedStatus =
        initialStatus?.status === "completed"
          ? initialStatus
          : await pollPipeline(activePipelineId, updatePipelineProcessingState);
      updatePipelineProcessingState(completedStatus);
      const result = await getPipelineResult(activePipelineId);
      applyPipelineResult(result);
    },
    [applyPipelineResult, updatePipelineProcessingState]
  );

  useEffect(() => {
    if (route.view !== "editor" || !route.sessionId) {
      setIsLoadingRouteSession(false);
      return;
    }
    if (activeSessionIdRef.current === route.sessionId) {
      setIsLoadingRouteSession(false);
      return;
    }

    let cancelled = false;
    setIsLoadingRouteSession(true);
    setSessionMessage(null);
    loadSession(route.sessionId)
      .then((restored) => {
        if (cancelled) return;
        applySession(restored);
        lastSavedPayloadRef.current = serializeSessionPayload({
          ...restored,
          transcript: restored.transcript ?? [],
        });
        const activePipeline = restored.latest_pipeline;
        if (
          activePipeline &&
          (activePipeline.status === "pending" || activePipeline.status === "processing")
        ) {
          void pollPipelineToResult(activePipeline.pipeline_id, activePipeline).catch((error) => {
            if (cancelled) return;
            const message =
              error instanceof Error ? error.message : "Pipeline konnte nicht fortgesetzt werden";
            setProcessingError(message);
            setProcessingStatus(`Fehler: ${message}`);
            setIsProcessing(true);
          });
        }
      })
      .catch((error) => {
        if (cancelled) return;
        const message =
          error instanceof Error ? error.message : "Sitzung konnte nicht geladen werden";
        setSessionMessage(message);
      })
      .finally(() => {
        if (!cancelled) setIsLoadingRouteSession(false);
      });

    return () => {
      cancelled = true;
    };
  }, [applySession, pollPipelineToResult, route.sessionId, route.view]);

  const activeSummaryJobId =
    summaryJob && ['pending', 'processing', 'cancelling'].includes(summaryJob.status)
      ? summaryJob.summary_job_id
      : null;

  const [summaryPollAttempt, setSummaryPollAttempt] = useState(0);
  useEffect(() => {
    if (!activeSummaryJobId || !sessionId) return;
    let cancelled = false;
    let retryTimer: number | undefined;
    const controller = new AbortController();
    setIsGeneratingSummary(true);
    const refresh = async () => {
      const refreshed = await loadSession(sessionId);
      if (cancelled || activeSessionIdRef.current !== sessionId) return;
      const merged = mergeSummarySession(latestPayloadRef.current, summaryBaselineRef.current, refreshed);
      summaryBaselineRef.current = refreshed;
      latestPayloadRef.current = merged;
      sessionRevisionRef.current = refreshed.revision ?? sessionRevisionRef.current;
      setSessionRevision(sessionRevisionRef.current);
      setTops(merged.tops);
      setTopIds(merged.top_ids ?? []);
      setTranscript(merged.transcript ?? []);
      setAssignments(merged.assignments);
      setSpeakerNames(merged.speaker_names);
      setAgendaProposals(merged.agenda_proposals ?? null);
      setSkippedAssignment(merged.skipped_assignment);
      if (merged.export_metadata) setExportMetadata(merged.export_metadata);
      setSummaries(normalizeSummaries(merged.summaries));
      setSummaryReviews(normalizeSummaryReviews(merged.summary_reviews));
      setSummaryStates(normalizeSummaryStates(merged.summary_states));
      setSessionMessage(null);
    };
    void pollSummaryJob(activeSummaryJobId, async (job) => {
      if (cancelled || !['pending', 'processing', 'cancelling'].includes(job.status)) return;
      await refresh();
      if (!cancelled) setSummaryJob(job);
    }, 1500, controller.signal)
      .then(async (completed) => {
        if (cancelled) return;
        // Keep the effect alive until the final results have been merged.
        await refresh();
        if (cancelled) return;
        setIsGeneratingSummary(false);
        setSummaryJob(completed);
      })
      .catch((error) => {
        if (cancelled) return;
        setSessionMessage(error instanceof Error ? error.message : 'Zusammenfassungsjob konnte nicht aktualisiert werden');
        retryTimer = window.setTimeout(() => setSummaryPollAttempt(value => value + 1), 3000);
      });
    return () => {
      cancelled = true;
      controller.abort();
      window.clearTimeout(retryTimer);
    };
  }, [activeSummaryJobId, sessionId, summaryPollAttempt]);

  const handleRestoreSession = async () => {
    if (!restoreCandidate) return;

    setIsRestoringSession(true);
    setSessionMessage(null);

    try {
      if (restoreCandidate.pipelineId) {
        const status = await getPipelineStatus(restoreCandidate.pipelineId);
        setRestoreCandidate(null);
        if (status.session_id) {
          setSessionId(status.session_id);
          activeSessionIdRef.current = status.session_id;
          localStorage.setItem(ACTIVE_SESSION_KEY, status.session_id);
        }
        setPipelineJob(status);
        if (status.status === "completed") {
          setPipelineId(status.pipeline_id);
          const result = await getPipelineResult(status.pipeline_id);
          applyPipelineResult(result);
        } else if (status.status === "failed" || status.status === "cancelled") {
          try {
            localStorage.removeItem(ACTIVE_PIPELINE_KEY);
          } catch (e) {
            console.error("Failed to clear active pipeline id:", e);
          }
          setPipelineId(null);
          setDirectProtocolAvailable(false);
          const inactiveMessage = status.error || "Pipeline ist nicht mehr aktiv";
          const fallbackSessionId = status.session_id ?? restoreCandidate.sessionId;
          if (fallbackSessionId) {
            try {
              const restored = await loadSession(fallbackSessionId);
              applySession(restored);
              setPipelineId(null);
              setSessionMessage(`${inactiveMessage}. Gespeicherte Sitzung wurde geladen.`);
            } catch (loadError) {
              if (restoreCandidate.draft) {
                applySession({ ...restoreCandidate.draft, pipeline_id: null });
                setPipelineId(null);
                setSessionMessage(`${inactiveMessage}. Lokaler Draft wurde geladen.`);
              } else {
                const message =
                  loadError instanceof Error ? loadError.message : inactiveMessage;
                setSessionMessage(message);
              }
            }
          } else if (restoreCandidate.draft) {
            applySession({ ...restoreCandidate.draft, pipeline_id: null });
            setPipelineId(null);
            setSessionMessage(`${inactiveMessage}. Lokaler Draft wurde geladen.`);
          } else {
            setSessionMessage(inactiveMessage);
          }
        } else {
          setPipelineId(status.pipeline_id);
          void pollPipelineToResult(status.pipeline_id, status).catch((error) => {
            const message =
              error instanceof Error ? error.message : "Pipeline konnte nicht fortgesetzt werden";
            setProcessingError(message);
            setProcessingStatus(`Fehler: ${message}`);
            setIsProcessing(true);
          });
        }
      } else if (restoreCandidate.sessionId) {
        const restored = await loadSession(restoreCandidate.sessionId);
        applySession(restored);
        if ((restored.current_step ?? 1) === 2 && hasAnySummaryArtifact(restored)) {
          const canGoDirect =
            hasFreshSessionSummaries(restored) && !hasReviewUncertainty(restored);
          setDirectProtocolAvailable(canGoDirect);
          setPipelineNotice(
            hasTechnicalProcessingFailure(restored) ? INCOMPLETE_PIPELINE_NOTICE : canGoDirect
              ? "Automatische Verarbeitung abgeschlossen. Sie können direkt zum Protokoll wechseln oder die Zuordnung prüfen."
              : "Automatische Verarbeitung abgeschlossen. Bitte prüfen Sie unsichere Zuordnungen vor dem Protokoll."
          );
        }
        lastSavedPayloadRef.current = serializeSessionPayload({
          ...restored,
          transcript: restored.transcript ?? [],
          summaries: normalizeSummaries(restored.summaries),
          summary_reviews: normalizeSummaryReviews(restored.summary_reviews),
        });
      } else if (restoreCandidate.draft) {
        applySession(restoreCandidate.draft);
        if ((restoreCandidate.draft.current_step ?? 1) === 2 && hasAnySummaryArtifact(restoreCandidate.draft)) {
          const draftSession = restoreCandidate.draft as SessionResponse;
          const canGoDirect =
            hasFreshSessionSummaries(restoreCandidate.draft) &&
            !hasReviewUncertainty(draftSession);
          setDirectProtocolAvailable(canGoDirect);
          setPipelineNotice(
            hasTechnicalProcessingFailure(draftSession) ? INCOMPLETE_PIPELINE_NOTICE : canGoDirect
              ? "Automatische Verarbeitung abgeschlossen. Sie können direkt zum Protokoll wechseln oder die Zuordnung prüfen."
              : "Automatische Verarbeitung abgeschlossen. Bitte prüfen Sie unsichere Zuordnungen vor dem Protokoll."
          );
        }
      }
      if (activeSessionIdRef.current) {
        navigate(
          `/sessions/${encodeURIComponent(activeSessionIdRef.current)}`,
          true
        );
      }
      setRestoreCandidate(null);
    } catch (error) {
      if (restoreCandidate.draft) {
        applySession(restoreCandidate.draft);
        setRestoreCandidate(null);
        setSessionMessage("Backend-Sitzung nicht verfügbar. Lokaler Draft wurde geladen.");
      } else {
        const message =
          error instanceof Error ? error.message : "Sitzung konnte nicht geladen werden";
        setSessionMessage(message);
      }
    } finally {
      setIsRestoringSession(false);
    }
  };

  // Start end-to-end pipeline via backend API
  const startPipeline = async () => {
    if (!audioFile) return;

    setIsProcessing(true);
    setProcessingProgress(0);
    setProcessingStatus("Audio wird hochgeladen und Pipeline gestartet...");
    setProcessingError(null);
    setPipelineNotice(null);
    setDirectProtocolAvailable(false);

    try {
      const submittedTops = skipAgendaDetection
        ? []
        : tops.map((top) => top.trim()).filter(Boolean);
      const shouldAutoDetectTopsFromPdf =
        (autoDetectTopsFromPdf || Boolean(pdfFile)) && submittedTops.length === 0;
      if (shouldAutoDetectTopsFromPdf && !skipAgendaDetection && !pdfFile) {
        throw new Error('PDF-Erkennung ist aktiviert. Bitte Einladung hochladen oder PDF-Erkennung ausschalten.');
      }
      const preparedSession = await saveSession(
        buildSessionPayload({
          current_step: 1,
          transcript: [],
          assignments: [],
          tops: submittedTops,
          summaries: {},
          summary_reviews: {},
          skipped_assignment: skipAgendaDetection,
        })
      );
      const activeSessionId = preparedSession.session_id;
      setSessionId(activeSessionId);
      activeSessionIdRef.current = activeSessionId;
      const preparedRevision = preparedSession.revision ?? null;
      setSessionRevision(preparedRevision);
      sessionRevisionRef.current = preparedRevision;
      navigate(`/sessions/${encodeURIComponent(activeSessionId)}`, true);
      localStorage.setItem(ACTIVE_SESSION_KEY, activeSessionId);

      const pipeline = await apiStartPipeline(audioFile, {
        sessionId: activeSessionId,
        tops: submittedTops,
        pdfFile,
        pdfSourceJobId: pdfExtraction?.document?.job_id,
        autoDetectTopsFromPdf: shouldAutoDetectTopsFromPdf,
        model: llmSettings.model,
        summarySystemPrompt: llmSettings.systemPrompt,
        rememberSpeakers,
        skipAgendaDetection,
      });
      updatePipelineProcessingState(pipeline);
      localStorage.setItem(ACTIVE_PIPELINE_KEY, pipeline.pipeline_id);
      await pollPipelineToResult(pipeline.pipeline_id, pipeline);
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : "Unbekannter Fehler";
      console.warn("Pipeline failed:", error);
      try {
        localStorage.removeItem(ACTIVE_PIPELINE_KEY);
      } catch (storageError) {
        console.error("Failed to clear active pipeline id:", storageError);
      }
      if (/abgebrochen|cancel/i.test(errorMessage)) {
        setProcessingError(errorMessage);
        setProcessingStatus(`Fehler: ${errorMessage}`);
        return;
      }
      setProcessingError(errorMessage);
      setProcessingStatus(`Fehler: ${errorMessage}`);
      setIsProcessing(true);
    }
  };

  const handleStartNewSession = () => {
    resetSession();
    navigate("/");
  };

  const handleOpenSession = (targetSessionId: string) => {
    resetSession();
    navigate(`/sessions/${encodeURIComponent(targetSessionId)}`);
  };

  const handleReloadCurrentSession = async () => {
    if (!sessionId) return;
    setIsLoadingRouteSession(true);
    try {
      const restored = await loadSession(sessionId);
      applySession(restored);
      setSessionMessage(null);
      lastSavedPayloadRef.current = serializeSessionPayload({
        ...restored,
        transcript: restored.transcript ?? [],
      });
    } catch (error) {
      setSessionMessage(
        error instanceof Error ? error.message : "Sitzung konnte nicht geladen werden"
      );
    } finally {
      setIsLoadingRouteSession(false);
    }
  };

  // Handle step navigation
  const handleStep1Next = () => {
    if (backendAvailable === false) {
      // Show error - backend not available
      alert("Backend nicht erreichbar.");
      return;
    }
    startPipeline();
  };

  const handleStep2Next = () => {
    setCurrentStep(3);
    setDirectProtocolAvailable(false);
    setPipelineNotice(null);
  };

  const handleStep2Back = () => {
    setDirectProtocolAvailable(false);
    setPipelineNotice(null);
    setCurrentStep(1);
    setTranscript([]);
    setAssignments([]);
    setAgendaProposals(null);
    setAgendaDetectionError(null);
    setProcessingError(null);
    setAudioUrl(null);
  };

  const handleStep3Back = () => {
    setDirectProtocolAvailable(false);
    setCurrentStep(2);
  };

  const handleRegenerateSummaries = async (topIndexes: number[]) => {
    if (!sessionId || activeSummaryJobId || isGeneratingSummary || !topIndexes.length) return;
    setIsGeneratingSummary(true);
    try {
      if (saveTimerRef.current !== null) {
        window.clearTimeout(saveTimerRef.current);
        saveTimerRef.current = null;
      }
      await saveChainRef.current;
      const persisted = await saveSession({
        ...buildSessionPayload({ current_step: 3 }),
        session_id: sessionId,
        revision: sessionRevisionRef.current,
      });
      const revision = persisted.revision ?? sessionRevisionRef.current;
      sessionRevisionRef.current = revision;
      setSessionRevision(revision);
      setSummaryStates(normalizeSummaryStates(persisted.summary_states));
      summaryBaselineRef.current = persisted;
      const selectedIds = [...new Set(topIndexes)].map(topIndex =>
        normalizeSummaryStates(persisted.summary_states)[topIndex]?.top_id
          ?? (persisted.top_ids ?? topIds)[topIndex]
          ?? `whole-session:${sessionId}`);
      const started = await startSummaryJob(sessionId, {
        revision,
        topIds: selectedIds,
        model: llmSettings.model,
        systemPrompt: tops.filter((top) => top.trim()).length === 0
          ? GENERIC_SUMMARY_PROMPT
          : llmSettings.systemPrompt,
      });
      setSummaryJob(started);
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : "Unbekannter Fehler";
      setSessionMessage(errorMessage);
      if (error instanceof SessionConflictError) setSessionConflict(errorMessage);
    } finally {
      setIsGeneratingSummary(false);
    }
  };

  const handleAcceptSummary = async (topIndex: number) => {
    if (!sessionId) return;
    setIsGeneratingSummary(true);
    try {
      if (saveTimerRef.current !== null) {
        window.clearTimeout(saveTimerRef.current);
        saveTimerRef.current = null;
      }
      await saveChainRef.current;
      const persisted = await saveSession({
        ...buildSessionPayload({ current_step: 3 }),
        session_id: sessionId,
        revision: sessionRevisionRef.current,
      });
      const revision = persisted.revision ?? sessionRevisionRef.current;
      const topId = normalizeSummaryStates(persisted.summary_states)[topIndex]?.top_id
        ?? (persisted.top_ids ?? topIds)[topIndex]
        ?? `whole-session:${sessionId}`;
      const accepted = await acceptExistingSummary(sessionId, topId, revision);
      applySession(accepted);
      setCurrentStep(3);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Zusammenfassung konnte nicht übernommen werden';
      setSessionMessage(message);
      if (error instanceof SessionConflictError) setSessionConflict(message);
    } finally {
      setIsGeneratingSummary(false);
    }
  };

  const handleCancelSummaryJob = async () => {
    if (!summaryJob) return;
    try {
      const cancelled = await cancelSummaryJob(summaryJob.summary_job_id);
      if (['pending', 'processing', 'cancelling'].includes(cancelled.status)) setSummaryJob(cancelled);
    } catch (error) {
      setSessionMessage(error instanceof Error ? error.message : 'Abbruch fehlgeschlagen');
    }
  };

  const handleRetry = () => {
    setProcessingError(null);
    setIsProcessing(false);
  };

  const handleDirectProtocol = () => {
    if (!hasFreshSummariesInState) {
      setDirectProtocolAvailable(false);
      setPipelineNotice("Bitte prüfen Sie die markierten TOP-Zusammenfassungen vor dem Export.");
      return;
    }
    setCurrentStep(3);
    setDirectProtocolAvailable(false);
    setPipelineNotice(null);
  };

  const handleDetectAgenda = async (fresh = false) => {
    agendaAbortRef.current?.abort();
    const controller = new AbortController();
    agendaAbortRef.current = controller;
    const requestId = ++agendaRequestRef.current;
    const inputEpoch = agendaInputEpochRef.current;
    const source = agendaSource(tops, topIds, transcript);
    setIsDetectingAgenda(true);
    setAgendaDetectionError(null);
    try {
      const result = await detectAgenda({
        tops, transcript, model: llmSettings.model,
        fresh, topIds, signal: controller.signal,
        onStatus: job => setAgendaJobPhase(job.state === 'queued' ? 'Wartet auf Verarbeitung' : job.state === 'retry_wait' ? 'Vorübergehend gestört; erneuter Versuch folgt' : (job.progress?.agenda_phase ?? job.progress?.phase)?.startsWith('independent') ? 'Unabhängige Quellenprüfung läuft …' : (job.progress?.agenda_phase ?? job.progress?.phase)?.startsWith('resolve') ? 'Abweichungen werden geklärt …' : job.progress?.phase === 'loading' ? 'Modell lädt / wartet auf erste Ausgabe' : 'Sitzungsverlauf und Zuordnungen werden ermittelt …'),
        preserveTranscriptStructure: true,
      });
      if (requestId !== agendaRequestRef.current) return;
      if (inputEpoch !== agendaInputEpochRef.current) {
        setAgendaDetectionError("TOPs oder Transkript wurden während der Erkennung geändert. Bitte erneut berechnen");
        return;
      }
      // Models may append independently reviewed additions, preserving every
      // existing title, identity and manual line assignment.
      if (result.tops.length < tops.length || tops.some((title, index) => result.tops[index] !== title)) {
        throw new Error("Das Ergebnis verändert vorhandene Tagesordnungspunkte");
      }
      const nextTopIds = [...topIds, ...result.tops.slice(tops.length).map((_, offset) => {
        const item = result.llm?.provenance?.identities?.find(top => top.top_index === tops.length + offset);
        return item?.top_uid ?? item?.top_id ?? crypto.randomUUID();
      })];
      const proposals: AgendaProposals = { version: 1, source: { ...source,
        tops: result.tops, top_ids: nextTopIds,
        pdf_extraction: agendaProposals?.source?.pdf_extraction ?? pdfExtraction }, result };
      if (!proposalsAreValid(proposals, result.tops, nextTopIds, transcript)) {
        throw new Error("Das Ergebnis passt nicht zum aktuellen Transkript oder zur Tagesordnung");
      }
      if (result.tops.length > tops.length) {
        setTops(result.tops);
        setTopIds(nextTopIds);
      }
      // Even manual assignments made while awaiting detection remain untouched.
      setAgendaProposals(proposals);
    } catch (error) {
      if (requestId === agendaRequestRef.current) {
        setAgendaDetectionError(error instanceof Error ? error.message : "Erkennung fehlgeschlagen");
      }
    } finally {
      if (requestId === agendaRequestRef.current) setIsDetectingAgenda(false);
    }
  };

  const handleTranscriptStructureChange = () => {
    setAgendaDetectionError(null);
    setDirectProtocolAvailable(false);
    setPipelineNotice(null);
  };

  const handleSpeakerNamesChange: Dispatch<SetStateAction<Record<string, string>>> = (action) => {
    setSpeakerNames((current) => {
      const next = typeof action === 'function' ? action(current) : action;
      const replacements: Record<string, string> = {};
      for (const speakerId of new Set([...Object.keys(current), ...Object.keys(next)])) {
        const oldLabel = current[speakerId]?.trim() || speakerId;
        const newLabel = next[speakerId]?.trim() || speakerId;
        if (oldLabel !== newLabel) replacements[oldLabel] = newLabel;
      }
      if (Object.keys(replacements).length > 0) {
        setSummaries((values) => replaceExactLabels(values, replacements));
        setSummaryReviews((values) => replaceExactLabels(values, replacements));
      }
      return next;
    });
  };

  const handleTopsChange = (nextTops: string[], nextTopIds: string[]) => {
    const previousIds = topIds;
    const remap = <T,>(values: Record<number, T>): Record<number, T> => {
      const next: Record<number, T> = {};
      nextTopIds.forEach((topId, nextIndex) => {
        const previousIndex = previousIds.indexOf(topId);
        if (previousIndex >= 0 && values[previousIndex] !== undefined) {
          next[nextIndex] = values[previousIndex]!;
        }
      });
      return next;
    };
    setSummaries((values) => remap(values));
    setSummaryReviews((values) => remap(values));
    setSummaryStates((values) => remap(values));
    setTopIds(nextTopIds);
    setTops(nextTops);
  };

  const handleCancelPipeline = async () => {
    if (!pipelineId) return;

    try {
      const cancelled = await cancelPipeline(pipelineId);
      setPipelineJob(cancelled);
      setProcessingProgress(cancelled.progress);
      setProcessingStatus("Pipeline abgebrochen");
      setProcessingError("Pipeline abgebrochen");
      setIsProcessing(false);
      setPipelineId(null);
      localStorage.removeItem(ACTIVE_PIPELINE_KEY);
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Pipeline konnte nicht abgebrochen werden";
      setProcessingError(message);
      setProcessingStatus(`Fehler: ${message}`);
    }
  };

  if (route.view === "history") {
    return (
      <Layout
        onSettingsClick={() => setIsSettingsOpen(true)}
        onHistoryClick={() => navigate("/sessions")}
        onNewSessionClick={handleStartNewSession}
      >
        <LLMSettingsPanel
          isOpen={isSettingsOpen}
          onClose={() => setIsSettingsOpen(false)}
          settings={llmSettings}
          onSettingsChange={setLlmSettings}
        />
        <SessionHistory
          onOpen={handleOpenSession}
          onNewSession={handleStartNewSession}
        />
      </Layout>
    );
  }

  if (isLoadingRouteSession) {
    return (
      <Layout
        onSettingsClick={() => setIsSettingsOpen(true)}
        onHistoryClick={() => navigate("/sessions")}
        onNewSessionClick={handleStartNewSession}
      >
        <div className="rounded-lg border border-gray-200 bg-white p-8 text-center text-gray-600">
          Sitzung wird geladen…
        </div>
      </Layout>
    );
  }

  // Filter out empty TOPs for display
  const validTops = tops.filter((t) => t.trim() !== "");

  return (
    <Layout
      wide={!isProcessing && currentStep > 1}
      onSettingsClick={() => setIsSettingsOpen(true)}
      onHistoryClick={() => navigate("/sessions")}
      onNewSessionClick={handleStartNewSession}
    >
      {/* LLM Settings Panel */}
      <LLMSettingsPanel
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        settings={llmSettings}
        onSettingsChange={setLlmSettings}
      />

      {sessionConflict && (
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3 rounded-lg border border-yellow-300 bg-yellow-50 p-4 text-sm text-yellow-900">
          <span>{sessionConflict}</span>
          <button
            type="button"
            onClick={() => void handleReloadCurrentSession()}
            className="rounded-md bg-yellow-900 px-3 py-2 font-medium text-white hover:bg-yellow-800"
          >
            Aktuellen Stand laden
          </button>
        </div>
      )}

      {/* Backend status indicator */}
      {backendAvailable === false && (
        <div className="mb-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg text-sm text-yellow-800">
          Backend nicht erreichbar.
        </div>
      )}

      {(restoreCandidate || sessionMessage) && (
        <div className="mb-4 bg-white border border-gray-200 rounded-lg p-3 flex flex-wrap items-center justify-between gap-3">
          <div>
            <div className="text-sm font-medium text-gray-800">
              {sessionId ? "Sitzung aktiv" : "Letzte Sitzung verfügbar"}
            </div>
            {sessionMessage && (
              <div className="text-sm text-yellow-700 mt-1">{sessionMessage}</div>
            )}
          </div>
          <div className="flex gap-2">
            {restoreCandidate && (
              <button
                type="button"
                onClick={handleRestoreSession}
                disabled={isRestoringSession}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
              >
                {isRestoringSession ? "Wird geladen..." : "Letzte Sitzung fortsetzen"}
              </button>
            )}
            <button
              type="button"
              onClick={handleStartNewSession}
              className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200"
            >
              Neue Sitzung starten
            </button>
          </div>
        </div>
      )}

      {!isProcessing && <PdfSources result={agendaProposals?.source?.pdf_extraction ?? pdfExtraction} />}

      {!isProcessing && <StepIndicator currentStep={currentStep} />}

      {!isProcessing && pipelineNotice && (
        <div className="mb-4 rounded-lg border border-blue-200 bg-blue-50 p-3 text-sm text-blue-900">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <span>{pipelineNotice}</span>
            {directProtocolAvailable && hasFreshSummariesInState &&
              agendaDetection && !agendaDetectionStale && !hasAgendaUncertainty(agendaDetection) && (
              <button
                type="button"
                onClick={handleDirectProtocol}
                className="rounded-lg bg-blue-600 px-4 py-2 font-medium text-white hover:bg-blue-700"
              >
                Direkt zum Protokoll
              </button>
            )}
          </div>
        </div>
      )}

      {isProcessing ? (
        <div>
          <ProcessingStep
            progress={processingProgress}
            status={processingStatus}
            pipeline={pipelineJob}
            canCancel={Boolean(pipelineId)}
            onCancel={handleCancelPipeline}
          />
          {processingError && (
            <div className="mt-4 text-center">
              <p className="text-red-600 mb-4">{processingError}</p>
              <button
                onClick={handleRetry}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
              >
                Erneut versuchen
              </button>
            </div>
          )}
        </div>
      ) : currentStep === 1 ? (
        <UploadStep
          onNext={handleStep1Next}
          audioFile={audioFile}
          setAudioFile={setAudioFile}
          pdfFile={pdfFile}
          setPdfFile={(file) => { setPdfFile(file); setPdfExtraction(null); }}
          onPdfExtracted={setPdfExtraction}
          tops={tops}
          setTops={setTops}
          llmSettings={llmSettings}
          rememberSpeakers={rememberSpeakers}
          setRememberSpeakers={setRememberSpeakers}
          skipAgendaDetection={skipAgendaDetection}
          setSkipAgendaDetection={setSkipAgendaDetection}
          autoDetectTopsFromPdf={autoDetectTopsFromPdf}
          setAutoDetectTopsFromPdf={setAutoDetectTopsFromPdf}
          exportMetadata={exportMetadata}
          setExportMetadata={setExportMetadata}
        />
      ) : currentStep === 2 ? (
        <AssignmentStep
          onNext={handleStep2Next}
          onBack={handleStep2Back}
          tops={validTops}
          setTops={setTops}
          topIds={topIds}
          onTopsChange={handleTopsChange}
          transcript={transcript}
          setTranscript={setTranscript}
          assignments={assignments}
          setAssignments={setAssignments}
          agendaDetection={agendaDetection}
          agendaDetectionError={agendaDetectionError}
          agendaDetectionStale={agendaDetectionStale}
          isDetectingAgenda={isDetectingAgenda}
          onDetectAgenda={handleDetectAgenda}
          onCancelAgenda={() => agendaAbortRef.current?.abort()}
          agendaJobPhase={agendaJobPhase}
          onTranscriptStructureChange={handleTranscriptStructureChange}
          audioUrl={audioUrl ?? undefined}
          speakerNames={speakerNames}
          setSpeakerNames={handleSpeakerNamesChange}
          sessionId={sessionId}
          hasSummaries={hasSummaryArtifactsInState}
          rememberSpeakers={rememberSpeakers}
        />
      ) : (
        <SummaryStep
          sessionId={sessionId}
          pipelineIncomplete={!!pipelineJob && pipelineJob.status !== 'completed'}
          onBack={handleStep3Back}
          tops={validTops}
          transcript={transcript}
          assignments={assignments}
          summaries={summaries}
          setSummaries={(next) => {
            const reviews = { ...summaryReviews };
            const states = { ...effectiveSummaryStates };
            Object.keys(next).map(Number).forEach(index => {
              if (next[index] === summaries[index]) return;
              delete reviews[index]; // Generated evidence no longer describes manual text.
              if (states[index]) states[index] = { ...states[index]!, status: 'ready', origin: 'manual' };
            });
            setSummaries(next);
            setSummaryReviews(reviews);
            setSummaryStates(states);
          }}
          summaryReviews={summaryReviews}
          summaryStates={effectiveSummaryStates}
          onRegenerateSummary={(index) => handleRegenerateSummaries([index])}
          onRegenerateSummaries={handleRegenerateSummaries}
          topIds={topIds}
          onAcceptSummary={handleAcceptSummary}
          summaryJob={summaryJob}
          onCancelSummaryJob={handleCancelSummaryJob}
          isGenerating={isGeneratingSummary}
          summariesAreFresh={summariesAreFresh}
          audioUrl={audioUrl ?? undefined}
          speakerNames={speakerNames}
          exportMetadata={exportMetadata}
          setExportMetadata={setExportMetadata}
        />
      )}
    </Layout>
  );
}
