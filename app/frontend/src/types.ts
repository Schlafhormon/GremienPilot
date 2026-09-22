import type { Dispatch, SetStateAction } from 'react';

/**
 * Shared type definitions for GremienPilot
 */

// API Types
export interface TranscriptLine {
  line_id?: string;
  speaker: string;
  text: string;
  start: number;  // Start time in seconds
  end: number;    // End time in seconds
  timing?: {
    source: string;
    words: { text: string; char_start: number; char_end: number; start: number; end: number }[];
    segments: { segment_id: string; start: number; end: number }[];
  } | null;
}

export interface AudioMetadata {
  filename?: string | null;
  content_type?: string | null;
  size_bytes?: number | null;
}

export interface SpeakerProfile {
  profile_id: string;
  display_name: string;
  scope?: string | null;
  created_at: number;
  updated_at: number;
  archived_at?: number | null;
  archived: boolean;
  embedding_count: number;
}

export interface SpeakerObservation {
  observation_id: number;
  job_id: string;
  session_id: string;
  local_speaker_id: string;
  local_display_name: string;
  profile_id?: string | null;
  profile_display_name?: string | null;
  profile?: SpeakerProfile | null;
  confidence?: number | null;
  status: 'suggested' | 'confirmed' | 'rejected' | 'manual' | string;
  display_name: string;
  embedding_warning?: string | null;
  created_at: number;
  updated_at: number;
}

export interface SpeakerEmbeddingBackfillResult {
  scanned_observation_count: number;
  processed_job_count: number;
  saved_embedding_count: number;
  skipped_count: number;
  errors: string[];
}

export interface SpeakerMatchDiagnostic {
  local_speaker_id: string;
  reason_code: string;
  reason: string;
  best_profile_id?: string | null;
  best_profile_display_name?: string | null;
  best_score?: number | null;
  suggest_threshold?: number | null;
  local_audio_seconds?: number | null;
  local_embedding_available: boolean;
  profile_embedding_count: number;
}

export interface SpeakerSuggestion {
  observation_id: number;
  local_speaker_id: string;
  profile_id: string;
  profile_display_name: string;
  confidence: number;
  status: string;
}

export interface TranscriptionJob {
  job_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  message: string;
  transcript?: TranscriptLine[];
  speaker_suggestions?: SpeakerSuggestion[];
  audio_url?: string;  // URL to stream audio for playback
  audio_metadata?: AudioMetadata | null;
  error?: string;
}

export type PipelineStatus = 'pending' | 'processing' | 'completed' | 'failed' | 'cancelled';

export type PipelineStage =
  | 'upload'
  | 'transcribe'
  | 'speaker_match'
  | 'agenda_detect'
  | 'summarize'
  | 'ready_for_review'
  | string;

export interface PipelineJob {
  pipeline_id: string;
  session_id?: string | null;
  transcription_job_id?: string | null;
  status: PipelineStatus;
  stage: PipelineStage;
  progress: number;
  warnings: string[];
  error?: string | null;
  created_at?: number | null;
  updated_at?: number | null;
}

export interface PipelineStartOptions {
  agendaFresh?: boolean;
  sessionId?: string | null;
  tops?: string[];
  pdfFile?: File | null;
  autoDetectTopsFromPdf?: boolean;
  model?: string;
  /** Legacy alias for summarySystemPrompt; used only for summaries. */
  systemPrompt?: string;
  summarySystemPrompt?: string;
  agendaSystemPrompt?: string;
  agendaUseLlm?: boolean | null;
  pdfSystemPrompt?: string;
  rememberSpeakers?: boolean;
  skipAgendaDetection?: boolean;
}

export interface PipelineResultResponse {
  pipeline: PipelineJob;
  session: SessionResponse;
  job?: TranscriptionJob | null;
  speaker_observations?: SpeakerObservation[];
  summary_reviews?: Record<number, SummaryReview> | Record<string, SummaryReview>;
  warnings: string[];
  agenda_detection?: AgendaDetectionResponse | null;
}

export interface SessionSavePayload {
  agenda_proposals?: AgendaProposals | null;
  session_id?: string | null;
  revision?: number | null;
  job_id?: string | null;
  current_step?: number | null;
  tops: string[];
  top_ids?: string[];
  transcript?: TranscriptLine[];
  assignments: (number | null)[];
  speaker_names: Record<string, string>;
  summaries: Record<number, string>;
  summary_reviews?: Record<number, SummaryReview>;
  summary_states?: Record<number, SummaryState>;
  export_metadata?: ExportMetadata;
  skipped_assignment: boolean;
}

export interface SessionResponse extends SessionSavePayload {
  session_id: string;
  revision?: number;
  created_at?: number | null;
  updated_at?: number | null;
  audio_url?: string | null;
  audio_metadata?: AudioMetadata | null;
  job?: TranscriptionJob | null;
  latest_pipeline?: PipelineJob | null;
  latest_summary_job?: SummaryJob | null;
}

export type SessionHistoryStatus =
  | 'draft'
  | 'processing'
  | 'review'
  | 'ready'
  | 'failed'
  | 'cancelled';

export interface SessionHistoryItem {
  session_id: string;
  title: string;
  committee: string;
  meeting_date: string;
  status: SessionHistoryStatus;
  current_step?: number | null;
  revision: number;
  created_at: number;
  updated_at: number;
  top_count: number;
  transcript_line_count: number;
  summary_count: number;
  audio_available: boolean;
  job_id?: string | null;
  job_status?: string | null;
  pipeline_job_id?: string | null;
  pipeline_status?: string | null;
  pipeline_stage?: string | null;
  pipeline_progress?: number | null;
}

export interface SessionHistoryResponse {
  items: SessionHistoryItem[];
  total: number;
  limit: number;
  offset: number;
}

export interface SummarizeRequest {
  top_title: string;
  lines: TranscriptLine[];
}

export interface StructuredSummary {
  discussion: string[];
  decisions: string[];
  votes: string[];
  action_items: string[];
  open_points: string[];
  uncertainties: string[];
}

export interface SummarySourceLink {
  section: keyof StructuredSummary | string;
  item_index: number;
  item_text: string;
  line_indices: number[];
  start?: number | null;
  end?: number | null;
  excerpt: string;
  confidence: number;
  missing_source: boolean;
}

export interface SummaryReviewWarning {
  kind: 'missing_source' | 'missing_decision_signal' | string;
  message: string;
  severity: 'info' | 'warning' | 'error' | string;
  keyword?: string | null;
  section?: keyof StructuredSummary | string | null;
  item_index?: number | null;
  line_indices: number[];
  start?: number | null;
  end?: number | null;
  excerpt: string;
}

export interface SummaryReview {
  structured?: StructuredSummary | null;
  source_links: SummarySourceLink[];
  review_warnings: SummaryReviewWarning[];
  fallback_used?: boolean;
  chunks_processed?: number;
  duration_seconds?: number;
  llm_usage?: Record<string, unknown>;
}

export type SummaryStatus =
  | 'ready'
  | 'review_required'
  | 'missing'
  | 'queued'
  | 'running'
  | 'failed';

export interface SummarySourceSnapshotLine {
  line_id: string;
  speaker: string;
  text: string;
}

export interface SummaryState {
  top_id: string;
  status: SummaryStatus | string;
  input_hash?: string;
  current_input_hash?: string;
  source_snapshot?: SummarySourceSnapshotLine[];
  change_reasons?: string[];
  origin?: string;
  generated_at?: number;
  accepted_at?: number;
  updated_at?: number;
}

export interface SummaryJob {
  summary_job_id: string;
  session_id: string;
  status: 'pending' | 'processing' | 'cancelling' | 'completed' | 'failed' | 'cancelled' | string;
  progress: number;
  current_top: number;
  total_tops: number;
  top_ids: string[];
  completed_tops?: number;
  processed_tops?: number;
  current_top_id?: string | null;
  outcomes?: Record<string, { status: 'completed' | 'failed'; error?: string }>;
  error?: string | null;
  created_at?: number | null;
  updated_at?: number | null;
}

export interface SummarizeResponse {
  summary: string;
  duration_seconds?: number;
  structured?: StructuredSummary | null;
  source_links?: SummarySourceLink[];
  review_warnings?: SummaryReviewWarning[];
  fallback_used?: boolean;
  chunks_processed?: number;
}

export interface AssignmentSuggestionSegment {
  top_index: number;
  top_title: string;
  start_index: number;
  end_index: number;
  confidence: number;
  uncertain: boolean;
  transition_type: 'explicit' | 'keyword' | 'inferred' | string;
  reason: string;
  evidence_index?: number | null;
  evidence_text?: string | null;
}

export interface AssignmentSuggestionsResponse {
  suggested_assignments: (number | null)[];
  segments: AssignmentSuggestionSegment[];
  strategy: string;
  uncertain_count: number;
}

export interface AgendaDetectionRequest {
  topIds?: string[];
  fresh?: boolean;
  cacheNamespace?: string;
  preserveTranscriptStructure?: boolean;
  useLlm?: boolean | null;
  tops?: string[];
  transcript: TranscriptLine[];
  model?: string;
  systemPrompt?: string;
}

export interface AgendaLLMUsage {
  provenance?: { model?: string; digest?: string; cache_namespace?: string; prompt_version?: string;
    context_tokens?: number; timeline_identity?: string };
  enabled: boolean;
  source: "server_default" | "request";
  timeout_seconds: number;
  status: "disabled" | "skipped" | "success" | "fallback" | "partial_fallback" | "failed" | "partial_failure";
  attempted_calls: number;
  failed_calls: number;
  failure_reasons: string[];
  validation_reasons?: string[];
  processed_lines?: number[];
  gaps?: { start_index: number; end_index: number; kind: "semantic" | "technical"; reason: string }[];
  chunks?: { start_index: number; end_index: number; status: string; duration_seconds: number }[];
}

export interface AgendaDetectionResponse {
  llm?: AgendaLLMUsage | null;
  warnings?: string[];
  tops: string[];
  transcript?: TranscriptLine[];
  assignments: (number | null)[];
  segments: AssignmentSuggestionSegment[];
  strategy: string;
  uncertain_count: number;
}

// Positional results are usable only against this exact, identity-bound input.
export interface AgendaProposals {
  version: 1;
  source: { tops: string[]; top_ids: string[]; transcript: TranscriptLine[] } | null;
  result: AgendaDetectionResponse;
}

export interface ExportMetadata {
  committee: string;
  date: string;
  location: string;
  title: string;
  participants: string[];
  includeSpeakerList: boolean;
  includeTranscript: boolean;
  includeTranscriptExcerpt?: boolean;
  groupTranscriptByTop: boolean;
  includeGenerationNote: boolean;
}

export interface PdfAgendaMetadata {
  committee?: string;
  date?: string;
  location?: string;
  title?: string;
}

export interface PdfAgendaExtractionResult {
  tops: string[];
  metadata: PdfAgendaMetadata;
}

export type ExportFormat = 'txt' | 'docx' | 'pdf';

// Component Props Types
export interface LayoutProps {
  children: React.ReactNode;
  wide?: boolean;
  onSettingsClick?: () => void;
  onHistoryClick?: () => void;
  onNewSessionClick?: () => void;
}

export interface StepIndicatorProps {
  currentStep: number;
}

export interface LLMSettings {
  model: string;
  systemPrompt: string;
}

export interface UploadStepProps {
  onNext: () => void;
  audioFile: File | null;
  setAudioFile: (file: File | null) => void;
  pdfFile: File | null;
  setPdfFile: (file: File | null) => void;
  tops: string[];
  setTops: (tops: string[]) => void;
  llmSettings?: LLMSettings;
  rememberSpeakers: boolean;
  setRememberSpeakers: (enabled: boolean) => void;
  skipAgendaDetection: boolean;
  setSkipAgendaDetection: (enabled: boolean) => void;
  autoDetectTopsFromPdf: boolean;
  setAutoDetectTopsFromPdf: (enabled: boolean) => void;
  exportMetadata: ExportMetadata;
  setExportMetadata: (metadata: ExportMetadata) => void;
}

export interface ProcessingStepProps {
  progress: number;
  status: string;
  pipeline?: PipelineJob | null;
  canCancel?: boolean;
  onCancel?: () => void;
}

export interface AssignmentStepProps {
  onNext: () => void;
  onBack: () => void;
  tops: string[];
  setTops: (tops: string[]) => void;
  topIds?: string[];
  onTopsChange?: (tops: string[], topIds: string[]) => void;
  transcript: TranscriptLine[];
  setTranscript: (transcript: TranscriptLine[]) => void;
  assignments: (number | null)[];
  setAssignments: (assignments: (number | null)[]) => void;
  agendaDetection?: AgendaDetectionResponse | null;
  agendaDetectionError?: string | null;
  agendaDetectionStale?: boolean;
  isDetectingAgenda?: boolean;
  onDetectAgenda?: (fresh?: boolean) => void;
  onTranscriptStructureChange?: () => void;
  audioUrl?: string;  // URL to stream audio for playback
  speakerNames: Record<string, string>;
  setSpeakerNames: Dispatch<SetStateAction<Record<string, string>>>;
  sessionId?: string | null;
  hasSummaries?: boolean;
  rememberSpeakers?: boolean;
}

export interface SummaryStepProps {
  onBack: () => void;
  tops: string[];
  transcript: TranscriptLine[];
  assignments: (number | null)[];
  summaries: Record<number, string>;
  setSummaries: (summaries: Record<number, string>) => void;
  summaryReviews?: Record<number, SummaryReview>;
  summaryStates?: Record<number, SummaryState>;
  onRegenerateSummary: (topIndex: number) => Promise<void>;
  onRegenerateSummaries?: (topIndexes: number[]) => Promise<void>;
  topIds?: string[];
  onAcceptSummary: (topIndex: number) => Promise<void>;
  summaryJob?: SummaryJob | null;
  onCancelSummaryJob?: () => Promise<void>;
  isGenerating: boolean;
  summariesAreFresh: boolean;
  audioUrl?: string;  // URL to stream audio for playback
  speakerNames: Record<string, string>;
  exportMetadata: ExportMetadata;
  setExportMetadata: (metadata: ExportMetadata) => void;
}

// Color palette type for TOPs
export interface TopColor {
  bg: string;
  border: string;
  text: string;
  dot: string;
}
export interface AgendaModelSettings {
  enabled: boolean;
  model: string;
  context_tokens: number;
  output_tokens: number;
  timeline_output_tokens: number;
  timeout_seconds: number;
  cpu_threads: number;
  temperature: number;
  seed: number | null;
  thinking: boolean;
}

export interface AgendaModelDiagnostics {
  settings: AgendaModelSettings;
  effective_model: string;
  summary_model: string;
  available: boolean | null;
  digest: string | null;
  message: string;
  last_error: { model: string; message: string } | null;
}
