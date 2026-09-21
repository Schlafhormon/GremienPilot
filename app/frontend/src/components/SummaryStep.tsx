import { useState, useRef, useEffect, useCallback, type ChangeEvent } from 'react';
import { exportProtocol } from '../api';
import type { ExportFormat, ExportMetadata, StructuredSummary, SummaryStepProps, TranscriptLine } from '../types';
import AudioPlayer from './AudioPlayer';
import { useAudioSync } from '../hooks/useAudioSync';

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

const SUMMARY_SECTIONS: Array<{
  key: keyof StructuredSummary;
  label: string;
  important?: boolean;
}> = [
  { key: 'discussion', label: 'Diskussion' },
  { key: 'decisions', label: 'Beschluss', important: true },
  { key: 'votes', label: 'Abstimmung', important: true },
  { key: 'action_items', label: 'Maßnahmen', important: true },
  { key: 'open_points', label: 'Offene Punkte' },
  { key: 'uncertainties', label: 'Unsicherheiten' },
];

const CHANGE_REASON_LABELS: Record<string, string> = {
  lines_added: 'Zeilen wurden hinzugefügt',
  lines_removed: 'Zeilen wurden entfernt',
  transcript_changed: 'Transkriptinhalt wurde geändert',
  line_order_changed: 'Reihenfolge wurde geändert',
  changed_while_queued: 'Inhalt wurde während der Wartezeit geändert',
  changed_during_generation: 'Inhalt wurde während der Generierung geändert',
  summary_input_changed: 'Inhalt oder TOP-Zuordnung wurde geändert',
};

export default function SummaryStep({
  onBack,
  tops,
  transcript,
  assignments,
  summaries,
  setSummaries,
  summaryReviews = {},
  summaryStates = {},
  onRegenerateSummary,
  onRegenerateSummaries,
  topIds = [],
  onAcceptSummary,
  summaryJob,
  onCancelSummaryJob,
  isGenerating,
  summariesAreFresh,
  audioUrl,
  speakerNames,
  exportMetadata,
  setExportMetadata,
}: SummaryStepProps) {
  const [selectedTop, setSelectedTop] = useState(0);
  const [editingTop, setEditingTop] = useState<number | null>(null);
  const [editText, setEditText] = useState('');
  const [copied, setCopied] = useState(false);
  const [activeSourceLine, setActiveSourceLine] = useState<number | null>(null);
  const [exportError, setExportError] = useState<string | null>(null);
  const [exportingFormat, setExportingFormat] = useState<ExportFormat | null>(null);
  const [acceptedSummaryWarnings, setAcceptedSummaryWarnings] = useState(false);
  const [regenerationCandidate, setRegenerationCandidate] = useState<number[] | null>(null);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const transcriptContainerRef = useRef<HTMLDivElement>(null);
  const transcriptLineRefs = useRef<Array<HTMLDivElement | null>>([]);

  useEffect(() => { setAcceptedSummaryWarnings(false); }, [summaryReviews]);

  // Audio sync hook (uses full transcript for seeking)
  const {
    seekTime,
    currentLineIndex,
    handleTimeUpdate,
    seekToLine,
    isAutoScroll,
  } = useAudioSync(transcript);

  const getTranscriptForTop = useCallback((topIndex: number) => {
    return transcript.filter((_, i) => assignments[i] === topIndex);
  }, [assignments, transcript]);

  useEffect(() => {
    setActiveSourceLine(null);
    transcriptLineRefs.current = [];
  }, [selectedTop]);

  // Auto-scroll to current line during playback (within filtered transcript)
  useEffect(() => {
    if (isAutoScroll && currentLineIndex >= 0 && transcriptContainerRef.current) {
      // Find position of current line within the filtered transcript
      const topLines = tops.length > 0 ? getTranscriptForTop(selectedTop) : transcript;
      const filteredIndex = topLines.findIndex((line) => {
        const originalIndex = transcript.indexOf(line);
        return originalIndex === currentLineIndex;
      });
      if (filteredIndex >= 0) {
        const lineElement = transcriptContainerRef.current.children[filteredIndex] as HTMLElement;
        if (lineElement) {
          lineElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
      }
    }
  }, [currentLineIndex, getTranscriptForTop, isAutoScroll, selectedTop, tops.length, transcript]);

  // Helper to get display name for a speaker
  const getDisplayName = (speakerId: string) => speakerNames[speakerId] || speakerId;

  const handleLineDoubleClick = (line: TranscriptLine) => {
    if (audioUrl) {
      const originalIndex = transcript.indexOf(line);
      if (originalIndex >= 0) {
        seekToLine(originalIndex, line);
      }
    }
  };

  const jumpToTranscriptLine = (localLineIndex: number | null | undefined) => {
    if (localLineIndex === null || localLineIndex === undefined) return;
    const line = topLines[localLineIndex];
    if (!line) return;

    setActiveSourceLine(localLineIndex);
    transcriptLineRefs.current[localLineIndex]?.scrollIntoView({
      behavior: 'smooth',
      block: 'center',
    });

    if (audioUrl) {
      const originalIndex = transcript.indexOf(line);
      if (originalIndex >= 0) {
        seekToLine(originalIndex, line);
      }
    }
  };

  const startEditing = (topIndex: number) => {
    setEditingTop(topIndex);
    setEditText(summaries[topIndex] || '');
  };

  const saveEdit = () => {
    if (editingTop !== null) {
      const newSummaries = { ...summaries };
      newSummaries[editingTop] = editText;
      setSummaries(newSummaries);
      setEditingTop(null);
    }
  };

  const cancelEdit = () => {
    setEditingTop(null);
    setEditText('');
  };

  const handleCopy = async () => {
    const text = summaries[selectedSummaryIndex];
    if (text) {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const updateExportMetadata = (patch: Partial<ExportMetadata>) => {
    setExportMetadata({ ...exportMetadata, ...patch });
  };

  const getSummaryIssueState = () => {
    const missingSummaryIndexes = tops.length > 0
      ? tops
          .map((_, index) => index)
          .filter((index) => !summaries[index]?.trim())
      : !summaries[0]?.trim()
        ? [0]
        : [];
    const reviewWarnings = Object.values(summaryReviews).flatMap(
      (review) => review?.review_warnings ?? []
    );
    const hasReviewWarnings = reviewWarnings.some((warning) =>
        ['warning', 'error'].includes(String(warning.severity ?? '').toLowerCase())
    );
    return {
      hasMissingSummaries: missingSummaryIndexes.length > 0,
      hasReviewWarnings,
      hasAnyIssue: missingSummaryIndexes.length > 0 || hasReviewWarnings,
    };
  };

  const handleExport = async (format: ExportFormat) => {
    setExportError(null);
    if (busy || !summariesAreFresh) {
      setExportError('Zusammenfassungen müssen nach den Korrekturen aktualisiert werden.');
      return;
    }
    if (summaryIssueState.hasMissingSummaries) {
      setExportError('Mindestens eine Zusammenfassung fehlt. Bitte aktualisieren Sie die Zusammenfassungen oder tragen Sie den Text manuell ein.');
      return;
    }
    if (summaryIssueState.hasReviewWarnings && !acceptedSummaryWarnings) {
      setExportError('Prüfhinweise müssen vor dem Export akzeptiert oder durch Aktualisierung behoben werden.');
      return;
    }
    setExportingFormat(format);
    try {
      const exportTops = hasTops ? tops : ['Gesamtes Gespräch'];
      const exportAssignments = hasTops ? assignments : transcript.map(() => 0);
      const blob = await exportProtocol({
        format,
        metadata: exportMetadata,
        tops: exportTops,
        transcript,
        assignments: exportAssignments,
        speakerNames,
        summaries,
        summaryReviews,
      });
      downloadBlob(blob, format);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Export fehlgeschlagen';
      setExportError(message);
    } finally {
      setExportingFormat(null);
    }
  };

  const downloadBlob = (blob: Blob, format: ExportFormat) => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `protokoll.${format}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const hasTops = tops.length > 0;
  const topIdAt = (index: number) => topIds[index] ?? summaryStates[index]?.top_id ?? `top-${index}`;
  const jobActive = Boolean(summaryJob && ['pending', 'processing', 'cancelling'].includes(summaryJob.status));
  const busy = isGenerating || jobActive;
  const selection = jobActive ? summaryJob?.top_ids ?? [] : selectedIds;
  const selectedIndexes = tops.map((_, index) => index).filter(index => selection.includes(topIdAt(index)));
  const jobTopTitle = (id: string) => {
    const index = tops.findIndex((_, i) => topIdAt(i) === id);
    return index >= 0 ? tops[index] : id.startsWith('whole-session:') ? 'Gesamtes Gespräch' : id;
  };
  const selectedSummaryIndex = hasTops ? selectedTop : 0;
  const topLines = hasTops ? getTranscriptForTop(selectedTop) : transcript;
  const selectedReview = summaryReviews[selectedSummaryIndex];
  const selectedSummaryState = summaryStates[selectedSummaryIndex];
  const selectedWarnings = selectedReview?.review_warnings ?? [];
  const structured = selectedReview?.structured ?? null;

  const getSourceLink = (section: keyof StructuredSummary, itemIndex: number) => {
    return selectedReview?.source_links.find(
      (link) => link.section === section && link.item_index === itemIndex
    );
  };

  const hasStructuredItems = Boolean(
    structured &&
      SUMMARY_SECTIONS.some((section) => structured[section.key]?.length)
  );
  const summaryIssueState = getSummaryIssueState();
  const speakerCount = Array.from(new Set(transcript.map((line) => line.speaker))).length;
  const assignedLineCount = hasTops
    ? assignments.filter((assignment) => assignment !== null && assignment !== undefined).length
    : transcript.length;
  const metadataMissing = [
    exportMetadata.committee.trim(),
    exportMetadata.date.trim(),
    exportMetadata.title.trim(),
  ].some((value) => !value);
  const exportBlocked =
    !summariesAreFresh ||
    summaryIssueState.hasMissingSummaries ||
    (summaryIssueState.hasReviewWarnings && !acceptedSummaryWarnings);
  const hasReviewContent =
    selectedWarnings.length > 0 ||
    hasStructuredItems ||
    Boolean(summaries[selectedSummaryIndex]);

  return (
    <div className="space-y-6">
      {!summariesAreFresh && (
        <div className="rounded-lg border border-yellow-200 bg-yellow-50 p-4 text-sm text-yellow-900">
          Änderungen betreffen einzelne TOPs. Die vorhandenen Zusammenfassungen bleiben sichtbar.
          Prüfen Sie die gelb markierten TOPs und übernehmen, bearbeiten oder regenerieren Sie nur diese.
        </div>
      )}

      {summaryJob && (
        <div className="rounded-lg border border-blue-300 bg-blue-50 p-4 text-sm text-blue-900">
          <div className="flex items-center justify-between gap-4">
            <div className="flex-1">
              <div className="font-medium">{jobActive ? 'Ausgewählte TOP-Zusammenfassungen werden erzeugt' : summaryJob.status === 'completed' ? 'Regenerierung abgeschlossen' : summaryJob.status === 'cancelled' ? 'Regenerierung abgebrochen' : 'Regenerierung mit Fehlern beendet'}</div>
              <div className="mt-1" aria-live="polite">
                {summaryJob.completed_tops ?? 0} von {summaryJob.total_tops} TOPs abgeschlossen
                {summaryJob.current_top_id && <div>Aktuell: {jobTopTitle(summaryJob.current_top_id)}</div>}
                {summaryJob.status === 'pending' && <div>Wartet auf Verarbeitung</div>}
                {summaryJob.status === 'cancelling' && <div>Abbruch angefordert</div>}
                {summaryJob.error && <div role="alert">{summaryJob.error}</div>}
                {Object.entries(summaryJob.outcomes ?? {}).filter(([, outcome]) => outcome.status === 'failed').map(([id, outcome]) => (
                  <div key={id} className="text-red-800">{jobTopTitle(id)}: {outcome.error}</div>
                ))}
              </div>
              <div role="progressbar" aria-label="Gesamtfortschritt" aria-valuemin={0} aria-valuemax={100} aria-valuenow={summaryJob.progress} className="mt-3 h-2 overflow-hidden rounded bg-blue-100">
                <div className="h-full bg-blue-600 transition-all" style={{ width: `${summaryJob.progress}%` }} />
              </div>
            </div>
            {jobActive && onCancelSummaryJob && summaryJob.status !== 'cancelling' && (
              <button type="button" onClick={() => void onCancelSummaryJob()} className="rounded border border-blue-300 px-3 py-2">
                Abbrechen
              </button>
            )}
          </div>
        </div>
      )}

      {summariesAreFresh && summaryIssueState.hasAnyIssue && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-900">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div className="space-y-3">
              <div>
                {summaryIssueState.hasMissingSummaries
                  ? 'Mindestens eine Zusammenfassung fehlt. Prüfen Sie die betroffenen TOPs oder starten Sie die Aktualisierung erneut.'
                  : 'Mindestens eine Zusammenfassung enthält Prüfhinweise. Sie können diese Hinweise nach Prüfung akzeptieren und trotzdem exportieren.'}
              </div>
              {!summaryIssueState.hasMissingSummaries && summaryIssueState.hasReviewWarnings && (
                <label className="inline-flex items-center gap-2 text-sm font-medium text-red-900">
                  <input
                    type="checkbox"
                    checked={acceptedSummaryWarnings}
                    onChange={(event) => setAcceptedSummaryWarnings(event.target.checked)}
                    className="h-4 w-4 rounded border-red-300"
                  />
                  Prüfhinweise akzeptieren und Export erlauben
                </label>
              )}
            </div>
          </div>
        </div>
      )}

      <div className="rounded-lg border border-blue-200 bg-blue-50 p-5">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <p className="text-sm font-medium text-blue-700">Exportfreigabe</p>
            <h2 className="mt-1 text-xl font-semibold text-gray-950">
              Protokoll prüfen und herunterladen
            </h2>
            <p className="mt-2 max-w-3xl text-sm text-blue-900">
              Zusammenfassungen, Belege und Exportdaten bleiben editierbar. Blockierende
              Hinweise müssen akzeptiert oder durch Aktualisierung behoben werden.
            </p>
          </div>
          <div className="grid min-w-0 gap-2 sm:grid-cols-2 lg:w-[520px]">
            <div className="rounded-md border border-blue-200 bg-white px-3 py-2">
              <div className="text-xs font-medium uppercase text-gray-400">Zusammenfassungen</div>
              <div className={`mt-1 text-sm font-semibold ${summariesAreFresh && !summaryIssueState.hasMissingSummaries ? 'text-green-700' : 'text-yellow-700'}`}>
                {summariesAreFresh && !summaryIssueState.hasMissingSummaries ? 'Aktuell' : 'Prüfen'}
              </div>
            </div>
            <div className="rounded-md border border-blue-200 bg-white px-3 py-2">
              <div className="text-xs font-medium uppercase text-gray-400">Hinweise</div>
              <div className={`mt-1 text-sm font-semibold ${summaryIssueState.hasReviewWarnings && !acceptedSummaryWarnings ? 'text-yellow-700' : 'text-green-700'}`}>
                {summaryIssueState.hasReviewWarnings && !acceptedSummaryWarnings ? 'Offen' : 'Erledigt'}
              </div>
            </div>
            <div className="rounded-md border border-blue-200 bg-white px-3 py-2">
              <div className="text-xs font-medium uppercase text-gray-400">Zuordnung</div>
              <div className="mt-1 text-sm font-semibold text-gray-900">
                {assignedLineCount}/{transcript.length} Zeilen
              </div>
            </div>
            <div className="rounded-md border border-blue-200 bg-white px-3 py-2">
              <div className="text-xs font-medium uppercase text-gray-400">Metadaten</div>
              <div className={`mt-1 text-sm font-semibold ${metadataMissing ? 'text-yellow-700' : 'text-green-700'}`}>
                {metadataMissing ? 'Unvollständig' : 'Bereit'}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Layout */}
      <div className="flex gap-6 h-[600px]">
        {/* TOPs Sidebar */}
        <div className="w-72 bg-white rounded-lg border border-gray-200 p-4 overflow-y-auto">
          <h3 className="font-medium text-gray-900 mb-4">Tagesordnung</h3>
          {hasTops && onRegenerateSummaries && (
            <div className="mb-4 space-y-2 text-sm">
              <div className="flex flex-wrap gap-2">
                <button type="button" disabled={busy} onClick={() => setSelectedIds(tops.map((_, i) => topIdAt(i)))} className="text-blue-700 disabled:opacity-50">Alle auswählen</button>
                <button type="button" disabled={busy || !selectedIndexes.length} onClick={() => setSelectedIds([])} className="text-blue-700 disabled:opacity-50">Auswahl aufheben</button>
              </div>
              <button type="button" disabled={busy || !selectedIndexes.length || editingTop !== null}
                onClick={() => setRegenerationCandidate(selectedIndexes)}
                className="rounded bg-blue-600 px-3 py-2 text-white disabled:opacity-50">
                Auswahl neu generieren ({selectedIndexes.length})
              </button>
            </div>
          )}
          <div className="space-y-2">
            {!hasTops ? (
              <div className="rounded-lg border border-gray-200 bg-gray-50 px-3 py-3 text-sm text-gray-600">
                Keine TOPs vorhanden.
              </div>
            ) : tops.map((top, index) => {
              const isSelected = selectedTop === index;
              const hasSummary = summaries[index] && summaries[index].trim();
              const state = summaryStates[index]?.status;
              return (
                <div key={topIdAt(index)} className="flex items-center gap-2">
                  {onRegenerateSummaries && <input type="checkbox" aria-label={`${top} auswählen`}
                    disabled={busy} checked={jobActive ? Boolean(summaryJob?.top_ids.includes(topIdAt(index))) : selectedIds.includes(topIdAt(index))}
                    onChange={(event) => setSelectedIds(current => event.target.checked ? [...current, topIdAt(index)] : current.filter(id => id !== topIdAt(index)))} />}
                <button
                  onClick={() => setSelectedTop(index)}
                  className={`w-full text-left px-3 py-3 rounded-lg border-2 transition-all ${
                    isSelected
                      ? 'bg-blue-50 border-blue-300 text-blue-700'
                      : 'border-transparent hover:bg-gray-50'
                  }`}
                >
                  <div className="flex items-start gap-2">
                    <div
                      className={`w-3 h-3 rounded-full mt-1 ${
                        state === 'review_required'
                          ? 'bg-yellow-500'
                          : state === 'failed' || state === 'missing'
                            ? 'bg-red-500'
                            : state === 'queued' || state === 'running'
                              ? 'bg-blue-500'
                              : hasSummary ? 'bg-green-500' : 'bg-gray-300'
                      }`}
                    />
                    <div className="flex-1 min-w-0">
                      <div
                        className="font-medium text-sm truncate"
                        title={top || 'Unbenannter Tagesordnungspunkt'}
                      >
                        {top || 'Unbenannter Tagesordnungspunkt'}
                      </div>
                      {jobActive && summaryJob?.top_ids.includes(topIdAt(index)) && <div className="text-xs text-blue-700">Im laufenden Auftrag</div>}
                    </div>
                  </div>
                </button>
                </div>
              );
            })}
          </div>
        </div>

        {/* Summary Content */}
        <div className="flex-1 flex flex-col gap-4">
          {/* Summary Box */}
          <div className="flex-1 bg-white rounded-lg border border-gray-200 overflow-hidden flex flex-col">
            <div className="px-4 py-3 border-b border-gray-200 bg-gray-50 flex items-center justify-between">
              <h3 className="font-medium text-gray-900">
                {hasTops ? tops[selectedTop] : 'Gesamtes Gespräch'}
              </h3>
              <div className="flex gap-2">
                {editingTop === selectedSummaryIndex ? (
                  <>
                    <button
                      onClick={saveEdit}
                      className="px-3 py-1 text-sm bg-green-500 text-white rounded hover:bg-green-600"
                    >
                      Speichern
                    </button>
                    <button
                      onClick={cancelEdit}
                      className="px-3 py-1 text-sm bg-gray-200 text-gray-700 rounded hover:bg-gray-300"
                    >
                      Abbrechen
                    </button>
                  </>
                ) : (
                  <>
                    <button
                      onClick={handleCopy}
                      disabled={!summaries[selectedSummaryIndex]}
                      className="p-2 text-gray-600 hover:bg-gray-200 rounded disabled:opacity-50 disabled:cursor-not-allowed"
                      title={copied ? 'Kopiert!' : 'In Zwischenablage kopieren'}
                    >
                      {copied ? (
                        <svg className="w-4 h-4 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                      ) : (
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                        </svg>
                      )}
                    </button>
                    <button
                      onClick={() => startEditing(selectedSummaryIndex)}
                      className="p-2 text-gray-600 hover:bg-gray-200 rounded"
                      title="Bearbeiten"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                      </svg>
                    </button>
                    {['review_required', 'failed'].includes(selectedSummaryState?.status ?? '') && summaries[selectedSummaryIndex]?.trim() && (
                      <button
                        type="button"
                        onClick={() => void onAcceptSummary(selectedSummaryIndex)}
                        disabled={busy}
                        className="rounded bg-green-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-green-700 disabled:opacity-50"
                      >
                        Bestehende übernehmen
                      </button>
                    )}
                    <button
                      onClick={() => setRegenerationCandidate([selectedSummaryIndex])}
                      disabled={busy}
                      className="rounded border border-blue-300 px-3 py-1.5 text-sm font-medium text-blue-700 hover:bg-blue-50 disabled:opacity-50"
                    >
                      Neu generieren
                    </button>
                  </>
                )}
              </div>
            </div>
            {selectedSummaryState?.status === 'review_required' && (
              <div className="border-b border-yellow-200 bg-yellow-50 px-4 py-3 text-sm text-yellow-900">
                Die Eingabe dieses TOPs wurde geändert. Prüfen Sie den vorhandenen Text und übernehmen Sie ihn,
                bearbeiten Sie ihn manuell oder starten Sie nur für diesen TOP eine Neugenerierung.
                {selectedSummaryState.change_reasons?.length ? (
                  <ul className="mt-2 list-disc pl-5">
                    {selectedSummaryState.change_reasons.map((reason) => (
                      <li key={reason}>{CHANGE_REASON_LABELS[reason] ?? reason}</li>
                    ))}
                  </ul>
                ) : null}
              </div>
            )}
            <div className="flex-1 p-4 overflow-y-auto">
              {editingTop === selectedSummaryIndex ? (
                <textarea
                  value={editText}
                  onChange={(e: ChangeEvent<HTMLTextAreaElement>) => setEditText(e.target.value)}
                  className="w-full h-full p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
                />
              ) : hasReviewContent ? (
                <div className="space-y-3">
                  {selectedWarnings.length > 0 && (
                    <div className="rounded-md border border-yellow-200 bg-yellow-50 px-3 py-2 text-sm text-yellow-900">
                      <div className="font-medium">Prüfhinweise</div>
                      <div className="mt-1 space-y-1">
                        {selectedWarnings.slice(0, 3).map((warning, index) => (
                          <button
                            key={`${warning.kind}-${warning.keyword ?? ''}-${index}`}
                            type="button"
                            onClick={() => jumpToTranscriptLine(warning.line_indices?.[0])}
                            className="block w-full text-left hover:underline disabled:no-underline"
                            disabled={!warning.line_indices?.length}
                            title={warning.excerpt || warning.message}
                          >
                            {warning.message}
                          </button>
                        ))}
                        {selectedWarnings.length > 3 && (
                          <div className="text-yellow-800">
                            +{selectedWarnings.length - 3} weitere Hinweise
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {hasStructuredItems && structured ? (
                    <div className="space-y-3">
                      {SUMMARY_SECTIONS.map((section) => {
                        const items = structured[section.key] ?? [];
                        if (items.length === 0) return null;

                        return (
                          <section
                            key={section.key}
                            className={`border-l-4 pl-3 ${
                              section.important
                                ? 'border-blue-500'
                                : section.key === 'uncertainties'
                                  ? 'border-yellow-400'
                                  : 'border-gray-200'
                            }`}
                          >
                            <h4
                              className={`mb-2 text-sm font-semibold ${
                                section.important ? 'text-blue-900' : 'text-gray-800'
                              }`}
                            >
                              {section.label}
                            </h4>
                            <ul className="space-y-2">
                              {items.map((item, itemIndex) => {
                                const source = getSourceLink(section.key, itemIndex);
                                return (
                                  <li
                                    key={`${section.key}-${itemIndex}`}
                                    className="text-sm text-gray-700"
                                  >
                                    <div>{item}</div>
                                    <div className="mt-1 flex flex-wrap gap-2">
                                      {source?.missing_source ? (
                                        <span className="rounded border border-yellow-300 bg-yellow-50 px-2 py-0.5 text-xs font-medium text-yellow-800">
                                          Quelle fehlt
                                        </span>
                                      ) : source?.line_indices?.length ? (
                                        <button
                                          type="button"
                                          onClick={() => jumpToTranscriptLine(source.line_indices[0])}
                                          className="rounded border border-blue-200 bg-blue-50 px-2 py-0.5 text-xs font-medium text-blue-700 hover:bg-blue-100"
                                          title={source.excerpt || 'Transkriptstelle anzeigen'}
                                        >
                                          Beleg {source.start != null ? formatTime(source.start) : ''}
                                        </button>
                                      ) : null}
                                    </div>
                                  </li>
                                );
                              })}
                            </ul>
                          </section>
                        );
                      })}
                    </div>
                  ) : summaries[selectedSummaryIndex] ? (
                    <div className="prose max-w-none text-gray-700 whitespace-pre-wrap">
                      {summaries[selectedSummaryIndex]}
                    </div>
                  ) : (
                    <div className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800">
                      Für diesen TOP liegt keine Zusammenfassung vor.
                    </div>
                  )}
                </div>
              ) : (
                <div className="flex items-center justify-center h-full text-gray-400">
                  Keine Zusammenfassung vorhanden. Klicken Sie auf "Neu generieren".
                </div>
              )}
            </div>
          </div>

          {/* Original Transcript for this TOP */}
          <div className="h-64 bg-white rounded-lg border border-gray-200 overflow-hidden flex flex-col">
            {/* Audio Player */}
            {audioUrl && (
              <div className="px-4 py-2 border-b border-gray-200 bg-gray-50">
                <AudioPlayer
                  audioUrl={audioUrl}
                  currentTime={seekTime}
                  onTimeUpdate={handleTimeUpdate}
                />
              </div>
            )}

            <div className="px-4 py-2 border-b border-gray-200 bg-gray-50">
              <h4 className="text-sm font-medium text-gray-700">
                Originaltranskript ({topLines.length} Zeilen)
                {audioUrl && <span className="text-gray-400 font-normal"> - Doppelklick zum Abspielen</span>}
              </h4>
            </div>
            <div ref={transcriptContainerRef} className="flex-1 overflow-y-auto p-3 text-sm">
              {topLines.length > 0 ? (
                topLines.map((line, index) => {
                  const originalIndex = transcript.indexOf(line);
                  const isCurrentLine = originalIndex === currentLineIndex;
                  return (
                    <div
                      key={index}
                      ref={(element) => {
                        transcriptLineRefs.current[index] = element;
                      }}
                      onDoubleClick={() => handleLineDoubleClick(line)}
                      className={`mb-1 px-2 py-1 rounded cursor-pointer hover:bg-gray-100 ${
                        isCurrentLine || activeSourceLine === index
                          ? 'ring-2 ring-blue-500 ring-offset-1 bg-blue-50'
                          : ''
                      }`}
                    >
                      <span className="font-medium text-gray-500">
                        {getDisplayName(line.speaker)}:
                      </span>{' '}
                      <span className="text-gray-700">{line.text}</span>
                      <span className="ml-2 text-xs text-gray-400">
                        [{formatTime(line.start)}]
                      </span>
                    </div>
                  );
                })
              ) : (
                <div className="text-gray-400 text-center py-4">
                  Keine Zeilen diesem TOP zugeordnet.
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {regenerationCandidate !== null && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4" role="dialog" aria-modal="true" aria-labelledby="regeneration-title">
          <div className="max-w-lg rounded-xl bg-white p-6 shadow-xl">
            <h3 id="regeneration-title" className="text-lg font-semibold text-gray-950">
              {regenerationCandidate.length === 1 ? 'TOP-Zusammenfassung wirklich neu generieren?' : `${regenerationCandidate.length} TOP-Zusammenfassungen wirklich neu generieren?`}
            </h3>
            <p className="mt-3 text-sm text-gray-700">
              Ausgewählt: {hasTops ? regenerationCandidate.map(index => tops[index]).join(', ') : 'Gesamtes Gespräch'}.
              Im CPU-Modus kann dies mehrere Stunden dauern und erhebliche Serverleistung beanspruchen.
              Prüfen Sie vorher, ob die vorhandene Zusammenfassung nicht bereits ausreicht.
            </p>
            <p className="mt-2 text-sm text-gray-600">
              Der Job läuft serverseitig weiter. Sie können die Seite verlassen und später zurückkehren.
            </p>
            <div className="mt-6 flex justify-end gap-3">
              <button type="button" onClick={() => setRegenerationCandidate(null)} className="rounded border border-gray-300 px-4 py-2 text-sm">
                Abbrechen
              </button>
              <button
                type="button"
                onClick={async () => {
                  const indexes = regenerationCandidate;
                  setRegenerationCandidate(null);
                  if (onRegenerateSummaries) await onRegenerateSummaries(indexes);
                  else await onRegenerateSummary(indexes[0]!);
                }}
                className="rounded bg-red-600 px-4 py-2 text-sm font-medium text-white hover:bg-red-700"
              >
                Regenerierung verbindlich starten
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Export Section */}
      <div className="bg-white rounded-lg border border-gray-200 p-4">
        <div className="space-y-4">
          <div className="flex items-start justify-between gap-4">
            <div>
              <h3 className="font-medium text-gray-900">Export</h3>
              <p className="text-sm text-gray-500">
                Metadaten final prüfen und Protokoll als Datei herunterladen
              </p>
              <p className="mt-1 text-xs text-gray-400">
                {speakerCount} Sprecher erkannt · {hasTops ? `${tops.length} TOPs` : 'Gesamtes Gespräch'}
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              {(['docx', 'pdf', 'txt'] as ExportFormat[]).map((format) => (
                <button
                  key={format}
                  onClick={() => handleExport(format)}
                  disabled={exportingFormat !== null || !summariesAreFresh || busy || editingTop !== null}
                  className={`px-4 py-2 rounded-lg text-sm font-medium disabled:opacity-50 ${
                    format === 'docx' && !exportBlocked
                      ? 'bg-blue-600 text-white hover:bg-blue-700'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  {exportingFormat === format
                    ? 'Exportiert...'
                    : format === 'docx'
                      ? 'DOCX'
                      : format === 'pdf'
                        ? 'PDF'
                        : 'Text (.txt)'}
                </button>
              ))}
            </div>
          </div>

          <div className="grid gap-3 md:grid-cols-2">
            <label className="text-sm font-medium text-gray-700">
              Gremium
              <input
                value={exportMetadata.committee}
                onChange={(event) => updateExportMetadata({ committee: event.target.value })}
                className="mt-1 w-full rounded-md border border-gray-300 px-3 py-2 font-normal text-gray-900 focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="z. B. Hauptausschuss"
              />
            </label>
            <label className="text-sm font-medium text-gray-700">
              Datum
              <input
                type="date"
                value={exportMetadata.date}
                onChange={(event) => updateExportMetadata({ date: event.target.value })}
                className="mt-1 w-full rounded-md border border-gray-300 px-3 py-2 font-normal text-gray-900 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </label>
            <label className="text-sm font-medium text-gray-700">
              Ort
              <input
                value={exportMetadata.location}
                onChange={(event) => updateExportMetadata({ location: event.target.value })}
                className="mt-1 w-full rounded-md border border-gray-300 px-3 py-2 font-normal text-gray-900 focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="z. B. Rathaus, Sitzungssaal"
              />
            </label>
            <label className="text-sm font-medium text-gray-700">
              Sitzungstitel
              <input
                value={exportMetadata.title}
                onChange={(event) => updateExportMetadata({ title: event.target.value })}
                className="mt-1 w-full rounded-md border border-gray-300 px-3 py-2 font-normal text-gray-900 focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Sitzungsprotokoll"
              />
            </label>
            <label className="text-sm font-medium text-gray-700 md:col-span-2">
              Teilnehmer
              <textarea
                value={exportMetadata.participants.join('\n')}
                onChange={(event) =>
                  updateExportMetadata({
                    participants: event.target.value
                      .split(/\n|;/)
                      .map((participant) => participant.trim())
                      .filter(Boolean),
                  })
                }
                className="mt-1 h-20 w-full resize-none rounded-md border border-gray-300 px-3 py-2 font-normal text-gray-900 focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Eine Person pro Zeile"
              />
            </label>
          </div>

          <div className="flex flex-wrap gap-4 text-sm text-gray-700">
            <label className="inline-flex items-center gap-2">
              <input
                type="checkbox"
                checked={exportMetadata.includeSpeakerList}
                onChange={(event) => updateExportMetadata({ includeSpeakerList: event.target.checked })}
                className="h-4 w-4 rounded border-gray-300"
              />
              Sprecherliste
            </label>
            <label className="inline-flex items-center gap-2">
              <input
                type="checkbox"
                checked={exportMetadata.includeTranscript}
                onChange={(event) => updateExportMetadata({ includeTranscript: event.target.checked })}
                className="h-4 w-4 rounded border-gray-300"
              />
              Transkript anfügen
            </label>
            <label className="inline-flex items-center gap-2">
              <input
                type="checkbox"
                checked={exportMetadata.groupTranscriptByTop}
                onChange={(event) => updateExportMetadata({ groupTranscriptByTop: event.target.checked })}
                disabled={!exportMetadata.includeTranscript}
                className="h-4 w-4 rounded border-gray-300 disabled:opacity-50"
              />
              TOP-Unterteilung
            </label>
            <label className="inline-flex items-center gap-2">
              <input
                type="checkbox"
                checked={exportMetadata.includeGenerationNote}
                onChange={(event) => updateExportMetadata({ includeGenerationNote: event.target.checked })}
                className="h-4 w-4 rounded border-gray-300"
              />
              Generierungshinweis
            </label>
          </div>

          {exportError && (
            <div className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
              {exportError}
            </div>
          )}
        </div>
      </div>

      {/* Actions */}
      <div className="flex justify-start">
        <button
          onClick={onBack}
          className="px-6 py-3 rounded-lg font-medium text-gray-600 hover:bg-gray-100 transition-colors flex items-center gap-2"
        >
          <span>←</span>
          Zurück zur Zuordnung
        </button>
      </div>
    </div>
  );
}
