import { useState, useRef, useEffect, type MouseEvent } from 'react';
import type {
  AssignmentSuggestionSegment,
  AssignmentStepProps,
  TranscriptLine,
  TopColor,
} from '../types';
import AudioPlayer from './AudioPlayer';
import { useAudioSync } from '../hooks/useAudioSync';
import SpeakerNameEditor from './SpeakerNameEditor';

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// Color palette for TOPs
const topColors: TopColor[] = [
  { bg: 'bg-blue-100', border: 'border-blue-300', text: 'text-blue-700', dot: 'bg-blue-500' },
  { bg: 'bg-green-100', border: 'border-green-300', text: 'text-green-700', dot: 'bg-green-500' },
  { bg: 'bg-yellow-100', border: 'border-yellow-300', text: 'text-yellow-700', dot: 'bg-yellow-500' },
  { bg: 'bg-purple-100', border: 'border-purple-300', text: 'text-purple-700', dot: 'bg-purple-500' },
  { bg: 'bg-pink-100', border: 'border-pink-300', text: 'text-pink-700', dot: 'bg-pink-500' },
  { bg: 'bg-indigo-100', border: 'border-indigo-300', text: 'text-indigo-700', dot: 'bg-indigo-500' },
  { bg: 'bg-red-100', border: 'border-red-300', text: 'text-red-700', dot: 'bg-red-500' },
  { bg: 'bg-orange-100', border: 'border-orange-300', text: 'text-orange-700', dot: 'bg-orange-500' },
];

export default function AssignmentStep({
  onNext,
  onBack,
  tops,
  setTops,
  topIds = [],
  onTopsChange,
  transcript,
  setTranscript,
  assignments,
  setAssignments,
  agendaDetection,
  agendaDetectionError,
  onTranscriptStructureChange,
  audioUrl,
  speakerNames,
  setSpeakerNames,
  sessionId,
  hasSummaries = false,
  rememberSpeakers = false,
}: AssignmentStepProps) {
  const [selectedTop, setSelectedTop] = useState(0);
  const [selectionStart, setSelectionStart] = useState<number | null>(null);
  const [selectedLineIndex, setSelectedLineIndex] = useState<number | null>(null);
  const [editingLine, setEditingLine] = useState<number | null>(null);
  const [editText, setEditText] = useState('');
  const [topTitleDraft, setTopTitleDraft] = useState(tops[0] ?? '');

  const stableTopIds = tops.map((_, index) => topIds[index] ?? `top-${index}`);
  const updateTops = (nextTops: string[], nextTopIds = stableTopIds) => {
    if (onTopsChange) onTopsChange(nextTops, nextTopIds);
    else setTops(nextTops);
  };
  const createStableId = () =>
    typeof crypto.randomUUID === 'function'
      ? crypto.randomUUID()
      : `local-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  const transcriptContainerRef = useRef<HTMLDivElement>(null);

  // Audio sync hook
  const {
    seekTime,
    currentLineIndex,
    handleTimeUpdate,
    seekToLine,
    isAutoScroll,
  } = useAudioSync(transcript);

  // Auto-scroll to current line during playback
  useEffect(() => {
    if (isAutoScroll && currentLineIndex >= 0 && transcriptContainerRef.current) {
      const lineElement = transcriptContainerRef.current.children[currentLineIndex] as HTMLElement;
      if (lineElement) {
        lineElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    }
  }, [currentLineIndex, isAutoScroll]);

  useEffect(() => {
    if (selectedTop >= tops.length) {
      setSelectedTop(Math.max(0, tops.length - 1));
    }
  }, [selectedTop, tops.length]);

  useEffect(() => {
    setTopTitleDraft(tops[selectedTop] ?? '');
  }, [selectedTop, tops]);

  // Helper to get display name for a speaker
  const getDisplayName = (speakerId: string) => speakerNames[speakerId] || speakerId;

  const getColor = (topIndex: number): TopColor => topColors[topIndex % topColors.length]!;

  const getAssignmentCounts = (): Record<number, number> => {
    const counts: Record<number, number> = {};
    tops.forEach((_, index) => {
      counts[index] = assignments.filter((a) => a === index).length;
    });
    return counts;
  };

  const counts = getAssignmentCounts();

  const handleLineClick = (lineIndex: number, event: MouseEvent<HTMLDivElement>) => {
    if (editingLine === lineIndex) {
      return;
    }
    setSelectedLineIndex(lineIndex);

    // Double-click to seek audio
    const line = transcript[lineIndex];
    if (event.detail === 2 && audioUrl && line) {
      seekToLine(lineIndex, line);
      return;
    }

    if (event.shiftKey && selectionStart !== null) {
      // Range selection
      const start = Math.min(selectionStart, lineIndex);
      const end = Math.max(selectionStart, lineIndex);
      const newAssignments = [...assignments];
      for (let i = start; i <= end; i++) {
        newAssignments[i] = selectedTop;
      }
      setAssignments(newAssignments);
      setSelectionStart(null);
    } else {
      // Single click - toggle or set
      const newAssignments = [...assignments];
      if (newAssignments[lineIndex] === selectedTop) {
        newAssignments[lineIndex] = null; // Unassign
      } else {
        newAssignments[lineIndex] = selectedTop;
      }
      setAssignments(newAssignments);
      setSelectionStart(lineIndex);
    }
  };

  const applySuggestionSegment = (segment: AssignmentSuggestionSegment) => {
    const newAssignments = [...assignments];
    for (let i = segment.start_index; i <= segment.end_index; i++) {
      if (i >= 0 && i < newAssignments.length) {
        newAssignments[i] = segment.top_index;
      }
    }
    setAssignments(newAssignments);
    setSelectedTop(segment.top_index);
    setSelectedLineIndex(segment.start_index);
  };

  const applyAllSafeSuggestions = () => {
    if (!agendaDetection) {
      return;
    }

    const newAssignments = [...assignments];
    agendaDetection.segments
      .filter((segment) => !segment.uncertain && segment.confidence >= 0.7)
      .forEach((segment) => {
        for (let i = segment.start_index; i <= segment.end_index; i++) {
          if (i >= 0 && i < newAssignments.length) {
            newAssignments[i] = segment.top_index;
          }
        }
      });
    setAssignments(newAssignments);
  };

  const applyAllSuggestions = () => {
    if (!agendaDetection) {
      return;
    }
    setAssignments(agendaDetection.assignments);
  };

  const renameSelectedTop = () => {
    const title = topTitleDraft.trim();
    if (!title || !tops[selectedTop]) {
      return;
    }
    const newTops = [...tops];
    newTops[selectedTop] = title;
    updateTops(newTops);
  };

  const addTop = () => {
    const insertionIndex = Math.min(selectedTop + 1, tops.length);
    const newTops = [...tops];
    newTops.splice(insertionIndex, 0, 'Neuer Tagesordnungspunkt');
    const newTopIds = [...stableTopIds];
    newTopIds.splice(insertionIndex, 0, createStableId());
    const newAssignments = assignments.map((assignment) => {
      if (assignment === null) return null;
      return assignment >= insertionIndex ? assignment + 1 : assignment;
    });
    updateTops(newTops, newTopIds);
    setAssignments(newAssignments);
    setSelectedTop(insertionIndex);
    setTopTitleDraft(newTops[insertionIndex] ?? '');
  };

  const deleteSelectedTop = () => {
    if (tops.length === 0) {
      return;
    }
    const newTops = tops.filter((_, index) => index !== selectedTop);
    const newTopIds = stableTopIds.filter((_, index) => index !== selectedTop);
    const newAssignments = assignments.map((assignment) => {
      if (assignment === null) return null;
      if (assignment === selectedTop) return null;
      return assignment > selectedTop ? assignment - 1 : assignment;
    });
    updateTops(newTops, newTopIds);
    setAssignments(newAssignments);
    setSelectedTop(Math.max(0, Math.min(selectedTop, newTops.length - 1)));
  };

  const mergeSelectedTopWithPrevious = () => {
    if (selectedTop <= 0) {
      return;
    }

    const newTops = [...tops];
    const newTopIds = [...stableTopIds];
    newTops[selectedTop - 1] = `${newTops[selectedTop - 1]} / ${newTops[selectedTop]}`;
    newTops.splice(selectedTop, 1);
    newTopIds.splice(selectedTop, 1);
    const newAssignments = assignments.map((assignment) => {
      if (assignment === null) return null;
      if (assignment === selectedTop) return selectedTop - 1;
      return assignment > selectedTop ? assignment - 1 : assignment;
    });
    updateTops(newTops, newTopIds);
    setAssignments(newAssignments);
    setSelectedTop(selectedTop - 1);
  };

  const getCurrentSegmentBounds = (lineIndex: number): { start: number; end: number } => {
    const value = assignments[lineIndex] ?? null;
    let start = lineIndex;
    let end = lineIndex;

    while (start > 0 && (assignments[start - 1] ?? null) === value) {
      start -= 1;
    }
    while (end < assignments.length - 1 && (assignments[end + 1] ?? null) === value) {
      end += 1;
    }

    return { start, end };
  };

  const assignCurrentSegmentToSelectedTop = () => {
    if (selectedLineIndex === null) {
      return;
    }
    const { start, end } = getCurrentSegmentBounds(selectedLineIndex);
    const newAssignments = [...assignments];
    for (let i = start; i <= end; i++) {
      newAssignments[i] = selectedTop;
    }
    setAssignments(newAssignments);
  };

  const splitCurrentSegmentAtSelectedLine = () => {
    if (selectedLineIndex === null) {
      return;
    }
    const { end } = getCurrentSegmentBounds(selectedLineIndex);
    const newAssignments = [...assignments];
    for (let i = selectedLineIndex; i <= end; i++) {
      newAssignments[i] = selectedTop;
    }
    setAssignments(newAssignments);
  };

  const mergeCurrentSegmentWithPrevious = () => {
    if (selectedLineIndex === null) {
      return;
    }
    const { start, end } = getCurrentSegmentBounds(selectedLineIndex);
    if (start === 0) {
      return;
    }
    const previousAssignment = assignments[start - 1] ?? null;
    const newAssignments = [...assignments];
    for (let i = start; i <= end; i++) {
      newAssignments[i] = previousAssignment;
    }
    setAssignments(newAssignments);
  };

  const mergeCurrentSegmentWithNext = () => {
    if (selectedLineIndex === null) {
      return;
    }
    const { start, end } = getCurrentSegmentBounds(selectedLineIndex);
    if (end >= assignments.length - 1) {
      return;
    }
    const nextAssignment = assignments[end + 1] ?? null;
    const newAssignments = [...assignments];
    for (let i = start; i <= end; i++) {
      newAssignments[i] = nextAssignment;
    }
    setAssignments(newAssignments);
  };

  const startLineEdit = (lineIndex: number, text: string) => {
    setEditingLine(lineIndex);
    setEditText(text);
  };

  const saveLineEdit = () => {
    if (editingLine === null) {
      return;
    }

    const currentLine = transcript[editingLine];
    if (!currentLine) {
      return;
    }

    const parts = editText
      .split(/\r?\n/)
      .map((part) => part.trim())
      .filter(Boolean);
    const texts = parts.length > 0 ? parts : [''];
    const duration = Math.max(0, currentLine.end - currentLine.start);
    const splitDuration = texts.length > 0 ? duration / texts.length : 0;
    const replacementLines = texts.map((text, index) => ({
      ...currentLine,
      line_id: index === 0 ? currentLine.line_id ?? createStableId() : createStableId(),
      text,
      start: currentLine.start + splitDuration * index,
      end:
        index === texts.length - 1
          ? currentLine.end
          : currentLine.start + splitDuration * (index + 1),
    }));

    const updatedTranscript = [
      ...transcript.slice(0, editingLine),
      ...replacementLines,
      ...transcript.slice(editingLine + 1),
    ];
    setTranscript(updatedTranscript);

    if (replacementLines.length !== 1) {
      const currentAssignment = assignments[editingLine] ?? null;
      setAssignments([
        ...assignments.slice(0, editingLine),
        ...replacementLines.map(() => currentAssignment),
        ...assignments.slice(editingLine + 1),
      ]);
      onTranscriptStructureChange?.();
    }

    setSelectedLineIndex(editingLine);
    setEditingLine(null);
    setEditText('');
  };

  const cancelLineEdit = () => {
    setEditingLine(null);
    setEditText('');
  };

  const mergeLineWithPrevious = () => {
    if (selectedLineIndex === null || selectedLineIndex <= 0) {
      return;
    }

    const previousLine = transcript[selectedLineIndex - 1];
    const currentLine = transcript[selectedLineIndex];
    if (!previousLine || !currentLine) {
      return;
    }

    const mergedLine = {
      ...previousLine,
      text: [previousLine.text, currentLine.text].filter(Boolean).join(' '),
      start: Math.min(previousLine.start, currentLine.start),
      end: Math.max(previousLine.end, currentLine.end),
    };

    setTranscript([
      ...transcript.slice(0, selectedLineIndex - 1),
      mergedLine,
      ...transcript.slice(selectedLineIndex + 1),
    ]);
    setAssignments([
      ...assignments.slice(0, selectedLineIndex),
      ...assignments.slice(selectedLineIndex + 1),
    ]);
    setSelectedLineIndex(selectedLineIndex - 1);
    onTranscriptStructureChange?.();
  };

  const mergeLineWithNext = () => {
    if (selectedLineIndex === null || selectedLineIndex >= transcript.length - 1) {
      return;
    }

    const currentLine = transcript[selectedLineIndex];
    const nextLine = transcript[selectedLineIndex + 1];
    if (!currentLine || !nextLine) {
      return;
    }

    const mergedLine = {
      ...currentLine,
      text: [currentLine.text, nextLine.text].filter(Boolean).join(' '),
      start: Math.min(currentLine.start, nextLine.start),
      end: Math.max(currentLine.end, nextLine.end),
    };

    setTranscript([
      ...transcript.slice(0, selectedLineIndex),
      mergedLine,
      ...transcript.slice(selectedLineIndex + 2),
    ]);
    setAssignments([
      ...assignments.slice(0, selectedLineIndex + 1),
      ...assignments.slice(selectedLineIndex + 2),
    ]);
    onTranscriptStructureChange?.();
  };

  const mergeConsecutiveSameSpeakerLines = () => {
    if (transcript.length <= 1) {
      return;
    }

    const mergedTranscript: TranscriptLine[] = [];
    const mergedAssignments: (number | null)[] = [];

    for (let index = 0; index < transcript.length; index++) {
      const line = transcript[index];
      if (!line) {
        continue;
      }

      const assignment = assignments[index] ?? null;
      const previousLine = mergedTranscript[mergedTranscript.length - 1];
      const previousAssignment = mergedAssignments[mergedAssignments.length - 1] ?? null;

      if (
        previousLine &&
        previousLine.speaker === line.speaker &&
        previousAssignment === assignment
      ) {
        mergedTranscript[mergedTranscript.length - 1] = {
          ...previousLine,
          text: [previousLine.text, line.text].filter(Boolean).join(' '),
          start: Math.min(previousLine.start, line.start),
          end: Math.max(previousLine.end, line.end),
        };
      } else {
        mergedTranscript.push(line);
        mergedAssignments.push(assignment);
      }
    }

    if (mergedTranscript.length === transcript.length) {
      return;
    }

    setTranscript(mergedTranscript);
    setAssignments(mergedAssignments);
    setSelectedLineIndex(null);
    setSelectionStart(null);
    setEditingLine(null);
    setEditText('');
    onTranscriptStructureChange?.();
  };

  const assignedCount = assignments.filter((a) => a !== null).length;
  const totalCount = transcript.length;
  const unassignedCount = Math.max(0, totalCount - assignedCount);
  const hasTops = tops.length > 0;
  const canProceed = hasTops ? assignedCount > 0 : true;
  const selectedSegmentBounds =
    selectedLineIndex !== null ? getCurrentSegmentBounds(selectedLineIndex) : null;
  const safeSuggestionCount =
    agendaDetection?.segments.filter((segment) => !segment.uncertain && segment.confidence >= 0.7)
      .length ?? 0;
  const speakerIds = Array.from(new Set(transcript.map((line) => line.speaker).filter(Boolean)));
  const openSpeakerCount = speakerIds.filter((speakerId) => {
    const displayName = speakerNames[speakerId]?.trim();
    return !displayName || displayName.toLowerCase() === speakerId.toLowerCase();
  }).length;
  const uncertainSegmentCount =
    agendaDetection?.segments.filter((segment) => segment.uncertain).length ?? 0;
  const topReviewStatus = !hasTops
    ? 'Nicht erforderlich'
    : uncertainSegmentCount > 0 || unassignedCount > 0
      ? 'Prüfen'
      : 'Bereit';
  const speakerReviewStatus = openSpeakerCount > 0 ? 'Prüfen' : 'Bereit';
  const protocolDraftStatus = hasSummaries ? 'Vorbereitet' : 'Wird nach Prüfung erstellt';

  const getDetectionSegmentForLine = (lineIndex: number) =>
    agendaDetection?.segments.find(
      (segment) => lineIndex >= segment.start_index && lineIndex <= segment.end_index
    ) ?? null;

  return (
    <div className="space-y-6">
      {/* Review overview */}
      <div className="rounded-lg border border-blue-200 bg-blue-50 p-5">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <p className="text-sm font-medium text-blue-700">Prüfung nach der Automatik</p>
            <h2 className="mt-1 text-xl font-semibold text-gray-950">
              Kontrollieren Sie nur die markierten Sprecher und TOP-Zuordnungen.
            </h2>
            <p className="mt-2 max-w-3xl text-sm text-blue-900">
              {hasTops
                ? 'Die Transkription und der Protokollentwurf sind vorbereitet. Änderungen markieren nur betroffene TOPs zur späteren Prüfung.'
                : 'Keine TOPs angelegt. Das gesamte Transkript wird ohne automatische TOP-Erkennung zusammengefasst.'}
            </p>
          </div>
          <div className="grid min-w-0 gap-2 sm:grid-cols-3 lg:w-[520px]">
            <div className="rounded-md border border-blue-200 bg-white px-3 py-2">
              <div className="text-xs font-medium uppercase text-gray-400">Sprecher</div>
              <div className={`mt-1 text-sm font-semibold ${openSpeakerCount > 0 ? 'text-yellow-700' : 'text-green-700'}`}>
                {speakerReviewStatus}
              </div>
              <div className="mt-1 text-xs text-gray-500">
                {openSpeakerCount > 0 ? `${openSpeakerCount} offen` : `${speakerIds.length} erkannt`}
              </div>
            </div>
            <div className="rounded-md border border-blue-200 bg-white px-3 py-2">
              <div className="text-xs font-medium uppercase text-gray-400">TOPs</div>
              <div className={`mt-1 text-sm font-semibold ${topReviewStatus === 'Prüfen' ? 'text-yellow-700' : 'text-green-700'}`}>
                {topReviewStatus}
              </div>
              <div className="mt-1 text-xs text-gray-500">
                {hasTops ? `${assignedCount}/${totalCount} Zeilen` : 'Gesamtgespräch'}
              </div>
            </div>
            <div className="rounded-md border border-blue-200 bg-white px-3 py-2">
              <div className="text-xs font-medium uppercase text-gray-400">Entwurf</div>
              <div className="mt-1 text-sm font-semibold text-gray-900">
                {protocolDraftStatus}
              </div>
              <div className="mt-1 text-xs text-gray-500">
                {hasSummaries ? 'Zusammenfassungen vorhanden' : 'Nächster Schritt'}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Speaker Name Editor */}
      <SpeakerNameEditor
        transcript={transcript}
        setTranscript={setTranscript}
        speakerNames={speakerNames}
        setSpeakerNames={setSpeakerNames}
        sessionId={sessionId}
        rememberSpeakers={rememberSpeakers}
        audioUrl={audioUrl}
      />

      {/* Agenda Detection */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
          <div>
            <h3 className="font-medium text-gray-900">Automatisch erkannte Segmente</h3>
            <p className="text-sm text-gray-600">
              Sichere Treffer sind bereits nutzbar; gelb markierte Bereiche sollten geprüft werden.
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={applyAllSafeSuggestions}
              disabled={!safeSuggestionCount}
              className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-200 disabled:text-gray-400"
            >
              Alle sicheren übernehmen
            </button>
            <button
              type="button"
              onClick={applyAllSuggestions}
              disabled={!agendaDetection?.segments.length}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-200 disabled:text-gray-400"
            >
              Alle übernehmen
            </button>
          </div>
        </div>

        {agendaDetection?.warnings?.map((warning) => (
          <p key={warning} role="status" className="text-amber-700">{warning}</p>
        ))}
        {agendaDetectionError && (
          <div className="mb-3 text-sm text-red-700 bg-red-50 border border-red-200 rounded-lg p-3">
            Automatische TOP-Erkennung fehlgeschlagen: {agendaDetectionError}. Die manuelle Zuordnung bleibt verfügbar.
          </div>
        )}
        {agendaDetection ? (
          <div className="space-y-2">
            <div className="text-xs text-gray-500">
              {agendaDetection.segments.length} Segmente, {agendaDetection.uncertain_count} unsicher · Strategie: {agendaDetection.strategy}
            </div>
            <div className="grid gap-2 md:grid-cols-2">
              {agendaDetection.segments.map((segment) => {
                const color = getColor(segment.top_index);
                return (
                  <div
                    key={`${segment.top_index}-${segment.start_index}`}
                    className={`border rounded-lg p-3 ${
                      segment.uncertain
                        ? 'border-yellow-400 bg-yellow-50 ring-1 ring-yellow-300'
                        : 'border-gray-200'
                    }`}
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <div className="flex items-center gap-2">
                          <span className={`w-2.5 h-2.5 rounded-full ${color.dot}`} />
                          <span className="font-medium text-sm text-gray-900 truncate">
                            {segment.top_title}
                          </span>
                        </div>
                        <div className="text-xs text-gray-500 mt-1">
                          Zeilen {segment.start_index + 1}-{segment.end_index + 1} · Confidence{' '}
                          {Math.round(segment.confidence * 100)}%
                        </div>
                        {segment.uncertain && (
                          <div className="mt-1 inline-flex items-center px-2 py-0.5 rounded bg-yellow-200 text-yellow-900 text-xs font-medium">
                            Unsicher prüfen
                          </div>
                        )}
                      </div>
                      <button
                        type="button"
                        onClick={() => applySuggestionSegment(segment)}
                        className="shrink-0 px-3 py-1.5 text-xs bg-gray-900 text-white rounded hover:bg-gray-700"
                      >
                        Übernehmen
                      </button>
                    </div>
                    <p className="text-xs text-gray-700 mt-2">{segment.reason}</p>
                    {segment.evidence_text && (
                      <p className="text-xs text-gray-500 mt-1 line-clamp-2">
                        Hinweis: {segment.evidence_text}
                      </p>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        ) : (
          <div className="text-sm text-gray-600">
            Keine automatischen Segmente verfügbar. Nutzen Sie bei Bedarf die manuelle Zeilenzuordnung.
          </div>
        )}
      </div>

      {/* Progress */}
      <div className="rounded-lg border border-gray-200 bg-white p-4">
        <div className="flex flex-wrap items-center justify-between gap-3 text-sm text-gray-600">
          <span>
            {assignedCount} von {totalCount} Zeilen zugeordnet
          </span>
          {hasTops && unassignedCount > 0 && (
            <span className="rounded-full bg-yellow-100 px-3 py-1 text-xs font-medium text-yellow-800">
              {unassignedCount} Zeilen ohne TOP
            </span>
          )}
          <div className="h-2 w-full overflow-hidden rounded-full bg-gray-200 sm:w-64">
            <div
              className="h-full bg-blue-500 transition-all"
              style={{ width: `${totalCount ? (assignedCount / totalCount) * 100 : 0}%` }}
            />
          </div>
        </div>
      </div>

      <div className="bg-gray-50 border border-gray-200 rounded-lg p-3 flex flex-wrap items-center gap-2 text-sm">
        <span className="mr-1 font-medium text-gray-900">Erweiterte Korrektur</span>
        <span className="text-gray-700">
          {selectedLineIndex === null
            ? 'Keine Zeile ausgewählt'
            : `Zeile ${selectedLineIndex + 1} ausgewählt${
                selectedSegmentBounds
                  ? ` · Segment ${selectedSegmentBounds.start + 1}-${selectedSegmentBounds.end + 1}`
                  : ''
              }`}
        </span>
        <button
          type="button"
          onClick={assignCurrentSegmentToSelectedTop}
          disabled={selectedLineIndex === null}
          className="px-3 py-1.5 bg-white border border-gray-300 rounded hover:bg-gray-100 disabled:opacity-50"
        >
          Segment zuordnen
        </button>
        <button
          type="button"
          onClick={splitCurrentSegmentAtSelectedLine}
          disabled={selectedLineIndex === null}
          className="px-3 py-1.5 bg-white border border-gray-300 rounded hover:bg-gray-100 disabled:opacity-50"
        >
          Grenze ab hier setzen
        </button>
        <button
          type="button"
          onClick={mergeCurrentSegmentWithPrevious}
          disabled={selectedLineIndex === null || selectedSegmentBounds?.start === 0}
          className="px-3 py-1.5 bg-white border border-gray-300 rounded hover:bg-gray-100 disabled:opacity-50"
        >
          Mit vorherigem Segment mergen
        </button>
        <button
          type="button"
          onClick={mergeCurrentSegmentWithNext}
          disabled={
            selectedLineIndex === null ||
            !selectedSegmentBounds ||
            selectedSegmentBounds.end >= assignments.length - 1
          }
          className="px-3 py-1.5 bg-white border border-gray-300 rounded hover:bg-gray-100 disabled:opacity-50"
        >
          Mit nächstem Segment mergen
        </button>
        <span className="w-px h-6 bg-gray-300 mx-1" />
        <button
          type="button"
          onClick={mergeLineWithPrevious}
          disabled={selectedLineIndex === null || selectedLineIndex <= 0}
          className="px-3 py-1.5 bg-white border border-gray-300 rounded hover:bg-gray-100 disabled:opacity-50"
        >
          Zeile mit vorheriger verbinden
        </button>
        <button
          type="button"
          onClick={mergeLineWithNext}
          disabled={selectedLineIndex === null || selectedLineIndex >= transcript.length - 1}
          className="px-3 py-1.5 bg-white border border-gray-300 rounded hover:bg-gray-100 disabled:opacity-50"
        >
          Zeile mit nächster verbinden
        </button>
        <button
          type="button"
          onClick={mergeConsecutiveSameSpeakerLines}
          disabled={transcript.length <= 1}
          className="px-3 py-1.5 bg-white border border-gray-300 rounded hover:bg-gray-100 disabled:opacity-50"
        >
          Gleiche Sprecher zusammenführen
        </button>
      </div>

      {/* Main Layout */}
      <div className="flex gap-6 h-[600px]">
        {/* TOPs Sidebar */}
        <div className="w-72 bg-white rounded-lg border border-gray-200 p-4 overflow-y-auto">
          <h3 className="font-medium text-gray-900 mb-4">Tagesordnung</h3>
          <div className="mb-4 space-y-2 border border-gray-200 rounded-lg p-3 bg-gray-50">
            <label className="block text-xs font-medium text-gray-600" htmlFor="selected-top-title">
              Ausgewählter TOP
            </label>
            <input
              id="selected-top-title"
              type="text"
              value={topTitleDraft}
              onChange={(event) => setTopTitleDraft(event.target.value)}
              className="w-full px-3 py-2 text-sm border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <div className="grid grid-cols-2 gap-2">
              <button
                type="button"
                onClick={renameSelectedTop}
                disabled={!topTitleDraft.trim()}
                className="px-2 py-1.5 text-xs bg-white border border-gray-300 rounded hover:bg-gray-100 disabled:opacity-50"
              >
                TOP umbenennen
              </button>
              <button
                type="button"
                onClick={addTop}
                className="px-2 py-1.5 text-xs bg-white border border-gray-300 rounded hover:bg-gray-100"
              >
                TOP hinzufügen
              </button>
              <button
                type="button"
                onClick={deleteSelectedTop}
                disabled={tops.length === 0}
                className="px-2 py-1.5 text-xs bg-white border border-gray-300 rounded hover:bg-red-50 hover:text-red-700 disabled:opacity-50"
              >
                TOP löschen
              </button>
              <button
                type="button"
                onClick={mergeSelectedTopWithPrevious}
                disabled={selectedTop <= 0}
                className="px-2 py-1.5 text-xs bg-white border border-gray-300 rounded hover:bg-gray-100 disabled:opacity-50"
              >
                TOP zusammenlegen
              </button>
            </div>
          </div>
          <div className="space-y-2">
            {tops.length === 0 ? (
              <div className="rounded-lg border border-gray-200 bg-gray-50 px-3 py-3 text-sm text-gray-600">
                Keine TOPs vorhanden.
              </div>
            ) : tops.map((top, index) => {
              const color = getColor(index);
              const isSelected = selectedTop === index;
              return (
                <button
                  key={index}
                  onClick={() => setSelectedTop(index)}
                  className={`w-full text-left px-3 py-3 rounded-lg border-2 transition-all ${
                    isSelected
                      ? `${color.bg} ${color.border} ${color.text}`
                      : 'border-transparent hover:bg-gray-50'
                  }`}
                >
                  <div className="flex items-start gap-2">
                    <div className={`w-3 h-3 rounded-full mt-1 ${color.dot}`} />
                    <div className="flex-1 min-w-0">
                      <div
                        className="font-medium text-sm truncate"
                        title={top || 'Unbenannter Tagesordnungspunkt'}
                      >
                        {top || 'Unbenannter Tagesordnungspunkt'}
                      </div>
                      <div className="text-xs text-gray-500 mt-1">
                        {counts[index]} Zeilen
                      </div>
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        {/* Transcript */}
        <div className="flex-1 bg-white rounded-lg border border-gray-200 overflow-hidden flex flex-col">
          {/* Audio Player */}
          {audioUrl && (
            <div className="px-4 py-3 border-b border-gray-200 bg-gray-50">
              <AudioPlayer
                audioUrl={audioUrl}
                currentTime={seekTime}
                onTimeUpdate={handleTimeUpdate}
              />
            </div>
          )}

          <div className="px-4 py-3 border-b border-gray-200 bg-gray-50">
            <h3 className="font-medium text-gray-900">Transkript</h3>
          </div>
          <div ref={transcriptContainerRef} className="flex-1 overflow-y-auto p-2">
            {transcript.map((line, index) => {
              const assignedTo = assignments[index] ?? null;
              const color = assignedTo !== null ? getColor(assignedTo) : null;
              const isCurrentLine = index === currentLineIndex;
              const isSelectedLine = index === selectedLineIndex;
              const detectionSegment = getDetectionSegmentForLine(index);
              return (
                <div
                  key={index}
                  onClick={(e) => handleLineClick(index, e)}
                  className={`px-3 py-2 rounded cursor-pointer transition-colors text-sm border-l-4 mb-1 ${
                    color
                      ? `${color.bg} ${color.border} hover:opacity-80`
                      : 'border-transparent hover:bg-gray-100'
                  } ${isCurrentLine ? 'ring-2 ring-blue-500 ring-offset-1' : ''} ${
                    isSelectedLine ? 'outline outline-2 outline-gray-800' : ''
                  } ${
                    detectionSegment?.uncertain ? 'ring-2 ring-yellow-400 ring-offset-1' : ''
                  }`}
                >
                  <span className="font-medium text-gray-600">
                    {getDisplayName(line.speaker)}:
                  </span>{' '}
                  {editingLine === index ? (
                    <div
                      className="mt-2 space-y-2"
                      onClick={(event) => event.stopPropagation()}
                    >
                      <textarea
                        value={editText}
                        onChange={(event) => setEditText(event.target.value)}
                        className="w-full min-h-20 p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                        aria-label={`Transkriptzeile ${index + 1} korrigieren`}
                      />
                      <p className="text-xs text-gray-500">
                        Mehrere Zeilen im Feld werden beim Speichern als getrennte Transkriptzeilen übernommen.
                      </p>
                      <div className="flex gap-2">
                        <button
                          type="button"
                          onClick={saveLineEdit}
                          className="px-3 py-1 text-xs bg-green-600 text-white rounded hover:bg-green-700"
                        >
                          Speichern
                        </button>
                        <button
                          type="button"
                          onClick={cancelLineEdit}
                          className="px-3 py-1 text-xs bg-gray-200 text-gray-700 rounded hover:bg-gray-300"
                        >
                          Abbrechen
                        </button>
                      </div>
                    </div>
                  ) : (
                    <>
                      <span className="text-gray-800">{line.text}</span>
                      {detectionSegment?.uncertain && (
                        <span className="ml-2 inline-flex items-center px-2 py-0.5 rounded bg-yellow-200 text-yellow-900 text-xs font-medium">
                          Unsicher
                        </span>
                      )}
                      <button
                        type="button"
                        onClick={(event) => {
                          event.stopPropagation();
                          startLineEdit(index, line.text);
                        }}
                        className="ml-2 text-xs text-gray-500 hover:text-blue-600"
                      >
                        Bearbeiten
                      </button>
                    </>
                  )}
                  <span className="ml-2 text-xs text-gray-400">
                    [{formatTime(line.start)}]
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Actions */}
      <div className="flex justify-between">
        <button
          onClick={onBack}
          className="px-6 py-3 rounded-lg font-medium text-gray-600 hover:bg-gray-100 transition-colors flex items-center gap-2"
        >
          <span>←</span>
          Zurück
        </button>
        <button
          onClick={onNext}
          disabled={!canProceed}
          className={`px-6 py-3 rounded-lg font-medium transition-colors flex items-center gap-2 ${
            canProceed
              ? 'bg-blue-600 text-white hover:bg-blue-700'
              : 'bg-gray-200 text-gray-400 cursor-not-allowed'
          }`}
        >
          Zum Protokollentwurf
          <span>→</span>
        </button>
      </div>
    </div>
  );
}
