import { useState, useRef, type DragEvent, type ChangeEvent } from 'react';
import type { UploadStepProps } from '../types';
import { extractAgendaDataFromPDF } from '../api';

export default function UploadStep({
  onNext,
  audioFile,
  setAudioFile,
  pdfFile,
  setPdfFile,
  tops,
  setTops,
  llmSettings,
  rememberSpeakers,
  setRememberSpeakers,
  skipAgendaDetection,
  setSkipAgendaDetection,
  autoDetectTopsFromPdf,
  setAutoDetectTopsFromPdf,
  exportMetadata,
  setExportMetadata,
}: UploadStepProps) {
  const audioInputRef = useRef<HTMLInputElement>(null);
  const pdfInputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);
  const [pdfDragActive, setPdfDragActive] = useState(false);
  const [showMetadata, setShowMetadata] = useState(false);
  const [showManualTops, setShowManualTops] = useState(false);

  // PDF extraction state
  const [isExtractingTops, setIsExtractingTops] = useState(false);
  const [extractionError, setExtractionError] = useState<string | null>(null);
  const [extractedCount, setExtractedCount] = useState<number | null>(null);

  // Audio drag handlers
  const handleDrag = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith('audio/') || file.name.match(/\.(mp3|wav|m4a)$/i)) {
        setAudioFile(file);
      }
    }
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setAudioFile(e.target.files[0]);
    }
  };

  // PDF drag handlers
  const handlePdfDrag = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setPdfDragActive(true);
    } else if (e.type === 'dragleave') {
      setPdfDragActive(false);
    }
  };

  const handlePdfDrop = async (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setPdfDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type === 'application/pdf' || file.name.endsWith('.pdf')) {
        setPdfFile(file);
        if (autoDetectTopsFromPdf) {
          setSkipAgendaDetection(false);
          setExtractionError(null);
          setExtractedCount(null);
        } else {
          await extractTopsFromFile(file);
        }
      }
    }
  };

  const handlePdfFileChange = async (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setPdfFile(file);
      if (autoDetectTopsFromPdf) {
        setSkipAgendaDetection(false);
        setExtractionError(null);
        setExtractedCount(null);
      } else {
        await extractTopsFromFile(file);
      }
    }
    // Reset the input so the same file can be selected again
    e.target.value = '';
  };

  const extractTopsFromFile = async (file: File) => {
    setIsExtractingTops(true);
    setExtractionError(null);
    setExtractedCount(null);

    try {
      const extracted = await extractAgendaDataFromPDF(file, {
        model: llmSettings?.model,
      });
      const extractedTops = extracted.tops.map((top) => top.trim()).filter(Boolean);
      applyDetectedMetadata(extracted.metadata ?? {});

      if (extractedTops.length > 0) {
        // Replace empty TOPs with extracted ones, but keep user-entered ones
        setTops(extractedTops);
        setSkipAgendaDetection(false);
        setAutoDetectTopsFromPdf(false);
        setShowManualTops(true);
        setExtractedCount(extractedTops.length);
      } else {
        setExtractionError('Keine TOPs im PDF gefunden. Bitte manuell eingeben.');
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unbekannter Fehler';
      setExtractionError(errorMessage);
    } finally {
      setIsExtractingTops(false);
    }
  };

  // TOP management
  const addTop = () => {
    setTops([...(tops.length ? tops : []), '']);
    setSkipAgendaDetection(false);
    setAutoDetectTopsFromPdf(false);
    setShowManualTops(true);
    setExtractedCount(null); // Clear success message when user modifies
  };

  const updateTop = (index: number, value: string) => {
    const newTops = [...tops];
    newTops[index] = value;
    setTops(newTops);
    if (value.trim()) {
      setSkipAgendaDetection(false);
      setAutoDetectTopsFromPdf(false);
      setShowManualTops(true);
    }
    setExtractedCount(null); // Clear success message when user modifies
  };

  const removeTop = (index: number) => {
    if (tops.length > 1) {
      setTops(tops.filter((_, i) => i !== index));
    } else {
      setTops([]);
      setSkipAgendaDetection(true);
      setAutoDetectTopsFromPdf(false);
    }
    setExtractedCount(null); // Clear success message when user modifies
  };

  const clearAllTops = () => {
    setTops([]);
    setSkipAgendaDetection(true);
    setAutoDetectTopsFromPdf(false);
    setExtractedCount(null);
    setExtractionError(null);
  };

  const updateExportMetadata = (patch: Partial<typeof exportMetadata>) => {
    setExportMetadata({ ...exportMetadata, ...patch });
  };

  const applyDetectedMetadata = (metadata: {
    committee?: string;
    date?: string;
    location?: string;
    title?: string;
  }) => {
    const patch: Partial<typeof exportMetadata> = {};
    const defaultDate = new Date().toISOString().slice(0, 10);
    if (metadata.committee?.trim() && !exportMetadata.committee.trim()) {
      patch.committee = metadata.committee.trim();
    }
    if (
      metadata.date?.trim() &&
      (
        !exportMetadata.date.trim() ||
        exportMetadata.date === defaultDate ||
        (
          !exportMetadata.committee.trim() &&
          !exportMetadata.location.trim() &&
          exportMetadata.title === 'Sitzungsprotokoll'
        )
      )
    ) {
      patch.date = metadata.date.trim();
    }
    if (metadata.location?.trim() && !exportMetadata.location.trim()) {
      patch.location = metadata.location.trim();
    }
    if (
      metadata.title?.trim() &&
      (!exportMetadata.title.trim() || exportMetadata.title === 'Sitzungsprotokoll')
    ) {
      patch.title = metadata.title.trim();
    }
    if (Object.keys(patch).length > 0) {
      setExportMetadata({ ...exportMetadata, ...patch });
    }
  };

  const canProceed = !!audioFile;
  const validTopCount = tops.filter((top) => top.trim() !== '').length;
  const hasMetadata =
    Boolean(exportMetadata.committee.trim()) ||
    Boolean(exportMetadata.location.trim()) ||
    Boolean(exportMetadata.title.trim() && exportMetadata.title !== 'Sitzungsprotokoll') ||
    exportMetadata.participants.length > 0;
  const automationMode = skipAgendaDetection
    ? 'Gesamtes Gespräch ohne TOP-Zuordnung'
    : validTopCount > 0
      ? `${validTopCount} manuell vorbereitete TOPs`
      : pdfFile && autoDetectTopsFromPdf
        ? 'TOPs aus PDF in der Pipeline erkennen'
        : 'TOPs automatisch aus dem Transkript erkennen';
  const shouldShowManualTops =
    showManualTops ||
    validTopCount > 0 ||
    skipAgendaDetection ||
    !autoDetectTopsFromPdf;

  return (
    <div className="space-y-6">
      <div className="rounded-lg border border-blue-200 bg-blue-50 p-5">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <p className="text-sm font-medium text-blue-700">Automatischer Protokolllauf</p>
            <h2 className="mt-1 text-2xl font-semibold text-gray-950">
              Aufnahme hochladen, Verarbeitung starten, nur offene Punkte prüfen.
            </h2>
            <p className="mt-2 max-w-3xl text-sm text-blue-900">
              Die Pipeline transkribiert, erkennt Sprecher, ordnet TOPs zu und erzeugt
              den Protokollentwurf. Danach bleiben nur Sprecher- und TOP-Prüfung sowie
              eventuelle Warnhinweise.
            </p>
          </div>
          <div className="rounded-md border border-blue-200 bg-white px-4 py-3 text-sm text-gray-700 lg:w-72">
            <div className="font-medium text-gray-900">Aktueller Modus</div>
            <div className="mt-1">{automationMode}</div>
            <div className="mt-2 text-xs text-gray-500">
              {rememberSpeakers
                ? 'Sprecherprofile werden nach Bestätigung genutzt.'
                : 'Sprecherprofile bleiben für diesen Lauf lokal.'}
            </div>
          </div>
        </div>
      </div>

      {/* Upload Section */}
      <div className="grid gap-4 lg:grid-cols-[minmax(0,1.1fr)_minmax(0,0.9fr)]">
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <h2 className="text-lg font-medium text-gray-900 mb-4 flex items-center gap-2">
            <span className="text-xl">🎙️</span>
            Audioaufnahme
          </h2>

          <div
            className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
              dragActive
                ? 'border-blue-500 bg-blue-50'
                : 'border-gray-300 hover:border-gray-400'
            }`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => audioInputRef.current?.click()}
          >
            <input
              ref={audioInputRef}
              type="file"
              accept="audio/*"
              onChange={handleFileChange}
              className="hidden"
            />

            {audioFile ? (
              <div className="flex flex-wrap items-center justify-center gap-3">
                <span className="text-green-500 text-2xl">✓</span>
                <span className="text-gray-700 font-medium">{audioFile.name}</span>
                <span className="text-gray-500">
                  ({(audioFile.size / 1024 / 1024).toFixed(1)} MB)
                </span>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setAudioFile(null);
                  }}
                  className="ml-2 text-red-500 hover:text-red-700"
                  aria-label="Audiodatei entfernen"
                >
                  ✕
                </button>
              </div>
            ) : (
              <div className="cursor-pointer">
                <div className="text-4xl mb-2">📁</div>
                <p className="text-gray-600">
                  Datei hierher ziehen oder klicken zum Auswählen
                </p>
                <p className="text-gray-400 text-sm mt-1">
                  MP3, WAV, M4A (max 500MB)
                </p>
              </div>
            )}
          </div>
        </div>

        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <h2 className="text-lg font-medium text-gray-900 mb-4 flex items-center gap-2">
            <span className="text-xl">📄</span>
            Einladung oder Tagesordnung
          </h2>
          <div
            className={`border-2 border-dashed rounded-lg p-5 text-center transition-all ${
              isExtractingTops
                ? 'border-blue-300 bg-blue-50'
                : pdfDragActive
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-200 hover:border-blue-300 hover:bg-gray-50'
            }`}
            onDragEnter={handlePdfDrag}
            onDragLeave={handlePdfDrag}
            onDragOver={handlePdfDrag}
            onDrop={handlePdfDrop}
            onClick={() => !isExtractingTops && pdfInputRef.current?.click()}
          >
            <input
              ref={pdfInputRef}
              type="file"
              accept=".pdf,application/pdf"
              onChange={handlePdfFileChange}
              className="hidden"
              disabled={isExtractingTops}
            />

            {isExtractingTops ? (
              <div className="flex items-center justify-center gap-3 py-3">
                <svg
                  className="animate-spin h-5 w-5 text-blue-600"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  />
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  />
                </svg>
                <span className="text-blue-700 font-medium">
                  TOPs werden extrahiert...
                </span>
              </div>
            ) : (
              <div className="cursor-pointer flex items-center justify-center gap-3 py-3">
                <span className="text-2xl">📄</span>
                <div className="text-left">
                  <p className="text-gray-700 font-medium">
                    PDF-Einladung hochladen
                  </p>
                  <p className="text-gray-400 text-sm">
                    {autoDetectTopsFromPdf
                      ? 'PDF wird in der Pipeline ausgewertet'
                      : 'TOPs werden automatisch extrahiert'}
                  </p>
                </div>
              </div>
            )}
          </div>

          {pdfFile && !isExtractingTops && (
            <div className="mt-3 flex items-center justify-between gap-3 rounded-lg border border-gray-200 bg-gray-50 px-3 py-2 text-sm">
              <span className="truncate text-gray-700">
                PDF für die automatische Verarbeitung: {pdfFile.name}
              </span>
              <button
                type="button"
                onClick={() => {
                  setPdfFile(null);
                  setExtractionError(null);
                  setExtractedCount(null);
                }}
                className="shrink-0 text-gray-500 hover:text-red-600"
              >
                Entfernen
              </button>
            </div>
          )}

          {extractionError && (
            <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-lg flex items-start gap-2">
              <span className="text-red-500 mt-0.5">⚠️</span>
              <div>
                <p className="text-red-700 text-sm">{extractionError}</p>
                <button
                  onClick={() => setExtractionError(null)}
                  className="text-red-600 text-sm underline hover:no-underline mt-1"
                >
                  Ausblenden
                </button>
              </div>
            </div>
          )}

          {extractedCount !== null && (
            <div className="mt-3 p-3 bg-green-50 border border-green-200 rounded-lg flex items-center gap-2">
              <span className="text-green-500">✓</span>
              <p className="text-green-700 text-sm">
                {extractedCount} {extractedCount === 1 ? 'TOP' : 'TOPs'} erfolgreich extrahiert.
                Sie können die Liste unten bearbeiten.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Automation choices */}
      <div className="bg-white rounded-lg border border-gray-200 p-6">
        <div className="mb-4 flex flex-col gap-1">
          <h2 className="text-lg font-medium text-gray-900">Automatisierung</h2>
          <p className="text-sm text-gray-500">
            Standard ist: alles automatisch auswerten und nur markierte Unsicherheiten prüfen.
          </p>
        </div>

        <div className="grid gap-3 lg:grid-cols-3">
          <label className="flex items-start gap-3 rounded-lg border border-blue-200 bg-blue-50 p-3 text-sm">
            <input
              type="checkbox"
              checked={autoDetectTopsFromPdf}
              onChange={(event) => {
                const enabled = event.target.checked;
                setAutoDetectTopsFromPdf(enabled);
                if (enabled) {
                  setSkipAgendaDetection(false);
                  setExtractionError(null);
                  setExtractedCount(null);
                }
              }}
              disabled={skipAgendaDetection}
              className="mt-1 h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500 disabled:opacity-50"
            />
            <span>
              <span className="block font-medium text-gray-900">
                TOPs automatisch aus PDF erkennen und direkt verarbeiten
              </span>
              <span className="block text-gray-600">
                PDF und Transkript werden in der Pipeline zusammen ausgewertet.
              </span>
            </span>
          </label>

          <label className="flex items-start gap-3 rounded-lg border border-gray-200 bg-gray-50 p-3 text-sm">
            <input
              type="checkbox"
              checked={skipAgendaDetection}
              onChange={(event) => {
                const enabled = event.target.checked;
                setSkipAgendaDetection(enabled);
                if (enabled) {
                  setAutoDetectTopsFromPdf(false);
                  setTops([]);
                  setPdfFile(null);
                  setExtractedCount(null);
                  setExtractionError(null);
                } else if (tops.length === 0) {
                  setTops(['']);
                }
              }}
              className="mt-1 h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
            />
            <span>
              <span className="block font-medium text-gray-900">
                Ohne TOPs und ohne automatische TOP-Erkennung fortfahren
              </span>
              <span className="block text-gray-600">
                Das gesamte Transkript wird als ein Gespräch zusammengefasst.
              </span>
            </span>
          </label>

          <label className="flex items-start gap-3 rounded-lg border border-gray-200 bg-gray-50 p-3 text-sm cursor-pointer">
            <input
              type="checkbox"
              checked={rememberSpeakers}
              onChange={(event) => setRememberSpeakers(event.target.checked)}
              className="mt-1 h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
            />
            <span>
              <span className="block font-medium text-gray-900">
                Sprecher dauerhaft merken
              </span>
              <span className="block text-gray-600">
                Vorschläge werden später geprüft, bevor Profile gespeichert werden.
              </span>
              <span className="sr-only">
                Dauerhafte Profile und Embeddings werden nur nach ausdrücklicher Sprecheraktion gespeichert.
              </span>
            </span>
          </label>
        </div>
      </div>

      {/* Export metadata */}
      <div className="bg-white rounded-lg border border-gray-200 p-6">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h2 className="text-lg font-medium text-gray-900 flex items-center gap-2">
              <span className="text-xl">📝</span>
              Sitzungsdaten für den Export
            </h2>
            <p className="mt-1 text-sm text-gray-500">
              Optional jetzt erfassen, sonst am Ende vor dem Export prüfen.
            </p>
          </div>
          <button
            type="button"
            onClick={() => setShowMetadata(!showMetadata)}
            className="rounded-lg border border-gray-300 px-3 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50"
          >
            {showMetadata ? 'Sitzungsdaten ausblenden' : hasMetadata ? 'Sitzungsdaten bearbeiten' : 'Sitzungsdaten ergänzen'}
          </button>
        </div>
        {!showMetadata && (
          <div className="mt-4 grid gap-3 text-sm text-gray-600 md:grid-cols-3">
            <div className="rounded-md bg-gray-50 px-3 py-2">
              <span className="block text-xs text-gray-400">Gremium</span>
              {exportMetadata.committee || 'Noch offen'}
            </div>
            <div className="rounded-md bg-gray-50 px-3 py-2">
              <span className="block text-xs text-gray-400">Datum</span>
              {exportMetadata.date}
            </div>
            <div className="rounded-md bg-gray-50 px-3 py-2">
              <span className="block text-xs text-gray-400">Teilnehmer</span>
              {exportMetadata.participants.length || 'Noch offen'}
            </div>
          </div>
        )}
        {showMetadata && (
        <div className="mt-4 grid gap-3 md:grid-cols-2">
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
        )}
      </div>

      {/* TOPs Section */}
      <div className="bg-white rounded-lg border border-gray-200 p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-medium text-gray-900 flex items-center gap-2">
            <span className="text-xl">📋</span>
            Tagesordnungspunkte
            <span className="text-sm font-normal text-gray-400">(optional)</span>
          </h2>
          <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={() => setShowManualTops(!showManualTops)}
            className="text-sm font-medium text-blue-600 hover:text-blue-700"
          >
            {shouldShowManualTops ? 'Manuelle TOPs einklappen' : 'TOPs manuell bearbeiten'}
          </button>
          {tops.some(t => t.trim() !== '') && (
            <button
              onClick={clearAllTops}
              className="text-sm text-gray-500 hover:text-red-500 transition-colors"
            >
              Alle löschen
            </button>
          )}
          </div>
        </div>

        <p className="text-sm text-gray-600 mb-3">
          Originalnummer mit eingeben, z. B. 2.1 Schulbau. Bei wiederholten Nummern
          den Abschnitt voranstellen: [Öffentlich] oder [Nichtöffentlich]. Ohne
          Nummer erfolgt die Zuordnung anhand des Inhalts.
        </p>
        {/* TOPs List */}
        {!shouldShowManualTops ? (
          <div className="rounded-lg border border-gray-200 bg-gray-50 px-4 py-3 text-sm text-gray-600">
            Keine manuelle Eingabe nötig. Die Pipeline erkennt TOPs automatisch oder nutzt das hochgeladene PDF.
          </div>
        ) : (
        <div className="space-y-3">
          {skipAgendaDetection && tops.length === 0 ? (
            <div className="rounded-lg border border-gray-200 bg-gray-50 px-4 py-3 text-sm text-gray-600">
              Keine TOPs angelegt. Die automatische TOP-Erkennung ist deaktiviert.
            </div>
          ) : tops.map((top, index) => (
            <div key={index} className="flex items-center gap-3">
              <span className="text-gray-500 text-sm">TOP</span>
              <input
                type="text"
                value={top}
                onChange={(e) => updateTop(index, e.target.value)}
                aria-label={`Tagesordnungseintrag ${index + 1}`}
                placeholder="Nummer und Titel, z. B. 2.1 Schulbau"
                className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
              <button
                onClick={() => removeTop(index)}
                className="p-2 rounded-lg transition-colors text-gray-400 hover:text-red-500 hover:bg-red-50"
                title="TOP entfernen"
              >
                ✕
              </button>
            </div>
          ))}
        </div>
        )}

        <button
          onClick={addTop}
          className="mt-4 text-blue-600 hover:text-blue-700 font-medium flex items-center gap-2"
        >
          <span className="text-lg">+</span>
          TOP hinzufügen
        </button>
      </div>

      {/* Action Button */}
      <div className="flex justify-end">
        <button
          onClick={onNext}
          disabled={!canProceed}
          className={`px-6 py-3 rounded-lg font-medium transition-colors flex items-center gap-2 ${
            canProceed
              ? 'bg-blue-600 text-white hover:bg-blue-700'
              : 'bg-gray-200 text-gray-400 cursor-not-allowed'
          }`}
        >
          Automatisch verarbeiten
          <span>→</span>
        </button>
      </div>
    </div>
  );
}
