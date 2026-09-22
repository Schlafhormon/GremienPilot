import { API_BASE } from '../api';
import type { ProcessingStepProps } from '../types';

const PIPELINE_STAGES = [
  { id: 'upload', label: 'Audio und optionale TOPs hochladen' },
  { id: 'transcribe', label: 'Transkription und Sprechertrennung' },
  { id: 'speaker_match', label: 'Sprecher abgleichen' },
  { id: 'agenda_detect', label: 'TOPs und Segmentgrenzen erkennen' },
  { id: 'summarize', label: 'Protokollentwurf erzeugen' },
  { id: 'ready_for_review', label: 'Bereit zur Prüfung' },
];

export default function ProcessingStep({
  progress,
  status,
  pipeline,
  canCancel = false,
  onCancel,
}: ProcessingStepProps) {
  const activeStage = pipeline?.stage ?? 'transcribe';
  const activeIndex = Math.max(
    0,
    PIPELINE_STAGES.findIndex((step) => step.id === activeStage)
  );
  const normalizedProgress = Math.max(0, Math.min(100, progress));
  const warnings = pipeline?.warnings ?? [];
  const steps = PIPELINE_STAGES.map((step, index) => ({
    ...step,
    done: normalizedProgress >= 100 || index < activeIndex,
    active: normalizedProgress < 100 && index === activeIndex,
  }));

  const executionLabels = {
    queued: 'Verarbeitung wartet...', running: 'Verarbeitung läuft...',
    retry_wait: 'Vorübergehend gestört – erneuter Versuch folgt',
    review_required: 'Verarbeitung benötigt Prüfung', completed: 'Verarbeitung abgeschlossen',
    failed: 'Verarbeitung technisch unvollständig', cancelled: 'Verarbeitung abgebrochen',
    superseded: 'Eingaben wurden geändert – Ergebnis nicht übernommen',
  };
  const headline = pipeline?.execution ? executionLabels[pipeline.execution.state] :
    pipeline?.status === 'pending'
      ? 'Verarbeitung wartet...'
      : pipeline?.status === 'completed'
        ? 'Verarbeitung abgeschlossen'
        : pipeline?.status === 'failed' ? 'Verarbeitung technisch unvollständig' : pipeline?.status === 'cancelled' ? 'Verarbeitung abgebrochen' : 'Verarbeitung läuft...';

  const stageStatus =
    status ||
    PIPELINE_STAGES.find((step) => step.id === activeStage)?.label ||
    'Pipeline wird verarbeitet...';

  const statusClass =
    pipeline?.status === 'failed'
      ? 'bg-red-50 text-red-800'
      : pipeline?.status === 'cancelled'
        ? 'bg-gray-50 text-gray-700'
        : 'bg-blue-50 text-blue-800';

  const canShowCancel =
    canCancel &&
    onCancel &&
    pipeline?.status !== 'completed' &&
    pipeline?.status !== 'failed' &&
    pipeline?.status !== 'cancelled';

  return (
    <div className="max-w-lg mx-auto">
      <div className="bg-white rounded-lg border border-gray-200 p-8">
        <div className="mb-6 flex items-start justify-between gap-4">
          <h2 className="text-xl font-medium text-gray-900">
            {headline}
          </h2>
          {canShowCancel && (
            <button
              type="button"
              onClick={onCancel}
              className="rounded-lg border border-gray-300 px-3 py-1.5 text-sm text-gray-700 hover:bg-gray-50"
            >
              Abbrechen
            </button>
          )}
        </div>

        {/* Progress Bar */}
        <div className="mb-8">
          <div className="h-3 bg-gray-200 rounded-full overflow-hidden">
            <div
              className="h-full bg-blue-500 transition-all duration-500"
              style={{ width: `${normalizedProgress}%` }}
            />
          </div>
          <div className="text-center mt-2 text-sm text-gray-600">
            {normalizedProgress}%
          </div>
        </div>

        {/* Steps */}
        <div className="space-y-3">
          {steps.map((step, index) => (
            <div key={step.id} className="flex items-center gap-3">
              <div
                className={`w-6 h-6 rounded-full flex items-center justify-center text-xs ${
                  step.done
                    ? 'bg-green-500 text-white'
                    : step.active
                      ? 'bg-blue-500 text-white animate-pulse'
                      : 'bg-gray-200 text-gray-400'
                }`}
              >
                {step.done ? '✓' : index + 1}
              </div>
              <span
                className={`text-sm ${
                  step.done
                    ? 'text-green-600'
                    : step.active
                      ? 'text-blue-700 font-medium'
                      : 'text-gray-500'
                }`}
              >
                {step.label}
              </span>
            </div>
          ))}
        </div>

        {/* Status Message */}
        {stageStatus && (
          <div className={`mt-6 p-4 rounded-lg ${statusClass}`}>
            <p className="text-sm">{stageStatus}</p>
            {pipeline?.execution?.progress?.page && <p className="text-sm">
              PDF: Seite {pipeline.execution.progress.page}/{pipeline.execution.progress.total_pages} ·
              {(pipeline.execution.progress.pdf_phase ?? pipeline.execution.progress.phase) === 'pdf_review' ? ' Nachprüfung' : ' Auswertung'}
              {pipeline.execution.progress.round ? ` (Runde ${pipeline.execution.progress.round})` : ''}
            </p>}
            {pipeline?.execution?.progress?.phase === 'loading' && <p className="text-sm">Modell lädt oder verarbeitet die Eingabe; noch keine Ausgabe empfangen.</p>}
            {pipeline?.execution?.progress?.last_delta_at != null && <p className="text-sm">Letzte Modellausgabe vor {Math.round(pipeline.execution.progress.silence_seconds ?? 0)} Sekunden.</p>}
          </div>
        )}

        {warnings.length > 0 && (
          <div className="mt-4 rounded-lg border border-yellow-200 bg-yellow-50 p-3 text-sm text-yellow-900">
            <div className="font-medium">Hinweise</div>
            <ul className="mt-1 list-disc pl-5">
              {warnings.map((warning, index) => (
                <li key={`${warning}-${index}`}>{warning}</li>
              ))}
            </ul>
          </div>
        )}

        {pipeline?.execution?.documents?.map(document => <p key={document.sha256} className="mt-3 text-sm">
          <a className="text-blue-700 underline" target="_blank" rel="noreferrer"
            href={`${API_BASE}/api/model-jobs/${pipeline.execution!.job_id}/documents/${document.sha256}`}>
            Hochgeladene Originaleinladung prüfen
          </a>
        </p>)}
        {/* Tip */}
        <div className="mt-8 text-center text-sm text-gray-500">
          Sie können die Sitzung später fortsetzen.
        </div>
      </div>
    </div>
  );
}
