import { useEffect, useState } from 'react';
import type { ModelJob } from '../api';

function phaseLabel(job: ModelJob): string {
  if (job.state === 'queued') return 'Wartet auf Verarbeitung';
  if (job.state === 'retry_wait') return 'Verbindung vorübergehend gestört; erneuter Versuch folgt';
  if (job.state === 'cancelled') return 'Abgebrochen';
  if (job.state === 'superseded') return 'Eingaben geändert; Ergebnis nicht übernommen';
  if (job.state === 'failed') return 'Fehlgeschlagen';
  if (['completed', 'review_required'].includes(job.state)) return 'Berechnung abgeschlossen';
  const progress = job.progress;
  if (progress?.phase === 'loading') return 'Modell lädt oder verarbeitet die Eingabe';
  const phase = progress?.agenda_phase ?? progress?.pdf_phase ?? progress?.phase ?? '';
  if (phase === 'pdf_merge') return 'Extrahierte TOPs werden zusammengeführt';
  if (phase === 'pdf_review' || phase === 'pdf_relations') return 'PDF-Ergebnis wird anhand der Quellen geprüft';
  if (phase === 'pdf_repair') return 'PDF-Antwort wird erneut angefordert';
  if (phase.includes('context')) return 'Sitzungsverlauf wird aufbereitet';
  if (phase.startsWith('independent')) return 'Unabhängige Quellenprüfung läuft';
  if (phase.startsWith('resolve')) return 'Abweichungen werden geklärt';
  if (phase.includes('discover')) return 'Tagesordnung wird ermittelt';
  if (phase.includes('reconstruct')) return 'Sitzungsverlauf wird rekonstruiert';
  if (phase.includes('detail')) return 'Transkript wird den TOPs zugeordnet';
  return job.kind === 'pdf' ? 'TOPs und Sitzungsdaten werden aus dem PDF extrahiert' : 'TOP-Zuordnung wird berechnet';
}

export default function ModelJobStatus({ title, job, busy, error, onCancel }: {
  title: string; job?: ModelJob | null; busy: boolean; error?: string | null; onCancel?: () => void;
}) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!busy) return;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [busy]);
  if (!job && !busy && !error) return null;
  const completed = job && ['completed', 'review_required'].includes(job.state);
  const progress = job?.progress;
  const lines = progress?.total_lines;
  const processed = progress?.processed_lines;
  const percent = completed ? 100 : lines && processed !== undefined && processed > 0
    ? Math.min(99, Math.round(100 * processed / lines)) : undefined;
  const seconds = job?.created_at
    ? Math.max(0, Math.floor(((busy ? now / 1000 : job.updated_at ?? now / 1000) - job.created_at))) : null;
  return <div className="mt-3 rounded-lg border border-blue-200 bg-blue-50 p-3 text-sm" aria-label={`${title}: Status`}>
    <div className="flex items-start justify-between gap-3">
      <div role="status" aria-live="polite">
        <p className="font-medium">{job ? phaseLabel(job) : 'Auftrag wird gestartet …'}</p>
        {lines !== undefined && <p>{completed ? lines : processed ?? 0} von {lines} Transkriptzeilen verarbeitet</p>}
        {progress?.page !== undefined && <p>PDF-Seite {progress.page}{progress.total_pages ? ` von ${progress.total_pages}` : ''}</p>}
        {seconds !== null && <p>Laufzeit: {Math.floor(seconds / 60)}:{String(seconds % 60).padStart(2, '0')} Min.</p>}
        {busy && (progress?.silence_seconds ?? 0) > 30 && <p>Warte auf die nächste Modellausgabe …</p>}
      </div>
      {busy && onCancel && <button type="button" onClick={onCancel}
        className="rounded border border-blue-300 bg-white px-3 py-2" aria-label={`${title} abbrechen`}>Abbrechen</button>}
    </div>
    {(busy || completed) && <div role="progressbar" aria-label={`${title}: Fortschritt`}
      aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent}
      aria-valuetext={percent === undefined ? 'Verarbeitung läuft; Gesamtfortschritt noch nicht bestimmbar' : `${percent} %`}
      className="mt-3 h-2 overflow-hidden rounded bg-blue-100">
      <div className={`h-full rounded bg-blue-600 ${percent === undefined ? 'w-1/3 animate-pulse' : 'transition-all'}`}
        style={percent === undefined ? undefined : { width: `${percent}%` }} />
    </div>}
    {(error || job?.error) && <p role="alert" className="mt-2 text-red-800">{error || job?.error}</p>}
  </div>;
}
