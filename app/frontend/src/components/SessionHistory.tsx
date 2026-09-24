import { FormEvent, useCallback, useEffect, useState } from 'react';
import { listSessions } from '../api';
import type { SessionHistoryItem, SessionHistoryStatus } from '../types';

interface SessionHistoryProps {
  onOpen: (sessionId: string) => void;
  onNewSession: () => void;
}

const PAGE_SIZE = 20;

const STATUS_OPTIONS: Array<{ value: '' | SessionHistoryStatus; label: string }> = [
  { value: '', label: 'Alle Status' },
  { value: 'processing', label: 'Wird verarbeitet' },
  { value: 'review', label: 'Prüfung offen' },
  { value: 'ready', label: 'Protokoll vorbereitet' },
  { value: 'draft', label: 'Entwurf' },
  { value: 'failed', label: 'Fehlgeschlagen' },
  { value: 'cancelled', label: 'Abgebrochen' },
];

const STATUS_LABELS: Record<SessionHistoryStatus, string> = {
  draft: 'Entwurf',
  processing: 'Wird verarbeitet',
  review: 'Prüfung offen',
  ready: 'Protokoll vorbereitet',
  failed: 'Fehlgeschlagen',
  cancelled: 'Abgebrochen',
};

const STATUS_STYLES: Record<SessionHistoryStatus, string> = {
  draft: 'bg-gray-100 text-gray-700',
  processing: 'bg-blue-100 text-blue-800',
  review: 'bg-yellow-100 text-yellow-800',
  ready: 'bg-green-100 text-green-800',
  failed: 'bg-red-100 text-red-800',
  cancelled: 'bg-gray-200 text-gray-700',
};

function formatTimestamp(timestamp: number): string {
  return new Intl.DateTimeFormat('de-DE', {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(new Date(timestamp * 1000));
}

function formatMeetingDate(value: string): string {
  if (!value) return '';
  const parsed = new Date(`${value}T00:00:00`);
  if (Number.isNaN(parsed.getTime())) return value;
  return new Intl.DateTimeFormat('de-DE', { dateStyle: 'long' }).format(parsed);
}

export default function SessionHistory({ onOpen, onNewSession }: SessionHistoryProps) {
  const [items, setItems] = useState<SessionHistoryItem[]>([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [queryInput, setQueryInput] = useState('');
  const [query, setQuery] = useState('');
  const [status, setStatus] = useState<'' | SessionHistoryStatus>('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadHistory = useCallback(async (showLoading = true) => {
    if (showLoading) setLoading(true);
    try {
      const result = await listSessions({
        limit: PAGE_SIZE,
        offset,
        query,
        status: status || undefined,
      });
      setItems(result.items);
      setTotal(result.total);
      setError(null);
    } catch (loadError) {
      setError(
        loadError instanceof Error
          ? loadError.message
          : 'Sitzungsverlauf konnte nicht geladen werden'
      );
    } finally {
      if (showLoading) setLoading(false);
    }
  }, [offset, query, status]);

  useEffect(() => {
    void loadHistory();
  }, [loadHistory]);

  useEffect(() => {
    const interval = window.setInterval(() => {
      void loadHistory(false);
    }, 10000);
    return () => window.clearInterval(interval);
  }, [loadHistory]);

  const handleSearch = (event: FormEvent) => {
    event.preventDefault();
    setOffset(0);
    setQuery(queryInput.trim());
  };

  const firstVisible = total === 0 ? 0 : offset + 1;
  const lastVisible = Math.min(offset + items.length, total);

  return (
    <section className="space-y-5" aria-labelledby="session-history-title">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h2 id="session-history-title" className="text-2xl font-semibold text-gray-900">
            Sitzungsverlauf
          </h2>
          <p className="mt-1 text-sm text-gray-600">
            Alle gespeicherten Sitzungen sind für Besucher dieser internen Anwendung sichtbar und bearbeitbar.
          </p>
        </div>
        <button
          type="button"
          onClick={onNewSession}
          className="rounded-lg bg-blue-600 px-4 py-2 font-medium text-white hover:bg-blue-700"
        >
          Neue Sitzung
        </button>
      </div>

      <form onSubmit={handleSearch} className="flex flex-col gap-3 rounded-lg border border-gray-200 bg-white p-4 sm:flex-row">
        <label className="flex-1 text-sm font-medium text-gray-700">
          Suchen
          <input
            type="search"
            value={queryInput}
            onChange={(event) => setQueryInput(event.target.value)}
            placeholder="Titel, Gremium, Datum oder Sitzungs-ID"
            className="mt-1 w-full rounded-md border border-gray-300 px-3 py-2 font-normal focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </label>
        <label className="text-sm font-medium text-gray-700 sm:w-56">
          Status
          <select
            value={status}
            onChange={(event) => {
              setOffset(0);
              setStatus(event.target.value as '' | SessionHistoryStatus);
            }}
            className="mt-1 w-full rounded-md border border-gray-300 px-3 py-2 font-normal focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {STATUS_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>{option.label}</option>
            ))}
          </select>
        </label>
        <button
          type="submit"
          className="self-end rounded-md border border-gray-300 px-4 py-2 font-medium text-gray-700 hover:bg-gray-50"
        >
          Suchen
        </button>
      </form>

      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-red-800">
          <p>{error}</p>
          <button type="button" onClick={() => void loadHistory()} className="mt-2 font-medium underline">
            Erneut versuchen
          </button>
        </div>
      )}

      {loading ? (
        <div className="rounded-lg border border-gray-200 bg-white p-8 text-center text-gray-500">
          Sitzungen werden geladen…
        </div>
      ) : items.length === 0 && !error ? (
        <div className="rounded-lg border border-gray-200 bg-white p-8 text-center">
          <p className="font-medium text-gray-800">Keine Sitzungen gefunden</p>
          <p className="mt-1 text-sm text-gray-500">Passen Sie die Suche an oder starten Sie eine neue Sitzung.</p>
        </div>
      ) : (
        <div className="grid gap-4">
          {items.map((session) => (
            <article key={session.session_id} className="rounded-lg border border-gray-200 bg-white p-5 shadow-sm">
              <div className="flex flex-col justify-between gap-4 md:flex-row md:items-center">
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <h3 className="truncate text-lg font-semibold text-gray-900" title={session.title}>
                      {session.title}
                    </h3>
                    <span className={`rounded-full px-2.5 py-1 text-xs font-medium ${STATUS_STYLES[session.status]}`}>
                      {STATUS_LABELS[session.status]}
                    </span>
                  </div>
                  <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-sm text-gray-600">
                    {session.committee && <span>{session.committee}</span>}
                    {session.meeting_date && <span>{formatMeetingDate(session.meeting_date)}</span>}
                    <span>Zuletzt geändert: {formatTimestamp(session.updated_at)}</span>
                  </div>
                  <div className="mt-3 flex flex-wrap gap-3 text-xs text-gray-500">
                    <span>{session.processing_mode === 'fast' ? 'Fast' : 'Slow'}</span>
                    <span>{session.top_count} TOPs</span>
                    <span>{session.transcript_line_count} Transkriptzeilen</span>
                    <span>{session.summary_count} Zusammenfassungen</span>
                    <span>{session.audio_available ? 'Audio verfügbar' : 'Kein Audio verfügbar'}</span>
                  </div>
                  {session.status === 'failed' && <div className="mt-3 text-sm text-red-800" role="status">
                    <p>Verarbeitung fehlgeschlagen · Phase: {({ agenda_detect: 'PDF-/TOP-Prüfung', transcribe: 'Transkription', summarize: 'Zusammenfassung', ready_for_review: 'Abschlussprüfung' } as Record<string, string>)[session.pipeline_stage ?? ''] ?? session.pipeline_stage ?? 'Unbekannt'}</p>
                    {session.pipeline_error && <p>{session.pipeline_error}</p>}
                    <p>Gespeicherte Ergebnisse bleiben erhalten. Kein vollständig geprüftes Protokoll.</p>
                  </div>}
                  {session.status === 'processing' && (
                    <div className="mt-3 h-1.5 max-w-md overflow-hidden rounded-full bg-gray-200">
                      <div
                        className="h-full rounded-full bg-blue-600 transition-all"
                        style={{ width: `${session.pipeline_progress ?? 0}%` }}
                      />
                    </div>
                  )}
                </div>
                <button
                  type="button"
                  onClick={() => onOpen(session.session_id)}
                  className="shrink-0 rounded-lg bg-blue-600 px-4 py-2 font-medium text-white hover:bg-blue-700"
                >
                  Öffnen
                </button>
              </div>
            </article>
          ))}
        </div>
      )}

      {total > PAGE_SIZE && (
        <div className="flex items-center justify-between gap-3 text-sm text-gray-600">
          <span>{firstVisible}–{lastVisible} von {total}</span>
          <div className="flex gap-2">
            <button
              type="button"
              disabled={offset === 0}
              onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
              className="rounded-md border border-gray-300 px-3 py-2 disabled:opacity-40"
            >
              Zurück
            </button>
            <button
              type="button"
              disabled={offset + PAGE_SIZE >= total}
              onClick={() => setOffset(offset + PAGE_SIZE)}
              className="rounded-md border border-gray-300 px-3 py-2 disabled:opacity-40"
            >
              Weiter
            </button>
          </div>
        </div>
      )}
    </section>
  );
}
