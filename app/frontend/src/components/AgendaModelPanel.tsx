import { useEffect, useState } from 'react';
import { getAgendaModelSettings, saveAgendaModelSettings } from '../api';
import type { AgendaModelDiagnostics, AgendaModelSettings } from '../types';

export default function AgendaModelPanel() {
  const [status, setStatus] = useState<AgendaModelDiagnostics | null>(null);
  const [draft, setDraft] = useState<AgendaModelSettings | null>(null);
  const [error, setError] = useState('');
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  useEffect(() => {
    let cancelled = false;
    getAgendaModelSettings().then(value => {
      if (!cancelled) { setStatus(value); setDraft(value.settings); }
    }).catch(() => { if (!cancelled) setError('TOP-Modellstatus konnte nicht geladen werden.'); });
    return () => { cancelled = true; };
  }, []);
  const update = (values: Partial<AgendaModelSettings>) => {
    setDraft(previous => previous && { ...previous, ...values });
    setSaved(false);
  };
  const save = async () => {
    if (!draft) return;
    setSaving(true); setError(''); setSaved(false);
    try {
      const value = await saveAgendaModelSettings(draft);
      setStatus(value); setDraft(value.settings); setSaved(true);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Speichern fehlgeschlagen.');
    } finally { setSaving(false); }
  };
  return <section className="space-y-3 border-b border-gray-200 pb-6" aria-label="TOP-Modell">
    <h3 className="font-semibold">Modell für TOP-Zuordnung</h3>
    {error && <p role="alert" className="text-sm text-red-700">{error}</p>}
    {!draft && !error && <p>Modellstatus wird geladen …</p>}
    {draft && <>
      <label className="flex gap-2 items-center">
        <input type="checkbox" checked={draft.enabled} disabled={saving}
          onChange={event => update({ enabled: event.target.checked })} />
        Separates Modell für TOP-Zuordnung verwenden
      </label>
      <p className="text-sm">Gespeichert: {status?.settings.enabled ? 'aktiv' : 'inaktiv'} · TOP-Zuordnung: {status?.effective_model}</p>
      <p className="text-sm">Zusammenfassungen und Neugenerierung: {status?.summary_model}. Die PDF-Extraktion behält ihre bisherige Modellwahl.</p>
      <fieldset disabled={saving} className="space-y-3">
        <label className="block text-sm">Lokales Ollama-Modell
          <input className="block w-full border rounded p-2" value={draft.model}
            onChange={event => update({ model: event.target.value })} />
        </label>
        <p className="text-sm" role="status">
          {draft.model === status?.settings.model
            ? `${status.available ? 'Lokal vorhanden' : 'Nicht verfügbar'}: ${status.message}`
            : 'Geändertes Modell wird beim Speichern geprüft.'}
        </p>
        {status?.last_error && <p role="alert" className="text-sm text-red-700">{status.last_error.message}</p>}
        <div className="grid grid-cols-2 gap-3">
          {([
            ['context_tokens', 'Kontextbudget (Tokens)', 8192, 262144],
            ['output_tokens', 'Ausgabebudget Zuordnung', 1024, 16384],
            ['timeline_output_tokens', 'Ausgabebudget Themenverlauf', 1024, 16384],
            ['timeout_seconds', 'Modellladen / erste Antwort (Sekunden)', 10, 7200],
            ['connect_timeout_seconds', 'Verbindungsaufbau (Sekunden)', 1, 120],
            ['idle_timeout_seconds', 'Inaktivität bei Ausgabe (Sekunden)', 10, 7200],
            ['total_timeout_seconds', 'Gesamtlimit je TOP-Aufruf (Sekunden)', 60, 172800],
            ['cpu_threads', 'CPU-Threads', 1, 128],
          ] as const).map(([key, label, min, max]) => <label key={key} className="text-sm">{label}
            <input type="number" min={min} max={max} step={1} className="block w-full border rounded p-2"
              value={draft[key] ?? ({ connect_timeout_seconds: 15, idle_timeout_seconds: 300, total_timeout_seconds: 43200 } as Record<string, number>)[key]} onChange={event => update({ [key]: Number(event.target.value) })} />
          </label>)}
        </div>
        <label className="flex gap-2 items-center text-sm">
          <input type="checkbox" checked={draft.thinking} onChange={event => update({ thinking: event.target.checked })} />
          Thinking verwenden (benötigt zusätzliches Ausgabebudget)
        </label>
      </fieldset>
      <p className="text-xs text-gray-600">Das Kontextbudget muss in RAM und VRAM passen. Die Modellgrenze von 256K ist keine Zusage für diesen Rechner. Fehlende oder nicht ladbare Modelle werden nicht automatisch ersetzt.</p>
      <button type="button" disabled={saving} onClick={save} className="px-3 py-2 rounded bg-blue-600 text-white disabled:opacity-50">
        {saving ? 'Speichert …' : 'TOP-Einstellungen speichern'}
      </button>
      {saved && <p role="status" className="text-sm">TOP-Einstellungen auf dem Server gespeichert.</p>}
      <p className="text-xs text-gray-600">Gilt für neue Verarbeitungen und frische TOP-Berechnungen. Bestehende Ergebnisse bleiben erhalten.</p>
    </>}
  </section>;
}
