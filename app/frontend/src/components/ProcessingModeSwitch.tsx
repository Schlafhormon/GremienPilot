import type { ProcessingMode } from '../types';

export default function ProcessingModeSwitch({ mode, onChange, disabled = false, existingResults = false }: {
  mode: ProcessingMode;
  onChange: (mode: ProcessingMode) => void;
  disabled?: boolean;
  existingResults?: boolean;
}) {
  return <div className="mb-4 rounded-lg border border-gray-200 bg-white p-4">
    <div className="flex flex-wrap items-center justify-between gap-3">
      <span className="font-medium text-gray-900">Verarbeitungsmodus dieser Sitzung</span>
      <div className="flex items-center gap-3">
        <span className={mode === 'fast' ? 'font-semibold text-blue-700' : 'text-gray-500'}>Fast</span>
        <button type="button" role="switch" aria-label="Slow-Modus" aria-checked={mode === 'slow'}
          disabled={disabled} onClick={() => onChange(mode === 'slow' ? 'fast' : 'slow')}
          className="relative h-7 w-12 rounded-full bg-blue-600 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-600 disabled:cursor-not-allowed disabled:opacity-50">
          <span className={`absolute top-1 h-5 w-5 rounded-full bg-white transition-transform ${mode === 'slow' ? 'left-1 translate-x-5' : 'left-1'}`} />
        </button>
        <span className={mode === 'slow' ? 'font-semibold text-blue-700' : 'text-gray-500'}>Slow</span>
      </div>
    </div>
    <p className="mt-2 text-sm text-gray-600" aria-live="polite">{mode === 'fast'
      ? 'Schnell, ohne zusätzliche KI-Prüfung. Ungenauere Ergebnisse möglich.'
      : 'Gründlich, mit Quellenprüfung und Korrekturdurchläufen. Dauert länger.'}</p>
    {disabled && <p className="mt-1 text-xs text-gray-500">Während der Verarbeitung bleibt der Modus festgelegt.</p>}
    {existingResults && <p className="mt-1 text-xs text-gray-500">Die Auswahl gilt für neue Berechnungen. Vorhandene Ergebnisse behalten ihren bisherigen Prüfstand.</p>}
  </div>;
}
