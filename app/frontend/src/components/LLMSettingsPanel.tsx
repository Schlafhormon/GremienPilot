import { useEffect, useRef } from 'react';
import SpeakerProfileManager from './SpeakerProfileManager';

export interface LLMSettings {
  model: string;
  // Summary preference; retain this key for existing localStorage settings.
  systemPrompt: string;
}

interface LLMSettingsPanelProps {
  isOpen: boolean;
  onClose: () => void;
  settings: LLMSettings;
  onSettingsChange: (settings: LLMSettings) => void;
}

export const DEFAULT_SYSTEM_PROMPT = `Du bist ein Experte für Niederschriften deutscher kommunaler Gremien.

Arbeite protokollarisch, sachlich und verwaltungsnah:
- formale Verwaltungssprache, dritte Person
- paraphrasieren statt wörtlich zitieren
- Beschlüsse, Abstimmungen und Aufträge nur aufnehmen, wenn sie aus dem Transkript hervorgehen
- fehlende oder unklare Informationen als Unsicherheit markieren
- Geschäftsordnungs- und Technikdetails nur aufnehmen, wenn sie fachlich relevant sind

Strukturiere die Auswertung fachlich nach:
- Diskussion
- Beschluss
- Abstimmung
- Maßnahmen und offene Punkte
- Unsicherheiten`;

export const DEFAULT_LLM_SETTINGS: LLMSettings = {
  model: '',
  systemPrompt: DEFAULT_SYSTEM_PROMPT,
};

export default function LLMSettingsPanel({
  isOpen,
  onClose,
  settings,
  onSettingsChange,
}: LLMSettingsPanelProps) {
  const panelRef = useRef<HTMLDivElement>(null);

  // Close on escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    if (isOpen) {
      document.addEventListener('keydown', handleEscape);
      return () => document.removeEventListener('keydown', handleEscape);
    }
  }, [isOpen, onClose]);

  // Close on click outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (panelRef.current && !panelRef.current.contains(e.target as Node)) {
        onClose();
      }
    };
    if (isOpen) {
      // Delay to prevent immediate close on the same click that opened it
      setTimeout(() => {
        document.addEventListener('mousedown', handleClickOutside);
      }, 0);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [isOpen, onClose]);

  const handlePromptChange = (systemPrompt: string) => {
    onSettingsChange({ ...settings, systemPrompt });
  };

  const handleResetPrompt = () => {
    onSettingsChange({ ...settings, systemPrompt: DEFAULT_SYSTEM_PROMPT });
  };

  if (!isOpen) return null;

  return (
    <>
      {/* Backdrop */}
      <div className="fixed inset-0 bg-black/20 z-40 transition-opacity" />

      {/* Panel */}
      <div
        ref={panelRef}
        className="fixed right-0 top-0 h-full w-full max-w-2xl bg-white shadow-xl z-50 flex flex-col transform transition-transform duration-200 ease-out"
        style={{ transform: isOpen ? 'translateX(0)' : 'translateX(100%)' }}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200">
          <h2 className="text-lg font-semibold text-gray-900">KI-Einstellungen</h2>
          <button
            onClick={onClose}
            className="p-1 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <svg className="w-5 h-5 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* System Prompt */}
          <div>
            <div className="flex items-center justify-between mb-3">
              <label className="block text-sm font-medium text-gray-700">
                System-Prompt
              </label>
              <button
                onClick={handleResetPrompt}
                className="text-xs text-blue-600 hover:text-blue-700 flex items-center gap-1"
              >
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
                Standard
              </button>
            </div>
            <textarea
              value={settings.systemPrompt}
              onChange={(e) => handlePromptChange(e.target.value)}
              aria-label="System-Prompt"
              rows={16}
              className="w-full px-3 py-2 text-sm border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none resize-none"
              placeholder="System-Prompt eingeben..."
            />
            <p className="mt-2 text-xs text-gray-500">
              Der System-Prompt definiert, wie die KI die Zusammenfassungen erstellt.
            </p>
          </div>

          <SpeakerProfileManager />
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-gray-200 bg-gray-50">
          <p className="text-xs text-gray-500 text-center">
            Einstellungen werden automatisch gespeichert
          </p>
        </div>
      </div>
    </>
  );
}
