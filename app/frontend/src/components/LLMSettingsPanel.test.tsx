import { fireEvent, render, screen } from '@testing-library/react';
import { useState } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import LLMSettingsPanel, { DEFAULT_LLM_SETTINGS, DEFAULT_CUSTOM_SUMMARY_PROMPT, GEMMA_SYSTEM_PROMPT } from './LLMSettingsPanel';

function jsonResponse(data: unknown) {
  return {
    ok: true,
    json: () => Promise.resolve(data),
  };
}

describe('LLMSettingsPanel', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('shows speaker profile management below the system prompt', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValueOnce(
        jsonResponse([
          {
            profile_id: 'rudolf',
            display_name: 'Herr Rudolf',
            scope: null,
            created_at: 1,
            updated_at: 1,
            archived: false,
            embedding_count: 2,
          },
        ])
      )
    );

    render(
      <LLMSettingsPanel
        isOpen
        onClose={vi.fn()}
        settings={DEFAULT_LLM_SETTINGS}
        onSettingsChange={vi.fn()}
      />
    );

    expect(screen.getByLabelText('System-Prompt')).toBeInTheDocument();
    expect(await screen.findByText('Profilverwaltung')).toBeInTheDocument();
    expect(screen.getByText('Herr Rudolf')).toBeInTheDocument();
  });

  it('shows the fixed Gemma prompt even when an old custom prompt is saved', () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse([])));
    render(<LLMSettingsPanel isOpen onClose={vi.fn()}
      settings={{...DEFAULT_LLM_SETTINGS, systemPrompt: 'Alter Prompt'}} onSettingsChange={vi.fn()} />);
    const prompt = screen.getByLabelText('System-Prompt');
    expect(prompt).toHaveValue(GEMMA_SYSTEM_PROMPT);
    expect(prompt).toHaveAttribute('readonly');
    expect(screen.getByRole('button', {name: 'Standard'})).toBeDisabled();
  });

  it('keeps the editable custom prompt across style switches and panel reopening', () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse([])));
    function Settings() {
      const [settings, setSettings] = useState(DEFAULT_LLM_SETTINGS);
      const [open, setOpen] = useState(true);
      return <><button onClick={() => setOpen(!open)}>Toggle</button>
        <LLMSettingsPanel isOpen={open} onClose={() => setOpen(false)} settings={settings} onSettingsChange={setSettings} /></>;
    }
    render(<Settings />);
    fireEvent.change(screen.getByLabelText('Zusammenfassungsstil'), { target: { value: 'gemma4-custom' } });
    expect(screen.getByLabelText('System-Prompt')).toHaveValue(DEFAULT_CUSTOM_SUMMARY_PROMPT);
    expect(screen.getByLabelText('System-Prompt')).not.toHaveAttribute('readonly');
    fireEvent.change(screen.getByLabelText('System-Prompt'), { target: { value: 'Meine Beschlusstabelle' } });
    fireEvent.change(screen.getByLabelText('Zusammenfassungsstil'), { target: { value: 'gemma4-lora' } });
    expect(screen.getByLabelText('System-Prompt')).toHaveValue(GEMMA_SYSTEM_PROMPT);
    fireEvent.click(screen.getByText('Toggle'));
    fireEvent.click(screen.getByText('Toggle'));
    fireEvent.change(screen.getByLabelText('Zusammenfassungsstil'), { target: { value: 'gemma4-custom' } });
    expect(screen.getByLabelText('System-Prompt')).toHaveValue('Meine Beschlusstabelle');
    expect(screen.getByLabelText('Modellüberschreibung')).toHaveValue('');
  });
});


it('shows persisted model overrides and allows returning to the server default', () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse([])));
  const change = vi.fn();
  render(<LLMSettingsPanel isOpen onClose={vi.fn()}
    settings={{...DEFAULT_LLM_SETTINGS, model: 'explicit-model'}} onSettingsChange={change} />);
  expect(screen.getByLabelText('Modellüberschreibung')).toHaveValue('explicit-model');
  fireEvent.click(screen.getByRole('button', {name: 'Servermodell verwenden'}));
  expect(change).toHaveBeenCalledWith({...DEFAULT_LLM_SETTINGS, model: ''});
});
