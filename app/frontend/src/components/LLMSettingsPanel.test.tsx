import { fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import LLMSettingsPanel, { DEFAULT_LLM_SETTINGS } from './LLMSettingsPanel';

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
