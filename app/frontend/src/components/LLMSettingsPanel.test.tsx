import { render, screen } from '@testing-library/react';
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
      vi.fn().mockImplementation((url: string) => url.includes('/settings/agenda-model')
        ? Promise.resolve(jsonResponse({ settings: { enabled: false, model: 'gemma4:31b-it-q4_K_M',
            context_tokens: 32768, output_tokens: 4096, timeline_output_tokens: 4096,
            timeout_seconds: 1800, cpu_threads: 8, temperature: 0.1, seed: 42, thinking: false },
            effective_model: 'qwen3.5:9b', summary_model: 'qwen3.5:9b', available: false, message: 'Nicht installiert' }))
        : Promise.resolve(
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
      ))
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
