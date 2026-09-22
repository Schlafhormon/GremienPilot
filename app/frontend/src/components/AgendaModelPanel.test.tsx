import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import AgendaModelPanel from './AgendaModelPanel';

const settings = { enabled: false, model: 'gemma4:31b-it-q4_K_M', context_tokens: 32768,
  output_tokens: 4096, timeline_output_tokens: 4096, timeout_seconds: 1800, cpu_threads: 8,
  temperature: 0.1, seed: 42, thinking: false };
const status = { settings, effective_model: 'qwen3.5:9b', summary_model: 'qwen3.5:9b',
  available: false, message: 'Modell ist nicht lokal installiert.', digest: null, last_error: null };
const response = (body: unknown, ok = true) => ({ ok, json: async () => body });

describe('separate TOP model settings', () => {
  afterEach(() => vi.unstubAllGlobals());
  it('loads saved state, saves only TOP parameters, and restores it on remount', async () => {
    let stored = status;
    const fetch = vi.fn(async (_url: string, init?: RequestInit) => {
      if (init?.method === 'PUT') {
        const next = JSON.parse(String(init.body));
        stored = { ...status, settings: next, effective_model: next.model };
      }
      return response(stored);
    });
    vi.stubGlobal('fetch', fetch);
    const { unmount } = render(<AgendaModelPanel />);
    const toggle = await screen.findByLabelText('Separates Modell für TOP-Zuordnung verwenden');
    expect(toggle).not.toBeChecked();
    expect(screen.getByText(/Modell ist nicht lokal installiert/)).toBeInTheDocument();
    fireEvent.click(toggle);
    fireEvent.change(screen.getByLabelText('Kontextbudget (Tokens)'), { target: { value: '24576' } });
    fireEvent.click(screen.getByRole('button', { name: 'TOP-Einstellungen speichern' }));
    await screen.findByText('TOP-Einstellungen auf dem Server gespeichert.');
    const writes = fetch.mock.calls.filter(([, init]) => init?.method === 'PUT');
    expect(JSON.parse(String(writes[0]?.[1]?.body))).toEqual({ ...settings, enabled: true, context_tokens: 24576 });
    expect(screen.getByText(/Zusammenfassungen und Neugenerierung: qwen3.5:9b/)).toBeInTheDocument();
    unmount();
    render(<AgendaModelPanel />);
    expect(await screen.findByLabelText('Separates Modell für TOP-Zuordnung verwenden')).toBeChecked();
    expect(fetch.mock.calls.every(([url]) => url.endsWith('/api/settings/agenda-model'))).toBe(true);
  });
  it('does not claim a failed save succeeded', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValueOnce(response(status))
      .mockResolvedValueOnce(response({ detail: 'Kontextbudget ungültig' }, false)));
    render(<AgendaModelPanel />);
    fireEvent.click(await screen.findByRole('button', { name: 'TOP-Einstellungen speichern' }));
    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('Kontextbudget ungültig'));
    expect(screen.queryByText('TOP-Einstellungen auf dem Server gespeichert.')).not.toBeInTheDocument();
  });
});
