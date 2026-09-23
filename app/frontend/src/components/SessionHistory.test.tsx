import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { listSessions } from '../api';
import SessionHistory from './SessionHistory';

vi.mock('../api', () => ({
  listSessions: vi.fn(),
}));

describe('SessionHistory', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(listSessions).mockResolvedValue({
      items: [
        {
          session_id: 'session-1',
          title: 'Haushaltsausschuss',
          committee: 'Finanzausschuss',
          meeting_date: '2026-07-13',
          status: 'ready',
          current_step: 3,
          revision: 2,
          created_at: 1783900000,
          updated_at: 1783903600,
          top_count: 5,
          transcript_line_count: 42,
          summary_count: 5,
          audio_available: true,
          pipeline_status: 'completed',
          pipeline_progress: 100,
        },
      ],
      total: 1,
      limit: 20,
      offset: 0,
    });
  });

  it('lists shared sessions and opens the selected entry', async () => {
    const onOpen = vi.fn();
    const user = userEvent.setup();
    render(<SessionHistory onOpen={onOpen} onNewSession={vi.fn()} />);

    expect(await screen.findByText('Haushaltsausschuss')).toBeInTheDocument();
    expect(screen.getByText('Finanzausschuss')).toBeInTheDocument();
    expect(screen.getAllByText('Protokoll vorbereitet')).toHaveLength(2);
    expect(screen.getByText('42 Transkriptzeilen')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Öffnen' }));
    expect(onOpen).toHaveBeenCalledWith('session-1');
  });

  it('passes search terms to the shared history endpoint', async () => {
    const user = userEvent.setup();
    render(<SessionHistory onOpen={vi.fn()} onNewSession={vi.fn()} />);
    await screen.findByText('Haushaltsausschuss');

    await user.type(screen.getByRole('searchbox', { name: 'Suchen' }), 'Finanzen');
    await user.click(screen.getByRole('button', { name: 'Suchen' }));

    expect(listSessions).toHaveBeenLastCalledWith(
      expect.objectContaining({ query: 'Finanzen' })
    );
  });
});


it('shows failed phase and retained artifacts even when summaries exist', async () => {
  vi.mocked(listSessions).mockResolvedValue({ items: [{ session_id: 'failed', title: 'Synthetic',
    status: 'failed', committee: '', meeting_date: '', revision: 1, created_at: 1, updated_at: 1,
    top_count: 2, transcript_line_count: 42, summary_count: 2, audio_available: true,
    pipeline_stage: 'agenda_detect', pipeline_error: 'PDF-Prüfung offen' }], total: 1, limit: 20, offset: 0 });
  render(<SessionHistory onOpen={vi.fn()} onNewSession={vi.fn()} />);
  expect(await screen.findByText(/Verarbeitung fehlgeschlagen · Phase: PDF-/)).toBeInTheDocument();
  expect(screen.getByText('42 Transkriptzeilen')).toBeInTheDocument();
  expect(screen.getByText(/Gespeicherte Ergebnisse bleiben erhalten/)).toBeInTheDocument();
});
