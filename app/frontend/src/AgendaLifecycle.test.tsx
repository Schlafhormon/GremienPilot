import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import App from './App';
import { checkBackendHealth, detectAgenda, loadSession, saveSession } from './api';
import { agendaSource } from './agendaProposals';
import type { AgendaDetectionRequest, AgendaDetectionResponse, SessionResponse } from './types';

vi.mock('./api', async (importOriginal) => ({
  ...await importOriginal<typeof import('./api')>(),
  checkBackendHealth: vi.fn(), loadSession: vi.fn(), saveSession: vi.fn(),
  detectAgenda: vi.fn(), listSpeakerProfiles: vi.fn(async () => []),
}));

function detection(input: AgendaDetectionRequest): AgendaDetectionResponse {
  const tops = input.tops!;
  return {
    tops, transcript: input.transcript, assignments: [0, tops.length - 1],
    llm: { enabled: true, source: 'request', status: 'success', timeout_seconds: 120,
      attempted_calls: 2, failed_calls: 0, failure_reasons: [], processing_complete: true, review_complete: true,
      provenance: {identities: tops.map((title, top_index) => ({title, top_index, top_id: `agenda:${top_index}`}))},
      line_results: input.transcript.map((line, index) => ({line_id: line.line_id!, index,
        top_ids: [`agenda:${index ? tops.length - 1 : 0}`], status: 'assigned', review_status: index ? 'agreed' : 'unresolved', reason: 'Modellprüfung', evidence: []})) },
    strategy: 'test', uncertain_count: 1, warnings: ['Grenze bitte prüfen.'],
    segments: [0, tops.length - 1].map((top_index, index) => ({
      top_index, top_title: tops[top_index]!, start_index: index, end_index: index,
      confidence: index ? 0.9 : 0.5, uncertain: index === 0,
      transition_type: 'explicit', reason: 'Aufruf', evidence_index: index,
      evidence_text: input.transcript[index]!.text,
    })),
  };
}

let stored: SessionResponse;
const draft = () => JSON.parse(localStorage.getItem('active-session-draft')!) as SessionResponse;
const assignments = () => draft().assignments;
async function reopen(view: ReturnType<typeof render>) {
  await waitFor(() => expect(stored.agenda_proposals).toEqual(draft().agenda_proposals));
  await waitFor(() => expect(stored.transcript).toEqual(draft().transcript));
  await waitFor(() => expect(stored.tops).toEqual(draft().tops));
  await waitFor(() => expect(stored.top_ids).toEqual(draft().top_ids));
  await waitFor(() => expect(stored.assignments).toEqual(assignments()));
  view.unmount();
  const next = render(<App />);
  await screen.findByRole('button', { name: 'Alle übernehmen' });
  return next;
}

beforeEach(() => {
  vi.clearAllMocks();
  localStorage.clear();
  window.history.replaceState(null, '', '/sessions/session-1');
  const tops = ['Haushalt', 'Schulbau'];
  const top_ids = ['top-a', 'top-b'];
  const transcript = [
    { line_id: 'line-a', speaker: 'S1', text: 'TOP 1 Haushalt.', start: 0, end: 2 },
    { line_id: 'line-b', speaker: 'S2', text: 'TOP 2 Schulbau.', start: 2, end: 4 },
  ];
  stored = {
    session_id: 'session-1', revision: 1, current_step: 2, tops, top_ids, transcript,
    assignments: [0, 1], speaker_names: { S1: 'Anna', S2: 'Ben' }, summaries: {},
    skipped_assignment: false,
    agenda_proposals: {
      version: 1, source: agendaSource(tops, top_ids, transcript),
      result: detection({ tops, transcript }),
    },
  };
  vi.mocked(checkBackendHealth).mockResolvedValue(true);
  vi.mocked(loadSession).mockImplementation(async () => structuredClone(stored));
  vi.mocked(saveSession).mockImplementation(async (payload) => {
    stored = { ...structuredClone(payload), session_id: 'session-1', revision: stored.revision! + 1 };
    return structuredClone(stored);
  });
  vi.mocked(detectAgenda).mockImplementation(async (input) => detection(input));
});

describe('agenda proposals across real editor state transitions', () => {
  it.each([
    ['TOP hinzufügen', [0, 2]],
    ['TOP löschen', [null, 0]],
    ['TOP zusammenlegen', [0, 0]],
    ['TOP umbenennen', [0, 1]],
  ])('blocks historical indices after %s and retains uncertainty on reopening', async (action, expected) => {
    const user = userEvent.setup();
    let view = render(<App />);
    await screen.findByRole('button', { name: 'Alle übernehmen' });
    if (action === 'TOP zusammenlegen') {
      await user.click(screen.getByRole('button', { name: /Schulbau.*1 Zeilen/ }));
    }
    if (action === 'TOP umbenennen') {
      await user.clear(screen.getByLabelText('Ausgewählter TOP'));
      await user.type(screen.getByLabelText('Ausgewählter TOP'), 'Finanzen');
    }
    await user.click(screen.getByRole('button', { name: action as string }));
    expect(assignments()).toEqual(expected);
    expect(screen.getByRole('button', { name: 'Alle übernehmen' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Alle sicheren übernehmen' })).toBeDisabled();
    for (const button of screen.getAllByRole('button', { name: 'Übernehmen' })) {
      expect(button).toBeDisabled();
      await user.click(button);
    }
    await user.click(screen.getByRole('button', { name: 'Alle übernehmen' }));
    expect(assignments()).toEqual(expected);
    view = await reopen(view);
    expect(screen.getByRole('button', { name: 'Alle übernehmen' })).toBeDisabled();
    expect(screen.getByText(/2 Segmente, 1 unsicher/)).toBeInTheDocument();
    expect(screen.getByText(/Vorschläge veraltet/)).toBeInTheDocument();
    expect(detectAgenda).not.toHaveBeenCalled();
    view.unmount();
  });

  it('recalculates explicitly, preserves concurrent manual edits, permits deliberate reapply, saves and restores evidence', async () => {
    const user = userEvent.setup();
    let view = render(<App />);
    await screen.findByRole('button', { name: 'Alle übernehmen' });
    await user.click(screen.getByRole('button', { name: 'TOP hinzufügen' }));
    let resolve!: (value: AgendaDetectionResponse) => void;
    vi.mocked(detectAgenda).mockImplementationOnce(() => new Promise((done) => { resolve = done; }));
    await user.click(screen.getByRole('button', { name: 'TOP-Erkennung erneut berechnen' }));
    await user.click(screen.getByText('TOP 1 Haushalt.'));
    expect(assignments()).toEqual([1, 2]);
    const request = vi.mocked(detectAgenda).mock.calls[0]![0];
    expect(request.preserveTranscriptStructure).toBe(true);
    await act(async () => resolve(detection(request)));
    expect(assignments()).toEqual([1, 2]);
    await user.click(screen.getByText('TOP 2 Schulbau.'));
    expect(assignments()).toEqual([1, 1]);
    await user.click(screen.getByRole('button', { name: 'Alle sicheren übernehmen' }));
    expect(assignments()).toEqual([1, 2]);
    expect(screen.queryByText(/Vorschläge veraltet/)).not.toBeInTheDocument();
    await user.click(screen.getAllByRole('button', { name: 'Übernehmen' })[0]!);
    expect(assignments()).toEqual([0, 2]);
    await user.click(screen.getByText('TOP 1 Haushalt.'));
    expect(assignments()).toEqual([null, 2]);
    await user.click(screen.getByRole('button', { name: 'Alle übernehmen' }));
    expect(assignments()).toEqual([0, 2]);
    view = await reopen(view);
    expect(screen.getByRole('button', { name: 'Alle übernehmen' })).toBeEnabled();
    expect(screen.getByText('Grenze bitte prüfen.')).toBeInTheDocument();
    expect(screen.getByText(/2 Segmente, 1 unsicher/)).toBeInTheDocument();

    await user.click(screen.getAllByRole('button', { name: 'Bearbeiten' })[0]!);
    await user.clear(screen.getByLabelText('Transkriptzeile 1 korrigieren'));
    await user.type(screen.getByLabelText('Transkriptzeile 1 korrigieren'), 'Text korrigiert.');
    await user.click(screen.getByRole('button', { name: 'Speichern' }));
    expect(screen.getByRole('button', { name: 'Alle übernehmen' })).toBeDisabled();
    expect(assignments()).toEqual([0, 2]);
    await reopen(view);
    expect(screen.getByRole('button', { name: 'Alle übernehmen' })).toBeDisabled();
    expect(screen.getByText(/2 Segmente, 1 unsicher/)).toBeInTheDocument();
    expect(stored.agenda_proposals!.source!.transcript[0]!.text).toBe('TOP 1 Haushalt.');
    expect(stored.transcript![0]!.text).toBe('Text korrigiert.');
  });

  it('rejects a late result after transcript splitting and preserves the old warning', async () => {
    const user = userEvent.setup();
    render(<App />);
    await screen.findByRole('button', { name: 'Alle übernehmen' });
    let resolve!: (value: AgendaDetectionResponse) => void;
    vi.mocked(detectAgenda).mockImplementationOnce(() => new Promise((done) => { resolve = done; }));
    await user.click(screen.getByRole('button', { name: 'TOP-Erkennung erneut berechnen' }));
    const request = vi.mocked(detectAgenda).mock.calls[0]![0];
    await user.click(screen.getAllByRole('button', { name: 'Bearbeiten' })[0]!);
    await user.clear(screen.getByLabelText('Transkriptzeile 1 korrigieren'));
    await user.type(screen.getByLabelText('Transkriptzeile 1 korrigieren'), 'Teil eins{enter}Teil zwei');
    await user.click(screen.getByRole('button', { name: 'Speichern' }));
    expect(assignments()).toEqual([0, 0, 1]);
    const ids = draft().transcript!.map((line) => line.line_id);
    expect(ids[0]).toBe('line-a');
    expect(new Set(ids).size).toBe(3);
    await act(async () => resolve(detection(request)));
    expect(screen.getByText(/während der Erkennung geändert/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Alle übernehmen' })).toBeDisabled();
    expect(screen.getByText(/2 Segmente, 1 unsicher/)).toBeInTheDocument();
    expect(assignments()).toEqual([0, 0, 1]);
  });

  it('keeps manual corrections and old evidence when recalculation fails', async () => {
    const user = userEvent.setup();
    render(<App />);
    await screen.findByRole('button', { name: 'Alle übernehmen' });
    await user.click(screen.getByText('TOP 1 Haushalt.'));
    const original = draft().agenda_proposals;
    vi.mocked(detectAgenda).mockRejectedValueOnce(new Error('Zeitlimit'));
    await user.click(screen.getByRole('button', { name: 'TOP-Erkennung erneut berechnen' }));
    await screen.findByText(/Zeitlimit/);
    expect(assignments()).toEqual([null, 1]);
    expect(draft().agenda_proposals).toEqual(original);
    expect(screen.getByRole('button', { name: 'Alle übernehmen' })).toBeEnabled();
  });

  it('does not attach an in-flight result to a newly opened session', async () => {
    const user = userEvent.setup();
    render(<App />);
    await screen.findByRole('button', { name: 'Alle übernehmen' });
    let resolve!: (value: AgendaDetectionResponse) => void;
    vi.mocked(detectAgenda).mockImplementationOnce(() => new Promise((done) => { resolve = done; }));
    await user.click(screen.getByRole('button', { name: 'TOP-Erkennung erneut berechnen' }));
    const request = vi.mocked(detectAgenda).mock.calls[0]![0];
    vi.mocked(loadSession).mockResolvedValueOnce({ ...stored, session_id: 'session-2', agenda_proposals: null });
    act(() => {
      window.history.pushState(null, '', '/sessions/session-2');
      window.dispatchEvent(new PopStateEvent('popstate'));
    });
    await screen.findByText(/Unsicherheiten sind unbekannt/);
    await act(async () => resolve(detection(request)));
    expect(screen.getByText(/Unsicherheiten sind unbekannt/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Alle übernehmen' })).toBeDisabled();
    expect(draft().agenda_proposals).toBeNull();
  });

  it('invalidates after joining transcript lines and restores the old uncertainty', async () => {
    const user = userEvent.setup();
    const view = render(<App />);
    await screen.findByRole('button', { name: 'Alle übernehmen' });
    await user.click(screen.getByText('TOP 1 Haushalt.'));
    await user.click(screen.getByRole('button', { name: 'Zeile mit nächster verbinden' }));
    expect(draft().transcript).toHaveLength(1);
    expect(draft().transcript![0]!.line_id).toBe('line-a');
    expect(assignments()).toEqual([null]);
    expect(screen.getByRole('button', { name: 'Alle übernehmen' })).toBeDisabled();
    await reopen(view);
    expect(screen.getByText(/2 Segmente, 1 unsicher/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Alle übernehmen' })).toBeDisabled();
  });

  it('shows unknown uncertainty for old sessions without detection evidence', async () => {
    stored.agenda_proposals = null;
    render(<App />);
    await screen.findByText(/Unsicherheiten sind unbekannt/);
    expect(screen.getByRole('button', { name: 'Alle übernehmen' })).toBeDisabled();
    expect(detectAgenda).not.toHaveBeenCalled();
  });

  it('does not restore old summary indices when a save finishes after TOP insertion', async () => {
    stored.summaries = { 0: 'Haushaltstext', 1: 'Schulbautext' };
    let resolve!: (value: SessionResponse) => void;
    vi.mocked(saveSession).mockImplementationOnce(() => new Promise((done) => { resolve = done; }));
    const user = userEvent.setup();
    render(<App />);
    await screen.findByRole('button', { name: 'Alle übernehmen' });
    await user.click(screen.getByText('TOP 1 Haushalt.'));
    await waitFor(() => expect(saveSession).toHaveBeenCalled());
    const submitted = structuredClone(vi.mocked(saveSession).mock.calls[0]![0]);
    await user.click(screen.getByRole('button', { name: 'TOP hinzufügen' }));
    expect(draft().summaries).toEqual({ 0: 'Haushaltstext', 2: 'Schulbautext' });
    await act(async () => resolve({ ...submitted, session_id: 'session-1', revision: 2 }));
    expect(draft().top_ids).toEqual(['top-a', expect.any(String), 'top-b']);
    expect(draft().summaries).toEqual({ 0: 'Haushaltstext', 2: 'Schulbautext' });
    expect(screen.getByRole('button', { name: 'Alle übernehmen' })).toBeDisabled();
  });

  it.each(['uncertain', 'stale', 'missing', 'warning'])('retains %s review status when restoring from the start screen', async (condition) => {
    stored.summaries = { 0: 'Haushaltstext', 1: 'Schulbautext' };
    if (condition === 'missing') stored.agenda_proposals = null;
    if (condition === 'stale') stored.tops = ['Finanzen', 'Schulbau'];
    if (condition === 'warning') {
      stored.agenda_proposals!.result.uncertain_count = 0;
      stored.agenda_proposals!.result.segments.forEach((segment) => { segment.uncertain = false; });
    }
    localStorage.setItem('active-session-id', stored.session_id);
    window.history.replaceState(null, '', '/');
    const user = userEvent.setup();
    render(<App />);
    await user.click(await screen.findByRole('button', { name: /letzte sitzung fortsetzen/i }));
    expect(await screen.findByText(/Bitte prüfen Sie unsichere Zuordnungen vor dem Protokoll/)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Direkt zum Protokoll' })).not.toBeInTheDocument();
  });
});

it('appends independently detected points while preserving existing IDs and manual assignments', async () => {
  vi.mocked(detectAgenda).mockImplementation(async input => {
    const result = detection(input);
    result.tops = [...input.tops!, 'Zusätzliche Beratung'];
    result.assignments = [0, 2];
    result.llm = { enabled: true, source: 'request', timeout_seconds: 120, status: 'success',
      attempted_calls: 6, failed_calls: 0, failure_reasons: [],
      provenance: { identities: [{ top_id: 'new-top', top_index: 2, title: 'Zusätzliche Beratung' }] } };
    return result;
  });
  render(<App />);
  const button = await screen.findByRole('button', { name: 'TOP-Erkennung erneut berechnen' });
  await userEvent.click(button);
  await waitFor(() => expect(draft().tops).toEqual(['Haushalt', 'Schulbau', 'Zusätzliche Beratung']));
  expect(draft().top_ids).toEqual(['top-a', 'top-b', 'new-top']);
  expect(assignments()).toEqual([0, 1]);
  expect(draft().agenda_proposals?.result.assignments).toEqual([0, 2]);
});
