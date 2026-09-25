import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import App from './App';
import { DEFAULT_SYSTEM_PROMPT } from './components/LLMSettingsPanel';
import {
  checkBackendHealth,
  extractAgendaDataFromPDF,
  getPipelineResult,
  getPipelineStatus,
  loadSession,
  listSessions,
  pollPipeline,
  startSummaryJob,
  pollSummaryJob,
  acceptExistingSummary,
  saveSession,
  exportProtocol,
  startPipeline,
} from './api';
import type { PipelineJob, PipelineResultResponse, SessionResponse, SummaryJob } from './types';
import { agendaSource } from './agendaProposals';

vi.mock('./api', () => ({
  checkBackendHealth: vi.fn(),
  loadSession: vi.fn(),
  listSessions: vi.fn(),
  saveSession: vi.fn(),
  startPipeline: vi.fn(),
  pollPipeline: vi.fn(),
  getPipelineStatus: vi.fn(),
  getPipelineResult: vi.fn(),
  cancelPipeline: vi.fn(),
  startSummaryJob: vi.fn(),
  pollSummaryJob: vi.fn(),
  cancelSummaryJob: vi.fn(),
  acceptExistingSummary: vi.fn(),
  extractAgendaDataFromPDF: vi.fn(),
  exportProtocol: vi.fn(),
  listSpeakerProfiles: vi.fn(() => Promise.resolve([])),
  deleteSpeakerProfileEmbeddings: vi.fn(),
  archiveSpeakerProfile: vi.fn(),
}));

const startedPipeline: PipelineJob = {
  pipeline_id: 'pipeline-1',
  session_id: 'session-1',
  transcription_job_id: 'job-1',
  status: 'pending',
  stage: 'upload',
  progress: 5,
  warnings: [],
};

const completedPipeline: PipelineJob = {
  ...startedPipeline,
  status: 'completed',
  stage: 'ready_for_review',
  progress: 100,
};

function pipelineResult(
  overrides: Partial<PipelineResultResponse['session']> = {},
  pipeline: PipelineJob = completedPipeline,
  warnings: string[] = [],
  agendaDetection: PipelineResultResponse['agenda_detection'] = null
): PipelineResultResponse {
  const result: PipelineResultResponse = {
    pipeline,
    session: {
      session_id: 'session-1',
      job_id: 'job-1',
      current_step: 3,
      tops: ['Haushalt'],
      top_ids: ['top-1'],
      transcript: [
        { line_id: 'legacy:0', speaker: 'SPEAKER_00', text: 'Der Haushalt wird beraten.', start: 0, end: 2 },
      ],
      assignments: [0],
      speaker_names: { SPEAKER_00: 'Alice' },
      summaries: { 0: 'Der Haushalt wurde serverseitig zusammengefasst.' },
      summary_reviews: {},
      summary_states: {
        0: {
          top_id: 'top-1',
          status: 'ready',
          source_snapshot: [
            { line_id: 'legacy:0', speaker: 'SPEAKER_00', text: 'Der Haushalt wird beraten.' },
          ],
        },
      },
      skipped_assignment: false,
      audio_url: '/api/audio/job-1',
      ...overrides,
    },
    job: {
      job_id: 'job-1',
      status: 'completed',
      progress: 100,
      message: 'Fertig',
      transcript: [
        { speaker: 'SPEAKER_00', text: 'Der Haushalt wird beraten.', start: 0, end: 2 },
      ],
      audio_url: '/api/audio/job-1',
    },
    speaker_observations: [],
    summary_reviews: overrides.summary_reviews ?? {},
    warnings,
    agenda_detection: agendaDetection,
  };
  if (!('agenda_proposals' in overrides)) {
    const session = result.session;
    const detection = {
      tops: session.tops, transcript: session.transcript ?? [],
      assignments: session.assignments, strategy: 'test', uncertain_count: 0,
      segments: [{
        top_index: 0, top_title: session.tops[0] ?? '', start_index: 0, end_index: 0,
        confidence: 0.9, uncertain: false, transition_type: 'explicit', reason: 'Aufruf',
      }],
      ...agendaDetection,
    };
    result.session.agenda_proposals = {
      version: 1, source: agendaSource(session.tops, session.top_ids ?? [], session.transcript ?? []),
      result: detection,
    };
    result.agenda_detection = detection;
  }
  return result;
}

async function uploadAndStart(user = userEvent.setup()) {
  const { container } = render(<App />);
  const input = container.querySelector<HTMLInputElement>('input[type="file"][accept="audio/*"]');
  await user.upload(input!, new File(['audio'], 'meeting.mp3', { type: 'audio/mpeg' }));
  await user.click(screen.getByRole('checkbox', { name: /TOPs automatisch aus PDF erkennen und direkt verarbeiten/i }));
  await user.click(screen.getByRole('button', { name: /automatisch verarbeiten/i }));
  return { container };
}

describe('App pipeline flow', () => {
  it('persists Fast, locks it during processing and opens the unreviewed result directly', async () => {
    const user = userEvent.setup();
    let finish!: (job: PipelineJob) => void;
    vi.mocked(pollPipeline).mockImplementationOnce(() => new Promise(resolve => { finish = resolve; }));
    vi.mocked(getPipelineResult).mockResolvedValue(pipelineResult({ processing_mode: 'fast', enforce_top_order: true }, completedPipeline,
      ['Fast – ohne automatische Inhaltsprüfung.']));
    const { container } = render(<App />);
    expect(screen.getByRole('checkbox', { name: /Feste TOP-Reihenfolge erzwingen/ })).not.toBeChecked();
    await user.click(screen.getByRole('checkbox', { name: /Feste TOP-Reihenfolge erzwingen/ }));
    expect(screen.getByRole('switch', { name: 'Slow-Modus' })).toBeChecked();
    await user.click(screen.getByRole('switch', { name: 'Slow-Modus' }));
    expect(screen.getByRole('switch', { name: 'Slow-Modus' })).not.toBeChecked();
    await user.upload(container.querySelector<HTMLInputElement>('input[accept="audio/*"]')!,
      new File(['audio'], 'meeting.mp3', { type: 'audio/mpeg' }));
    await user.click(screen.getByRole('checkbox', { name: /TOPs automatisch aus PDF erkennen und direkt verarbeiten/i }));
    await user.click(screen.getByRole('button', { name: /automatisch verarbeiten/i }));
    await waitFor(() => expect(startPipeline).toHaveBeenCalledWith(expect.any(File), expect.objectContaining({ processingMode: 'fast', enforceTopOrder: true })));
    expect(saveSession).toHaveBeenCalledWith(expect.objectContaining({ processing_mode: 'fast', enforce_top_order: true }));
    expect(screen.getByRole('switch', { name: 'Slow-Modus' })).toBeDisabled();
    expect(screen.getByRole('checkbox', { name: /Feste TOP-Reihenfolge erzwingen/ })).toBeDisabled();
    await act(async () => { finish(completedPipeline); });
    await waitFor(() => expect(screen.getByRole('button', { name: /text \(\.txt\)/i })).toBeInTheDocument());
    expect(screen.getByRole('switch', { name: 'Slow-Modus' })).not.toBeChecked();
  });

  it('extracts the PDF again after switching its Fast result to Slow', async () => {
    const user = userEvent.setup();
    vi.mocked(extractAgendaDataFromPDF).mockResolvedValue({ tops: ['Haushalt'], metadata: {},
      processing_mode: 'fast', processing_complete: true, review_status: 'skipped', review_required: true,
      document: { job_id: 'fast-pdf', sha256: 'hash', page_count: 1 } });
    const { container } = render(<App />);
    await user.click(screen.getByRole('switch', { name: 'Slow-Modus' }));
    await user.click(screen.getByRole('checkbox', { name: /TOPs automatisch aus PDF erkennen und direkt verarbeiten/i }));
    await user.upload(container.querySelector<HTMLInputElement>('input[accept=".pdf,application/pdf"]')!,
      new File(['pdf'], 'invitation.pdf', { type: 'application/pdf' }));
    await waitFor(() => expect(screen.getByText(/1 TOP erfolgreich extrahiert/)).toBeInTheDocument());
    await user.click(screen.getByRole('switch', { name: 'Slow-Modus' }));
    await user.upload(container.querySelector<HTMLInputElement>('input[accept="audio/*"]')!,
      new File(['audio'], 'meeting.mp3', { type: 'audio/mpeg' }));
    await user.click(screen.getByRole('button', { name: /automatisch verarbeiten/i }));
    await waitFor(() => expect(startPipeline).toHaveBeenCalledWith(expect.any(File), expect.objectContaining({
      processingMode: 'slow', tops: [], autoDetectTopsFromPdf: true, pdfSourceJobId: undefined,
    })));
  });

  it('restores the saved mode and resets new sessions to Slow', async () => {
    const user = userEvent.setup();
    window.history.replaceState(null, '', '/sessions/session-1');
    vi.mocked(loadSession).mockResolvedValue(pipelineResult({ processing_mode: 'fast', enforce_top_order: true }).session);
    render(<App />);
    await waitFor(() => expect(screen.getByRole('switch', { name: 'Slow-Modus' })).not.toBeChecked());
    expect(screen.getByRole('checkbox', { name: /Feste TOP-Reihenfolge erzwingen/ })).toBeChecked();
    await user.click(screen.getByRole('button', { name: /neue sitzung/i }));
    await waitFor(() => expect(screen.getByRole('switch', { name: 'Slow-Modus' })).toBeChecked());
    expect(screen.getByRole('checkbox', { name: /Feste TOP-Reihenfolge erzwingen/ })).not.toBeChecked();
  });


  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    window.history.replaceState(null, '', '/');
    vi.mocked(checkBackendHealth).mockResolvedValue(true);
    vi.mocked(listSessions).mockResolvedValue({
      items: [],
      total: 0,
      limit: 20,
      offset: 0,
    });
    vi.mocked(saveSession).mockResolvedValue({
      session_id: 'session-1',
      tops: ['Haushalt'],
      transcript: [],
      assignments: [],
      speaker_names: {},
      summaries: {},
      skipped_assignment: false,
    });
    vi.mocked(startPipeline).mockResolvedValue(startedPipeline);
    vi.mocked(pollPipeline).mockImplementation(async (_pipelineId, onStatus) => {
      onStatus?.({
        ...startedPipeline,
        status: 'processing',
        stage: 'transcribe',
        progress: 25,
      });
      onStatus?.({
        ...startedPipeline,
        status: 'processing',
        stage: 'summarize',
        progress: 82,
      });
      return completedPipeline;
    });
    vi.mocked(getPipelineResult).mockResolvedValue(pipelineResult());
    vi.mocked(loadSession).mockResolvedValue(pipelineResult().session);
    vi.mocked(startSummaryJob).mockResolvedValue({
      summary_job_id: 'summary-job-1',
      session_id: 'session-1',
      status: 'pending',
      progress: 0,
      current_top: 0,
      total_tops: 1,
      top_ids: ['top-1'],
    });
    vi.mocked(pollSummaryJob).mockResolvedValue({
      summary_job_id: 'summary-job-1',
      session_id: 'session-1',
      status: 'completed',
      progress: 100,
      current_top: 1,
      total_tops: 1,
      top_ids: ['top-1'],
    });
    vi.mocked(acceptExistingSummary).mockResolvedValue(pipelineResult().session);
  });

  it('starts the end-to-end pipeline from the upload UI', async () => {
    await uploadAndStart();

    await waitFor(() => {
      expect(startPipeline).toHaveBeenCalledWith(
        expect.any(File),
        expect.objectContaining({
          sessionId: 'session-1',
          tops: [],
          pdfFile: null,
          model: expect.any(String),
          summarySystemPrompt: DEFAULT_SYSTEM_PROMPT,
        })
      );
    });
    const options = vi.mocked(startPipeline).mock.calls[0]![1]!;
    expect(options).not.toHaveProperty('systemPrompt');
    expect(options).not.toHaveProperty('agendaSystemPrompt');
    expect(options).not.toHaveProperty('pdfSystemPrompt');
  });

  it('keeps a saved legacy prompt as the summary preference', async () => {
    localStorage.setItem('llm-settings', JSON.stringify({ model: 'saved-model', systemPrompt: 'Gespeicherte Fachvorgabe' }));
    await uploadAndStart();
    await waitFor(() => expect(startPipeline).toHaveBeenCalled());
    const options = vi.mocked(startPipeline).mock.calls[0]![1]!;
    expect(options.summarySystemPrompt).toBe('Gespeicherte Fachvorgabe');
    expect(options.model).toBe('saved-model');
    expect(options).not.toHaveProperty('systemPrompt');
    expect(options).not.toHaveProperty('agendaSystemPrompt');
    expect(options).not.toHaveProperty('pdfSystemPrompt');
    expect(JSON.parse(localStorage.getItem('llm-settings')!).systemPrompt).toBe('Gespeicherte Fachvorgabe');
  });

  it('starts auto-PDF processing without requiring immediate TOP extraction', async () => {
    const user = userEvent.setup();
    const { container } = render(<App />);
    const audioInput = container.querySelector<HTMLInputElement>('input[type="file"][accept="audio/*"]');
    const pdfInput = container.querySelector<HTMLInputElement>('input[accept=".pdf,application/pdf"]');
    const pdf = new File(['pdf'], 'agenda.pdf', { type: 'application/pdf' });

    await user.upload(audioInput!, new File(['audio'], 'meeting.mp3', { type: 'audio/mpeg' }));
    await user.upload(pdfInput!, pdf);
    await user.click(screen.getByRole('button', { name: /automatisch verarbeiten/i }));

    await waitFor(() => {
      expect(extractAgendaDataFromPDF).not.toHaveBeenCalled();
      expect(startPipeline).toHaveBeenCalledWith(
        expect.any(File),
        expect.objectContaining({
          sessionId: 'session-1',
          tops: [],
          pdfFile: pdf,
          autoDetectTopsFromPdf: true,
          skipAgendaDetection: false,
        })
      );
    });
  });

  it('sends PDF-extracted TOPs to the pipeline instead of re-detecting them', async () => {
    const user = userEvent.setup();
    const { container } = render(<App />);
    const audioInput = container.querySelector<HTMLInputElement>('input[type="file"][accept="audio/*"]');
    const pdfInput = container.querySelector<HTMLInputElement>('input[accept=".pdf,application/pdf"]');
    const pdf = new File(['pdf'], 'agenda.pdf', { type: 'application/pdf' });
    vi.mocked(extractAgendaDataFromPDF).mockResolvedValue({
      tops: ['Eröffnung', 'Haushalt'],
      metadata: {},
    });

    await user.click(screen.getByRole('checkbox', {
      name: /TOPs automatisch aus PDF erkennen und direkt verarbeiten/i,
    }));
    await user.upload(audioInput!, new File(['audio'], 'meeting.mp3', { type: 'audio/mpeg' }));
    await user.upload(pdfInput!, pdf);
    await screen.findByDisplayValue('Eröffnung');
    await screen.findByDisplayValue('Haushalt');
    await user.click(screen.getByRole('button', { name: /automatisch verarbeiten/i }));

    await waitFor(() => {
      expect(startPipeline).toHaveBeenCalledWith(
        expect.any(File),
        expect.objectContaining({
          tops: ['Eröffnung', 'Haushalt'],
          pdfFile: pdf,
          autoDetectTopsFromPdf: false,
        })
      );
    });
  });

  it('polls status and opens the protocol step for safe results', async () => {
    await uploadAndStart();

    await waitFor(() => {
      expect(pollPipeline).toHaveBeenCalledWith('pipeline-1', expect.any(Function));
    });
    expect(await screen.findByText('Der Haushalt wurde serverseitig zusammengefasst.')).toBeInTheDocument();
  });

  it('loads the completed result into app state without regenerating summaries', async () => {
    await uploadAndStart();

    expect(await screen.findByText('Der Haushalt wurde serverseitig zusammengefasst.')).toBeInTheDocument();
    expect(startSummaryJob).not.toHaveBeenCalled();
  });

  it('merges a delayed single-job result before stopping polling and exports and saves that result', async () => {
    const user = userEvent.setup();
    const original = pipelineResult({ revision: 1, summary_reviews: {
      0: { source_links: [], review_warnings: [], duration_seconds: 99, llm_usage: { calls: 1 } },
    } }).session;
    const review = { structured: null, source_links: [], review_warnings: [], fallback_used: true,
      chunks_processed: 3, duration_seconds: 12, llm_usage: { calls: 3 } };
    const refreshed = { ...original, revision: 4, summaries: { 0: 'Aktuelle Einzelzusammenfassung' },
      summary_reviews: { 0: review } };
    let finishLoad!: (value: SessionResponse) => void;
    vi.mocked(loadSession).mockResolvedValueOnce(original).mockImplementationOnce(() =>
      new Promise(resolve => { finishLoad = resolve; }));
    vi.mocked(saveSession).mockImplementation(async payload => ({ ...payload,
      session_id: 'session-1', revision: (payload.revision ?? 0) + 1 }));
    vi.mocked(exportProtocol).mockResolvedValue(new Blob(['Protokoll']));
    Object.defineProperty(URL, 'createObjectURL', { configurable: true, value: vi.fn(() => 'blob:test') });
    Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: vi.fn() });
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => undefined);
    window.history.replaceState(null, '', '/sessions/session-1');
    render(<App />);
    await screen.findByText(original.summaries[0]!);
    await user.click(screen.getByRole('button', { name: 'Neu generieren' }));
    await user.click(screen.getByRole('button', { name: /verbindlich starten/i }));
    await waitFor(() => expect(finishLoad).toBeDefined());
    await act(async () => { finishLoad(refreshed); });
    expect(await screen.findByText('Aktuelle Einzelzusammenfassung')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Neu generieren' })).toBeEnabled();
    await user.click(screen.getByRole('button', { name: 'Text (.txt)' }));
    expect(exportProtocol).toHaveBeenCalledWith(expect.objectContaining({
      summaries: refreshed.summaries, summaryReviews: refreshed.summary_reviews,
    }));
    await waitFor(() => expect(saveSession).toHaveBeenLastCalledWith(expect.objectContaining({
      revision: 4, summaries: refreshed.summaries, summary_reviews: refreshed.summary_reviews,
    })));
    expect(startSummaryJob).toHaveBeenCalledWith('session-1', expect.objectContaining({ topIds: ['top-1'] }));
  });

  it('refreshes partial results while retaining local text, unselected TOPs and export metadata', async () => {
    const user = userEvent.setup();
    const original = pipelineResult({ revision: 1, tops: ['Haushalt', 'Schulbau', 'Sport'],
      top_ids: ['top-1', 'top-2', 'top-3'],
      summaries: { 0: 'Alt Haushalt', 1: 'Alt Schulbau', 2: 'Alt Sport' },
    }).session;
    const job: SummaryJob = { summary_job_id: 'batch', session_id: 'session-1', status: 'processing',
      top_ids: ['top-1', 'top-2'], total_tops: 2, current_top: 1, progress: 0,
      completed_tops: 0, current_top_id: 'top-1' };
    let report!: (job: SummaryJob) => void | Promise<void>;
    let finish!: (job: SummaryJob) => void;
    vi.mocked(startSummaryJob).mockResolvedValue({ ...job, status: 'pending' });
    vi.mocked(pollSummaryJob).mockImplementation((_id, callback) => {
      report = callback!;
      return new Promise(resolve => { finish = resolve; });
    });
    vi.mocked(saveSession).mockImplementation(async payload => ({ ...payload,
      session_id: 'session-1', revision: (payload.revision ?? 0) + 1 }));
    vi.mocked(loadSession).mockResolvedValue(original);
    window.history.replaceState(null, '', '/sessions/session-1');
    render(<App />);
    await screen.findByText('Alt Haushalt');
    await user.click(screen.getByRole('checkbox', { name: 'Haushalt auswählen' }));
    await user.click(screen.getByRole('checkbox', { name: 'Schulbau auswählen' }));
    await user.click(screen.getByRole('button', { name: 'Auswahl neu generieren (2)' }));
    await user.click(screen.getByRole('button', { name: /verbindlich starten/i }));
    await waitFor(() => expect(report).toBeDefined());
    expect(startSummaryJob).toHaveBeenCalledWith('session-1', expect.objectContaining({ topIds: ['top-1', 'top-2'] }));
    fireEvent.change(screen.getByLabelText('Gremium'), { target: { value: 'Manuelles Gremium' } });
    await user.click(screen.getByRole('button', { name: /Schulbau/ }));
    await user.click(screen.getByTitle('Bearbeiten'));
    fireEvent.change(screen.getAllByRole('textbox').find(element => (element as HTMLTextAreaElement).value === 'Alt Schulbau')!,
      { target: { value: 'Manueller Schulbau' } });
    await user.click(screen.getByRole('button', { name: 'Speichern' }));
    const remote = { ...original, revision: 5, summaries: { ...original.summaries, 0: 'Neu Haushalt' } };
    vi.mocked(loadSession).mockResolvedValue(remote);
    await act(async () => { await report({ ...job, progress: 50, completed_tops: 1, current_top_id: 'top-2' }); });
    expect(screen.getByText('Manueller Schulbau')).toBeInTheDocument();
    expect(screen.getByText('1 von 2 TOPs abgeschlossen')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: /Haushalt/ }));
    expect(screen.getByText('Neu Haushalt')).toBeInTheDocument();
    await act(async () => { finish({ ...job, status: 'failed', progress: 100, completed_tops: 1,
      current_top_id: null, error: '1 von 2 TOPs fehlgeschlagen',
      outcomes: { 'top-1': { status: 'completed' }, 'top-2': { status: 'failed', error: 'Modellfehler' } } }); });
    expect(await screen.findByText('Regenerierung mit Fehlern beendet')).toBeInTheDocument();
    expect(screen.getByLabelText('Gremium')).toHaveValue('Manuelles Gremium');
    await waitFor(() => expect(saveSession).toHaveBeenLastCalledWith(expect.objectContaining({
      summaries: { 0: 'Neu Haushalt', 1: 'Manueller Schulbau', 2: 'Alt Sport' },
      export_metadata: expect.objectContaining({ committee: 'Manuelles Gremium' }),
    })));
  });

  it('sets agenda detection from the pipeline result', async () => {
    vi.mocked(getPipelineResult).mockResolvedValue(
      pipelineResult(
        { speaker_names: { SPEAKER_00: 'SPEAKER_00' } },
        completedPipeline,
        [],
        {
          tops: ['Haushalt'],
          assignments: [0],
          strategy: 'known_agenda_heuristic',
          uncertain_count: 0,
          segments: [
            {
              top_index: 0,
              top_title: 'Haushalt',
              start_index: 0,
              end_index: 0,
              confidence: 0.95,
              uncertain: false,
              transition_type: 'explicit',
              reason: 'Explizite TOP-Nennung',
              evidence_index: 0,
              evidence_text: 'Der Haushalt wird beraten.',
            },
          ],
        }
      )
    );

    await uploadAndStart();

    expect(await screen.findByText(/1 Segmente, 0 unsicher/i)).toBeInTheDocument();
    expect(screen.getByText(/Strategie: known_agenda_heuristic/i)).toBeInTheDocument();
  });

  it('hides direct protocol when agenda detection is uncertain', async () => {
    vi.mocked(getPipelineResult).mockResolvedValue(
      pipelineResult(
        {},
        completedPipeline,
        [],
        {
          tops: ['Haushalt'],
          assignments: [0],
          strategy: 'llm_repaired',
          uncertain_count: 1,
          segments: [
            {
              top_index: 0,
              top_title: 'Haushalt',
              start_index: 0,
              end_index: 0,
              confidence: 0.45,
              uncertain: true,
              transition_type: 'repaired',
              reason: 'Segment wurde repariert',
              evidence_index: 0,
              evidence_text: 'Der Haushalt wird beraten.',
            },
          ],
        }
      )
    );

    await uploadAndStart();

    expect(await screen.findByText(/1 Segmente, 1 unsicher/i)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /direkt zum protokoll/i })).not.toBeInTheDocument();
  });

  it('keeps the speaker review step when no TOPs are available', async () => {
    vi.mocked(getPipelineResult).mockResolvedValue(
      pipelineResult({
        current_step: 2,
        tops: [],
        assignments: [null],
        skipped_assignment: true,
        summaries: { 0: 'Gesamtes Gespräch wurde zusammengefasst.' },
      })
    );

    await uploadAndStart();

    expect(await screen.findByText(/Sprecher umbenennen und Profile prüfen/i)).toBeInTheDocument();
    expect(screen.getByText(/Keine TOPs angelegt/i)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /direkt zum protokoll/i })).not.toBeInTheDocument();
  });

  it('keeps a pipeline summary when only the TOP title changes', async () => {
    const user = userEvent.setup();
    // Autosave may fire during typing on a busy test runner. Echo the saved
    // payload instead of returning the default empty-session stub.
    vi.mocked(saveSession).mockImplementation(async payload => ({
      ...payload, session_id: 'session-1', revision: (payload.revision ?? 0) + 1,
    }));
    vi.mocked(getPipelineResult).mockResolvedValue(
      pipelineResult({ speaker_names: { SPEAKER_00: 'SPEAKER_00' } })
    );

    await uploadAndStart(user);

    expect(screen.getByRole('button', { name: /^zum protokoll/i })).toBeInTheDocument();

    const topInput = screen.getByLabelText(/ausgewählter top/i);
    await user.clear(topInput);
    await user.type(topInput, 'Neuer Haushalt');
    await user.click(screen.getByRole('button', { name: /top umbenennen/i }));

    await user.click(screen.getByRole('button', { name: /^zum protokoll/i }));

    expect(await screen.findByText('Der Haushalt wurde serverseitig zusammengefasst.')).toBeInTheDocument();
    expect(startSummaryJob).not.toHaveBeenCalled();
  });

  it('does not auto-regenerate when the pipeline already returned a summary error review', async () => {
    const user = userEvent.setup();
    vi.mocked(getPipelineResult).mockResolvedValue(
      pipelineResult({
        summaries: { 0: '' },
        summary_reviews: {
          0: {
            structured: null,
            source_links: [],
            review_warnings: [
              {
                kind: 'summary_failed',
                message: 'Leere Antwort des LLM',
                severity: 'error',
                line_indices: [],
                excerpt: '',
              },
            ],
          },
        },
      })
    );

    await uploadAndStart(user);

    expect(await screen.findByText(/Sprecher umbenennen und Profile prüfen/i)).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: /^zum protokoll/i }));

    expect(await screen.findByText('Leere Antwort des LLM')).toBeInTheDocument();
    expect(screen.queryByText(/Zuordnung, Sprecher oder TOPs wurden geändert/i)).not.toBeInTheDocument();
  });

  it('shows a pipeline error without starting a hidden legacy workflow', async () => {
    vi.mocked(startPipeline).mockRejectedValue(new Error('Pipeline nicht erreichbar'));

    await uploadAndStart();

    expect(await screen.findByText('Pipeline nicht erreichbar')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /erneut versuchen/i })).toBeInTheDocument();
  });

  it('keeps the review step when automatic generation has uncertain assignments', async () => {
    vi.mocked(getPipelineResult).mockResolvedValue(
      pipelineResult({
        tops: ['Haushalt', 'Schulbau'],
        transcript: [
          { speaker: 'SPEAKER_00', text: 'Haushalt.', start: 0, end: 1 },
          { speaker: 'SPEAKER_01', text: 'Vielleicht Schulbau.', start: 2, end: 3 },
        ],
        assignments: [0, null],
        summaries: {
          0: 'Haushalt wurde zusammengefasst.',
          1: 'Schulbau wurde zusammengefasst.',
        },
      })
    );

    await uploadAndStart();

    expect(await screen.findByText('Automatisch erkannte Segmente')).toBeInTheDocument();
    expect(screen.getByText('1 von 2 Zeilen zugeordnet')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /direkt zum protokoll/i })).not.toBeInTheDocument();
  });

  it('resumes an active pipeline after reload and applies the finished result', async () => {
    const user = userEvent.setup();
    localStorage.setItem('active-pipeline-id', 'pipeline-1');
    vi.mocked(getPipelineStatus).mockResolvedValue({
      ...startedPipeline,
      status: 'processing',
      stage: 'agenda_detect',
      progress: 72,
    });

    render(<App />);
    await user.click(
      await screen.findByRole('button', { name: /letzte sitzung fortsetzen/i })
    );

    await waitFor(() => {
      expect(getPipelineStatus).toHaveBeenCalledWith('pipeline-1');
      expect(pollPipeline).toHaveBeenCalledWith('pipeline-1', expect.any(Function));
    });
    expect(await screen.findByText('Der Haushalt wurde serverseitig zusammengefasst.')).toBeInTheDocument();
  });

  it('clears a failed restored pipeline id and loads the saved session', async () => {
    const user = userEvent.setup();
    localStorage.setItem('active-pipeline-id', 'pipeline-1');
    localStorage.setItem('active-session-id', 'session-1');
    vi.mocked(getPipelineStatus).mockResolvedValue({
      ...startedPipeline,
      status: 'failed',
      stage: 'summarize',
      progress: 0,
      error: 'Pipeline wurde durch Backend-Neustart unterbrochen',
    });

    render(<App />);
    await user.click(
      await screen.findByRole('button', { name: /letzte sitzung fortsetzen/i })
    );

    await waitFor(() => {
      expect(loadSession).toHaveBeenCalledWith('session-1');
    });
    expect(localStorage.getItem('active-pipeline-id')).toBeNull();
    expect(await screen.findByText('Der Haushalt wurde serverseitig zusammengefasst.')).toBeInTheDocument();
  });

  it('opens the shared session history from the root-page button', async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole('button', { name: 'Verlauf' }));

    expect(window.location.pathname).toBe('/sessions');
    expect(await screen.findByRole('heading', { name: 'Sitzungsverlauf' })).toBeInTheDocument();
    expect(listSessions).toHaveBeenCalled();
  });

  it('loads a shared session directly by URL without browser session data', async () => {
    window.history.replaceState(null, '', '/sessions/session-1');
    render(<App />);

    await waitFor(() => {
      expect(loadSession).toHaveBeenCalledWith('session-1');
    });
    expect(
      await screen.findByText('Der Haushalt wurde serverseitig zusammengefasst.')
    ).toBeInTheDocument();
  });
  it('labels a completed pipeline with LLM failure as technically incomplete', async () => {
    const result = pipelineResult({ assignments: [null], summaries: {} });
    result.agenda_detection!.llm = {
      enabled: true, source: 'request', status: 'failed', timeout_seconds: 1800,
      attempted_calls: 1, failed_calls: 1, failure_reasons: ['TimeoutError'],
      gaps: [{ start_index: 0, end_index: 0, kind: 'technical', reason: 'TimeoutError' }],
    };
    result.session.agenda_proposals!.result = result.agenda_detection!;
    vi.mocked(getPipelineResult).mockResolvedValue(result);
    await uploadAndStart();
    expect(await screen.findByText(/Automatische Verarbeitung technisch unvollständig/)).toBeInTheDocument();
    expect(screen.queryByText(/Das Protokoll ist vorbereitet/)).not.toBeInTheDocument();
  });

});
