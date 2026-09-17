import { render, screen, waitFor } from '@testing-library/react';
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
  startPipeline,
} from './api';
import type { PipelineJob, PipelineResultResponse } from './types';
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
  await user.click(screen.getByRole('button', { name: /automatisch verarbeiten/i }));
  return { container };
}

describe('App pipeline flow', () => {
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
});
