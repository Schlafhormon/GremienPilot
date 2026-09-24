import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { DEFAULT_SYSTEM_PROMPT } from './components/LLMSettingsPanel';
import {
  archiveSpeakerProfile,
  backfillSpeakerEmbeddings,
  getPipelineResult,
  pollPipeline,
  pollSummaryJob,
  pollModelJob,
  confirmSpeakerObservation,
  createManualSpeakerObservation,
  createSpeakerProfile,
  deleteSpeakerProfileEmbeddings,
  detectAgenda,
  exportProtocol,
  extractAgendaDataFromPDF,
  extractTOPsFromPDF,
  reextractSessionPDF,
  generateSummary,
  listSpeakerMatchDiagnostics,
  listSpeakerObservations,
  listSpeakerProfiles,
  listSessions,
  loadSession,
  saveSession,
  startPipeline,
  startTranscription,
  unassignSpeakerObservation,
  updateSpeakerProfile,
} from './api';

describe('api session client', () => {
  it('sends Fast to every standalone generation endpoint and accepts its unreviewed PDF', async () => {
    const pdf = { tops: ['Haushalt'], metadata: {}, processing_complete: true,
      processing_mode: 'fast', review_status: 'skipped', review_required: true };
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => pdf });
    vi.stubGlobal('fetch', fetchMock);
    const file = new File(['pdf'], 'invitation.pdf', { type: 'application/pdf' });
    await expect(extractAgendaDataFromPDF(file, { processingMode: 'fast' })).resolves.toMatchObject(pdf);
    expect((fetchMock.mock.calls[0]![1]!.body as FormData).get('processing_mode')).toBe('fast');
    await expect(extractAgendaDataFromPDF(file, { processingMode: 'slow' })).rejects.toThrow(/nicht vollständig geprüft/);
    const lines = [{ speaker: 'Rat', text: 'Beratung', start: 0, end: 2 }];
    await detectAgenda({ transcript: lines, processingMode: 'fast' });
    expect(JSON.parse(fetchMock.mock.calls[2]![1]!.body).processing_mode).toBe('fast');
    await generateSummary('Haushalt', lines, { processingMode: 'fast' });
    expect(JSON.parse(fetchMock.mock.calls[3]![1]!.body).processing_mode).toBe('fast');
    await startPipeline(new File(['audio'], 'meeting.mp3'), { processingMode: 'fast' });
    expect((fetchMock.mock.calls[4]![1]!.body as FormData).get('processing_mode')).toBe('fast');
  });


  it('awaits summary result synchronization and returns partial job failures', async () => {
    const job = { summary_job_id: 'batch', session_id: 'session', top_ids: ['a', 'b'],
      total_tops: 2, current_top: 2, completed_tops: 1, processed_tops: 1, progress: 50 };
    const fetchMock = vi.fn()
      .mockResolvedValueOnce({ ok: true, json: async () => ({ ...job, status: 'processing' }) })
      .mockResolvedValueOnce({ ok: true, json: async () => ({ ...job, status: 'failed', processed_tops: 2,
        outcomes: { a: { status: 'completed' }, b: { status: 'failed', error: 'Modellfehler' } } }) });
    vi.stubGlobal('fetch', fetchMock);
    let release!: () => void;
    const onStatus = vi.fn().mockImplementationOnce(() => new Promise<void>(resolve => { release = resolve; }));
    const result = pollSummaryJob('batch', onStatus, 0);
    await vi.waitFor(() => expect(onStatus).toHaveBeenCalledTimes(1));
    expect(fetchMock).toHaveBeenCalledTimes(1);
    release();
    expect((await result).status).toBe('failed');
    expect(onStatus).toHaveBeenCalledTimes(2);
  });

  it('does not continue polling after the owner aborts', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
    const controller = new AbortController();
    controller.abort();
    await expect(pollSummaryJob('batch', undefined, 0, controller.signal)).rejects.toThrow();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it('passes the session id when starting transcription', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: () =>
        Promise.resolve({
          job_id: 'job-1',
          status: 'pending',
          progress: 0,
          message: 'Transkription gestartet',
        }),
    });
    vi.stubGlobal('fetch', fetchMock);

    await startTranscription(
      new File(['audio'], 'meeting.mp3', { type: 'audio/mpeg' }),
      'session-1'
    );

    const body = fetchMock.mock.calls[0]![1]!.body as FormData;
    expect(fetchMock).toHaveBeenCalledWith('/api/transcribe', {
      method: 'POST',
      body,
    });
    expect(body.get('session_id')).toBe('session-1');
    expect(body.get('remember_speakers')).toBe('false');
  });

  it('starts a pipeline with audio, optional PDF, known TOPs and model settings', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: () =>
        Promise.resolve({
          pipeline_id: 'pipeline-1',
          session_id: 'session-1',
          transcription_job_id: 'job-1',
          status: 'pending',
          stage: 'upload',
          progress: 5,
          warnings: [],
        }),
    });
    vi.stubGlobal('fetch', fetchMock);

    await startPipeline(new File(['audio'], 'meeting.mp3', { type: 'audio/mpeg' }), {
      sessionId: 'session-1',
      pdfFile: new File(['pdf'], 'agenda.pdf', { type: 'application/pdf' }),
      tops: ['Begruessung', 'Haushalt'],
      model: 'qwen3:8b',
      systemPrompt: 'Prompt',
      rememberSpeakers: true,
      autoDetectTopsFromPdf: true,
    });

    const body = fetchMock.mock.calls[0]![1]!.body as FormData;
    expect(fetchMock).toHaveBeenCalledWith('/api/pipeline/start', {
      method: 'POST',
      body,
    });
    expect(body.get('session_id')).toBe('session-1');
    expect(body.get('tops')).toBe(JSON.stringify(['Begruessung', 'Haushalt']));
    expect(body.get('model')).toBe('qwen3:8b');
    expect(body.get('system_prompt')).toBe('Prompt');
    expect(body.get('remember_speakers')).toBe('true');
    expect(body.get('skip_agenda_detection')).toBe('false');
    expect(body.get('auto_detect_tops_from_pdf')).toBe('true');
    expect(body.get('audio')).toBeInstanceOf(File);
    expect(body.get('pdf')).toBeInstanceOf(File);
    expect(body.get('transcript')).toBeNull();
    expect(body.get('summaries')).toBeNull();
  });

  it('sends the actual UI default only as a summary prompt', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ pipeline_id: 'p', warnings: [] }) });
    vi.stubGlobal('fetch', fetchMock);
    await startPipeline(new File(['audio'], 'meeting.mp3'), { summarySystemPrompt: DEFAULT_SYSTEM_PROMPT });
    const body = fetchMock.mock.calls[0]![1]!.body as FormData;
    expect(body.get('summary_system_prompt')).toBe(DEFAULT_SYSTEM_PROMPT);
    expect(body.has('system_prompt')).toBe(false);
    expect(body.has('agenda_system_prompt')).toBe(false);
    expect(body.has('pdf_system_prompt')).toBe(false);
  });

  it('serializes separate task prompts including an explicit empty summary preference', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ pipeline_id: 'p', warnings: [] }) });
    vi.stubGlobal('fetch', fetchMock);
    await startPipeline(new File(['audio'], 'meeting.mp3'), {
      systemPrompt: 'Legacy', summarySystemPrompt: '', agendaSystemPrompt: 'Grenzen', pdfSystemPrompt: 'Einladung',
    });
    const body = fetchMock.mock.calls[0]![1]!.body as FormData;
    expect(body.get('system_prompt')).toBe('Legacy');
    expect(body.get('summary_system_prompt')).toBe('');
    expect(body.get('agenda_system_prompt')).toBe('Grenzen');
    expect(body.get('pdf_system_prompt')).toBe('Einladung');
  });

  it('extracts PDF agenda data including session metadata', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: () =>
        Promise.resolve({
          tops: ['Begruessung', 'Haushalt'],
          processing_complete: true, review_required: false,
          metadata: {
            committee: 'Hauptausschuss',
            date: '2026-06-30',
            location: 'Rathaus',
            title: 'Sitzung Hauptausschuss',
          },
        }),
    });
    vi.stubGlobal('fetch', fetchMock);

    const result = await extractAgendaDataFromPDF(
      new File(['pdf'], 'agenda.pdf', { type: 'application/pdf' }),
      { model: 'qwen3:8b', systemPrompt: 'Prompt' }
    );

    const body = fetchMock.mock.calls[0]![1]!.body as FormData;
    expect(fetchMock).toHaveBeenCalledWith('/api/extract-tops/jobs', {
      method: 'POST',
      body,
    });
    expect(body.get('model')).toBe('qwen3:8b');
    expect(body.get('system_prompt')).toBe('Prompt');
    expect(result.tops).toEqual(['Begruessung', 'Haushalt']);
    expect(result.metadata).toEqual(
      expect.objectContaining({
        committee: 'Hauptausschuss',
        date: '2026-06-30',
      })
    );
  });

  it('keeps the TOP-only PDF extraction helper backward compatible', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        json: () =>
          Promise.resolve({
            tops: ['Begruessung'],
            processing_complete: true, review_required: false,
            metadata: { committee: 'Hauptausschuss' },
          }),
      })
    );

    await expect(
      extractTOPsFromPDF(new File(['pdf'], 'agenda.pdf', { type: 'application/pdf' }))
    ).resolves.toEqual(['Begruessung']);
  });

  it('polls pipeline status until completion', async () => {
    vi.useFakeTimers();
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve({
            pipeline_id: 'pipeline-1',
            status: 'processing',
            stage: 'transcribe',
            progress: 25,
            warnings: [],
          }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve({
            pipeline_id: 'pipeline-1',
            status: 'completed',
            stage: 'ready_for_review',
            progress: 100,
            warnings: [],
          }),
      });
    vi.stubGlobal('fetch', fetchMock);
    const updates: string[] = [];

    const promise = pollPipeline('pipeline-1', (status) => updates.push(status.stage));
    await vi.advanceTimersByTimeAsync(1000);
    const result = await promise;

    expect(result.status).toBe('completed');
    expect(updates).toEqual(['transcribe', 'ready_for_review']);
    expect(fetchMock.mock.calls.map((call) => call[0])).toEqual([
      '/api/pipeline/pipeline-1',
      '/api/pipeline/pipeline-1',
    ]);
    vi.useRealTimers();
  });

  it('loads a completed pipeline result', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: () =>
        Promise.resolve({
          pipeline: {
            pipeline_id: 'pipeline-1',
            status: 'completed',
            stage: 'ready_for_review',
            progress: 100,
          },
          session: {
            session_id: 'session-1',
            tops: ['Haushalt'],
            transcript: [{ speaker: 'SPEAKER_00', text: 'Hallo', start: 0, end: 1 }],
            assignments: [0],
            speaker_names: {},
            summaries: { 0: 'Zusammenfassung' },
            skipped_assignment: false,
          },
          agenda_detection: {
            tops: ['Haushalt'],
            assignments: [0],
            segments: [
              {
                top_index: 0,
                top_title: 'Haushalt',
                start_index: 0,
                end_index: 0,
                confidence: 0.91,
                uncertain: false,
                transition_type: 'explicit',
                reason: 'Explizite TOP-Nennung',
              },
            ],
            strategy: 'known_agenda_heuristic',
            uncertain_count: 0,
          },
          warnings: ['Hinweis'],
        }),
    });
    vi.stubGlobal('fetch', fetchMock);

    const result = await getPipelineResult('pipeline-1');

    expect(fetchMock.mock.calls[0]![0]).toBe('/api/pipeline/pipeline-1/result');
    expect(result.pipeline.warnings).toEqual([]);
    expect(result.warnings).toEqual(['Hinweis']);
    expect(result.session.summaries[0]).toBe('Zusammenfassung');
    expect(result.agenda_detection?.segments).toHaveLength(1);
    expect(result.agenda_detection?.uncertain_count).toBe(0);
  });

  it('saves and loads persisted session state', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve({
            session_id: 'session-1',
            job_id: 'job-1',
            current_step: 3,
            tops: ['TOP 1'],
            transcript: [
              {
                speaker: 'SPEAKER_00',
                text: 'Korrigierter Text',
                start: 0,
                end: 1,
              },
            ],
            assignments: [0],
            speaker_names: { SPEAKER_00: 'Alice' },
            summaries: { 0: 'Zusammenfassung' },
            skipped_assignment: false,
          }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve({
            session_id: 'session-1',
            tops: ['TOP 1'],
            assignments: [0],
            speaker_names: {},
            summaries: {},
            skipped_assignment: false,
          }),
      });
    vi.stubGlobal('fetch', fetchMock);

    await saveSession({
      session_id: 'session-1',
      job_id: 'job-1',
      current_step: 3,
      tops: ['TOP 1'],
      transcript: [
        { speaker: 'SPEAKER_00', text: 'Korrigierter Text', start: 0, end: 1 },
      ],
      assignments: [0],
      speaker_names: { SPEAKER_00: 'Alice' },
      summaries: { 0: 'Zusammenfassung' },
      skipped_assignment: false,
    });
    await loadSession('session-1');

    expect(fetchMock.mock.calls[0]![0]).toBe('/api/sessions/session-1');
    expect(fetchMock.mock.calls[0]![1]!.method).toBe('PUT');
    expect(JSON.parse(fetchMock.mock.calls[0]![1]!.body as string)).toMatchObject({
      transcript: [{ text: 'Korrigierter Text' }],
      speaker_names: { SPEAKER_00: 'Alice' },
      summaries: { 0: 'Zusammenfassung' },
    });
    expect(fetchMock.mock.calls[1]![0]).toBe('/api/sessions/session-1');
    expect(fetchMock.mock.calls[1]![1]).toBeUndefined();
  });

  it('loads the shared session history with filters', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ items: [], total: 0, limit: 20, offset: 0 }),
    });
    vi.stubGlobal('fetch', fetchMock);

    await listSessions({ limit: 20, offset: 20, query: 'Haushalt', status: 'ready' });

    expect(fetchMock.mock.calls[0]![0]).toBe(
      '/api/sessions?limit=20&offset=20&query=Haushalt&status=ready'
    );
  });

  it('reports a revision conflict while saving a shared session', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false,
      status: 409,
      json: () => Promise.resolve({
        detail: {
          message: 'Diese Sitzung wurde zwischenzeitlich geändert.',
          actual_revision: 4,
        },
      }),
    }));

    await expect(saveSession({
      session_id: 'session-1',
      revision: 3,
      tops: [],
      assignments: [],
      speaker_names: {},
      summaries: {},
      skipped_assignment: false,
    })).rejects.toMatchObject({
      name: 'SessionConflictError',
      actualRevision: 4,
    });
  });

  it('posts protocol export data and returns the generated blob', async () => {
    const blob = new Blob(['docx-bytes'], {
      type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    });
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      blob: () => Promise.resolve(blob),
    });
    vi.stubGlobal('fetch', fetchMock);

    const result = await exportProtocol({
      format: 'docx',
      metadata: {
        committee: 'Hauptausschuss',
        date: '2026-06-30',
        location: 'Rathaus',
        title: 'Sitzung Hauptausschuss',
        participants: ['Alice'],
        includeSpeakerList: true,
        includeTranscript: true,
        groupTranscriptByTop: true,
        includeGenerationNote: true,
      },
      tops: ['Begruessung'],
      transcript: [{ speaker: 'SPEAKER_00', text: 'Hallo', start: 0, end: 1 }],
      assignments: [0],
      speakerNames: { SPEAKER_00: 'Alice' },
      summaries: { 0: 'Diskussion:\nHallo' },
      summaryReviews: {},
    });

    expect(result).toBe(blob);
    expect(fetchMock.mock.calls[0]![0]).toBe('/api/export');
    const body = JSON.parse(fetchMock.mock.calls[0]![1]!.body as string);
    expect(body).toMatchObject({
      format: 'docx',
      metadata: {
        committee: 'Hauptausschuss',
        participants: ['Alice'],
      },
      appendix: {
        include_speaker_list: true,
        include_transcript: true,
        include_transcript_excerpt: true,
        group_transcript_by_top: true,
        include_generation_note: true,
      },
      speaker_names: { SPEAKER_00: 'Alice' },
    });
  });

  it('posts agenda detection data with optional known TOPs and model settings', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: () =>
        Promise.resolve({
          tops: ['Begruessung', 'Haushalt'],
          transcript: [
            { speaker: 'SPEAKER_00', text: 'Hallo', start: 0, end: 1 },
            { speaker: 'SPEAKER_01', text: 'Haushalt', start: 2, end: 3 },
          ],
          assignments: [0, 1],
          segments: [],
          strategy: 'known_agenda_heuristic',
          uncertain_count: 0,
        }),
    });
    vi.stubGlobal('fetch', fetchMock);

    const result = await detectAgenda({
      tops: ['Begruessung', 'Haushalt'],
      transcript: [
        { speaker: 'SPEAKER_00', text: 'Hallo', start: 0, end: 1 },
        { speaker: 'SPEAKER_01', text: 'Haushalt', start: 2, end: 3 },
      ],
      model: 'qwen3:8b',
      systemPrompt: 'Prompt',
    });

    expect(result.assignments).toEqual([0, 1]);
    expect(result.transcript).toHaveLength(2);
    expect(fetchMock.mock.calls[0]![0]).toBe('/api/agenda-detection/jobs');
    expect(fetchMock.mock.calls[0]![1]!.method).toBe('POST');
    expect(JSON.parse(fetchMock.mock.calls[0]![1]!.body as string)).toMatchObject({
      tops: ['Begruessung', 'Haushalt'],
      transcript: [{ text: 'Hallo' }, { text: 'Haushalt' }],
      model: 'qwen3:8b',
      system_prompt: 'Prompt',
    });
  });

  it('sends full long meetings to the durable agenda job without the legacy browser limit', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce({ ok: true, json: async () => ({ job_id: 'long-meeting', state: 'queued' }) })
      .mockResolvedValueOnce({ ok: true, json: async () => ({ job_id: 'long-meeting', state: 'completed', result: { tops: [], transcript: [], assignments: [] } }) });
    vi.stubGlobal('fetch', fetchMock);

    await detectAgenda({
        transcript: [
          {
            speaker: 'SPEAKER_00',
            text: 'Langer Transkripttext '.repeat(7000),
            start: 0,
            end: 1,
          },
        ],
      });
    expect(JSON.parse(fetchMock.mock.calls[0]![1].body).transcript[0].text).toBe('Langer Transkripttext '.repeat(7000));
    expect(fetchMock.mock.calls[1]![0]).toBe('/api/model-jobs/long-meeting');
  });

  it('does not send oversized legacy summary requests from the browser', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);

    await expect(
      generateSummary('Haushalt', [
        {
          speaker: 'SPEAKER_00',
          text: 'Langer Transkripttext '.repeat(7000),
          start: 0,
          end: 1,
        },
      ])
    ).rejects.toThrow(/Browser-Workflow zu groß/i);

    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('uses speaker profile and observation endpoints', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve([
            {
              profile_id: 'alice',
              display_name: 'Alice Global',
              scope: null,
              created_at: 1,
            updated_at: 1,
            archived: false,
            embedding_count: 0,
          },
        ]),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve({
            profile_id: 'alice',
            display_name: 'Alice Lokal',
            scope: null,
            created_at: 1,
            updated_at: 2,
            archived: false,
            embedding_count: 1,
          }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve({
            profile_id: 'alice',
            display_name: 'Alice Lokal',
            scope: null,
            created_at: 1,
            updated_at: 3,
            archived: true,
            archived_at: 3,
            embedding_count: 1,
          }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve({
            profile_id: 'alice',
            deleted_count: 2,
          }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve({
            scanned_observation_count: 2,
            processed_job_count: 1,
            saved_embedding_count: 3,
            skipped_count: 0,
            errors: [],
          }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve({
            profile_id: 'bob',
            display_name: 'Bob Global',
            scope: 'committee-1',
            created_at: 1,
            updated_at: 1,
            archived: false,
            embedding_count: 0,
          }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve([
            {
              observation_id: 7,
              job_id: 'job-1',
              session_id: 'session-1',
              local_speaker_id: 'SPEAKER_00',
              local_display_name: 'SPEAKER_00',
              profile_id: 'alice',
              profile_display_name: 'Alice Lokal',
              confidence: 0.82,
              status: 'suggested',
              display_name: 'Alice Lokal',
              created_at: 1,
              updated_at: 1,
            },
          ]),
      })
      .mockResolvedValue({
        ok: true,
        json: () =>
          Promise.resolve({
            observation_id: 7,
            job_id: 'job-1',
            session_id: 'session-1',
            local_speaker_id: 'SPEAKER_00',
            local_display_name: 'SPEAKER_00',
            profile_id: 'alice',
            profile_display_name: 'Alice Lokal',
            confidence: 1,
            status: 'manual',
            display_name: 'Alice Lokal',
            created_at: 1,
            updated_at: 4,
          }),
      });
    vi.stubGlobal('fetch', fetchMock);

    await listSpeakerProfiles({ includeArchived: true });
    await updateSpeakerProfile('alice', { displayName: 'Alice Lokal' });
    await archiveSpeakerProfile('alice');
    await deleteSpeakerProfileEmbeddings('alice');
    await backfillSpeakerEmbeddings('alice');
    await createSpeakerProfile({ displayName: 'Bob Global', scope: 'committee-1' });
    await listSpeakerObservations('session-1');
    await listSpeakerMatchDiagnostics('session-1');
    await confirmSpeakerObservation('session-1', 7, { profileId: 'alice' });
    await unassignSpeakerObservation('session-1', 7);
    await createManualSpeakerObservation('session-1', {
      localSpeakerId: 'SPEAKER_00',
      profileId: 'alice',
      observationId: 7,
    });

    expect(fetchMock.mock.calls[0]![0]).toBe(
      '/api/speaker-profiles?include_archived=true'
    );
    expect(fetchMock.mock.calls[1]![0]).toBe('/api/speaker-profiles/alice');
    expect(fetchMock.mock.calls[1]![1]!.method).toBe('PUT');
    expect(JSON.parse(fetchMock.mock.calls[1]![1]!.body as string)).toEqual({
      display_name: 'Alice Lokal',
    });
    expect(fetchMock.mock.calls[2]![0]).toBe('/api/speaker-profiles/alice');
    expect(fetchMock.mock.calls[2]![1]!.method).toBe('DELETE');
    expect(fetchMock.mock.calls[3]![0]).toBe('/api/speaker-profiles/alice/embeddings');
    expect(fetchMock.mock.calls[3]![1]!.method).toBe('DELETE');
    expect(fetchMock.mock.calls[4]![0]).toBe('/api/speaker-embeddings/backfill?profile_id=alice');
    expect(fetchMock.mock.calls[4]![1]!.method).toBe('POST');
    expect(fetchMock.mock.calls[5]![0]).toBe('/api/speaker-profiles');
    expect(fetchMock.mock.calls[5]![1]!.method).toBe('POST');
    expect(JSON.parse(fetchMock.mock.calls[5]![1]!.body as string)).toMatchObject({
      display_name: 'Bob Global',
      scope: 'committee-1',
    });
    expect(fetchMock.mock.calls[6]![0]).toBe(
      '/api/sessions/session-1/speaker-observations'
    );
    expect(fetchMock.mock.calls[7]![0]).toBe(
      '/api/sessions/session-1/speaker-match-diagnostics'
    );
    expect(fetchMock.mock.calls[8]![0]).toBe(
      '/api/sessions/session-1/speaker-observations/7/confirm'
    );
    expect(JSON.parse(fetchMock.mock.calls[8]![1]!.body as string)).toMatchObject({
      profile_id: 'alice',
    });
    expect(fetchMock.mock.calls[9]![0]).toBe(
      '/api/sessions/session-1/speaker-observations/7/unassign'
    );
    expect(fetchMock.mock.calls[9]![1]!.method).toBe('POST');
    expect(fetchMock.mock.calls[10]![0]).toBe(
      '/api/sessions/session-1/speaker-observations/manual'
    );
    expect(JSON.parse(fetchMock.mock.calls[10]![1]!.body as string)).toMatchObject({
      local_speaker_id: 'SPEAKER_00',
      profile_id: 'alice',
      observation_id: 7,
    });
  });
});


describe('explicit agenda LLM policy', () => {
  afterEach(() => vi.unstubAllGlobals());

  it.each([undefined, null, false, true])('preserves decision %s for detection and pipeline', async (decision) => {
    const llm = { enabled: true, source: 'request', timeout_seconds: 8, status: 'fallback', attempted_calls: 1, failed_calls: 1, failure_reasons: ['timeout'] };
    const warnings = ['Heuristische Ersatzverarbeitung verwendet.'];
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ tops: [], segments: [], assignments: [], llm, warnings }),
    });
    vi.stubGlobal('fetch', fetchMock);
    const result = await detectAgenda({ transcript: [], model: 'chosen', systemPrompt: 'custom', useLlm: decision });
    const payload = JSON.parse(fetchMock.mock.calls[0]![1].body);
    expect(payload.use_llm).toBe(decision);
    expect(result.llm).toEqual(llm);
    expect(result.warnings).toEqual(warnings);
    await startPipeline(new File(['fake'], 'test.mp3'), { agendaUseLlm: decision });
    const form = fetchMock.mock.calls[1]![1].body as FormData;
    expect(form.get('agenda_use_llm')).toBe(decision == null ? null : String(decision));
  });

  it('retains persisted diagnostics and warnings when loading a pipeline result', async () => {
    const agenda = { tops: [], assignments: [], segments: [], strategy: 'known_agenda_heuristic_llm_fallback', uncertain_count: 0, llm: { status: 'fallback', failure_reasons: ['timeout'] }, warnings: ['TOP-Zuordnung prüfen.'] };
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ({ pipeline: {}, session: {}, agenda_detection: agenda }) }));
    const result = await getPipelineResult('test');
    expect(result.agenda_detection?.llm).toEqual(agenda.llm);
    expect(result.agenda_detection?.warnings).toEqual(agenda.warnings);
  });
});


describe('durable model jobs', () => {
  afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals(); });

  it('reconnects after a backend interruption without submitting another job', async () => {
    vi.useFakeTimers();
    const fetchMock = vi.fn()
      .mockRejectedValueOnce(new TypeError('offline'))
      .mockResolvedValueOnce({ ok: true, json: async () => ({ job_id: 'j', state: 'running' }) })
      .mockResolvedValueOnce({ ok: true, json: async () => ({ job_id: 'j', state: 'completed', result: { tops: ['A'] } }) });
    vi.stubGlobal('fetch', fetchMock);
    const result = pollModelJob('j');
    await vi.advanceTimersByTimeAsync(3100);
    expect(await result).toEqual({ tops: ['A'] });
    expect(fetchMock.mock.calls.every(call => call[0] === '/api/model-jobs/j')).toBe(true);
  });

  it('requests server cancellation immediately while a status request is outstanding', async () => {
    let resolveStatus!: (value: unknown) => void;
    const fetchMock = vi.fn()
      .mockImplementationOnce(() => new Promise(resolve => { resolveStatus = resolve; }))
      .mockResolvedValueOnce({ ok: true });
    vi.stubGlobal('fetch', fetchMock);
    const controller = new AbortController();
    const result = pollModelJob('j', controller.signal);
    const rejected = expect(result).rejects.toThrow('abgebrochen');
    controller.abort();
    expect(fetchMock).toHaveBeenLastCalledWith('/api/model-jobs/j/cancel', expect.objectContaining({ method: 'POST' }));
    resolveStatus({ ok: true, json: async () => ({ job_id: 'j', state: 'completed', result: { tops: ['late'] } }) });
    await rejected;
  });

  it('does not return technical partial results as successful', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ({ state: 'failed', result: { tops: ['partial'] } }) }));
    await expect(pollModelJob('j')).rejects.toThrow('Verarbeitung fehlgeschlagen');
  });
});

it('reextracts a retained session PDF in Fast mode and polls its own job', async () => {
  const onStatus = vi.fn();
  const result = { tops: ['Haushalt'], metadata: {}, processing_mode: 'fast', processing_complete: true, review_status: 'skipped' };
  const fetchMock = vi.fn()
    .mockResolvedValueOnce({ ok: true, json: async () => ({ job_id: 'pdf-new', kind: 'pdf', state: 'queued' }) })
    .mockResolvedValueOnce({ ok: true, json: async () => ({ job_id: 'pdf-new', kind: 'pdf', state: 'review_required', result }) });
  vi.stubGlobal('fetch', fetchMock);
  expect(await reextractSessionPDF('s1', { processingMode: 'fast', onStatus })).toEqual(result);
  expect(fetchMock.mock.calls[0]![0]).toBe('/api/sessions/s1/pdf-jobs');
  expect(JSON.parse(fetchMock.mock.calls[0]![1].body)).toMatchObject({ processing_mode: 'fast' });
  expect(fetchMock.mock.calls[1]![0]).toBe('/api/model-jobs/pdf-new');
  expect(onStatus.mock.calls.map(call => call[0].state)).toEqual(['queued', 'review_required']);
});

it('rejects enabled PDF detection without uploading or creating a job', async () => {
  const fetchMock = vi.fn();
  vi.stubGlobal('fetch', fetchMock);
  await expect(startPipeline(new File(['audio'], 'meeting.mp3'), { autoDetectTopsFromPdf: true }))
    .rejects.toThrow(/keine Einladung/);
  expect(fetchMock).not.toHaveBeenCalled();
});

it('does not accept an unverified PDF result', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ({
    tops: ['Rat'], metadata: {}, processing_complete: false, review_required: true,
  }) }));
  await expect(extractAgendaDataFromPDF(new File(['pdf'], 'agenda.pdf')))
    .rejects.toThrow(/nicht vollständig geprüft/);
});
