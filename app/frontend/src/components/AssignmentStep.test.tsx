import { fireEvent, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { AgendaLLMUsage, AssignmentStepProps, TranscriptLine } from '../types';
import AssignmentStep from './AssignmentStep';

const transcript: TranscriptLine[] = [
  { line_id: 'line-0', speaker: 'SPEAKER_00', text: 'Hallo zusammen', start: 0, end: 4 },
  { line_id: 'line-1', speaker: 'SPEAKER_01', text: 'Wir beraten den Haushalt', start: 5, end: 9 },
];

function reviewedLines(count: number, unassigned: number[] = []): AgendaLLMUsage {
  return { enabled: true, source: 'request', status: 'success', timeout_seconds: 120,
    attempted_calls: 2, failed_calls: 0, failure_reasons: [], chunks: [],
    processing_complete: true, review_complete: true,
    line_results: Array.from({length: count}, (_, index) => ({ line_id: `line-${index}`, index,
      top_ids: unassigned.includes(index) ? [] : ['model'], status: unassigned.includes(index) ? 'unassigned' : 'assigned', review_status: 'agreed', reason: 'Geprüft.', evidence: [] })) };
}

const defaultProps: AssignmentStepProps = {
  onNext: vi.fn(),
  onBack: vi.fn(),
  tops: ['Begruessung', 'Haushalt'],
  setTops: vi.fn(),
  transcript,
  setTranscript: vi.fn(),
  assignments: [null, null],
  setAssignments: vi.fn(),
  agendaDetection: {
    tops: ['Begruessung', 'Haushalt'],
    assignments: [0, 1],
    strategy: 'model_agenda_v1',
    llm: reviewedLines(2),
    uncertain_count: 1,
    segments: [
      {
        top_index: 0,
        top_title: 'Begruessung',
        start_index: 0,
        end_index: 0,
        confidence: 0.6,
        uncertain: true,
        transition_type: 'inferred',
        reason: 'Erster TOP beginnt am Anfang des Transkripts.',
        evidence_index: 0,
        evidence_text: 'Hallo zusammen',
      },
      {
        top_index: 1,
        top_title: 'Haushalt',
        start_index: 1,
        end_index: 1,
        confidence: 0.82,
        uncertain: false,
        transition_type: 'keyword',
        reason: 'Starker Begriffsabgleich mit dem TOP-Titel.',
        evidence_index: 1,
        evidence_text: 'Wir beraten den Haushalt',
      },
    ],
  },
  speakerNames: {},
  setSpeakerNames: vi.fn(),
};

function renderAssignmentStep(overrides: Partial<AssignmentStepProps> = {}) {
  return render(<AssignmentStep {...defaultProps} {...overrides} />);
}

describe('AssignmentStep', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('assigns a clicked transcript line to the selected TOP', () => {
    const setAssignments = vi.fn();
    renderAssignmentStep({ setAssignments });

    fireEvent.click(screen.getByText('Hallo zusammen'));

    expect(setAssignments).toHaveBeenCalledWith([0, null]);
  });

  it('assigns and unassigns a line with the keyboard, including a range with Shift', async () => {
    const user = userEvent.setup();
    const setAssignments = vi.fn();
    const { rerender } = renderAssignmentStep({ setAssignments });
    const firstLine = screen.getByRole('button', { name: 'Zeile 1 zuordnen' });
    firstLine.focus();
    await user.keyboard('{Enter}');
    expect(setAssignments).toHaveBeenLastCalledWith([0, null]);

    rerender(<AssignmentStep {...defaultProps} setAssignments={setAssignments} assignments={[0, null]} />);
    expect(firstLine).toHaveAttribute('aria-pressed', 'true');
    await user.keyboard(' ');
    expect(setAssignments).toHaveBeenLastCalledWith([null, null]);

    screen.getByRole('button', { name: 'Zeile 2 zuordnen' }).focus();
    await user.keyboard('{Shift>}{Enter}{/Shift}');
    expect(setAssignments).toHaveBeenLastCalledWith([0, 0]);
  });

  it('seeks audio by keyboard without changing assignments', async () => {
    const user = userEvent.setup();
    const setAssignments = vi.fn();
    renderAssignmentStep({ setAssignments, audioUrl: '/audio/test.wav' });
    screen.getByRole('button', { name: 'Audio ab 0:05' }).focus();
    await user.keyboard('{Enter}');
    expect(document.querySelector('audio')?.currentTime).toBe(5);
    expect(setAssignments).not.toHaveBeenCalled();
  });

  it('enables continuing once at least one line is assigned', async () => {
    const user = userEvent.setup();
    const onNext = vi.fn();
    const { rerender } = renderAssignmentStep({ onNext });

    const nextButton = screen.getByRole('button', {
      name: /zum protokollentwurf/i,
    });
    expect(nextButton).toBeDisabled();

    rerender(
      <AssignmentStep
        {...defaultProps}
        onNext={onNext}
        assignments={[0, null]}
      />,
    );

    await user.click(screen.getByRole('button', { name: /zum protokollentwurf/i }));
    expect(onNext).toHaveBeenCalledTimes(1);
  });

  it('updates corrected transcript line text', async () => {
    const user = userEvent.setup();
    const setTranscript = vi.fn();
    renderAssignmentStep({ setTranscript });

    await user.click(screen.getAllByRole('button', { name: /bearbeiten/i })[0]!);
    const textarea = screen.getByLabelText(/transkriptzeile 1 korrigieren/i);
    await user.clear(textarea);
    await user.type(textarea, 'Hallo korrigiert');
    await user.click(screen.getByRole('button', { name: /^speichern$/i }));

    expect(setTranscript).toHaveBeenCalledWith([
      { ...transcript[0]!, text: 'Hallo korrigiert', timing: { source: 'manual_estimate', words: [], segments: [] } },
      transcript[1],
    ]);
  });

  it('requests fresh model answers explicitly without replacing assignments', async () => {
    const user = userEvent.setup();
    const onDetectAgenda = vi.fn();
    const setAssignments = vi.fn();
    renderAssignmentStep({ onDetectAgenda, setAssignments });
    await user.click(screen.getByRole('button', { name: 'Frische TOP-Berechnung' }));
    expect(onDetectAgenda).toHaveBeenCalledWith(true);
    expect(setAssignments).not.toHaveBeenCalled();
  });

  it('splits a corrected transcript line by line breaks and keeps assignments aligned', async () => {
    const user = userEvent.setup();
    const setTranscript = vi.fn();
    const setAssignments = vi.fn();
    renderAssignmentStep({
      assignments: [0, 1],
      setTranscript,
      setAssignments,
    });

    await user.click(screen.getAllByRole('button', { name: /bearbeiten/i })[0]!);
    const textarea = screen.getByLabelText(/transkriptzeile 1 korrigieren/i);
    fireEvent.change(textarea, { target: { value: 'Hallo\nzusammen' } });
    await user.click(screen.getByRole('button', { name: /^speichern$/i }));

    expect(setTranscript).toHaveBeenCalledWith([
      { ...transcript[0]!, text: 'Hallo', start: 0, end: 2, timing: { source: 'manual_estimate', words: [], segments: [] } },
      { ...transcript[0]!, line_id: expect.any(String), text: 'zusammen', start: 2, end: 4, timing: { source: 'manual_estimate', words: [], segments: [] } },
      transcript[1],
    ]);
    expect(setAssignments).toHaveBeenCalledWith([0, 0, 1]);
  });

  it('merges an accidentally separated speaker into an existing speaker', async () => {
    const user = userEvent.setup();
    const setTranscript = vi.fn();
    const setSpeakerNames = vi.fn();
    renderAssignmentStep({
      setTranscript,
      setSpeakerNames,
      speakerNames: {
        SPEAKER_00: 'Alice',
        SPEAKER_01: 'Alicia',
      },
    });

    await user.selectOptions(
      screen.getByLabelText('SPEAKER_01 mit Sprecher zusammenführen'),
      'SPEAKER_00'
    );
    await user.click(within(screen.getByRole('region', { name: 'Sprecherprüfung' })).getAllByRole('button', { name: /mergen/i })[1]!);

    expect(setTranscript).toHaveBeenCalledWith([
      transcript[0],
      { ...transcript[1]!, speaker: 'SPEAKER_00' },
    ]);
    expect(setSpeakerNames).toHaveBeenCalledWith({
      SPEAKER_00: 'Alice',
      SPEAKER_01: 'Alice', // Keep historical speaker labels resolvable.
    });
  });

  it('merges a selected transcript line with the previous line', async () => {
    const user = userEvent.setup();
    const setTranscript = vi.fn();
    const setAssignments = vi.fn();
    renderAssignmentStep({
      assignments: [0, 1],
      setTranscript,
      setAssignments,
    });

    await user.click(screen.getByText('Wir beraten den Haushalt'));
    await user.click(screen.getByRole('button', { name: /zeile mit vorheriger verbinden/i }));

    expect(setTranscript).toHaveBeenCalledWith([
      {
        ...transcript[0]!,
        text: 'Hallo zusammen Wir beraten den Haushalt',
        start: 0,
        end: 9,
      },
    ]);
    expect(setAssignments).toHaveBeenLastCalledWith([0]);
  });

  it('merges consecutive lines only when speaker and TOP assignment match', async () => {
    const user = userEvent.setup();
    const setTranscript = vi.fn();
    const setAssignments = vi.fn();
    const onTranscriptStructureChange = vi.fn();
    const splitTranscript: TranscriptLine[] = [
      { speaker: 'SPEAKER_00', text: 'Erster Satz.', start: 0, end: 1 },
      { speaker: 'SPEAKER_00', text: 'Zweiter Satz.', start: 1, end: 2 },
      { speaker: 'SPEAKER_00', text: 'Neuer TOP.', start: 2, end: 3 },
      { speaker: 'SPEAKER_01', text: 'Anderer Sprecher.', start: 3, end: 4 },
      { speaker: 'SPEAKER_01', text: 'Gleicher TOP.', start: 4, end: 5 },
    ];

    renderAssignmentStep({
      transcript: splitTranscript,
      assignments: [0, 0, 1, 1, 1],
      setTranscript,
      setAssignments,
      onTranscriptStructureChange,
    });

    await user.click(screen.getByRole('button', { name: /gleiche sprecher zusammenführen/i }));

    expect(setTranscript).toHaveBeenCalledWith([
      {
        speaker: 'SPEAKER_00',
        text: 'Erster Satz. Zweiter Satz.',
        start: 0,
        end: 2,
      },
      splitTranscript[2],
      {
        speaker: 'SPEAKER_01',
        text: 'Anderer Sprecher. Gleicher TOP.',
        start: 3,
        end: 5,
      },
    ]);
    expect(setAssignments).toHaveBeenCalledWith([0, 1, 1]);
    expect(onTranscriptStructureChange).toHaveBeenCalledTimes(1);
  });

  it('applies generated assignment suggestions after review', async () => {
    const user = userEvent.setup();
    const setAssignments = vi.fn();
    renderAssignmentStep({ setAssignments });

    await screen.findByText(/starker begriffsabgleich/i);
    await user.click(screen.getByRole('button', { name: /^alle übernehmen$/i }));

    expect(setAssignments).toHaveBeenCalledWith([0, 1]);
  });

  it('applies only safe agenda detections and keeps uncertain lines unchanged', async () => {
    const user = userEvent.setup();
    const setAssignments = vi.fn();
    renderAssignmentStep({ setAssignments, assignments: [null, null] });

    await user.click(screen.getByRole('button', { name: /alle sicheren übernehmen/i }));

    expect(setAssignments).toHaveBeenCalledWith([null, 1]);
    expect(screen.getAllByText(/unsicher/i).length).toBeGreaterThan(0);
  });

  it('can split the selected segment from a transcript line', async () => {
    const user = userEvent.setup();
    const setAssignments = vi.fn();
    renderAssignmentStep({ assignments: [0, 0], setAssignments });

    await user.click(screen.getByRole('button', { name: /Haushalt/i }));
    await user.click(screen.getByText('Wir beraten den Haushalt'));
    await user.click(screen.getByRole('button', { name: /grenze ab hier setzen/i }));

    expect(setAssignments).toHaveBeenCalledWith([0, 1]);
  });

  it('renames, adds, deletes and merges TOPs while preserving manual correction', async () => {
    const user = userEvent.setup();
    const setTops = vi.fn();
    const setAssignments = vi.fn();
    renderAssignmentStep({
      setTops,
      setAssignments,
      tops: ['Begruessung', 'Haushalt', 'Schulbau'],
      assignments: [0, 1],
    });

    await user.clear(screen.getByLabelText(/ausgewählter top/i));
    await user.type(screen.getByLabelText(/ausgewählter top/i), 'Eroeffnung');
    await user.click(screen.getByRole('button', { name: /top umbenennen/i }));
    expect(setTops).toHaveBeenCalledWith(['Eroeffnung', 'Haushalt', 'Schulbau']);

    await user.click(screen.getByRole('button', { name: /top hinzufügen/i }));
    expect(setTops).toHaveBeenLastCalledWith(['Begruessung', 'Neuer Tagesordnungspunkt', 'Haushalt', 'Schulbau']);
    expect(setAssignments).toHaveBeenLastCalledWith([0, 2]);

    await user.click(screen.getByRole('button', { name: /Haushalt/i }));
    await user.click(screen.getByRole('button', { name: /top löschen/i }));
    expect(setTops).toHaveBeenLastCalledWith(['Begruessung', 'Schulbau']);
    expect(setAssignments).toHaveBeenLastCalledWith([0, null]);

    await user.click(screen.getByRole('button', { name: /Haushalt/i }));
    await user.click(screen.getByRole('button', { name: /top zusammenlegen/i }));
    expect(setTops).toHaveBeenLastCalledWith(['Begruessung / Haushalt', 'Schulbau']);
    expect(setAssignments).toHaveBeenLastCalledWith([0, 0]);
  });
});

it('shows an agenda fallback warning alongside reviewable segments', () => {
  render(<AssignmentStep {...defaultProps} agendaDetection={{ ...defaultProps.agendaDetection!, warnings: ['TOP-Erkennung: timeout. Heuristische Ersatzverarbeitung verwendet.'] }} />);
  expect(screen.getByRole('status')).toHaveTextContent('Heuristische Ersatzverarbeitung verwendet.');
});

it('shows missing evidence even when no uncertain segments exist', () => {
  renderAssignmentStep({
    agendaDetection: {
      tops: defaultProps.tops,
      assignments: [null, null],
      segments: [],
      uncertain_count: 0,
      strategy: 'model_agenda_v1',
      llm: undefined,
    },
  });
  expect(screen.getByText('2 Zeilen ohne automatischen Zuordnungsvorschlag.')).toBeInTheDocument();
  expect(screen.getByText(/Ohne Segmentnachweis: Begruessung, Haushalt/)).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Alle sicheren übernehmen' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Alle übernehmen' })).toBeDisabled();
});

it('applies reordered and resumed safe segments while leaving gaps and uncertain rows open', () => {
  const setAssignments = vi.fn();
  const line = transcript[0]!;
  const segment = defaultProps.agendaDetection!.segments[1]!;
  renderAssignmentStep({
    transcript: Array.from({ length: 5 }, () => ({ ...line })),
    assignments: [null, null, null, null, null],
    setAssignments,
    agendaDetection: {
      tops: defaultProps.tops,
      assignments: [null, 1, 0, 1, 0],
      segments: [
        { ...segment, top_index: 1, start_index: 1, end_index: 1 },
        { ...segment, top_index: 0, start_index: 2, end_index: 2 },
        { ...segment, top_index: 1, start_index: 3, end_index: 3 },
        { ...segment, top_index: 0, start_index: 4, end_index: 4, confidence: 0.5, uncertain: true },
      ],
      uncertain_count: 1,
      strategy: 'model_agenda_v1',
      llm: reviewedLines(5, [0]),
    },
  });
  fireEvent.click(screen.getByRole('button', { name: 'Alle sicheren übernehmen' }));
  expect(setAssignments).toHaveBeenCalledWith([null, 1, 0, 1, null]);
  expect(screen.getByText('1 Zeilen ohne automatischen Zuordnungsvorschlag.')).toBeInTheDocument();
  expect(screen.getByText('Unsicher prüfen')).toBeInTheDocument();
});

it('distinguishes technical gaps from semantic uncertainty without replacing assignments', () => {
  renderAssignmentStep({agendaDetection: {
    ...defaultProps.agendaDetection!,
    llm: { enabled: true, source: 'request', status: 'partial_failure', timeout_seconds: 1800,
      attempted_calls: 2, failed_calls: 1, failure_reasons: ['TimeoutError'], gaps: [
        {start_index: 0, end_index: 0, kind: 'technical', reason: 'TimeoutError'},
        {start_index: 1, end_index: 1, kind: 'semantic', reason: 'Kein eindeutiger Themenbezug'},
      ]},
  }});
  expect(screen.getByText(/Technisch nicht ausgewertet/)).toBeInTheDocument();
  expect(screen.getByText(/Fachlich begründet unzugeordnet/)).toBeInTheDocument();
  expect(defaultProps.setAssignments).not.toHaveBeenCalled();
});

it('shows joint model decisions, agenda status and technical review gaps without scalar reassignment', () => {
  const setAssignments = vi.fn();
  renderAssignmentStep({ setAssignments, agendaDetection: {
    tops: defaultProps.tops, transcript, assignments: [null, null], segments: [],
    strategy: 'model_agenda_v1', uncertain_count: 0,
    llm: { enabled: true, source: 'request', timeout_seconds: 120, status: 'partial_failure',
      attempted_calls: 4, failed_calls: 1, failure_reasons: ['TimeoutError'],
      processing_complete: true, review_complete: false, chunks: [],
      provenance: { identities: [{ top_id: 'a', top_index: 0, title: 'Begruessung' }, { top_id: 'b', top_index: 1, title: 'Haushalt' }] },
      agenda_states: [{ top_id: 'b', status: 'deferred', reason: 'Weitere Beratung im Dezember.', review_status: 'agreed', evidence: [] }],
      line_results: [
        { line_id: 'line-0', index: 0, top_ids: ['a', 'b'], status: 'assigned', reason: 'Zusammen beraten.', review_status: 'agreed', evidence: [] },
        { line_id: 'line-1', index: 1, top_ids: [], status: 'unassigned', reason: 'Pause.', review_status: 'technical_pending', evidence: [] },
      ],
    },
  }});
  expect(screen.getByText(/Technisch vollständig verarbeitet/)).toBeInTheDocument();
  expect(screen.getByText(/Unabhängige Modellprüfung offen/)).toBeInTheDocument();
  expect(screen.getByText(/Gemeinsame Beratung/)).toBeInTheDocument();
  expect(screen.getByText(/Vertagt/)).toBeInTheDocument();
  expect(screen.getByText(/Technische Prüflücke/)).toBeInTheDocument();
  expect(setAssignments).not.toHaveBeenCalled();
  expect(screen.getByRole('button', { name: 'Alle übernehmen' })).toBeDisabled();
});


it('does not certify legacy confidence without completed independent checks', () => {
  renderAssignmentStep({agendaDetection: { ...defaultProps.agendaDetection!, llm: undefined,
    segments: defaultProps.agendaDetection!.segments.map(segment => ({...segment, confidence: 1, uncertain: false})) }});
  expect(screen.getByRole('button', {name: 'Alle sicheren übernehmen'})).toBeDisabled();
});
