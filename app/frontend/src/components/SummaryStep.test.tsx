import { fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { exportProtocol } from '../api';
import type { SummaryStepProps, TranscriptLine } from '../types';
import SummaryStep from './SummaryStep';

vi.mock('../api', () => ({
  exportProtocol: vi.fn(),
}));

const transcript: TranscriptLine[] = [
  { speaker: 'SPEAKER_00', text: 'Willkommen', start: 0, end: 4 },
  { speaker: 'SPEAKER_01', text: 'Haushalt wird beraten', start: 5, end: 9 },
];

const defaultProps: SummaryStepProps = {
  onBack: vi.fn(),
  tops: ['Begruessung', 'Haushalt'],
  transcript,
  assignments: [0, 1],
  summaries: {
    0: 'Die Sitzung wurde eroeffnet.',
    1: 'Der Haushalt wurde beraten.',
  },
  setSummaries: vi.fn(),
  onRegenerateSummary: vi.fn().mockResolvedValue(undefined),
  onAcceptSummary: vi.fn().mockResolvedValue(undefined),
  isGenerating: false,
  summariesAreFresh: true,
  speakerNames: {},
  exportMetadata: {
    committee: 'Hauptausschuss',
    date: '2026-06-30',
    location: 'Rathaus',
    title: 'Sitzung Hauptausschuss',
    participants: ['Alice', 'Bob'],
    includeSpeakerList: true,
    includeTranscript: false,
    groupTranscriptByTop: false,
    includeGenerationNote: true,
  },
  setExportMetadata: vi.fn(),
};

function renderSummaryStep(overrides: Partial<SummaryStepProps> = {}) {
  return render(<SummaryStep {...defaultProps} {...overrides} />);
}

function readBlobAsText(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(reader.error);
    reader.onload = () => resolve(String(reader.result));
    reader.readAsText(blob);
  });
}

describe('SummaryStep', () => {
  it('seeks a transcript source by keyboard', async () => {
    const user = userEvent.setup();
    renderSummaryStep({ audioUrl: '/audio/test.wav' });
    await user.click(screen.getByRole('button', { name: 'Haushalt' }));
    screen.getByRole('button', { name: 'Audio ab 0:05' }).focus();
    await user.keyboard('{Enter}');
    expect(document.querySelector('audio')?.currentTime).toBe(5);
  });

  it('selects multiple TOPs, clears the selection and submits one job', async () => {
    const user = userEvent.setup();
    const regenerate = vi.fn().mockResolvedValue(undefined);
    renderSummaryStep({ onRegenerateSummaries: regenerate });
    expect(screen.getByRole('button', { name: 'Auswahl neu generieren (0)' })).toBeDisabled();
    await user.click(screen.getByRole('button', { name: 'Alle auswählen' }));
    expect(screen.getByRole('checkbox', { name: 'Begruessung auswählen' })).toBeChecked();
    expect(screen.getByRole('checkbox', { name: 'Haushalt auswählen' })).toBeChecked();
    await user.click(screen.getByRole('button', { name: 'Auswahl aufheben' }));
    expect(screen.getByRole('button', { name: 'Auswahl neu generieren (0)' })).toBeDisabled();
    await user.click(screen.getByRole('checkbox', { name: 'Haushalt auswählen' }));
    await user.click(screen.getByRole('checkbox', { name: 'Begruessung auswählen' }));
    await user.click(screen.getByRole('button', { name: 'Auswahl neu generieren (2)' }));
    await user.click(screen.getByRole('button', { name: /verbindlich starten/i }));
    expect(regenerate).toHaveBeenCalledExactlyOnceWith([0, 1]);
  });

  it('shows real aggregate progress, the current TOP and the immutable job selection', () => {
    renderSummaryStep({
      onRegenerateSummaries: vi.fn(), topIds: ['a', 'b'],
      summaryJob: {
        summary_job_id: 'job', session_id: 'session', status: 'processing',
        progress: 42, current_top: 4, total_tops: 7, completed_tops: 3,
        processed_tops: 3, current_top_id: 'b', top_ids: ['b'],
      },
    });
    expect(screen.getByText('3 von 7 TOPs abgeschlossen')).toBeInTheDocument();
    expect(screen.getByText('Aktuell: Haushalt')).toBeInTheDocument();
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '42');
    expect(screen.getByRole('checkbox', { name: 'Haushalt auswählen' })).toBeChecked();
    expect(screen.getByRole('checkbox', { name: 'Begruessung auswählen' })).not.toBeChecked();
    expect(screen.getByRole('button', { name: 'Alle auswählen' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Neu generieren' })).toBeDisabled();
  });

  it('keeps individual errors and partial completion visible', () => {
    renderSummaryStep({ topIds: ['a', 'b'], summaryJob: {
      summary_job_id: 'job', session_id: 'session', status: 'failed',
      progress: 100, current_top: 2, total_tops: 2, completed_tops: 1,
      processed_tops: 2, top_ids: ['a', 'b'], error: '1 von 2 TOPs fehlgeschlagen',
      outcomes: { a: { status: 'completed' }, b: { status: 'failed', error: 'Modellfehler' } },
    } });
    expect(screen.getByText('Regenerierung mit Fehlern beendet')).toBeInTheDocument();
    expect(screen.getByText('1 von 2 TOPs abgeschlossen')).toBeInTheDocument();
    expect(screen.getByText('Haushalt: Modellfehler')).toBeInTheDocument();
  });

  beforeEach(() => {
    vi.clearAllMocks();
    Element.prototype.scrollIntoView = vi.fn();
  });

  it('exports protocol data through the backend and downloads the result', async () => {
    const user = userEvent.setup();
    const exportedBlobs: Blob[] = [];
    const createObjectURL = vi.fn((blob: Blob | MediaSource) => {
      exportedBlobs.push(blob as Blob);
      return 'blob:test-protocol';
    });
    const revokeObjectURL = vi.fn();
    const click = vi
      .spyOn(HTMLAnchorElement.prototype, 'click')
      .mockImplementation(() => undefined);

    Object.defineProperty(URL, 'createObjectURL', {
      configurable: true,
      value: createObjectURL,
    });
    Object.defineProperty(URL, 'revokeObjectURL', {
      configurable: true,
      value: revokeObjectURL,
    });
    vi.mocked(exportProtocol).mockResolvedValue(
      new Blob(['SITZUNGSPROTOKOLL'], { type: 'text/plain;charset=utf-8' })
    );

    renderSummaryStep();

    await user.click(screen.getByRole('button', { name: /text \(\.txt\)/i }));

    expect(click).toHaveBeenCalledTimes(1);
    expect(exportProtocol).toHaveBeenCalledWith(
      expect.objectContaining({
        format: 'txt',
        metadata: defaultProps.exportMetadata,
        tops: defaultProps.tops,
        summaries: defaultProps.summaries,
      })
    );
    expect(createObjectURL).toHaveBeenCalledTimes(1);
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:test-protocol');

    const blob = exportedBlobs[0];
    expect(blob).toBeDefined();
    expect(blob).toBeInstanceOf(Blob);
    const content = await readBlobAsText(blob!);

    expect(content).toContain('SITZUNGSPROTOKOLL');
  });

  it('updates export metadata from the form', async () => {
    const setExportMetadata = vi.fn();
    renderSummaryStep({ setExportMetadata });

    fireEvent.change(screen.getByLabelText(/gremium/i), {
      target: { value: 'Finanzausschuss' },
    });

    expect(setExportMetadata).toHaveBeenLastCalledWith(
      expect.objectContaining({ committee: 'Finanzausschuss' })
    );
  });

  it('offers full transcript export with optional TOP grouping', async () => {
    const user = userEvent.setup();
    const setExportMetadata = vi.fn();
    renderSummaryStep({ setExportMetadata });

    const groupingCheckbox = screen.getByRole('checkbox', { name: /TOP-Unterteilung/i });
    expect(groupingCheckbox).toBeDisabled();

    await user.click(screen.getByRole('checkbox', { name: /Transkript anfügen/i }));

    expect(setExportMetadata).toHaveBeenCalledWith(
      expect.objectContaining({ includeTranscript: true })
    );
  });

  it('shows structured review sections and jumps from evidence to transcript', async () => {
    const user = userEvent.setup();

    renderSummaryStep({
      summaries: {
        0: 'Beschluss:\nDer Ausschuss beschloss die Vorlage.',
      },
      summaryReviews: {
        0: {
          structured: {
            discussion: [],
            decisions: ['Der Ausschuss beschloss die Vorlage.'],
            votes: [],
            action_items: [],
            open_points: [],
            uncertainties: [],
          },
          source_links: [
            {
              section: 'decisions',
              item_index: 0,
              item_text: 'Der Ausschuss beschloss die Vorlage.',
              line_indices: [0],
              start: 0,
              end: 4,
              excerpt: 'SPEAKER_00: Willkommen',
              confidence: 0.8,
              missing_source: false,
            },
          ],
          review_warnings: [],
        },
      },
    });

    expect(screen.getByText('Beschluss')).toBeInTheDocument();
    expect(screen.getByText('Der Ausschuss beschloss die Vorlage.')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: /beleg 0:00/i }));

    const transcriptLine = screen.getByText('Willkommen').closest('div');
    expect(transcriptLine?.className).toContain('ring-2');
  });

  it('shows review warnings and missing source markers', () => {
    renderSummaryStep({
      summaries: {
        0: 'Diskussion:\nDer Antrag wurde diskutiert.',
      },
      summaryReviews: {
        0: {
          structured: {
            discussion: ['Der Antrag wurde diskutiert.'],
            decisions: ['Der Antrag wurde angenommen.'],
            votes: [],
            action_items: [],
            open_points: [],
            uncertainties: ['Die Stimmenzahl ist unklar.'],
          },
          source_links: [
            {
              section: 'decisions',
              item_index: 0,
              item_text: 'Der Antrag wurde angenommen.',
              line_indices: [],
              start: null,
              end: null,
              excerpt: '',
              confidence: 0,
              missing_source: true,
            },
          ],
          review_warnings: [
            {
              kind: 'missing_decision_signal',
              message: 'Im Transkript kommt "abgelehnt" vor, in der Zusammenfassung aber nicht.',
              severity: 'warning',
              keyword: 'abgelehnt',
              line_indices: [1],
              start: 5,
              end: 9,
              excerpt: 'SPEAKER_01: Haushalt wird beraten',
            },
          ],
        },
      },
    });

    expect(screen.getByText('Prüfhinweise')).toBeInTheDocument();
    expect(screen.getByText(/abgelehnt/)).toBeInTheDocument();
    expect(screen.getByText('Quelle fehlt')).toBeInTheDocument();
    expect(screen.getByText('Unsicherheiten')).toBeInTheDocument();
    expect(screen.getByText('Die Stimmenzahl ist unklar.')).toBeInTheDocument();
  });

  it('allows export after review warnings are explicitly accepted', async () => {
    const user = userEvent.setup();
    const click = vi
      .spyOn(HTMLAnchorElement.prototype, 'click')
      .mockImplementation(() => undefined);
    Object.defineProperty(URL, 'createObjectURL', {
      configurable: true,
      value: vi.fn(() => 'blob:test-protocol'),
    });
    Object.defineProperty(URL, 'revokeObjectURL', {
      configurable: true,
      value: vi.fn(),
    });
    vi.mocked(exportProtocol).mockResolvedValue(
      new Blob(['SITZUNGSPROTOKOLL'], { type: 'text/plain;charset=utf-8' })
    );

    renderSummaryStep({
      summaries: {
        0: 'Der TOP wurde zur Kenntnis genommen.',
        1: 'Der Haushalt wurde beraten.',
      },
      summaryReviews: {
        0: {
          structured: {
            discussion: [],
            decisions: [],
            votes: [],
            action_items: [],
            open_points: [],
            uncertainties: ['Keine Beschlusslage erkannt.'],
          },
          source_links: [],
          review_warnings: [
            {
              kind: 'missing_decision_signal',
              message: 'Keine Beschlusslage erkannt.',
              severity: 'warning',
              line_indices: [],
              excerpt: '',
            },
          ],
        },
      },
    });

    await user.click(screen.getByRole('button', { name: /text \(\.txt\)/i }));
    expect(exportProtocol).not.toHaveBeenCalled();
    expect(screen.getByText(/Prüfhinweise müssen vor dem Export akzeptiert/i)).toBeInTheDocument();

    await user.click(screen.getByRole('checkbox', {
      name: /Prüfhinweise akzeptieren und Export erlauben/i,
    }));
    await user.click(screen.getByRole('button', { name: /text \(\.txt\)/i }));

    expect(exportProtocol).toHaveBeenCalledTimes(1);
    expect(click).toHaveBeenCalledTimes(1);
  });

  it('requires an explicit warning confirmation before selective regeneration', async () => {
    const user = userEvent.setup();
    const onRegenerateSummary = vi.fn().mockResolvedValue(undefined);
    renderSummaryStep({
      onRegenerateSummary,
      summaryStates: {
        0: {
          top_id: 'top-0',
          status: 'review_required',
          change_reasons: ['lines_added'],
        },
      },
      summariesAreFresh: false,
    });

    expect(screen.getByText('Zeilen wurden hinzugefügt')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: /neu generieren/i }));
    expect(onRegenerateSummary).not.toHaveBeenCalled();
    expect(screen.getByRole('dialog')).toHaveTextContent(/mehrere Stunden/i);

    await user.click(screen.getByRole('button', { name: /verbindlich starten/i }));
    expect(onRegenerateSummary).toHaveBeenCalledWith(0);
  });
});
