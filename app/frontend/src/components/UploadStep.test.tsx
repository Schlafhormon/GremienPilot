import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { extractAgendaDataFromPDF } from '../api';
import type { UploadStepProps } from '../types';
import UploadStep from './UploadStep';

vi.mock('../api', () => ({
  extractAgendaDataFromPDF: vi.fn(),
}));

const defaultProps: UploadStepProps = {
  onNext: vi.fn(),
  audioFile: null,
  setAudioFile: vi.fn(),
  pdfFile: null,
  setPdfFile: vi.fn(),
  tops: [''],
  setTops: vi.fn(),
  llmSettings: { model: 'qwen3:8b', systemPrompt: '' },
  rememberSpeakers: false,
  setRememberSpeakers: vi.fn(),
  skipAgendaDetection: false,
  setSkipAgendaDetection: vi.fn(),
  autoDetectTopsFromPdf: false,
  setAutoDetectTopsFromPdf: vi.fn(),
  exportMetadata: {
    committee: '',
    date: '2026-07-07',
    location: '',
    title: 'Sitzungsprotokoll',
    participants: [],
    includeSpeakerList: true,
    includeTranscript: false,
    groupTranscriptByTop: false,
    includeGenerationNote: true,
  },
  setExportMetadata: vi.fn(),
};

function renderUploadStep(overrides: Partial<UploadStepProps> = {}) {
  return render(<UploadStep {...defaultProps} {...overrides} />);
}

describe('UploadStep', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('keeps the next button disabled until an audio file is selected', async () => {
    const user = userEvent.setup();
    const onNext = vi.fn();
    const audioFile = new File(['audio'], 'meeting.mp3', { type: 'audio/mpeg' });
    const setAudioFile = vi.fn();
    const { container, rerender } = renderUploadStep({ onNext, setAudioFile });

    const nextButton = screen.getByRole('button', { name: /automatisch verarbeiten/i });
    expect(nextButton).toBeDisabled();

    const input = container.querySelector<HTMLInputElement>('input[type="file"][accept="audio/*"]');
    expect(input).not.toBeNull();
    await user.upload(input!, audioFile);

    expect(setAudioFile).toHaveBeenCalledWith(audioFile);

    rerender(
      <UploadStep
        {...defaultProps}
        onNext={onNext}
        audioFile={audioFile}
        setAudioFile={setAudioFile}
        pdfFile={null}
        setPdfFile={vi.fn()}
        rememberSpeakers={false}
        setRememberSpeakers={vi.fn()}
        skipAgendaDetection={false}
        setSkipAgendaDetection={vi.fn()}
        autoDetectTopsFromPdf={false}
        setAutoDetectTopsFromPdf={vi.fn()}
      />,
    );

    await user.click(screen.getByRole('button', { name: /automatisch verarbeiten/i }));
    expect(onNext).toHaveBeenCalledTimes(1);
  });

  it('extracts TOPs from a PDF and replaces the TOP list', async () => {
    vi.mocked(extractAgendaDataFromPDF).mockResolvedValue({
      tops: ['Begruessung', 'Haushalt'],
      metadata: {
        committee: 'Hauptausschuss',
        date: '2026-06-30',
        location: 'Rathaus',
        title: 'Sitzung Hauptausschuss',
      },
    });
    const setTops = vi.fn();
    const setPdfFile = vi.fn();
    const setExportMetadata = vi.fn();
    const { container } = renderUploadStep({ setTops, setPdfFile, setExportMetadata });

    const input = container.querySelector<HTMLInputElement>('input[accept=".pdf,application/pdf"]');
    expect(input).not.toBeNull();
    const pdf = new File(['pdf'], 'einladung.pdf', { type: 'application/pdf' });

    fireEvent.change(input!, { target: { files: [pdf] } });

    await waitFor(() => {
      expect(extractAgendaDataFromPDF).toHaveBeenCalledWith(pdf, { model: 'qwen3:8b' });
      expect(setTops).toHaveBeenCalledWith(['Begruessung', 'Haushalt']);
      expect(setPdfFile).toHaveBeenCalledWith(pdf);
      expect(setExportMetadata).toHaveBeenCalledWith(
        expect.objectContaining({
          committee: 'Hauptausschuss',
          date: '2026-06-30',
          location: 'Rathaus',
          title: 'Sitzung Hauptausschuss',
        })
      );
    });
    expect(await screen.findByText(/2 TOPs erfolgreich extrahiert/i)).toBeInTheDocument();
  });

  it('keeps a PDF for pipeline extraction without extracting TOPs immediately in auto-PDF mode', async () => {
    const setPdfFile = vi.fn();
    const { container } = renderUploadStep({
      autoDetectTopsFromPdf: true,
      setPdfFile,
    });

    const input = container.querySelector<HTMLInputElement>('input[accept=".pdf,application/pdf"]');
    expect(input).not.toBeNull();
    const pdf = new File(['pdf'], 'einladung.pdf', { type: 'application/pdf' });

    fireEvent.change(input!, { target: { files: [pdf] } });

    await waitFor(() => {
      expect(setPdfFile).toHaveBeenCalledWith(pdf);
    });
    expect(extractAgendaDataFromPDF).not.toHaveBeenCalled();
    expect(screen.queryByText(/erfolgreich extrahiert/i)).not.toBeInTheDocument();
  });

  it('enables pipeline PDF detection separately from skip agenda detection', async () => {
    const user = userEvent.setup();
    const setAutoDetectTopsFromPdf = vi.fn();
    const setSkipAgendaDetection = vi.fn();
    renderUploadStep({ setAutoDetectTopsFromPdf, setSkipAgendaDetection });

    await user.click(screen.getByRole('checkbox', {
      name: /TOPs automatisch aus PDF erkennen und direkt verarbeiten/i,
    }));

    expect(setAutoDetectTopsFromPdf).toHaveBeenCalledWith(true);
    expect(setSkipAgendaDetection).toHaveBeenCalledWith(false);
  });

  it('can explicitly continue without TOPs or automatic TOP detection', async () => {
    const user = userEvent.setup();
    const setTops = vi.fn();
    const setSkipAgendaDetection = vi.fn();
    renderUploadStep({ tops: ['Begruessung'], setTops, setSkipAgendaDetection });

    await user.click(screen.getByRole('checkbox', {
      name: /ohne TOPs und ohne automatische TOP-Erkennung/i,
    }));

    expect(setSkipAgendaDetection).toHaveBeenCalledWith(true);
    expect(defaultProps.setAutoDetectTopsFromPdf).toHaveBeenCalledWith(false);
    expect(setTops).toHaveBeenCalledWith([]);
  });

  it('keeps persistent speaker memory off until the user opts in', async () => {
    const user = userEvent.setup();
    const setRememberSpeakers = vi.fn();
    renderUploadStep({ setRememberSpeakers });

    const checkbox = screen.getByRole('checkbox', {
      name: /Sprecher dauerhaft merken/i,
    });
    expect(checkbox).not.toBeChecked();
    expect(screen.getByText(/Dauerhafte Profile und Embeddings/i)).toBeInTheDocument();

    await user.click(checkbox);

    expect(setRememberSpeakers).toHaveBeenCalledWith(true);
  });
});

it('keeps original numbers and sections in manual input without positional numbering', () => {
  const setTops = vi.fn();
  const tops = ['2 Haushalt', '[Öffentlich] 2.1 Schulbau', '[Nichtöffentlich] 2.1 Vergabe', '7 Anfragen'];
  renderUploadStep({ tops, setTops });
  tops.forEach((top) => expect(screen.getByDisplayValue(top)).toBeInTheDocument());
  fireEvent.change(screen.getByLabelText('Tagesordnungseintrag 2'), {
    target: { value: '[Öffentlich] 2.2 Schulbau' },
  });
  expect(setTops).toHaveBeenCalledWith(['2 Haushalt', '[Öffentlich] 2.2 Schulbau', '[Nichtöffentlich] 2.1 Vergabe', '7 Anfragen']);
});
