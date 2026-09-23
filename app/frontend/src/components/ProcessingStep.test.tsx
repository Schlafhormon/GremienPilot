import { act, cleanup, render, screen } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import ProcessingStep from './ProcessingStep';
import type { ProcessingStepProps } from '../types';

afterEach(() => { cleanup(); vi.useRealTimers(); });

it('keeps the failed phase visible and advances time since the last model output', () => {
  vi.useFakeTimers();
  vi.setSystemTime(100_000);
  const props: ProcessingStepProps = { progress: 72, status: 'TOP-Zuordnung technisch unvollständig', pipeline: {
    pipeline_id: 'pipeline', progress: 72, stage: 'agenda_detect', status: 'failed', warnings: [], execution: {
      job_id: 'pipeline', kind: 'pipeline', state: 'failed', progress: { last_delta_at: 90, silence_seconds: 0, agenda_phase: 'primary:context' },
    },
  }};
  render(<ProcessingStep {...props} />);
  expect(screen.getByText('Letzte Modellausgabe vor 10 Sekunden.')).toBeInTheDocument();
  expect(screen.getByText('Prüfphase: primary:context')).toBeInTheDocument();
  act(() => { vi.advanceTimersByTime(5000); });
  expect(screen.getByText('Letzte Modellausgabe vor 15 Sekunden.')).toBeInTheDocument();
  expect(screen.getByText('Verarbeitung technisch unvollständig')).toBeInTheDocument();
});
