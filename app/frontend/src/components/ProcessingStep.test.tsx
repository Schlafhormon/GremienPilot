import { render, screen } from '@testing-library/react';
import { expect, it } from 'vitest';
import ProcessingStep from './ProcessingStep';

it('shows live agenda activity without declaring incomplete work successful', () => {
  const { rerender } = render(<ProcessingStep progress={72} status="TOP-Verarbeitung" pipeline={{
    pipeline_id: 'job', status: 'processing', stage: 'agenda_detect', progress: 72, warnings: [],
    agenda_progress: { active_call: { model: 'gemma', phase: 'generating', elapsed_seconds: 1900,
      response_chunks: 1000, output_characters: 4000 }, processed_lines: [] },
  }} />);
  expect(screen.getByRole('status')).toHaveTextContent('31 Minuten');
  expect(screen.getByRole('status')).toHaveTextContent('4000 Ausgabezeichen');
  rerender(<ProcessingStep progress={72} status="Transkript bleibt zugänglich" pipeline={{
    pipeline_id: 'job', status: 'failed', stage: 'agenda_detect', progress: 72, warnings: [],
  }} />);
  expect(screen.getByRole('heading')).toHaveTextContent('Verarbeitung fehlgeschlagen');
});
