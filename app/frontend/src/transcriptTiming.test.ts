import { expect, it } from 'vitest';
import { mergedTiming } from './transcriptTiming';
import { agendaSource, proposalsAreValid } from './agendaProposals';
import type { TranscriptLine, AgendaProposals } from './types';

it('merges aligned words with adjusted text offsets and original audio intervals', () => {
  const left: TranscriptLine = { line_id: 'a', text: 'Hallo.', speaker: 'S', start: 3, end: 4,
    timing: { source: 'word_alignment', words: [{ text: 'Hallo.', char_start: 0, char_end: 6, start: 3, end: 4 }],
      segments: [{ segment_id: 'raw', start: 0, end: 9 }] } };
  const right: TranscriptLine = { ...left, line_id: 'b', start: 6, end: 7,
    timing: { ...left.timing!, words: [{ ...left.timing!.words[0]!, start: 6, end: 7 }] } };
  const timing = mergedTiming(left, right)!;
  expect(timing.words[1]).toMatchObject({ char_start: 7, char_end: 13, start: 6, end: 7 });
  expect(timing.segments).toHaveLength(1);
  const source = agendaSource(['Begrüßung'], ['t'], [left]);
  const proposal: AgendaProposals = { version: 1, source, result: { tops: source.tops, transcript: [left],
    assignments: [0], segments: [], strategy: 'test', uncertain_count: 0 } };
  expect(proposalsAreValid(proposal, source.tops, source.top_ids, [left])).toBe(true);
  expect(proposalsAreValid(proposal, source.tops, source.top_ids,
    [{ ...left, timing: { ...left.timing!, source: 'manual_estimate' } }])).toBe(false);
});
