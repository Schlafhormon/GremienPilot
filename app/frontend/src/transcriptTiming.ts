import type { TranscriptLine } from './types';

export function mergedTiming(left: TranscriptLine, right: TranscriptLine): TranscriptLine['timing'] {
  if (!left.timing && !right.timing) return undefined;
  const offset = left.text.length + (left.text && right.text ? 1 : 0);
  return {
    source: left.timing?.source === 'word_alignment' && right.timing?.source === 'word_alignment'
      ? 'word_alignment' : 'mixed',
    words: [...(left.timing?.words ?? []), ...(right.timing?.words ?? []).map((w) => ({
      ...w, char_start: w.char_start + offset, char_end: w.char_end + offset,
    }))],
    segments: [...new Map([...(left.timing?.segments ?? []), ...(right.timing?.segments ?? [])]
      .map((s) => [s.segment_id, s])).values()],
  };
}
