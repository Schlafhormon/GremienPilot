import type { AgendaProposals, TranscriptLine } from './types';

export function agendaSource(tops: string[], top_ids: string[], transcript: TranscriptLine[]) {
  return {
    tops,
    top_ids,
    transcript: transcript.map(({ line_id, speaker, text, start, end }) => ({
      line_id, speaker, text, start, end,
    })),
  };
}

export function proposalsAreValid(
  proposals: AgendaProposals | null | undefined,
  tops: string[], topIds: string[], transcript: TranscriptLine[],
): boolean {
  if (!proposals?.source || proposals.version !== 1) return false;
  if (topIds.length !== tops.length || new Set(topIds).size !== tops.length) return false;
  if (topIds.some((id) => !id) || transcript.some((line) => !line.line_id)) return false;
  if (new Set(transcript.map((line) => line.line_id)).size !== transcript.length) return false;
  const source = proposals.source;
  if (JSON.stringify(agendaSource(source.tops, source.top_ids, source.transcript)) !==
      JSON.stringify(agendaSource(tops, topIds, transcript))) return false;
  const result = proposals.result;
  const validTop = (index: number) => Number.isInteger(index) && index >= 0 && index < tops.length;
  return JSON.stringify(agendaSource(tops, topIds, result.transcript ?? [])) ===
    JSON.stringify(agendaSource(tops, topIds, transcript)) &&
    JSON.stringify(result.tops) === JSON.stringify(tops) &&
    result.assignments.length === transcript.length &&
    result.assignments.every((index) => index === null || validTop(index)) &&
    result.segments.every((segment) => validTop(segment.top_index) &&
      Number.isInteger(segment.start_index) && Number.isInteger(segment.end_index) &&
      segment.start_index >= 0 && segment.end_index >= segment.start_index &&
      segment.end_index < transcript.length);
}
