import type { AgendaProposals, TranscriptLine } from './types';

export function agendaSource(tops: string[], top_ids: string[], transcript: TranscriptLine[]) {
  return {
    tops,
    top_ids,
    transcript: transcript.map(({ line_id, speaker, text, start, end, timing }) => ({
      line_id, speaker, text, start, end, ...(timing ? { timing } : {}),
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
  const details = result.llm?.line_results;
  if (details?.length) {
    const identities = result.llm?.provenance?.identities ?? [];
    if (identities.length !== tops.length || new Set(identities.map(top => top.top_id)).size !== identities.length ||
        identities.some(top => !validTop(top.top_index) || top.title !== tops[top.top_index] ||
          (top.top_uid !== undefined && top.top_uid !== topIds[top.top_index]))) return false;
    const byId = new Map(identities.map(top => [top.top_id, top.top_index]));
    const sources = new Map(transcript.map(line => [line.line_id, line.text]));
    if (details.length !== transcript.length || details.some((line, index) =>
      line.index !== index || line.line_id !== transcript[index]?.line_id ||
      !Array.isArray(line.top_ids) || new Set(line.top_ids).size !== line.top_ids.length ||
      line.top_ids.some(id => !byId.has(id)) ||
      !['assigned', 'unassigned', 'not_processed'].includes(line.status) ||
      (line.status === 'assigned') !== (line.top_ids.length > 0) ||
      result.assignments[index] !== (line.top_ids.length === 1 ? byId.get(line.top_ids[0]!) : null) ||
      !Array.isArray(line.evidence) || line.evidence.some(ref => !sources.has(ref.line_id) || !sources.get(ref.line_id)!.includes(ref.quote)))) return false;
    if (result.llm?.agenda_states?.some(state => !byId.has(state.top_id))) return false;
  }
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
