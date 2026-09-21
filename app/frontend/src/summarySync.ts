import type { SessionSavePayload, SummaryReview } from './types';

const equal = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);
const review = (value: SummaryReview | undefined) => value ? {
  ...value, structured: value.structured ?? null,
  source_links: value.source_links ?? [], review_warnings: value.review_warnings ?? [],
} : undefined;
const ids = (session: SessionSavePayload) => session.skipped_assignment || !session.tops.length
  ? [`whole-session:${session.session_id}`]
  : session.top_ids ?? session.tops.map((_, index) => `top-${index}`);

/** Three-way merge: job responses may arrive while the user is editing locally. */
export function mergeSummarySession(
  current: SessionSavePayload,
  baseline: SessionSavePayload,
  incoming: SessionSavePayload,
): SessionSavePayload {
  const merged = { ...current, revision: incoming.revision };
  // Refresh unchanged fields, keeping local edits (including export metadata).
  for (const key of ['tops', 'top_ids', 'transcript', 'assignments', 'speaker_names',
    'export_metadata', 'agenda_proposals', 'skipped_assignment'] as const) {
    if (equal(current[key], baseline[key])) Object.assign(merged, { [key]: incoming[key] });
  }
  merged.summaries = {};
  merged.summary_reviews = {};
  merged.summary_states = {};
  const currentIds = ids(current), baselineIds = ids(baseline), incomingIds = ids(incoming);
  ids(merged).forEach((id, index) => {
    const local = currentIds.indexOf(id), before = baselineIds.indexOf(id), remote = incomingIds.indexOf(id);
    // Keep text, review and generation metadata together as a single result.
    const locallyEdited = !equal(current.summaries[local], baseline.summaries[before]) ||
      !equal(review(current.summary_reviews?.[local]), review(baseline.summary_reviews?.[before]));
    const source = locallyEdited || remote < 0 ? current : incoming;
    const sourceIndex = source === current ? local : remote;
    if (source.summaries[sourceIndex] !== undefined) merged.summaries[index] = source.summaries[sourceIndex]!;
    if (source.summary_reviews?.[sourceIndex]) merged.summary_reviews![index] = source.summary_reviews[sourceIndex]!;
    if (source.summary_states?.[sourceIndex]) merged.summary_states![index] = source.summary_states[sourceIndex]!;
  });
  return merged;
}
