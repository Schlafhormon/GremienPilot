import { describe, expect, it } from 'vitest';
import { mergeSummarySession } from './summarySync';
import type { SessionSavePayload } from './types';

const baseline: SessionSavePayload = {
  session_id: 's', revision: 1, tops: ['A', 'B', 'C'], top_ids: ['a', 'b', 'c'],
  assignments: [], speaker_names: {}, skipped_assignment: false,
  summaries: { 0: 'Alt A', 1: 'Alt B', 2: 'Alt C' },
  summary_reviews: { 0: { source_links: [], review_warnings: [], fallback_used: false } },
  summary_states: { 0: { top_id: 'a', status: 'ready', origin: 'pipeline' } },
};

describe('summary job result merge', () => {
  it('merges result bundles by stable TOP identity and preserves manual text across polls', () => {
    const local = { ...baseline, summaries: { ...baseline.summaries, 1: 'Manuell B' },
      speaker_names: { S: 'Manueller Name' } };
    const remote = { ...baseline, revision: 4, tops: ['B', 'A', 'C'], top_ids: ['b', 'a', 'c'],
      summaries: { 0: 'Neu B', 1: 'Neu A', 2: 'Alt C' },
      summary_reviews: { 1: { source_links: [], review_warnings: [], fallback_used: true } },
      summary_states: { 1: { top_id: 'a', status: 'ready' as const, origin: 'manual_regeneration' } } };
    const merged = mergeSummarySession(local, baseline, remote);
    expect(merged.summaries).toEqual({ 0: 'Manuell B', 1: 'Neu A', 2: 'Alt C' });
    expect(merged.summary_reviews?.[1]).toEqual(remote.summary_reviews[1]);
    expect(merged.summary_states?.[1]).toEqual(remote.summary_states[1]);
    expect(merged.speaker_names).toEqual(local.speaker_names);
    expect(merged.revision).toBe(4);
    expect(mergeSummarySession(merged, remote, remote).summaries).toEqual(merged.summaries);
  });
});
