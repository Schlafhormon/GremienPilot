import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import PdfSources from './PdfSources';

describe('PDF sources', () => {
  it('shows distinct repeated numbers, hierarchy and original page links', () => {
    render(<PdfSources result={{ tops: [], metadata: { date: '2026-10-01' },
      document: { sha256: 'digest', page_count: 2, url: '/api/model-jobs/job/documents/digest' },
      items: [
        { id: 'a', number: '01', title: 'Rat', section: 'public', kind: 'agenda', parent_id: null, sources: [{ page: 1, quote: 'Rat' }] },
        { id: 'b', number: '01', title: 'Personal', section: 'nonpublic', kind: 'agenda', parent_id: 'a', sources: [{ page: 2, quote: null }] },
      ], metadata_sources: { date: [{ page: 1, quote: null }] },
    }} />);
    expect(screen.getByText(/Unterpunkt zu Rat/)).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Seite 2', hidden: true })).toHaveAttribute('href', '/api/model-jobs/job/documents/digest#page=2');
    expect(screen.getByText(/Dokument-ID: digest/)).toBeInTheDocument();
  });
});
