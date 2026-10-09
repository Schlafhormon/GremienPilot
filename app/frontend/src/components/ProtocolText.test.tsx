import { render, screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import ProtocolText from './ProtocolText';

const example = `## Zu TOP 02: Entscheidung über eventuelle Einwendungen gegen die Niederschrift

Der Ausschuss beschließt einstimmig (9 : 0 : 0) die Niederschrift vom 19.01.2026.

Aus der Beratung

Heide (SPD) bittet, die Ausführungen zu TOP 7 zur Kleinen Elster nochmals zu prüfen.

Frau Stasch werde die entsprechende Stelle prüfen.

Abstimmung:

| Abstimmungsergebnis | Abstimmungsergebnis | Abstimmungsergebnis | Abstimmungsergebnis |
|---------------------|---------------------|---------------------|---------------------|
|                     | JA                  | NEIN                | ENTH.               |
| AUK                 | 9                   | 0                   | 0                   |`;

describe('ProtocolText', () => {
  it('renders the supplied protocol layout and all vote cells without Markdown delimiters', () => {
    const { container } = render(<ProtocolText text={example} />);
    expect(screen.getByRole('heading', { name: /Zu TOP 02/, level: 2 })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Aus der Beratung', level: 3 })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Abstimmung:', level: 3 })).toBeInTheDocument();
    const table = screen.getByRole('table');
    expect(within(table).getAllByRole('columnheader')).toHaveLength(4);
    expect(within(table).getAllByRole('cell').map(cell => cell.textContent)).toEqual(['', 'JA', 'NEIN', 'ENTH.', 'AUK', '9', '0', '0']);
    expect(screen.getByRole('region', { name: 'Protokolltabelle' })).toHaveAttribute('tabindex', '0');
    expect(container.textContent).not.toContain('##');
    expect(container.textContent).not.toContain('-----');
    expect(screen.getByText(/Heide \(SPD\).*TOP 7.*Kleinen Elster/)).toBeInTheDocument();
    expect(screen.getByText(/Frau Stasch/)).toBeInTheDocument();
  });

  it('formats explicit and bold layout headings, lists and inline emphasis in their original order', () => {
    const { container } = render(<ProtocolText text={'### Beschlüsse\n\n**Aus der Beratung**\n\nEin **wichtiger** Punkt.\n\n3. Erster Auftrag\n4. Zweiter Auftrag\n\nOffene Punkte\n\nNoch offen.'} />);
    expect(screen.getByRole('heading', { name: 'Beschlüsse' })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Aus der Beratung' })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Offene Punkte' })).toBeInTheDocument();
    expect(container.querySelector('ol')).toHaveAttribute('start', '3');
    expect(container.querySelector('p strong')).toHaveTextContent('wichtiger');
    expect(screen.getAllByRole('listitem').map(item => item.textContent)).toEqual(['Erster Auftrag', 'Zweiter Auftrag']);
    expect(container.querySelector('p:last-child')).toHaveTextContent('Noch offen.');
  });

  it('keeps malformed tables and ordinary paragraphs readable', () => {
    const { container } = render(<ProtocolText text={'Ein Absatz.\nZweite Zeile.\n\n| A | B |\nKein Tabellentrenner.\n\nLetzter Absatz.'} />);
    expect(screen.queryByRole('table')).not.toBeInTheDocument();
    expect(container.textContent).toContain('| A | B |');
    expect(screen.getByText('Letzter Absatz.')).toBeInTheDocument();
  });

  it('treats embedded HTML and unsafe links as data and never loads generated images', () => {
    const { container } = render(<ProtocolText text={'<img src="https://example.org/a" onerror="alert(1)">\n\n[Link](javascript:alert(1))\n\n![Plan](https://example.org/plan.png)'} />);
    expect(container.querySelector('img')).toBeNull();
    expect(container.querySelector('script')).toBeNull();
    expect(container.querySelector('a[href^="javascript:"]')).toBeNull();
    expect(container.textContent).toContain('Plan');
  });
});
