import type { PdfAgendaExtractionResult } from '../types';
import { API_BASE } from '../api';

export default function PdfSources({ result }: { result?: PdfAgendaExtractionResult | null }) {
  if (!result?.document?.url) return null;
  const url = API_BASE + result.document.url;
  return <details className="my-4 rounded border border-gray-200 bg-white p-3 text-sm">
    <summary>PDF-Quellen und geprüfte Originalagenda ({result.document.page_count} Seiten)</summary>
    <p className="my-2">Diese Originalauswertung bleibt auch nach manueller Bearbeitung zugänglich.</p>
    <a className="text-blue-700 underline" href={url} target="_blank" rel="noreferrer">Originaleinladung öffnen</a>
    <p className="break-all text-xs text-gray-500">Dokument-ID: {result.document.sha256}</p>
    <ul className="my-2 space-y-2">{result.items?.map(item => <li key={item.id}>
      {item.section && <span>[{item.section}] </span>}{item.number !== null && `${item.number} `}{item.title}
      {item.kind === 'heading' && ' (Abschnitt)'}
      {item.parent_id && ` · Unterpunkt zu ${result.items?.find(parent => parent.id === item.parent_id)?.title ?? item.parent_id}`}
      {' · '}{item.sources.map((source, index) => <a key={index} href={`${url}#page=${source.page}`}
        target="_blank" rel="noreferrer" className="mr-2 text-blue-700 underline" title={source.quote ?? undefined}>
        Seite {source.page}</a>)}
    </li>)}</ul>
    <ul>{Object.entries(result.metadata_sources ?? {}).map(([key, sources]) => <li key={key}>
      {({ committee: 'Gremium', date: 'Sitzungstermin', time: 'Uhrzeit', location: 'Ort', title: 'Titel' } as Record<string, string>)[key] ?? key}: {' '}
      {result.metadata[key as keyof typeof result.metadata]} {' '}
      {sources.map((source, index) => <a key={index} href={`${url}#page=${source.page}`} target="_blank"
        rel="noreferrer" className="mr-2 text-blue-700 underline">Seite {source.page}</a>)}
    </li>)}</ul>
  </details>;
}
