import type { ReactNode } from 'react';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

// These literal layout labels also occur without Markdown in the LoRA output.
// Recognizing their typography does not assign categories to the protocol text.
const layoutHeadings = new Set([
  'aus der beratung', 'beschlüsse und festlegungen', 'beschlüsse/festlegungen',
  'beschlüsse', 'festlegungen', 'abstimmung', 'abstimmungen', 'abstimmungsergebnis',
  'maßnahmen', 'offene punkte', 'unsicherheiten',
]);

function plainText(children: ReactNode): string {
  if (typeof children === 'string' || typeof children === 'number') return String(children);
  if (Array.isArray(children)) return children.map(plainText).join('');
  if (children && typeof children === 'object' && 'props' in children) {
    return plainText((children.props as { children?: ReactNode }).children);
  }
  return '';
}

export default function ProtocolText({ text }: { text: string }) {
  return (
    <div className="protocol-prose min-w-0 text-sm text-gray-700" aria-label="Formatierter Protokolltext">
      <Markdown remarkPlugins={[remarkGfm]} components={{
        p: ({ children }) => layoutHeadings.has(plainText(children).trim().replace(/:$/, '').toLocaleLowerCase('de'))
          ? <h3 className="protocol-layout-heading">{children}</h3>
          : <p>{children}</p>,
        table: ({ children }) => (
          <div className="protocol-table-scroll" role="region" aria-label="Protokolltabelle" tabIndex={0}>
            <table>{children}</table>
          </div>
        ),
        a: ({ href, children }) => href ? <a href={href} target="_blank" rel="noopener noreferrer">{children}</a> : <span>{children}</span>,
        // Protocols are text. Keep image descriptions readable without loading
        // external resources from generated Markdown.
        img: ({ alt, src }) => <span>{alt}{src ? ` (${src})` : ''}</span>,
      }}>{text}</Markdown>
    </div>
  );
}
