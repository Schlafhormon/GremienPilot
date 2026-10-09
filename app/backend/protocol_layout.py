"""Presentation-only Markdown layout; never categorize or rewrite stored prose."""
from dataclasses import dataclass, field

from markdown_it import MarkdownIt


LAYOUT_HEADINGS = {label.casefold() for label in (
    'Aus der Beratung', 'Beschlüsse und Festlegungen', 'Beschlüsse/Festlegungen',
    'Beschlüsse', 'Festlegungen', 'Abstimmung', 'Abstimmungen', 'Abstimmungsergebnis',
    'Maßnahmen', 'Offene Punkte', 'Unsicherheiten',
)}


@dataclass
class Run:
    text: str
    bold: bool = False
    italic: bool = False
    strike: bool = False
    code: bool = False


@dataclass
class Block:
    kind: str
    runs: list[Run] = field(default_factory=list)
    rows: list[list[list[Run]]] = field(default_factory=list)
    level: int = 0
    prefix: str = ''
    indent: int = 0
    quote: bool = False


def inline_runs(tokens):
    result, formats, links = [], {'bold': 0, 'italic': 0, 'strike': 0}, []
    tags = {'strong': 'bold', 'em': 'italic', 's': 'strike'}
    for token in tokens or []:
        tag, _, edge = token.type.partition('_')
        if tag in tags and edge in {'open', 'close'}:
            formats[tags[tag]] += 1 if edge == 'open' else -1
            continue
        if token.type == 'link_open':
            links.append((token.attrGet('href') or '', len(result)))
            continue
        if token.type == 'link_close':
            target, start = links.pop()
            if target and ''.join(run.text for run in result[start:]) != target:
                result.append(Run(' (' + target + ')'))
            continue
        if token.type == 'image':
            text = token.content
            target = token.attrGet('src') or ''
            if target:
                text += ' (' + target + ')'
        elif token.type in {'softbreak', 'hardbreak'}:
            text = '\n'
        else:
            text = token.content
        if text:
            result.append(Run(text, **{key: count > 0 for key, count in formats.items()},
                              code=token.type == 'code_inline'))
    return result


def plain(runs):
    return ''.join(run.text for run in runs)


def parse_protocol(text):
    """Keep block order, original words, list numbering and every table cell."""
    tokens = MarkdownIt('commonmark', {'html': False, 'typographer': False}).enable(['table', 'strikethrough']).parse(text)
    blocks, lists, quote_depth = [], [], 0
    index = 0
    while index < len(tokens):
        token = tokens[index]
        kind = token.type
        if kind == 'table_open':
            rows, row = [], []
            while tokens[index].type != 'table_close':
                current = tokens[index]
                if current.type == 'tr_open':
                    row = []
                elif current.type == 'inline':
                    row.append(inline_runs(current.children))
                elif current.type == 'tr_close':
                    rows.append(row)
                index += 1
            blocks.append(Block('table', rows=rows))
        elif kind in {'bullet_list_open', 'ordered_list_open'}:
            lists.append(dict(ordered=kind == 'ordered_list_open',
                              number=int(token.attrGet('start') or 1) - 1, first=False))
        elif kind in {'bullet_list_close', 'ordered_list_close'}:
            lists.pop()
        elif kind == 'list_item_open':
            lists[-1]['number'] += 1
            lists[-1]['first'] = True
        elif kind == 'blockquote_open':
            quote_depth += 1
        elif kind == 'blockquote_close':
            quote_depth -= 1
        elif kind in {'paragraph_open', 'heading_open'}:
            runs = inline_runs(tokens[index + 1].children)
            heading = kind == 'heading_open' or (not lists and
                plain(runs).strip().rstrip(':').casefold() in LAYOUT_HEADINGS)
            prefix = ''
            if lists and lists[-1]['first']:
                prefix = str(lists[-1]['number']) + '. ' if lists[-1]['ordered'] else '• '
                lists[-1]['first'] = False
            blocks.append(Block('heading' if heading else 'paragraph', runs=runs,
                                level=int(token.tag[1:]) if kind == 'heading_open' else 3 if heading else 0,
                                prefix=prefix, indent=len(lists), quote=quote_depth > 0))
            index += 2
        elif kind in {'fence', 'code_block'}:
            blocks.append(Block('code', runs=[Run(token.content, code=True)]))
        elif kind == 'hr':
            blocks.append(Block('rule'))
        index += 1
    return blocks


def text_table(rows):
    """Aligned plain-text columns, wrapping large cells without dropping text."""
    from textwrap import wrap
    cells = [[plain(cell) for cell in row] for row in rows]
    widths = [min(48, max(3, max(len(line) for row in cells for line in row[col].split('\n'))))
              for col in range(len(cells[0]))]
    result = []
    for index, row in enumerate(cells):
        columns = [[piece for line in cell.split('\n')
                    for piece in (wrap(line, width=width, replace_whitespace=False, drop_whitespace=False) or [''])]
                   for cell, width in zip(row, widths)]
        for line in range(max(len(column) for column in columns)):
            result.append(' | '.join((column[line] if line < len(column) else '').ljust(width)
                                     for column, width in zip(columns, widths)).rstrip())
        if index == 0:
            result.append('-+-'.join('-' * width for width in widths))
    return '\n'.join(result)


def render_text(text):
    return '\n\n'.join(text_table(block.rows) if block.kind == 'table' else
                         '-' * 40 if block.kind == 'rule' else
                         '  ' * block.indent + block.prefix + plain(block.runs)
                         for block in parse_protocol(text))
