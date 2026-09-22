"""Lossless agenda labels and conservative reference resolution.

Storage/API keep their existing strings and independent stable top_ids. Parsed
labels are a view, never an identity or a reason to renumber an existing item.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


def fold(value: str) -> str:
    return unicodedata.normalize('NFKD', value.lower().replace('ß', 'ss')).encode('ascii', 'ignore').decode()


# Consume a complete hierarchy before considering a list delimiter.
LABEL = re.compile(
    r'^(?:(?i:TOP|Tagesordnungspunkt)\s*)?'
    r'(?P<number>\d+(?:\.\d+)*|[IVXLCDM]+|[a-z])'
    r'(?:[.)](?!\d)|\s*[:\-](?=\s|$)|(?=\s|$))\s*(?P<title>.*)$'
)
SECTION_PREFIX = re.compile(r'^\[(Öffentlich|Nichtöffentlich)\]\s*', re.I)


@dataclass(frozen=True)
class AgendaLabel:
    original_number: str | None
    title: str
    section: str | None = None

    @property
    def number_key(self) -> str | None:
        if self.original_number and re.fullmatch(r'\d+(?:\.\d+)*', self.original_number):
            return '.'.join(str(int(part)) for part in self.original_number.split('.'))
        return self.original_number


def parse_agenda_label(value: str) -> AgendaLabel:
    text = value.strip()
    section_match = SECTION_PREFIX.match(text)
    section = None
    if section_match:
        section = 'nonpublic' if fold(section_match[1]).startswith('nicht') else 'public'
        text = text[section_match.end():]
    match = LABEL.match(text)
    if match:
        if not re.match(r'^[.,/]\s*\d', match['title']):
            return AgendaLabel(match['number'], match['title'].strip(), section)
    return AgendaLabel(None, text, section)


def with_section(value: str, section: str | None) -> str:
    if not section or SECTION_PREFIX.match(value):
        return value
    return f"[{'Nichtöffentlich' if section == 'nonpublic' else 'Öffentlich'}] {value}"


def section_heading(value: str) -> str | None:
    text = re.sub(r'[^a-z0-9]+', ' ', fold(value)).strip()
    text = re.sub(r'^(?:top\s*)?(?:[ivx]+|\d+)\s+', '', text)
    if re.fullmatch(r'(?:nicht\s*)?(?:offentliche[rn]?|oeffentliche[rn]?) teil', text):
        return 'nonpublic' if text.startswith('nicht') else 'public'
    return None


def label_from_json(item: object) -> str | None:
    """Accept legacy strings and the lossless extraction object format."""
    if isinstance(item, str):
        return item.strip() or None
    if not isinstance(item, dict):
        return None
    title = item.get('title', item.get('top_title'))
    if not isinstance(title, str) or not title.strip():
        return None
    title = title.strip()
    number = item.get('number')
    # Decimal JSON numbers cannot preserve hierarchical spelling (2.10 != 2.1).
    if isinstance(number, str) and re.fullmatch(r'\d+(?:\.\d+)*|[IVXLCDM]+|[a-z]', number):
        if parse_agenda_label(title).original_number != number:
            title = f'{number}. {title}'
    section = item.get('section')
    if section not in ('public', 'nonpublic'):
        section = section_heading(f'{section} Teil') if isinstance(section, str) else None
    return with_section(title, section)
