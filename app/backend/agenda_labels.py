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


_UNITS = ['null', 'eins', 'zwei', 'drei', 'vier', 'fünf', 'sechs', 'sieben', 'acht', 'neun']
_TEENS = ['zehn', 'elf', 'zwölf', 'dreizehn', 'vierzehn', 'fünfzehn', 'sechzehn', 'siebzehn', 'achtzehn', 'neunzehn']
_TENS = ['zwanzig', 'dreißig', 'vierzig', 'fünfzig', 'sechzig', 'siebzig', 'achtzig', 'neunzig']
WORDS: dict[str, str] = {}
for n in range(100):
    word = (_UNITS[n] if n < 10 else _TEENS[n - 10] if n < 20 else
            (('ein' if n % 10 == 1 else _UNITS[n % 10]) + 'und' if n % 10 else '') + _TENS[n // 10 - 2])
    WORDS[fold(word)] = str(n)
    WORDS[fold(word.replace('ü', 'ue').replace('ö', 'oe'))] = str(n)
WORDS['ein'] = '1'

# Read the entire token, including unsupported suffixes, before validating it.
# That prevents TOP 2a / 2,1 / 2/1 from silently becoming TOP 2.
REFERENCE = re.compile(r'\b(?:top|tagesordnungspunkt|punkt)\s*(?P<token>[\w]+(?:[.,/\-][\w]+)*)(?!\w)', re.I)


@dataclass(frozen=True)
class AgendaReference:
    number: str | None
    original_number: str
    start: int
    end: int


def agenda_references(text: str) -> list[AgendaReference]:
    refs = []
    for match in REFERENCE.finditer(text):
        token = fold(match['token'])
        if re.fullmatch(r'\d+(?:\.\d+)*', token):
            number = '.'.join(str(int(part)) for part in token.split('.'))
        else:
            number = WORDS.get(token)
            if number is None and not any(char.isdigit() for char in token):
                continue
        # Spaced hierarchy and spoken decimals are deliberately unsupported.
        if re.match(r'(?:\s*[.,/]\s*\d|\s+(?:punkt|komma)\s+\w+)', fold(text[match.end():])):
            number = None
        refs.append(AgendaReference(number, match["token"] if token[0].isdigit() else number or token, match.start(), match.end()))
    return refs


def reference_sections(text: str) -> set[str]:
    """Consume negation together with öffentlich, including spaced spelling."""
    return {
        "nonpublic" if match["private"] else "public"
        for match in re.finditer(
            r"\b(?P<private>nicht[\s-]*)?(?:o|oe)ffentlich\w*\b", fold(text)
        )
    }


def reference_targets(text: str, tops: list[str]) -> tuple[bool, set[int]]:
    """Return (has reference, candidates). Only a singleton is unambiguous."""
    refs = agenda_references(text)
    if not refs:
        return False, set()
    numbers = {ref.number for ref in refs}
    if None in numbers or len(numbers) != 1:
        return True, set()
    # Coordinated/range references without repeated TOP marker are ambiguous.
    for ref in refs:
        continuation = re.match(r'\s*(?:,|und\b|bis\b|sowie\b|[-–])\s*(\w+)', fold(text[ref.end:]))
        if continuation and (continuation[1][0].isdigit() or continuation[1] in WORDS):
            return True, set()
    sections = reference_sections(text)
    if len(sections) > 1:
        return True, set()
    section = next(iter(sections), None)
    targets = {
        index for index, top in enumerate(tops)
        if (label := parse_agenda_label(top)).number_key in numbers
        and (section is None or label.section is None or label.section == section)
    }
    return True, targets
