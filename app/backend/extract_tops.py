"""
PDF TOP extraction module for German municipal meeting invitations.

Extracts agenda items (Tagesordnungspunkte/TOPs) from PDF invitation documents
using pdfplumber for text extraction and Ollama LLM for intelligent parsing.

Configuration via environment variables:
- LLM_BASE_URL: API endpoint (local default: http://localhost:11434/v1,
  Docker default: http://ollama:11434/v1)
- LLM_MODEL: Model name (default: qwen3:8b)
"""

import logging
import os
import re
import json
from dataclasses import dataclass, field
from typing import Optional

from agenda_labels import label_from_json, parse_agenda_label, section_heading, with_section
from summarize import LLM_MAX_RETRIES, get_llm_config
from llm_transport import complete

logger = logging.getLogger(__name__)

# LLM server configuration (same as summarize.py)
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3:8b")
LLM_BASE_URL = get_llm_config().base_url
LLM_API_KEY = get_llm_config().api_key
NO_THINK_DIRECTIVE = "/no_think"

# Default system prompt for TOP extraction. Keep this short: reasoning models can
# otherwise spend the whole response budget on hidden reasoning and return no
# message.content through Ollama's OpenAI-compatible endpoint.
DEFAULT_EXTRACTION_PROMPT = """Du bist ein Extraktor. Antworte ohne Erklärung, nur mit einer nummerierten Liste der Tagesordnungspunkte.
Extrahiere aus der Einladung alle eigentlichen TOPs aus öffentlichem und nichtöffentlichem Teil.
Ignoriere Abschnittsüberschriften wie "TOP I. Öffentlicher Teil" und "TOP II. Nichtöffentlicher Teil" als eigene TOPs.
Ignoriere Bullet-Unterpunkte wie "- Fäkalienentsorgungssatzung - FES".
Entferne Zusatzinfos wie "BE:", "Beschlussvorlage:", "Antrag:" oder "Drucksache:".
Erhalte die Originalnummer inklusive Unterpunkten und Lücken; erfinde keine Nummern.
Jeder TOP kommt auf eine eigene Zeile im Format: 2.1. Titel (ohne Nummer, falls unbekannt).
Stelle bei bekanntem Abschnitt [Öffentlich] oder [Nichtöffentlich] voran."""

DEFAULT_AGENDA_DATA_EXTRACTION_PROMPT = """Du bist ein Extraktor. Antworte ohne Erklärung, nur mit validem JSON.
Extrahiere aus der Einladung:
- alle eigentlichen Tagesordnungspunkte aus öffentlichem und nichtöffentlichem Teil
- die Sitzungsmetadaten Gremium, Sitzungsdatum, Ort und Sitzungstitel

Regeln:
- Verwende als datum das Datum der Sitzung, nicht das Datum des Schreibens.
- datum muss im Format YYYY-MM-DD stehen, falls eindeutig erkennbar.
- Ignoriere Abschnittsüberschriften wie "TOP I. Öffentlicher Teil" und "TOP II. Nichtöffentlicher Teil" als eigene TOPs.
- Ignoriere Bullet-Unterpunkte wie "- Fäkalienentsorgungssatzung - FES".
- Entferne Zusatzinfos wie "BE:", "Beschlussvorlage:", "Antrag:" oder "Drucksache:".
- Erhalte Originalnummern als Strings inklusive Unterpunkten, führenden Nullen und Lücken.
- Nummern niemals aus Listenpositionen erzeugen. Unbekannte Nummer: null.
- section ist public, nonpublic oder null. Wiederholte Nummern bleiben separate TOPs.
- Nummerierte Unterpunkte sind eigene TOPs.
- Lass unbekannte Metadatenfelder als leere Strings.

JSON-Schema:
{
  "tops": [{"number": "2.1", "title": "Titel", "section": "public"}],
  "metadata": {
    "committee": "Gremium",
    "date": "YYYY-MM-DD",
    "location": "Ort",
    "title": "Sitzungstitel"
  }
}"""


@dataclass
class PdfSessionMetadata:
    committee: str = ""
    date: str = ""
    location: str = ""
    title: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "committee": self.committee,
            "date": self.date,
            "location": self.location,
            "title": self.title,
        }


@dataclass
class PdfAgendaExtractionResult:
    tops: list[str] = field(default_factory=list)
    metadata: PdfSessionMetadata = field(default_factory=PdfSessionMetadata)
    provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "tops": self.tops,
            "metadata": self.metadata.to_dict(),
            "provenance": self.provenance,
        }


def agenda_data_response_format():
    """Constrain the extraction shape, without changing summary generation."""
    metadata_keys = ['committee', 'date', 'location', 'title']
    schema = {'type': 'object', 'properties': {
        'tops': {'type': 'array', 'items': {'type': 'object', 'properties': {
            'number': {'type': ['string', 'null']}, 'title': {'type': 'string'},
            'section': {'enum': ['public', 'nonpublic', None]}},
            'required': ['number', 'title', 'section'], 'additionalProperties': False}},
        'metadata': {'type': 'object', 'properties': {k: {'type': 'string'} for k in metadata_keys},
            'required': metadata_keys, 'additionalProperties': False}},
        'required': ['tops', 'metadata'], 'additionalProperties': False}
    return {'type': 'json_schema', 'json_schema': {'name': 'pdf_agenda', 'strict': True, 'schema': schema}}


def build_extraction_system_prompt(system_prompt: Optional[str] = None) -> str:
    """Preserve the numbered-list contract and the configured reasoning mode."""
    return _build_extraction_prompt(DEFAULT_EXTRACTION_PROMPT, system_prompt)


def build_agenda_data_extraction_system_prompt(system_prompt: Optional[str] = None) -> str:
    """Return the structured extraction prompt with optional caller context."""
    return _build_extraction_prompt(DEFAULT_AGENDA_DATA_EXTRACTION_PROMPT, system_prompt)


def _build_extraction_prompt(base_prompt: str, system_prompt: Optional[str]) -> str:
    custom_prompt = (system_prompt or "").strip()
    if custom_prompt.startswith(NO_THINK_DIRECTIVE):
        custom_prompt = custom_prompt[len(NO_THINK_DIRECTIVE):].strip()
    # Preserve the legacy Qwen3 prompt when no API setting was supplied. An
    # explicit reasoning mode is controlled solely by the API, including none.
    prompt = (
        f"{NO_THINK_DIRECTIVE}\n{base_prompt}"
        if get_llm_config().reasoning_effort is None else base_prompt
    )
    if custom_prompt and custom_prompt != base_prompt.strip():
        prompt += (
            "\n\nZusätzliche fachliche Vorgaben des Nutzers. Diese nur anwenden, "
            "soweit sie der Extraktionsaufgabe, den Regeln und dem Ausgabeformat "
            "oben nicht widersprechen; diese haben Vorrang:\n"
            + custom_prompt
        )
    return prompt


def repair_common_pdf_text(text: str) -> str:
    """Repair common replacement-character artifacts from municipal PDFs."""
    replacements = {
        "f�r": "für",
        "F�r": "Für",
        "�ffentlich": "öffentlich",
        "�ffentlicher": "Öffentlicher",
        "�ffentliche": "öffentliche",
        "�ffentlichen": "öffentlichen",
        "nicht�ffentlich": "nichtöffentlich",
        "Nicht�ffentlich": "Nichtöffentlich",
        "�ber": "über",
        "gem��": "gemäß",
        "ordnungsgem��en": "ordnungsgemäßen",
        "Best�tigung": "Bestätigung",
        "Geb�hren": "Gebühren",
        "F�kalien": "Fäkalien",
        "Schlie�ung": "Schließung",
        "Ausschusssitzung": "Ausschusssitzung",
    }
    repaired = text
    for broken, fixed in replacements.items():
        repaired = repaired.replace(broken, fixed)
    return repaired


def normalize_session_date(value: str | None) -> str:
    """Normalize German dates to YYYY-MM-DD when possible."""
    if not value:
        return ""
    text = str(value).strip()
    iso_match = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", text)
    if iso_match:
        return iso_match.group(0)
    german_match = re.search(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b", text)
    if german_match:
        day, month, year = german_match.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"
    return text


def normalize_metadata(raw_metadata: object) -> PdfSessionMetadata:
    """Normalize metadata from JSON or heuristic extraction."""
    if not isinstance(raw_metadata, dict):
        return PdfSessionMetadata()

    def pick(*keys: str) -> str:
        for key in keys:
            value = raw_metadata.get(key)
            if value is not None and str(value).strip():
                return repair_common_pdf_text(str(value).strip())
        return ""

    return PdfSessionMetadata(
        committee=pick("committee", "gremium", "ausschuss"),
        date=normalize_session_date(pick("date", "datum", "sitzungsdatum")),
        location=pick("location", "ort", "sitzungsort"),
        title=pick("title", "titel", "sitzungstitel"),
    )


def merge_metadata(
    primary: PdfSessionMetadata,
    fallback: PdfSessionMetadata,
) -> PdfSessionMetadata:
    """Fill empty primary metadata fields from fallback values."""
    return PdfSessionMetadata(
        committee=primary.committee or fallback.committee,
        date=primary.date or fallback.date,
        location=primary.location or fallback.location,
        title=primary.title or fallback.title,
    )


def extract_session_metadata_from_text(pdf_text: str) -> PdfSessionMetadata:
    """Extract common session metadata directly from invitation text."""
    repaired_text = repair_common_pdf_text(pdf_text)
    lines = [line.strip() for line in repaired_text.splitlines() if line.strip()]

    committee = ""
    for line in lines[:12]:
        if re.search(
            r"\b(Ausschuss|Rat|Beirat|Gemeindevertretung|"
            r"Stadtverordnetenversammlung|Ortsbeirat)\b",
            line,
            flags=re.IGNORECASE,
        ):
            committee = line
            break

    title = ""
    title_match = re.search(
        r"\bzur\s+(.+?)\s+am\s+\d{1,2}\.\d{1,2}\.\d{4}\b",
        repaired_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if title_match:
        title = re.sub(r"\s+", " ", title_match.group(1)).strip()

    date = ""
    session_date_match = re.search(
        r"\bam\s+(\d{1,2}\.\d{1,2}\.\d{4})\b",
        repaired_text,
        flags=re.IGNORECASE,
    )
    if session_date_match:
        date = normalize_session_date(session_date_match.group(1))

    location = ""
    for index, line in enumerate(lines):
        location_match = re.match(
            r"^in\s+(?:das|den|die|der)\s+(.+)$",
            line,
            flags=re.IGNORECASE,
        )
        if not location_match:
            location_match = re.match(r"^im\s+(.+)$", line, flags=re.IGNORECASE)
        if location_match:
            location = location_match.group(1).strip().rstrip(".")
            break
        if line.lower() == "in" and index + 1 < len(lines):
            location = lines[index + 1].strip().rstrip(".")
            break

    return PdfSessionMetadata(
        committee=committee,
        date=date,
        location=location,
        title=title,
    )


def extract_tops_heuristically_from_text(pdf_text: str, *, audit: dict | None = None) -> list[str]:
    """Extract numbered agenda items directly from invitation text."""
    repaired_text = repair_common_pdf_text(pdf_text)
    lines = [line.strip() for line in repaired_text.splitlines() if line.strip()]
    try:
        start_index = next(
            index
            for index, line in enumerate(lines)
            if re.fullmatch(r"tagesordnung", line, flags=re.IGNORECASE)
        ) + 1
    except StopIteration:
        start_index = 0

    audit = audit if audit is not None else {}
    audit.update(candidates=[], ignored=[])
    section: str | None = None
    current: list[str] | None = None
    items: list[str] = []
    source_lines: list[dict] = []
    subpoints: list[dict] = []
    signed = False

    def flush_current() -> None:
        nonlocal current, source_lines, subpoints
        if not current:
            return
        title = re.sub(r"\s+", " ", " ".join(current)).strip()
        if title and not is_agenda_section_heading(title):
            items.append(with_section(title, section))
            audit['candidates'].append({'title': items[-1], 'source_lines': source_lines, 'subpoints': subpoints})
        current = None
        source_lines = []
        subpoints = []

    for line_index, original in enumerate(lines[start_index:], start_index):
        line = re.sub(r'[_=]{3,}|[-–—]{5,}', '', original).strip()
        if is_agenda_section_heading(line):
            flush_current()
            section = section_heading(line)
            signed = False
            continue
        artifact = pdf_artifact_kind(line)
        if not line or artifact or signed:
            audit['ignored'].append({'line_index': line_index, 'text': original,
                                     'reason': artifact or ('signature_block' if signed else 'separator')})
            if artifact in {'signature', 'office'}:
                if artifact == 'office' and current and len(current) > 1 and re.fullmatch(
                        r'(?:Dr\.\s+)?[A-ZÄÖÜ][a-zäöüß]+(?:[-\s][A-ZÄÖÜ][a-zäöüß]+){1,3}', current[-1]):
                    audit['ignored'].append({**source_lines.pop(), 'reason': 'name_before_office'})
                    current.pop()
                flush_current()
                signed = True
            continue
        if line.startswith(("-", "–", "•", "*")):
            if current:
                subpoints.append({'line_index': line_index, 'text': original})
            continue
        if line.startswith(("BE:", "Beschlussvorlage:", "Antrag:", "Drucksache:")):
            continue
        item = parse_agenda_label(line)
        if item.original_number is not None:
            flush_current()
            current = [line]
            source_lines = [{'line_index': line_index, 'text': original}]
            continue

        if current is not None:
            current.append(line)
            source_lines.append({'line_index': line_index, 'text': original})

    flush_current()
    return items


def pdf_artifact_kind(line: str) -> str | None:
    if re.match(r'^(?:gez\s*\.|gezeichnet\b|Unterschrift\b|Mit freundlichen Grüßen\b)', line, re.I):
        return 'signature'
    if re.fullmatch(r'(?:Seite\s+)?\d+\s*(?:(?:von|/)\s*\d+)?', line, re.I):
        return 'page_footer'
    if re.match(r'^(?:\d+\.\s*)?(?:(?:der|die)\s+)?(?:(?:stellv\.|stellvertretende[rns]?)\s+)?'
                r'(?:[A-Za-zÄÖÜäöüß]*vorsitzende[rns]?|Bürgermeister(?:in)?|Schriftführer(?:in)?)\b', line, re.I):
        return 'office'
    if re.match(r'^(?:Beleg:|Dokumentnummer:|www\.|https?://)', line, re.I):
        return 'page_footer'
    return None


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text content from a PDF file.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Extracted text as a single string

    Raises:
        RuntimeError: If pdfplumber is not installed or extraction fails
    """
    try:
        import pdfplumber
    except ImportError:
        raise RuntimeError(
            "pdfplumber nicht installiert. Installieren Sie mit: uv add pdfplumber"
        )

    logger.info("Extracting text from uploaded PDF")

    try:
        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                # Marginal rotated stamps are not body text. Preserve a fully
                # rotated document; filter only a small minority orientation.
                upright = sum(bool(c.get('upright', True)) for c in page.chars)
                body = page.filter(lambda obj: obj.get('object_type') != 'char' or obj.get('upright', True)
                                   or (obj.get('x0', 0) > 40 and obj.get('x1', page.width) < page.width - 40)) if upright > len(page.chars) * .8 else page
                page_text = body.extract_text()
                if page_text:
                    text_parts.append(page_text)
                    logger.debug(f"Page {i + 1}: extracted {len(page_text)} characters")

        full_text = "\n\n".join(text_parts)
        logger.info(f"Total extracted text: {len(full_text)} characters from {len(text_parts)} pages")
        return full_text

    except Exception as e:
        logger.error(
            "Failed to extract text from uploaded PDF (%s)",
            e.__class__.__name__,
        )
        raise RuntimeError(f"PDF-Text konnte nicht extrahiert werden: {str(e)}")


def extract_tops_from_text(
    pdf_text: str,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> list[str]:
    """
    Extract TOPs from PDF text using LLM.

    Args:
        pdf_text: Full text extracted from the PDF
        model: LLM model to use (default: from env or qwen3:8b)
        system_prompt: Optional context supplementing the mandatory extraction prompt

    Returns:
        List of TOP titles (including numbering)

    Raises:
        RuntimeError: If OpenAI client is not installed or LLM call fails
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError(
            "OpenAI client nicht installiert. Installieren Sie mit: uv add openai"
        )

    config = get_llm_config(model)
    actual_model = config.model
    actual_system_prompt = build_extraction_system_prompt(system_prompt)

    logger.info(f"Extracting TOPs using model: {actual_model}")

    client = OpenAI(
        base_url=config.base_url,
        api_key=config.api_key,
        timeout=config.timeout_seconds,
        max_retries=LLM_MAX_RETRIES,
    )

    user_prompt = f"""Extrahiere alle Tagesordnungspunkte aus diesem Einladungsdokument:

{pdf_text}

TOPs:"""

    try:
        response = complete(client, config,
            model=actual_model,
            messages=[
                {"role": "system", "content": actual_system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=2048,
            temperature=0.1,  # Very low temperature for consistent extraction
            **config.reasoning_options,
        )

        raw_response = response.choices[0].message.content or ""
        logger.debug("LLM TOP extraction returned %s characters", len(raw_response))

        # Parse the response into individual TOPs
        tops = parse_tops_response(raw_response)
        logger.info(f"Extracted {len(tops)} TOPs")

        return tops

    except Exception as e:
        logger.error("LLM TOP extraction failed (%s)", e.__class__.__name__)
        raise RuntimeError(f"TOP-Extraktion fehlgeschlagen: {str(e)}")


def parse_tops_response(response_text: str) -> list[str]:
    """Parse standard lists without discarding original numbering or scope."""
    tops = []
    section = None
    for line in response_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        heading = section_heading(line)
        if heading:
            section = heading
            continue
        if line.startswith(("●", "•", "-", "*", "–")):
            continue
        label = parse_agenda_label(line)
        if label.original_number is not None:
            if label.title:
                tops.append(with_section(line, section))
        elif len(line) > 5 and not any(
            skip in line.lower() for skip in ["beschlussvorlage", "antrag:", "drucksache", "seite"]
        ):
            tops.append(with_section(line, section))
    return tops


def _extract_json_object(response_text: str) -> dict[str, object] | None:
    """Extract a JSON object from plain or fenced model output."""
    text = response_text.strip()
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    elif not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def title_comparison_key(title):
    # A leading council-paper identifier and typographic quotation marks are
    # formatting differences, not missing title content. Retain originals.
    title = re.sub(r'^\d{1,6}/\d{2,4}\s+', '', title)
    title = title.translate(str.maketrans({'„': '"', '“': '"', '”': '"', '«': '"', '»': '"'}))
    return re.sub(r'\s+', ' ', re.sub(r'\s*-\s*', '-', title)).casefold().strip()


def compatible_titles(source, model):
    refs = [re.match(r'^(\d{1,6}/\d{2,4})\s+', t) for t in (source, model)]
    if all(refs) and refs[0][1] != refs[1][1]:
        return False
    return title_comparison_key(source) == title_comparison_key(model)


def parse_agenda_data_response(
    response_text: str,
    fallback_text: str = "",
) -> PdfAgendaExtractionResult:
    """Parse structured LLM output, falling back to legacy TOP parsing."""
    payload = _extract_json_object(response_text)
    audit = {'version': 'pdf-reconciliation-v1', 'raw_model_response': response_text, 'conflicts': []}
    fallback_tops = (
        extract_tops_heuristically_from_text(fallback_text, audit=audit)
        if fallback_text
        else []
    )
    fallback_metadata = (
        extract_session_metadata_from_text(fallback_text)
        if fallback_text
        else PdfSessionMetadata()
    )
    malformed_json = payload is None and response_text.lstrip().startswith(('{', '```json'))
    if malformed_json:
        audit['conflicts'].append({'kind': 'invalid_model_json', 'resolution': 'numbered_source_only'})
    raw_tops = (payload.get("tops") or payload.get("agenda") or []) if payload else ([] if malformed_json else response_text)
    if isinstance(raw_tops, list):
        # Structured arrays are already item boundaries; never enumerate them
        # into invented agenda numbers or discard short unnumbered titles.
        tops = []
        section = None
        for item in raw_tops:
            label = label_from_json(item)
            if not label:
                continue
            heading = section_heading(label)
            if heading:
                section = heading
            else:
                tops.append(with_section(label, section))
    else:
        tops = parse_tops_response(str(raw_tops))

    audit['model_tops'] = tops[:]
    if fallback_tops:
        # Match occurrences, not list length. Original numbered boundaries and
        # their actual text win; model-only items remain explicit conflicts.
        unmatched = list(enumerate(tops))
        for candidate in audit['candidates']:
            source = parse_agenda_label(candidate['title'])
            matches = [(i, top) for i, top in unmatched
                       if (parse_agenda_label(top).number_key in {None, source.number_key})
                       and (parse_agenda_label(top).section in {None, source.section})
                       and compatible_titles(source.title, parse_agenda_label(top).title)]
            matches = [(i, top) for i, top in matches if sum(
                compatible_titles(parse_agenda_label(other).title, parse_agenda_label(top).title)
                and parse_agenda_label(top).number_key in {None, parse_agenda_label(other).number_key}
                and parse_agenda_label(top).section in {None, parse_agenda_label(other).section}
                for other in fallback_tops) == 1]
            if matches:
                i, top = matches[0]
                unmatched.remove((i, top))
                candidate.update(origin='source_and_model', model_index=i)
                candidate['match_basis'] = ('exact_title' if source.title == parse_agenda_label(top).title
                                            else 'format_normalized_title_original_preserved')
            else:
                candidate['origin'] = 'numbered_source'
                audit['conflicts'].append({'kind': 'source_not_confirmed_by_model', 'source': candidate['title']})
        for i, top in unmatched:
            audit['conflicts'].append({'kind': 'model_item_not_confirmed_by_source', 'model_index': i, 'title': top})
        tops = fallback_tops
    else:
        tops = [top for top in tops if not pdf_artifact_kind(parse_agenda_label(top).title)]
        if fallback_text and tops:
            audit['conflicts'].append({'kind': 'no_numbered_source_boundaries', 'titles': tops[:]})
    audit['requires_review'] = bool(audit['conflicts'])

    metadata = merge_metadata(
        normalize_metadata((payload.get("metadata") or payload) if payload else None),
        fallback_metadata,
    )
    return PdfAgendaExtractionResult(tops=tops, metadata=metadata, provenance=audit)


def is_agenda_section_heading(value: str) -> bool:
    """Return true for agenda section labels, not actual agenda items."""
    return section_heading(value) is not None


def extract_tops_from_pdf(
    pdf_path: str,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> list[str]:
    """
    Extract TOPs from a PDF file (convenience function).

    Combines text extraction and LLM parsing in one call.

    Args:
        pdf_path: Path to the PDF file
        model: LLM model to use (optional)
        system_prompt: Custom system prompt (optional)

    Returns:
        List of TOP titles
    """
    pdf_text = extract_text_from_pdf(pdf_path)
    return extract_tops_from_text(pdf_text, model, system_prompt)


def extract_agenda_data_from_text(
    pdf_text: str,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> PdfAgendaExtractionResult:
    """
    Extract TOPs and session metadata from PDF text using LLM plus heuristics.

    Args:
        pdf_text: Full text extracted from the PDF
        model: LLM model to use
        system_prompt: Optional additional system prompt context

    Returns:
        Structured agenda extraction result
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError(
            "OpenAI client nicht installiert. Installieren Sie mit: uv add openai"
        )

    config = get_llm_config(model)
    actual_model = config.model
    actual_system_prompt = build_agenda_data_extraction_system_prompt(system_prompt)
    repaired_text = repair_common_pdf_text(pdf_text)

    logger.info("Extracting TOPs and session metadata using model: %s", actual_model)

    client = OpenAI(
        base_url=config.base_url,
        api_key=config.api_key,
        timeout=config.timeout_seconds,
        max_retries=LLM_MAX_RETRIES,
    )

    user_prompt = f"""Extrahiere Tagesordnungspunkte und Sitzungsmetadaten aus diesem Einladungsdokument:

{repaired_text}

JSON:"""

    try:
        response = complete(client, config,
            model=actual_model,
            messages=[
                {"role": "system", "content": actual_system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=3072,
            temperature=0.1,
            response_format=agenda_data_response_format(),
            **config.reasoning_options,
        )

        raw_response = response.choices[0].message.content or ""
        logger.debug("LLM agenda data extraction returned %s characters", len(raw_response))

        result = parse_agenda_data_response(raw_response, fallback_text=repaired_text)
        logger.info(
            "Extracted %s TOPs with metadata fields: %s",
            len(result.tops),
            [key for key, value in result.metadata.to_dict().items() if value],
        )
        return result

    except Exception as e:
        logger.error("LLM agenda data extraction failed (%s)", e.__class__.__name__)
        raise RuntimeError(f"PDF-Datenextraktion fehlgeschlagen: {str(e)}")


def extract_agenda_data_from_pdf(
    pdf_path: str,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> PdfAgendaExtractionResult:
    """
    Extract TOPs and session metadata from a PDF file.

    Keeps PDF text extraction shared with the legacy TOP-only path.
    """
    pdf_text = extract_text_from_pdf(pdf_path)
    return extract_agenda_data_from_text(pdf_text, model, system_prompt)
