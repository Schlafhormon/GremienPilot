#!/usr/bin/env python3
"""Unified research adapter: full-source model workflows, no legacy heuristics.

Input is a UTF-8 JSON list of {speaker, text, start?, end?}. Existing research
outputs are untouched. Run --help for the replacement command-line interface.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transcript', type=Path, required=True)
    parser.add_argument('--pdf', type=Path)
    parser.add_argument('--tops', type=Path, help='JSON array of original agenda labels')
    parser.add_argument('--model', default=None, help='Default: central LLM configuration')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'app/backend'))
    from agenda_detection import detect_agenda_from_transcript, segment_known_agenda
    from assignment_suggestions import TranscriptUtterance
    from summarize import summarize_segment, meeting_context_from_transcript
    from extract_tops import extract_agenda_data_from_pdf
    lines = json.loads(args.transcript.read_text())
    tops = json.loads(args.tops.read_text()) if args.tops else []
    if args.pdf:
        extraction = extract_agenda_data_from_pdf(str(args.pdf), model=args.model)
        if not extraction['processing_complete'] or extraction['review_required']:
            raise RuntimeError('PDF incomplete or requires review')
        tops = extraction['tops']
    rows = [TranscriptUtterance(line['speaker'], line['text']) for line in lines]
    agenda = (segment_known_agenda(rows, tops, model=args.model, use_llm=True) if tops
              else detect_agenda_from_transcript(rows, model=args.model, use_llm=True))
    if not agenda.llm.processing_complete or not agenda.llm.review_complete:
        raise RuntimeError('Agenda incomplete; no minutes generated')
    result = dict(agenda=asdict(agenda), summaries={})
    for index, title in enumerate(agenda.tops):
        source = [f"{line['speaker']}: {line['text']}" for i, line in enumerate(lines) if agenda.assignments[i] == index]
        if source:
            result['summaries'][index] = asdict(summarize_segment(title, '\n'.join(source),
                source_lines=source, model=args.model, meeting_context=meeting_context_from_transcript(lines)))
    with args.output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
