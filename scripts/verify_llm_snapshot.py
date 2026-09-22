#!/usr/bin/env python3
"""Offline quality evaluation against approved human references. Never calls a model.

Inventory: --inventory testdata --output /tmp/private-quality/inventory.json
Evaluation: --reference reference.json --candidate candidate.json
            --adjudication human-review.json --output /tmp/private-quality/report.json

Human alignment is bound to both immutable artifacts; model-generated annotations
are rejected. Missing approval/reference data produces no fabricated quality score.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_sha(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def approved(value):
    if value.get('origin') != 'human' or value.get('approved') is not True or not str(value.get('approved_by', '')).strip():
        raise ValueError('Approved human reference/adjudication required')


def keyed(rows):
    result = {row['id']: row for row in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate identifiers')
    return result


def evaluate(reference, candidate, adjudication):
    approved(reference)
    approved(adjudication)
    if adjudication.get('reference_sha256') != sha(reference) or adjudication.get('candidate_sha256') != sha(candidate):
        raise ValueError('Human review belongs to a different immutable artifact')
    if candidate.get('source_files') != reference.get('source_files') or not reference.get('source_files'):
        raise ValueError('Sources differ or are missing')
    expected = keyed(reference['agenda'])
    actual = keyed(candidate['agenda'])
    mapping = adjudication['top_matches']
    if set(mapping) != set(actual) or any(value is not None and value not in expected for value in mapping.values()):
        raise ValueError('Every candidate TOP requires explicit human alignment')
    matched = {value for value in mapping.values() if value is not None}
    duplicate_tops = sum(max(0, list(mapping.values()).count(top)-1) for top in matched)
    numbering_errors = sum(actual[key].get('number') != expected[value].get('number') for key, value in mapping.items() if value)
    section_errors = sum(actual[key].get('section') != expected[value].get('section') for key, value in mapping.items() if value)
    lines = reference['transcript']
    ids = [line['id'] for line in lines]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate reference transcript IDs')
    if any(not set(line['top_ids']) <= set(expected) for line in lines):
        raise ValueError('Unknown reference TOP')
    assignments = candidate['assignments']
    if not set(assignments) <= set(ids):
        raise ValueError('Unknown candidate transcript lines')
    if any(not set(values) <= set(actual) for values in assignments.values()):
        raise ValueError('Unknown assigned TOP')
    mapped = {key: {mapping[top] or ('unmatched:' + top) for top in values}
              for key, values in assignments.items()}
    assignment_errors = sum(set(line['top_ids']) != mapped.get(line['id'], {None}) for line in lines)
    expected_boundaries = {i for i in range(1, len(lines)) if set(lines[i]['top_ids']) != set(lines[i-1]['top_ids'])}
    actual_boundaries = {i for i in range(1, len(lines)) if mapped.get(ids[i]) != mapped.get(ids[i-1])}
    facts = keyed(reference['facts'])
    claims = keyed(candidate['claims'])
    matches = adjudication['claim_matches']
    if set(matches) != set(claims):
        raise ValueError('Every generated claim needs independent human adjudication')
    kinds = {'decision', 'vote', 'action', 'open_point', 'discussion'}
    if any(row.get('kind') not in kinds for row in [*facts.values(), *claims.values()]):
        raise ValueError('Unknown reference/candidate fact category')
    source_validity = adjudication.get('claim_source_validity', {})
    if set(source_validity) != set(claims) or any(type(value) is not bool for value in source_validity.values()):
        raise ValueError('Independent source assessment required for every claim')
    remaining = adjudication.get('remaining_review_questions')
    if not isinstance(remaining, list) or any(not isinstance(question, str) or not question.strip() for question in remaining):
        raise ValueError('Independent assessment of remaining review work required')
    covered = set()
    unsupported = {'decision': 0, 'vote': 0, 'other': 0}
    attribution_errors = 0
    for claim_id, fact_ids in matches.items():
        if not isinstance(fact_ids, list) or not set(fact_ids) <= set(facts):
            raise ValueError('Unknown reference fact')
        claim = claims[claim_id]
        if claim.get('top_id') not in actual:
            raise ValueError('Claim refers to unknown TOP')
        if any(facts[ident]['kind'] != claim['kind'] for ident in fact_ids):
            raise ValueError('Human fact alignment has mismatched outcome types')
        if not fact_ids:
            kind = claim['kind'] if claim['kind'] in unsupported else 'other'
            unsupported[kind] += 1
        for ident in fact_ids:
            covered.add(ident)
            attribution_errors += int(mapping[claim['top_id']] != facts[ident]['top_id'])
    missing = {kind: sum(fact['kind'] == kind and ident not in covered for ident, fact in facts.items())
               for kind in ('decision', 'vote', 'action', 'open_point', 'discussion')}
    return dict(reference_sha256=sha(reference), candidate_sha256=sha(candidate),
        adjudication_sha256=sha(adjudication), evaluation='human_reference_offline',
        top_completeness=dict(found=len(matched), expected=len(expected), missing=sorted(set(expected)-matched),
                              extra=sum(value is None for value in mapping.values()), duplicates=duplicate_tops),
        original_number_errors=numbering_errors, meeting_section_errors=section_errors,
        assignment_errors=assignment_errors, transcript_lines=len(lines),
        unprocessed_lines=len(set(ids)-set(assignments)), attribution_errors=attribution_errors,
        boundary_errors=len(expected_boundaries ^ actual_boundaries),
        missing_boundaries=sorted(expected_boundaries-actual_boundaries),
        extra_boundaries=sorted(actual_boundaries-expected_boundaries),
        unsupported_outcomes=unsupported, missing_outcomes=missing,
        source_attribution_errors=sum(not valid for valid in source_validity.values()),
        remaining_manual_questions=len(remaining),
        candidate_reported_questions=len(candidate.get('review_questions', [])),
        processing_complete=candidate.get('processing_complete') is True)


def inventory(directory):
    records = []
    for path in sorted(directory.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in {'.pdf', '.mp3', '.wav', '.json'}:
            continue
        item = dict(path=str(path.relative_to(directory)), bytes=path.stat().st_size, sha256=file_sha(path))
        if path.suffix.lower() == '.pdf':
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                item['pages'] = len(pdf.pages)
                item['page_text_characters'] = [len(page.extract_text() or '') for page in pdf.pages]
        records.append(item)
    return dict(evaluation='inventory_only', model_calls=0, quality_metrics=None, files=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--adjudication', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.inventory:
        if any((args.reference, args.candidate, args.adjudication)):
            parser.error('Choose inventory or evaluation')
        report = inventory(args.inventory.resolve())
    else:
        if not all((args.reference, args.candidate, args.adjudication)):
            parser.error('Evaluation requires approved reference, candidate and human adjudication')
        reference, candidate, adjudication = [json.loads(path.read_text()) for path in
                                              (args.reference, args.candidate, args.adjudication)]
        # Verify originals read-only before accepting reference measurements.
        for source in reference.get('source_files', []):
            path = (args.reference.parent / source['path']).resolve()
            if file_sha(path) != source['sha256']:
                raise ValueError('Original source changed')
        report = evaluate(reference, candidate, adjudication)
    os.umask(0o077)
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(dict(evaluation=report['evaluation'], output=str(args.output))))


if __name__ == '__main__':
    main()
