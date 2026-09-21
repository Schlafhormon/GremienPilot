#!/usr/bin/env python3
"""Isolated real inference from a session's immutable agenda source, never current assignments.

Run on the Docker host with the backend's Python dependencies. No production API
writes, no audio decoding, no container lifecycle operations. Private artifacts
contain transcripts and model replies; keep the output directory access restricted.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, required=True, help='Production SQLite, opened mode=ro only')
    parser.add_argument('--session', required=True)
    parser.add_argument('--output', type=Path, required=True, help='Separate private test directory')
    parser.add_argument('--base-url', default='http://127.0.0.1:11434/v1')
    parser.add_argument('--model', default='qwen3.5:9b')
    args = parser.parse_args()
    database = args.database.resolve()
    output = args.output.resolve()
    if output == database.parent or database.is_relative_to(output):
        parser.error('Output must be separate from the production data directory')
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(output, 0o700)
    os.umask(0o077)

    def read_database():
        return sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)

    def ensure_idle():
        with read_database() as connection:
            for table in ('transcription_jobs', 'pipeline_jobs', 'summary_jobs'):
                active = connection.execute(
                    f"SELECT count(*) FROM {table} WHERE status NOT IN ('completed','failed','cancelled')"
                ).fetchone()[0]
                if active:
                    raise SystemExit('User job active; stop isolated inference and resume later')

    ensure_idle()
    with read_database() as connection:
        row = connection.execute('SELECT agenda_proposals_json FROM sessions WHERE session_id=?',
                                 (args.session,)).fetchone()
        if row is None or not row[0]:
            raise RuntimeError('No immutable agenda proposal snapshot available')
        proposals = json.loads(row[0])
    source = proposals['source']
    original_hash = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    previous = output / 'source-hash.txt'
    if previous.exists() and previous.read_text() != original_hash:
        raise RuntimeError('Output directory contains a different source snapshot')
    previous.write_text(original_hash)
    (output / 'original.json').write_text(json.dumps(proposals, ensure_ascii=False, indent=2))
    os.environ.update(
        LLM_BASE_URL=args.base_url, LLM_MODEL=args.model, LLM_REASONING_EFFORT='none',
        LLM_OLLAMA_NATIVE='true', LLM_CONTEXT_TOKENS='16384', LLM_CPU_THREADS='16',
        LLM_TIMEOUT_SECONDS='1800', AGENDA_DETECTION_TIMEOUT_SECONDS='1800',
        LLM_MAX_RETRIES='0', LLM_CHUNK_CHARS='7000', LLM_REPAIR_SPLIT_DEPTH='1',
        LLM_SUMMARY_FACT_REVIEW_MAX_CALLS='3',
        LLM_SUMMARY_GROUNDING_MAX_CALLS='32',
        LLM_SUMMARY_GROUNDING_THINK='false',
        LLM_SUMMARY_FACT_REVIEW_THINK='true', LLM_SUMMARY_FACT_REVIEW_MAX_TOKENS='5120',
        AGENDA_DETECTION_CHUNK_LINES='160', AGENDA_DETECTION_CHUNK_OVERLAP_LINES='12',
        AGENDA_DETECTION_GAP_REVIEW_MAX_CALLS='3',
        AGENDA_DETECTION_BOUNDARY_REVIEW_MAX_CALLS='4',
        LLM_CACHE_DIR=str(output / 'cache'), LLM_AUDIT_DIR=str(output / 'audit'),
        PERSISTENCE_DB_PATH=str(output / 'verification.sqlite3'),
    )
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app' / 'backend'))
    import llm_transport
    from agenda_detection import segment_known_agenda
    from assignment_suggestions import TranscriptUtterance
    from summarize import summarize_segment, build_summary_review, meeting_context_from_transcript
    from persistence import init_db, save_session, load_session
    original_complete = llm_transport._complete

    def guarded_complete(*positional, **kwargs):
        ensure_idle()  # Also recheck between calls, not only at startup.
        return original_complete(*positional, **kwargs)
    llm_transport._complete = guarded_complete

    transcript = source['transcript']
    tops = source['tops']
    started_at = time.time()
    began = time.monotonic()
    def save_progress(usage):
        (output / 'agenda-progress.json').write_text(json.dumps(asdict(usage), ensure_ascii=False, indent=2))
        print(f"Agenda: {len(usage.processed_lines)}/{len(transcript)} lines validated; {usage.failed_calls} failed calls", flush=True)
    result = segment_known_agenda([TranscriptUtterance(line['speaker'], line['text']) for line in transcript],
                                  tops, use_llm=True, progress_callback=save_progress)
    agenda_result = dict(asdict(result), transcript=transcript, warnings=result.llm.warnings)
    (output / 'assignment.json').write_text(json.dumps(agenda_result, ensure_ascii=False, indent=2))
    init_db()
    state = dict(source, current_step=2, assignments=result.assignments,
                 agenda_proposals={'version': 1, 'source': source, 'result': agenda_result},
                 summaries={}, summary_reviews={})
    save_session('isolated-verification', state)
    errors = []
    for top_index, title in enumerate(tops):
        ensure_idle()
        lines = [line for i, line in enumerate(transcript) if result.assignments[i] == top_index]
        if not lines:
            (output / f'summary-{top_index}.json').unlink(missing_ok=True)
            errors.append({'top_index': top_index, 'reason': 'no_assigned_lines'})
            continue
        try:
            summary = summarize_segment(title, '\n'.join(f"{line['speaker']}: {line['text']}" for line in lines),
                                        meeting_context=meeting_context_from_transcript(transcript))
            review = build_summary_review(structured=summary.structured, summary=summary.summary, lines=lines)
            artifact = dict(asdict(summary), source_line_indices=[i for i, a in enumerate(result.assignments) if a == top_index],
                            review=review.to_dict())
            (output / f'summary-{top_index}.json').write_text(json.dumps(artifact, ensure_ascii=False, indent=2))
            state['summaries'][top_index] = summary.summary
            state['summary_reviews'][top_index] = {
                'structured': summary.structured.to_dict() if summary.structured else None,
                'source_links': artifact['review']['source_links'],
                'review_warnings': artifact['review']['warnings'],
                'fallback_used': summary.fallback_used, 'chunks_processed': summary.chunks_processed,
                'llm_usage': summary.llm_usage, 'duration_seconds': summary.duration_seconds,
            }
            save_session('isolated-verification', state)
            if summary.llm_usage.get('grounding_incomplete'):
                errors.append({'top_index': top_index, 'reason': 'incomplete_source_check'})
            print(f"Summary {top_index+1}/{len(tops)} saved ({len(lines)} source lines)", flush=True)
        except Exception as exc:
            (output / f'summary-{top_index}.json').unlink(missing_ok=True)
            errors.append({'top_index': top_index, 'reason': type(exc).__name__})
            print(f"Summary {top_index+1}/{len(tops)} failed ({type(exc).__name__})", flush=True)
    reloaded = load_session('isolated-verification')
    assert reloaded['assignments'] == result.assignments
    assert reloaded['transcript'] == transcript
    assert reloaded['summaries'] == state['summaries']
    report = {'seconds': time.monotonic()-began, 'started_at': started_at, 'completed_at': time.time(),
              'source_sha256': original_hash, 'model': args.model,
              'llm': asdict(result.llm), 'unassigned_lines': result.assignments.count(None),
              'saved_summaries': len(state['summaries']), 'errors': errors}
    (output / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != 'llm'}), flush=True)
    if errors or result.llm.status != 'success':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
