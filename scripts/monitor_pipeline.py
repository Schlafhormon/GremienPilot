"""Local content-free acceptance measurements; never reads prompts or transcripts.

Run from the repository root: python scripts/monitor_pipeline.py JOB_ID
Reports to ignored data/measurements. Sampling continues across Docker outages.
"""
import argparse
import hashlib
import inspect
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import subprocess
import time
import urllib.request


def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=20)
        return result.stdout.strip() if result.returncode == 0 else 'unavailable'
    except (OSError, subprocess.TimeoutExpired):
        return 'unavailable'


def snapshot(db_path, job_id):
    with sqlite3.connect(f'{db_path.resolve().as_uri()}?mode=ro', uri=True, timeout=5) as db:
        db.row_factory = sqlite3.Row
        row = db.execute('SELECT state,attempt,progress,heartbeat_at FROM durable_jobs WHERE job_id=?', (job_id,)).fetchone()
        data = dict(row)
        progress = json.loads(data.pop('progress') or '{}')
        data['progress'] = {k: progress[k] for k in ('phase', 'pdf_phase', 'agenda_phase', 'page',
            'round', 'processed_lines', 'total_lines', 'model_calls', 'elapsed_seconds') if k in progress}
        data['steps'] = db.execute('SELECT COUNT(*) FROM durable_steps WHERE job_id=?', (job_id,)).fetchone()[0]
        transcript = db.execute("SELECT value,completed_at FROM durable_steps WHERE job_id=? AND step_key='pipeline:transcript'", (job_id,)).fetchone()
        data['transcript_completed_at'] = transcript[1] if transcript else None
        data['transcript_sha256'] = hashlib.sha256(transcript[0].encode()).hexdigest() if transcript else None
        data['published'] = bool(db.execute("SELECT 1 FROM durable_steps WHERE job_id=? AND step_key='pipeline:published'", (job_id,)).fetchone())
        pipeline = db.execute('SELECT result_refs_json FROM pipeline_jobs WHERE pipeline_job_id=?', (job_id,)).fetchone()
        refs = json.loads(pipeline[0] or '{}') if pipeline else {}
        pdf = refs.get('pdf_extraction') or {}
        agenda = (refs.get('agenda') or {}).get('llm') or refs.get('agenda_progress') or {}
        summaries = (refs.get('summary_progress') or {}).get('summary_reviews') or {}
        # Whitelist numeric/boolean quality signals. Never export titles, quotes,
        # speaker names, free-form errors, questions or provider responses.
        data['quality'] = dict(
            processing_complete=refs.get('processing_complete') is True,
            pdf_complete=pdf.get('processing_complete') is True,
            pdf_entries=len(pdf.get('items') or []), pdf_tops=len(pdf.get('tops') or []),
            pdf_questions=len(pdf.get('review_questions') or []),
            agenda_complete=agenda.get('processing_complete') is True,
            agenda_review_complete=agenda.get('review_complete') is True,
            agenda_processed_lines=len(agenda.get('processed_lines') or []),
            agenda_gaps=len(agenda.get('gaps') or []),
            agenda_evidence_questions=sum(len((r.get('grounding') or {}).get('questions',[])) for r in agenda.get('line_results',[])),
            agenda_exactly_supported=sum((r.get('grounding') or {}).get('evidence_status') == 'exact' for r in agenda.get('line_results',[])),
            summary_results=len(summaries),
            summary_verified=sum((r.get('llm_usage') or {}).get('processing_complete') is True for r in summaries.values()),
            summary_questions=sum(len((r.get('structured') or {}).get('review_questions') or []) for r in summaries.values()),
            summary_evidence_questions=sum((r.get('llm_usage') or {}).get('open_evidence_questions',0) for r in summaries.values()),
            summary_repair_rounds=sum((r.get('llm_usage') or {}).get('reconciliation_rounds',0) for r in summaries.values()),
            summary_errors=sum(bool(r.get('error')) for r in summaries.values()),
        )
        phases = {}
        for row in db.execute('SELECT phase,metrics FROM durable_metrics WHERE job_id=?', (job_id,)):
            phase = phases.setdefault(row[0], dict(calls=0, prompt_tokens=0, generated_tokens=0, wall_seconds=0,
                                                  eval_duration_seconds=0, prompt_eval_duration_seconds=0, load_duration_seconds=0))
            phase['calls'] += 1
            metrics = json.loads(row[1])
            for key in phase.keys() - {'calls'}:
                phase[key] += metrics.get(key, 0)
        data['phases'] = phases
        return data


def live_snapshot(container, job_id):
    # All SQLite access must use the same Linux filesystem locks as the writer.
    # Opening a Docker Desktop bind-mounted live SQLite file with Windows SQLite
    # can incorrectly recover the Linux writer's journal and corrupt the database.
    code = ('import sqlite3,json,os,sys,hashlib\nfrom pathlib import Path\n' + inspect.getsource(snapshot) +
            '\nprint(json.dumps(snapshot(Path(os.environ["PERSISTENCE_DB_PATH"]), sys.argv[1])))')
    result = subprocess.run(['docker', 'exec', '-i', container, 'python', '-', job_id],
                            input=code, capture_output=True, text=True, timeout=15)
    if result.returncode:
        raise OSError('Container measurement unavailable')
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job_id')
    parser.add_argument('--container', default='gremienpilot-backend-1')
    parser.add_argument('--hours', type=float, default=12)
    parser.add_argument('--interval', type=float, default=30)
    args = parser.parse_args()
    output = Path('data/measurements')
    output.mkdir(parents=True, exist_ok=True)
    target = output / (args.job_id + '.jsonl')
    deadline = time.monotonic() + args.hours * 3600
    active_seen = False
    while time.monotonic() < deadline:
        sample = dict(timestamp=datetime.now(timezone.utc).isoformat())
        try:
            sample.update(live_snapshot(args.container, args.job_id))
        except (sqlite3.Error, OSError, TypeError, ValueError, subprocess.TimeoutExpired):
            sample['storage'] = 'unavailable'
        try:
            models = json.load(urllib.request.urlopen('http://localhost:11434/api/ps', timeout=5))['models']
            sample['models'] = [{k: m.get(k) for k in ('name', 'digest', 'size_vram', 'context_length')} for m in models]
        except (OSError, ValueError):
            sample['models'] = []
        sample['containers'] = command(['docker', 'stats', '--no-stream', '--format', '{{.Name}} {{.CPUPerc}} {{.MemUsage}}'])
        sample['gpu'] = command(['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw', '--format=csv,noheader,nounits'])
        with target.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(sample) + '\n')
        active_seen |= sample.get('state') in {'running', 'queued', 'retry_wait'}
        if active_seen and sample.get('state') in {'completed', 'review_required', 'failed', 'cancelled', 'superseded'}:
            break
        time.sleep(max(1, min(60, args.interval)))


if __name__ == '__main__':
    main()
