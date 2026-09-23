"""Explicit, offline migration to compact assignments without repeating completed work.

Only supports an unfinished PDF/agenda stage and narrowly scoped configuration
changes. Run while the backend is stopped. The old configuration and checkpoints
remain recorded; ordinary worker version checks are never disabled.
"""
import argparse
import json
import sqlite3
import time
import uuid
from pathlib import Path

import durable_jobs as durable
import persistence


def validate_versions(old, new):
    for section in set(old) | set(new):
        if section not in {'model', 'overrides', 'policy', 'code'} and old.get(section) != new.get(section):
            raise ValueError('Unsupported version change')
    allowed = {
        'model': {'thinking', 'thinking_tokens', 'config_id'},
        'policy': {'AGENDA_COMPACT_ASSIGNMENTS', 'AGENDA_OUTPUT_TOKENS_PER_LINE', 'AGENDA_DETECTION_CHUNK_LINES'},
        'code': {'agenda_llm.py', 'durable_jobs.py', 'main.py'},
    }
    changes = {}
    def compare(section, before, after, permitted):
        changed = {k for k in set(before) | set(after) if before.get(k) != after.get(k)}
        if changed - permitted:
            raise ValueError('Unsupported changes in ' + section + ': ' + ', '.join(sorted(changed - permitted)))
        changes[section] = sorted(changed)
    for section, permitted in allowed.items():
        compare(section, old.get(section, {}), new.get(section, {}), permitted)
    if set(old.get('overrides', {})) != set(new.get('overrides', {})):
        raise ValueError('Model overrides changed')
    for name, before in old.get('overrides', {}).items():
        compare('override:' + name, before, new['overrides'][name], allowed['model'])
    if new['model']['thinking'] is not False or new['model']['thinking_tokens'] != 0:
        raise ValueError('This migration requires thinking disabled')
    return changes


def resume(job_id, *, apply=False, fork_session=False):
    with durable.ProcessLock():
        job = durable.load(job_id)
        if not job or job['kind'] != 'pipeline' or job['state'] not in {'queued', 'running', 'retry_wait', 'failed'}:
            raise ValueError('Expected an interrupted pipeline, not a cancelled or completed job')
        pipeline = persistence.load_pipeline_job(job_id)
        if not pipeline or pipeline['stage'] != 'agenda_detect':
            raise ValueError('Only the unfinished agenda stage is supported')
        session = persistence.load_session(pipeline['session_id'])
        changed_session = not session or session.get('revision') != job['payload'].get('session_revision')
        if changed_session and not fork_session:
            raise ValueError('Session changed since submission')
        for doc in job['documents'] or []:
            if durable.document(doc['path'])['sha256'] != doc['sha256']:
                raise ValueError('Source document changed')
        versions = durable.version_snapshot(job['payload'])
        changes = validate_versions(job['payload']['versions'], versions)
        with persistence.connect() as db:
            steps = dict(db.execute('SELECT step_key,value FROM durable_steps WHERE job_id=?', (job_id,)))
        for key in steps:
            if not (key in {'model-identities', 'pipeline:transcript', 'pipeline:agenda-transcript:v1'}
                    or key.startswith(('model:', 'pdf:v2:', 'operator:compact-resume:'))):
                raise ValueError('Unsupported completed step: ' + key)
        transcript = json.loads(steps.get('pipeline:transcript', 'null'))
        if not isinstance(transcript, list) or not transcript:
            raise ValueError('A completed transcript checkpoint is required')
        if 'model-identities' not in steps:
            raise ValueError('Immutable model identity is required')
        report = dict(job_id=job_id, transcript_lines=len(transcript), retained_steps=len(steps), changes=changes,
                      fork_session=fork_session, applied=apply)
        if not apply:
            return report
        backup = persistence.get_db_path().parent / 'backups' / ('compact-resume-' + str(uuid.uuid4()) + '.sqlite3')
        backup.parent.mkdir(exist_ok=True)
        with persistence.connect() as source, sqlite3.connect(backup) as dest:
            source.backup(dest)
        now = time.time()
        payload = dict(job['payload'], versions=versions)
        session_id = pipeline['session_id']
        if fork_session:
            session_id = str(uuid.uuid4())
            initial = dict(payload.get('session_snapshot') or {})
            initial.update(transcript=transcript, current_step=1, job_id=pipeline['transcription_job_id'])
            fork = persistence.save_session(session_id, initial)
            payload.update(session_snapshot=fork, session_revision=fork['revision'],
                           legacy_snapshot=dict(payload['legacy_snapshot'], session_id=session_id))
        history = dict(previous_versions=job['payload']['versions'], current_versions=versions,
                       retained_steps=list(steps), previous_attempt=job['attempt'], applied_at=now,
                       previous_session_id=pipeline['session_id'], current_session_id=session_id,
                       reason='Operator requested compact assignments and disabled thinking; completed PDF steps retained under original provenance.')
        with persistence.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute('INSERT INTO durable_steps VALUES (?,?,?,?)',
                       (job_id, 'operator:compact-resume:' + str(uuid.uuid4()), json.dumps(history), now))
            db.execute("""UPDATE durable_jobs SET payload=?,state='queued',owner=NULL,lease_until=NULL,
                heartbeat_at=NULL,attempt=0,available_at=0,error=NULL,result=NULL,progress=?,updated_at=? WHERE job_id=?""",
                       (json.dumps(payload), json.dumps({'phase': 'resume_from_checkpoint'}), now, job_id))
            refs = dict(pipeline['result_refs'], execution_state='queued')
            refs.pop('cancel_requested', None)
            db.execute("""UPDATE pipeline_jobs SET status='pending',error=NULL,result_refs_json=?,updated_at=?,session_id=?
                          WHERE pipeline_job_id=?""", (json.dumps(refs), now, session_id, job_id))
        return {**report, 'backup': str(backup), 'session_id': session_id}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job_id')
    parser.add_argument('--apply', action='store_true', help='Apply after backup; default validates only')
    parser.add_argument('--fork-session', action='store_true', help='Continue in a new session, preserving any edits to the old session')
    args = parser.parse_args()
    print(json.dumps(resume(args.job_id, apply=args.apply, fork_session=args.fork_session), ensure_ascii=False, indent=2))
