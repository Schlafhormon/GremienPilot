"""Explicit migration of interrupted agenda work without repeating transcription.

Only supports an unfinished PDF/agenda stage and narrowly scoped configuration
changes. Run while the backend is stopped. The old configuration and checkpoints
remain recorded; ordinary worker version checks are never disabled. With
--pdf-contract, the local model service must be reachable for digest verification.
"""
import argparse
import json
import sqlite3
import time
import uuid
from copy import deepcopy
from pathlib import Path

import durable_jobs as durable
import persistence


def recover_fast_draft(job_id, *, apply=False):
    """Fork a failed Fast agenda run, retaining accepted audio and raw PDF work."""
    from llm_config import get_llm_config
    from llm_transport import model_fingerprint
    from processing_mode import processing_scope

    with durable.ProcessLock():
        job = durable.load(job_id)
        pipeline = persistence.load_pipeline_job(job_id)
        if (not job or not pipeline or job['kind'] != 'pipeline' or job['state'] != 'failed'
                or pipeline['stage'] != 'agenda_detect'
                or pipeline['result_refs'].get('options', {}).get('processing_mode') != 'fast'):
            raise ValueError('Expected a failed Fast agenda pipeline')
        previous = job['payload']['versions']
        current = durable.version_snapshot(job['payload'])
        if any(previous.get(section) != current.get(section) for section in previous if section != 'code'):
            raise ValueError('Model, transcription or policy configuration changed')
        changed = {name for name in set(previous['code']) | set(current['code'])
                   if previous['code'].get(name) != current['code'].get(name)}
        if changed - {'main.py', 'agenda_llm.py', 'extract_tops.py'}:
            raise ValueError('Unrelated code changed: ' + ','.join(sorted(changed - {'main.py', 'agenda_llm.py', 'extract_tops.py'})))
        with persistence.connect() as db:
            if db.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
                raise ValueError('Database integrity check failed')
            if db.execute("SELECT 1 FROM durable_jobs WHERE state IN ('queued','running','retry_wait')").fetchone():
                raise ValueError('Other work remains active')
            records = db.execute('''SELECT s.step_key,s.value,i.sha256,s.completed_at FROM durable_steps s
                LEFT JOIN durable_step_integrity i ON i.job_id=s.job_id AND i.step_key=s.step_key
                WHERE s.job_id=?''', (job_id,)).fetchall()
        steps = {r[0]: r[1] for r in records}
        retained = {key: value for key, value in steps.items()
                    if key in {'model-identities', 'pipeline:transcript', 'pipeline:agenda-transcript:v1'}
                    or key.startswith(('model:', 'pdf:fast:v1:'))}
        if not {'model-identities', 'pipeline:transcript', 'pipeline:agenda-transcript:v1'} <= retained.keys():
            raise ValueError('Accepted transcript or model checkpoint missing')
        for key, value, digest, _ in records:
            if key in retained and (not digest or digest != durable.hash_value(value)):
                raise ValueError('Checkpoint integrity mismatch: ' + key)
        transcript = json.loads(retained['pipeline:transcript'])
        transcription = persistence.load_job(pipeline['transcription_job_id'])
        if (not transcription or transcription['status'] != 'completed' or
                transcription['transcript'] != [{k: v for k, v in row.items() if k != 'line_id'} for row in transcript]):
            raise ValueError('Accepted transcript differs from completed transcription')
        with processing_scope('fast'):
            for name, identity in json.loads(retained['model-identities']).items():
                if not identity.get('digest') or model_fingerprint(get_llm_config(name)) != identity:
                    raise ValueError('Model identity changed')
        for document in job['documents']:
            if durable.document(document['path'])['sha256'] != document['sha256']:
                raise ValueError('Original document changed')
        child_id = str(uuid.uuid5(uuid.UUID(job_id), 'recover-fast-draft:' + durable.hash_value(json.dumps(current, sort_keys=True))))
        existing = durable.load(child_id)
        if existing:
            return dict(job_id=child_id, session_id=existing['payload']['legacy_snapshot']['session_id'], already_created=True)
        report = dict(parent_job_id=job_id, job_id=child_id, transcript_lines=len(transcript),
                      retained_steps=len(retained), changed_code=sorted(changed), applied=apply)
        if not apply:
            return report
        backup = persistence.get_db_path().parent / 'backups' / ('fast-draft-resume-' + child_id + '.sqlite3')
        backup.parent.mkdir(exist_ok=True)
        with persistence.connect() as source, sqlite3.connect(backup) as target:
            source.backup(target, pages=128, sleep=.02)
            if target.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                raise ValueError('Backup integrity failed')
        session_id = str(uuid.uuid5(uuid.UUID(child_id), 'session'))
        if persistence.load_session(session_id):
            raise ValueError('Recovery session already exists')
        snapshot = deepcopy(job['payload'].get('session_snapshot') or {})
        snapshot.update(transcript=transcript, current_step=1, job_id=pipeline['transcription_job_id'],
                        tops=[], top_ids=[], assignments=[], summaries={}, summary_reviews={},
                        summary_states={}, agenda_proposals=None)
        fork = persistence.save_session(session_id, snapshot)
        refs = {key: deepcopy(pipeline['result_refs'][key]) for key in
                ('audio_path', 'pdf_path', 'known_tops', 'options', 'remember_speakers')
                if key in pipeline['result_refs']}
        refs.update(parent_pipeline_id=job_id, processing_complete=False, publication_status='pending')
        now = time.time()
        new_pipeline = dict(pipeline, pipeline_job_id=child_id, session_id=session_id,
                            status='pending', stage='agenda_detect', progress=72, error=None,
                            result_refs=refs, created_at=now, updated_at=now)
        payload = dict(job['payload'], versions=current, session_snapshot=fork,
                       session_revision=fork['revision'], legacy_snapshot=new_pipeline)
        history = dict(parent_job_id=job_id, parent_session_id=pipeline['session_id'],
                       previous_versions=previous, current_versions=current,
                       retained_hashes={key: durable.hash_value(value) for key, value in retained.items()},
                       backup=str(backup))
        with persistence.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute('''INSERT INTO pipeline_jobs (pipeline_job_id,session_id,transcription_job_id,status,stage,
                progress,error,result_refs_json,created_at,updated_at) VALUES (?,?,?,'pending','agenda_detect',72,NULL,?,?,?)''',
                (child_id, session_id, pipeline['transcription_job_id'], json.dumps(refs), now, now))
            db.execute("INSERT INTO durable_jobs (job_id,kind,state,payload,documents,created_at,updated_at) VALUES (?,'pipeline','queued',?,?,?,?)",
                       (child_id, json.dumps(payload), json.dumps(job['documents']), now, now))
            for key, value in {**retained, 'operator:fast-draft-resume': json.dumps(history)}.items():
                db.execute('INSERT INTO durable_steps VALUES (?,?,?,?)', (child_id, key, value, now))
                db.execute('INSERT INTO durable_step_integrity VALUES (?,?,?)',
                           (child_id, key, durable.hash_value(value)))
        return {**report, 'session_id': session_id, 'backup': str(backup)}


def resume_sources(job_id, *, apply=False, reconstruction=False):
    """Fork a failed graded-source continuation; the historical job stays intact.

    Run under the same Linux volume with the backend stopped. Every reusable
    source, model, PDF audit and integrity binding is checked before enqueueing.
    The deterministic child ID makes operator retries idempotent.
    """
    from llm_config import get_llm_config
    from llm_transport import model_fingerprint
    import extract_tops as pdf
    with durable.ProcessLock():
        persistence.init_db()
        job = durable.load(job_id)
        pipeline = persistence.load_pipeline_job(job_id)
        if not job or not pipeline or job['kind'] != 'pipeline' or job['state'] != 'failed':
            raise ValueError('Expected failed pipeline')
        versions = durable.version_snapshot(job['payload'])
        previous = job['payload']['versions']
        session = persistence.load_session(pipeline['session_id'])
        if reconstruction and (pipeline['stage'] != 'agenda_detect' or not session):
            raise ValueError('Expected unfinished agenda stage with retained session')
        for section in set(previous) | set(versions):
            if section != 'code' and previous.get(section) != versions.get(section):
                raise ValueError('Source migration does not permit configuration changes: ' + section)
        allowed = {'agenda_llm.py', 'summary_grounding.py', 'summarize.py', 'main.py', 'durable_jobs.py', 'source_contract.py'}
        if reconstruction:
            # Reviewed baseline: only reconstruction requests and transport failure
            # diagnostics changed. Reused calls still require identical cache keys
            # (full source/model/configuration/prompt/schema) and current validation.
            baseline = {
                'agenda_llm.py': '0d203fb0373bb825d46993b0bddd3353edc48fcdb7a52e5c381a83dbec109d93',
                'llm_transport.py': '76d906f5e47d6838ceaf0c94c489662f4c4924e526f83aafee626c13b9418195',
            }
            allowed = set(baseline)
            for name, digest in baseline.items():
                if previous['code'].get(name) != digest:
                    raise ValueError('Unsupported reconstruction baseline: ' + name)
        changed = {name for name in set(previous['code']) | set(versions['code'])
                   if previous['code'].get(name) != versions['code'].get(name)}
        if changed - allowed:
            raise ValueError('Unrelated code changed: ' + ','.join(sorted(changed - allowed)))
        contract = 'bounded-reconstruction-v1' if reconstruction else 'graded-sources-v1'
        child_id = str(uuid.uuid5(uuid.UUID(job_id), contract + ':' + durable.hash_value(json.dumps(versions,sort_keys=True))))
        existing = durable.load(child_id)
        if existing:
            return dict(job_id=child_id, session_id=existing['payload']['legacy_snapshot']['session_id'], already_created=True)
        for doc in job['documents']:
            if durable.document(doc['path'])['sha256'] != doc['sha256']:
                raise ValueError('Original source changed')
        with persistence.connect() as db:
            if db.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
                raise ValueError('Database integrity check failed')
            if db.execute("select 1 from durable_jobs where state in ('queued','running','retry_wait')").fetchone():
                raise ValueError('Other work remains active')
            records = db.execute('''SELECT s.step_key,s.value,i.sha256,s.completed_at FROM durable_steps s
                LEFT JOIN durable_step_integrity i ON i.job_id=s.job_id AND i.step_key=s.step_key WHERE s.job_id=?''', (job_id,)).fetchall()
            attempts = db.execute("SELECT step_key,value,sha256 FROM durable_artifacts WHERE job_id=? AND kind='model_attempt'", (job_id,)).fetchall()
        steps = {r[0]: r[1] for r in records}
        retained = {k: v for k, v in steps.items() if k in {'model-identities', 'pipeline:transcript',
            'pipeline:agenda-transcript:v1'} or k.startswith(('model:', 'pdf:v2:', 'pdf:page-evidence-v3:'))}
        reused_agenda = set()
        if reconstruction:
            if any(k in steps for k in ('pipeline:agenda', 'pipeline:summaries', 'pipeline:published')):
                raise ValueError('Reconstruction migration cannot reuse completed downstream work')
            for key, value, sha in attempts:
                phase = json.loads(value).get('phase', '')
                if phase in {'primary:context', 'independent:context', 'primary:discover',
                             'independent:discover', 'resolve:discover'} and key in steps:
                    if not sha or sha != durable.hash_value(value):
                        raise ValueError('Agenda attempt integrity mismatch')
                    reused_agenda.add(key)
                    retained[key] = steps[key]
                    cache = 'cache:' + key.removeprefix('agenda:')
                    if cache in steps:
                        if json.loads(steps[cache]) != json.loads(steps[key]):
                            raise ValueError('Agenda cache differs from checkpoint')
                        retained[cache] = steps[cache]
        completed_at = {r[0]:r[3] for r in records}
        for key, value, sha, _ in records:
            if key in retained and (not sha or sha != durable.hash_value(value)):
                raise ValueError('Checkpoint integrity binding missing or changed')
        identities = json.loads(steps['model-identities'])
        for name, identity in identities.items():
            if not identity.get('digest') or model_fingerprint(get_llm_config(name)) != identity:
                raise ValueError('Model identity changed')
        transcript = json.loads(steps['pipeline:transcript'])
        transcription = persistence.load_job(pipeline['transcription_job_id'])
        if not transcription or transcription['status'] != 'completed' or transcription['transcript'] != [
                {k:v for k,v in row.items() if k != 'line_id'} for row in transcript]:
            raise ValueError('Accepted transcript differs from completed audio checkpoint')
        audio = pipeline['result_refs'].get('audio_path')
        retained_audio = durable.document(audio)
        if not any(d['sha256'] == retained_audio['sha256'] and Path(d['path']).stat().st_size == Path(audio).stat().st_size
                   for d in job['documents']):
            raise ValueError('Retained audio lacks input hash binding')
        documents = deepcopy(job['documents'])
        if not any(d['path'] == audio for d in documents):
            documents.append(retained_audio)
        pdf_path = pipeline['result_refs'].get('pdf_path')
        document = durable.document(pdf_path)
        prefix = 'pdf:v2:' + document['sha256']
        manifest = json.loads(steps[prefix + ':manifest'])
        pages = [json.loads(steps[f'{prefix}:page:{i}:render']) for i in range(1, manifest['page_count']+1)]
        candidate = pdf._validate(json.loads(steps[prefix + ':merged']), list(range(1,len(pages)+1)))
        for number, originals, projection, prompt in [
            *[(p['page'], [p], pdf.page_projection(candidate,p['page']), pdf.PAGE_AUDIT_PROMPT) for p in pages],
            (0,pages,candidate,pdf.RELATION_AUDIT_PROMPT)]:
            key = 'pdf:page-evidence-v3:' + document['sha256'] + ':audit:' + pdf._digest([
                number,projection,[p['image_sha256'] for p in originals],prompt])
            audit = json.loads(steps[key])
            pdf.validate_audit(audit,projection,[p['page'] for p in originals],number)
            if audit['issues']:
                raise ValueError('PDF audit still requires review')
        result_key = 'pdf:page-evidence-v3:' + document['sha256'] + ':result:' + pdf._digest(candidate)
        result = json.loads(steps[result_key])
        if result.get('contract_version') != 'page-evidence-v3' or not result['processing_complete'] or result['review_required']:
            raise ValueError('PDF verification incomplete')
        retained['pipeline:pdf:page-evidence-v3'] = steps[result_key]
        completed_at['pipeline:pdf:page-evidence-v3'] = completed_at[result_key]
        report = dict(parent_job_id=job_id, job_id=child_id, transcript_lines=len(transcript),
            agenda_lines=len(json.loads(steps['pipeline:agenda-transcript:v1'])),
            retained_steps=len(retained), discarded_success_markers=[k for k in steps if k not in retained],
            reused_agenda_calls=len(reused_agenda), parent_session_revision=session.get('revision') if session else None,
            changed_code=sorted(changed), applied=apply)
        if not apply:
            return report
        backup = persistence.get_db_path().parent/'backups'/('graded-resume-'+str(uuid.uuid4())+'.sqlite3')
        backup.parent.mkdir(exist_ok=True)
        with persistence.connect() as source, sqlite3.connect(backup) as target:
            source.backup(target,pages=128,sleep=.02)
            if target.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                raise ValueError('Backup integrity failed')
        session_id = str(uuid.uuid5(uuid.UUID(child_id), 'session'))
        if persistence.load_session(session_id) is not None:
            # A failed operator transaction may have left a reviewable fork.
            # Never overwrite it (including edits made after that interruption).
            session_id = str(uuid.uuid4())
        snapshot = deepcopy(job['payload'].get('session_snapshot') or {})
        snapshot.update(transcript=transcript,current_step=1,job_id=pipeline['transcription_job_id'],
                        tops=[],top_ids=[],assignments=[],summaries={},summary_reviews={},summary_states={},agenda_proposals=None)
        fork = persistence.save_session(session_id,snapshot)
        refs = {k:deepcopy(pipeline['result_refs'][k]) for k in ('audio_path','pdf_path','known_tops','options','remember_speakers')
                if k in pipeline['result_refs']}
        refs.update(parent_pipeline_id=job_id,processing_complete=False,publication_status='pending')
        new_pipeline = dict(pipeline, pipeline_job_id=child_id, session_id=session_id, status='pending',
            stage='agenda_detect',progress=72,error=None,result_refs=refs,created_at=time.time(),updated_at=time.time())
        payload = dict(job['payload'],versions=versions,session_snapshot=fork,session_revision=fork['revision'],
                       legacy_snapshot=new_pipeline)
        history = dict(parent_job_id=job_id,parent_session_id=pipeline['session_id'],source_contract=contract,
            previous_versions=previous,current_versions=versions,created_at=time.time(),
            retained_hashes={k:durable.hash_value(v) for k,v in retained.items()},
            retained_completed_at={k:completed_at[k] for k in retained},
            retained_audio_binding=dict(path=audio,sha256=retained_audio['sha256'],
                original_paths=[d['path'] for d in job['documents'] if d['sha256'] == retained_audio['sha256']]),
            archived_in_parent=[k for k in steps if k not in retained],backup=str(backup))
        # Child queue, pipeline row and copied checkpoints become visible atomically.
        with persistence.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if reconstruction and persistence.load_session(pipeline['session_id'])['revision'] != session['revision']:
                raise ValueError('Parent session changed during migration')
            now=time.time()
            db.execute('''INSERT INTO pipeline_jobs (pipeline_job_id,session_id,transcription_job_id,status,stage,
                progress,error,result_refs_json,created_at,updated_at) VALUES (?,?,?,'pending','agenda_detect',72,NULL,?,?,?)''',
                (child_id,session_id,pipeline['transcription_job_id'],json.dumps(refs),now,now))
            db.execute("INSERT INTO durable_jobs (job_id,kind,state,payload,documents,created_at,updated_at) VALUES (?,'pipeline','queued',?,?,?,?)",
                (child_id,json.dumps(payload),json.dumps(documents),now,now))
            for key,value in {**retained,'operator:source-contract-resume':json.dumps(history)}.items():
                db.execute('INSERT INTO durable_steps VALUES (?,?,?,?)',(child_id,key,value,completed_at.get(key,now)))
                db.execute('INSERT INTO durable_step_integrity VALUES (?,?,?)',(child_id,key,durable.hash_value(value)))
        return {**report,'session_id':session_id,'backup':str(backup)}


def validate_versions(old, new, *, pdf_contract=False):
    for section in set(old) | set(new):
        if section not in {'model', 'overrides', 'policy', 'code'} and old.get(section) != new.get(section) and not (
                pdf_contract and section == 'transcription' and section not in old):
            raise ValueError('Unsupported version change')
    allowed = {
        'model': {'thinking', 'thinking_tokens', 'config_id'},
        'policy': {'AGENDA_COMPACT_ASSIGNMENTS', 'AGENDA_OUTPUT_TOKENS_PER_LINE', 'AGENDA_DETECTION_CHUNK_LINES'},
        'code': {'agenda_llm.py', 'durable_jobs.py', 'main.py'},
    }
    if pdf_contract:
        allowed['code'].update({'extract_tops.py', 'llm_transport.py', 'summary_grounding.py', 'persistence.py'})
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
    config = new['model']
    disabled = (config.get('thinking') is False or
                (config.get('thinking') is None and config.get('reasoning_effort') == 'none'))
    if not disabled or config['thinking_tokens'] != 0:
        raise ValueError('This migration requires thinking disabled')
    return changes


def resume(job_id, *, apply=False, fork_session=False, pdf_contract=False):
    with durable.ProcessLock():
        persistence.init_db()
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
        changes = validate_versions(job['payload']['versions'], versions, pdf_contract=pdf_contract)
        with persistence.connect() as db:
            steps = dict(db.execute('SELECT step_key,value FROM durable_steps WHERE job_id=?', (job_id,)))
        for key in steps:
            if not (key in {'model-identities', 'pipeline:transcript', 'pipeline:agenda-transcript:v1'}
                    or key.startswith(('model:', 'pdf:v2:', 'operator:compact-resume:', 'operator:pdf-contract-resume:'))):
                raise ValueError('Unsupported completed step: ' + key)
        transcript = json.loads(steps.get('pipeline:transcript', 'null'))
        if not isinstance(transcript, list) or not transcript:
            raise ValueError('A completed transcript checkpoint is required')
        if 'model-identities' not in steps:
            raise ValueError('Immutable model identity is required')
        audio_document = None
        identities = json.loads(steps['model-identities'])
        if pdf_contract:
            from llm_config import get_llm_config
            from llm_transport import model_fingerprint
            for name, identity in identities.items():
                current = model_fingerprint(get_llm_config(name))
                if not identity.get('digest') or identity != current:
                    raise ValueError('Model digest changed; migration refused')
            audio = pipeline['result_refs'].get('audio_path')
            if not audio or not Path(audio).is_file():
                raise ValueError('Retained audio is required for provenance')
            audio_document = durable.document(audio)
            # Historical jobs did not bind audio hashes. Do not pretend otherwise:
            # anchor the retained source now and keep the accepted transcript intact.
            transcription = persistence.load_job(pipeline['transcription_job_id'])
            saved_lines = [{k: v for k, v in line.items() if k != 'line_id'} for line in transcript]
            if not transcription or transcription['status'] != 'completed' or transcription.get('transcript') != saved_lines:
                raise ValueError('Transcript checkpoint differs from completed transcription')
        report = dict(job_id=job_id, transcript_lines=len(transcript), retained_steps=len(steps), changes=changes,
                      fork_session=fork_session, applied=apply)
        if not apply:
            return report
        backup = persistence.get_db_path().parent / 'backups' / ('pdf-resume-' + str(uuid.uuid4()) + '.sqlite3')
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
                       reason=('PDF page-evidence-v3: old inventories/merge are drafts; all audits rerun with new contract.' if pdf_contract else
                               'Operator requested compact assignments and disabled thinking; completed PDF steps retained under original provenance.'),
                       checkpoint_hashes={k: durable.hash_value(v) for k, v in steps.items()},
                       audio_binding=('retained source hash verified now; historical input hash unavailable' if audio_document else None))
        with persistence.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if fork_session:
                # Keep the failed phase visible on the previous session after the
                # live job moves to its new target. This row never enters the queue.
                archived = dict(db.execute('SELECT * FROM pipeline_jobs WHERE pipeline_job_id=?', (job_id,)).fetchone())
                archived_id = str(uuid.uuid4())
                archived.update(pipeline_job_id=archived_id, status='failed',
                                error=archived.get('error') or 'Unterbrochener Auftrag wird in einer neuen Sitzung fortgesetzt')
                archived_refs = json.loads(archived['result_refs_json'] or '{}')
                archived['result_refs_json'] = json.dumps({**archived_refs, 'archived_job_id': job_id,
                                                          'resumed_session_id': session_id})
                columns = list(archived)
                db.execute('INSERT INTO pipeline_jobs (' + ','.join(columns) + ') VALUES (' +
                           ','.join('?' for _ in columns) + ')', [archived[k] for k in columns])
                history['archived_pipeline_id'] = archived_id
            db.execute('INSERT INTO durable_steps VALUES (?,?,?,?)',
                       (job_id, ('operator:pdf-contract-resume:' if pdf_contract else 'operator:compact-resume:') + str(uuid.uuid4()), json.dumps(history), now))
            documents = list(job['documents'] or [])
            if audio_document and audio_document['path'] not in {d['path'] for d in documents}:
                documents.append(audio_document)
            db.execute("""UPDATE durable_jobs SET payload=?,documents=?,state='queued',owner=NULL,lease_until=NULL,
                heartbeat_at=NULL,attempt=0,available_at=0,error=NULL,result=NULL,progress=?,updated_at=? WHERE job_id=?""",
                       (json.dumps(payload), json.dumps(documents), json.dumps({'phase': 'resume_from_checkpoint'}), now, job_id))
            for key, value in steps.items():
                db.execute('INSERT OR IGNORE INTO durable_step_integrity VALUES (?,?,?)', (job_id, key, durable.hash_value(value)))
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
    parser.add_argument('--pdf-contract', action='store_true', help='Migrate unfinished PDF checks to page-evidence-v3; verify local model digest and retained audio')
    parser.add_argument('--source-contract', action='store_true', help='Fork failed work with graded sources; retain verified PDF and accepted audio')
    parser.add_argument('--reconstruction', action='store_true', help='Fork the verified graded-source baseline; reuse unchanged context/discovery calls')
    parser.add_argument('--recover-fast-draft', action='store_true', help='Fork failed Fast agenda work; reuse accepted audio and raw PDF checkpoints')
    args = parser.parse_args()
    result = recover_fast_draft(args.job_id,apply=args.apply) if args.recover_fast_draft else resume_sources(args.job_id,apply=args.apply,reconstruction=args.reconstruction) if args.source_contract or args.reconstruction else resume(
        args.job_id, apply=args.apply, fork_session=args.fork_session, pdf_contract=args.pdf_contract)
    print(json.dumps(result, ensure_ascii=False, indent=2))
