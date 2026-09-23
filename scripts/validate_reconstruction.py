"""Bounded local validation on a queued reconstruction fork, with the backend stopped.

Uses the production checkpoints and model configuration. Successful individual
calls can be reused by the worker; the probe never publishes a completed agenda.
Originals and responses stay in the Linux state volume. Stdout contains counts.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import threading
import time
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app' / 'backend'))
import durable_jobs as durable
import persistence
from agenda_llm import Workflow, obj, array, TEXT, EVIDENCE
from agenda_context import model_agenda
from agenda_detection import AgendaLLMUsage
from assignment_suggestions import TranscriptUtterance
from llm_transport import model_fingerprint, request_control
from llm_config import get_llm_config


def validate(job_id, output):
    if os.name == 'nt':
        raise RuntimeError('Run in Linux with the backend stopped')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with durable.ProcessLock():
        job = durable.load(job_id)
        pipeline = persistence.load_pipeline_job(job_id)
        if not job or job['state'] != 'queued' or job['kind'] != 'pipeline':
            raise ValueError('Expected queued reconstruction fork')
        if job['payload']['versions'] != durable.version_snapshot(job['payload']):
            raise ValueError('Code/configuration changed')
        with persistence.connect() as db:
            if db.execute("SELECT count(*) FROM durable_jobs WHERE state IN ('queued','running','retry_wait')").fetchone()[0] != 1:
                raise ValueError('Other work is active')
            records = list(db.execute('''SELECT s.step_key,s.value,i.sha256 FROM durable_steps s
                LEFT JOIN durable_step_integrity i ON i.job_id=s.job_id AND i.step_key=s.step_key
                WHERE s.job_id=?''', (job_id,)))
        if any(not sha or durable.hash_value(value) != sha for _, value, sha in records):
            raise ValueError('Checkpoint integrity mismatch')
        steps = {key: json.loads(value) for key, value, _ in records}
        history = steps['operator:source-contract-resume']
        if history['source_contract'] != 'bounded-reconstruction-v1':
            raise ValueError('Wrong migration contract')
        for document in job['documents']:
            if durable.document(document['path'])['sha256'] != document['sha256']:
                raise ValueError('Original input changed')
        options = pipeline['result_refs']['options']
        model = options.get('agenda_model') or options.get('model')
        config = get_llm_config(model)
        if urlparse(config.base_url).hostname not in {'ollama', 'localhost', '127.0.0.1', 'host.docker.internal'}:
            raise ValueError('Local model service required')
        if model_fingerprint(config) != steps['model-identities'][config.model]:
            raise ValueError('Model identity changed')
        transcript = [TranscriptUtterance(**{k: row.get(k) for k in ('speaker', 'text', 'line_id', 'start', 'end')})
                      for row in steps['pipeline:agenda-transcript:v1']]
        pdf = steps['pipeline:pdf:page-evidence-v3']
        agenda = model_agenda(pdf['tops'], [item['id'] for item in pdf['items'] if item['kind'] == 'agenda'])
        parent = persistence.load_pipeline_job(history['parent_job_id'])
        old_usage = parent['result_refs']['agenda']['llm']
        if agenda != [{k: v for k, v in item.items() if k not in {'top_index', 'top_uid'}}
                      for item in old_usage['provenance']['identities']]:
            raise ValueError('Probe expects unchanged verified PDF agenda')
        owner = 'reconstruction-validation'
        claimed = durable.claim(owner, 7200)
        if claimed['job_id'] != job_id:
            raise ValueError('Unexpected claimed job')
        runtime = durable.Runtime(claimed, owner, threading.Event())
        token = durable.CURRENT.set(runtime)
        began = time.monotonic()
        usage = AgendaLLMUsage(True, 'local_validation')
        try:
            with request_control(durable.check, durable.progress):
                work = Workflow(transcript, usage, model, options.get('agenda_system_prompt'), None,
                                options.get('agenda_cache_namespace', ''))
                if work.provenance != {k: old_usage['provenance'][k] for k in work.provenance}:
                    raise ValueError('Agenda source/model/prompt/configuration binding changed')
                # Read exact existing checkpoints. Any new context/discovery request
                # indicates a binding mismatch and must fail before inference.
                import agenda_llm
                real_complete = agenda_llm.complete
                def unexpected(*args, **kwargs):
                    raise RuntimeError('Expected unchanged context/discovery checkpoint')
                agenda_llm.complete = unexpected
                try:
                    contexts = [work.context(role, agenda) for role in ('primary', 'independent')]
                    inventories = [work.discover(role + ':discover', context, known_agenda=agenda)
                                   for role, context in zip(('primary', 'independent'), contexts)]
                    selected = inventories[0]
                    if inventories[0]['items'] != inventories[1]['items']:
                        selected = work.discover('resolve:discover', contexts[0], inventories, known_agenda=agenda)
                    if selected['items']:
                        raise ValueError('Additional agenda requires a different probe')
                finally:
                    agenda_llm.complete = real_complete
                reused = len(usage.chunks)
                result = work.reconstruction('independent:reconstruct', contexts[1], agenda)
                work.state_entries(result['agenda_states'], [item['top_id'] for item in agenda])
                durable.artifact('validation:reconstruction', 'reconstruction_probe', result)
                # Additional bounded content review of the actual cited originals.
                # This does not certify absence claims or replace either full reader.
                indices = {work.by_id[e['line_id']]['index'] for state in result['agenda_states'] for e in state['evidence']}
                indices = sorted({i for index in indices for i in range(max(0,index-1), min(len(work.rows),index+2))})
                schema = obj({'issues': array(obj({'kind': {'enum': ['unsupported', 'omission', 'contradiction', 'unclear']},
                    'top_id': {'enum': [item['top_id'] for item in agenda]}, 'question': TEXT, 'evidence': EVIDENCE}))})
                instruction = ('Prüfe die unbestätigten TOP-Statusangaben gegen die vorliegenden ORIGINALZEILEN. '
                    'Melde Widersprüche, ausgelassene Einschränkungen und nicht gestützte Behauptungen. '
                    'Prüfe heutige Beratung gegenüber Erwähnung/Rückblick, Wiederaufnahmen, gemeinsame TOPs und Verneinungen. '
                    'Eine fehlende Erwähnung in diesen Ausschnitten beweist keine Abwesenheit in der Sitzung. '
                    'Fehlende Gesamtprüfung offen lassen. Keine automatische Bestätigung durch gültige Quellen-IDs.')
                def check_review(data):
                    for issue in data['issues']:
                        if issue['kind'] not in ('unsupported', 'omission', 'contradiction', 'unclear'):
                            raise ValueError('Invalid issue kind')
                        if issue['top_id'] not in {item['top_id'] for item in agenda}:
                            raise ValueError('Invalid issue identity')
                        work.text(issue['question'])
                        work.evidence(issue['evidence'], required=False)
                body = dict(agenda=agenda, candidate_states=result['agenda_states'],
                            originals=[work.rows[i] for i in indices])
                review = work.call('validation:cited_originals', instruction, body, schema, check_review)
                durable.artifact('validation:reconstruction', 'original_review', review)
            with persistence.connect() as db:
                metrics = [json.loads(row[0]) for row in db.execute('SELECT metrics FROM durable_metrics WHERE job_id=?', (job_id,))]
            states = result['agenda_states']
            with persistence.connect() as db:
                primary = None
                for row in db.execute("SELECT step_key,value,sha256 FROM durable_artifacts WHERE job_id=? AND kind='model_attempt'", (history['parent_job_id'],)):
                    if json.loads(row[1]).get('phase') != 'primary:reconstruct':
                        continue
                    saved = db.execute('''SELECT s.value,i.sha256 FROM durable_steps s JOIN durable_step_integrity i
                        ON i.job_id=s.job_id AND i.step_key=s.step_key WHERE s.job_id=? AND s.step_key=?''',
                        (history['parent_job_id'], row[0])).fetchone()
                    if saved and durable.hash_value(saved[0]) == saved[1]:
                        primary = {s['top_id']: s['status'] for s in json.loads(saved[0])['agenda_states']}
            report = dict(job_id=job_id, session_id=pipeline['session_id'], model=config.model,
                digest=work.provenance['digest'], configuration=config.public_snapshot(),
                expected_states=len(agenda), actual_states=len(states), unique_states=len({s['top_id'] for s in states}),
                episodes=len(result['episodes']), reused_calls=reused, new_calls=usage.attempted_calls,
                failed_calls=usage.failed_calls,
                status_counts=dict(Counter(s['status'] for s in states)),
                reference_status_counts=dict(Counter(s['grounding']['reference_status'] for s in states)),
                content_status_counts=dict(Counter(s['grounding']['content_status'] for s in states)),
                cited_originals_reviewed=len(indices), review_issue_counts=dict(Counter(i['kind'] for i in review['issues'])),
                primary_status_disagreements=sum(primary.get(s['top_id']) != s['status'] for s in states) if primary else None,
                open_content_checks=len(states), duration_seconds=round(time.monotonic()-began, 2),
                prompt_tokens=sum(m.get('prompt_tokens', 0) for m in metrics),
                generated_tokens=sum(m.get('generated_tokens', 0) for m in metrics),
                end_to_end_verified=False)
            with persistence.connect() as db:
                report['repairs'] = db.execute("SELECT count(*) FROM durable_artifacts WHERE job_id=? AND kind='technical_diagnostic'", (job_id,)).fetchone()[0]
                durable.fence(db)
                db.execute("UPDATE durable_jobs SET state='queued',owner=NULL,lease_until=NULL,progress=?,updated_at=? WHERE job_id=?",
                           (json.dumps({'phase': 'validated_reconstruction_resume'}), time.time(), job_id))
            (output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(json.dumps(report), flush=True)
        except BaseException as exc:
            with persistence.connect() as db:
                db.execute("UPDATE durable_jobs SET state='failed',error=?,owner=NULL,lease_until=NULL,updated_at=? WHERE job_id=? AND owner=?",
                           ('Local validation: ' + type(exc).__name__, time.time(), job_id, owner))
            print(json.dumps({'validation_failed': type(exc).__name__, 'job_id': job_id}), flush=True)
            raise
        finally:
            durable.CURRENT.reset(token)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job_id')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    validate(args.job_id, args.output)
