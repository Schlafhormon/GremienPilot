import copy
import json
import time

import pytest
import durable_jobs as jobs
import persistence
import resume_pipeline


@pytest.fixture
def interrupted(monkeypatch):
    monkeypatch.setenv('LLM_PROVIDER', 'ollama')
    monkeypatch.setenv('LLM_THINKING', 'false')
    monkeypatch.setenv('LLM_THINKING_TOKENS', '0')
    session = persistence.save_session('original', {'tops': [], 'transcript': []})
    with persistence.connect() as db:
        db.execute("""INSERT INTO pipeline_jobs
            (pipeline_job_id,session_id,status,stage,progress,created_at,updated_at,result_refs_json)
            VALUES ('job','original','processing','agenda_detect',72,?,?, '{}')""", (time.time(), time.time()))
    payload = dict(legacy_snapshot=persistence.load_pipeline_job('job'), session_snapshot=session,
                   session_revision=session['revision'])
    job = jobs.submit('pipeline', payload, 'job')
    with persistence.connect() as db:
        for key, value in [('pipeline:transcript', [dict(line_id='line-0', speaker='S', text='Original')]),
                           ('model-identities', {'model': {'digest': 'same'}}),
                           ('pdf:v2:hash:merged', {'items': []})]:
            db.execute('INSERT INTO durable_steps VALUES (?,?,?,?)', ('job', key, json.dumps(value), time.time()))
    return job


def test_resume_keeps_steps_and_records_previous_config(interrupted):
    before = jobs.load('job')['payload']['versions']
    report = resume_pipeline.resume('job', apply=True)
    assert report['transcript_lines'] == 1
    assert jobs.load('job')['state'] == 'queued'
    with persistence.connect() as db:
        steps = dict(db.execute('SELECT step_key,value FROM durable_steps WHERE job_id=?', ('job',)))
    assert 'pdf:v2:hash:merged' in steps and 'pipeline:transcript' in steps
    history = json.loads(next(v for k, v in steps.items() if k.startswith('operator:compact-resume:')))
    assert history['previous_versions'] == before


def test_changed_session_requires_fork_and_remains_untouched(interrupted):
    edited = persistence.save_session('original', {'tops': ['Manuell bearbeitet']})
    with pytest.raises(ValueError, match='Session changed'):
        resume_pipeline.resume('job', apply=True)
    report = resume_pipeline.resume('job', apply=True, fork_session=True)
    assert report['session_id'] != 'original'
    assert persistence.load_session('original') == edited
    fork = persistence.load_session(report['session_id'])
    assert fork['transcript'][0]['text'] == 'Original'
    assert jobs.load('job')['payload']['session_revision'] == fork['revision']


@pytest.mark.parametrize('section,key', [('model', 'model'), ('code', 'extract_tops.py'), ('policy', 'PDF_RENDER_DPI')])
def test_migration_rejects_unrelated_changes(interrupted, section, key):
    old = interrupted['payload']['versions']
    new = copy.deepcopy(old)
    new[section][key] = 'changed'
    with pytest.raises(ValueError, match='Unsupported changes'):
        resume_pipeline.validate_versions(old, new)


def test_cannot_reuse_already_published_agenda(interrupted):
    with persistence.connect() as db:
        db.execute('INSERT INTO durable_steps VALUES (?,?,?,?)', ('job', 'pipeline:agenda', '[]', time.time()))
    with pytest.raises(ValueError, match='Unsupported completed step'):
        resume_pipeline.resume('job', apply=True)


def test_dry_run_never_mutates(interrupted):
    before = jobs.load('job')
    resume_pipeline.resume('job')
    assert jobs.load('job') == before
