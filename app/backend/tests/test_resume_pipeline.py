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


def test_pdf_contract_migration_requires_unchanged_model_and_transcription(interrupted):
    old = interrupted['payload']['versions']
    changed = copy.deepcopy(old)
    changed['code']['extract_tops.py'] = 'new contract'
    assert 'extract_tops.py' in resume_pipeline.validate_versions(old, changed, pdf_contract=True)['code']
    changed['model']['model'] = 'other model'
    with pytest.raises(ValueError, match='Unsupported changes'):
        resume_pipeline.validate_versions(old, changed, pdf_contract=True)

    changed = copy.deepcopy(old)
    changed['transcription']['policy']['WHISPER_MODEL'] = 'other whisper'
    with pytest.raises(ValueError, match='Unsupported version'):
        resume_pipeline.validate_versions(old, changed, pdf_contract=True)


def test_pdf_migration_binds_audio_preserves_edits_and_keeps_old_audits_as_history(interrupted, tmp_path, monkeypatch):
    import llm_transport
    audio = tmp_path / 'synthetic.wav'
    audio.write_bytes(b'synthetic audio')
    transcription = [dict(speaker='S', text='Original', start=0, end=1)]
    persistence.save_job('audio-job', dict(status='completed', transcript=transcription, file_path=str(audio)))
    # Use exactly the committed transcription representation, plus its stable ID.
    transcript = [dict(line_id='line-0', **r) for r in persistence.load_job('audio-job')['transcript']]
    with persistence.connect() as db:
        db.execute("UPDATE pipeline_jobs SET transcription_job_id='audio-job',result_refs_json=? WHERE pipeline_job_id='job'",
                   (json.dumps({'audio_path': str(audio)}),))
        db.execute("UPDATE durable_steps SET value=? WHERE step_key='pipeline:transcript'", (json.dumps(transcript),))
        db.execute("INSERT INTO durable_steps VALUES ('job','pdf:v2:hash:review:0:1',?,?)",
                   (json.dumps({'complete': True}), time.time()))
    edited = persistence.save_session('original', {'tops': ['Manual edit']})
    monkeypatch.setattr(llm_transport, 'model_fingerprint', lambda _: {'digest': 'different'})
    with pytest.raises(ValueError, match='digest changed'):
        resume_pipeline.resume('job', pdf_contract=True, fork_session=True)
    monkeypatch.setattr(llm_transport, 'model_fingerprint', lambda _: {'digest': 'same'})
    report = resume_pipeline.resume('job', apply=True, pdf_contract=True, fork_session=True)
    assert persistence.load_session('original') == edited
    assert persistence.load_session(report['session_id'])['transcript'] == transcript
    archived = persistence.load_latest_pipeline_job_for_session('original')
    assert archived['status'] == 'failed' and archived['stage'] == 'agenda_detect'
    assert jobs.load(archived['pipeline_job_id']) is None
    assert jobs.load('job')['documents'][0]['sha256'] == jobs.document(audio)['sha256']
    with persistence.connect() as db:
        assert db.execute("SELECT value FROM durable_steps WHERE step_key='pdf:v2:hash:review:0:1'").fetchone()
        assert db.execute("SELECT 1 FROM durable_steps WHERE step_key LIKE 'pdf:page-evidence-v3:%'").fetchone() is None
        history = json.loads(db.execute("SELECT value FROM durable_steps WHERE step_key LIKE 'operator:pdf-contract-resume:%'").fetchone()[0])
        assert history['checkpoint_hashes']['pipeline:transcript']
        assert 'historical input hash unavailable' in history['audio_binding']
