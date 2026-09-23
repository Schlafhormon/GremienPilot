import importlib.util
import json
from pathlib import Path
import sqlite3


spec = importlib.util.spec_from_file_location('monitor_pipeline', Path(__file__).parents[1] / 'monitor_pipeline.py')
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


def test_snapshot_contains_quality_counts_but_no_source_or_error_content(tmp_path):
    path = tmp_path / 'fixture.sqlite3'
    private = 'PRIVATE SYNTHETIC SENTINEL'
    with sqlite3.connect(path) as db:
        db.executescript('''
            CREATE TABLE durable_jobs(job_id,state,attempt,progress,heartbeat_at);
            CREATE TABLE durable_steps(job_id,step_key,value,completed_at);
            CREATE TABLE durable_metrics(job_id,phase,metrics);
            CREATE TABLE pipeline_jobs(pipeline_job_id,result_refs_json);
        ''')
        db.execute('INSERT INTO durable_jobs VALUES (?,?,?,?,?)', ('job', 'running', 1,
            json.dumps(dict(phase='summary_final_review', prompt=private, error=private)), 1))
        db.execute('INSERT INTO durable_steps VALUES (?,?,?,?)', ('job', 'pipeline:transcript', private, 1))
        refs = dict(pdf_extraction=dict(processing_complete=True, items=[dict(title=private)],
            tops=[private], review_questions=[dict(description=private)]), summary_progress=dict(
            summary_reviews={'0': dict(structured=dict(review_questions=[dict(question=private)]),
                                      llm_usage=dict(processing_complete=True), error=private)}))
        db.execute('INSERT INTO pipeline_jobs VALUES (?,?)', ('job', json.dumps(refs)))
    result = monitor.snapshot(path, 'job')
    assert private not in json.dumps(result)
    assert result['quality']['pdf_entries'] == 1
    assert result['quality']['summary_verified'] == 1
    assert result['quality']['summary_questions'] == 1
    assert result['quality']['summary_errors'] == 1
    assert result['transcript_completed_at'] == 1
    assert len(result['transcript_sha256']) == 64
    assert not result['published']
    with sqlite3.connect(path) as db:
        db.execute('DELETE FROM durable_steps')
    assert monitor.snapshot(path, 'job')['transcript_sha256'] is None


def test_live_snapshot_reads_sqlite_inside_container(monkeypatch):
    calls = []
    def run(args, **kwargs):
        calls.append((args, kwargs))
        return type('Result', (), dict(returncode=0, stdout='{"state":"running"}'))()
    monkeypatch.setattr(monitor.subprocess, 'run', run)
    assert monitor.live_snapshot('backend', 'job') == {'state': 'running'}
    assert calls[0][0] == ['docker', 'exec', '-i', 'backend', 'python', '-', 'job']
    assert 'os.environ["PERSISTENCE_DB_PATH"]' in calls[0][1]['input']
