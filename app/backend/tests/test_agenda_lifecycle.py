"""Exercise actual HTTP persistence, legacy reads, and positional detector contracts."""
import copy
import sqlite3

from fastapi.testclient import TestClient

import main
import persistence


def test_proposals_survive_edits_old_clients_and_revision_conflicts(tmp_path, monkeypatch):
    monkeypatch.setenv('PERSISTENCE_DB_PATH', str(tmp_path / 'sessions.sqlite3'))
    persistence.init_db()
    with TestClient(main.app) as client:
        session = client.post('/api/sessions', json={
            'current_step': 2, 'tops': ['Haushalt', 'Schulbau'],
            'top_ids': ['top-a', 'top-b'],
            'transcript': [
                {'line_id': 'line-a', 'speaker': 'S1', 'text': 'TOP 1 Haushalt.', 'start': 0, 'end': 1},
                {'line_id': 'line-b', 'speaker': 'S2', 'text': 'TOP 2 Schulbau.', 'start': 1, 'end': 2},
            ], 'assignments': [0, 1],
        }).json()
        detected = client.post('/api/agenda-detection', json={
            'tops': session['tops'], 'transcript': session['transcript'],
            'preserve_transcript_structure': True, 'use_llm': False,
        })
        assert detected.status_code == 200
        result = detected.json()
        # Include an explicit review warning to verify lossless storage of uncertainty.
        result['uncertain_count'] = 1
        result['segments'][0]['uncertain'] = True
        result['warnings'] = ['Grenze prüfen']
        proposals = {
            'version': 1,
            'source': {key: copy.deepcopy(session[key]) for key in ('tops', 'top_ids', 'transcript')},
            'result': result,
        }
        session['agenda_proposals'] = proposals
        url = '/api/sessions/' + session['session_id']
        saved = client.put(url, json=session)
        assert saved.status_code == 200
        session = saved.json()
        original_revision = session['revision']
        # TOP insertion, manual assignment, then save and reopen.
        session['tops'].insert(1, 'Neuer TOP')
        session['top_ids'].insert(1, 'top-new')
        session['assignments'] = [1, 2]
        saved = client.put(url, json=session)
        assert saved.status_code == 200
        restored = client.get(url).json()
        assert restored['assignments'] == [1, 2]
        assert restored['agenda_proposals'] == proposals
        assert restored['agenda_proposals']['source']['top_ids'] != restored['top_ids']
        assert restored['revision'] == original_revision + 1
        # A stale client cannot overwrite either the assignments or their evidence.
        conflict = client.put(url, json=session)
        assert conflict.status_code == 409
        assert client.get(url).json() == restored
        # An older client omits the new field. Keep the old immutable evidence,
        # even when the client changes text and the source becomes stale.
        restored.pop('agenda_proposals')
        restored['transcript'][0]['text'] = 'Korrigierter Wortlaut'
        assert client.put(url, json=restored).status_code == 200
        reopened = client.get(url).json()
        assert reopened['agenda_proposals'] == proposals
        assert reopened['transcript'][0]['text'] == 'Korrigierter Wortlaut'
        assert reopened['agenda_proposals']['result']['uncertain_count'] == 1
        assert reopened['agenda_proposals']['result']['segments'][0]['uncertain'] is True


def test_migration_retains_legacy_uncertainty_without_rebinding_indices(tmp_path, monkeypatch):
    db_path = tmp_path / 'sessions.sqlite3'
    monkeypatch.setenv('PERSISTENCE_DB_PATH', str(db_path))
    persistence.init_db()
    persistence.save_session('legacy', {'tops': ['Geändert'], 'assignments': [0]})
    persistence.save_pipeline_job('old-pipeline', {
        'session_id': 'legacy', 'status': 'completed', 'stage': 'ready_for_review',
        'progress': 100, 'result_refs': {'agenda': {
            'strategy': 'legacy', 'uncertain_count': 3, 'segments': [],
            'warnings': ['Alte Unsicherheit'],
        }},
    })
    # Simulate the schema before this change and run the migration twice.
    with sqlite3.connect(db_path) as db:
        db.execute('ALTER TABLE sessions DROP COLUMN agenda_proposals_json')
    persistence.init_db()
    persistence.init_db()
    with TestClient(main.app) as client:
        session = client.get('/api/sessions/legacy').json()
        proposals = session['agenda_proposals']
        assert proposals['source'] is None
        assert proposals['result']['uncertain_count'] == 3
        assert proposals['result']['warnings'] == ['Alte Unsicherheit']
        assert proposals['result']['assignments'] == []
        assert session['assignments'] == [0]
        assert client.put('/api/sessions/legacy', json=session).status_code == 200
        assert client.get('/api/sessions/legacy').json()['agenda_proposals'] == proposals
        pipeline = client.get('/api/pipeline/old-pipeline/result').json()
        assert pipeline['agenda_detection'] == proposals['result'] | {'llm': None}


def test_explicit_detection_preserves_edited_line_identity_and_structure(tmp_path, monkeypatch):
    monkeypatch.setenv('PERSISTENCE_DB_PATH', str(tmp_path / 'sessions.sqlite3'))
    persistence.init_db()
    transcript = [{
        'line_id': 'edited-line', 'speaker': 'S1',
        'text': 'TOP 1 Haushalt. Wir beraten. TOP 2 Schulbau. Weitere Beratung.',
        'start': 0, 'end': 20,
    }]
    with TestClient(main.app) as client:
        response = client.post('/api/agenda-detection', json={
            'tops': ['Haushalt', 'Schulbau'], 'transcript': transcript,
            'preserve_transcript_structure': True, 'use_llm': False,
        })
    assert response.status_code == 200
    result = response.json()
    assert result['transcript'] == transcript
    assert len(result['assignments']) == 1
    assert all(segment['start_index'] == segment['end_index'] == 0 for segment in result['segments'])
