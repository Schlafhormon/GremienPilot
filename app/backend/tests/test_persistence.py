import sqlite3

import pytest

from fastapi.testclient import TestClient

import main
import persistence


def test_persistence_initializes_expected_tables(tmp_path, monkeypatch):
    db_path = tmp_path / "sessions.sqlite3"
    monkeypatch.setenv("PERSISTENCE_DB_PATH", str(db_path))

    persistence.init_db()

    with sqlite3.connect(db_path) as db:
        tables = {
            row[0]
            for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }

    assert {
        "sessions",
        "transcription_jobs",
        "transcript_lines",
        "tops",
        "assignments",
        "speaker_names",
        "speaker_profiles",
        "speaker_embeddings",
        "job_speaker_embeddings",
        "speaker_observations",
        "pipeline_jobs",
        "summaries",
        "summary_reviews",
        "summary_states",
        "summary_jobs",
        "session_transcript_lines",
    }.issubset(tables)


def test_persistence_migration_is_idempotent_and_keeps_session_state(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "sessions.sqlite3"
    monkeypatch.setenv("PERSISTENCE_DB_PATH", str(db_path))

    persistence.init_db()
    persistence.save_session(
        "session-idempotent",
        {
            "current_step": 2,
            "speaker_names": {"SPEAKER_00": "Alice"},
            "tops": ["Begrüßung"],
            "assignments": [0],
        },
    )

    persistence.init_db()
    persistence.init_db()

    session = persistence.load_session("session-idempotent")

    assert session["current_step"] == 2
    assert session["speaker_names"] == {"SPEAKER_00": "Alice"}
    assert session["tops"] == ["Begrüßung"]
    assert session["assignments"] == [0]


def test_session_history_and_revision_conflicts(tmp_path, monkeypatch):
    db_path = tmp_path / "sessions.sqlite3"
    monkeypatch.setenv("PERSISTENCE_DB_PATH", str(db_path))
    persistence.init_db()

    first = persistence.save_session(
        "session-draft",
        {
            "current_step": 1,
            "tops": [],
            "export_metadata": {
                "title": "Planungssitzung",
                "committee": "Hauptausschuss",
                "date": "2026-07-13",
            },
        },
    )
    persistence.save_session(
        "session-ready",
        {
            "current_step": 3,
            "tops": ["Haushalt"],
            "transcript": [
                {
                    "speaker": "SPEAKER_00",
                    "text": "Beratung",
                    "start": 0,
                    "end": 1,
                }
            ],
            "assignments": [0],
            "summaries": {0: "Der Haushalt wurde beraten."},
            "export_metadata": {
                "title": "Haushaltsausschuss",
                "committee": "Finanzausschuss",
                "date": "2026-07-12",
            },
        },
    )

    updated = persistence.save_session(
        "session-draft",
        {
            "current_step": 2,
            "tops": ["Eröffnung"],
            "export_metadata": first["export_metadata"],
        },
        expected_revision=first["revision"],
    )

    assert updated["revision"] == first["revision"] + 1
    with pytest.raises(persistence.SessionConflictError):
        persistence.save_session(
            "session-draft",
            {"current_step": 3},
            expected_revision=first["revision"],
        )

    items, total = persistence.list_sessions(query="Finanzausschuss")
    assert total == 1
    assert items[0]["session_id"] == "session-ready"
    assert items[0]["status"] == "ready"
    assert items[0]["transcript_line_count"] == 1
    assert items[0]["summary_count"] == 1

    reviews, review_total = persistence.list_sessions(status="review")
    assert review_total == 0
    assert reviews == []


def test_persistence_migrates_pre_speaker_pipeline_schema(tmp_path, monkeypatch):
    db_path = tmp_path / "legacy.sqlite3"
    monkeypatch.setenv("PERSISTENCE_DB_PATH", str(db_path))
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as db:
        db.executescript(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                job_id TEXT,
                current_step INTEGER,
                skipped_assignment INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE transcription_jobs (
                job_id TEXT PRIMARY KEY,
                session_id TEXT,
                status TEXT NOT NULL,
                progress INTEGER NOT NULL DEFAULT 0,
                message TEXT NOT NULL DEFAULT '',
                file_path TEXT,
                audio_path TEXT,
                audio_filename TEXT,
                audio_content_type TEXT,
                audio_size_bytes INTEGER,
                error TEXT,
                telemetry_json TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            INSERT INTO sessions (
                session_id, job_id, current_step, skipped_assignment, created_at, updated_at
            ) VALUES ('legacy-session', 'legacy-job', 1, 0, 1.0, 1.0);
            INSERT INTO transcription_jobs (
                job_id, session_id, status, progress, message, created_at, updated_at
            ) VALUES ('legacy-job', 'legacy-session', 'pending', 5, 'wartet', 1.0, 1.0);
            """
        )

    persistence.init_db()
    persistence.mark_interrupted_jobs()

    with sqlite3.connect(db_path) as db:
        db.row_factory = sqlite3.Row
        job_columns = {
            row["name"]
            for row in db.execute("PRAGMA table_info(transcription_jobs)").fetchall()
        }
        session_columns = {
            row["name"] for row in db.execute("PRAGMA table_info(sessions)").fetchall()
        }
        tables = {
            row["name"]
            for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }

    assert {"remember_speakers", "cancellation_requested"}.issubset(job_columns)
    assert "telemetry_json" not in job_columns
    assert {"export_metadata_json", "revision"}.issubset(session_columns)
    assert {
        "speaker_profiles",
        "speaker_embeddings",
        "job_speaker_embeddings",
        "speaker_observations",
        "pipeline_jobs",
        "summary_reviews",
    }.issubset(tables)
    legacy_job = persistence.load_job("legacy-job")
    assert legacy_job["status"] == "failed"
    assert legacy_job["remember_speakers"] is False
    assert legacy_job["cancellation_requested"] is False


def test_speaker_profiles_embeddings_observations_and_pipeline_jobs_roundtrip(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "sessions.sqlite3"
    monkeypatch.setenv("PERSISTENCE_DB_PATH", str(db_path))
    persistence.init_db()

    persistence.save_job(
        "job-speakers",
        {
            "created_at": 1.0,
            "updated_at": 1.0,
            "session_id": "session-speakers",
            "status": "completed",
            "progress": 100,
            "message": "Transkription abgeschlossen",
            "remember_speakers": True,
            "transcript": [
                {
                    "speaker": "SPEAKER_00",
                    "text": "Hallo",
                    "start": 0.0,
                    "end": 1.0,
                }
            ],
        },
    )
    persistence.save_session(
        "session-speakers",
        {
            "job_id": "job-speakers",
            "speaker_names": {"SPEAKER_00": "Alice lokal"},
            "tops": ["TOP 1"],
            "assignments": [0],
        },
    )

    profile = persistence.create_speaker_profile(
        "Alice Global",
        scope="committee-1",
        profile_id="profile-alice",
    )
    renamed = persistence.rename_speaker_profile("profile-alice", "Alice Beispiel")
    embedding = persistence.save_speaker_embedding(
        "profile-alice",
        [0.1, 0.2, 0.3],
        model_name="test-speaker-model",
        quality=0.87,
        metadata={"source_job_id": "job-speakers"},
    )
    job_embedding = persistence.save_job_speaker_embedding(
        job_id="job-speakers",
        local_speaker_id="SPEAKER_00",
        embedding=[0.2, 0.1, 0.4],
        model_name="test-speaker-model",
        quality=0.7,
        quality_metadata={"total_seconds": 9.0},
    )
    blob_embedding = persistence.save_speaker_embedding(
        "profile-alice",
        b"binary-embedding",
        model_name="test-speaker-model-bin",
    )

    observation = persistence.save_speaker_observation(
        job_id="job-speakers",
        session_id="session-speakers",
        local_speaker_id="SPEAKER_00",
        profile_id="profile-alice",
        confidence=0.76,
        status="suggested",
    )
    confirmed = persistence.confirm_speaker_observation(
        observation["observation_id"],
        confidence=0.91,
    )
    rejected = persistence.reject_speaker_observation(observation["observation_id"])

    pipeline_job = persistence.save_pipeline_job(
        "pipeline-1",
        {
            "session_id": "session-speakers",
            "transcription_job_id": "job-speakers",
            "status": "processing",
            "stage": "assignment",
            "progress": 40,
            "result_refs": {
                "transcription_job_id": "job-speakers",
                "session_id": "session-speakers",
            },
        },
    )
    loaded_pipeline_job = persistence.load_pipeline_job("pipeline-1")

    assert profile["display_name"] == "Alice Global"
    assert renamed["display_name"] == "Alice Beispiel"
    assert persistence.load_speaker_profile("profile-alice")["scope"] == "committee-1"
    assert persistence.load_speaker_profiles(scope="committee-1")[0]["profile_id"] == (
        "profile-alice"
    )
    assert embedding["embedding"] == [0.1, 0.2, 0.3]
    assert embedding["model_name"] == "test-speaker-model"
    assert embedding["metadata"] == {"source_job_id": "job-speakers"}
    assert job_embedding["embedding"] == [0.2, 0.1, 0.4]
    assert persistence.load_job("job-speakers")["remember_speakers"] is True
    assert persistence.load_job_speaker_embeddings("job-speakers")[0][
        "quality_metadata"
    ] == {"total_seconds": 9.0}
    assert blob_embedding["embedding"] == b"binary-embedding"
    assert persistence.load_speaker_embeddings(
        "profile-alice",
        model_name="test-speaker-model",
    )[0]["quality"] == 0.87
    assert confirmed["status"] == "confirmed"
    assert confirmed["confidence"] == 0.91
    assert rejected["status"] == "rejected"
    assert persistence.load_speaker_observations(
        job_id="job-speakers",
        session_id="session-speakers",
    )[0]["local_speaker_id"] == "SPEAKER_00"
    assert pipeline_job["progress"] == 40
    assert loaded_pipeline_job["result_refs"]["session_id"] == "session-speakers"

    archived = persistence.archive_speaker_profile("profile-alice")

    assert archived["archived_at"] is not None
    assert persistence.load_speaker_profile("profile-alice") is None
    assert persistence.load_speaker_profile(
        "profile-alice",
        include_archived=True,
    )["display_name"] == "Alice Beispiel"
    assert persistence.load_session("session-speakers")["speaker_names"] == {
        "SPEAKER_00": "Alice lokal"
    }


def test_prune_speaker_embeddings_keeps_best_limited_profile_references(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "sessions.sqlite3"
    monkeypatch.setenv("PERSISTENCE_DB_PATH", str(db_path))
    persistence.init_db()
    persistence.create_speaker_profile("Alice", profile_id="alice")

    for index, quality in enumerate([0.1, 0.9, 0.4, 0.8]):
        persistence.save_speaker_embedding(
            "alice",
            [float(index), 1.0],
            model_name="test-model",
            quality=quality,
            metadata={"index": index},
        )

    deleted = persistence.prune_speaker_embeddings(
        "alice",
        model_name="test-model",
        max_count=2,
    )
    remaining = persistence.load_speaker_embeddings(
        "alice",
        model_name="test-model",
    )

    assert deleted == 2
    assert [item["metadata"]["index"] for item in remaining] == [1, 3]


def test_transcription_job_is_restored_from_sqlite_after_memory_cache_is_cleared(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "sessions.sqlite3"
    audio_path = tmp_path / "meeting.mp3"
    audio_path.write_bytes(b"abcdef")
    monkeypatch.setenv("PERSISTENCE_DB_PATH", str(db_path))
    persistence.init_db()

    persistence.save_job(
        "job-1",
        {
            "created_at": 1.0,
            "updated_at": 1.0,
            "status": "completed",
            "progress": 100,
            "message": "Transkription abgeschlossen",
            "file_path": str(audio_path),
            "audio_path": str(audio_path),
            "audio_filename": "meeting.mp3",
            "audio_content_type": "audio/mpeg",
            "audio_size_bytes": audio_path.stat().st_size,
            "transcript": [
                {
                    "speaker": "SPEAKER_00",
                    "text": "Hallo zusammen",
                    "start": 0.0,
                    "end": 1.2,
                }
            ],
        },
    )

    main.jobs.clear()
    client = TestClient(main.app)

    response = client.get("/api/transcribe/job-1")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["audio_url"] == "/api/audio/job-1"
    assert body["audio_metadata"] == {
        "filename": "meeting.mp3",
        "content_type": "audio/mpeg",
        "size_bytes": 6,
    }
    assert body["transcript"] == [
        {
            "line_id": None,
            "speaker": "SPEAKER_00",
            "text": "Hallo zusammen",
            "start": 0.0,
            "end": 1.2,
        }
    ]

    audio_response = client.get("/api/audio/job-1", headers={"Range": "bytes=1-3"})
    assert audio_response.status_code == 206
    assert audio_response.content == b"bcd"

    main.jobs.clear()


def test_session_state_is_saved_and_loaded_with_linked_job(tmp_path, monkeypatch):
    db_path = tmp_path / "sessions.sqlite3"
    audio_path = tmp_path / "meeting.mp3"
    audio_path.write_bytes(b"audio")
    monkeypatch.setenv("PERSISTENCE_DB_PATH", str(db_path))
    persistence.init_db()

    persistence.save_job(
        "job-2",
        {
            "created_at": 1.0,
            "updated_at": 1.0,
            "status": "completed",
            "progress": 100,
            "message": "Transkription abgeschlossen",
            "file_path": str(audio_path),
            "audio_path": str(audio_path),
            "transcript": [
                {
                    "speaker": "SPEAKER_00",
                    "text": "Bericht zu TOP 1",
                    "start": 0.0,
                    "end": 2.0,
                },
                {
                    "speaker": "SPEAKER_01",
                    "text": "Beschluss zu TOP 2",
                    "start": 2.0,
                    "end": 4.0,
                },
            ],
        },
    )

    main.jobs.clear()
    client = TestClient(main.app)

    save_response = client.put(
        "/api/sessions/session-1",
        json={
            "job_id": "job-2",
            "current_step": 3,
            "tops": ["Begrüßung", "Haushalt"],
            "transcript": [
                {
                    "speaker": "SPEAKER_00",
                    "text": "Korrigierter Bericht zu TOP 1",
                    "start": 0.0,
                    "end": 2.0,
                },
                {
                    "speaker": "SPEAKER_01",
                    "text": "Korrigierter Beschluss zu TOP 2",
                    "start": 2.0,
                    "end": 4.0,
                },
            ],
            "assignments": [0, 1],
            "speaker_names": {
                "SPEAKER_00": "Alice",
                "SPEAKER_01": "Bob",
            },
            "summaries": {
                "0": "Begrüßung wurde abgeschlossen.",
                "1": "Haushalt wurde beschlossen.",
            },
            "export_metadata": {
                "committee": "Hauptausschuss",
                "date": "2026-06-30",
                "location": "Rathaus",
                "title": "Sitzung Hauptausschuss",
                "participants": ["Alice", "Bob"],
                "includeSpeakerList": True,
                "includeTranscriptExcerpt": False,
                "includeGenerationNote": True,
            },
            "summary_reviews": {
                "1": {
                    "structured": {
                        "discussion": [],
                        "decisions": ["Haushalt wurde beschlossen."],
                        "votes": [],
                        "action_items": [],
                        "open_points": [],
                        "uncertainties": [],
                    },
                    "source_links": [
                        {
                            "section": "decisions",
                            "item_index": 0,
                            "item_text": "Haushalt wurde beschlossen.",
                            "line_indices": [1],
                            "start": 2.0,
                            "end": 4.0,
                            "excerpt": "Beschluss zu TOP 2",
                            "confidence": 0.9,
                            "missing_source": False,
                        }
                    ],
                    "review_warnings": [],
                }
            },
            "skipped_assignment": False,
        },
    )

    assert save_response.status_code == 200
    assert save_response.json()["session_id"] == "session-1"

    main.jobs.clear()
    load_response = client.get("/api/sessions/session-1")

    assert load_response.status_code == 200
    body = load_response.json()
    assert body["job_id"] == "job-2"
    assert body["current_step"] == 3
    assert body["tops"] == ["Begrüßung", "Haushalt"]
    assert body["assignments"] == [0, 1]
    assert body["speaker_names"] == {
        "SPEAKER_00": "Alice",
        "SPEAKER_01": "Bob",
    }
    assert body["summaries"] == {
        "0": "Begrüßung wurde abgeschlossen.",
        "1": "Haushalt wurde beschlossen.",
    }
    assert body["summary_reviews"]["1"]["source_links"][0]["line_indices"] == [1]
    assert body["export_metadata"]["committee"] == "Hauptausschuss"
    assert body["export_metadata"]["participants"] == ["Alice", "Bob"]
    assert body["transcript"][0]["text"] == "Korrigierter Bericht zu TOP 1"
    assert body["transcript"][1]["text"] == "Korrigierter Beschluss zu TOP 2"
    assert body["audio_url"] == "/api/audio/job-2"
    assert body["job"]["status"] == "completed"

    main.jobs.clear()


def test_interrupted_summary_job_restores_previous_top_status(tmp_path, monkeypatch):
    db_path = tmp_path / "summary-sessions.sqlite3"
    monkeypatch.setenv("PERSISTENCE_DB_PATH", str(db_path))
    persistence.init_db()
    session = persistence.save_session(
        "summary-session",
        {
            "tops": ["Haushalt"],
            "top_ids": ["top-budget"],
            "transcript": [
                {
                    "line_id": "line-1",
                    "speaker": "SPEAKER_00",
                    "text": "Der Haushalt wird beraten.",
                    "start": 0,
                    "end": 1,
                }
            ],
            "assignments": [0],
            "summaries": {0: "Vorhandene Zusammenfassung."},
            "summary_states": {
                0: {
                    "top_id": "top-budget",
                    "status": "running",
                    "input_hash": "before",
                }
            },
        },
    )
    persistence.save_summary_job(
        "summary-job",
        {
            "session_id": session["session_id"],
            "status": "processing",
            "refs": {
                "top_ids": ["top-budget"],
                "previous_statuses": {"top-budget": "review_required"},
            },
        },
    )

    persistence.mark_interrupted_summary_jobs()

    restored = persistence.load_session("summary-session")
    job = persistence.load_summary_job("summary-job")
    assert restored["summary_states"][0]["status"] == "review_required"
    assert job["status"] == "failed"


def test_agenda_labels_and_stable_ids_survive_api_edit_reorder_and_reload(tmp_path, monkeypatch):
    monkeypatch.setenv('PERSISTENCE_DB_PATH', str(tmp_path / 'agenda.sqlite3'))
    persistence.init_db()
    tops = ['Alte Sitzung ohne Nummer', '[Öffentlich] 02.1 Schulbau', '[Nichtöffentlich] 02.1 Vergabe', '7 Anfragen']
    ids = ['legacy-id', 'public-id', 'private-id', 'questions-id']
    persistence.save_session('agenda-roundtrip', {
        'tops': tops, 'top_ids': ids, 'assignments': [1, 2],
        'transcript': [
            {'speaker': 'MOD', 'text': 'Schulbau', 'start': 0, 'end': 1},
            {'speaker': 'MOD', 'text': 'Vergabe', 'start': 1, 'end': 2},
        ],
    })
    client = TestClient(main.app)
    loaded = client.get('/api/sessions/agenda-roundtrip').json()
    assert loaded['tops'] == tops
    assert loaded['top_ids'] == ids
    order = [3, 2, 1, 0]
    updated_tops = [tops[i] for i in order]
    updated_tops[2] = '[Öffentlich] 02.1 Schulbau geändert'
    response = client.put('/api/sessions/agenda-roundtrip', json={
        'revision': loaded['revision'], 'tops': updated_tops,
        'top_ids': [ids[i] for i in order], 'transcript': loaded['transcript'],
        'assignments': [2, 1],
    })
    assert response.status_code == 200
    persistence.init_db()
    stored = persistence.load_session('agenda-roundtrip')
    assert stored['tops'] == updated_tops
    assert stored['top_ids'] == [ids[i] for i in order]
    assert stored['assignments'] == [2, 1]
