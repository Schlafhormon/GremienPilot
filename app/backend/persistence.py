"""SQLite persistence for sessions and transcription jobs."""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any


DEFAULT_DB_PATH = Path(os.environ.get("PERSISTENCE_DB_PATH", "data/sessions.sqlite3"))


class SessionConflictError(RuntimeError):
    """Raised when a client tries to overwrite a newer session revision."""

    def __init__(self, session_id: str, expected_revision: int, actual_revision: int):
        self.session_id = session_id
        self.expected_revision = expected_revision
        self.actual_revision = actual_revision
        super().__init__(
            f"Session {session_id} changed from revision "
            f"{expected_revision} to {actual_revision}"
        )


def get_db_path() -> Path:
    return Path(os.environ.get("PERSISTENCE_DB_PATH", str(DEFAULT_DB_PATH)))


def connect(db_path: Path | None = None, *, timeout: float = 5) -> sqlite3.Connection:
    path = db_path or get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=timeout)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_db(db_path: Path | None = None) -> None:
    """Create or migrate the local SQLite database."""
    with connect(db_path) as db:
        # The production database lives on a local Linux volume. Readers must
        # not prevent the queue heartbeat from committing a short write.
        db.execute('PRAGMA journal_mode = WAL')
        from durable_jobs import init_schema
        init_schema(db)
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                job_id TEXT,
                current_step INTEGER,
                skipped_assignment INTEGER NOT NULL DEFAULT 0,
                export_metadata_json TEXT,
                revision INTEGER NOT NULL DEFAULT 1,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS transcription_jobs (
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
                remember_speakers INTEGER NOT NULL DEFAULT 0,
                cancellation_requested INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                    ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS transcript_lines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                line_index INTEGER NOT NULL,
                speaker TEXT NOT NULL,
                text TEXT NOT NULL,
                start REAL NOT NULL,
                end REAL NOT NULL,
                FOREIGN KEY (job_id) REFERENCES transcription_jobs(job_id)
                    ON DELETE CASCADE,
                UNIQUE (job_id, line_index)
            );

            CREATE TABLE IF NOT EXISTS tops (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                top_index INTEGER NOT NULL,
                top_uid TEXT,
                title TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                    ON DELETE CASCADE,
                UNIQUE (session_id, top_index)
            );

            CREATE TABLE IF NOT EXISTS assignments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                line_index INTEGER NOT NULL,
                top_index INTEGER,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                    ON DELETE CASCADE,
                UNIQUE (session_id, line_index)
            );

            CREATE TABLE IF NOT EXISTS speaker_names (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                speaker_id TEXT NOT NULL,
                display_name TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                    ON DELETE CASCADE,
                UNIQUE (session_id, speaker_id)
            );

            CREATE TABLE IF NOT EXISTS speaker_profiles (
                profile_id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                scope TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                archived_at REAL
            );

            CREATE TABLE IF NOT EXISTS speaker_embeddings (
                embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id TEXT NOT NULL,
                embedding_json TEXT,
                embedding_blob BLOB,
                model_name TEXT NOT NULL,
                quality REAL,
                metadata_json TEXT,
                created_at REAL NOT NULL,
                FOREIGN KEY (profile_id) REFERENCES speaker_profiles(profile_id)
                    ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS job_speaker_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                local_speaker_id TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                model_name TEXT NOT NULL,
                quality REAL,
                quality_metadata_json TEXT,
                created_at REAL NOT NULL,
                FOREIGN KEY (job_id) REFERENCES transcription_jobs(job_id)
                    ON DELETE CASCADE,
                UNIQUE (job_id, local_speaker_id, model_name)
            );

            CREATE TABLE IF NOT EXISTS speaker_observations (
                observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                local_speaker_id TEXT NOT NULL,
                profile_id TEXT,
                confidence REAL,
                status TEXT NOT NULL CHECK (
                    status IN ('suggested', 'confirmed', 'rejected', 'manual')
                ),
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                FOREIGN KEY (job_id) REFERENCES transcription_jobs(job_id)
                    ON DELETE CASCADE,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                    ON DELETE CASCADE,
                FOREIGN KEY (profile_id) REFERENCES speaker_profiles(profile_id)
                    ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS pipeline_jobs (
                pipeline_job_id TEXT PRIMARY KEY,
                session_id TEXT,
                transcription_job_id TEXT,
                status TEXT NOT NULL,
                stage TEXT NOT NULL,
                progress INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                result_refs_json TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                    ON DELETE SET NULL,
                FOREIGN KEY (transcription_job_id) REFERENCES transcription_jobs(job_id)
                    ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                top_index INTEGER NOT NULL,
                summary TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                    ON DELETE CASCADE,
                UNIQUE (session_id, top_index)
            );

            CREATE TABLE IF NOT EXISTS summary_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                top_index INTEGER NOT NULL,
                review_json TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                    ON DELETE CASCADE,
                UNIQUE (session_id, top_index)
            );

            CREATE TABLE IF NOT EXISTS summary_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                top_index INTEGER NOT NULL,
                top_uid TEXT NOT NULL,
                state_json TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                    ON DELETE CASCADE,
                UNIQUE (session_id, top_index),
                UNIQUE (session_id, top_uid)
            );

            CREATE TABLE IF NOT EXISTS summary_jobs (
                summary_job_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                status TEXT NOT NULL,
                progress INTEGER NOT NULL DEFAULT 0,
                current_top INTEGER NOT NULL DEFAULT 0,
                total_tops INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                refs_json TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                    ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS session_transcript_lines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                line_index INTEGER NOT NULL,
                line_uid TEXT,
                speaker TEXT NOT NULL,
                text TEXT NOT NULL,
                start REAL NOT NULL,
                end REAL NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                    ON DELETE CASCADE,
                UNIQUE (session_id, line_index)
            );

            CREATE INDEX IF NOT EXISTS idx_transcription_jobs_session_id
                ON transcription_jobs(session_id);
            CREATE INDEX IF NOT EXISTS idx_transcript_lines_job_id
                ON transcript_lines(job_id);
            CREATE INDEX IF NOT EXISTS idx_session_transcript_lines_session_id
                ON session_transcript_lines(session_id);
            CREATE INDEX IF NOT EXISTS idx_speaker_profiles_scope
                ON speaker_profiles(scope);
            CREATE INDEX IF NOT EXISTS idx_speaker_embeddings_profile_id
                ON speaker_embeddings(profile_id);
            CREATE INDEX IF NOT EXISTS idx_job_speaker_embeddings_job_id
                ON job_speaker_embeddings(job_id);
            CREATE INDEX IF NOT EXISTS idx_speaker_observations_job_session
                ON speaker_observations(job_id, session_id);
            CREATE INDEX IF NOT EXISTS idx_speaker_observations_profile_id
                ON speaker_observations(profile_id);
            CREATE INDEX IF NOT EXISTS idx_pipeline_jobs_session_id
                ON pipeline_jobs(session_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_updated_at
                ON sessions(updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_summary_jobs_session_id
                ON summary_jobs(session_id, created_at DESC);
            """
        )
        existing_columns = {
            row["name"]
            for row in db.execute("PRAGMA table_info(transcription_jobs)").fetchall()
        }
        if "telemetry_json" in existing_columns:
            try:
                db.execute("ALTER TABLE transcription_jobs DROP COLUMN telemetry_json")
                existing_columns.remove("telemetry_json")
            except sqlite3.OperationalError:
                db.execute("UPDATE transcription_jobs SET telemetry_json = NULL")
        if "cancellation_requested" not in existing_columns:
            db.execute(
                """
                ALTER TABLE transcription_jobs
                ADD COLUMN cancellation_requested INTEGER NOT NULL DEFAULT 0
                """
            )
        if "remember_speakers" not in existing_columns:
            db.execute(
                """
                ALTER TABLE transcription_jobs
                ADD COLUMN remember_speakers INTEGER NOT NULL DEFAULT 0
                """
            )
        session_columns = {
            row["name"] for row in db.execute("PRAGMA table_info(sessions)").fetchall()
        }
        if "export_metadata_json" not in session_columns:
            db.execute(
                """
                ALTER TABLE sessions
                ADD COLUMN export_metadata_json TEXT
                """
            )
        if "enforce_top_order" not in session_columns:
            db.execute("ALTER TABLE sessions ADD COLUMN enforce_top_order INTEGER NOT NULL DEFAULT 0")
        if "processing_mode" not in session_columns:
            db.execute("ALTER TABLE sessions ADD COLUMN processing_mode TEXT NOT NULL DEFAULT 'slow' CHECK (processing_mode IN ('fast', 'slow'))")
        if "agenda_proposals_json" not in session_columns:
            db.execute("ALTER TABLE sessions ADD COLUMN agenda_proposals_json TEXT")
        if "pdf_source_job_id" not in session_columns:
            db.execute("ALTER TABLE sessions ADD COLUMN pdf_source_job_id TEXT")
        if "revision" not in session_columns:
            db.execute(
                """
                ALTER TABLE sessions
                ADD COLUMN revision INTEGER NOT NULL DEFAULT 1
                """
            )
        top_columns = {
            row["name"] for row in db.execute("PRAGMA table_info(tops)").fetchall()
        }
        # Additive migration: old rows keep their original timing semantics.
        for table in ('transcript_lines', 'session_transcript_lines'):
            columns = {r['name'] for r in db.execute(f'PRAGMA table_info({table})')}
            if 'timing_json' not in columns:
                db.execute(f'ALTER TABLE {table} ADD COLUMN timing_json TEXT')
        if "top_uid" not in top_columns:
            db.execute("ALTER TABLE tops ADD COLUMN top_uid TEXT")
        legacy_tops = db.execute(
            "SELECT id FROM tops WHERE top_uid IS NULL OR top_uid = ''"
        ).fetchall()
        for row in legacy_tops:
            db.execute(
                "UPDATE tops SET top_uid = ? WHERE id = ?",
                (str(uuid.uuid4()), row["id"]),
            )
        session_line_columns = {
            row["name"]
            for row in db.execute(
                "PRAGMA table_info(session_transcript_lines)"
            ).fetchall()
        }
        if "line_uid" not in session_line_columns:
            db.execute(
                "ALTER TABLE session_transcript_lines ADD COLUMN line_uid TEXT"
            )
        legacy_lines = db.execute(
            """
            SELECT id FROM session_transcript_lines
            WHERE line_uid IS NULL OR line_uid = ''
            """
        ).fetchall()
        for row in legacy_lines:
            db.execute(
                "UPDATE session_transcript_lines SET line_uid = ? WHERE id = ?",
                (str(uuid.uuid4()), row["id"]),
            )
        embedding_columns = {
            row["name"]
            for row in db.execute("PRAGMA table_info(speaker_embeddings)").fetchall()
        }
        if "metadata_json" not in embedding_columns:
            db.execute(
                """
                ALTER TABLE speaker_embeddings
                ADD COLUMN metadata_json TEXT
                """
            )


def _to_json(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def _from_json(value: str | None) -> Any:
    if not value:
        return None
    return json.loads(value)


def _speaker_profile_from_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def create_speaker_profile(
    display_name: str,
    *,
    scope: str | None = None,
    profile_id: str | None = None,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Create a global speaker profile after explicit user opt-in."""
    now = time.time()
    profile_id = profile_id or str(uuid.uuid4())
    with connect(db_path) as db:
        db.execute(
            """
            INSERT INTO speaker_profiles (
                profile_id, display_name, scope, created_at, updated_at, archived_at
            )
            VALUES (?, ?, ?, ?, ?, NULL)
            """,
            (profile_id, display_name, scope, now, now),
        )
    profile = load_speaker_profile(profile_id, db_path=db_path)
    if profile is None:
        raise RuntimeError("Speaker profile could not be loaded after creation")
    return profile


def rename_speaker_profile(
    profile_id: str,
    display_name: str,
    db_path: Path | None = None,
) -> dict[str, Any] | None:
    now = time.time()
    with connect(db_path) as db:
        db.execute(
            """
            UPDATE speaker_profiles
            SET display_name = ?, updated_at = ?
            WHERE profile_id = ?
            """,
            (display_name, now, profile_id),
        )
    return load_speaker_profile(profile_id, include_archived=True, db_path=db_path)


def update_speaker_profile(
    profile_id: str,
    *,
    display_name: str | None = None,
    scope: str | None = None,
    db_path: Path | None = None,
) -> dict[str, Any] | None:
    now = time.time()
    with connect(db_path) as db:
        db.execute(
            """
            UPDATE speaker_profiles
            SET display_name = COALESCE(?, display_name),
                scope = ?,
                updated_at = ?
            WHERE profile_id = ?
            """,
            (display_name, scope, now, profile_id),
        )
    return load_speaker_profile(profile_id, include_archived=True, db_path=db_path)


def archive_speaker_profile(
    profile_id: str,
    db_path: Path | None = None,
) -> dict[str, Any] | None:
    now = time.time()
    with connect(db_path) as db:
        db.execute(
            """
            UPDATE speaker_profiles
            SET archived_at = COALESCE(archived_at, ?), updated_at = ?
            WHERE profile_id = ?
            """,
            (now, now, profile_id),
        )
    return load_speaker_profile(profile_id, include_archived=True, db_path=db_path)


def anonymize_speaker_observations_for_profile(
    profile_id: str,
    db_path: Path | None = None,
) -> int:
    """Detach observations from an archived profile so responses no longer expose its name."""
    now = time.time()
    with connect(db_path) as db:
        cursor = db.execute(
            """
            UPDATE speaker_observations
            SET profile_id = NULL,
                updated_at = ?
            WHERE profile_id = ?
            """,
            (now, profile_id),
        )
        return int(cursor.rowcount or 0)


def delete_speaker_embeddings(
    profile_id: str,
    db_path: Path | None = None,
) -> int:
    """Delete all persisted global embeddings for a speaker profile."""
    with connect(db_path) as db:
        cursor = db.execute(
            "DELETE FROM speaker_embeddings WHERE profile_id = ?",
            (profile_id,),
        )
        return int(cursor.rowcount or 0)


def delete_speaker_embeddings_for_source(
    profile_id: str,
    *,
    job_id: str,
    local_speaker_id: str,
    observation_id: int | None = None,
    db_path: Path | None = None,
) -> int:
    """Delete only profile embeddings copied from one local session speaker."""
    delete_ids: list[int] = []
    for embedding in load_speaker_embeddings(profile_id, db_path=db_path):
        metadata = embedding.get("metadata") or {}
        if metadata.get("source_job_id") != job_id:
            continue
        if metadata.get("source_local_speaker_id") != local_speaker_id:
            continue
        if observation_id is not None:
            source_observation_id = metadata.get("source_observation_id")
            if (
                source_observation_id is not None
                and source_observation_id != observation_id
            ):
                continue
        delete_ids.append(int(embedding["embedding_id"]))

    if not delete_ids:
        return 0

    placeholders = ",".join("?" for _ in delete_ids)
    with connect(db_path) as db:
        cursor = db.execute(
            f"""
            DELETE FROM speaker_embeddings
            WHERE profile_id = ?
              AND embedding_id IN ({placeholders})
            """,
            [profile_id, *delete_ids],
        )
        return int(cursor.rowcount or 0)


def count_speaker_embeddings(
    profile_id: str,
    *,
    model_name: str | None = None,
    db_path: Path | None = None,
) -> int:
    conditions = ["profile_id = ?"]
    params: list[Any] = [profile_id]
    if model_name is not None:
        conditions.append("model_name = ?")
        params.append(model_name)

    with connect(db_path) as db:
        row = db.execute(
            f"""
            SELECT COUNT(*) AS count
            FROM speaker_embeddings
            WHERE {' AND '.join(conditions)}
            """,
            params,
        ).fetchone()
    return int(row["count"] if row else 0)


def count_all_speaker_embeddings(db_path: Path | None = None) -> int:
    with connect(db_path) as db:
        row = db.execute(
            "SELECT COUNT(*) AS count FROM speaker_embeddings"
        ).fetchone()
    return int(row["count"] if row else 0)


def count_job_speaker_embeddings(
    job_id: str | None = None,
    db_path: Path | None = None,
) -> int:
    conditions = []
    params: list[Any] = []
    if job_id is not None:
        conditions.append("job_id = ?")
        params.append(job_id)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    with connect(db_path) as db:
        row = db.execute(
            f"SELECT COUNT(*) AS count FROM job_speaker_embeddings {where}",
            params,
        ).fetchone()
    return int(row["count"] if row else 0)


def prune_speaker_embeddings(
    profile_id: str,
    *,
    model_name: str,
    max_count: int,
    db_path: Path | None = None,
) -> int:
    """Keep only the best bounded set of embeddings for one profile/model."""
    if max_count < 1:
        raise ValueError("max_count must be at least 1")

    with connect(db_path) as db:
        keep_rows = db.execute(
            """
            SELECT embedding_id
            FROM speaker_embeddings
            WHERE profile_id = ? AND model_name = ?
            ORDER BY COALESCE(quality, 0) DESC, created_at DESC, embedding_id DESC
            LIMIT ?
            """,
            (profile_id, model_name, max_count),
        ).fetchall()
        keep_ids = {int(row["embedding_id"]) for row in keep_rows}
        if not keep_ids:
            return 0

        placeholders = ",".join("?" for _ in keep_ids)
        cursor = db.execute(
            f"""
            DELETE FROM speaker_embeddings
            WHERE profile_id = ?
              AND model_name = ?
              AND embedding_id NOT IN ({placeholders})
            """,
            [profile_id, model_name, *keep_ids],
        )
        return int(cursor.rowcount or 0)


def load_speaker_profile(
    profile_id: str,
    *,
    include_archived: bool = False,
    db_path: Path | None = None,
) -> dict[str, Any] | None:
    with connect(db_path) as db:
        row = db.execute(
            """
            SELECT *
            FROM speaker_profiles
            WHERE profile_id = ?
              AND (? OR archived_at IS NULL)
            """,
            (profile_id, 1 if include_archived else 0),
        ).fetchone()
    return _speaker_profile_from_row(row)


def load_speaker_profiles(
    *,
    scope: str | None = None,
    include_archived: bool = False,
    db_path: Path | None = None,
) -> list[dict[str, Any]]:
    conditions = []
    params: list[Any] = []
    if scope is not None:
        conditions.append("scope = ?")
        params.append(scope)
    if not include_archived:
        conditions.append("archived_at IS NULL")
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    with connect(db_path) as db:
        rows = db.execute(
            f"""
            SELECT *
            FROM speaker_profiles
            {where}
            ORDER BY display_name, created_at
            """,
            params,
        ).fetchall()
    return [dict(row) for row in rows]


def save_speaker_embedding(
    profile_id: str,
    embedding: Any,
    *,
    model_name: str,
    quality: float | None = None,
    metadata: dict[str, Any] | None = None,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Store one speaker embedding without deriving or logging personal data."""
    embedding_json = None
    embedding_blob = None
    if isinstance(embedding, bytes):
        embedding_blob = sqlite3.Binary(embedding)
    else:
        embedding_json = _to_json(embedding)

    now = time.time()
    with connect(db_path) as db:
        cursor = db.execute(
            """
            INSERT INTO speaker_embeddings (
                profile_id, embedding_json, embedding_blob, model_name, quality,
                metadata_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                profile_id,
                embedding_json,
                embedding_blob,
                model_name,
                quality,
                _to_json(metadata or {}),
                now,
            ),
        )
        embedding_id = int(cursor.lastrowid)

    embeddings = load_speaker_embeddings(profile_id, db_path=db_path)
    for item in embeddings:
        if item["embedding_id"] == embedding_id:
            return item
    raise RuntimeError("Speaker embedding could not be loaded after creation")


def load_speaker_embeddings(
    profile_id: str,
    *,
    model_name: str | None = None,
    db_path: Path | None = None,
) -> list[dict[str, Any]]:
    conditions = ["profile_id = ?"]
    params: list[Any] = [profile_id]
    if model_name is not None:
        conditions.append("model_name = ?")
        params.append(model_name)

    with connect(db_path) as db:
        rows = db.execute(
            f"""
            SELECT *
            FROM speaker_embeddings
            WHERE {' AND '.join(conditions)}
            ORDER BY created_at, embedding_id
            """,
            params,
        ).fetchall()

    embeddings = []
    for row in rows:
        item = dict(row)
        blob = item.pop("embedding_blob")
        embedding_json = item.pop("embedding_json")
        item["embedding"] = (
            bytes(blob) if blob is not None else _from_json(embedding_json)
        )
        item["metadata"] = _from_json(item.pop("metadata_json", None)) or {}
        embeddings.append(item)
    return embeddings


def save_job_speaker_embedding(
    *,
    job_id: str,
    local_speaker_id: str,
    embedding: Any,
    model_name: str,
    quality: float | None = None,
    quality_metadata: dict[str, Any] | None = None,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Store a local speaker embedding extracted from one transcription job."""
    now = time.time()
    with connect(db_path) as db:
        db.execute(
            """
            INSERT INTO job_speaker_embeddings (
                job_id, local_speaker_id, embedding_json, model_name, quality,
                quality_metadata_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id, local_speaker_id, model_name) DO UPDATE SET
                embedding_json = excluded.embedding_json,
                quality = excluded.quality,
                quality_metadata_json = excluded.quality_metadata_json
            """,
            (
                job_id,
                local_speaker_id,
                _to_json(embedding) or "[]",
                model_name,
                quality,
                _to_json(quality_metadata or {}),
                now,
            ),
        )

    embedding_record = load_job_speaker_embedding(
        job_id,
        local_speaker_id,
        model_name=model_name,
        db_path=db_path,
    )
    if embedding_record is None:
        raise RuntimeError("Job speaker embedding could not be loaded after save")
    return embedding_record


def load_job_speaker_embedding(
    job_id: str,
    local_speaker_id: str,
    *,
    model_name: str | None = None,
    db_path: Path | None = None,
) -> dict[str, Any] | None:
    conditions = ["job_id = ?", "local_speaker_id = ?"]
    params: list[Any] = [job_id, local_speaker_id]
    if model_name is not None:
        conditions.append("model_name = ?")
        params.append(model_name)

    with connect(db_path) as db:
        row = db.execute(
            f"""
            SELECT *
            FROM job_speaker_embeddings
            WHERE {' AND '.join(conditions)}
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """,
            params,
        ).fetchone()
    if row is None:
        return None

    item = dict(row)
    item["embedding"] = _from_json(item.pop("embedding_json", None)) or []
    item["quality_metadata"] = (
        _from_json(item.pop("quality_metadata_json", None)) or {}
    )
    return item


def load_job_speaker_embeddings(
    job_id: str,
    *,
    model_name: str | None = None,
    db_path: Path | None = None,
) -> list[dict[str, Any]]:
    conditions = ["job_id = ?"]
    params: list[Any] = [job_id]
    if model_name is not None:
        conditions.append("model_name = ?")
        params.append(model_name)

    with connect(db_path) as db:
        rows = db.execute(
            f"""
            SELECT *
            FROM job_speaker_embeddings
            WHERE {' AND '.join(conditions)}
            ORDER BY local_speaker_id, created_at, id
            """,
            params,
        ).fetchall()

    embeddings = []
    for row in rows:
        item = dict(row)
        item["embedding"] = _from_json(item.pop("embedding_json", None)) or []
        item["quality_metadata"] = (
            _from_json(item.pop("quality_metadata_json", None)) or {}
        )
        embeddings.append(item)
    return embeddings


def save_speaker_observation(
    *,
    job_id: str,
    session_id: str,
    local_speaker_id: str,
    profile_id: str | None = None,
    confidence: float | None = None,
    status: str = "suggested",
    observation_id: int | None = None,
    db_path: Path | None = None,
) -> dict[str, Any]:
    now = time.time()
    with connect(db_path) as db:
        if observation_id is None:
            cursor = db.execute(
                """
                INSERT INTO speaker_observations (
                    job_id, session_id, local_speaker_id, profile_id, confidence,
                    status, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    session_id,
                    local_speaker_id,
                    profile_id,
                    confidence,
                    status,
                    now,
                    now,
                ),
            )
            observation_id = int(cursor.lastrowid)
        else:
            db.execute(
                """
                UPDATE speaker_observations
                SET job_id = ?,
                    session_id = ?,
                    local_speaker_id = ?,
                    profile_id = ?,
                    confidence = ?,
                    status = ?,
                    updated_at = ?
                WHERE observation_id = ?
                """,
                (
                    job_id,
                    session_id,
                    local_speaker_id,
                    profile_id,
                    confidence,
                    status,
                    now,
                    observation_id,
                ),
            )

    observation = load_speaker_observation(observation_id, db_path=db_path)
    if observation is None:
        raise RuntimeError("Speaker observation could not be loaded after save")
    return observation


def load_speaker_observation(
    observation_id: int,
    db_path: Path | None = None,
) -> dict[str, Any] | None:
    with connect(db_path) as db:
        row = db.execute(
            "SELECT * FROM speaker_observations WHERE observation_id = ?",
            (observation_id,),
        ).fetchone()
    return dict(row) if row is not None else None


def load_speaker_observations(
    *,
    job_id: str | None = None,
    session_id: str | None = None,
    profile_id: str | None = None,
    status: str | None = None,
    db_path: Path | None = None,
) -> list[dict[str, Any]]:
    conditions = []
    params: list[Any] = []
    if job_id is not None:
        conditions.append("job_id = ?")
        params.append(job_id)
    if session_id is not None:
        conditions.append("session_id = ?")
        params.append(session_id)
    if profile_id is not None:
        conditions.append("profile_id = ?")
        params.append(profile_id)
    if status is not None:
        conditions.append("status = ?")
        params.append(status)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    with connect(db_path) as db:
        rows = db.execute(
            f"""
            SELECT *
            FROM speaker_observations
            {where}
            ORDER BY created_at, observation_id
            """,
            params,
        ).fetchall()
    return [dict(row) for row in rows]


def confirm_speaker_observation(
    observation_id: int,
    *,
    profile_id: str | None = None,
    confidence: float | None = None,
    db_path: Path | None = None,
) -> dict[str, Any] | None:
    now = time.time()
    with connect(db_path) as db:
        db.execute(
            """
            UPDATE speaker_observations
            SET status = 'confirmed',
                profile_id = COALESCE(?, profile_id),
                confidence = COALESCE(?, confidence),
                updated_at = ?
            WHERE observation_id = ?
            """,
            (profile_id, confidence, now, observation_id),
        )
    return load_speaker_observation(observation_id, db_path=db_path)


def reject_speaker_observation(
    observation_id: int,
    db_path: Path | None = None,
) -> dict[str, Any] | None:
    now = time.time()
    with connect(db_path) as db:
        db.execute(
            """
            UPDATE speaker_observations
            SET status = 'rejected',
                updated_at = ?
            WHERE observation_id = ?
            """,
            (now, observation_id),
        )
    return load_speaker_observation(observation_id, db_path=db_path)


def save_pipeline_job(
    pipeline_job_id: str,
    job_data: dict[str, Any],
    db_path: Path | None = None,
) -> dict[str, Any]:
    now = time.time()
    created_at = float(job_data.get("created_at") or now)
    updated_at = float(job_data.get("updated_at") or now)

    with connect(db_path) as db:
        db.execute(
            """
            INSERT INTO pipeline_jobs (
                pipeline_job_id, session_id, transcription_job_id, status, stage,
                progress, error, result_refs_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(pipeline_job_id) DO UPDATE SET
                session_id = excluded.session_id,
                transcription_job_id = excluded.transcription_job_id,
                status = excluded.status,
                stage = excluded.stage,
                progress = excluded.progress,
                error = excluded.error,
                result_refs_json = excluded.result_refs_json,
                updated_at = excluded.updated_at
            """,
            (
                pipeline_job_id,
                job_data.get("session_id"),
                job_data.get("transcription_job_id"),
                job_data.get("status", "pending"),
                job_data.get("stage", "pending"),
                int(job_data.get("progress", 0)),
                job_data.get("error"),
                _to_json(job_data.get("result_refs") or {}),
                created_at,
                updated_at,
            ),
        )

    job = load_pipeline_job(pipeline_job_id, db_path=db_path)
    if job is None:
        raise RuntimeError("Pipeline job could not be loaded after save")
    return job


def load_pipeline_job(
    pipeline_job_id: str,
    db_path: Path | None = None,
) -> dict[str, Any] | None:
    with connect(db_path) as db:
        row = db.execute(
            "SELECT * FROM pipeline_jobs WHERE pipeline_job_id = ?",
            (pipeline_job_id,),
        ).fetchone()
    if row is None:
        return None

    job = dict(row)
    job["result_refs"] = _from_json(job.pop("result_refs_json", None)) or {}
    return job


def load_pipeline_jobs(
    *,
    session_id: str | None = None,
    status: str | None = None,
    db_path: Path | None = None,
) -> list[dict[str, Any]]:
    conditions = []
    params: list[Any] = []
    if session_id is not None:
        conditions.append("session_id = ?")
        params.append(session_id)
    if status is not None:
        conditions.append("status = ?")
        params.append(status)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    with connect(db_path) as db:
        rows = db.execute(
            f"""
            SELECT pipeline_job_id
            FROM pipeline_jobs
            {where}
            ORDER BY created_at
            """,
            params,
        ).fetchall()
    return [
        job
        for row in rows
        if (job := load_pipeline_job(row["pipeline_job_id"], db_path=db_path))
        is not None
    ]


def save_summary_job(
    summary_job_id: str,
    job_data: dict[str, Any],
    db_path: Path | None = None,
) -> dict[str, Any]:
    now = time.time()
    created_at = float(job_data.get("created_at") or now)
    updated_at = float(job_data.get("updated_at") or now)
    with connect(db_path) as db:
        db.execute(
            """
            INSERT INTO summary_jobs (
                summary_job_id, session_id, status, progress, current_top,
                total_tops, error, refs_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(summary_job_id) DO UPDATE SET
                session_id = excluded.session_id,
                status = excluded.status,
                progress = excluded.progress,
                current_top = excluded.current_top,
                total_tops = excluded.total_tops,
                error = excluded.error,
                refs_json = excluded.refs_json,
                updated_at = excluded.updated_at
            """,
            (
                summary_job_id,
                job_data["session_id"],
                job_data.get("status", "pending"),
                int(job_data.get("progress", 0)),
                int(job_data.get("current_top", 0)),
                int(job_data.get("total_tops", 0)),
                job_data.get("error"),
                _to_json(job_data.get("refs") or {}),
                created_at,
                updated_at,
            ),
        )
    job = load_summary_job(summary_job_id, db_path=db_path)
    if job is None:
        raise RuntimeError("Summary job could not be loaded after save")
    return job


def load_summary_job(
    summary_job_id: str,
    db_path: Path | None = None,
) -> dict[str, Any] | None:
    with connect(db_path) as db:
        row = db.execute(
            "SELECT * FROM summary_jobs WHERE summary_job_id = ?",
            (summary_job_id,),
        ).fetchone()
    if row is None:
        return None
    job = dict(row)
    job["refs"] = _from_json(job.pop("refs_json", None)) or {}
    return job


def load_latest_summary_job_for_session(
    session_id: str,
    db_path: Path | None = None,
) -> dict[str, Any] | None:
    with connect(db_path) as db:
        row = db.execute(
            """
            SELECT summary_job_id FROM summary_jobs
            WHERE session_id = ?
            ORDER BY updated_at DESC, created_at DESC
            LIMIT 1
            """,
            (session_id,),
        ).fetchone()
    if row is None:
        return None
    return load_summary_job(row["summary_job_id"], db_path=db_path)


def save_job(job_id: str, job_data: dict[str, Any], db_path: Path | None = None) -> None:
    """Upsert a transcription job and its transcript lines."""
    now = time.time()
    created_at = float(job_data.get("created_at") or now)
    updated_at = float(job_data.get("updated_at") or now)
    transcript = job_data.get("transcript")

    with connect(db_path) as db:
        if job_data.get("session_id"):
            db.execute(
                """
                INSERT INTO sessions (
                    session_id, job_id, current_step, skipped_assignment,
                    created_at, updated_at
                )
                VALUES (?, ?, NULL, 0, ?, ?)
                ON CONFLICT(session_id) DO NOTHING
                """,
                (job_data["session_id"], job_id, created_at, updated_at),
            )

        db.execute(
            """
            INSERT INTO transcription_jobs (
                job_id, session_id, status, progress, message, file_path, audio_path,
                audio_filename, audio_content_type, audio_size_bytes,
                remember_speakers, cancellation_requested, error,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET
                session_id = excluded.session_id,
                status = excluded.status,
                progress = excluded.progress,
                message = excluded.message,
                file_path = excluded.file_path,
                audio_path = excluded.audio_path,
                audio_filename = excluded.audio_filename,
                audio_content_type = excluded.audio_content_type,
                audio_size_bytes = excluded.audio_size_bytes,
                remember_speakers = excluded.remember_speakers,
                cancellation_requested = excluded.cancellation_requested,
                error = excluded.error,
                updated_at = excluded.updated_at
            """,
            (
                job_id,
                job_data.get("session_id"),
                job_data.get("status", "pending"),
                int(job_data.get("progress", 0)),
                job_data.get("message", ""),
                job_data.get("file_path"),
                job_data.get("audio_path"),
                job_data.get("audio_filename"),
                job_data.get("audio_content_type"),
                job_data.get("audio_size_bytes"),
                1 if job_data.get("remember_speakers") else 0,
                1 if job_data.get("cancellation_requested") else 0,
                job_data.get("error"),
                created_at,
                updated_at,
            ),
        )

        if job_data.get("session_id"):
            db.execute(
                """
                UPDATE sessions
                SET job_id = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (job_id, updated_at, job_data["session_id"]),
            )

        if transcript is not None:
            db.execute("DELETE FROM transcript_lines WHERE job_id = ?", (job_id,))
            db.executemany(
                """
                INSERT INTO transcript_lines (
                    job_id, line_index, speaker, text, start, end, timing_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        job_id,
                        index,
                        str(line.get("speaker", "")),
                        str(line.get("text", "")),
                        float(line.get("start", 0)),
                        float(line.get("end", 0)),
                        _to_json(line.get('timing')),
                    )
                    for index, line in enumerate(transcript)
                ],
            )


def load_job(job_id: str, db_path: Path | None = None) -> dict[str, Any] | None:
    with connect(db_path) as db:
        row = db.execute(
            """
            SELECT
                job_id, session_id, status, progress, message, file_path, audio_path,
                audio_filename, audio_content_type, audio_size_bytes,
                remember_speakers, cancellation_requested, error,
                created_at, updated_at
            FROM transcription_jobs
            WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()
        if row is None:
            return None

        lines = db.execute(
            """
            SELECT speaker, text, start, end, timing_json
            FROM transcript_lines
            WHERE job_id = ?
            ORDER BY line_index
            """,
            (job_id,),
        ).fetchall()

    job = dict(row)
    job["cancellation_requested"] = bool(job.get("cancellation_requested"))
    job["remember_speakers"] = bool(job.get("remember_speakers"))
    job["transcript"] = [_transcript_row(line) for line in lines] if lines else None
    return job


def load_jobs(db_path: Path | None = None) -> dict[str, dict[str, Any]]:
    with connect(db_path) as db:
        job_ids = [
            row["job_id"]
            for row in db.execute(
                "SELECT job_id FROM transcription_jobs ORDER BY created_at"
            ).fetchall()
        ]

    return {
        job_id: job
        for job_id in job_ids
        if (job := load_job(job_id, db_path=db_path)) is not None
    }


def mark_interrupted_jobs(db_path: Path | None = None) -> None:
    now = time.time()
    with connect(db_path) as db:
        db.execute(
            """
            UPDATE transcription_jobs
            SET status = 'failed',
                progress = CASE WHEN progress >= 100 THEN progress ELSE 0 END,
                message = 'Transkription wurde durch Backend-Neustart unterbrochen',
                error = 'Transkription wurde durch Backend-Neustart unterbrochen',
                updated_at = ?
            WHERE status IN ('pending', 'processing')
            """,
            (now,),
        )


def mark_interrupted_pipeline_jobs(db_path: Path | None = None) -> None:
    now = time.time()
    message = "Pipeline wurde durch Backend-Neustart unterbrochen"
    with connect(db_path) as db:
        db.execute(
            """
            UPDATE pipeline_jobs
            SET status = 'failed',
                progress = CASE WHEN progress >= 100 THEN progress ELSE 0 END,
                error = ?,
                updated_at = ?
            WHERE status IN ('pending', 'processing')
            """,
            (message, now),
        )


def mark_interrupted_summary_jobs(db_path: Path | None = None) -> None:
    now = time.time()
    with connect(db_path) as db:
        interrupted = db.execute(
            """
            SELECT session_id, refs_json FROM summary_jobs
            WHERE status IN ('pending', 'processing', 'cancelling')
            """
        ).fetchall()
        db.execute(
            """
            UPDATE summary_jobs
            SET status = 'failed',
                error = 'Zusammenfassungsjob wurde durch Backend-Neustart unterbrochen',
                updated_at = ?
            WHERE status IN ('pending', 'processing', 'cancelling')
            """,
            (now,),
        )
        for job in interrupted:
            refs = _from_json(job["refs_json"]) or {}
            previous_statuses = refs.get("previous_statuses") or {}
            for top_uid in refs.get("top_ids") or []:
                row = db.execute(
                    """
                    SELECT id, state_json FROM summary_states
                    WHERE session_id = ? AND top_uid = ?
                    """,
                    (job["session_id"], top_uid),
                ).fetchone()
                if row is None:
                    continue
                state = _from_json(row["state_json"]) or {}
                if state.get("status") not in {"queued", "running"}:
                    continue
                state["status"] = previous_statuses.get(top_uid) or "review_required"
                state["updated_at"] = now
                db.execute(
                    "UPDATE summary_states SET state_json = ? WHERE id = ?",
                    (_to_json(state) or "{}", row["id"]),
                )


def save_session(
    session_id: str,
    state: dict[str, Any],
    *,
    expected_revision: int | None = None,
    bump_revision: bool = True,
    db_path: Path | None = None,
) -> dict[str, Any]:
    # Prepare potentially multi-megabyte JSON before holding SQLite's single
    # writer lock. Revision checks and all publications remain transactional.
    metadata_json = _to_json(state.get('export_metadata') or {})
    proposals_json = _to_json(state.get('agenda_proposals'))
    reviews_json = {int(index): _to_json(value) or '{}' for index, value in (state.get('summary_reviews') or {}).items()}
    states_json = {int(index): _to_json(value) or '{}' for index, value in (state.get('summary_states') or {}).items()}
    transcript_rows = None if state.get('transcript') is None else [
        (session_id, index, str(line.get('line_id') or uuid.uuid4()),
         str(line.get('speaker', '')), str(line.get('text', '')),
         float(line.get('start', 0)), float(line.get('end', 0)), _to_json(line.get('timing')))
        for index, line in enumerate(state['transcript'])]
    now = time.time()
    with connect(db_path) as db:
        from durable_jobs import fence, record_publication
        # Serialize the revision check and write so concurrent autosaves cannot
        # both validate the same revision before either one commits.
        db.execute("BEGIN IMMEDIATE")
        fence(db)
        existing = db.execute(
            "SELECT created_at, revision, processing_mode, enforce_top_order, pdf_source_job_id FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        created_at = float(existing["created_at"]) if existing else now
        if existing is not None and expected_revision is not None:
            actual_revision = int(existing["revision"])
            if actual_revision != expected_revision:
                raise SessionConflictError(
                    session_id,
                    expected_revision,
                    actual_revision,
                )

        db.execute(
            """
            INSERT INTO sessions (
                session_id, job_id, current_step, skipped_assignment, export_metadata_json,
                agenda_proposals_json, processing_mode, enforce_top_order, pdf_source_job_id, revision, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                job_id = excluded.job_id,
                current_step = excluded.current_step,
                skipped_assignment = excluded.skipped_assignment,
                export_metadata_json = excluded.export_metadata_json,
                agenda_proposals_json = excluded.agenda_proposals_json,
                processing_mode = excluded.processing_mode,
                enforce_top_order = excluded.enforce_top_order,
                pdf_source_job_id = excluded.pdf_source_job_id,
                revision = sessions.revision + ?,
                updated_at = excluded.updated_at
            """,
            (
                session_id,
                state.get("job_id"),
                state.get("current_step"),
                1 if state.get("skipped_assignment") else 0,
                metadata_json,
                proposals_json,
                state.get("processing_mode", existing["processing_mode"] if existing else "slow"),
                int(state.get("enforce_top_order", existing["enforce_top_order"] if existing else False)),
                state.get("pdf_source_job_id", existing["pdf_source_job_id"] if existing else None),
                created_at,
                now,
                1 if bump_revision else 0,
            ),
        )

        if state.get("job_id"):
            # Multiple sessions may retain the same confirmed recording. Saving a
            # fork must not transfer ownership or alter the source job's timestamp.
            db.execute(
                """
                UPDATE transcription_jobs
                SET session_id = ?, updated_at = ?
                WHERE job_id = ? AND session_id IS NULL
                """,
                (session_id, now, state["job_id"]),
            )

        db.execute("DELETE FROM tops WHERE session_id = ?", (session_id,))
        top_ids = list(state.get("top_ids") or [])
        titles = list(state.get("tops") or [])
        if len(top_ids) != len(titles):
            top_ids = [str(uuid.uuid4()) for _ in titles]
        db.executemany(
            """
            INSERT INTO tops (session_id, top_index, top_uid, title)
            VALUES (?, ?, ?, ?)
            """,
            [
                (session_id, index, str(top_ids[index]), str(title))
                for index, title in enumerate(titles)
            ],
        )

        db.execute("DELETE FROM assignments WHERE session_id = ?", (session_id,))
        db.executemany(
            """
            INSERT INTO assignments (session_id, line_index, top_index)
            VALUES (?, ?, ?)
            """,
            [
                (session_id, index, assignment)
                for index, assignment in enumerate(state.get("assignments") or [])
            ],
        )

        db.execute("DELETE FROM speaker_names WHERE session_id = ?", (session_id,))
        db.executemany(
            """
            INSERT INTO speaker_names (session_id, speaker_id, display_name)
            VALUES (?, ?, ?)
            """,
            [
                (session_id, str(speaker_id), str(display_name))
                for speaker_id, display_name in (
                    state.get("speaker_names") or {}
                ).items()
            ],
        )

        db.execute("DELETE FROM summaries WHERE session_id = ?", (session_id,))
        db.executemany(
            "INSERT INTO summaries (session_id, top_index, summary) VALUES (?, ?, ?)",
            [
                (session_id, int(top_index), str(summary))
                for top_index, summary in (state.get("summaries") or {}).items()
            ],
        )

        db.execute("DELETE FROM summary_reviews WHERE session_id = ?", (session_id,))
        db.executemany(
            """
            INSERT INTO summary_reviews (session_id, top_index, review_json)
            VALUES (?, ?, ?)
            """,
            [
                (session_id, top_index, review_json)
                for top_index, review_json in reviews_json.items()
            ],
        )

        db.execute("DELETE FROM summary_states WHERE session_id = ?", (session_id,))
        db.executemany(
            """
            INSERT INTO summary_states (
                session_id, top_index, top_uid, state_json
            )
            VALUES (?, ?, ?, ?)
            """,
            [
                (
                    session_id,
                    int(top_index),
                    str(summary_state.get("top_id") or (
                        top_ids[int(top_index)]
                        if 0 <= int(top_index) < len(top_ids)
                        else f"whole-session:{session_id}"
                    )),
                    states_json[int(top_index)],
                )
                for top_index, summary_state in (
                    state.get("summary_states") or {}
                ).items()
            ],
        )

        if state.get("transcript") is not None:
            db.execute(
                "DELETE FROM session_transcript_lines WHERE session_id = ?",
                (session_id,),
            )
            db.executemany(
                """
                INSERT INTO session_transcript_lines (
                    session_id, line_index, line_uid, speaker, text, start, end, timing_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                transcript_rows,
            )

        record_publication(db)

    return load_session(session_id, db_path=db_path) or {}


def load_session(
    session_id: str,
    db_path: Path | None = None,
) -> dict[str, Any] | None:
    with connect(db_path) as db:
        # A summary worker can commit between these SELECTs. Read the revision,
        # text, reviews and metadata from one consistent database snapshot.
        db.execute("BEGIN")
        row = db.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return None

        tops = db.execute(
            """
            SELECT top_uid, title FROM tops
            WHERE session_id = ?
            ORDER BY top_index
            """,
            (session_id,),
        ).fetchall()
        assignments = db.execute(
            """
            SELECT line_index, top_index FROM assignments
            WHERE session_id = ?
            ORDER BY line_index
            """,
            (session_id,),
        ).fetchall()
        speaker_names = db.execute(
            """
            SELECT speaker_id, display_name FROM speaker_names
            WHERE session_id = ?
            ORDER BY speaker_id
            """,
            (session_id,),
        ).fetchall()
        summaries = db.execute(
            """
            SELECT top_index, summary FROM summaries
            WHERE session_id = ?
            ORDER BY top_index
            """,
            (session_id,),
        ).fetchall()
        summary_reviews = db.execute(
            """
            SELECT top_index, review_json FROM summary_reviews
            WHERE session_id = ?
            ORDER BY top_index
            """,
            (session_id,),
        ).fetchall()
        summary_states = db.execute(
            """
            SELECT top_index, top_uid, state_json FROM summary_states
            WHERE session_id = ?
            ORDER BY top_index
            """,
            (session_id,),
        ).fetchall()
        transcript = db.execute(
            """
            SELECT line_uid AS line_id, speaker, text, start, end, timing_json
            FROM session_transcript_lines
            WHERE session_id = ?
            ORDER BY line_index
            """,
            (session_id,),
        ).fetchall()

    session = dict(row)
    session["enforce_top_order"] = bool(session.get("enforce_top_order", False))
    session["skipped_assignment"] = bool(session["skipped_assignment"])
    session["tops"] = [item["title"] for item in tops]
    session["top_ids"] = [item["top_uid"] for item in tops]
    session["assignments"] = [item["top_index"] for item in assignments]
    session["speaker_names"] = {
        item["speaker_id"]: item["display_name"] for item in speaker_names
    }
    session["summaries"] = {
        int(item["top_index"]): item["summary"] for item in summaries
    }
    session["summary_reviews"] = {
        int(item["top_index"]): _from_json(item["review_json"]) or {}
        for item in summary_reviews
    }
    session["summary_states"] = {
        int(item["top_index"]): {
            **(_from_json(item["state_json"]) or {}),
            "top_id": item["top_uid"],
        }
        for item in summary_states
    }
    session["agenda_proposals"] = _from_json(session.pop("agenda_proposals_json", None))
    session["export_metadata"] = _from_json(session.get("export_metadata_json")) or {}
    session["transcript"] = [_transcript_row(line) for line in transcript] if transcript else None
    return session


def _transcript_row(row):
    value = dict(row)
    timing = _from_json(value.pop('timing_json', None))
    if timing is not None:
        value['timing'] = timing
    return value


def load_latest_pipeline_job_for_session(
    session_id: str,
    db_path: Path | None = None,
) -> dict[str, Any] | None:
    with connect(db_path) as db:
        row = db.execute(
            """
            SELECT pipeline_job_id
            FROM pipeline_jobs
            WHERE session_id = ?
            ORDER BY updated_at DESC, created_at DESC
            LIMIT 1
            """,
            (session_id,),
        ).fetchone()
    if row is None:
        return None
    return load_pipeline_job(row["pipeline_job_id"], db_path=db_path)


def list_sessions(
    *,
    query: str | None = None,
    status: str | None = None,
    limit: int = 25,
    offset: int = 0,
    db_path: Path | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Return lightweight session history entries without loading transcripts."""
    with connect(db_path) as db:
        rows = db.execute(
            """
            SELECT
                s.session_id,
                s.job_id,
                s.current_step,
                s.processing_mode,
                s.export_metadata_json,
                s.revision,
                s.created_at,
                s.updated_at,
                tj.status AS job_status,
                tj.audio_path,
                tj.file_path,
                latest_pipeline.pipeline_job_id,
                latest_pipeline.status AS pipeline_status,
                latest_pipeline.stage AS pipeline_stage,
                latest_pipeline.error AS pipeline_error,
                latest_pipeline.progress AS pipeline_progress,
                (SELECT COUNT(*) FROM tops t WHERE t.session_id = s.session_id)
                    AS top_count,
                (SELECT COUNT(*) FROM session_transcript_lines st
                    WHERE st.session_id = s.session_id) AS transcript_line_count,
                (SELECT COUNT(*) FROM summaries sm WHERE sm.session_id = s.session_id)
                    AS summary_count,
                (SELECT COUNT(*) FROM summary_states ss
                    WHERE ss.session_id = s.session_id
                      AND ss.state_json NOT LIKE '%"status": "ready"%')
                    AS unresolved_summary_count,
                (SELECT sj.status FROM summary_jobs sj
                    WHERE sj.session_id = s.session_id
                    ORDER BY sj.updated_at DESC LIMIT 1)
                    AS summary_job_status
            FROM sessions s
            LEFT JOIN transcription_jobs tj ON tj.job_id = s.job_id
            LEFT JOIN pipeline_jobs latest_pipeline
                ON latest_pipeline.pipeline_job_id = (
                    SELECT p.pipeline_job_id
                    FROM pipeline_jobs p
                    WHERE p.session_id = s.session_id
                    ORDER BY p.updated_at DESC, p.created_at DESC
                    LIMIT 1
                )
            ORDER BY s.updated_at DESC, s.session_id DESC
            """
        ).fetchall()

    normalized_query = (query or "").strip().casefold()
    normalized_status = (status or "").strip().lower()
    entries: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        metadata = _from_json(item.pop("export_metadata_json", None)) or {}
        top_count = int(item.get("top_count") or 0)
        transcript_count = int(item.get("transcript_line_count") or 0)
        summary_count = int(item.get("summary_count") or 0)
        unresolved_summary_count = int(item.get("unresolved_summary_count") or 0)
        summary_job_status = item.get("summary_job_status")
        pipeline_status = item.get("pipeline_status")
        job_status = item.get("job_status")

        if pipeline_status in {"pending", "processing"} or job_status in {
            "pending",
            "processing",
        } or summary_job_status in {"pending", "processing", "cancelling"}:
            derived_status = "processing"
        elif pipeline_status == "failed" or job_status == "failed":
            derived_status = "failed"
        elif pipeline_status == "cancelled" or job_status == "cancelled":
            derived_status = "cancelled"
        elif unresolved_summary_count > 0:
            derived_status = "review"
        elif summary_count > 0:
            derived_status = "ready"
        elif transcript_count > 0:
            derived_status = "review"
        elif pipeline_status == "failed" or job_status == "failed":
            derived_status = "failed"
        elif pipeline_status == "cancelled" or job_status == "cancelled":
            derived_status = "cancelled"
        else:
            derived_status = "draft"

        title = str(metadata.get("title") or "").strip()
        committee = str(metadata.get("committee") or "").strip()
        meeting_date = str(metadata.get("date") or "").strip()
        if not title or title == "Sitzungsprotokoll":
            title = committee or (
                f"Sitzung vom {meeting_date}" if meeting_date else "Unbenannte Sitzung"
            )

        item.update(
            {
                "title": title,
                "committee": committee,
                "meeting_date": meeting_date,
                "status": derived_status,
                "top_count": top_count,
                "transcript_line_count": transcript_count,
                "summary_count": summary_count,
                "audio_available": bool(
                    (item.get("audio_path") and os.path.exists(item["audio_path"]))
                    or (item.get("file_path") and os.path.exists(item["file_path"]))
                ),
            }
        )
        item.pop("audio_path", None)
        item.pop("file_path", None)

        if normalized_status and derived_status != normalized_status:
            continue
        if normalized_query:
            haystack = " ".join(
                [title, committee, meeting_date, item["session_id"]]
            ).casefold()
            if normalized_query not in haystack:
                continue
        entries.append(item)

    total = len(entries)
    return entries[offset : offset + limit], total
