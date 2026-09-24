"""Durable local work queue. One backend process per local SQLite database.

A lifetime flock enforces that boundary; transactional leases fence stale workers.
Only model-output deltas update model progress. Heartbeats prove ownership only.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import asyncio
try:
    import fcntl
except ImportError:  # Native Windows tests/operator tools; production uses Linux flock.
    fcntl = None
    import msvcrt
import hashlib
import json
import os
import sqlite3
import logging
from pathlib import Path
import threading
import time
import uuid

import persistence
from llm_transport import LLMCancelledError, request_control, retryable

ACTIVE = {"queued", "running", "retry_wait"}
STORAGE_ERRORS = (sqlite3.OperationalError, OSError)
TERMINAL = {"completed", "review_required", "failed", "cancelled", "superseded"}
class LeaseLost(LLMCancelledError):
    """Ownership expired; this is a recoverable interruption, not a user cancellation."""


class WorkerStopped(LLMCancelledError):
    pass


CURRENT = ContextVar("durable_work", default=None)
PUBLIC = ("job_id", "kind", "state", "attempt", "heartbeat_at", "lease_until",
          "progress", "error", "created_at", "updated_at", "result", "documents")


def init_schema(db):
    db.executescript("""
        CREATE TABLE IF NOT EXISTS durable_jobs (
            job_id TEXT PRIMARY KEY, kind TEXT NOT NULL, state TEXT NOT NULL,
            payload TEXT NOT NULL, result TEXT, progress TEXT, error TEXT,
            owner TEXT, lease_until REAL, heartbeat_at REAL,
            attempt INTEGER NOT NULL DEFAULT 0, available_at REAL NOT NULL DEFAULT 0,
            created_at REAL NOT NULL, updated_at REAL NOT NULL,
            documents TEXT NOT NULL DEFAULT '[]'
        );
        CREATE INDEX IF NOT EXISTS durable_jobs_ready ON durable_jobs(state, available_at);
        CREATE TABLE IF NOT EXISTS durable_steps (
            job_id TEXT NOT NULL REFERENCES durable_jobs(job_id),
            step_key TEXT NOT NULL, value TEXT NOT NULL, completed_at REAL NOT NULL,
            PRIMARY KEY(job_id, step_key)
        );
        CREATE TABLE IF NOT EXISTS durable_step_integrity (
            job_id TEXT NOT NULL, step_key TEXT NOT NULL, sha256 TEXT NOT NULL,
            PRIMARY KEY(job_id, step_key)
        );
        CREATE TABLE IF NOT EXISTS durable_metrics (
            request_id TEXT PRIMARY KEY, job_id TEXT NOT NULL, phase TEXT NOT NULL,
            recorded_at REAL NOT NULL, metrics TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS durable_artifacts (
            artifact_id TEXT PRIMARY KEY, job_id TEXT NOT NULL, step_key TEXT NOT NULL,
            kind TEXT NOT NULL, value TEXT NOT NULL, sha256 TEXT NOT NULL, recorded_at REAL NOT NULL
        );
    """)


def _decode(row):
    if row is None:
        return None
    data = dict(row)
    for key in ("payload", "result", "progress", "documents"):
        data[key] = json.loads(data[key]) if data[key] else None
    return data


def load(job_id):
    with persistence.connect() as db:
        return _decode(db.execute("SELECT * FROM durable_jobs WHERE job_id=?", (job_id,)).fetchone())


def public(job):
    result = {key: job.get(key) for key in PUBLIC if key != "documents"} | {
        "documents": [{k: v for k, v in doc.items() if k != "path"} for doc in job.get("documents") or []]
    }
    if (result.get('progress') or {}).get('last_delta_at') is not None:
        result['progress'] = {**result['progress'], 'silence_seconds':
            max(0, time.time() - result['progress']['last_delta_at'])}
    if job['kind'] == 'agenda' and job['state'] in TERMINAL and job.get('payload', {}).get('session_id'):
        request = job['payload']['request']
        result['source'] = {key: request.get(key) for key in ('tops', 'top_ids', 'transcript', 'processing_mode')}
    return result


def latest_for_session(session_id, kind):
    with persistence.connect() as db:
        row = db.execute("""SELECT * FROM durable_jobs WHERE kind=?
            AND json_extract(payload, '$.session_id')=? ORDER BY created_at DESC, rowid DESC LIMIT 1""",
            (kind, session_id)).fetchone()
        return _decode(row)


def artifact(step, kind, value):
    """Private local evidence, never a successful processing checkpoint or API log."""
    ctx = CURRENT.get()
    if ctx is None:
        return
    encoded = json.dumps(value, ensure_ascii=False)
    with persistence.connect() as db:
        fence(db)
        db.execute('INSERT INTO durable_artifacts VALUES (?,?,?,?,?,?,?)',
                   (str(uuid.uuid4()), ctx.job_id, step, kind, encoded, hash_value(encoded), time.time()))


def version_snapshot(payload=None):
    # Code hashes include prompts, validators and chunking policy. No credentials.
    from llm_config import get_llm_config
    from processing_mode import VERSION as processing_version
    files = ("processing_mode.py", "llm_config.py", "durable_jobs.py", "summarize.py", "summary_grounding.py", "agenda_llm.py", "agenda_detection.py",
             "extract_tops.py", "llm_transport.py", "main.py", "agenda_context.py",
             "agenda_labels.py", "assignment_suggestions.py", "persistence.py", "source_contract.py")
    policy_keys = (
        "PDF_RENDER_DPI", "PDF_MAX_PAGE_PIXELS", "PDF_MAX_PAGES", "PDF_OUTPUT_TOKENS",
        "PDF_MODEL_ATTEMPTS", "PDF_REVIEW_ROUNDS",
        "LLM_CHUNK_CHARS", "SUMMARY_OUTPUT_TOKENS", "SUMMARY_MODEL_ATTEMPTS",
        "SUMMARY_RECONCILIATION_ROUNDS",
        "AGENDA_DETECTION_USE_LLM", "AGENDA_DETECTION_CHUNK_LINES", "AGENDA_DETECTION_CHUNK_OVERLAP_LINES",
        "AGENDA_DETECTION_CONTEXT_WINDOW_BEFORE", "AGENDA_DETECTION_CONTEXT_WINDOW_AFTER",
        "AGENDA_OUTPUT_TOKENS", "AGENDA_FAST_OUTPUT_TOKENS", "AGENDA_OUTPUT_TOKENS_PER_LINE", "AGENDA_REPAIR_SPLIT_DEPTH",
        "AGENDA_COMPACT_ASSIGNMENTS",
        "AGENDA_MODEL_ATTEMPTS", "AGENDA_SOURCE_REQUEST_ROUNDS",
    )
    payload = payload or {}
    legacy = payload.get('legacy_snapshot') or {}
    refs = legacy.get('refs') or legacy.get('result_refs') or {}
    if isinstance(refs, str):
        refs = json.loads(refs)
    request = payload.get('request') or refs.get('options') or refs
    mode = request.get('processing_mode', payload.get('processing_mode', 'slow'))
    models = {name: get_llm_config(request.get(name), processing_mode=mode).public_snapshot()
              for name in ('model', 'agenda_model', 'summary_model') if request.get(name)}
    return {"processing": {"mode": request.get("processing_mode", payload.get("processing_mode", "slow")),
                           "version": processing_version}, "transcription": {
        "code": hashlib.sha256(Path(__file__).with_name('transcribe.py').read_bytes()).hexdigest(),
        "policy": {key: os.environ.get(key) for key in ('WHISPER_MODEL', 'WHISPER_DEVICE',
            'WHISPER_LANGUAGE', 'WHISPER_BATCH_SIZE', 'WHISPER_CPU_THREADS', 'SPEAKER_EMBEDDING_MODEL')}
    }, "overrides": models, "model": get_llm_config(request.get("model"), processing_mode=mode).public_snapshot(), "policy": {
        key: os.environ.get(key) for key in policy_keys
    }, "code": {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in files}}


def submit(kind, payload, job_id=None, documents=None):
    job_id = job_id or str(uuid.uuid4())
    payload = {**payload, "versions": version_snapshot(payload)}
    now = time.time()
    with persistence.connect() as db:
        db.execute("""INSERT OR IGNORE INTO durable_jobs
            (job_id,kind,state,payload,documents,created_at,updated_at) VALUES (?,?,'queued',?,?,?,?)""",
            (job_id, kind, json.dumps(payload), json.dumps(documents or []), now, now))
    return load(job_id)


def claim(owner, lease_seconds, now=None):
    now = time.time() if now is None else now
    with persistence.connect() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("""SELECT job_id FROM durable_jobs
            WHERE (state IN ('queued','retry_wait') AND available_at<=?)
               OR (state='running' AND lease_until<?)
            ORDER BY created_at LIMIT 1""", (now, now)).fetchone()
        if row is None:
            return None
        db.execute("""UPDATE durable_jobs SET state='running',owner=?,lease_until=?,heartbeat_at=?,
            attempt=attempt+1,updated_at=? WHERE job_id=?""",
            (owner, now + lease_seconds, now, now, row[0]))
        return _decode(db.execute("SELECT * FROM durable_jobs WHERE job_id=?", (row[0],)).fetchone())


def fence(db, ctx=None):
    ctx = ctx or CURRENT.get()
    if ctx is None:
        return
    row = db.execute("SELECT state,owner,lease_until FROM durable_jobs WHERE job_id=?", (ctx.job_id,)).fetchone()
    if ctx.stop.is_set():
        raise WorkerStopped("Worker shutdown")
    if row is not None and row[0] == "cancelled":
        raise LLMCancelledError("Job cancelled")
    if row is None or row[0] != "running" or row[1] != ctx.owner or row[2] <= time.time():
        raise LeaseLost("Job ownership expired")


def check():
    if CURRENT.get():
        with persistence.connect() as db:
            fence(db)


def cancel(job_id):
    with persistence.connect() as db:
        db.execute("""UPDATE durable_jobs SET state='cancelled',updated_at=?
            WHERE job_id=? AND state IN ('queued','running','retry_wait')""", (time.time(), job_id))
    return load(job_id)


def hash_value(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def checkpoint(key, operation):
    ctx = CURRENT.get()
    if ctx is None:
        return operation()
    with persistence.connect() as db:
        fence(db)
        row = db.execute("""SELECT s.value, i.sha256 FROM durable_steps s
            LEFT JOIN durable_step_integrity i ON i.job_id=s.job_id AND i.step_key=s.step_key
            WHERE s.job_id=? AND s.step_key=?""", (ctx.job_id, key)).fetchone()
    if row:
        if row[1] and row[1] != hash_value(row[0]):
            raise ValueError('Checkpoint integrity mismatch')
        return json.loads(row[0])
    value = operation()
    with persistence.connect() as db:
        db.execute("BEGIN IMMEDIATE")
        fence(db)
        encoded = json.dumps(value)
        db.execute("INSERT INTO durable_steps VALUES (?,?,?,?)", (ctx.job_id, key, encoded, time.time()))
        db.execute('INSERT INTO durable_step_integrity VALUES (?,?,?)', (ctx.job_id, key, hash_value(encoded)))
    return value


def draft_checkpoint(key, operation, complete):
    """Retain inspectable partial results without certifying a completed step."""
    class IncompleteDraft(Exception):
        def __init__(self, value):
            self.value = value
    def run():
        value = operation()
        if not complete(value):
            raise IncompleteDraft(value)
        return value
    try:
        value = checkpoint(key, run)
    except IncompleteDraft as exc:
        artifact(key, 'incomplete_draft', exc.value)
        return exc.value
    if not complete(value):
        raise ValueError('Invalid completed checkpoint; explicit migration required')
    return value


def progress(value):
    ctx = CURRENT.get()
    if ctx:
        if 'elapsed_seconds' not in value and value.get('phase'):
            ctx.phase = value['phase']
        with persistence.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            fence(db)
            if 'elapsed_seconds' in value:
                row = db.execute('SELECT progress FROM durable_jobs WHERE job_id=?', (ctx.job_id,)).fetchone()
                previous = json.loads(row[0]) if row and row[0] else {}
                if previous.get('agenda_phase'):
                    value = {**{k: previous[k] for k in ('agenda_phase', 'processed_lines', 'total_lines', 'model_calls') if k in previous}, **value}
                phase = previous.get('pdf_phase', previous.get('phase', ''))
                if phase.startswith('pdf_') and phase != 'pdf_verified':
                    value = {**{k: previous[k] for k in ('page', 'total_pages', 'round') if k in previous},
                             **value, 'pdf_phase': phase}
            db.execute("UPDATE durable_jobs SET progress=?,updated_at=? WHERE job_id=?",
                       (json.dumps(value), time.time(), ctx.job_id))


@contextmanager
def publication(key):
    """Record a publication in the SAME transaction as the session revision."""
    ctx = CURRENT.get()
    if ctx is None:
        yield
        return
    old = ctx.publication
    ctx.publication = key
    try:
        yield
    finally:
        ctx.publication = old


def record_publication(db):
    ctx = CURRENT.get()
    if ctx and ctx.publication:
        db.execute("INSERT OR REPLACE INTO durable_steps VALUES (?,?,?,?)",
                   (ctx.job_id, ctx.publication, 'true', time.time()))


def published(key):
    ctx = CURRENT.get()
    if not ctx:
        return False
    with persistence.connect() as db:
        return db.execute("SELECT 1 FROM durable_steps WHERE job_id=? AND step_key=?", (ctx.job_id, key)).fetchone() is not None


class Runtime:
    def __init__(self, job, owner, stop):
        self.job_id, self.owner, self.stop = job["job_id"], owner, stop
        self.payload = job["payload"]
        self.publication = None
        self.phase = 'starting'


def record_metric(value):
    """Content-free local measurements, tied to the active pipeline phase."""
    ctx = CURRENT.get()
    if ctx:
        with persistence.connect() as db:
            fence(db)
            db.execute('INSERT INTO durable_metrics VALUES (?,?,?,?,?)',
                       (str(uuid.uuid4()), ctx.job_id, ctx.phase, time.time(), json.dumps(value)))


class ProcessLock:
    def __enter__(self):
        path = persistence.get_db_path().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(str(path) + ".worker.lock", "a")
        try:
            if fcntl is not None:
                fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            self.handle.close()
            raise RuntimeError("Only one backend process per SQLite database is supported (workers=1, replicas=1)")
        return self

    def __exit__(self, *args):
        self.handle.close()


class Manager:
    def __init__(self, runner, mirror=lambda *_: None):
        self.runner, self.mirror = runner, mirror
        self.stop_event = threading.Event()
        self.started = False
        self.lock = None
        self.lease = max(3, float(os.environ.get("MODEL_JOB_LEASE_SECONDS", "60")))
        self.max_attempts = max(1, int(os.environ.get("MODEL_JOB_MAX_ATTEMPTS", "3")))
        self.owner = str(uuid.uuid4())
        self.storage_fault = None

    async def start(self):
        if self.started:
            return
        if self.lock is None:
            self.lock = ProcessLock().__enter__()
        # Exclusive process lock proves no old worker still runs; don't wait a full lease after a crash.
        with persistence.connect() as db:
            db.execute("UPDATE durable_jobs SET lease_until=0 WHERE state='running'")
        with persistence.connect() as db:
            terminal = [_decode(row) for row in db.execute(
                "SELECT * FROM durable_jobs WHERE state NOT IN ('queued','running','retry_wait')").fetchall()]
        for job in terminal:
            self.mirror(job)
        self.started = True
        self.thread = threading.Thread(target=self._loop, name="durable-model-worker", daemon=True)
        self.thread.start()

    async def stop(self):
        if not self.started:
            return
        self.stop_event.set()
        # Keep lifetime lock until cooperative model/resource cancellation has actually finished.
        await asyncio.to_thread(self.thread.join)
        self.started = False
        self.lock.__exit__()

    def _loop(self):
        while not self.stop_event.is_set():
            try:
                job = claim(self.owner + "/" + str(uuid.uuid4()), self.lease)
                self.storage_fault = None
            except STORAGE_ERRORS as exc:
                self.storage_fault = type(exc).__name__
                logging.getLogger(__name__).warning("Durable queue database temporarily unavailable")
                self.stop_event.wait(1)
                continue
            except sqlite3.DatabaseError:
                # Do not silently kill the thread or pretend a corrupt database
                # is a valid empty queue. Health exposes the need for recovery.
                self.storage_fault = 'DatabaseError'
                logging.getLogger(__name__).error('Durable database integrity failure; operator recovery required')
                self.stop_event.wait(5)
                continue
            if job is None:
                self.stop_event.wait(0.5)
                continue
            try:
                self.execute(job)
            except Exception:
                # A failed persistence write leaves the lease recoverable. The attempt
                # counter still bounds repeated execution once that lease expires.
                logging.getLogger(__name__).warning("Durable worker could not persist job transition; lease retained")
                self.stop_event.wait(1)

    def execute(self, job):
        ctx = Runtime(job, job["owner"], self.stop_event)
        token = CURRENT.set(ctx)
        done = threading.Event()
        def heartbeat():
            delay = self.lease / 3
            while not done.wait(delay):
                try:
                    with persistence.connect(timeout=min(1, self.lease / 6)) as db:
                        db.execute('BEGIN IMMEDIATE')
                        # Calculate expiry AFTER obtaining the write lock. Never
                        # extend an expired lease or revive a replaced worker.
                        now = time.time()
                        changed = db.execute("""UPDATE durable_jobs SET heartbeat_at=?,lease_until=?
                            WHERE job_id=? AND owner=? AND state='running' AND lease_until>?""",
                            (now, now + self.lease, ctx.job_id, ctx.owner, now)).rowcount
                    if not changed:
                        return
                    delay = self.lease / 3
                except STORAGE_ERRORS as exc:
                    # Retry promptly after brief autosave contention instead of
                    # losing another full heartbeat interval while idle.
                    delay = min(1, self.lease / 12)
                    logging.getLogger(__name__).warning('Durable heartbeat storage temporarily unavailable (%s)',
                                                       getattr(exc, 'sqlite_errorname', type(exc).__name__))
        heart = threading.Thread(target=heartbeat, daemon=True)
        heart.start()
        state, result, error = "completed", None, None
        try:
            if job['attempt'] > self.max_attempts:
                raise RuntimeError("Job retry budget exhausted")
            if job["payload"]["versions"] != version_snapshot(job["payload"]):
                state, error = "failed", "Modell-/Promptkonfiguration geändert; neue Verarbeitung erforderlich"
            else:
                with request_control(check, progress):
                    if job['kind'] in {'summary', 'pipeline', 'agenda', 'pdf'}:
                        from llm_config import get_llm_config
                        from llm_transport import model_fingerprint
                        snapshots = job['payload']['versions']
                        names = {snapshots['model']['model']} | {
                            cfg['model'] for cfg in snapshots.get('overrides', {}).values()}
                        identities = {name: model_fingerprint(get_llm_config(name)) for name in sorted(names)}
                        original = checkpoint('model-identities', lambda: identities)
                        if original != identities or (job['attempt'] > 1 and
                                any(not identity.get('digest') for identity in identities.values())):
                            raise RuntimeError('Model identity changed or is not immutable; start a new job')
                    result, state = self.runner(job)
                check()
        except WorkerStopped:
            state = "queued"
        except LeaseLost:
            state = "retry_wait" if job['attempt'] < self.max_attempts else "failed"
            error = "LeaseLost"
        except LLMCancelledError:
            state = "cancelled"
        except persistence.SessionConflictError:
            state, error = "superseded", "Sitzung wurde bearbeitet; Ergebnis nicht übernommen"
        except Exception as exc:
            cause = exc
            while cause.__cause__ or cause.__context__:
                cause = cause.__cause__ or cause.__context__
            state = "retry_wait" if (retryable(cause) or isinstance(cause, STORAGE_ERRORS)) and job['attempt'] < self.max_attempts else "failed"
            error = getattr(exc, "public_message", type(exc).__name__)  # Never persist provider bodies/secrets.
            if hasattr(exc, 'review_result'):
                state, result = 'review_required', exc.review_result
        finally:
            done.set()
            heart.join()
            CURRENT.reset(token)
        with persistence.connect() as db:
            db.execute("""UPDATE durable_jobs SET state=?,result=?,error=?,updated_at=?,available_at=?,owner=NULL,lease_until=NULL
                WHERE job_id=? AND owner=? AND state='running'
                  AND (lease_until>? OR ? IN ('queued','retry_wait','cancelled','failed'))""",
                (state, json.dumps(result), error, time.time(), time.time() + min(300, 2 ** job['attempt']),
                 ctx.job_id, ctx.owner, time.time(), state))
            db.execute("UPDATE durable_jobs SET owner=NULL,lease_until=NULL WHERE job_id=? AND owner=? AND state='cancelled'",
                       (ctx.job_id, ctx.owner))
        self.mirror(load(ctx.job_id))


def cleanup_documents(*, dry_run=True, now=None):
    """Explicit operator maintenance, disabled by default; never purge review/error inputs."""
    days = float(os.environ.get("MODEL_DOCUMENT_RETENTION_DAYS", "0"))
    if days <= 0:
        return []
    now = time.time() if now is None else now
    candidates = []
    with persistence.connect() as db:
        db.execute("BEGIN IMMEDIATE")
        all_jobs = [_decode(row) for row in db.execute("SELECT * FROM durable_jobs").fetchall()]
        eligible = lambda job: (job['state'] in {'completed', 'cancelled'} and
            job['updated_at'] < now - days * 86400 and
            (not job['owner'] or (job['lease_until'] or 0) < now))
        protected = {doc['path'] for job in all_jobs if not eligible(job)
                     for doc in job['documents'] or [] if not doc.get('deleted_at')}
        rows = [job for job in all_jobs if eligible(job)]
        for job in rows:
            for doc in job['documents']:
                if doc.get('deleted_at') or doc['path'] in protected:
                    continue
                candidates.append({'job_id': job['job_id'], 'sha256': doc['sha256'], 'path': doc['path']})
                if not dry_run:
                    path = Path(doc['path'])
                    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != doc['sha256']:
                        raise RuntimeError('Document changed; refusing cleanup')
                    path.unlink(missing_ok=True)
                    doc['deleted_at'] = now
            if not dry_run:
                db.execute('UPDATE durable_jobs SET documents=? WHERE job_id=?', (json.dumps(job['documents']), job['job_id']))
    return candidates


def document(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'retained_at': time.time(), 'deleted_at': None}


def raise_if_transient(exc):
    if CURRENT.get() is None:
        return
    cause = exc
    while cause.__cause__ or cause.__context__:
        cause = cause.__cause__ or cause.__context__
    if retryable(cause) or isinstance(cause, STORAGE_ERRORS) or getattr(exc, 'transient', False):
        raise exc


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Explicit PDF retention maintenance')
    parser.add_argument('--cleanup-documents', action='store_true', required=True)
    parser.add_argument('--apply', action='store_true', help='Delete eligible files; default is dry-run')
    args = parser.parse_args()
    persistence.init_db()
    print(json.dumps(cleanup_documents(dry_run=not args.apply), indent=2))
