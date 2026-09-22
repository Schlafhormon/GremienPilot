import asyncio
from types import SimpleNamespace

import pytest

import main
import persistence
from summarize import SummarizationResult


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.setenv("PERSISTENCE_DB_PATH", str(tmp_path / "jobs.sqlite3"))
    persistence.init_db()
    async def enqueue(_):
        pass
    async def manager():
        return SimpleNamespace(enqueue=enqueue)
    monkeypatch.setattr(main, "get_or_create_summary_job_manager", manager)
    state = {
        "session_id": "meeting", "tops": ["A", "B", "C", "D"],
        "top_ids": ["a", "b", "c", "d"], "assignments": [0, 1, 2, 3],
        "transcript": [{"line_id": f"line-{i}", "speaker": "S", "text": title,
                        "start": i, "end": i + 1} for i, title in enumerate("ABCD")],
        "speaker_names": {}, "summaries": {i: f"Alt {title}" for i, title in enumerate("ABCD")},
        "summary_reviews": {3: {"duration_seconds": 42, "review_warnings": []}},
        "export_metadata": {"title": "Sitzung"}, "current_step": 3,
    }
    return persistence.save_session("meeting", main.reconcile_session_summaries(None, state))


def start(session, ids):
    return asyncio.run(main.create_summary_job("meeting", main.SummaryJobCreateRequest(
        revision=session["revision"], top_ids=ids))).summary_job_id


def test_multi_top_progress_partial_failure_and_unselected_content(session, monkeypatch):
    job_id = start(session, ["a", "b", "c", "a"])
    calls = []
    def generate(title, *args, **kwargs):
        calls.append(title)
        job = main.build_summary_job_response(persistence.load_summary_job(job_id))
        assert job.current_top_id == title.lower()
        assert job.processed_tops == len(calls) - 1
        assert job.progress == int((len(calls) - 1) / 3 * 100)
        if title == "C":
            assert job.completed_tops == 1
            assert job.outcomes["b"]["status"] == "failed"
            assert persistence.load_session("meeting")["summaries"][0] == "Neu A"
        if title == "B":
            raise RuntimeError("private transport details")
        return SummarizationResult(summary=f"Neu {title}", duration_seconds=1.5)
    monkeypatch.setattr(main, "summarize_segment", generate)
    main.run_summary_job(job_id)
    job = main.build_summary_job_response(persistence.load_summary_job(job_id))
    saved = persistence.load_session("meeting")
    assert calls == ["A", "B", "C"]
    assert job.status == "failed"
    assert job.completed_tops == 2 and job.processed_tops == 3
    assert job.progress == 100 and job.current_top_id is None
    assert "private transport details" not in str(job)
    assert saved["summaries"] == {0: "Neu A", 1: "Alt B", 2: "Neu C", 3: "Alt D"}
    assert saved["summary_reviews"][0]["duration_seconds"] == 1.5
    assert saved["summary_reviews"][3] == session["summary_reviews"][3]
    assert saved["summary_states"][3] == session["summary_states"][3]
    assert saved["export_metadata"] == session["export_metadata"]
    assert saved["summary_states"][1]["status"] == "failed"
    assert saved["summary_reviews"][1]["review_warnings"][-1]["kind"] == "summary_failed"
    assert saved["revision"] > session["revision"]


@pytest.mark.parametrize("change", ["manual", "source", "reorder", "delete"])
def test_edits_during_generation_are_preserved(session, monkeypatch, change):
    job_id = start(session, ["a"])
    def generate(*args, **kwargs):
        latest = persistence.load_session("meeting")
        edited = {**latest, "export_metadata": {"title": "Neue Metadaten"}}
        if change == "manual":
            edited["summaries"] = {**latest["summaries"], 0: "Manuelle Korrektur"}
        elif change == "source":
            edited["transcript"][0]["text"] = "Geänderter Inhalt"
        elif change == "reorder":
            edited["tops"] = ["B", "A", "C", "D"]
            edited["top_ids"] = ["b", "a", "c", "d"]
            edited["assignments"] = [1, 0, 2, 3]
            for key in ("summaries", "summary_reviews", "summary_states"):
                edited[key] = {({0: 1, 1: 0}.get(i, i)): value for i, value in latest[key].items()}
        else:
            edited["tops"] = ["B", "C", "D"]
            edited["top_ids"] = ["b", "c", "d"]
            edited["assignments"] = [None, 0, 1, 2]
            for key in ("summaries", "summary_reviews", "summary_states"):
                edited[key] = {i - 1: value for i, value in latest[key].items() if i > 0}
        persistence.save_session("meeting", main.reconcile_session_summaries(latest, edited))
        return SummarizationResult(summary="LLM-Ergebnis", duration_seconds=1)
    monkeypatch.setattr(main, "summarize_segment", generate)
    main.run_summary_job(job_id)
    saved = persistence.load_session("meeting")
    job = persistence.load_summary_job(job_id)
    assert saved["export_metadata"]["title"] == "Neue Metadaten"
    if change == "reorder":
        assert saved["summaries"][1] == "LLM-Ergebnis"
        assert saved["summaries"][0] == "Alt B"
        assert job["status"] == "completed"
    else:
        assert "LLM-Ergebnis" not in saved["summaries"].values()
        assert job["status"] == "failed"
    if change == "manual":
        assert saved["summaries"][0] == "Manuelle Korrektur"
        assert saved["summary_states"][0]["origin"] == "manual"


def test_publish_retries_revision_race_without_repeating_llm(session, monkeypatch):
    job_id = start(session, ["a"])
    calls = []
    monkeypatch.setattr(main, "summarize_segment", lambda *a, **k:
        calls.append(1) or SummarizationResult(summary="Neu A", duration_seconds=1))
    original_save = main.save_session
    raced = False
    def save(session_id, state, **kwargs):
        nonlocal raced
        if state["summaries"][0] == "Neu A" and not raced:
            raced = True
            latest = persistence.load_session(session_id)
            latest["summaries"][3] = "Manuell D"
            persistence.save_session(session_id, latest)
        return original_save(session_id, state, **kwargs)
    monkeypatch.setattr(main, "save_session", save)
    main.run_summary_job(job_id)
    assert raced and calls == [1]
    assert persistence.load_session("meeting")["summaries"][3] == "Manuell D"
    assert persistence.load_summary_job(job_id)["status"] == "completed"


def test_empty_top_does_not_abort_remaining_selection(session, monkeypatch):
    session["assignments"][0] = None
    session = persistence.save_session("meeting", session)
    job_id = start(session, ["a", "b"])
    calls = []
    monkeypatch.setattr(main, "summarize_segment", lambda title, *a, **k:
        calls.append(title) or SummarizationResult(summary="Neu B", duration_seconds=1))
    main.run_summary_job(job_id)
    assert calls == ["B"]
    assert persistence.load_summary_job(job_id)["status"] == "failed"
    assert persistence.load_session("meeting")["summaries"][1] == "Neu B"


def test_manual_edit_while_queued_skips_only_that_top(session, monkeypatch):
    job_id = start(session, ["a", "b", "c"])
    calls = []
    def generate(title, *args, **kwargs):
        calls.append(title)
        if title == "A":
            latest = persistence.load_session("meeting")
            edited = {**latest, "summaries": {**latest["summaries"], 1: "Manuell B"}}
            persistence.save_session("meeting", main.reconcile_session_summaries(latest, edited))
        return SummarizationResult(summary=f"Neu {title}", duration_seconds=1)
    monkeypatch.setattr(main, "summarize_segment", generate)
    main.run_summary_job(job_id)
    assert calls == ["A", "C"]
    saved = persistence.load_session("meeting")
    assert saved["summaries"] == {0: "Neu A", 1: "Manuell B", 2: "Neu C", 3: "Alt D"}
    assert saved["summary_states"][1]["status"] == "ready"
    assert persistence.load_summary_job(job_id)["status"] == "failed"


def test_cancellation_keeps_completed_results_and_restores_unprocessed_tops(session, monkeypatch):
    job_id = start(session, ["a", "b", "c"])
    def generate(title, *args, **kwargs):
        if title == "B":
            main.update_summary_job(job_id, status="cancelling", refs={"cancel_requested": True})
        return SummarizationResult(summary=f"Neu {title}", duration_seconds=1)
    monkeypatch.setattr(main, "summarize_segment", generate)
    main.run_summary_job(job_id)
    saved = persistence.load_session("meeting")
    job = main.build_summary_job_response(persistence.load_summary_job(job_id))
    assert job.status == "cancelled" and job.completed_tops == 1
    assert job.current_top_id is None
    assert saved["summaries"] == {0: "Neu A", 1: "Alt B", 2: "Alt C", 3: "Alt D"}
    assert all(state["status"] == "ready" for state in saved["summary_states"].values())


def test_manual_summary_does_not_keep_generated_evidence(session):
    session["summary_reviews"][0] = {"structured": {"decisions": ["Alter Beschluss"]},
                                     "source_links": [{"item_text": "Alter Beschluss"}], "duration_seconds": 12}
    edited = {**session, "summaries": {**session["summaries"], 0: "Manueller Text"}, "summary_reviews": {}}
    saved = main.reconcile_session_summaries(session, edited)
    assert 0 not in saved["summary_reviews"]
    assert saved["summary_reviews"][3] == session["summary_reviews"][3]


def test_poll_reads_text_and_review_from_one_revision(session, monkeypatch):
    # WAL lets the writer commit while the polling reader holds its snapshot.
    with persistence.connect() as db:
        db.execute("PRAGMA journal_mode=WAL")
    connect = persistence.connect
    injected = False
    def during_read(sql):
        nonlocal injected
        if "FROM summary_reviews" in sql and not injected:
            injected = True
            changed = {**session, "summaries": {**session["summaries"], 3: "Neues Ergebnis"},
                       "summary_reviews": {3: {"duration_seconds": 7}}}
            persistence.save_session("meeting", changed)
    def traced_connect(*args, **kwargs):
        db = connect(*args, **kwargs)
        db.set_trace_callback(during_read)
        return db
    monkeypatch.setattr(persistence, "connect", traced_connect)
    read = persistence.load_session("meeting")
    assert injected
    assert read["revision"] == session["revision"]
    assert read["summaries"][3] == "Alt D"
    assert read["summary_reviews"][3]["duration_seconds"] == 42
    assert persistence.load_session("meeting")["summaries"][3] == "Neues Ergebnis"


def test_cancellation_reaches_waiting_llm_and_preserves_manual_content(session, monkeypatch):
    from llm_transport import _CONTROL
    job_id = start(session, ['a', 'b'])
    def generate(*args, **kwargs):
        check_cancel, progress = _CONTROL.get()
        assert check_cancel is not None and progress is not None
        progress({'phase': 'waiting', 'model': 'test', 'config_id': 'id'})
        assert persistence.load_summary_job(job_id)['refs']['llm_progress']['phase'] == 'waiting'
        main.update_summary_job(job_id, status='cancelling', refs={'cancel_requested': True})
        check_cancel()
        pytest.fail('Cancellation did not interrupt generation')
    monkeypatch.setattr(main, 'summarize_segment', generate)
    main.run_summary_job(job_id)
    assert persistence.load_summary_job(job_id)['status'] == 'cancelled'
    assert persistence.load_session('meeting')['summaries'] == session['summaries']
