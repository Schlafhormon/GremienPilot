import json
import time
import threading
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

import main
import persistence
import summarize
import pytest
import agenda_detection
import extract_tops
from pdf_fixtures import agenda, item, audit, pdf_bytes
from conftest import FakeTranscriptionResult
from speaker_recognition import LocalSpeakerEmbedding


def wait_until(predicate, timeout=15.0, interval=0.02):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def configure_test_app(tmp_path, monkeypatch, *, concurrency=1):
    db_path = tmp_path / "sessions.sqlite3"
    upload_dir = tmp_path / "uploads"
    monkeypatch.setenv("PERSISTENCE_DB_PATH", str(db_path))
    monkeypatch.setattr(main, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(main, "TRANSCRIPTION_CONCURRENCY", concurrency)
    persistence.init_db()
    main.jobs.clear()
    return upload_dir


@pytest.mark.parametrize("agenda_source, prompt_mode", [
    ("transcript", "legacy"),
    ("manual", "legacy"),
    ("pdf", "legacy"),
    ("pdf", "legacy_options"),
    ("pdf", "scoped_form"),
    ("pdf", "scoped_options"),
    ("pdf", "blank_options"),
    ("pdf", "blank_form"),
])
def test_pipeline_routes_prompts_to_actual_model_messages(
    tmp_path, monkeypatch, fake_openai_module, frontend_summary_prompt,
    agenda_source, prompt_mode, agenda_model):
    configure_test_app(tmp_path, monkeypatch)
    monkeypatch.setattr(main, "transcribe_audio", lambda *args, **kwargs: FakeTranscriptionResult(
        transcript=[{"speaker": "MOD", "text": "TOP 1 Haushalt.", "start": 0.0, "end": 2.0}],
        audio_duration_seconds=2.0,
    ))
    # Only PDF decoding, transcription and the model transport are doubles.
    # Routing, prompt builders, parsing and persistence run as in production.
    monkeypatch.setenv("LLM_IMAGE_TOKENS", "1024")
    data = {"model": "test-model", "system_prompt": frontend_summary_prompt, "agenda_use_llm": "true"}
    scoped = {
        "summary_system_prompt": "SUMMARY_ONLY: Sachliche Niederschrift.",
        "agenda_system_prompt": "AGENDA_ONLY: Moderationssignale beachten.",
        "pdf_system_prompt": "PDF_ONLY: Einladung beachten.",
    }
    if prompt_mode == "scoped_form":
        data.update(scoped)
        data["options"] = json.dumps({key: "OVERRIDDEN_OPTION" for key in scoped})
    elif prompt_mode == "scoped_options":
        data["options"] = json.dumps(scoped)
    elif prompt_mode == "blank_options":
        data["options"] = json.dumps({key: "" for key in scoped})
    elif prompt_mode == "blank_form":
        data.update({key: "" for key in scoped})
        data["options"] = json.dumps({key: "OVERRIDDEN_OPTION" for key in scoped})
    elif prompt_mode == "legacy_options":
        data["options"] = json.dumps({"system_prompt": data.pop("system_prompt")})
    files = {"audio": ("meeting.mp3", b"audio", "audio/mpeg")}
    if agenda_source == "manual":
        data["tops"] = json.dumps(["Haushalt"])
    if agenda_source == "pdf":
        data["auto_detect_tops_from_pdf"] = "true"
        files["pdf"] = ("agenda.pdf", pdf_bytes(), "application/pdf")
        fake_openai_module.responses.extend(json.dumps(v) for v in [agenda(), agenda(), audit(), audit(0)])
    from summary_fixtures import SummaryModel
    fake_openai_module.content = SummaryModel()
    with TestClient(main.app) as client:
        response = client.post("/api/pipeline/start", data=data, files=files)
        assert response.status_code == 200
        pipeline_id = response.json()["pipeline_id"]
        assert wait_until(lambda: client.get(f"/api/pipeline/{pipeline_id}").json()["stage"] == "ready_for_review")
        result = client.get(f"/api/pipeline/{pipeline_id}/result").json()

    calls = [call for instance in fake_openai_module.instances for call in instance.calls]
    assert len(calls) == (10 if agenda_source == "pdf" else 6)
    assert len(agenda_model.calls) == 8
    routed = [(call, 'agenda_system_prompt') for _, call in agenda_model.calls]
    routed += [(call, 'pdf_system_prompt' if i < 2 else 'audit') for i, call in enumerate(calls[:-6])]
    routed += [(call, 'summary_system_prompt') for call in calls[-6:]]
    for call, key in routed:
        assert call['model'] == 'test-model'
        prompt = call['messages'][0]['content']
        assert 'OVERRIDDEN_OPTION' not in prompt
        assert (frontend_summary_prompt in prompt) == (key == 'summary_system_prompt' and prompt_mode.startswith('legacy'))
        for scoped_key, custom_prompt in scoped.items():
            assert (custom_prompt in prompt) == (prompt_mode.startswith('scoped_') and key == scoped_key)
    expected_top = 'Haushalt'
    assert f'TOP: {expected_top}' in calls[-1]['messages'][0]['content']
    assert result['session']['tops'] == [expected_top]
    assert result["session"]["assignments"] == [0]
    proposals = result["session"]["agenda_proposals"]
    assert {key: proposals["source"][key] for key in ("tops", "top_ids", "transcript")} == {
        key: result["session"][key] for key in ("tops", "top_ids", "transcript")
    }
    assert proposals["result"]["assignments"] == [0]
    persisted = persistence.load_session(result["session"]["session_id"])
    assert persisted["agenda_proposals"] == proposals
    assert result["agenda_detection"]["strategy"] == "model_agenda_v1"
    assert result["session"]["summary_reviews"]["0"]["fallback_used"] is False


def test_audio_upload_validation_accepts_supported_content_types_and_extensions():
    assert main.is_allowed_audio_file("recording.bin", "audio/mpeg")
    assert main.is_allowed_audio_file("recording.M4A", "application/octet-stream")
    assert main.is_allowed_audio_file("meeting.wav", None)


def test_audio_upload_validation_rejects_unsupported_files():
    assert not main.is_allowed_audio_file("agenda.pdf", "application/pdf")
    assert not main.is_allowed_audio_file("notes.txt", "text/plain")
    assert not main.is_allowed_audio_file(None, None)


def test_audio_filename_is_normalized_and_not_used_directly_in_path(tmp_path, monkeypatch):
    upload_dir = configure_test_app(tmp_path, monkeypatch)

    def fake_transcribe(file_path, models, progress_callback=None):
        return FakeTranscriptionResult(
            transcript=[
                {
                    "speaker": "SPEAKER_00",
                    "text": "Hallo",
                    "start": 0.0,
                    "end": 1.0,
                }
            ],
            audio_duration_seconds=1.0,
        )

    monkeypatch.setattr(main, "transcribe_audio", fake_transcribe)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/transcribe",
            files={"audio": ("..\\unsafe name ä.mp3", b"audio", "audio/mpeg")},
        )

        assert response.status_code == 200
        job_id = response.json()["job_id"]

        assert wait_until(
            lambda: client.get(f"/api/transcribe/{job_id}").json()["status"]
            == "completed"
        )
        job = main.get_job_from_cache_or_db(job_id)

    assert job["audio_filename"] == "unsafe_name_a.mp3"
    assert Path(job["file_path"]).parent == upload_dir
    assert Path(job["file_path"]).name == f"{job_id}.mp3"


def test_audio_upload_rejects_files_larger_than_configured_limit(tmp_path, monkeypatch):
    upload_dir = configure_test_app(tmp_path, monkeypatch)
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 5)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/transcribe",
            files={"audio": ("meeting.mp3", b"123456", "audio/mpeg")},
        )

    assert response.status_code == 413
    assert not list(upload_dir.glob("*"))


def test_transcription_jobs_are_queued_and_respect_concurrency_limit(tmp_path, monkeypatch):
    configure_test_app(tmp_path, monkeypatch, concurrency=1)
    first_started = threading.Event()
    release_first = threading.Event()
    calls = []

    def fake_transcribe(file_path, models, progress_callback=None):
        calls.append(file_path)
        if len(calls) == 1:
            first_started.set()
            assert release_first.wait(timeout=5)
        if progress_callback:
            progress_callback(80, "Fast fertig")
        return FakeTranscriptionResult(
            transcript=[
                {
                    "speaker": "SPEAKER_00",
                    "text": Path(file_path).name,
                    "start": 0.0,
                    "end": 1.0,
                }
            ],
            audio_duration_seconds=1.0,
        )

    monkeypatch.setattr(main, "transcribe_audio", fake_transcribe)

    with TestClient(main.app) as client:
        first = client.post(
            "/api/transcribe",
            files={"audio": ("first.mp3", b"one", "audio/mpeg")},
        ).json()["job_id"]
        second = client.post(
            "/api/transcribe",
            files={"audio": ("second.mp3", b"two", "audio/mpeg")},
        ).json()["job_id"]

        assert first_started.wait(timeout=5)
        assert wait_until(
            lambda: client.get(f"/api/transcribe/{first}").json()["status"]
            == "processing"
        )
        assert client.get(f"/api/transcribe/{second}").json()["status"] == "pending"
        assert len(calls) == 1

        release_first.set()

        assert wait_until(
            lambda: client.get(f"/api/transcribe/{first}").json()["status"]
            == "completed"
        )
        assert wait_until(
            lambda: client.get(f"/api/transcribe/{second}").json()["status"]
            == "completed"
        )


def test_transcription_creates_reviewable_speaker_suggestions(
    tmp_path, monkeypatch
):
    configure_test_app(tmp_path, monkeypatch, concurrency=1)
    monkeypatch.setenv("SPEAKER_EMBEDDING_MODEL", "test-embedding")
    monkeypatch.setenv("SPEAKER_MATCH_AUTO_THRESHOLD", "0.98")
    monkeypatch.setenv("SPEAKER_MATCH_SUGGEST_THRESHOLD", "0.9")
    profile = persistence.create_speaker_profile("Alice Global", profile_id="alice")
    persistence.save_speaker_embedding(
        profile["profile_id"],
        [1.0, 0.0],
        model_name="test-embedding",
        quality=1.0,
    )

    def fake_transcribe(file_path, models, progress_callback=None):
        return SimpleNamespace(
            transcript=[
                {
                    "speaker": "SPEAKER_00",
                    "text": "Hallo",
                    "start": 0.0,
                    "end": 1.0,
                }
            ],
            audio_duration_seconds=1.0,
            speaker_embeddings=[
                LocalSpeakerEmbedding(
                    local_speaker_id="SPEAKER_00",
                    embedding=[0.95, 0.05],
                    model_name="test-embedding",
                    quality=0.9,
                    quality_metadata={"total_seconds": 9.0},
                )
            ],
        )

    monkeypatch.setattr(main, "transcribe_audio", fake_transcribe)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/transcribe",
            data={
                "session_id": "session-speaker-match",
                "remember_speakers": "true",
            },
            files={"audio": ("meeting.mp3", b"audio", "audio/mpeg")},
        )
        assert response.status_code == 200
        job_id = response.json()["job_id"]

        assert wait_until(
            lambda: client.get(f"/api/transcribe/{job_id}").json()["status"]
            == "completed"
        )
        body = client.get(f"/api/transcribe/{job_id}").json()

    assert body["speaker_suggestions"] == [
        {
            "observation_id": 1,
            "local_speaker_id": "SPEAKER_00",
            "profile_id": "alice",
            "profile_display_name": "Alice Global",
            "confidence": body["speaker_suggestions"][0]["confidence"],
            "status": "suggested",
        }
    ]
    assert body["speaker_suggestions"][0]["confidence"] >= 0.98
    observations = persistence.load_speaker_observations(
        job_id=job_id,
        session_id="session-speaker-match",
    )
    assert observations[0]["status"] == "suggested"


def test_transcription_does_not_suggest_global_speakers_without_opt_in(
    tmp_path, monkeypatch
):
    configure_test_app(tmp_path, monkeypatch, concurrency=1)
    monkeypatch.setenv("SPEAKER_EMBEDDING_MODEL", "test-embedding")
    profile = persistence.create_speaker_profile("Alice Global", profile_id="alice")
    persistence.save_speaker_embedding(
        profile["profile_id"],
        [1.0, 0.0],
        model_name="test-embedding",
        quality=1.0,
    )

    monkeypatch.setattr(
        main,
        "transcribe_audio",
        lambda file_path, models, progress_callback=None: SimpleNamespace(
            transcript=[
                {
                    "speaker": "SPEAKER_00",
                    "text": "Hallo",
                    "start": 0.0,
                    "end": 1.0,
                }
            ],
            audio_duration_seconds=1.0,
            speaker_embeddings=[
                LocalSpeakerEmbedding(
                    local_speaker_id="SPEAKER_00",
                    embedding=[1.0, 0.0],
                    model_name="test-embedding",
                    quality=0.9,
                )
            ],
        ),
    )

    with TestClient(main.app) as client:
        response = client.post(
            "/api/transcribe",
            data={"session_id": "session-speaker-match"},
            files={"audio": ("meeting.mp3", b"audio", "audio/mpeg")},
        )
        job_id = response.json()["job_id"]

        assert wait_until(
            lambda: client.get(f"/api/transcribe/{job_id}").json()["status"]
            == "completed"
        )
        body = client.get(f"/api/transcribe/{job_id}").json()

    assert body["speaker_suggestions"] in (None, [])
    assert persistence.load_speaker_observations(
        job_id=job_id,
        session_id="session-speaker-match",
    ) == []


def test_archived_speaker_profiles_are_not_suggested(tmp_path, monkeypatch):
    configure_test_app(tmp_path, monkeypatch, concurrency=1)
    monkeypatch.setenv("SPEAKER_EMBEDDING_MODEL", "test-embedding")
    profile = persistence.create_speaker_profile("Alice Global", profile_id="alice")
    persistence.save_speaker_embedding(
        profile["profile_id"],
        [1.0, 0.0],
        model_name="test-embedding",
        quality=1.0,
    )
    persistence.archive_speaker_profile(profile["profile_id"])

    monkeypatch.setattr(
        main,
        "transcribe_audio",
        lambda file_path, models, progress_callback=None: SimpleNamespace(
            transcript=[
                {
                    "speaker": "SPEAKER_00",
                    "text": "Hallo",
                    "start": 0.0,
                    "end": 1.0,
                }
            ],
            audio_duration_seconds=1.0,
            speaker_embeddings=[
                LocalSpeakerEmbedding(
                    local_speaker_id="SPEAKER_00",
                    embedding=[1.0, 0.0],
                    model_name="test-embedding",
                    quality=0.9,
                )
            ],
        ),
    )

    with TestClient(main.app) as client:
        response = client.post(
            "/api/transcribe",
            data={
                "session_id": "session-speaker-match",
                "remember_speakers": "true",
            },
            files={"audio": ("meeting.mp3", b"audio", "audio/mpeg")},
        )
        job_id = response.json()["job_id"]

        assert wait_until(
            lambda: client.get(f"/api/transcribe/{job_id}").json()["status"]
            == "completed"
        )
        body = client.get(f"/api/transcribe/{job_id}").json()

    assert body["speaker_suggestions"] in (None, [])
    assert persistence.load_speaker_observations(
        job_id=job_id,
        session_id="session-speaker-match",
    ) == []


def test_cancel_pending_job_marks_cancelled_and_cleans_upload(tmp_path, monkeypatch):
    configure_test_app(tmp_path, monkeypatch, concurrency=1)
    first_started = threading.Event()
    release_first = threading.Event()

    def fake_transcribe(file_path, models, progress_callback=None):
        first_started.set()
        assert release_first.wait(timeout=5)
        return FakeTranscriptionResult(transcript=[], audio_duration_seconds=0)

    monkeypatch.setattr(main, "transcribe_audio", fake_transcribe)

    with TestClient(main.app) as client:
        first = client.post(
            "/api/transcribe",
            files={"audio": ("first.mp3", b"one", "audio/mpeg")},
        ).json()["job_id"]
        second = client.post(
            "/api/transcribe",
            files={"audio": ("second.mp3", b"two", "audio/mpeg")},
        ).json()["job_id"]

        assert first_started.wait(timeout=5)
        pending_job = main.get_job_from_cache_or_db(second)
        pending_upload = Path(pending_job["file_path"])
        assert pending_upload.exists()

        cancel_response = client.post(f"/api/transcribe/{second}/cancel")

        assert cancel_response.status_code == 200
        assert cancel_response.json()["status"] == "cancelled"
        assert not pending_upload.exists()
        assert main.get_job_from_cache_or_db(second)["file_path"] is None

        release_first.set()
        assert wait_until(
            lambda: client.get(f"/api/transcribe/{first}").json()["status"]
            == "completed"
        )


def test_cancel_processing_job_is_terminal_and_retains_source_for_review(
    tmp_path, monkeypatch
):
    configure_test_app(tmp_path, monkeypatch, concurrency=1)
    started = threading.Event()

    def fake_transcribe(file_path, models, progress_callback=None):
        started.set()
        assert wait_until(
            lambda: any(
                job.get("cancellation_requested") for job in main.jobs.values()
            ),
            timeout=5,
        )
        if progress_callback:
            progress_callback(50, "Wird abgebrochen")
        return FakeTranscriptionResult(transcript=[], audio_duration_seconds=0)

    monkeypatch.setattr(main, "transcribe_audio", fake_transcribe)

    with TestClient(main.app) as client:
        job_id = client.post(
            "/api/transcribe",
            files={"audio": ("meeting.mp3", b"audio", "audio/mpeg")},
        ).json()["job_id"]

        assert started.wait(timeout=5)
        job = main.get_job_from_cache_or_db(job_id)
        upload_path = Path(job["file_path"])
        assert upload_path.exists()

        cancel_response = client.post(f"/api/transcribe/{job_id}/cancel")

        assert cancel_response.status_code == 200
        assert cancel_response.json()["status"] == "cancelled"
        assert upload_path.exists()
        final_job = client.get(f"/api/transcribe/{job_id}").json()
        assert final_job["status"] == "cancelled"
        assert final_job["audio_url"] == f"/api/audio/{job_id}"


def test_pdf_upload_validation_accepts_pdf_mime_or_extension():
    assert main.is_allowed_pdf_file("agenda.bin", "application/pdf")
    assert main.is_allowed_pdf_file("agenda.PDF", "application/octet-stream")


def test_extract_tops_endpoint_returns_metadata(tmp_path, monkeypatch):
    configure_test_app(tmp_path, monkeypatch)

    monkeypatch.setattr(
        main,
        "extract_agenda_data_from_pdf",
        lambda pdf_path, model=None, system_prompt=None, processing_mode=None: fake_pdf_result(
            tops=["Eröffnung", "Haushalt"],
            metadata=SimpleNamespace(
                to_dict=lambda: {
                    "committee": "Hauptausschuss",
                    "date": "2026-06-30",
                    "location": "Rathaus",
                    "title": "Sitzung Hauptausschuss",
                }
            ),
        ),
    )

    with TestClient(main.app) as client:
        response = client.post(
            "/api/extract-tops",
            data={"model": "fake-llm"},
            files={"pdf": ("agenda.pdf", b"%PDF-1.4", "application/pdf")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["tops"] == ["Eröffnung", "Haushalt"]
    assert body["metadata"]["committee"] == "Hauptausschuss"
    assert body["metadata"]["date"] == "2026-06-30"


def test_assignment_suggestions_endpoint_returns_reviewable_segments(agenda_model):
    agenda_model.labels = {0: [], 1: ["agenda:1"]}
    with TestClient(main.app) as client:
        response = client.post(
            "/api/assignment-suggestions",
            json={
                "tops": ["1. Begrüßung", "2. Haushalt"],
                "transcript": [
                    {
                        "speaker": "SPEAKER_00",
                        "text": "Ich eröffne die Sitzung.",
                        "start": 0,
                        "end": 3,
                    },
                    {
                        "speaker": "SPEAKER_00",
                        "text": "Kommen wir zu TOP 2 Haushalt.",
                        "start": 4,
                        "end": 8,
                    },
                ],
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["suggested_assignments"] == [None, 1]
    assert data["segments"][0]["confidence"] >= 0.7
    assert data["segments"][0]["reason"]


def test_agenda_detection_endpoint_detects_tops_without_pdf_or_manual_list(agenda_model):
    agenda_model.labels = {0: ['detected:0'], 1: ['detected:0'], 2: ['detected:1']}
    def inventory(body):
        rows = body['context']['original_transcript']
        return {'items': [{'title': title, 'number': number, 'section': None,
            'evidence': agenda_model.evidence(rows[index])} for title, number, index in [('Haushalt', '1', 0), ('Schulbau', '2', 2)]], 'reason': 'Belegte Agenda'}
    agenda_model.overrides.update({'primary:discover': inventory, 'independent:discover': inventory})
    with TestClient(main.app) as client:
        response = client.post(
            "/api/agenda-detection",
            json={
                "transcript": [
                    {
                        "speaker": "SPEAKER_00",
                        "text": "Kommen wir zu TOP 1 Haushalt.",
                        "start": 0,
                        "end": 3,
                    },
                    {
                        "speaker": "SPEAKER_01",
                        "text": "Der Haushalt wird beraten.",
                        "start": 4,
                        "end": 8,
                    },
                    {
                        "speaker": "SPEAKER_00",
                        "text": "Als nächstes rufe ich TOP 2 Schulbau auf.",
                        "start": 9,
                        "end": 12,
                    },
                ],
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["tops"] == ["1. Haushalt", "2. Schulbau"]
    assert data["assignments"] == [0, 0, 1]
    assert data["segments"][0]["evidence_text"] == "Kommen wir zu TOP 1 Haushalt."
    assert data["strategy"] == "model_agenda_v1"


@pytest.mark.parametrize("tops", [[], ["Haushalt", "Schulbau"]])
def test_agenda_detection_endpoint_retains_short_tail_on_repeated_requests(tops):
    text = (
        "Dies ist ein ausreichend langer erster Satz mit Erläuterungen zur bisherigen Beratung. "
        "Danach folgt ein weiterer längerer Satz zum gleichen Thema. TOP 2."
    )
    source = [{"line_id": "source-line", "speaker": "SPEAKER_04", "text": text,
               "start": 10.0, "end": 20.0}]
    with TestClient(main.app) as client:
        response = client.post("/api/agenda-detection", json={
            "transcript": source, "tops": tops, "use_llm": False,
        })

        assert response.status_code == 200
        data = response.json()
        lines = data["transcript"]
        assert len(lines) == 3
        assert lines[-1]["text"] == "TOP 2."
        assert " ".join(line["text"] for line in lines) == text
        assert lines[0]["line_id"] == "source-line"
        assert len(data["assignments"]) == len(lines)

        repeated = client.post("/api/agenda-detection", json={
            "transcript": lines, "tops": tops, "use_llm": False,
        })
        assert repeated.status_code == 200
        assert repeated.json()["transcript"] == lines
        assert repeated.json()["assignments"] == data["assignments"]

def test_agenda_detection_endpoint_splits_mid_utterance_top_transition(agenda_model):
    agenda_model.labels = {i: ["agenda:0" if i < 3 else "agenda:2"] for i in range(5)}
    with TestClient(main.app) as client:
        response = client.post(
            "/api/agenda-detection",
            json={
                    "tops": [
                        "1. Eröffnung",
                        "2. Protokoll der letzten Sitzung",
                        "3. Verpflichtung Herr Krull",
                    ],
                "transcript": [
                    {
                        "speaker": "SPEAKER_04",
                        "text": "Ich eröffne die Sitzung.",
                        "start": 0,
                        "end": 2,
                    },
                    {
                        "speaker": "SPEAKER_04",
                        "text": (
                            "Das Protokoll wird geändert. Das ist einstimmig. "
                            "Kommen wir zum Tagesordnungspunkt 3. "
                            "Wir haben heute Herrn Krull zu verpflichten."
                        ),
                        "start": 2,
                        "end": 12,
                    }
                ],
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert [line["text"] for line in data["transcript"]] == [
        "Ich eröffne die Sitzung.",
        "Das Protokoll wird geändert.",
        "Das ist einstimmig.",
        "Kommen wir zum Tagesordnungspunkt 3.",
        "Wir haben heute Herrn Krull zu verpflichten.",
    ]
    assert data["assignments"] == [0, 0, 0, 2, 2]
    assert data["segments"][1]["start_index"] == 3


def test_llm_diagnostics_endpoint_reports_configured_model(fake_openai_module):
    with TestClient(main.app) as client:
        response = client.get("/api/llm/diagnostics?model=test-model")

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["model"] == "test-model"
    assert data["model_available"] is True


def test_pipeline_runs_to_reviewable_result_and_persists_status_after_cache_clear(
    tmp_path, monkeypatch, agenda_model):
    agenda_model.labels = {0: ['agenda:0'], 1: ['agenda:0'], 2: ['agenda:1']}
    configure_test_app(tmp_path, monkeypatch, concurrency=1)

    def fake_transcribe(file_path, models, progress_callback=None):
        if progress_callback:
            progress_callback(40, "Transkription läuft")
        return FakeTranscriptionResult(
            transcript=[
                {
                    "speaker": "SPEAKER_00",
                    "text": "Ich eröffne TOP 1 Haushalt.",
                    "start": 0.0,
                    "end": 2.0,
                },
                {
                    "speaker": "SPEAKER_01",
                    "text": "Der Haushalt wird beraten.",
                    "start": 2.0,
                    "end": 5.0,
                },
                {
                    "speaker": "SPEAKER_00",
                    "text": "Kommen wir zu TOP 2 Schulbau.",
                    "start": 5.0,
                    "end": 7.0,
                },
            ],
            audio_duration_seconds=7.0,
        )

    def fake_summarize_segment(top_title, transcript_text, model=None, system_prompt=None, meeting_context=None, source_lines=None, processing_mode=None):
        structured = summarize.StructuredSummary(
            discussion=[f"{top_title} wurde beraten."]
        )
        return summarize.SummarizationResult(llm_usage={'processing_complete': True},
            summary=f"Zusammenfassung {top_title}",
            duration_seconds=0.01,
            structured=structured,
        )

    monkeypatch.setattr(main, "transcribe_audio", fake_transcribe)
    monkeypatch.setattr(main, "summarize_segment", fake_summarize_segment)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/pipeline/start",
            data={
                "session_id": "pipeline-session",
                "tops": json.dumps(["TOP 1 Haushalt", "TOP 2 Schulbau"]),
                "options": json.dumps({"model": "fake-llm"}),
            },
            files={"audio": ("meeting.mp3", b"audio", "audio/mpeg")},
        )

        assert response.status_code == 200
        pipeline_id = response.json()["pipeline_id"]

        assert wait_until(
            lambda: client.get(f"/api/pipeline/{pipeline_id}").json()["stage"]
            == "ready_for_review"
        )
        status = client.get(f"/api/pipeline/{pipeline_id}").json()
        assert status["status"] == "completed"
        assert status["progress"] == 100

        main.jobs.clear()
        restored_status = client.get(f"/api/pipeline/{pipeline_id}").json()
        assert restored_status["stage"] == "ready_for_review"

        result = client.get(f"/api/pipeline/{pipeline_id}/result").json()
        session = result["session"]
        assert session["session_id"] == "pipeline-session"
        assert session["job"]["status"] == "completed"
        assert session["tops"] == ["TOP 1 Haushalt", "TOP 2 Schulbau"]
        assert session["assignments"] == [0, 0, 1]
        assert session["summaries"]["0"] == "Zusammenfassung TOP 1 Haushalt"
        assert session["summary_reviews"]["0"]["structured"]["discussion"]
        assert session["audio_url"].startswith("/api/audio/")
        assert result["agenda_detection"]["tops"] == [
            "TOP 1 Haushalt",
            "TOP 2 Schulbau",
        ]
        assert result["agenda_detection"]["assignments"] == [0, 0, 1]
        assert result["agenda_detection"]["uncertain_count"] == 0
        assert result["agenda_detection"]["segments"][0]["top_title"] == "TOP 1 Haushalt"

        session["summaries"]["0"] = "Manuell korrigierte Zusammenfassung."
        save_response = client.put(
            "/api/sessions/pipeline-session",
            json={
                "job_id": session["job_id"],
                "current_step": session["current_step"],
                "tops": session["tops"],
                "transcript": session["transcript"],
                "assignments": session["assignments"],
                "speaker_names": session["speaker_names"],
                "summaries": session["summaries"],
                "summary_reviews": session["summary_reviews"],
                "export_metadata": {
                    "committee": "Hauptausschuss",
                    "date": "2026-07-01",
                    "location": "Rathaus",
                    "title": "Pipeline-Protokoll",
                    "participants": [],
                },
                "skipped_assignment": False,
            },
        )
        assert save_response.status_code == 200

        export_response = client.post(
            "/api/export",
            json={
                "format": "txt",
                "metadata": {
                    "committee": "Hauptausschuss",
                    "date": "2026-07-01",
                    "location": "Rathaus",
                    "title": "Pipeline-Protokoll",
                    "participants": [],
                },
                "appendix": {
                    "include_speaker_list": True,
                    "include_transcript_excerpt": False,
                    "include_generation_note": True,
                },
                "tops": session["tops"],
                "transcript": session["transcript"],
                "assignments": session["assignments"],
                "speaker_names": session["speaker_names"],
                "summaries": session["summaries"],
                "summary_reviews": session["summary_reviews"],
            },
        )
        assert export_response.status_code == 200
        assert "Manuell korrigierte Zusammenfassung." in export_response.text


def test_selective_summary_job_updates_requested_top_with_speaker_names(
    tmp_path, monkeypatch
):
    configure_test_app(tmp_path, monkeypatch, concurrency=1)
    captured = []
    persistence.save_session(
        "summary-session",
        {
            "job_id": None,
            "current_step": 2,
            "tops": ["Haushalt", "Schulbau"],
            "transcript": [
                {
                    "speaker": "SPEAKER_00",
                    "text": "Der Haushalt wird beraten.",
                    "start": 0.0,
                    "end": 2.0,
                },
                {
                    "speaker": "SPEAKER_01",
                    "text": "Der Schulbau bleibt unverändert.",
                    "start": 2.0,
                    "end": 4.0,
                },
            ],
            "assignments": [0, 1],
            "speaker_names": {"SPEAKER_00": "Frau Beispiel"},
            "summaries": {0: "Alt.", 1: "Schulbau bleibt bestehen."},
            "summary_reviews": {},
            "skipped_assignment": False,
            "export_metadata": {"title": "Sitzung"},
        },
    )

    def fake_summarize_segment(top_title, transcript_text, model=None, system_prompt=None, meeting_context=None, source_lines=None, processing_mode=None):
        captured.append(
            {
                "top_title": top_title,
                "transcript_text": transcript_text,
                "model": model,
                "system_prompt": system_prompt,
            }
        )
        return summarize.SummarizationResult(llm_usage={'processing_complete': True},
            summary=f"Neu: {top_title}",
            duration_seconds=0.01,
            structured=summarize.StructuredSummary(
                discussion=["Der Haushalt wurde beraten."]
            ),
        )

    monkeypatch.setattr(main, "summarize_segment", fake_summarize_segment)

    with TestClient(main.app) as client:
        session = client.get("/api/sessions/summary-session").json()
        response = client.post(
            "/api/sessions/summary-session/summary-jobs",
            json={
                "revision": session["revision"],
                "top_ids": [session["top_ids"][0]],
                "model": "fake-llm",
                "system_prompt": "Kurz",
            },
        )
        assert response.status_code == 200
        summary_job_id = response.json()["summary_job_id"]
        assert wait_until(
            lambda: client.get(f"/api/summary-jobs/{summary_job_id}").json()["status"]
            == "completed"
        )
        body = client.get("/api/sessions/summary-session").json()

    assert body["summaries"]["0"] == "Neu: Haushalt"
    assert body["summaries"]["1"] == "Schulbau bleibt bestehen."
    assert body["summary_states"]["0"]["status"] == "ready"
    assert body["summary_states"]["1"]["origin"] == "legacy"
    assert len(captured) == 1
    assert body["export_metadata"]["title"] == "Sitzung"
    assert captured[0]["transcript_text"] == (
        "Frau Beispiel: Der Haushalt wird beraten."
    )
    assert captured[0]["model"] == "fake-llm"
    assert captured[0]["system_prompt"] == "Kurz"


def test_speaker_rename_updates_summary_without_llm(tmp_path, monkeypatch):
    configure_test_app(tmp_path, monkeypatch, concurrency=1)
    transcript = [{
        "line_id": "line-1",
        "speaker": "SPEAKER_04",
        "text": "Der Bericht wird vorgestellt.",
        "start": 0.0,
        "end": 2.0,
    }]
    states = main.build_generated_summary_states(
        session_id="rename-session",
        transcript=transcript,
        tops=["Bericht"],
        top_ids=["top-report"],
        assignments=[0],
        summaries={0: "SPEAKER_04 stellt den Bericht vor."},
        summary_reviews={},
        origin="pipeline",
    )
    saved = persistence.save_session("rename-session", {
        "tops": ["Bericht"],
        "top_ids": ["top-report"],
        "transcript": transcript,
        "assignments": [0],
        "speaker_names": {"SPEAKER_04": "SPEAKER_04"},
        "summaries": {0: "SPEAKER_04 stellt den Bericht vor."},
        "summary_reviews": {},
        "summary_states": states,
    })
    monkeypatch.setattr(
        main,
        "summarize_segment",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("LLM called")),
    )

    with TestClient(main.app) as client:
        response = client.put("/api/sessions/rename-session", json={
            "revision": saved["revision"],
            "tops": ["Bericht"],
            "top_ids": ["top-report"],
            "transcript": transcript,
            "assignments": [0],
            "speaker_names": {"SPEAKER_04": "Herr Reiche"},
            # The UI applies this deterministic replacement immediately.
            "summaries": {0: "Herr Reiche stellt den Bericht vor."},
            "summary_reviews": {},
            "summary_states": states,
        })

    assert response.status_code == 200
    body = response.json()
    assert body["summaries"]["0"] == "Herr Reiche stellt den Bericht vor."
    assert body["summary_states"]["0"]["status"] == "ready"
    assert body["summary_states"]["0"]["origin"] == "pipeline"


def test_assignment_change_marks_only_source_and_target_top(tmp_path, monkeypatch):
    configure_test_app(tmp_path, monkeypatch, concurrency=1)
    transcript = [
        {"line_id": "line-a", "speaker": "S1", "text": "A", "start": 0, "end": 1},
        {"line_id": "line-b", "speaker": "S2", "text": "B", "start": 1, "end": 2},
        {"line_id": "line-c", "speaker": "S3", "text": "C", "start": 2, "end": 3},
    ]
    tops = ["A", "B", "C"]
    top_ids = ["top-a", "top-b", "top-c"]
    summaries = {0: "A.", 1: "B.", 2: "C."}
    states = main.build_generated_summary_states(
        session_id="move-session",
        transcript=transcript,
        tops=tops,
        top_ids=top_ids,
        assignments=[0, 1, 2],
        summaries=summaries,
        summary_reviews={},
        origin="pipeline",
    )
    saved = persistence.save_session("move-session", {
        "tops": tops,
        "top_ids": top_ids,
        "transcript": transcript,
        "assignments": [0, 1, 2],
        "speaker_names": {},
        "summaries": summaries,
        "summary_reviews": {},
        "summary_states": states,
    })

    with TestClient(main.app) as client:
        response = client.put("/api/sessions/move-session", json={
            "revision": saved["revision"],
            "tops": tops,
            "top_ids": top_ids,
            "transcript": transcript,
            "assignments": [1, 1, 2],
            "speaker_names": {},
            "summaries": summaries,
            "summary_reviews": {},
            "summary_states": states,
        })

    assert response.status_code == 200
    statuses = response.json()["summary_states"]
    assert statuses["0"]["status"] == "review_required"
    assert statuses["1"]["status"] == "review_required"
    assert statuses["2"]["status"] == "ready"


def test_shared_session_history_and_conflict_response(tmp_path, monkeypatch):
    configure_test_app(tmp_path, monkeypatch)
    saved = persistence.save_session(
        "shared-session",
        {
            "current_step": 3,
            "tops": ["Haushalt"],
            "transcript": [
                {
                    "speaker": "SPEAKER_00",
                    "text": "Beratung",
                    "start": 0.0,
                    "end": 1.0,
                }
            ],
            "assignments": [0],
            "summaries": {0: "Zusammenfassung"},
            "export_metadata": {
                "title": "Öffentliche interne Sitzung",
                "committee": "Hauptausschuss",
                "date": "2026-07-13",
            },
        },
    )

    with TestClient(main.app) as client:
        history_response = client.get("/api/sessions?query=Hauptausschuss")
        assert history_response.status_code == 200
        history = history_response.json()
        assert history["total"] == 1
        assert history["items"][0]["session_id"] == "shared-session"
        assert history["items"][0]["status"] == "ready"

        update_response = client.put(
            "/api/sessions/shared-session",
            json={
                "revision": saved["revision"],
                "current_step": 3,
                "tops": ["Haushalt"],
                "transcript": saved["transcript"],
                "assignments": [0],
                "summaries": {"0": "Korrigiert"},
                "export_metadata": saved["export_metadata"],
                "skipped_assignment": False,
            },
        )
        assert update_response.status_code == 200

        conflict_response = client.put(
            "/api/sessions/shared-session",
            json={
                "revision": saved["revision"],
                "current_step": 1,
                "tops": [],
                "skipped_assignment": False,
            },
        )
        assert conflict_response.status_code == 409
        assert conflict_response.json()["detail"]["actual_revision"] > saved["revision"]


def test_pipeline_uses_pdf_tops_when_auto_pdf_mode_is_enabled(tmp_path, monkeypatch, agenda_model):
    agenda_model.labels = {0: ['agenda:0'], 1: ['agenda:1']}
    configure_test_app(tmp_path, monkeypatch, concurrency=1)
    extracted_calls = []

    def fake_transcribe(file_path, models, progress_callback=None):
        return FakeTranscriptionResult(
            transcript=[
                {
                    "speaker": "SPEAKER_00",
                    "text": "Ich eröffne TOP 1 Haushalt.",
                    "start": 0.0,
                    "end": 2.0,
                },
                {
                    "speaker": "SPEAKER_00",
                    "text": "Kommen wir zu TOP 2 Schulbau.",
                    "start": 2.0,
                    "end": 4.0,
                },
            ],
            audio_duration_seconds=4.0,
        )

    def fake_extract_agenda_data_from_pdf(pdf_path, model=None, system_prompt=None, processing_mode=None):
        extracted_calls.append(
            {
                "pdf_path": pdf_path,
                "model": model,
                "system_prompt": system_prompt,
            }
        )
        return fake_pdf_result(
            tops=["TOP 1 Haushalt", "TOP 2 Schulbau"],
            metadata=SimpleNamespace(
                to_dict=lambda: {
                    "committee": "Hauptausschuss",
                    "date": "2026-07-01",
                    "location": "Rathaus",
                    "title": "Sitzung Hauptausschuss",
                }
            ),
        )

    def fake_summarize_segment(top_title, transcript_text, model=None, system_prompt=None, meeting_context=None, source_lines=None, processing_mode=None):
        return summarize.SummarizationResult(llm_usage={'processing_complete': True},
            summary=f"Zusammenfassung {top_title}",
            duration_seconds=0.01,
            structured=summarize.StructuredSummary(
                discussion=[f"{top_title} wurde beraten."]
            ),
        )

    monkeypatch.setattr(main, "transcribe_audio", fake_transcribe)
    monkeypatch.setattr(main, "extract_agenda_data_from_pdf", fake_extract_agenda_data_from_pdf)
    monkeypatch.setattr(main, "summarize_segment", fake_summarize_segment)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/pipeline/start",
            data={
                "session_id": "pipeline-pdf-session",
                "model": "fake-llm",
                "auto_detect_tops_from_pdf": "true",
            },
            files={
                "audio": ("meeting.mp3", b"audio", "audio/mpeg"),
                "pdf": ("agenda.pdf", b"%PDF-1.4", "application/pdf"),
            },
        )

        assert response.status_code == 200
        pipeline_id = response.json()["pipeline_id"]

        assert wait_until(
            lambda: client.get(f"/api/pipeline/{pipeline_id}").json()["stage"]
            == "ready_for_review"
        )
        result = client.get(f"/api/pipeline/{pipeline_id}/result").json()

    assert extracted_calls
    assert extracted_calls[0]["model"] == "fake-llm"
    assert result["session"]["tops"] == ["TOP 1 Haushalt", "TOP 2 Schulbau"]
    assert result["session"]["export_metadata"]["committee"] == "Hauptausschuss"
    assert result["session"]["export_metadata"]["date"] == "2026-07-01"
    assert result["session"]["export_metadata"]["location"] == "Rathaus"
    assert result["session"]["assignments"] == [0, 1]
    assert result["session"]["summaries"]["0"] == "Zusammenfassung TOP 1 Haushalt"
    assert result["agenda_detection"]["tops"] == ["TOP 1 Haushalt", "TOP 2 Schulbau"]
    assert result["agenda_detection"]["assignments"] == [0, 1]
    assert result["agenda_detection"]["segments"][0]["top_title"] == "TOP 1 Haushalt"
    assert result["agenda_detection"]["uncertain_count"] == 0


def test_pipeline_keeps_known_tops_when_pdf_auto_mode_is_stale(tmp_path, monkeypatch):
    configure_test_app(tmp_path, monkeypatch, concurrency=1)
    extracted_calls = []

    def fake_transcribe(file_path, models, progress_callback=None):
        return FakeTranscriptionResult(
            transcript=[
                {
                    "speaker": "SPEAKER_00",
                    "text": "TOP 1 Eröffnung.",
                    "start": 0.0,
                    "end": 1.0,
                },
                {
                    "speaker": "SPEAKER_00",
                    "text": "TOP 2 Haushalt.",
                    "start": 1.0,
                    "end": 2.0,
                },
            ],
            audio_duration_seconds=2.0,
        )

    def fake_extract_agenda_data_from_pdf(pdf_path, model=None, system_prompt=None, processing_mode=None):
        extracted_calls.append(pdf_path)
        return fake_pdf_result(
            tops=["TOP I. Öffentlicher Teil", "01 Eröffnung", "02 Haushalt"],
            metadata=SimpleNamespace(to_dict=lambda: {"committee": "Nicht genutzt"}),
        )

    def fake_summarize_segment(top_title, transcript_text, model=None, system_prompt=None, meeting_context=None, source_lines=None, processing_mode=None):
        return summarize.SummarizationResult(llm_usage={'processing_complete': True},
            summary=f"Zusammenfassung {top_title}",
            duration_seconds=0.01,
            structured=summarize.StructuredSummary(
                discussion=[f"{top_title} wurde beraten."]
            ),
        )

    monkeypatch.setattr(main, "transcribe_audio", fake_transcribe)
    monkeypatch.setattr(main, "extract_agenda_data_from_pdf", fake_extract_agenda_data_from_pdf)
    monkeypatch.setattr(main, "summarize_segment", fake_summarize_segment)

    known_tops = ["Eröffnung", "Haushalt"]
    with TestClient(main.app) as client:
        response = client.post(
            "/api/pipeline/start",
            data={
                "session_id": "pipeline-known-tops-session",
                "tops": json.dumps(known_tops),
                "auto_detect_tops_from_pdf": "true",
            },
            files={
                "audio": ("meeting.mp3", b"audio", "audio/mpeg"),
                "pdf": ("agenda.pdf", b"%PDF-1.4", "application/pdf"),
            },
        )

        assert response.status_code == 200
        pipeline_id = response.json()["pipeline_id"]

        assert wait_until(
            lambda: client.get(f"/api/pipeline/{pipeline_id}").json()["status"]
            == "failed"
        )
        result = client.get(f"/api/pipeline/{pipeline_id}/result").json()

    assert extracted_calls == []
    assert result["session"]["tops"] == known_tops
    assert result["agenda_detection"]["tops"] == known_tops


def test_pipeline_failure_keeps_technical_gap_without_invented_agenda(
    tmp_path, monkeypatch
):
    configure_test_app(tmp_path, monkeypatch, concurrency=1)

    monkeypatch.setattr(
        main,
        "transcribe_audio",
        lambda file_path, models, progress_callback=None: FakeTranscriptionResult(
            transcript=[
                {
                    "speaker": "SPEAKER_00",
                    "text": "Wir beraten ohne erkennbare Tagesordnung.",
                    "start": 0.0,
                    "end": 3.0,
                }
            ],
            audio_duration_seconds=3.0,
        ),
    )
    monkeypatch.setattr(
        main,
        "detect_agenda_from_transcript",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("agenda kaputt")),
    )
    monkeypatch.setattr(
        main,
        "summarize_segment",
        lambda *args, **kwargs: summarize.SummarizationResult(llm_usage={'processing_complete': True},
            summary="Gesamtes Gespräch wurde zusammengefasst.",
            duration_seconds=0.01,
            structured=None,
            fallback_used=True,
        ),
    )

    with TestClient(main.app) as client:
        response = client.post(
            "/api/pipeline/start",
            files={"audio": ("meeting.mp3", b"audio", "audio/mpeg")},
        )
        pipeline_id = response.json()["pipeline_id"]

        assert wait_until(
            lambda: client.get(f"/api/pipeline/{pipeline_id}").json()["status"]
            == "failed"
        )
        result = client.get(f"/api/pipeline/{pipeline_id}/result").json()

    assert result["session"]["tops"] == []
    assert result["session"]["assignments"] == [None]
    assert result["warnings"]
    assert result["agenda_detection"]["tops"] == []
    assert result["agenda_detection"]["assignments"] == [None]
    assert result["agenda_detection"]["uncertain_count"] == 0
    assert result["agenda_detection"]["strategy"] == "known_agenda_failed"


def test_pipeline_without_tops_stays_on_review_step_for_speaker_assignment(
    tmp_path, monkeypatch
):
    configure_test_app(tmp_path, monkeypatch, concurrency=1)

    monkeypatch.setattr(
        main,
        "transcribe_audio",
        lambda file_path, models, progress_callback=None: FakeTranscriptionResult(
            transcript=[
                {
                    "speaker": "SPEAKER_00",
                    "text": "Wir beraten ohne Tagesordnung.",
                    "start": 0.0,
                    "end": 3.0,
                }
            ],
            audio_duration_seconds=3.0,
        ),
    )
    monkeypatch.setattr(
        main,
        "summarize_segment",
        lambda *args, **kwargs: summarize.SummarizationResult(llm_usage={'processing_complete': True},
            summary="Gesamtes Gespräch wurde zusammengefasst.",
            duration_seconds=0.01,
            structured=None,
        ),
    )

    with TestClient(main.app) as client:
        response = client.post(
            "/api/pipeline/start",
            data={"skip_agenda_detection": "true"},
            files={"audio": ("meeting.mp3", b"audio", "audio/mpeg")},
        )
        pipeline_id = response.json()["pipeline_id"]

        assert wait_until(
            lambda: client.get(f"/api/pipeline/{pipeline_id}").json()["stage"]
            == "ready_for_review"
        )
        result = client.get(f"/api/pipeline/{pipeline_id}/result").json()

    assert result["session"]["current_step"] == 2
    assert result["session"]["skipped_assignment"] is True
    assert result["session"]["tops"] == []
    assert result["session"]["assignments"] == [None]
    assert result["session"]["summaries"]["0"] == (
        "Gesamtes Gespräch wurde zusammengefasst."
    )


def test_pipeline_result_exposes_agenda_segments_and_uncertainty(
    tmp_path, monkeypatch
):
    configure_test_app(tmp_path, monkeypatch, concurrency=1)
    persistence.save_session(
        "session-agenda",
        {
            "job_id": None,
            "current_step": 3,
            "tops": ["TOP 1 Haushalt", "TOP 2 Schulbau"],
            "transcript": [
                {
                    "speaker": "SPEAKER_00",
                    "text": "TOP 1 Haushalt.",
                    "start": 0.0,
                    "end": 1.0,
                },
                {
                    "speaker": "SPEAKER_00",
                    "text": "Vielleicht TOP 2 Schulbau.",
                    "start": 1.0,
                    "end": 2.0,
                },
            ],
            "assignments": [0, 1],
            "speaker_names": {},
            "summaries": {0: "Haushalt.", 1: "Schulbau."},
            "summary_reviews": {},
            "skipped_assignment": False,
        },
    )
    persistence.save_pipeline_job(
        "pipeline-agenda",
        {
            "session_id": "session-agenda",
            "transcription_job_id": None,
            "status": "completed",
            "stage": "ready_for_review",
            "progress": 100,
            "result_refs": {
                "warnings": [],
                "agenda": {
                    "strategy": "llm_repaired",
                    "uncertain_count": 1,
                    "segments": [
                        {
                            "top_index": 0,
                            "top_title": "TOP 1 Haushalt",
                            "start_index": 0,
                            "end_index": 0,
                            "confidence": 0.92,
                            "uncertain": False,
                            "transition_type": "explicit",
                            "reason": "Explizite TOP-Nennung",
                            "evidence_index": 0,
                            "evidence_text": "TOP 1 Haushalt.",
                        },
                        {
                            "top_index": 1,
                            "top_title": "TOP 2 Schulbau",
                            "start_index": 1,
                            "end_index": 1,
                            "confidence": 0.42,
                            "uncertain": True,
                            "transition_type": "repaired",
                            "reason": "Segment wurde repariert und muss geprüft werden",
                            "evidence_index": 1,
                            "evidence_text": "Vielleicht TOP 2 Schulbau.",
                        },
                    ],
                },
            },
        },
    )

    with TestClient(main.app) as client:
        response = client.get("/api/pipeline/pipeline-agenda/result")

    assert response.status_code == 200
    agenda_detection = response.json()["agenda_detection"]
    # Legacy indices cannot be rebound to the current editing state.
    assert response.json()["session"]["agenda_proposals"]["source"] is None
    assert agenda_detection["tops"] == []
    assert agenda_detection["assignments"] == []
    assert agenda_detection["uncertain_count"] == 1
    assert agenda_detection["segments"][1]["uncertain"] is True


def test_pipeline_marks_failed_top_summary_but_stays_reviewable(
    tmp_path, monkeypatch, agenda_model):
    agenda_model.labels = {0: ['agenda:0'], 1: ['agenda:1']}
    configure_test_app(tmp_path, monkeypatch, concurrency=1)

    monkeypatch.setattr(
        main,
        "transcribe_audio",
        lambda file_path, models, progress_callback=None: FakeTranscriptionResult(
            transcript=[
                {
                    "speaker": "SPEAKER_00",
                    "text": "Kommen wir zu TOP 1 Haushalt.",
                    "start": 0.0,
                    "end": 2.0,
                },
                {
                    "speaker": "SPEAKER_00",
                    "text": "Kommen wir zu TOP 2 Fehler.",
                    "start": 2.0,
                    "end": 4.0,
                },
            ],
            audio_duration_seconds=4.0,
        ),
    )

    def maybe_fail_summary(top_title, transcript_text, model=None, system_prompt=None, meeting_context=None, source_lines=None, processing_mode=None):
        if "Fehler" in top_title:
            raise RuntimeError("LLM nicht verfügbar")
        return summarize.SummarizationResult(llm_usage={'processing_complete': True},
            summary="Haushalt wurde zusammengefasst.",
            duration_seconds=0.01,
            structured=None,
        )

    monkeypatch.setattr(main, "summarize_segment", maybe_fail_summary)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/pipeline/start",
            data={"tops": json.dumps(["TOP 1 Haushalt", "TOP 2 Fehler"])},
            files={"audio": ("meeting.mp3", b"audio", "audio/mpeg")},
        )
        pipeline_id = response.json()["pipeline_id"]

        assert wait_until(
            lambda: client.get(f"/api/pipeline/{pipeline_id}").json()["status"]
            == "failed"
        )
        result = client.get(f"/api/pipeline/{pipeline_id}/result").json()

    assert result["pipeline"]["status"] == "failed"
    assert result["session"]["summaries"]["0"] == "Haushalt wurde zusammengefasst."
    assert result["session"]["summaries"]["1"] == ""
    assert result["session"]["summary_reviews"]["1"]["review_warnings"][0]["severity"] == "error"
    assert result["warnings"]
    assert "LLM nicht verfügbar" not in json.dumps(result, ensure_ascii=False)


@pytest.mark.parametrize('grounding_incomplete', [False, True])
def test_pipeline_persists_incomplete_source_check_without_losing_draft(tmp_path, monkeypatch, grounding_incomplete, agenda_model):
    configure_test_app(tmp_path, monkeypatch)
    monkeypatch.setattr(main, 'transcribe_audio', lambda *args, **kwargs: FakeTranscriptionResult(
        transcript=[{'speaker': 'MOD', 'text': 'Kommen wir zu TOP 1 Haushalt.', 'start': 0, 'end': 2}],
        audio_duration_seconds=2))
    monkeypatch.setattr(main, 'summarize_segment', lambda *args, **kwargs: summarize.SummarizationResult(
        summary='Entwurf bleibt prüfbar.', duration_seconds=0.01,
        llm_usage={'grounding_incomplete': grounding_incomplete, 'processing_complete': not grounding_incomplete}))
    with TestClient(main.app) as client:
        started = client.post('/api/pipeline/start', data={
            'tops': json.dumps(['1 Haushalt']), 'agenda_use_llm': 'true',
        }, files={'audio': ('meeting.mp3', b'audio', 'audio/mpeg')})
        pipeline_id = started.json()['pipeline_id']
        assert wait_until(lambda: client.get(f'/api/pipeline/{pipeline_id}').json()['status'] == ('failed' if grounding_incomplete else 'completed'))
        result = client.get(f'/api/pipeline/{pipeline_id}/result').json()
        session = client.get('/api/sessions/' + result['session']['session_id']).json()
    assert main.load_pipeline_job(pipeline_id)['result_refs']['processing_complete'] is not grounding_incomplete
    assert session['summaries']['0'] == 'Entwurf bleibt prüfbar.'
    assert session['summary_reviews']['0']['llm_usage']['grounding_incomplete'] is grounding_incomplete
    assert session['summary_states']['0']['status'] == ('failed' if grounding_incomplete else 'ready')


def test_pipeline_fails_clearly_when_transcription_fails(tmp_path, monkeypatch):
    configure_test_app(tmp_path, monkeypatch, concurrency=1)

    def fail_transcription(file_path, models, progress_callback=None):
        raise RuntimeError("Audioqualität zu schlecht")

    monkeypatch.setattr(main, "transcribe_audio", fail_transcription)

    with TestClient(main.app) as client:
        response = client.post(
            "/api/pipeline/start",
            files={"audio": ("meeting.mp3", b"audio", "audio/mpeg")},
        )
        pipeline_id = response.json()["pipeline_id"]

        assert wait_until(
            lambda: client.get(f"/api/pipeline/{pipeline_id}").json()["status"]
            == "failed"
        )
        status = client.get(f"/api/pipeline/{pipeline_id}").json()

    assert status["stage"] == "transcribe"
    assert status["error"] == "RuntimeError"
    transcription_job_id = status["transcription_job_id"]
    assert persistence.load_job(transcription_job_id)["status"] == "failed"


def test_pipeline_restart_marks_active_pipeline_failed(tmp_path, monkeypatch):
    db_path = tmp_path / "sessions.sqlite3"
    monkeypatch.setenv("PERSISTENCE_DB_PATH", str(db_path))
    persistence.init_db()
    persistence.save_session(
        "session-1",
        {
            "tops": [],
            "assignments": [],
            "speaker_names": {},
            "summaries": {},
            "skipped_assignment": False,
        },
    )
    persistence.save_pipeline_job(
        "interrupted-pipeline",
        {
            "session_id": "session-1",
            "transcription_job_id": None,
            "status": "processing",
            "stage": "summarize",
            "progress": 82,
            "result_refs": {"warnings": []},
        },
    )

    persistence.mark_interrupted_pipeline_jobs()

    job = persistence.load_pipeline_job("interrupted-pipeline")
    assert job["status"] == "failed"
    assert job["progress"] == 0
    assert job["error"] == "Pipeline wurde durch Backend-Neustart unterbrochen"


def test_pipeline_cancel_endpoint_marks_pending_pipeline_cancelled(tmp_path, monkeypatch):
    configure_test_app(tmp_path, monkeypatch, concurrency=1)
    audio_path = tmp_path / "pending.mp3"
    audio_path.write_bytes(b"audio")

    with TestClient(main.app) as client:
        persistence.save_job(
            "cancel-transcription",
            {
                "session_id": "cancel-session",
                "status": "pending",
                "progress": 0,
                "message": "wartet",
                "file_path": str(audio_path),
                "audio_path": str(audio_path),
            },
        )
        persistence.save_pipeline_job(
            "cancel-pipeline",
            {
                "session_id": "cancel-session",
                "transcription_job_id": "cancel-transcription",
                "status": "pending",
                "stage": "upload",
                "progress": 5,
                "result_refs": {"audio_path": str(audio_path), "warnings": []},
            },
        )
        response = client.post("/api/pipeline/cancel-pipeline/cancel")

    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
    assert persistence.load_pipeline_job("cancel-pipeline")["result_refs"][
        "cancel_requested"
    ]


def test_cleanup_old_jobs_removes_expired_jobs_and_upload_files(tmp_path, monkeypatch):
    old_file_path = tmp_path / "old-upload.mp3"
    old_audio_path = tmp_path / "old-audio.mp3"
    fresh_file_path = tmp_path / "fresh-upload.mp3"
    old_file_path.write_bytes(b"old upload")
    old_audio_path.write_bytes(b"old audio")
    fresh_file_path.write_bytes(b"fresh upload")

    main.jobs.clear()
    monkeypatch.setattr(main, "JOB_MAX_AGE_SECONDS", 60)
    monkeypatch.setattr(main, "JOB_MAX_COUNT", 10)
    monkeypatch.setattr(main, "DELETE_UPLOADS_ON_JOB_CLEANUP", True)

    now = time.time()
    main.jobs["old"] = {
        "created_at": now - 120,
        "file_path": str(old_file_path),
        "audio_path": str(old_audio_path),
    }
    main.jobs["fresh"] = {
        "created_at": now,
        "file_path": str(fresh_file_path),
    }

    assert main.cleanup_old_jobs() == 1

    assert list(main.jobs) == ["fresh"]
    assert not old_file_path.exists()
    assert not old_audio_path.exists()
    assert fresh_file_path.exists()

    main.jobs.clear()


def test_cleanup_old_jobs_enforces_max_count(tmp_path, monkeypatch):
    main.jobs.clear()
    monkeypatch.setattr(main, "JOB_MAX_AGE_SECONDS", 3600)
    monkeypatch.setattr(main, "JOB_MAX_COUNT", 2)
    monkeypatch.setattr(main, "DELETE_UPLOADS_ON_JOB_CLEANUP", True)

    files = []
    for index in range(3):
        file_path = tmp_path / f"job-{index}.mp3"
        file_path.write_bytes(b"audio")
        files.append(file_path)
        main.jobs[f"job-{index}"] = {
            "created_at": time.time(),
            "file_path": str(file_path),
        }

    assert main.cleanup_old_jobs() == 1

    assert list(main.jobs) == ["job-1", "job-2"]
    assert not files[0].exists()
    assert files[1].exists()
    assert files[2].exists()

    main.jobs.clear()


@pytest.mark.parametrize("known_tops", [False, True])
@pytest.mark.parametrize("decision, default, enabled", [(None, False, False), (None, True, True), (False, True, False), (True, False, True)])
def test_agenda_api_explicit_llm_policy(monkeypatch, fake_openai_module, known_tops, decision, default, enabled):
    monkeypatch.setattr(agenda_detection, "AGENDA_DETECTION_USE_LLM", default)
    fake_openai_module.responses = [TimeoutError("SECRET")]
    # No lifespan or real models needed for the stateless detection endpoint.
    with TestClient(main.app) as client:
        payload = {
            "transcript": [{"speaker": "MOD", "text": "TOP 1 Haushalt.", "start": 0, "end": 1}],
            "tops": ["Haushalt"] if known_tops else [],
            "model": "test-model", "system_prompt": "custom", "use_llm": decision,
        }
        response = client.post("/api/agenda-detection", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["llm"]["enabled"] is enabled
        assert data["llm"]["status"] == ("failed" if enabled else "disabled")
        assert data["warnings"]
        assert "SECRET" not in response.text

@pytest.mark.parametrize("decision", ["false", "true", 1, 0, [], {}])
def test_agenda_json_rejects_non_boolean_decisions(decision):
    response = TestClient(main.app).post("/api/agenda-detection", json={"transcript": [], "use_llm": decision})
    assert response.status_code == 422
    with pytest.raises(main.HTTPException):
        main.parse_pipeline_options(json.dumps({"agenda_use_llm": decision}))


@pytest.mark.parametrize("form_value, option_value, server_default, enabled", [
    (None, None, False, False), (None, None, True, True),
    (None, False, True, False), (None, True, False, True),
    ("false", True, True, False), ("true", False, False, True),
])
def test_pipeline_llm_policy_and_persisted_fallback(
    tmp_path, monkeypatch, fake_openai_module, form_value, option_value, server_default, enabled,
):
    configure_test_app(tmp_path, monkeypatch)
    monkeypatch.setattr(agenda_detection, "AGENDA_DETECTION_USE_LLM", server_default)
    monkeypatch.setattr(main, "transcribe_audio", lambda *args, **kwargs: FakeTranscriptionResult(
        transcript=[{"speaker": "MOD", "text": "TOP 1 Haushalt.", "start": 0, "end": 1}],
        audio_duration_seconds=1,
    ))
    monkeypatch.setattr(main, "summarize_segment", lambda *args, **kwargs: summarize.SummarizationResult(llm_usage={'processing_complete': True},
        summary="Haushalt beraten.", duration_seconds=0.01,
        structured=summarize.StructuredSummary(discussion=["Haushalt beraten."]),
    ))
    fake_openai_module.responses = [TimeoutError("SECRET")]
    data = {"model": "test-model", "agenda_system_prompt": "custom", "options": json.dumps({"agenda_use_llm": option_value})}
    if form_value is not None:
        data["agenda_use_llm"] = form_value
    with TestClient(main.app) as client:
        response = client.post("/api/pipeline/start", data=data, files={"audio": ("test.mp3", b"fake", "audio/mpeg")})
        assert response.status_code == 200
        pipeline_id = response.json()["pipeline_id"]
        assert wait_until(lambda: client.get(f"/api/pipeline/{pipeline_id}").json()["status"] in {"completed", "failed"})
        main.jobs.clear()
        result = client.get(f"/api/pipeline/{pipeline_id}/result").json()
    usage = result["agenda_detection"]["llm"]
    assert usage["enabled"] is enabled
    assert usage["status"] == ("failed" if enabled else "disabled")
    assert usage["source"] == ("server_default" if option_value is None and form_value is None else "request")
    assert len(fake_openai_module.instances) == int(enabled)
    if enabled:
        assert result["agenda_detection"]["warnings"]
        assert result["warnings"]
    assert "SECRET" not in str(result)


def test_pdf_agenda_api_preserves_scope_and_numbers_through_assignment(tmp_path, monkeypatch, fake_openai_module, agenda_model):
    agenda_model.labels = {i: [f"agenda:{i//2}"] for i in range(8)}
    from io import BytesIO
    from reportlab.pdfgen.canvas import Canvas

    configure_test_app(tmp_path, monkeypatch)
    pdf = BytesIO()
    canvas = Canvas(pdf)
    for i, line in enumerate(['Tagesordnung', 'Öffentlicher Teil', '2 Haushalt', '2.1 Schulbau', '7 Anfragen', 'Nichtöffentlicher Teil', '2 Vergabe']):
        canvas.drawString(50, 750 - i * 25, line)
    canvas.save()
    monkeypatch.setenv('LLM_IMAGE_TOKENS', '1024')
    value = agenda(items=[item('a', number='2', section='public'),
        item('b', number='2.1', title='Schulbau', section='public', parent_id='a'),
        item('c', number='7', title='Anfragen', section='public'),
        item('d', number='2', title='Vergabe', section='nonpublic')])
    fake_openai_module.responses = [json.dumps(v) for v in [value, value, audit(), audit(0)]]
    with TestClient(main.app) as client:
        response = client.post('/api/extract-tops', files={'pdf': ('agenda.pdf', pdf.getvalue(), 'application/pdf')})
        assert response.status_code == 200
        tops = response.json()['tops']
        assert tops == ['[Öffentlich] 2 Haushalt', '[Öffentlich] 2.1 Schulbau', '[Öffentlich] 7 Anfragen', '[Nichtöffentlich] 2 Vergabe']
        response = client.post('/api/agenda-detection', json={
            'tops': tops, 'use_llm': True,
            'transcript': [{'speaker': 'MOD', 'text': text, 'start': i, 'end': i+1} for i, text in enumerate([
                'TOP 2 öffentlich Haushalt', 'Beratung', 'TOP 2.1 Schulbau', 'Beratung',
                'TOP sieben Anfragen', 'Beratung', 'TOP 2 Vergabe', 'Beratung',
            ])],
        })
        assert response.status_code == 200
        result = response.json()
        assert result['tops'] == tops
        assert not result['segments'][1]['uncertain']
        assert not result['segments'][2]['uncertain']
        assert result['assignments'][6:] == [3, 3]
        assert len(result['segments']) == 4

def test_pipeline_known_fallback_keeps_absence_of_evidence_visible(monkeypatch):
    transcript = [{'speaker': 'A', 'text': 'Allgemeine Diskussion.'}]
    tops = ['Begrüßung', 'Haushalt']
    actual_tops, assignments, info = main.fallback_agenda(transcript, tops)
    assert actual_tops == tops
    assert assignments == [None]
    assert info['segments'] == []

    def fail(*args, **kwargs):
        raise RuntimeError('unavailable')

    monkeypatch.setattr(main, 'suggest_assignments', fail)
    assert main.fallback_agenda(transcript, tops)[1] == [None]


@pytest.mark.parametrize('top_id,expected', [('agenda:1', 1), ('agenda:99', None)])
def test_agenda_api_preserves_identity_and_validation_diagnostics(fake_openai_module, top_id, expected):
    fake_openai_module.content = json.dumps({'tops': [{
        'top_id': top_id, 'top_title': 'Schulbau', 'start_index': 0, 'end_index': 0,
        'confidence': 0.99, 'evidence_text': 'SECRET invented quotation', 'evidence_index': 0,
    }]})
    with TestClient(main.app) as client:
        response = client.post('/api/agenda-detection', json={
            'transcript': [{'speaker': 'A', 'text': 'Allgemeine Diskussion.', 'start': 0, 'end': 1}],
            'tops': ['Haushalt', 'Schulbau'], 'use_llm': True,
        })
        assert response.status_code == 200
        data = response.json()
        assert data['tops'] == ['Haushalt', 'Schulbau']
        assert data['assignments'] == [None]
        assert data['llm']['gaps'][0]['kind'] == 'technical'
        assert data['warnings']
        assert data['llm']['status'] == 'failed'  # Invalid evidence is not a successful LLM result.
        assert 'SECRET' not in response.text
        assert data['segments'] == []

def test_known_agenda_total_failure_is_persisted_as_incomplete(tmp_path, monkeypatch, fake_openai_module):
    configure_test_app(tmp_path, monkeypatch)
    monkeypatch.setattr(main, 'transcribe_audio', lambda *args, **kwargs: FakeTranscriptionResult(
        transcript=[{'speaker': 'MOD', 'text': 'Kommen wir zu TOP 1 Haushalt.', 'start': 0, 'end': 2}],
        audio_duration_seconds=2))
    fake_openai_module.responses = [TimeoutError('SECRET')]
    with TestClient(main.app) as client:
        started = client.post('/api/pipeline/start', data={
            'tops': json.dumps(['1 Haushalt']), 'agenda_use_llm': 'true', 'model': 'test-model',
        }, files={'audio': ('meeting.mp3', b'audio', 'audio/mpeg')})
        pipeline_id = started.json()['pipeline_id']
        assert wait_until(lambda: client.get(f'/api/pipeline/{pipeline_id}').json()['status'] == 'failed')
        result = client.get(f'/api/pipeline/{pipeline_id}/result').json()
        session = client.get('/api/sessions/' + result['session']['session_id']).json()
    assert result['agenda_detection']['llm']['status'] == 'failed'
    assert session['assignments'] == [None]
    assert session['agenda_proposals']['result']['llm']['gaps'][0]['kind'] == 'technical'
    assert not main.load_pipeline_job(pipeline_id)['result_refs']['processing_complete']
    assert not session['summaries'].get('0')
    pipeline = main.load_pipeline_job(pipeline_id)
    assert pipeline['stage'] == 'agenda_detect' and pipeline['progress'] < 100
    with persistence.connect() as db:
        keys={row[0] for row in db.execute('SELECT step_key FROM durable_steps WHERE job_id=?',(pipeline_id,))}
    assert not {'pipeline:agenda','pipeline:summaries','pipeline:published'} & keys
    assert 'pipeline:draft:agenda' in keys
    assert result['warnings']
    assert 'SECRET' not in str(result)
    assert len(fake_openai_module.instances) == 1  # No summary over heuristic replacement text.


def fake_pdf_result(tops, metadata, **kwargs):
    return extract_tops.PdfAgendaExtractionResult(tops=tops,
        metadata=extract_tops.PdfSessionMetadata(**metadata.to_dict()),
        processing_complete=True, review_required=False, **kwargs)


def test_reuse_verified_pdf_keeps_source_and_ids_without_browser_file(tmp_path, monkeypatch, fake_openai_module):
    configure_test_app(tmp_path, monkeypatch)
    monkeypatch.setenv('LLM_IMAGE_TOKENS', '1024')
    fake_openai_module.responses = [json.dumps(v) for v in [agenda(), agenda(), audit(), audit(0)]]
    monkeypatch.setattr(main, 'transcribe_audio', lambda *a, **kw: FakeTranscriptionResult(
        transcript=[{'speaker': 'MOD', 'text': 'Allgemeine Beratung', 'start': 0, 'end': 1}], audio_duration_seconds=1))
    monkeypatch.setattr(main, 'summarize_segment', lambda *a, **kw: summarize.SummarizationResult(summary='Beratung', duration_seconds=0, llm_usage={'processing_complete': True}))
    with TestClient(main.app) as client:
        extracted = client.post('/api/extract-tops', files={'pdf': ('source.pdf', pdf_bytes(), 'application/pdf')}).json()
        response = client.post('/api/pipeline/start', files={'audio': ('a.mp3', b'audio', 'audio/mpeg')},
            data={'pdf_source_job_id': extracted['document']['job_id'], 'agenda_use_llm': 'false'})
        assert response.status_code == 200, response.text
        pipeline_id = response.json()['pipeline_id']
        assert wait_until(lambda: client.get(f'/api/pipeline/{pipeline_id}').json()['status'] == 'failed')
        session = client.get(f'/api/pipeline/{pipeline_id}/result').json()['session']
        assert session['tops'] == extracted['tops']
        assert session['top_ids'] == [i['id'] for i in extracted['items'] if i['kind'] == 'agenda']
        assert session['agenda_proposals']['source']['pdf_extraction'] == extracted
        assert client.get(extracted['document']['url']).status_code == 200
        assert main.durable.load(pipeline_id)['documents']
