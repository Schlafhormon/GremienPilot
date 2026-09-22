"""GPU handover tests with fake models and HTTP; no CUDA/server required."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace
import weakref

import httpx
import pytest

import gpu_resources as gpu
import llm_transport
from summarize import get_llm_config


@pytest.fixture(autouse=True)
def isolated_gpu(monkeypatch):
    monkeypatch.setenv("GPU_MODEL_SWITCHING", "true")
    monkeypatch.setenv("GPU_MODEL_UNLOAD_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("LLM_BASE_URL", "http://ollama:11434/v1")
    monkeypatch.setattr(gpu, "_release_error", False)
    monkeypatch.setattr(httpx, "get", lambda *a, **k: pytest.fail("Unexpected HTTP GET"))
    monkeypatch.setattr(httpx, "post", lambda *a, **k: pytest.fail("Unexpected HTTP POST"))


def response(method, url, payload, status=200):
    return httpx.Response(status, request=httpx.Request(method, url), json=payload)


def test_unload_waits_for_every_local_runner_and_verifies_absence(monkeypatch):
    states = iter([
        {"models": [{"model": "qwen3:8b"}, {"name": "other:latest"}]},
        {"models": [{"model": "qwen3:8b"}]},
        {"models": []},
    ])
    calls = []
    monkeypatch.setattr(httpx, "get", lambda url, **kw: response("GET", url, next(states)))
    def post(url, **kwargs):
        calls.append((url, kwargs))
        return response("POST", url, {"done": True})
    monkeypatch.setattr(httpx, "post", post)
    with gpu.gpu_slot():
        gpu.unload_local_ollama(get_llm_config())
    assert [item[1]["json"] for item in calls] == [
        {"model": "qwen3:8b", "keep_alive": 0, "stream": False},
        {"model": "other:latest", "keep_alive": 0, "stream": False},
    ]
    assert all(url == "http://ollama:11434/api/generate" for url, _ in calls)
    assert all(0 < kw["timeout"] <= 1 for _, kw in calls)


@pytest.mark.parametrize("payload", [{}, [], {"models": None}, {"models": [None]}, {"models": [{}]}])
def test_bad_status_never_counts_as_memory_release(monkeypatch, payload):
    monkeypatch.setattr(httpx, "get", lambda url, **kw: response("GET", url, payload))
    with pytest.raises(gpu.GPUResourceError):
        gpu.unload_local_ollama(get_llm_config())


def test_unload_timeout_does_not_claim_release(monkeypatch):
    monkeypatch.setenv("GPU_MODEL_UNLOAD_TIMEOUT_SECONDS", "0.03")
    monkeypatch.setattr(httpx, "get", lambda url, **kw: response("GET", url, {"models": [{"name": "busy"}]}))
    monkeypatch.setattr(httpx, "post", lambda url, **kw: response("POST", url, {"done": True}))
    with pytest.raises(gpu.GPUResourceError, match="Zeitlimit"):
        gpu.unload_local_ollama(get_llm_config())


def test_failed_unload_http_response_is_not_ignored(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda url, **kw: response("GET", url, {"models": [{"name": "busy"}]}))
    monkeypatch.setattr(httpx, "post", lambda url, **kw: response("POST", url, {}, 503))
    with pytest.raises(gpu.GPUResourceError, match="nicht bestaetigt"):
        gpu.unload_local_ollama(get_llm_config())


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "wrong"])
def test_invalid_unload_timeout(value, monkeypatch):
    monkeypatch.setenv("GPU_MODEL_UNLOAD_TIMEOUT_SECONDS", value)
    with pytest.raises(ValueError):
        gpu.unload_timeout()


def test_invalid_switch_setting(monkeypatch):
    monkeypatch.setenv("GPU_MODEL_SWITCHING", "maybe")
    with pytest.raises(ValueError, match="GPU_MODEL_SWITCHING"):
        gpu.switching_enabled()


def test_complete_waits_until_transcription_releases_gpu(monkeypatch):
    entered = threading.Event()
    attempted = threading.Event()
    def complete(*args, **kwargs):
        entered.set()
        return "answer"
    monkeypatch.setattr(llm_transport, "_complete", complete)
    def run():
        attempted.set()
        return llm_transport.complete(None, get_llm_config())
    with ThreadPoolExecutor(1) as pool:
        with gpu.gpu_slot():
            task = pool.submit(run)
            assert attempted.wait(2)
            assert not entered.wait(0.05)
        assert task.result(timeout=2) == "answer"


def test_external_llm_does_not_wait_or_get_unloaded(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://provider.example/v1")
    config = get_llm_config()
    gpu.unload_local_ollama(config)  # HTTP mocks fail if contacted.
    monkeypatch.setattr(llm_transport, "_complete", lambda *a, **k: "external")
    with ThreadPoolExecutor(1) as pool:
        with gpu.gpu_slot():
            assert pool.submit(llm_transport.complete, None, config).result(timeout=2) == "external"


def test_cancel_while_waiting_for_gpu_does_not_acquire_or_load():
    class Cancelled(Exception):
        pass
    def cancel():
        raise Cancelled()
    def run():
        with gpu.gpu_slot(cancel):
            pytest.fail("Cancelled worker entered GPU section")
    with ThreadPoolExecutor(1) as pool:
        with gpu.gpu_slot():
            task = pool.submit(run)
            with pytest.raises(Cancelled):
                task.result(timeout=2)
    with gpu.gpu_slot():
        pass  # Cancellation did not leave the lock held.


@pytest.fixture
def real_transcribe(monkeypatch):
    """Import actual lifecycle code, replacing only the heavyweight libraries."""
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: True, synchronize=lambda: None, empty_cache=lambda: None)
    whisperx = ModuleType("whisperx")
    diarize = ModuleType("whisperx.diarize")
    diarize.DiarizationPipeline = object
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "whisperx", whisperx)
    monkeypatch.setitem(sys.modules, "whisperx.diarize", diarize)
    name = "_transcribe_gpu_test"
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / "transcribe.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "WHISPER_DEVICE", "cuda")
    return module


class FakeGPUModel:
    pass


def install_loader(module, monkeypatch, events):
    refs = []
    def load():
        events.append("load")
        resources = [FakeGPUModel() for _ in range(4)]
        refs.extend(weakref.ref(item) for item in resources)
        return module.TranscriptionModels(
            resources[0], resources[1], {}, resources[2], "cuda",
            speaker_embedding_inference=resources[3],
            speaker_embedding_model_name="test-embedding",
        )
    monkeypatch.setattr(module, "_load_models", load)
    monkeypatch.setattr(module, "unload_local_ollama", lambda *a: events.append("unload-ollama"))
    def release():
        assert all(ref() is None for ref in refs)
        events.append("release")
    monkeypatch.setattr(module.torch.cuda, "empty_cache", release)
    return refs


def test_models_reload_for_each_job_and_release_before_llm(real_transcribe, monkeypatch):
    module = real_transcribe
    events = []
    install_loader(module, monkeypatch, events)
    holder = module.load_models()
    assert holder.gpu_managed and holder.whisper_model is None
    assert events == []  # Startup never loads models or contacts Ollama.
    def infer(path, models, callback):
        assert models.whisper_model is not None
        assert models.speaker_embedding_inference is not None
        events.append("infer")
        return module.TranscriptionResult([], 1)
    monkeypatch.setattr(module, "_transcribe_audio", infer)
    monkeypatch.setattr(llm_transport, "_complete", lambda *a, **kw: events.append("llm"))
    for _ in range(2):
        module.transcribe_audio("test.wav", holder)
        assert holder.whisper_model is None
        assert holder.speaker_embedding_inference is None
        assert holder.speaker_embedding_model_name == "test-embedding"
        llm_transport.complete(None, get_llm_config())
    assert events == ["unload-ollama", "load", "infer", "release", "llm"] * 2


def test_transcription_cannot_unload_ollama_during_active_llm(real_transcribe, monkeypatch):
    module = real_transcribe
    events = []
    install_loader(module, monkeypatch, events)
    holder = module.load_models()
    waiting = threading.Event()
    def work():
        with module.transcription_model_session(holder, lambda *a: waiting.set()):
            events.append("infer")
    with ThreadPoolExecutor(1) as pool:
        with gpu.llm_gpu_slot(get_llm_config()):
            task = pool.submit(work)
            assert waiting.wait(2)
            assert events == []
        task.result(timeout=2)
    assert events == ["unload-ollama", "load", "infer", "release"]


def test_cancel_after_model_load_releases_models_and_allows_next_job(real_transcribe, monkeypatch):
    module = real_transcribe
    events = []
    install_loader(module, monkeypatch, events)
    holder = module.load_models()
    class Cancelled(Exception):
        pass
    def cancel_after_load(*args):
        if holder.whisper_model is not None:
            raise Cancelled()
    with pytest.raises(Cancelled):
        with module.transcription_model_session(holder, cancel_after_load):
            pytest.fail("Inference ran after cancellation")
    assert events == ["unload-ollama", "load", "release"]
    with module.transcription_model_session(holder):
        assert holder.whisper_model is not None
    assert events[-1] == "release"


@pytest.mark.parametrize("failure_stage", ["load", "infer", "chained_load"])
def test_failure_tracebacks_do_not_keep_gpu_models_alive(real_transcribe, monkeypatch, failure_stage):
    module = real_transcribe
    events = []
    refs = install_loader(module, monkeypatch, events)
    holder = module.load_models()
    if failure_stage in {"load", "chained_load"}:
        def fail():
            partial_model = FakeGPUModel()
            refs.append(weakref.ref(partial_model))
            raise RuntimeError("load failed")
        def chained_fail():
            try:
                fail()
            except RuntimeError as exc:
                raise RuntimeError("wrapped load failed") from exc
        monkeypatch.setattr(module, "_load_models", chained_fail if failure_stage == "chained_load" else fail)
    else:
        def fail(path, models, callback):
            retained_model = models.whisper_model
            assert retained_model is not None
            raise RuntimeError("infer failed")
        monkeypatch.setattr(module, "_transcribe_audio", fail)
    with pytest.raises(RuntimeError, match="failed"):
        module.transcribe_audio("test.wav", holder)
    assert all(ref() is None for ref in refs)
    assert holder.whisper_model is None and holder.gpu_managed
    assert events[-1] == "release"
    with gpu.llm_gpu_slot(get_llm_config()):
        pass


def test_unconfirmed_ollama_release_prevents_transcription_load(real_transcribe, monkeypatch):
    module = real_transcribe
    monkeypatch.setattr(module, "_load_models", lambda: pytest.fail("Loaded before handover"))
    holder = module.load_models()
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: (_ for _ in ()).throw(httpx.ConnectError("offline")))
    with pytest.raises(gpu.GPUResourceError, match="nicht bestaetigt"):
        module.transcribe_audio("test.wav", holder)
    assert holder.whisper_model is None


def test_failed_cuda_release_blocks_later_llm_requests(real_transcribe, monkeypatch):
    module = real_transcribe
    events = []
    install_loader(module, monkeypatch, events)
    holder = module.load_models()
    def fail():
        raise RuntimeError("CUDA release failed")
    monkeypatch.setattr(module.torch.cuda, "empty_cache", fail)
    with pytest.raises(RuntimeError, match="CUDA"):
        with module.transcription_model_session(holder):
            pass
    monkeypatch.setattr(llm_transport, "_complete", lambda *a, **kw: pytest.fail("Used poisoned GPU"))
    with pytest.raises(gpu.GPUResourceError, match="neu starten"):
        llm_transport.complete(None, get_llm_config())


@pytest.mark.parametrize("device, enabled", [("cpu", "true"), ("cuda", "false")])
def test_cpu_and_disabled_mode_keep_eager_loading(real_transcribe, monkeypatch, device, enabled):
    module = real_transcribe
    monkeypatch.setenv("GPU_MODEL_SWITCHING", enabled)
    monkeypatch.setattr(module, "WHISPER_DEVICE", device)
    expected = module.TranscriptionModels(object(), None, None, None, device)
    monkeypatch.setattr(module, "_load_models", lambda: expected)
    assert module.load_models() is expected
    with module.transcription_model_session(expected):
        pass
    assert expected.whisper_model is not None


def test_speaker_backfill_uses_same_lifecycle(monkeypatch, tmp_path, real_transcribe):
    import main
    module = real_transcribe
    events = []
    install_loader(module, monkeypatch, events)
    holder = module.load_models()
    monkeypatch.setattr(main, "transcription_model_session", module.transcription_model_session)
    def extract(job, *, models):
        assert models.speaker_embedding_inference is not None
        events.append("speaker")
        return []
    monkeypatch.setattr(main, "_extract_job_speaker_embeddings_from_transcript", extract)
    assert main.extract_job_speaker_embeddings_from_transcript({}, models=holder) == []
    assert events == ["unload-ollama", "load", "speaker", "release"]


def test_health_and_speaker_backfill_accept_unloaded_on_demand_models(monkeypatch, tmp_path, real_transcribe):
    import main
    import persistence
    from fastapi.testclient import TestClient
    monkeypatch.setenv("PERSISTENCE_DB_PATH", str(tmp_path / "sessions.sqlite3"))
    persistence.init_db()
    monkeypatch.setattr(main.app.state, "models", real_transcribe.load_models(), raising=False)
    monkeypatch.setattr(main.app.state, "models_loaded", True, raising=False)
    client = TestClient(main.app)
    result = client.get("/health")
    assert result.status_code == 200
    assert result.json()["models_on_demand"] is True
    assert result.json()["models_loaded"] is False
    diagnostics = result.json()["speaker_embeddings"]
    assert diagnostics["loaded"] is False and diagnostics["on_demand"] is True
    # An empty backfill should succeed without loading a model or network calls.
    assert main.backfill_speaker_profile_embeddings().processed_job_count == 0


def test_remote_native_ollama_is_not_unloaded_for_local_transcription(monkeypatch):
    monkeypatch.setenv('LLM_PROVIDER', 'ollama')
    monkeypatch.setenv('LLM_BASE_URL', 'https://remote-model.example/v1')
    monkeypatch.setattr(httpx, 'get', lambda *a, **kw: pytest.fail('Contacted remote Ollama to unload'))
    gpu.unload_local_ollama(get_llm_config())
