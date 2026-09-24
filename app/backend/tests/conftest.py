import sys
import types
from dataclasses import dataclass
from pathlib import Path
from contextlib import nullcontext
import re

import pytest


@pytest.fixture(autouse=True)
def isolated_llm_transport(monkeypatch, request):
    # Unit tests must never accidentally contact a local model or reuse private caches.
    monkeypatch.setenv("LLM_OLLAMA_NATIVE", "false")
    monkeypatch.setenv("LLM_MODEL", "qwen3:8b")
    # Keep boundary/splitting fixtures deterministic; deployment-default tests
    # explicitly remove this override to exercise the production configuration.
    monkeypatch.setenv("LLM_CONTEXT_TOKENS", "16384")
    # Historical per-line/repair fixtures exercise the still supported legacy
    # contract. Fresh-install integration tests explicitly clear these settings.
    monkeypatch.setenv("AGENDA_COMPACT_ASSIGNMENTS", "false")
    monkeypatch.setenv("AGENDA_OUTPUT_TOKENS", "4096")
    monkeypatch.setenv("SUMMARY_OUTPUT_TOKENS", "4096")
    monkeypatch.setenv("LLM_RETRY_BACKOFF_SECONDS", "0")
    monkeypatch.setenv("LLM_MAX_RETRIES", "0")
    monkeypatch.setenv("LLM_MODEL_REVISION", "test-revision")
    monkeypatch.setenv("LLM_SUMMARY_FACT_REVIEW_MAX_CALLS", "0")
    monkeypatch.setenv("LLM_SUMMARY_GROUNDING_MAX_CALLS", "0")
    monkeypatch.delenv("LLM_CACHE_DIR", raising=False)
    monkeypatch.delenv("LLM_AUDIT_DIR", raising=False)
    monkeypatch.setenv("GPU_MODEL_SWITCHING", "false")
    import llm_transport
    async def forbidden_network(*args, **kwargs):
        raise AssertionError('Model transport must be mocked in tests')
        yield  # async generator contract
    if request.node.path.name not in {'test_llm_transport.py', 'test_llm_reasoning.py'}:
        monkeypatch.setattr(llm_transport, '_openai_stream', forbidden_network)



@pytest.fixture
def frontend_summary_prompt():
    """Read the actual UI default so this regression cannot drift from production."""
    source = (
        Path(__file__).resolve().parents[2]
        / "frontend/src/components/LLMSettingsPanel.tsx"
    ).read_text(encoding="utf-8")
    match = re.search(r"export const DEFAULT_SYSTEM_PROMPT = `([^`]+)`;", source)
    assert match, "Update this fixture if the frontend prompt representation changes"
    return match.group(1)


@dataclass
class FakeTranscriptionModels:
    device: str = "cpu"


@dataclass
class FakeTranscriptionResult:
    transcript: list[dict]
    audio_duration_seconds: float


fake_transcribe = types.ModuleType("transcribe")
fake_transcribe.TranscriptionModels = FakeTranscriptionModels
fake_transcribe.TranscriptionResult = FakeTranscriptionResult
fake_transcribe.WHISPER_MODEL = "test-whisper"
fake_transcribe.WHISPER_BATCH_SIZE = 1
fake_transcribe.load_models = lambda: FakeTranscriptionModels()
fake_transcribe.transcription_model_session = lambda models, progress_callback=None: nullcontext(models)
fake_transcribe._cleanup_memory = lambda device: None
fake_transcribe.transcribe_audio = lambda file_path, models, progress_callback=None: FakeTranscriptionResult(
    transcript=[],
    audio_duration_seconds=0,
)

sys.modules.setdefault("transcribe", fake_transcribe)


@pytest.fixture
def fake_openai_module(monkeypatch):
    class FakeCompletions:
        def __init__(self, owner):
            self.owner = owner

        def create(self, **kwargs):
            self.owner.calls.append(kwargs)
            if type(self.owner).responses:
                response = type(self.owner).responses.pop(0)
                if isinstance(response, Exception):
                    raise response
                content = response
            else:
                content = self.owner.content
            if callable(content):
                content = content(kwargs)
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(
                            content=content,
                            reasoning=self.owner.reasoning,
                        )
                    )
                ]
            )

    class FakeModels:
        def __init__(self, owner):
            self.owner = owner

        def list(self):
            response = type(self.owner).models_response
            if isinstance(response, Exception):
                raise response
            return types.SimpleNamespace(
                data=[
                    types.SimpleNamespace(id=model_id)
                    for model_id in response
                ]
            )

    class FakeOpenAI:
        instances = []
        content = ""
        reasoning = ""
        responses = []
        models_response = ["qwen3:8b", "test-model"]

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.calls = []
            self.content = type(self).content
            self.chat = types.SimpleNamespace(
                completions=FakeCompletions(self),
            )
            self.models = FakeModels(self)
            type(self).instances.append(self)

    FakeOpenAI.instances = []
    FakeOpenAI.content = ""
    FakeOpenAI.reasoning = ""
    FakeOpenAI.responses = []
    FakeOpenAI.models_response = ["qwen3:8b", "test-model"]
    module = types.ModuleType("openai")
    module.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", module)
    import llm_transport
    async def stream(client, config, payload):
        response = client.chat.completions.create(**payload)
        yield {'choices': [{'delta': {'content': response.choices[0].message.content}, 'finish_reason': 'stop'}]}
    monkeypatch.setattr(llm_transport, '_openai_stream', stream)
    return FakeOpenAI


@pytest.fixture(autouse=True)
def isolated_job_storage(tmp_path_factory, monkeypatch):
    """Even lifespan/API tests without an explicit database must never touch user jobs."""
    tmp_path = tmp_path_factory.mktemp("job-storage")
    monkeypatch.setenv('PERSISTENCE_DB_PATH', str(tmp_path / 'isolated.sqlite3'))
    import persistence
    persistence.init_db()
    if 'main' in sys.modules:
        monkeypatch.setattr(sys.modules['main'], 'UPLOAD_DIR', tmp_path / 'uploads')


@pytest.fixture
def agenda_model(monkeypatch, fake_openai_module):
    from agenda_fixtures import AgendaModel
    return AgendaModel(monkeypatch)


@pytest.fixture
def summary_model(fake_openai_module):
    from summary_fixtures import SummaryModel
    model = SummaryModel()
    fake_openai_module.content = model
    return model
