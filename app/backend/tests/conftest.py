import sys
import types
from dataclasses import dataclass
from pathlib import Path
import re

import pytest


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
    return FakeOpenAI
