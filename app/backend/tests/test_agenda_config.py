"""Load startup configuration in fresh processes without making model requests."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parents[1]


def load_config(overrides):
    env = {key: value for key, value in os.environ.items() if not key.startswith("AGENDA_DETECTION_")}
    env.update(overrides)
    return subprocess.run([
        sys.executable, "-c",
        "import agenda_detection as a; import json; "
        "print(json.dumps([a.AGENDA_DETECTION_USE_LLM, a.AGENDA_DETECTION_TIMEOUT_SECONDS, "
        "a.AGENDA_DETECTION_CHUNK_LINES, a.AGENDA_DETECTION_CHUNK_OVERLAP_LINES, "
        "a.AGENDA_DETECTION_CONTEXT_WINDOW_BEFORE, a.AGENDA_DETECTION_CONTEXT_WINDOW_AFTER]))",
    ], cwd=BACKEND, env=env, capture_output=True, text=True)


def test_unset_server_defaults():
    result = load_config({})
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [False, 8, 160, 12, 4, 8]


@pytest.mark.parametrize("path", [ROOT / ".env.example", BACKEND / ".env.example"])
def test_documented_defaults_are_loaded(path):
    values = dict(line.split("=", 1) for line in path.read_text().splitlines() if line.startswith("AGENDA_DETECTION_"))
    result = load_config(values)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [False, 8, 160, 12, 4, 8]


def test_startup_overrides_are_effective():
    result = load_config({
        "AGENDA_DETECTION_USE_LLM": "true",
        "AGENDA_DETECTION_TIMEOUT_SECONDS": "2.5",
        "AGENDA_DETECTION_CHUNK_LINES": "50",
        "AGENDA_DETECTION_CHUNK_OVERLAP_LINES": "3",
        "AGENDA_DETECTION_CONTEXT_WINDOW_BEFORE": "2",
        "AGENDA_DETECTION_CONTEXT_WINDOW_AFTER": "6",
        "LLM_TIMEOUT_SECONDS": "999",
    })
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [True, 2.5, 50, 3, 2, 6]


@pytest.mark.parametrize("timeout", ["0", "-1", "nan", "inf", "invalid"])
def test_invalid_timeouts_fail_at_startup(timeout):
    result = load_config({"AGENDA_DETECTION_TIMEOUT_SECONDS": timeout})
    assert result.returncode != 0
    assert "ValueError" in result.stderr


def test_invalid_server_mode_fails_at_startup():
    result = load_config({"AGENDA_DETECTION_USE_LLM": "maybe"})
    assert result.returncode != 0
    assert "AGENDA_DETECTION_USE_LLM must be true or false" in result.stderr
