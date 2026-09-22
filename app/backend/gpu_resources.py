"""Exclusive GPU use by local Ollama and on-demand transcription models.

The application runs one backend process. Its worker threads share this gate;
other applications using the GPU are outside this coordinator's control.
"""
from contextlib import contextmanager
import logging
import math
import os
import threading
import time

logger = logging.getLogger(__name__)
_GPU_LOCK = threading.RLock()
_release_error = False


class GPUResourceError(RuntimeError):
    pass


def switching_enabled():
    value = os.environ.get("GPU_MODEL_SWITCHING", "false").strip().lower()
    if value not in {"true", "false"}:
        raise ValueError("GPU_MODEL_SWITCHING must be true or false")
    return value == "true"


def unload_timeout():
    value = float(os.environ.get("GPU_MODEL_UNLOAD_TIMEOUT_SECONDS", "120"))
    if not math.isfinite(value) or value <= 0:
        raise ValueError("GPU_MODEL_UNLOAD_TIMEOUT_SECONDS must be positive and finite")
    return value


def block_after_release_failure():
    # Keep no exception/traceback here: it could retain the GPU tensors.
    global _release_error
    _release_error = True


@contextmanager
def gpu_slot(check_cancel=None):
    while not _GPU_LOCK.acquire(timeout=0.25):
        if check_cancel is not None:
            check_cancel()
    try:
        if check_cancel is not None:
            check_cancel()
        if _release_error:
            raise GPUResourceError(
                "GPU-Speicher konnte nicht freigegeben werden; Backend neu starten."
            )
        yield
    finally:
        _GPU_LOCK.release()


@contextmanager
def llm_gpu_slot(config):
    if switching_enabled() and config.uses_ollama:
        with gpu_slot():
            yield
    else:
        yield


def unload_local_ollama(config, check_cancel=None, *, only_model=None, except_model=None):
    """Unload all runners on our local Ollama service and verify completion.

Call only while holding gpu_slot. Refuse to load Whisper if Ollama cannot
confirm the handover, including after a timed-out inference request.
"""
    if not config.uses_ollama:
        return
    import httpx

    base = config.base_url.rstrip("/").removesuffix("/v1")
    headers = {"Authorization": "Bearer " + config.api_key}
    deadline = time.monotonic() + unload_timeout()
    requested = set()

    def remaining():
        if check_cancel is not None:
            check_cancel()
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise GPUResourceError("Zeitlimit beim Entladen der Ollama-Modelle erreicht.")
        return seconds

    try:
        while True:
            response = httpx.get(base + "/api/ps", headers=headers, timeout=remaining())
            response.raise_for_status()
            payload = response.json()
            entries = payload.get("models") if isinstance(payload, dict) else None
            if not isinstance(entries, list):
                raise GPUResourceError("Ollama liefert keinen gueltigen Modellstatus.")
            if any(not isinstance(entry, dict) or not isinstance(entry.get('model') or entry.get('name'), str)
                   or not (entry.get('model') or entry.get('name')) for entry in entries):
                raise GPUResourceError("Ollama liefert einen ungueltigen Modellnamen.")
            normalize = lambda name: name if ':' in name else name + ':latest'
            if only_model:
                entries = [e for e in entries if normalize(e.get('model') or e.get('name', '')) == normalize(only_model)]
            if except_model:
                entries = [e for e in entries if normalize(e.get('model') or e.get('name', '')) != normalize(except_model)]
            if not entries:
                return
            for entry in entries:
                name = (entry.get("model") or entry.get("name")) if isinstance(entry, dict) else None
                if not isinstance(name, str) or not name:
                    raise GPUResourceError("Ollama liefert einen ungueltigen Modellnamen.")
                if name not in requested:
                    logger.info("Unloading Ollama model before transcription: %s", name)
                    response = httpx.post(
                        base + "/api/generate", headers=headers, timeout=remaining(),
                        json={"model": name, "keep_alive": 0, "stream": False},
                    )
                    response.raise_for_status()
                    requested.add(name)
            time.sleep(min(0.1, remaining()))
    except (httpx.HTTPError, ValueError) as exc:
        raise GPUResourceError(
            "Ollama-Speicherfreigabe konnte nicht bestaetigt werden; "
            "Transkriptionsmodelle wurden nicht geladen."
        ) from exc


def unload_ollama_model(config):
    """Release only this task's runner, including after a failed/late request."""
    unload_local_ollama(config, only_model=config.model)
