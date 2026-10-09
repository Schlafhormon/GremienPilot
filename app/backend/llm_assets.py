"""Pinned tokenizer data, installed at image build time; never downloaded by jobs."""
import hashlib
from pathlib import Path
from urllib.request import urlopen

MODEL = 'qwen3.5:9b'
REVISION = 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'
TOKENIZER_SHA256 = '5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42'
TOKENIZER_BYTES = 12807982
TOKENIZER_PATH = Path(__file__).resolve().parent / 'model_assets' / 'qwen3.5-9b-tokenizer.json'
TOKENIZER_URL = f'https://huggingface.co/Qwen/Qwen3.5-9B/resolve/{REVISION}/tokenizer.json'
GEMMA_MODEL = 'gemma-4-31b'
GEMMA_REVISION = '153460cf9a7c566df4a809a2f454c7b854c2405c'
GEMMA_SHA256 = 'cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f'
GEMMA_BYTES = 32169626
GEMMA_PATH = TOKENIZER_PATH.with_name('gemma-4-31b-tokenizer.json')


def bundled_tokenizer(model):
    if model == GEMMA_MODEL and GEMMA_PATH.is_file():
        return str(GEMMA_PATH)
    return str(TOKENIZER_PATH) if model == MODEL and TOKENIZER_PATH.is_file() else ''


def install_gemma_tokenizer():
    if GEMMA_PATH.is_file() and hashlib.sha256(GEMMA_PATH.read_bytes()).hexdigest() == GEMMA_SHA256:
        return
    url = f'https://huggingface.co/aihpi/gemma-4-31b-protokoll/resolve/{GEMMA_REVISION}/tokenizer.json'
    with urlopen(url, timeout=120) as response:
        content = response.read(GEMMA_BYTES + 1)
    if len(content) != GEMMA_BYTES or hashlib.sha256(content).hexdigest() != GEMMA_SHA256:
        raise ValueError('Gemma tokenizer size or checksum mismatch')
    GEMMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = GEMMA_PATH.with_suffix('.tmp')
    try:
        temporary.write_bytes(content)
        temporary.replace(GEMMA_PATH)
    finally:
        temporary.unlink(missing_ok=True)


def install_tokenizer():
    """Small public data download; no weights, authentication or remote code."""
    if TOKENIZER_PATH.is_file() and hashlib.sha256(TOKENIZER_PATH.read_bytes()).hexdigest() == TOKENIZER_SHA256:
        return
    with urlopen(TOKENIZER_URL, timeout=120) as response:
        content = response.read(TOKENIZER_BYTES + 1)
    if len(content) != TOKENIZER_BYTES or hashlib.sha256(content).hexdigest() != TOKENIZER_SHA256:
        raise ValueError('Qwen tokenizer size or checksum mismatch')
    TOKENIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = TOKENIZER_PATH.with_suffix('.tmp')
    try:
        temporary.write_bytes(content)
        temporary.replace(TOKENIZER_PATH)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    install_gemma_tokenizer()
    print('Gemma 4 tokenizer ready (pinned revision and SHA-256 verified).')
