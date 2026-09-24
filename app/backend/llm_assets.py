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


def bundled_tokenizer(model):
    return str(TOKENIZER_PATH) if model == MODEL and TOKENIZER_PATH.is_file() else ''


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
    install_tokenizer()
    print('Qwen3.5:9b tokenizer ready (pinned revision and SHA-256 verified).')
