"""Pinned tokenizer data, installed at image build time; never downloaded by jobs."""
import hashlib
from pathlib import Path
from urllib.request import urlopen

MODEL = 'Aleph-Alpha/Kolibri-1'
REVISION = '35bc4d3be745502227a67247de77d70e691614ee'
TOKENIZER_SHA256 = '5d4798f2a8c9d598d6dc005614216989d4b6aeffaf9f6f5e136f0ec5fc4a4c13'
TOKENIZER_BYTES = 9454692
TOKENIZER_PATH = Path(__file__).resolve().parent / 'model_assets' / 'kolibri-1-tokenizer.json'
TOKENIZER_URL = f'https://huggingface.co/{MODEL}/resolve/{REVISION}/tokenizer.json'


def bundled_tokenizer(model):
    return str(TOKENIZER_PATH) if model == MODEL and TOKENIZER_PATH.is_file() else ''


def install_tokenizer():
    """Small public data download; no weights, authentication or remote code."""
    if TOKENIZER_PATH.is_file() and hashlib.sha256(TOKENIZER_PATH.read_bytes()).hexdigest() == TOKENIZER_SHA256:
        return
    with urlopen(TOKENIZER_URL, timeout=120) as response:
        content = response.read(TOKENIZER_BYTES + 1)
    if len(content) != TOKENIZER_BYTES or hashlib.sha256(content).hexdigest() != TOKENIZER_SHA256:
        raise ValueError('Kolibri tokenizer size or checksum mismatch')
    TOKENIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = TOKENIZER_PATH.with_suffix('.tmp')
    try:
        temporary.write_bytes(content)
        temporary.replace(TOKENIZER_PATH)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    install_tokenizer()
    print('Kolibri-1 tokenizer ready (pinned revision and SHA-256 verified).')
