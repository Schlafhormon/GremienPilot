"""Verify resumable asset downloads without contacting a model registry."""
import hashlib
import importlib.util
import io
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('prepare_gemma4', Path(__file__).resolve().parents[3] / 'scripts/prepare_gemma4.py')
assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assets)


def test_ranges_resume_and_verify_before_publishing(tmp_path, monkeypatch):
    data = b'0123456789abcdef'
    target = tmp_path / 'model.gguf'
    target.with_suffix('.part0000').write_bytes(data[:4])
    calls = []
    def open_request(request, **kwargs):
        begin, end = map(int, request.headers['Range'].removeprefix('bytes=').split('-'))
        calls.append(begin)
        response = io.BytesIO(data[begin:end+1])
        response.status = 206
        response.headers = {'Content-Range': f'bytes {begin}-{end}/{len(data)}'}
        return response
    monkeypatch.setattr(assets.urllib.request, 'urlopen', open_request)
    original_replace = Path.replace
    replacements = []
    def temporarily_locked(path, destination):
        replacements.append(path)
        if len(replacements) == 1:
            raise PermissionError('Scanner still reading')
        return original_replace(path, destination)
    monkeypatch.setattr(Path, 'replace', temporarily_locked)
    monkeypatch.setattr(assets.time, 'sleep', lambda _: None)
    args = ('https://example.invalid/model', target, len(data), hashlib.sha256(data).hexdigest())
    assets.download(*args, chunk_size=4, workers=2)
    assert target.read_bytes() == data and sorted(calls) == [4, 8, 12]
    assert len(replacements) == 2
    assert not list(tmp_path.glob('*.part*'))
    assets.download(*args, chunk_size=4, workers=2)
    assert len(calls) == 3  # A verified existing file makes no network request.


def test_corrupt_ranges_are_never_published(tmp_path, monkeypatch):
    target = tmp_path / 'model.gguf'
    def open_request(request, **kwargs):
        begin, end = map(int, request.headers['Range'].removeprefix('bytes=').split('-'))
        response = io.BytesIO(b'x' * (end-begin+1))
        response.status = 206
        response.headers = {'Content-Range': f'bytes {begin}-{end}/8'}
        return response
    monkeypatch.setattr(assets.urllib.request, 'urlopen', open_request)
    with pytest.raises(ValueError, match='Checksum mismatch'):
        assets.download('https://example.invalid/model', target, 8,
                        hashlib.sha256(b'correct!').hexdigest(), chunk_size=4, workers=1)
    assert not target.exists()
