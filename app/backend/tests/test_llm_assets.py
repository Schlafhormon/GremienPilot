import hashlib
import io
import pytest
import llm_assets
from llm_config import get_llm_config


def test_tokenizer_install_verifies_bytes_and_reuses_valid_asset(tmp_path,monkeypatch):
    content=b'{"test":"tokenizer"}'
    path=tmp_path/'model_assets'/'tokenizer.json'
    monkeypatch.setattr(llm_assets,'TOKENIZER_PATH',path)
    monkeypatch.setattr(llm_assets,'TOKENIZER_BYTES',len(content))
    monkeypatch.setattr(llm_assets,'TOKENIZER_SHA256',hashlib.sha256(content).hexdigest())
    calls=[]
    def download(url,timeout):
        calls.append(url)
        return io.BytesIO(content)
    monkeypatch.setattr(llm_assets,'urlopen',download)
    llm_assets.install_tokenizer(); llm_assets.install_tokenizer()
    assert path.read_bytes()==content and calls==[llm_assets.TOKENIZER_URL]
    assert llm_assets.REVISION in calls[0]
    monkeypatch.setattr(llm_assets,'TOKENIZER_SHA256','invalid')
    with pytest.raises(ValueError,match='checksum'):
        llm_assets.install_tokenizer()
    assert path.read_bytes()==content and not path.with_suffix('.tmp').exists()


def test_bundled_tokenizer_is_offline_and_model_specific(tmp_path,monkeypatch):
    path=tmp_path/'tokenizer.json';path.write_text('{}')
    monkeypatch.setattr(llm_assets,'TOKENIZER_PATH',path)
    monkeypatch.setenv('LLM_PROVIDER','ollama')
    monkeypatch.setenv('LLM_MODEL',llm_assets.MODEL)
    monkeypatch.delenv('LLM_TOKENIZER_PATH',raising=False)
    monkeypatch.delenv('LLM_TOKENIZER_MODEL',raising=False)
    def forbidden(*args,**kwargs): raise AssertionError('No runtime download')
    monkeypatch.setattr(llm_assets,'urlopen',forbidden)
    config=get_llm_config()
    assert config.tokenizer_path==str(path) and config.tokenizer_model==llm_assets.MODEL
    assert not get_llm_config('other-model').tokenizer_path
    monkeypatch.setenv('LLM_TOKENIZER_PATH','explicit.json')
    monkeypatch.setenv('LLM_TOKENIZER_MODEL',llm_assets.MODEL)
    assert get_llm_config().tokenizer_path=='explicit.json'
    monkeypatch.delenv('LLM_TOKENIZER_PATH');monkeypatch.delenv('LLM_TOKENIZER_MODEL')
    path.unlink()
    assert not get_llm_config().tokenizer_path  # Source checkout retains safe byte fallback.
