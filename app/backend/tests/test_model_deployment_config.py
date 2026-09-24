"""Keep the versioned configuration surfaces in sync without reading local secrets."""
from pathlib import Path
import re
import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize('mode', ['fast', 'slow'])
def test_shared_context_default_reaches_both_modes_and_deployments(monkeypatch, mode):
    from llm_config import get_llm_config
    from processing_mode import processing_scope
    monkeypatch.delenv('LLM_CONTEXT_TOKENS', raising=False)
    with processing_scope(mode):
        assert get_llm_config().context_tokens == 131072
    name, default = 'LLM_CONTEXT_TOKENS', '131072'
    for path in ['.env.example', 'app/backend/.env.example']:
        assert f'{name}={default}' in (ROOT / path).read_text()
    assert f'{name}=${{{name}:-{default}}}' in (ROOT / 'docker-compose.yml').read_text()
    assert f'{name}: "{default}"' in (ROOT / 'k8s/backend/configmap.yaml').read_text()


@pytest.mark.parametrize('mode', ['fast', 'slow'])
def test_local_context_override_is_respected_in_both_modes(monkeypatch, mode):
    from llm_config import get_llm_config
    from processing_mode import processing_scope
    monkeypatch.setenv('LLM_CONTEXT_TOKENS', '65536')
    with processing_scope(mode):
        assert get_llm_config().context_tokens == 65536


def test_central_settings_reach_compose_examples_and_kubernetes():
    source = (ROOT / 'app/backend/llm_config.py').read_text()
    variables = set(re.findall(r"['\"](LLM_[A-Z_]+)['\"]", source))
    variables -= {'LLM_API_KEY', 'LLM_BASE_URL', 'LLM_MODEL'}
    surfaces = [ROOT / '.env.example', ROOT / 'app/backend/.env.example',
                ROOT / 'docker-compose.yml', ROOT / 'k8s/backend/configmap.yaml']
    for path in surfaces:
        missing = variables - set(re.findall(r'\bLLM_[A-Z_]+\b', path.read_text()))
        assert not missing, (path, missing)


def test_installation_defaults_match_without_overwriting_external_provider_model():
    default = 'qwen3.5:9b'
    for relative in ['setup.sh', 'setup.ps1', 'scripts/ollama-entrypoint.sh',
                     '.env.example', 'app/backend/.env.example', 'docker-compose.yml',
                     'app/backend/llm_config.py']:
        assert default in (ROOT / relative).read_text()
    assert 'LLM_PROVIDER: "ollama"' in (ROOT / 'k8s/backend/configmap.yaml').read_text()
    assert 'OLLAMA_LOAD_TIMEOUT=${OLLAMA_LOAD_TIMEOUT:-30m}' in (ROOT / 'docker-compose.yml').read_text()


def test_mandatory_summary_policy_is_consistent_and_not_disablable_by_legacy_flags():
    values = {'SUMMARY_OUTPUT_TOKENS': '8192', 'SUMMARY_MODEL_ATTEMPTS': '2', 'SUMMARY_RECONCILIATION_ROUNDS': '2'}
    for name, default in values.items():
        for path in ['.env.example', 'app/backend/.env.example']:
            assert f'{name}={default}' in (ROOT / path).read_text()
        assert f'{name}=${{{name}:-{default}}}' in (ROOT / 'docker-compose.yml').read_text()
        assert f'{name}: "{default}"' in (ROOT / 'k8s/backend/configmap.yaml').read_text()
    for path in ['summarize.py', 'summary_grounding.py']:
        assert 'LLM_SUMMARY_GROUNDING_MAX_CALLS' not in (ROOT / 'app/backend' / path).read_text()
        assert 'LLM_STRUCTURED_FALLBACK' not in (ROOT / 'app/backend' / path).read_text()


def test_kubernetes_default_endpoint_has_a_matching_internal_service():
    assert 'ollama/deployment.yaml' in (ROOT / 'k8s/kustomization.yaml').read_text()
    text = (ROOT / 'k8s/ollama/deployment.yaml').read_text()
    assert 'kind: Service' in text and 'name: ollama' in text
    assert 'key: LLM_MODEL' in text


def test_fast_agenda_budget_reaches_every_deployment_and_job_snapshot(monkeypatch):
    import durable_jobs
    name, default = 'AGENDA_FAST_OUTPUT_TOKENS', '8192'
    for path in ['.env.example', 'app/backend/.env.example']:
        assert f'{name}={default}' in (ROOT / path).read_text()
    assert f'{name}=${{{name}:-{default}}}' in (ROOT / 'docker-compose.yml').read_text()
    assert f'{name}: "{default}"' in (ROOT / 'k8s/backend/configmap.yaml').read_text()
    monkeypatch.setenv(name, default)
    before = durable_jobs.version_snapshot({'processing_mode': 'fast'})
    monkeypatch.setenv(name, '6144')
    after = durable_jobs.version_snapshot({'processing_mode': 'fast'})
    assert before['policy'][name] == default
    assert after['policy'][name] == '6144'
    assert before != after


def test_docker_sqlite_lives_on_linux_volume_not_windows_bind_mount():
    compose = (ROOT / 'docker-compose.yml').read_text()
    assert 'backend_state:/app/state' in compose
    assert 'PERSISTENCE_DB_PATH=${PERSISTENCE_DB_PATH:-/app/state/sessions.sqlite3}' in compose
    assert 'PERSISTENCE_DB_PATH=/app/state/sessions.sqlite3' in (ROOT / '.env.example').read_text()
    monitor = (ROOT / 'scripts/monitor_pipeline.py').read_text()
    assert "['docker', 'exec', '-i', container, 'python'" in monitor
    assert 'sample.update(live_snapshot(' in monitor


@pytest.mark.parametrize('mode', ['fast', 'slow'])
def test_fresh_install_uses_qwen_profile_without_manual_tuning(monkeypatch, mode):
    from llm_config import get_llm_config
    for key in ['LLM_MODEL','LLM_CONTEXT_TOKENS','LLM_THINKING_TOKENS','LLM_IMAGE_TOKENS',
                'LLM_TEMPERATURE','LLM_TOP_P','LLM_TOP_K']:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('LLM_PROVIDER','ollama')
    config=get_llm_config(processing_mode=mode)
    assert config.model=='qwen3.5:9b' and config.context_tokens==131072
    assert config.thinking_tokens==(0 if mode=='fast' else 4096)
    assert config.image_tokens==17408
    assert (config.temperature,config.top_p,config.top_k)==(1.0,0.95,20)
    # Unknown models never inherit a tokenizer, vision reserve or sampling profile.
    other=get_llm_config('custom-vision',processing_mode=mode)
    assert other.image_tokens==0 and not other.tokenizer_path
    assert (other.temperature,other.top_p,other.top_k)==(None,None,None)
    monkeypatch.setenv('LLM_IMAGE_TOKENS','0')
    monkeypatch.setenv('LLM_THINKING_TOKENS','2048')
    assert get_llm_config(processing_mode=mode).image_tokens==0
    assert get_llm_config(processing_mode=mode).thinking_tokens==(0 if mode=='fast' else 2048)


@pytest.mark.parametrize('mode', ['fast', 'slow'])
def test_fresh_install_plans_eighty_line_blocks_and_preserves_reviews(monkeypatch, agenda_model, mode):
    from test_agenda_llm import run
    for key in ['AGENDA_OUTPUT_TOKENS','AGENDA_FAST_OUTPUT_TOKENS','AGENDA_OUTPUT_TOKENS_PER_LINE',
                'AGENDA_DETECTION_CHUNK_LINES','AGENDA_COMPACT_ASSIGNMENTS','LLM_CONTEXT_TOKENS']:
        monkeypatch.delenv(key, raising=False)
    result=run(['Beratung.']*161,processing_mode=mode)
    assert result.llm.processing_complete
    phases=['fast:detail'] if mode=='fast' else ['primary:detail','independent:detail']
    for phase in phases:
        calls=[(body,args) for body,args in agenda_model.calls if body['phase']==phase]
        assert [len(body['target_lines']) for body,_ in calls]==[80,80,1]
        assert all(args['max_tokens']==8192 for _,args in calls)
        assert all('response' in args['response_format']['json_schema']['schema']['properties'] for _,args in calls)
    assert result.llm.review_complete==(mode=='slow')


def test_long_meeting_defaults_match_deployments():
    values={'AGENDA_OUTPUT_TOKENS':'8192','AGENDA_OUTPUT_TOKENS_PER_LINE':'40',
            'AGENDA_DETECTION_CHUNK_LINES':'80','AGENDA_COMPACT_ASSIGNMENTS':'true'}
    for name,value in values.items():
        for path in ['.env.example','app/backend/.env.example']:
            assert f'{name}={value}' in (ROOT/path).read_text()
        assert f'{name}=${{{name}:-{value}}}' in (ROOT/'docker-compose.yml').read_text()
        assert f'{name}: "{value}"' in (ROOT/'k8s/backend/configmap.yaml').read_text()


def test_upload_limits_allow_two_gib_files_and_multipart_overhead():
    import main
    assert main.MAX_UPLOAD_BYTES==2147483648
    for path in ['.env.example','app/backend/.env.example','docker-compose.yml','k8s/backend/configmap.yaml']:
        assert '2147483648' in (ROOT/path).read_text()
    assert 'client_max_body_size 2050M;' in (ROOT/'app/frontend/nginx.conf').read_text()
    assert 'proxy-body-size: "2050m"' in (ROOT/'k8s/frontend/ingress.example.yaml').read_text()
