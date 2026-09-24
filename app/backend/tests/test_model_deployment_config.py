"""Keep the versioned configuration surfaces in sync without reading local secrets."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]


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
    default = 'gemma4:31b-it-q4_K_M'
    for relative in ['setup.sh', 'setup.ps1', 'scripts/ollama-entrypoint.sh',
                     '.env.example', 'app/backend/.env.example', 'docker-compose.yml',
                     'app/backend/llm_config.py']:
        assert default in (ROOT / relative).read_text()
    assert 'LLM_PROVIDER: "ollama"' in (ROOT / 'k8s/backend/configmap.yaml').read_text()
    assert 'OLLAMA_LOAD_TIMEOUT=${OLLAMA_LOAD_TIMEOUT:-5m}' in (ROOT / 'docker-compose.yml').read_text()


def test_mandatory_summary_policy_is_consistent_and_not_disablable_by_legacy_flags():
    values = {'SUMMARY_OUTPUT_TOKENS': '4096', 'SUMMARY_MODEL_ATTEMPTS': '2', 'SUMMARY_RECONCILIATION_ROUNDS': '2'}
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
