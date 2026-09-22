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
    assert 'LLM_PROVIDER: "openai-compatible"' in (ROOT / 'k8s/backend/configmap.yaml').read_text()
    assert 'OLLAMA_LOAD_TIMEOUT=${OLLAMA_LOAD_TIMEOUT:-5m}' in (ROOT / 'docker-compose.yml').read_text()
