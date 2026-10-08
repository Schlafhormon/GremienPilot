"""Exercise Bash setup functions without Docker or modifying the checkout."""
import os
from pathlib import Path
import re
import subprocess
import pytest

pytestmark = pytest.mark.skipif(os.name == 'nt', reason='Run POSIX shell contracts in the Linux test container')

ROOT=Path(__file__).resolve().parents[3]


def function(name):
    script=(ROOT/'setup.sh').read_text()
    return re.search(r'^'+name+r'\(\) \{\n.*?^\}',script,re.M|re.S).group()


def shell(code,path):
    return subprocess.run(['bash','-c',code],cwd=path,env=dict(os.environ,SCRIPT_DIR=str(path)),
                          text=True,capture_output=True)


def test_bash_creates_defaults_once_and_preserves_existing_settings(tmp_path):
    (tmp_path/'.env.example').write_text('LLM_MODEL=qwen3.5:9b\n')
    code='info(){ :; }; error(){ :; };\n'+function('initialize_configuration')+'\ninitialize_configuration'
    assert shell(code,tmp_path).returncode==0
    assert (tmp_path/'.env').read_text()=='LLM_MODEL=qwen3.5:9b\n'
    (tmp_path/'.env').write_text('LLM_MODEL=custom-model\n')
    assert shell(code,tmp_path).returncode==0
    assert (tmp_path/'.env').read_text()=='LLM_MODEL=custom-model\n'


def test_bash_port_settings_follow_compose_precedence(tmp_path):
    (tmp_path/'.env').write_text("TEST_GEMMA_PORT='3001' # local port\n")
    code = function('configured_port') + '\nconfigured_port TEST_GEMMA_PORT 3000'
    assert shell(code, tmp_path).stdout.strip() == '3001'
    code = "export TEST_GEMMA_PORT=3002\n" + code
    assert shell(code, tmp_path).stdout.strip() == '3002'


@pytest.mark.parametrize('valid_checksum', [True, False])
def test_gemma_start_preserves_library_directory_and_checks_assets(tmp_path, valid_checksum):
    import hashlib
    runtime = tmp_path / 'app'
    models = tmp_path / 'models'
    runtime.mkdir()
    models.mkdir()
    # Like the pinned image, the server requires its own working directory.
    server = runtime / 'llama-server'
    server.write_text('''#!/bin/sh
test "$PWD" = "$(dirname "$0")" || exit 42
if [ "$1" = --version ]; then echo VERSION; else echo SERVER_STARTED; fi
''')
    server.chmod(0o755)
    (models / 'weights.gguf').write_bytes(b'model')
    digest = hashlib.sha256(b'model' if valid_checksum else b'corrupt').hexdigest()
    (models / 'checksums.sha256').write_text(f'{digest}  weights.gguf\n')
    script = (ROOT / 'scripts/llama-entrypoint.sh').read_text()
    script = script.replace('/app', str(runtime)).replace('/models', str(models))
    result = subprocess.run(['sh', '-c', script], cwd=tmp_path, text=True,
                            capture_output=True, timeout=10)
    assert ('SERVER_STARTED' in result.stdout) == valid_checksum
    assert (result.returncode == 0) == valid_checksum, result.stderr
    assert 'VERSION' in result.stdout


@pytest.mark.parametrize('scenario',['fresh','existing','invalid'])
def test_bash_start_installs_only_when_no_containers_exist(tmp_path,scenario):
    code='''info(){ :; }; error(){ :; }; check_docker(){ return 0; }
do_build(){ echo BUILD; }; wait_for_services(){ echo READY; }
docker(){
    if [[ "$*" == *" ps "* ]]; then
        case "$scenario" in existing) echo container-id;; invalid) return 1;; esac
    elif [[ "$*" == *" start"* ]]; then echo START; fi
}
'''+f'scenario={scenario}\n'+function('do_start')+'\ndo_start'
    result=shell(code,tmp_path)
    assert ('BUILD' in result.stdout)==(scenario=='fresh')
    assert ('START' in result.stdout)==(scenario=='existing')
    assert result.returncode==(1 if scenario=='invalid' else 0)


@pytest.mark.parametrize('present',['yes','no'])
def test_ollama_bootstrap_checks_exact_model_not_similar_cached_tag(tmp_path,present):
    executable=tmp_path/'ollama'
    executable.write_text('''#!/bin/bash
case "$1" in
  serve) exit 0;;
  list) echo 'qwen3.5:9b-other';;
  show) printf '%s' "$2" > "$CHECKED"; test "$PRESENT" = yes;;
  pull) printf '%s' "$2" > "$PULLED";;
esac
''')
    executable.chmod(0o755)
    result=subprocess.run(['bash',str(ROOT/'scripts/ollama-entrypoint.sh')],
        env=dict(os.environ,PATH=str(tmp_path)+os.pathsep+os.environ['PATH'],
                 OLLAMA_MODEL='qwen3.5:9b',PRESENT=present,CHECKED=str(tmp_path/'checked'),PULLED=str(tmp_path/'pulled')),
        text=True,capture_output=True,timeout=10)
    assert result.returncode==0,result.stderr
    assert (tmp_path/'checked').read_text()=='qwen3.5:9b'
    assert (tmp_path/'pulled').exists()==(present=='no')
    if present=='no': assert (tmp_path/'pulled').read_text()=='qwen3.5:9b'
