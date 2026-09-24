"""Exercise Bash setup functions without Docker or modifying the checkout."""
import os
from pathlib import Path
import re
import subprocess
import pytest

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
