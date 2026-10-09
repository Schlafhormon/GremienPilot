"""Download pinned Gemma 4 assets and convert the HPI LoRA (no base-weight merge).

Run in a separate conversion venv; see docs/llm-configuration.md. Downloads are
resumable and checked against the immutable Hugging Face file metadata.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
LLAMA_TAG = 'b11429'
REPOS = {
    'weights': ('unsloth/gemma-4-31B-it-GGUF', 'c1ac76e99d5513b141e8adde7288b85c3f9c32ec'),
    'adapter': ('aihpi/gemma-4-31b-protokoll', '153460cf9a7c566df4a809a2f454c7b854c2405c'),
    'base': ('google/gemma-4-31B-it', '842da3794eaa0b77d5f08bae87a17459d91ff475'),
}
FILES = {
    'weights': ['gemma-4-31B-it-Q4_K_M.gguf', 'mmproj-F16.gguf'],
    'adapter': ['adapter_config.json', 'adapter_model.safetensors', 'tokenizer.json', 'tokenizer_config.json'],
    'base': ['config.json'],
}


def sha256(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(4 * 1024 * 1024):
            checksum.update(block)
    return checksum.hexdigest()


def download(url, target, size, checksum=None, git_oid=None, *, chunk_size=32 * 1024 * 1024, workers=8):
    def valid(path):
        if not path.is_file() or path.stat().st_size != size:
            return False
        if checksum:
            return sha256(path) == checksum
        return hashlib.sha1(f'blob {size}\0'.encode() + path.read_bytes()).hexdigest() == git_oid

    if valid(target):
        print(f'Verified {target.name}', flush=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + '.partial')
    def publish():
        # Windows scanners can briefly retain a handle after checksum reads.
        for attempt in range(5):
            try:
                partial.replace(target)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.5 * (attempt + 1))
    if size > chunk_size:
        # Bounded range requests also work behind buffering corporate proxies.
        def part(index):
            start = index * chunk_size
            end = min(size, start + chunk_size) - 1
            path = target.with_suffix(f'.part{index:04d}')
            if path.exists() and path.stat().st_size == end - start + 1:
                return path
            for attempt in range(4):
                try:
                    request = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
                    with urllib.request.urlopen(request, timeout=120) as response:
                        if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{size}':
                            raise ValueError('Invalid download range')
                        with path.open('wb') as stream:
                            shutil.copyfileobj(response, stream, 1024 * 1024)
                    if path.stat().st_size != end - start + 1:
                        raise ValueError('Incomplete range')
                    return path
                except (OSError, ValueError):
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)
        count = (size + chunk_size - 1) // chunk_size
        print(f'Downloading {target.name} in {count} resumable ranges', flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            paths = []
            for path in pool.map(part, range(count)):
                paths.append(path)
                if len(paths) % 16 == 0:
                    print(f'{target.name}: {len(paths)}/{count} ranges', flush=True)
        with partial.open('wb') as stream:
            for path in paths:
                with path.open('rb') as source:
                    shutil.copyfileobj(source, stream, 4 * 1024 * 1024)
        if not valid(partial):
            raise ValueError(f'Checksum mismatch: {target.name}')
        publish()
        for path in paths:
            path.unlink()
        print(f'Verified {target.name}', flush=True)
        return
    offset = partial.stat().st_size if partial.exists() else 0
    if offset == size and valid(partial):
        publish()
        return
    request = urllib.request.Request(url, headers={'Range': f'bytes={offset}-'} if offset else {})
    print(f'Downloading {target.name} ({size / 1e9:.2f} GB, resume {offset})', flush=True)
    with urllib.request.urlopen(request, timeout=120) as response:
        append = offset > 0 and response.status == 206
        if append and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
            raise ValueError('Invalid download range')
        with partial.open('ab' if append else 'wb') as stream:
            while block := response.read(4 * 1024 * 1024):
                stream.write(block)
    if not valid(partial):
        raise ValueError(f'Checksum/size mismatch: {target.name}; remove the .partial file and retry')
    publish()
    print(f'Verified {target.name}', flush=True)


def prepare(destination, llama, download_only=False):
    manifest = {'repositories': REPOS, 'llama_tag': LLAMA_TAG, 'files': {}}
    # Small assets first, allowing conversion while the large model downloads.
    for group in ('adapter', 'base', 'weights'):
        repo, revision = REPOS[group]
        with urllib.request.urlopen(f'https://huggingface.co/api/models/{repo}/tree/{revision}', timeout=60) as response:
            entries = {item['path']: item for item in json.load(response)}
        for name in FILES[group]:
            entry = entries[name]
            target = destination / group / name
            download(f'https://huggingface.co/{repo}/resolve/{revision}/{name}', target,
                     entry['size'], entry.get('lfs', {}).get('oid'), entry['oid'])
            manifest['files'][str(target.relative_to(destination)).replace('\\', '/')] = entry.get('lfs', {}).get('oid') or sha256(target)
    if not download_only:
        commit = subprocess.check_output(['git', '-C', str(llama), 'rev-parse', 'HEAD'], text=True).strip()
        expected = subprocess.check_output(['git', '-C', str(llama), 'rev-parse', f'{LLAMA_TAG}^{{commit}}'], text=True).strip()
        if commit != expected:
            raise ValueError(f'Converter must be checked out at {LLAMA_TAG}')
        adapter = destination / 'gemma-4-31b-protokoll-f16.gguf'
        subprocess.run([sys.executable, str(llama / 'convert_lora_to_gguf.py'),
                        '--base', str(destination / 'base'), '--outtype', 'f16',
                        '--outfile', str(adapter), str(destination / 'adapter')], check=True)
        manifest['files'][adapter.name] = sha256(adapter)
        manifest['llama_commit'] = commit
        runtime_files = ['weights/gemma-4-31B-it-Q4_K_M.gguf', 'weights/mmproj-F16.gguf', adapter.name]
        (destination / 'checksums.sha256').write_text(''.join(
            f'{manifest["files"][name]}  {name}\n' for name in runtime_files), encoding='ascii')
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    if not (3, 11) <= sys.version_info[:2] < (3, 14):
        sys.exit('Gemma asset preparation requires Python 3.11-3.13. '
                 'Use setup build to provide Python 3.12 automatically via Docker.')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT / 'data' / 'gemma4')
    parser.add_argument('--llama-cpp', type=Path, default=ROOT / 'data' / 'gemma4' / 'llama.cpp')
    parser.add_argument('--download-only', action='store_true')
    parser.add_argument('--bootstrap', action='store_true', help='Create an isolated converter venv and pinned llama.cpp checkout')
    parser.add_argument('--tools-only', action='store_true', help='With --bootstrap: install converter tools without downloading model assets')
    args = parser.parse_args()
    if args.tools_only and not args.bootstrap:
        parser.error('--tools-only requires --bootstrap')
    if args.bootstrap:
        import venv
        directory = args.directory.resolve()
        llama = args.llama_cpp.resolve()
        if not llama.exists():
            git = ['git', '-c', 'http.sslBackend=schannel'] if sys.platform == 'win32' else ['git']
            subprocess.run(git + ['clone', '--depth', '1', '--branch', LLAMA_TAG,
                                  'https://github.com/ggml-org/llama.cpp.git', str(llama)], check=True)
        environment = directory / 'converter-venv'
        python = environment / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
        if not python.exists():
            venv.create(environment, with_pip=True)
        subprocess.run([str(python), '-m', 'pip', 'install', '--use-feature=truststore',
                        'torch==2.11.0', '--index-url', 'https://download.pytorch.org/whl/cpu'], check=True)
        subprocess.run([str(python), '-m', 'pip', 'install', '--use-feature=truststore',
                        'transformers==4.57.6', 'sentencepiece==0.2.2', 'protobuf==4.25.9',
                        'safetensors==0.8.0', 'gguf==0.19.0', 'numpy==2.2.6'], check=True)
        if args.tools_only:
            sys.exit(0)
        subprocess.run([str(python), str(Path(__file__).resolve()), '--directory', str(directory),
                        '--llama-cpp', str(llama)] + (['--download-only'] if args.download_only else []), check=True)
        sys.exit(0)
    prepare(args.directory.resolve(), args.llama_cpp.resolve(), args.download_only)
