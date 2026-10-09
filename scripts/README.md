# Scripts

This directory is intentionally small.

- `llama-entrypoint.sh` verifies Gemma assets and starts the local llama.cpp service.
- `Dockerfile.gemma-converter` provides Python 3.12 and pinned conversion tools for
  both setup scripts, without requiring host Python. Its Docker build context is
  restricted by `Dockerfile.gemma-converter.dockerignore`.
- `prepare_gemma4.py` downloads Q4_K_M assets and converts the HPI adapter. Its
  `--bootstrap --tools-only` mode builds the converter image without model downloads.
- `ollama-entrypoint.sh` remains available for legacy Ollama installations.
- `verify_llm_snapshot.py` inventories local originals or evaluates immutable results against approved human references without model calls.
- `research/` delegates historical inference commands to the verified production workflows and retains a separate explicit Kubernetes experiment.

Do not add one-off analysis scripts to this directory root. Put them under `scripts/research/` and document required local data, credentials, and expected outputs there.
