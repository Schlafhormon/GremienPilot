# Scripts

This directory is intentionally small.

- `ollama-entrypoint.sh` is used by `docker-compose.yml` for the local Ollama service.
- `verify_llm_snapshot.py` inventories local originals or evaluates immutable results against approved human references without model calls.
- `research/` delegates historical inference commands to the verified production workflows and retains a separate explicit Kubernetes experiment.

Do not add one-off analysis scripts to this directory root. Put them under `scripts/research/` and document required local data, credentials, and expected outputs there.
