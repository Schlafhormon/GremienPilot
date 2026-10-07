# Scripts

This directory is intentionally small.

- `kolibri/` builds the pinned llama.cpp runtime and downloads the checksum-verified Kolibri IQ2_XS model.
- `verify_llm_snapshot.py` inventories local originals or evaluates immutable results against approved human references without model calls.
- `research/` delegates historical inference commands to the verified production workflows and retains a separate explicit Kubernetes experiment.

Do not add one-off analysis scripts to this directory root. Put them under `scripts/research/` and document required local data, credentials, and expected outputs there.
