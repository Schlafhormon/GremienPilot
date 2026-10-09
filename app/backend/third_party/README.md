# Model assets

The default image downloads the tokenizer from
[aihpi/gemma-4-31b-protokoll](https://huggingface.co/aihpi/gemma-4-31b-protokoll),
revision `153460cf9a7c566df4a809a2f454c7b854c2405c`, verified by SHA-256.
The adapter card specifies the Gemma terms; weights are downloaded separately
by `scripts/prepare_gemma4.py` and are not included in the application image.

`prompt_gemma.txt` is copied from the HPI reference application; its MIT notice
is included unchanged in [HPI-protocol-MIT.txt](HPI-protocol-MIT.txt).

## Optional legacy Qwen tokenizer

The image build downloads `tokenizer.json` from Qwen/Qwen3.5-9B by the Qwen Team,
revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. The data is distributed under
the Apache License 2.0, included unchanged in [Qwen3.5-LICENSE.txt](Qwen3.5-LICENSE.txt).

Source: https://huggingface.co/Qwen/Qwen3.5-9B/tree/c202236235762e1c871ad0ccb60c8ee5ba337b9a

Only tokenizer data is included, not model weights or executable model code.
`llm_assets.py` verifies the pinned file's SHA-256 during the build.
