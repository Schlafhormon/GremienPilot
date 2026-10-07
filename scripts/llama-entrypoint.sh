#!/bin/sh
set -eu
cd /models
if [ ! -s checksums.sha256 ]; then
    echo 'Gemma assets missing. Run: python scripts/prepare_gemma4.py --bootstrap' >&2
    exit 1
fi
sha256sum -c checksums.sha256
exec /app/llama-server \
    --model /models/weights/gemma-4-31B-it-Q4_K_M.gguf \
    --mmproj /models/weights/mmproj-F16.gguf --no-mmproj-offload \
    --lora /models/gemma-4-31b-protokoll-f16.gguf --lora-init-without-apply \
    --alias gemma-4-31b --host 0.0.0.0 --port 8080 \
    --ctx-size "${LLM_CONTEXT_TOKENS:-16384}" --parallel 1 \
    --gpu-layers "${LLM_GPU_LAYERS:-16}" --threads "${LLM_CPU_THREADS:-12}" \
    --batch-size 256 --ubatch-size 128 --flash-attn on \
    --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 \
    --image-max-tokens 1120 --jinja --reasoning-budget 0 \
    --repeat-penalty 1.0 --sleep-idle-seconds 30 --no-context-shift
