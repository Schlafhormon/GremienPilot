#!/bin/sh
set -eu

# Complete 78B model, mixed IQ2_XS quantization, 20.94 GiB. No expert pruning.
revision=0fd72f818529e4697ef7942de68867c51c6e9507
file=Sakura-Kolibri-1-IQ2_XS-20.94GiB.gguf
sha=5a55e6f918d7f045ce013df193df29083819904be0e9d80607c7f91a2840d838
model=/models/$file
mkdir -p /models
if [ ! -f "$model" ]; then
    echo "Downloading Kolibri-1 IQ2_XS (22.5 GB); interrupted downloads resume."
    aria2c --ca-certificate=/etc/ssl/certs/ca-certificates.crt \
        --continue=true --file-allocation=none --auto-file-renaming=false \
        --max-connection-per-server=8 --split=8 --min-split-size=16M \
        --max-tries=10 --retry-wait=5 --connect-timeout=15 --timeout=60 \
        --summary-interval=30 --enable-color=false --show-console-readout=false \
        --checksum="sha-256=$sha" --check-integrity=true --dir=/models --out="$file.partial" \
        "https://huggingface.co/webmp3/Sakura-MicroQuality-Kolibri-1-GGUF/resolve/$revision/$file"
    mv "$model.partial" "$model"
else
    # Verify cached weights too; never start with silently corrupted weights.
    echo "$sha  $model" | sha256sum -c -
fi
echo "Loading Kolibri-1; context=${LLM_CONTEXT_TOKENS:-32768}, GPU layers=${KOLIBRI_GPU_LAYERS:-0}."
exec /opt/llama/llama-server --model "$model" --alias Aleph-Alpha/Kolibri-1 \
    --host 0.0.0.0 --port 8080 --ctx-size "${LLM_CONTEXT_TOKENS:-32768}" \
    --parallel 1 --threads "${LLM_CPU_THREADS:-8}" --threads-batch "${LLM_CPU_THREADS:-8}" \
    --gpu-layers "${KOLIBRI_GPU_LAYERS:-0}" --batch-size 512 --ubatch-size 128 \
    --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
    --jinja --reasoning-format deepseek --no-context-shift \
    --sleep-idle-seconds 15 --cache-ram 0 \
    --temp 1.0 --top-p 0.97 --top-k 128
