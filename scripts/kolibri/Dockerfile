# syntax=docker/dockerfile:1.7
# Kolibri is not supported by stock llama.cpp/Ollama yet. This release contains
# the MIT-licensed kolibri1 patch; both runtime and weights are checksum pinned.
FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl aria2 libgomp1 && rm -rf /var/lib/apt/lists/*
RUN --mount=type=secret,id=custom_ca,required=false \
    if [ -s /run/secrets/custom_ca ]; then \
      cp /run/secrets/custom_ca /usr/local/share/ca-certificates/gremienpilot.crt && update-ca-certificates; \
    fi
RUN aria2c --ca-certificate=/etc/ssl/certs/ca-certificates.crt -x 8 -s 8 --file-allocation=none \
    --dir=/tmp --out=llama.tar.gz \
    https://github.com/Seraphiel102/llama.cpp/releases/download/hob-b11434/hob-llama.cpp-b11434-bin-ubuntu-cuda-12.8-x64.tar.gz \
    && echo 'bb888c6c3334cba28e8d55025738627118ba7456325e5bca992f71aa363826d3  /tmp/llama.tar.gz' | sha256sum -c - \
    && mkdir /opt/llama && tar -xzf /tmp/llama.tar.gz -C /opt/llama --strip-components=1 \
    && rm /tmp/llama.tar.gz
RUN aria2c --ca-certificate=/etc/ssl/certs/ca-certificates.crt -x 8 -s 8 --file-allocation=none \
    --dir=/tmp --out=cudart.tar.gz \
    https://github.com/Seraphiel102/llama.cpp/releases/download/hob-b11434/cudart-hob-llama.cpp-b11434-bin-ubuntu-cuda-12.8-x64.tar.gz \
    && echo 'be6b8e5adc7061f5363d17836892a7186356dd3a09a2ea330b6dc5bef00756c9  /tmp/cudart.tar.gz' | sha256sum -c - \
    && tar -xzf /tmp/cudart.tar.gz -C /opt/llama --strip-components=1 \
    && rm /tmp/cudart.tar.gz
COPY entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh
ENV LD_LIBRARY_PATH=/opt/llama:/usr/local/cuda/lib64
EXPOSE 8080
ENTRYPOINT ["/entrypoint.sh"]
