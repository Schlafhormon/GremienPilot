# syntax=docker/dockerfile:1.7
# Backend Dockerfile for CPU-only deployment with optional pre-bundled ML models.
# For GPU support, use Dockerfile.gpu.
#
# Build with pre-cached models:
#   DOCKER_BUILDKIT=1 docker build --secret id=hf_token,env=HF_TOKEN -t backend:cpu .
# Build without pre-cached models (HF_TOKEN required at runtime for diarization):
#   DOCKER_BUILDKIT=1 docker build --build-arg PRECACHE_MODELS=0 -t backend:cpu .

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (ffmpeg required by whisperx, curl for health checks).
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Optional local CA for HTTPS inspection (provided by setup.ps1 via BuildKit).
RUN --mount=type=secret,id=custom_ca,required=false \
    if [ -s /run/secrets/custom_ca ]; then \
      cp /run/secrets/custom_ca /usr/local/share/ca-certificates/gremienpilot.crt && \
      update-ca-certificates; \
    fi
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    PIP_CERT=/etc/ssl/certs/ca-certificates.crt

# Install uv for package management
RUN pip install --no-cache-dir uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY *.py ./

# Install dependencies
ENV UV_NO_CACHE=1
# Use the container's trusted CA certificates, including installed proxy CAs.
ENV UV_SYSTEM_CERTS=true
RUN uv pip install --system -e .

# Create runtime directories after editable install so setuptools package
# discovery does not treat them as Python packages.
RUN mkdir -p uploads data

# Public tokenizer only (~13 MB); no Hugging Face token or model weights needed.
COPY third_party ./third_party
RUN python llm_assets.py

# Set cache directories inside the container
ENV HF_HOME=/app/.cache/huggingface
ENV TORCH_HOME=/app/.cache/torch

# Download all ML models at build time when enabled. The HuggingFace token is
# provided via a BuildKit secret and is never persisted as an ARG, ENV or layer.
ARG PRECACHE_MODELS=1
RUN --mount=type=secret,id=hf_token,required=false \
    if [ "${PRECACHE_MODELS}" = "1" ]; then \
      if [ -f /run/secrets/hf_token ]; then export HF_TOKEN="$(cat /run/secrets/hf_token)"; fi; \
      python -c "from transcribe import load_models; load_models()"; \
    else \
      echo "Skipping model pre-cache; provide HF_TOKEN at runtime if diarization requires it."; \
    fi

# Set flag indicating models are pre-cached (runtime won't require HF_TOKEN)
ENV MODELS_PRECACHED=${PRECACHE_MODELS}

# Expose port
EXPOSE 8010

# Health check with reduced start_period since models are pre-loaded
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=120s \
    CMD curl -f http://localhost:8010/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8010"]
