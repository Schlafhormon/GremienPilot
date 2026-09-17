# GremienPilot Kubernetes Deployment

Kubernetes manifests for deploying GremienPilot on the HPI cluster with GPU-accelerated transcription.

## Prerequisites

- `kubectl` configured with access to the HPI Kubernetes cluster
- A node with an NVIDIA A30 GPU (label: `accelerator: a30`)
- NVIDIA runtime class configured (`runtimeClassName: nvidia`)
- An API key for the [AISC LLM API](https://api.aisc.hpi.de)

## Architecture

```
Internet → LoadBalancer Service (tops-frontend)
              ↓
         Frontend (nginx:80) — serves React SPA
              ↓ proxy_pass /api/ & /health
         Backend (FastAPI:8010) — WhisperX transcription + summarization
              ↓ OpenAI SDK
         AISC LLM API (api.aisc.hpi.de) — LLM inference
```

- **Frontend**: nginx serving the React SPA, proxies API requests to the backend
- **Backend**: FastAPI with WhisperX (GPU) for transcription and PyAnnote for diarization; uses the AISC LLM API for summarization via the OpenAI SDK
- **No Ollama needed**: The AISC LLM API replaces the local Ollama instance used in Docker Compose
- **External access**: The checked-in default is a `LoadBalancer` Service. An optional Ingress example is provided but not applied by Kustomize.

## Directory Structure

```
k8s/
├── kustomization.yaml          # Kustomize root — lists all resources
├── namespace.yaml              # tops namespace
├── backend/
│   ├── configmap.yaml          # Whisper + LLM environment variables
│   ├── pvc.yaml                # Persistent backend data/uploads volume
│   ├── deployment.yaml         # GPU pod (A30), WhisperX + FastAPI
│   └── service.yaml            # ClusterIP "backend" on port 8010
├── frontend/
│   ├── deployment.yaml         # CPU pod, nginx + React SPA
│   ├── service.yaml            # LoadBalancer on port 80
│   └── ingress.example.yaml    # Optional Ingress template, not in Kustomize
├── secrets/
│   ├── example-secret.yaml     # Template — copy to secret.yaml
│   └── .gitkeep
├── deploy.sh                   # Ordered deployment script
└── README.md                   # This file
```

## Configuration

### Backend ConfigMap (`backend/configmap.yaml`)

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL` | `large-v2` | WhisperX model size |
| `WHISPER_DEVICE` | `cuda` | Compute device (cuda for GPU) |
| `WHISPER_BATCH_SIZE` | `16` | Transcription batch size |
| `WHISPER_LANGUAGE` | `de` | Audio language |
| `LLM_BASE_URL` | `https://api.aisc.hpi.de` | AISC LLM API endpoint |
| `LLM_MODEL` | `llama-3-3-70b` | Model name for summarization |
| `LLM_TIMEOUT_SECONDS` | `120` | Summary/diagnostics request timeout (not transcript agenda detection) |
| `LLM_MAX_RETRIES` | `2` | Retries for transient LLM errors |
| `LLM_RETRY_BACKOFF_SECONDS` | `0.5` | Backoff between retries |
| `LLM_CHUNK_CHARS` | `12000` | Target chunk size for long TOP transcripts |
| `LLM_STRUCTURED_FALLBACK` | `true` | Use plain-text fallback if structured output parsing fails |
| `PERSISTENCE_DB_PATH` | `/app/data/sessions.sqlite3` | SQLite session database path |
| `JOB_MAX_AGE_SECONDS` | `7200` | Max age for in-memory job cache cleanup |
| `JOB_MAX_COUNT` | `100` | Max jobs retained in memory |
| `MAX_UPLOAD_BYTES` | `524288000` | Maximum upload size |
| `TRANSCRIPTION_CONCURRENCY` | `1` | Concurrent transcription workers |

### Storage

`backend/pvc.yaml` creates a `ReadWriteOnce` PVC named `tops-backend-data`.
The backend mounts it as:

- `/app/data` for the SQLite session database
- `/app/uploads` for retained audio playback after session restore

If your cluster has no default `StorageClass`, set one in `backend/pvc.yaml`
before applying the manifests.

### Secrets

The backend expects a `LLM_API_KEY` for authenticating with the AISC LLM API. Set this up:

```bash
cd k8s/secrets
cp example-secret.yaml secret.yaml
# Edit secret.yaml — replace YOUR_AISC_API_KEY_HERE with a real key
kubectl apply -f secret.yaml
```

The secret is referenced as `optional: true` in the deployment, so the pod will still start without it (defaulting to `"ollama"`).

### External Access

By default, `frontend/service.yaml` exposes the app as a `LoadBalancer` Service:

```bash
kubectl get svc -n tops tops-frontend
```

If your cluster uses an Ingress controller instead, copy
`frontend/ingress.example.yaml`, set the real host and ingress class, and apply
it separately. The example is intentionally not included in `kustomization.yaml`
so existing LoadBalancer deployments keep working.

## Deployment

### Quick Start

```bash
cd k8s
./deploy.sh
```

The script will:
1. Create the `tops` namespace
2. Apply secrets (if `secrets/secret.yaml` exists)
3. Apply all Kustomize manifests
4. Wait for frontend readiness (~10s)
5. Wait for backend readiness (~3-5 min while ML models load)

### Manual Deployment

```bash
kubectl apply -f namespace.yaml
kubectl apply -f secrets/secret.yaml          # if you have one
kubectl apply -k .
```

## Verification

```bash
# Check pod status
kubectl get pods -n tops

# Watch backend model loading
kubectl logs -f -n tops -l app=tops-backend

# Port-forward to test locally
kubectl port-forward -n tops svc/tops-frontend 3000:80
# Open http://localhost:3000
```

## Updating the Application

When new container images are pushed to GHCR (handled by the existing GitHub Actions pipeline on push to `main`):

```bash
# The GitHub Actions deploy job pins manifests to commit-specific sha tags.
# Apply the updated manifests, then watch the rollout:
kubectl apply -k k8s
kubectl rollout status deployment/tops-backend -n tops
kubectl rollout status deployment/tops-frontend -n tops
```

## Troubleshooting

### Backend pod stuck in Pending
- Check GPU availability: `kubectl describe node -l accelerator=a30`
- Verify the NVIDIA runtime class exists: `kubectl get runtimeclass nvidia`

### Backend pod CrashLoopBackOff
- Check logs: `kubectl logs -n tops -l app=tops-backend --previous`
- The startup probe allows up to ~5 minutes for model loading — if it's crashing before that, it's likely an OOM or GPU issue

### LLM summarization fails
- Test connectivity from the backend pod:
  ```bash
  kubectl exec -n tops deploy/tops-backend -- curl -s https://api.aisc.hpi.de/health
  ```
- Check the API key is set: `kubectl get secret tops-secret -n tops -o yaml`

### Frontend returns 502 for API requests
- The frontend nginx proxies to `http://backend:8010` — verify the backend service exists:
  ```bash
  kubectl get svc backend -n tops
  ```
- Check backend pod is ready: `kubectl get pods -n tops -l app=tops-backend`

## Relationship to Docker Compose

| Concern | Docker Compose | Kubernetes |
|---------|---------------|------------|
| LLM inference | Local Ollama container | AISC LLM API (api.aisc.hpi.de) |
| GPU access | Docker `--gpus` / compose `deploy.resources` | `runtimeClassName: nvidia` + `nodeSelector` |
| Backend image | `backend:cpu-latest` or `backend:gpu-latest` | Commit-pinned `sha-...-gpu` image (always GPU) |
| Networking | Docker network (service names) | K8s Services + DNS |
| Storage | `./uploads` and `./data` bind mounts | `tops-backend-data` PVC mounted at `/app/uploads` and `/app/data` |
| External access | `localhost:3000` | LoadBalancer by default; optional Ingress example |

Agenda detection uses the `AGENDA_DETECTION_*` values in
`backend/configmap.yaml` (LLM default off, 8-second timeout per chunk, no SDK
retries). Explicit API `use_llm` / pipeline `agenda_use_llm` overrides the server
default; model/prompt alone does not enable it. See the [configuration contract](../README.md#llm-nutzung-für-automatische-top-zuordnung).
Restart backend pods after ConfigMap changes to load the new values.
