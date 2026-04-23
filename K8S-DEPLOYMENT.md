# Kubernetes Deployment for Biomedical AI Assistant

> This extends the project with production-grade K8s manifests.
> Run locally on minikube (M4 Mac) — same patterns apply to SciLifeLab's HPC clusters.

## Prerequisites

```bash
# Install on Mac
brew install minikube helm kubectl

# Start minikube with enough resources for ML workloads
minikube start --cpus=4 --memory=8192 --driver=docker

# Enable addons
minikube addons enable ingress
minikube addons enable metrics-server
```

---

## Project structure (K8s additions)

```
scilifelab-ai-assistant/
├── k8s/
│   ├── namespace.yaml
│   ├── model-pvc.yaml              # Persistent volume for model weights
│   ├── configmap.yaml              # Model config + versioning
│   ├── secrets.yaml                # API keys (LLM provider, etc.)
│   ├── deployment-api.yaml         # FastAPI deployment
│   ├── deployment-agent.yaml       # LangChain agent worker
│   ├── service.yaml                # ClusterIP services
│   ├── ingress.yaml                # Ingress routing
│   ├── hpa.yaml                    # Horizontal Pod Autoscaler
│   ├── servicemonitor.yaml         # Prometheus auto-discovery
│   └── init-job.yaml               # Model download job
├── helm/
│   └── scilifelab-ai/
│       ├── Chart.yaml
│       ├── values.yaml
│       └── templates/
│           ├── deployment.yaml
│           ├── service.yaml
│           ├── ingress.yaml
│           ├── pvc.yaml
│           ├── configmap.yaml
│           ├── hpa.yaml
│           └── _helpers.tpl
└── Dockerfile.multi               # Multi-stage production build
```

---

## 1. Multi-stage Dockerfile (ML-optimized)

This is different from a regular app — you need to handle model weights efficiently.

```dockerfile
# Dockerfile.multi
# Stage 1: Download and cache model weights at build time
FROM python:3.11-slim AS model-builder

RUN pip install --no-cache-dir transformers torch --index-url https://download.pytorch.org/whl/cpu

# Pre-download the base model so it's baked into the image
# In production, you'd pull your fine-tuned model from a model registry
RUN python -c "
from transformers import AutoTokenizer, AutoModelForSequenceClassification
AutoTokenizer.from_pretrained('distilbert-base-uncased')
AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=5)
print('Model cached successfully')
"

# Stage 2: Production runtime
FROM python:3.11-slim AS runtime

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy cached models from builder stage
COPY --from=model-builder /root/.cache/huggingface /root/.cache/huggingface

# Copy application code
COPY api/ ./api/
COPY agent/ ./agent/

# Non-root user (security best practice)
RUN useradd -m appuser
USER appuser

# Health check for Kubernetes probes
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

**Why this matters for SciLifeLab:** Models are often gigabytes. Baking them into the image (or using init containers to pull from a model registry) avoids cold-start delays when pods scale up. This is a real production pattern for ML serving.

---

## 2. Kubernetes Manifests

### Namespace

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: biomedical-ai
  labels:
    app.kubernetes.io/part-of: scilifelab-ai
    team: data-centre
```

### PersistentVolumeClaim for model weights

This is **ML-specific** — regular web apps don't need persistent storage for multi-GB model files.

```yaml
# k8s/model-pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-weights
  namespace: biomedical-ai
  labels:
    app.kubernetes.io/component: model-storage
spec:
  accessModes:
    - ReadOnlyMany # Multiple pods read the same model weights
  resources:
    requests:
      storage: 5Gi # DistilBERT is ~250MB; room for multiple versions
  storageClassName: standard
```

**Key ML detail:** `ReadOnlyMany` — multiple API pods share the same model weights without each needing their own copy. At SciLifeLab, with larger models (ESM-2, AlphaFold), this saves significant storage.

### ConfigMap for model versioning

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: model-config
  namespace: biomedical-ai
data:
  # Model versioning — critical for reproducibility in research
  MODEL_NAME: "distilbert-pubmed-rct"
  MODEL_VERSION: "v1.0.0"
  MODEL_PATH: "/models/best_model"
  BASE_MODEL: "distilbert-base-uncased"
  NUM_LABELS: "5"
  MAX_SEQUENCE_LENGTH: "128"

  # Inference settings — tune these for latency vs throughput
  BATCH_SIZE: "32"
  NUM_WORKERS: "2"

  # Agent configuration
  AGENT_LLM_MODEL: "llama3.2:3b"
  AGENT_MAX_ITERATIONS: "5"
  PUBMED_MAX_RESULTS: "3"
```

### Secrets

```yaml
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: api-secrets
  namespace: biomedical-ai
type: Opaque
stringData:
  # In production: use external secret managers (Vault, AWS SM)
  PUBMED_EMAIL: "researcher@scilifelab.se"
  # If using hosted LLM instead of local Ollama:
  OPENAI_API_KEY: ""
```

### Init Job — download fine-tuned model

This pattern separates model download from serving — a pod doesn't need to download 500MB on every restart.

```yaml
# k8s/init-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: model-init
  namespace: biomedical-ai
spec:
  template:
    spec:
      containers:
        - name: model-downloader
          image: python:3.11-slim
          command:
            - python
            - -c
            - |
              from transformers import AutoTokenizer, AutoModelForSequenceClassification
              import os

              model_path = "/models/best_model"
              os.makedirs(model_path, exist_ok=True)

              # In production: pull from your model registry (MLflow, S3, etc.)
              print("Downloading model...")
              tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
              model = AutoModelForSequenceClassification.from_pretrained(
                  "distilbert-base-uncased", num_labels=5
              )

              tokenizer.save_pretrained(model_path)
              model.save_pretrained(model_path)
              print(f"Model saved to {model_path}")
          volumeMounts:
            - name: model-weights
              mountPath: /models
      volumes:
        - name: model-weights
          persistentVolumeClaim:
            claimName: model-weights
      restartPolicy: Never
  backoffLimit: 3
```

### Deployment — API server

```yaml
# k8s/deployment-api.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: biomedical-api
  namespace: biomedical-ai
  labels:
    app.kubernetes.io/name: biomedical-api
    app.kubernetes.io/component: api-server
spec:
  replicas: 2
  selector:
    matchLabels:
      app.kubernetes.io/name: biomedical-api
  template:
    metadata:
      labels:
        app.kubernetes.io/name: biomedical-api
      annotations:
        # Tell Prometheus to scrape this pod
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      # Init container: wait for model weights to be ready
      initContainers:
        - name: wait-for-model
          image: busybox:1.36
          command:
            [
              "sh",
              "-c",
              'until [ -f /models/best_model/config.json ]; do echo "Waiting for model weights..."; sleep 5; done',
            ]
          volumeMounts:
            - name: model-weights
              mountPath: /models
              readOnly: true

      containers:
        - name: api
          image: scilifelab-ai/biomedical-api:v1.0.0
          ports:
            - containerPort: 8000
              name: http
              protocol: TCP

          envFrom:
            - configMapRef:
                name: model-config
            - secretRef:
                name: api-secrets

          volumeMounts:
            - name: model-weights
              mountPath: /models
              readOnly: true # Model weights are read-only at inference time

          # --- ML-specific resource management ---
          resources:
            requests:
              cpu: "500m"
              memory: "1Gi" # DistilBERT needs ~500MB in memory
            limits:
              cpu: "2"
              memory: "2Gi"
              # For GPU-accelerated models at SciLifeLab:
              # nvidia.com/gpu: "1"

          # Liveness: is the process alive?
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30 # Model loading takes time
            periodSeconds: 15
            failureThreshold: 3

          # Readiness: is the model loaded and ready to serve?
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 45 # Higher than liveness — model must be loaded
            periodSeconds: 10
            failureThreshold: 5

          # Startup: give the pod time to load the model on first boot
          startupProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 10
            failureThreshold: 12 # 12 * 10s = 2 min max startup time

      volumes:
        - name: model-weights
          persistentVolumeClaim:
            claimName: model-weights
```

**ML-specific details you should understand for the interview:**

1. **`startupProbe`** — ML models take 10-60 seconds to load into memory. Without a startup probe, Kubernetes kills the pod before the model finishes loading. Regular web apps don't need this.

2. **`readinessProbe` with higher `initialDelaySeconds`** — The pod shouldn't receive traffic until the model is actually in memory. Your `/health` endpoint should verify this.

3. **Memory `requests`** — A model consumes a fixed amount of RAM once loaded. DistilBERT is ~250MB, but ESM-2 (protein model) is 8GB+. You must size requests accurately or pods get OOMKilled.

4. **`ReadOnlyMany` PVC** — All replicas share the same model weights. No need to download per-pod.

### Deployment — Agent Worker

```yaml
# k8s/deployment-agent.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-worker
  namespace: biomedical-ai
  labels:
    app.kubernetes.io/name: agent-worker
    app.kubernetes.io/component: agent
spec:
  replicas: 1
  selector:
    matchLabels:
      app.kubernetes.io/name: agent-worker
  template:
    metadata:
      labels:
        app.kubernetes.io/name: agent-worker
    spec:
      containers:
        - name: agent
          image: scilifelab-ai/biomedical-api:v1.0.0
          command: ["python", "-m", "agent.agent"]

          envFrom:
            - configMapRef:
                name: model-config
            - secretRef:
                name: api-secrets

          resources:
            requests:
              cpu: "1"
              memory: "2Gi" # Agent + LLM needs more memory
            limits:
              cpu: "4"
              memory: "4Gi"
```

### Service

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: biomedical-api
  namespace: biomedical-ai
  labels:
    app.kubernetes.io/name: biomedical-api
spec:
  type: ClusterIP
  ports:
    - port: 80
      targetPort: 8000
      protocol: TCP
      name: http
  selector:
    app.kubernetes.io/name: biomedical-api
```

### Ingress

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: biomedical-api
  namespace: biomedical-ai
  annotations:
    # Rate limiting for ML endpoints (inference is expensive)
    nginx.ingress.kubernetes.io/limit-rps: "10"
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"
    # Longer timeout for agent queries (LLM reasoning takes time)
    nginx.ingress.kubernetes.io/proxy-read-timeout: "120"
spec:
  ingressClassName: nginx
  rules:
    - host: biomedical-ai.local
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: biomedical-api
                port:
                  number: 80
```

**ML-specific Ingress detail:** `proxy-read-timeout: 120` — agent queries with LLM reasoning can take 30-60 seconds. Default nginx timeout is 60s. Without this, complex queries get 504 errors.

### HorizontalPodAutoscaler

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: biomedical-api
  namespace: biomedical-ai
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: biomedical-api
  minReplicas: 2
  maxReplicas: 8
  metrics:
    # Scale on CPU (model inference is CPU-bound without GPU)
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    # Scale on memory (each model instance holds the full model in RAM)
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60 # Wait 1 min before scaling up
      policies:
        - type: Pods
          value: 2
          periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300 # Wait 5 min before scaling down
      policies: # ML pods are expensive to restart
        - type: Pods
          value: 1
          periodSeconds: 120
```

**ML-specific HPA detail:** Slow scale-down (`stabilizationWindowSeconds: 300`). ML pods take 30-60 seconds to start serving because the model must load into memory. Aggressive scale-down means cold starts on the next traffic spike.

### ServiceMonitor (Prometheus Operator)

```yaml
# k8s/servicemonitor.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: biomedical-api
  namespace: biomedical-ai
spec:
  selector:
    matchLabels:
      app.kubernetes.io/name: biomedical-api
  endpoints:
    - port: http
      path: /metrics
      interval: 15s
```

---

## 3. Helm Chart

Package everything into a reusable Helm chart — this is how SciLifeLab would deploy multiple model-serving stacks.

### Chart.yaml

```yaml
# helm/scilifelab-ai/Chart.yaml
apiVersion: v2
name: scilifelab-ai
description: Biomedical AI model serving platform
version: 0.1.0
appVersion: "1.0.0"
keywords:
  - machine-learning
  - model-serving
  - biomedical
  - scilifelab
```

### values.yaml

```yaml
# helm/scilifelab-ai/values.yaml

# --- Model configuration ---
model:
  name: distilbert-pubmed-rct
  version: v1.0.0
  path: /models/best_model
  baseModel: distilbert-base-uncased
  numLabels: 5
  maxSequenceLength: 128
  # For GPU models at SciLifeLab:
  # gpu:
  #   enabled: true
  #   count: 1
  #   type: nvidia.com/gpu

# --- API server ---
api:
  replicaCount: 2
  image:
    repository: scilifelab-ai/biomedical-api
    tag: v1.0.0
    pullPolicy: IfNotPresent
  resources:
    requests:
      cpu: 500m
      memory: 1Gi
    limits:
      cpu: "2"
      memory: 2Gi
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 8
    targetCPUUtilization: 70

# --- Agent worker ---
agent:
  enabled: true
  replicaCount: 1
  llmModel: llama3.2:3b
  maxIterations: 5

# --- Storage ---
storage:
  modelWeights:
    size: 5Gi
    storageClassName: standard
    accessMode: ReadOnlyMany

# --- Monitoring ---
monitoring:
  enabled: true
  serviceMonitor:
    enabled: true
    interval: 15s

# --- Ingress ---
ingress:
  enabled: true
  host: biomedical-ai.local
  rateLimitRPS: 10
  readTimeout: 120
```

---

## 4. Deploy it

```bash
# Build the image
docker build -f Dockerfile.multi -t scilifelab-ai/biomedical-api:v1.0.0 .

# Load into minikube
minikube image load scilifelab-ai/biomedical-api:v1.0.0

# Apply raw manifests (quick way)
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/

# OR deploy with Helm (production way)
helm install scilifelab-ai ./helm/scilifelab-ai \
  --namespace biomedical-ai \
  --create-namespace

# Verify
kubectl get pods -n biomedical-ai -w

# Port-forward to test
kubectl port-forward svc/biomedical-api -n biomedical-ai 8000:80

# Test it
curl http://localhost:8000/health
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "We enrolled 500 patients in a randomized controlled trial."}'
```

---

## 5. Key ML-on-K8s patterns to discuss in the interview

### Pattern 1: Model versioning with rolling updates

```bash
# Update the model version in ConfigMap
kubectl edit configmap model-config -n biomedical-ai
# Change MODEL_VERSION to v1.1.0

# Rolling update — zero downtime model swap
kubectl rollout restart deployment/biomedical-api -n biomedical-ai
kubectl rollout status deployment/biomedical-api -n biomedical-ai
```

### Pattern 2: Canary deployment for new models

```yaml
# Deploy v2 model alongside v1, route 10% traffic
# This is critical for research — you need to A/B test models
apiVersion: apps/v1
kind: Deployment
metadata:
  name: biomedical-api-canary
  namespace: biomedical-ai
spec:
  replicas: 1 # 1 canary vs 2 stable = ~33% traffic
  selector:
    matchLabels:
      app.kubernetes.io/name: biomedical-api
      track: canary
  template:
    metadata:
      labels:
        app.kubernetes.io/name: biomedical-api
        track: canary
    spec:
      containers:
        - name: api
          image: scilifelab-ai/biomedical-api:v2.0.0-rc1
          env:
            - name: MODEL_VERSION
              value: "v2.0.0-rc1"
```

### Pattern 3: GPU scheduling (for SciLifeLab's HPC)

```yaml
# When deploying larger models (ESM-2, AlphaFold)
resources:
  limits:
    nvidia.com/gpu: 1
nodeSelector:
  accelerator: nvidia-a100 # Target GPU nodes
tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
```

### Pattern 4: Model health beyond HTTP 200

```python
# Enhanced health check that verifies model is actually working
@app.get("/health")
def health_check():
    try:
        clf = get_classifier()
        # Run a smoke test prediction
        test_result = clf("test sentence", truncation=True, max_length=128)
        return {
            "status": "healthy",
            "model_loaded": True,
            "model_version": os.getenv("MODEL_VERSION", "unknown"),
            "smoke_test": "passed",
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )
```

---
