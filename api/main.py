"""
Production API for the biomedical research assistant.
Includes health checks, metrics, and structured logging.
"""

import time
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from agent.tools import get_classifier

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("biomedical-api")

# --- Prometheus metrics ---
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total API requests",
    ["endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "api_request_duration_seconds",
    "Request latency in seconds",
    ["endpoint"],
)
MODEL_PREDICTIONS = Counter(
    "model_predictions_total",
    "Total model predictions",
    ["predicted_label"],
)

# --- App ---
app = FastAPI(
    title="SciLifeLab Biomedical Research Assistant",
    description="AI-powered tools for life science researchers",
    version="0.1.0",
)


# --- Request/Response models ---
class ClassifyRequest(BaseModel):
    text: str

    class Config:
        json_schema_extra = {
            "example": {
                "text": "We recruited 500 patients for a randomized controlled trial."
            }
        }


class ClassifyResponse(BaseModel):
    label: str
    confidence: float
    text: str


# --- Endpoints ---
@app.get("/health")
def health_check():
    """Health check endpoint for Kubernetes liveness/readiness probes."""
    return {"status": "healthy", "model_loaded": get_classifier() is not None}


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint for monitoring."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/classify", response_model=ClassifyResponse)
def classify(request: ClassifyRequest):
    """Classify a biomedical text sentence."""
    start_time = time.time()

    try:
        clf = get_classifier()
        result = clf(request.text, truncation=True, max_length=128)
        label = result[0]["label"]
        score = result[0]["score"]

        # Track metrics
        MODEL_PREDICTIONS.labels(predicted_label=label).inc()
        REQUEST_COUNT.labels(endpoint="/classify", status="success").inc()
        REQUEST_LATENCY.labels(endpoint="/classify").observe(time.time() - start_time)

        logger.info(f"Classified: '{request.text[:50]}...' → {label} ({score:.2%})")

        return ClassifyResponse(
            label=label,
            confidence=score,
            text=request.text,
        )

    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/classify", status="error").inc()
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
def model_info():
    """Return model metadata — useful for model versioning."""
    return {
        "model_name": "distilbert-pubmed-rct",
        "base_model": "distilbert-base-uncased",
        "task": "sequence-classification",
        "labels": ["BACKGROUND", "CONCLUSIONS", "METHODS", "OBJECTIVE", "RESULTS"],
        "training_data": "pubmed-rct20k",
        "version": "0.1.0",
    }
