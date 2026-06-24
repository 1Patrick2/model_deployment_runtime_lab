"""Inference request/response protocol definitions.

Defines the wire format for all inference services in the lab using
Pydantic models.  Every backend (fake, onnx, torch, rknn) speaks this
protocol, so client code never depends on backend internals.
"""

from __future__ import annotations

import uuid
from typing import Optional

from pydantic import BaseModel, Field


# ────────────────────────────────────────────── Error type constants ──

INVALID_REQUEST = "INVALID_REQUEST"
UNSUPPORTED_BACKEND = "UNSUPPORTED_BACKEND"
MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
RUNTIME_ERROR = "RUNTIME_ERROR"
TIMEOUT = "TIMEOUT"


# ────────────────────────────────────────────── Request schema ──

class InferenceRequest(BaseModel):
    """Client → Server inference request."""

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    schema_version: str = "inference_request.v1"
    backend: str
    model_id: str = "fake_classifier"
    model_variant: str = "fake_v1"
    input_type: str = "image_path"
    input: str
    return_debug: bool = False


# ────────────────────────────────────────────── Response sub-models ──

class Prediction(BaseModel):
    """Single prediction output from a runner."""

    class_id: int
    class_name: str
    confidence: float


class LatencyMs(BaseModel):
    """Timing breakdown for a single inference call."""

    preprocess: float = 0.0
    inference: float = 0.0
    postprocess: float = 0.0
    total: float = 0.0


# ────────────────────────────────────────────── Response schema ──

class InferenceResponse(BaseModel):
    """Server → Client inference response (success or error)."""

    request_id: str
    schema_version: str = "inference_response.v1"
    status: str  # "ok" | "error"
    prediction: Optional[Prediction] = None
    latency_ms: Optional[LatencyMs] = None
    backend: str
    model_id: Optional[str] = None
    model_variant: Optional[str] = None
    error_type: Optional[str] = None
    message: Optional[str] = None


# ────────────────────────────────────────────── Helpers ──

def make_error_response(
    request_id: str,
    backend: str,
    error_type: str,
    message: str,
) -> InferenceResponse:
    """Convenience factory for building an error InferenceResponse."""
    return InferenceResponse(
        request_id=request_id,
        status="error",
        backend=backend,
        error_type=error_type,
        message=message,
    )


__all__ = [
    "INVALID_REQUEST",
    "UNSUPPORTED_BACKEND",
    "MODEL_NOT_FOUND",
    "RUNTIME_ERROR",
    "TIMEOUT",
    "InferenceRequest",
    "Prediction",
    "LatencyMs",
    "InferenceResponse",
    "make_error_response",
]
