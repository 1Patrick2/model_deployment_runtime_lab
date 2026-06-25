"""HTTP API request/response schemas."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class InferRequest(BaseModel):
    """POST /infer request body."""

    input_type: str = Field(default="dummy", description="dummy | image_path")
    input: str = Field(default="dummy", description="Input path or dummy marker")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of top predictions")


class PredictionItem(BaseModel):
    """A single top-k entry."""

    class_id: int
    class_name: str
    score: float


class LatencyInfo(BaseModel):
    """Latency breakdown in milliseconds."""

    preprocess: float = 0.0
    inference: float = 0.0
    postprocess: float = 0.0
    total: float = 0.0


class InferResponse(BaseModel):
    """POST /infer response."""

    status: str
    backend: str
    model_id: str
    model_variant: str
    input_type: str
    top_k: int
    top_k_predictions: List[PredictionItem]
    latency_ms: LatencyInfo


class HealthResponse(BaseModel):
    """GET /health response."""

    status: str = "ok"


class MetadataResponse(BaseModel):
    """GET /metadata response."""

    model_id: str
    model_variant: str
    backend: str
    task: str
    input_shape: List[int]
    input_dtype: str


__all__ = [
    "InferRequest",
    "PredictionItem",
    "LatencyInfo",
    "InferResponse",
    "HealthResponse",
    "MetadataResponse",
]
