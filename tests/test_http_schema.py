"""Tests for HTTP request/response schemas."""

import json

import pytest
from pydantic import ValidationError

from src.server.http_schema import InferRequest, InferResponse, HealthResponse, MetadataResponse, PredictionItem, LatencyInfo
from src.runtime.classification_postprocess import top_k_to_dicts, TopKPrediction


class TestInferRequest:
    """Request schema validation."""

    def test_defaults(self):
        req = InferRequest()
        assert req.input_type == "dummy"
        assert req.input == "dummy"
        assert req.top_k == 5

    def test_can_be_parsed_from_dict(self):
        data = {"input_type": "image_path", "input": "test.jpg", "top_k": 3}
        req = InferRequest(**data)
        assert req.input_type == "image_path"
        assert req.input == "test.jpg"
        assert req.top_k == 3

    def test_top_k_below_minimum_raises_error(self):
        with pytest.raises(ValidationError):
            InferRequest(top_k=0)

    def test_top_k_above_maximum_raises_error(self):
        with pytest.raises(ValidationError):
            InferRequest(top_k=100)


class TestInferResponse:
    """Response schema construction and serialization."""

    def test_response_can_be_built(self):
        resp = InferResponse(
            status="ok",
            backend="onnx",
            model_id="test",
            model_variant="v1",
            input_type="dummy",
            top_k=2,
            top_k_predictions=[
                PredictionItem(class_id=1, class_name="cat", score=0.9),
                PredictionItem(class_id=2, class_name="dog", score=0.05),
            ],
            latency_ms=LatencyInfo(preprocess=0.1, inference=1.0, postprocess=0.1, total=1.2),
        )
        data = json.loads(resp.model_dump_json())
        assert data["status"] == "ok"
        assert len(data["top_k_predictions"]) == 2
        assert data["latency_ms"]["total"] == 1.2

    def test_health_response(self):
        resp = HealthResponse()
        assert resp.status == "ok"

    def test_metadata_response(self):
        resp = MetadataResponse(
            model_id="test", model_variant="v1", backend="onnx",
            task="classification", input_shape=[1, 3, 224, 224], input_dtype="float32",
        )
        assert resp.model_id == "test"


class TestTopKToDicts:
    """Top-k prediction list serialization."""

    def test_conversion(self):
        preds = [TopKPrediction(1, "cat", 0.9), TopKPrediction(2, "dog", 0.05)]
        dicts = top_k_to_dicts(preds)
        assert dicts[0]["class_name"] == "cat"
        assert dicts[1]["class_id"] == 2
