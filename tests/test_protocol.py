"""Tests for inference request/response protocol."""

import json
import pytest
from pydantic import ValidationError

from src.server.protocol import (
    INVALID_REQUEST,
    UNSUPPORTED_BACKEND,
    InferenceRequest,
    InferenceResponse,
    Prediction,
    LatencyMs,
    make_error_response,
)


class TestInferenceRequest:
    """InferenceRequest parsing and validation."""

    def test_valid_request_can_be_parsed(self):
        payload = {
            "request_id": "req-001",
            "backend": "fake",
            "input": "samples/images/test.jpg",
        }
        req = InferenceRequest(**payload)
        assert req.request_id == "req-001"
        assert req.backend == "fake"
        assert req.input == "samples/images/test.jpg"
        assert req.schema_version == "inference_request.v1"

    def test_request_id_is_auto_generated_when_missing(self):
        req = InferenceRequest(backend="fake", input="test.jpg")
        assert req.request_id is not None
        assert isinstance(req.request_id, str)
        assert len(req.request_id) > 0

    def test_missing_backend_raises_error(self):
        with pytest.raises(ValidationError):
            InferenceRequest(input="test.jpg")

    def test_missing_input_raises_error(self):
        with pytest.raises(ValidationError):
            InferenceRequest(backend="fake")


class TestInferenceResponse:
    """InferenceResponse construction and serialization."""

    def test_ok_response_can_be_serialized_to_dict(self):
        resp = InferenceResponse(
            request_id="req-001",
            status="ok",
            backend="fake",
            prediction=Prediction(class_id=1, class_name="warning", confidence=0.87),
            latency_ms=LatencyMs(preprocess=0.1, inference=0.2, postprocess=0.1, total=0.4),
            model_id="fake_classifier",
            model_variant="fake_v1",
        )
        data = resp.model_dump()
        assert data["request_id"] == "req-001"
        assert data["status"] == "ok"
        assert data["prediction"]["class_id"] == 1
        assert data["latency_ms"]["total"] == 0.4

    def test_ok_response_can_be_serialized_to_json(self):
        resp = InferenceResponse(
            request_id="req-001",
            status="ok",
            backend="fake",
            prediction=Prediction(class_id=0, class_name="safe", confidence=0.95),
        )
        raw = resp.model_dump_json()
        data = json.loads(raw)
        assert data["request_id"] == "req-001"
        assert data["prediction"]["class_name"] == "safe"

    def test_error_response_has_no_prediction(self):
        resp = make_error_response(
            request_id="req-001",
            backend="unknown",
            error_type=UNSUPPORTED_BACKEND,
            message="backend 'abc' is not supported",
        )
        assert resp.status == "error"
        assert resp.prediction is None
        assert resp.error_type == UNSUPPORTED_BACKEND
        assert "not supported" in resp.message

    def test_latency_ms_field_exists_on_ok(self):
        resp = InferenceResponse(
            request_id="r",
            status="ok",
            backend="fake",
            prediction=Prediction(class_id=0, class_name="safe", confidence=1.0),
            latency_ms=LatencyMs(total=0.5),
        )
        assert resp.latency_ms is not None
        assert resp.latency_ms.total == 0.5

    def test_error_response_can_be_serialized_to_json(self):
        resp = make_error_response(
            request_id="req-002",
            backend="fake",
            error_type=INVALID_REQUEST,
            message="missing field",
        )
        raw = resp.model_dump_json()
        data = json.loads(raw)
        assert data["status"] == "error"
        assert data["error_type"] == INVALID_REQUEST
