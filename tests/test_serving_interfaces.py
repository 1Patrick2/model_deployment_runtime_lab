"""Consolidated tests -- merged from test_zmq_client.py, test_zmq_server_fake_runtime.py, test_http_schema.py, test_http_benchmark.py."""

# -- From test_zmq_client.py --
"""Tests for the ZMQ server runner factory and client payload builder."""

from pathlib import Path

import pytest

from src.server.zmq_client import build_request_payload


class TestBuildRequestPayload:
    """Client payload construction — no ZMQ socket needed."""

    def test_default_input_type_is_image_path(self):
        payload = build_request_payload(backend="fake", input_value="test.jpg")
        assert payload["backend"] == "fake"
        assert payload["input_type"] == "image_path"
        assert payload["input"] == "test.jpg"

    def test_dummy_input_type(self):
        payload = build_request_payload(
            backend="onnx", input_value="dummy", input_type="dummy"
        )
        assert payload["backend"] == "onnx"
        assert payload["input_type"] == "dummy"
        assert payload["input"] == "dummy"

    def test_request_id_is_generated(self):
        payload = build_request_payload(backend="fake", input_value="x.jpg")
        assert "request_id" in payload
# -- From test_zmq_server_fake_runtime.py --
"""Tests for the ZMQ server's request handler.

These tests exercise ``handle_request_json`` directly without a real
ZMQ socket — fast, deterministic, and CI-friendly.
"""

from pathlib import Path

import pytest

from src.runtime.fake_runner import FakeRunner
from src.runtime.onnx_runner import OnnxRunner
from src.server.protocol import (
    INVALID_REQUEST,
    UNSUPPORTED_BACKEND,
    RUNTIME_ERROR,
)
from src.server.zmq_server import handle_request_json, create_runner


@pytest.fixture
def runner():
    r = FakeRunner()
    r.load()
    yield r
    r.close()


class TestHandleRequestJson:
    """Core request dispatch logic."""

    def test_valid_request_returns_ok(self, runner):
        payload = {
            "request_id": "ut-001",
            "backend": "fake",
            "input": "samples/images/danger_scene.jpg",
        }
        result = handle_request_json(payload, runner)
        assert result["status"] == "ok"
        assert result["prediction"]["class_name"] == "danger"

    def test_missing_fields_return_error(self, runner):
        payload = {"backend": "fake"}  # no input
        result = handle_request_json(payload, runner)
        assert result["status"] == "error"
        assert result["error_type"] in (INVALID_REQUEST, RUNTIME_ERROR)

    def test_empty_payload_returns_error(self, runner):
        result = handle_request_json({}, runner)
        assert result["status"] == "error"

    def test_request_id_is_preserved_on_success(self, runner):
        payload = {"request_id": "my-id-42", "backend": "fake", "input": "test.jpg"}
        result = handle_request_json(payload, runner)
        assert result["request_id"] == "my-id-42"

    def test_request_id_is_preserved_on_error(self, runner):
        payload = {"request_id": "err-99", "backend": "fake"}
        result = handle_request_json(payload, runner)
        assert result["request_id"] == "err-99"
        assert result["status"] == "error"

    def test_prediction_is_none_on_error(self, runner):
        payload = {"request_id": "r", "backend": "fake"}
        result = handle_request_json(payload, runner)
        assert result["prediction"] is None

    def test_latency_ms_included_on_success(self, runner):
        payload = {"request_id": "r", "backend": "fake", "input": "test.jpg"}
        result = handle_request_json(payload, runner)
        assert result["latency_ms"] is not None
        assert result["latency_ms"]["total"] > 0

    def test_backend_mismatch_returns_error(self, runner):
        """Request with backend=onnx sent to a fake runner should be rejected."""
        payload = {
            "request_id": "bad-backend",
            "backend": "onnx",
            "input": "test.jpg",
        }
        result = handle_request_json(payload, runner)
        assert result["status"] == "error"
        assert result["error_type"] == UNSUPPORTED_BACKEND
        assert result["request_id"] == "bad-backend"


class TestCreateRunner:
    """Runner factory."""

    def test_fake_runner_can_be_created(self):
        runner = create_runner("fake")
        assert runner.backend_name == "fake"
        runner.close()

    def test_onnx_backend_requires_manifest(self):
        with pytest.raises(ValueError, match="requires --manifest"):
            create_runner("onnx")

    def test_unknown_backend_raises_error(self):
        with pytest.raises(ValueError, match="unsupported backend"):
            create_runner("nonexistent")


# ── Optional ONNX smoke test ─────────────────────────────────────


@pytest.mark.skipif(
    not Path("outputs/onnx/mobilenetv3_small.onnx").exists(),
    reason="ONNX artifact not generated; run export_onnx.py first",
)
class TestHandleRequestJsonWithOnnxRunner:
    """These tests only run when the ONNX artifact is present locally."""

    @pytest.fixture(autouse=True)
    def _setup_runner(self):
        self.runner = OnnxRunner(
            manifest_path="models/manifests/mobilenetv3_small_onnx_fp32.json",
        )
        self.runner.load()
        yield
        self.runner.close()

    def test_dummy_request_returns_ok(self):
        payload = {
            "backend": "onnx",
            "input_type": "dummy",
            "input": "dummy",
        }
        result = handle_request_json(payload, self.runner)
        assert result["status"] == "ok"
        assert result["backend"] == "onnx"
        assert result["prediction"] is not None
        assert result["latency_ms"]["total"] > 0
# -- From test_http_schema.py --
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
# -- From test_http_benchmark.py --
"""Tests for HTTP benchmark — no real server needed."""

from pathlib import Path

import pytest
import yaml

from src.benchmark.http_benchmark import (
    make_health_endpoint,
    run_http_benchmark,
    send_infer_request,
    _write_http_markdown,
)


class TestMakeHealthEndpoint:
    """Health endpoint derivation."""

    def test_from_infer_path(self):
        assert (
            make_health_endpoint("http://127.0.0.1:8001/infer")
            == "http://127.0.0.1:8001/health"
        )

    def test_from_root_path(self):
        assert (
            make_health_endpoint("http://127.0.0.1:8001")
            == "http://127.0.0.1:8001/health"
        )


class TestRunHttpBenchmarkAllSuccess:
    """All requests succeed."""

    def test_success_counts_and_latency(self, monkeypatch):
        call_count = 0

        def fake_send(endpoint, payload, timeout_sec):
            nonlocal call_count
            call_count += 1
            return 5.0, {
                "status": "ok",
                "backend": "onnx",
                "model_id": "mobilenetv3_small",
                "model_variant": "onnx_fp32_v1",
                "top_k_predictions": [
                    {"class_id": 1, "class_name": "goldfish", "score": 0.9}
                ],
                "latency_ms": {
                    "preprocess": 0.1,
                    "inference": 1.5,
                    "postprocess": 0.2,
                    "total": 1.8,
                },
            }

        monkeypatch.setattr(
            "src.benchmark.http_benchmark.send_infer_request",
            fake_send,
        )

        config = {
            "endpoint": "http://127.0.0.1:8001/infer",
            "request": {"input_type": "dummy", "input": "dummy", "top_k": 5},
            "benchmark": {"warmup": 2, "repeat": 3, "timeout_sec": 10},
        }

        report = run_http_benchmark(config)

        assert call_count == 5  # warmup 2 + repeat 3
        assert report["success_count"] == 3
        assert report["failure_count"] == 0
        assert report["success_rate"] == 1.0
        assert report["backend"] == "onnx"
        assert report["model_id"] == "mobilenetv3_small"
        assert report["server_total_ms"]["mean"] == 1.8
        assert report["client_total_ms"]["mean"] == 5.0
        assert report["sample_predictions"][0]["class_name"] == "goldfish"


class TestRunHttpBenchmarkPartialFailure:
    """Some requests fail."""

    def test_failure_counts_and_latency(self, monkeypatch):
        responses = [
            (4.0, {"status": "ok", "latency_ms": {"total": 1.0}}),
            (6.0, None),
            (5.0, {"status": "ok", "latency_ms": {"total": 2.0}}),
        ]

        def fake_send(endpoint, payload, timeout_sec):
            return responses.pop(0)

        monkeypatch.setattr(
            "src.benchmark.http_benchmark.send_infer_request",
            fake_send,
        )

        config = {
            "endpoint": "http://127.0.0.1:8001/infer",
            "request": {"input_type": "dummy", "input": "dummy", "top_k": 5},
            "benchmark": {"warmup": 0, "repeat": 3, "timeout_sec": 10},
        }

        report = run_http_benchmark(config)

        assert report["success_count"] == 2
        assert report["failure_count"] == 1
        assert report["success_rate"] == 0.6667
        assert report["client_total_ms"]["mean"] == 5.0
        assert report["server_total_ms"]["mean"] == 1.5


class TestHttpBenchmarkOutput:
    """Markdown report writing."""

    def test_markdown_contains_expected_fields(self, tmp_path):
        report = {
            "endpoint": "http://127.0.0.1:8001/infer",
            "input_type": "dummy",
            "backend": "onnx",
            "model_id": "test",
            "model_variant": "v1",
            "top_k": 5,
            "repeat": 50,
            "success_rate": 1.0,
            "client_total_ms": {
                "mean": 4.12, "p50": 3.95, "p95": 5.80, "min": 3.60, "max": 7.20,
            },
            "server_total_ms": {
                "mean": 3.40, "p50": 3.30, "p95": 4.20, "min": 3.00, "max": 5.00,
            },
        }
        p = tmp_path / "report.md"
        _write_http_markdown(report, p)
        content = p.read_text(encoding="utf-8")
        assert "HTTP Inference Benchmark Report" in content
        assert "Latency" in content


class TestConfigLoad:
    """Config file loading."""

    def test_dummy_config_can_be_loaded(self):
        p = Path("configs/http_benchmark_fp32_dummy.yaml")
        assert p.exists()
        with open(p, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert cfg["endpoint"] == "http://127.0.0.1:8001/infer"
        assert cfg["request"]["input_type"] == "dummy"

    def test_image_config_can_be_loaded(self):
        p = Path("configs/http_benchmark_fp32_image.yaml")
        assert p.exists()
        with open(p, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert cfg["request"]["input_type"] == "image_path"
