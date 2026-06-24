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
