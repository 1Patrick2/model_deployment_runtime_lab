"""Tests for the ONNX Runtime runner."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.models.manifest import ModelManifest, load_manifest, resolve_artifact_path
from src.models.registry import load_registry, resolve_manifest_path
from src.runtime.onnx_runner import OnnxRunner
from src.server.protocol import InferenceRequest


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def dummy_manifest_path(tmp_path: Path) -> Path:
    """Create a minimal valid manifest with backend=onnx."""
    data = {
        "model_id": "test_model",
        "model_variant": "v1",
        "backend": "onnx",
        "task": "classification",
        "artifact_path": "outputs/onnx/nonexistent.onnx",
        "input_name": "input",
        "output_name": "logits",
        "input_shape": [1, 3, 224, 224],
        "input_dtype": "float32",
    }
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


@pytest.fixture
def non_onnx_manifest_path(tmp_path: Path) -> Path:
    """Create a manifest with a non-onnx backend."""
    data = {
        "model_id": "test_model",
        "model_variant": "v1",
        "backend": "rknn",
        "task": "classification",
        "artifact_path": "outputs/rknn/test.rknn",
        "input_name": "input",
        "output_name": "logits",
        "input_shape": [1, 3, 224, 224],
    }
    p = tmp_path / "rknn_manifest.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


# ── Tests that do NOT require a real ONNX artifact ────────────────


class TestOnnxRunnerInit:
    """Construction and load() — no real ONNX artifact needed."""

    def test_runner_rejects_missing_artifact(self, dummy_manifest_path):
        runner = OnnxRunner(
            manifest_path=dummy_manifest_path,
            project_root=Path.cwd(),
        )
        with pytest.raises(FileNotFoundError, match="ONNX artifact not found"):
            runner.load()

    def test_runner_rejects_non_onnx_manifest(self, non_onnx_manifest_path):
        runner = OnnxRunner(
            manifest_path=non_onnx_manifest_path,
            project_root=Path.cwd(),
        )
        with pytest.raises(ValueError, match="expected 'onnx'"):
            runner.load()


# ── Optional smoke test (requires real ONNX artifact) ─────────────


@pytest.mark.skipif(
    not Path("outputs/onnx/mobilenetv3_small.onnx").exists(),
    reason="ONNX artifact not generated; run export_onnx.py first",
)
class TestOnnxRunnerRealSmoke:
    """These tests only run when the ONNX artifact is present locally."""

    @pytest.fixture(autouse=True)
    def _setup_runner(self):
        self.runner = OnnxRunner(
            manifest_path="models/manifests/mobilenetv3_small_onnx_fp32.json",
        )
        self.runner.load()
        yield
        self.runner.close()

    def test_dummy_input_returns_ok(self):
        req = InferenceRequest(
            backend="onnx",
            input_type="dummy",
            input="dummy",
        )
        resp = self.runner.predict(req)
        assert resp.status == "ok"
        assert resp.backend == "onnx"

    def test_prediction_has_valid_fields(self):
        req = InferenceRequest(
            backend="onnx",
            input_type="dummy",
            input="dummy",
        )
        resp = self.runner.predict(req)
        assert resp.prediction is not None
        assert len(resp.prediction.class_name) > 0
        assert resp.prediction.class_id >= 0
        assert 0.0 <= resp.prediction.confidence <= 1.0

    def test_latency_ms_contains_all_fields(self):
        req = InferenceRequest(
            backend="onnx",
            input_type="dummy",
            input="dummy",
        )
        resp = self.runner.predict(req)
        assert resp.latency_ms is not None
        assert resp.latency_ms.preprocess >= 0
        assert resp.latency_ms.inference >= 0
        assert resp.latency_ms.postprocess >= 0
        assert resp.latency_ms.total > 0
