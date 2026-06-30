"""Consolidated tests -- merged from test_protocol.py, test_fake_runner.py, test_onnx_runner.py, test_model_manifest.py, test_artifact_size.py, test_latency_stats.py."""

# -- From test_protocol.py --
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
# -- From test_fake_runner.py --
"""Tests for the fake runtime backend."""

import pytest

from src.runtime.fake_runner import FakeRunner
from src.server.protocol import InferenceRequest


@pytest.fixture
def runner():
    r = FakeRunner()
    r.load()
    yield r
    r.close()


class TestFakeRunnerInputHeuristics:
    """Input-based prediction heuristics."""

    def test_danger_input_returns_danger(self, runner):
        req = InferenceRequest(backend="fake", input="samples/images/danger_scene.jpg")
        resp = runner.predict(req)
        assert resp.prediction is not None
        assert resp.prediction.class_name == "danger"
        assert resp.prediction.class_id == 2

    def test_safe_input_returns_safe(self, runner):
        req = InferenceRequest(backend="fake", input="samples/images/safe_scene.jpg")
        resp = runner.predict(req)
        assert resp.prediction is not None
        assert resp.prediction.class_name == "safe"
        assert resp.prediction.class_id == 0

    def test_ordinary_input_returns_warning(self, runner):
        req = InferenceRequest(backend="fake", input="samples/images/test.jpg")
        resp = runner.predict(req)
        assert resp.prediction is not None
        assert resp.prediction.class_name == "warning"
        assert resp.prediction.class_id == 1


class TestFakeRunnerResponseShape:
    """Response structure and required fields."""

    def test_status_is_ok(self, runner):
        req = InferenceRequest(backend="fake", input="test.jpg")
        resp = runner.predict(req)
        assert resp.status == "ok"

    def test_backend_is_fake(self, runner):
        req = InferenceRequest(backend="fake", input="test.jpg")
        resp = runner.predict(req)
        assert resp.backend == "fake"

    def test_latency_ms_total_exists(self, runner):
        req = InferenceRequest(backend="fake", input="test.jpg")
        resp = runner.predict(req)
        assert resp.latency_ms is not None
        assert resp.latency_ms.total > 0

    def test_confidence_is_reasonable(self, runner):
        req = InferenceRequest(backend="fake", input="danger.jpg")
        resp = runner.predict(req)
        assert resp.prediction is not None
        assert 0.0 <= resp.prediction.confidence <= 1.0
# -- From test_onnx_runner.py --
"""Tests for the ONNX Runtime runner."""

import json
from pathlib import Path

import pytest

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

    def test_predict_before_load_raises_error(self, dummy_manifest_path):
        runner = OnnxRunner(
            manifest_path=dummy_manifest_path,
            project_root=Path.cwd(),
        )
        req = InferenceRequest(backend="onnx", input_type="dummy", input="dummy")
        with pytest.raises(RuntimeError, match="not loaded"):
            runner.predict(req)


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

    def test_runner_rejects_backend_mismatch(self):
        req = InferenceRequest(
            backend="fake",
            input_type="dummy",
            input="dummy",
        )
        with pytest.raises(ValueError, match="does not match"):
            self.runner.predict(req)
# -- From test_model_manifest.py --
"""Tests for model manifest and registry."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.models.manifest import ModelManifest, load_manifest, resolve_artifact_path
from src.models.registry import load_registry, resolve_manifest_path


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def manifest_path(tmp_path: Path) -> Path:
    """Write a valid manifest to a temporary directory."""
    data = {
        "model_id": "mobilenetv3_small",
        "model_variant": "onnx_fp32_v1",
        "backend": "onnx",
        "task": "classification",
        "artifact_path": "outputs/onnx/mobilenetv3_small.onnx",
        "input_name": "input",
        "output_name": "logits",
        "input_shape": [1, 3, 224, 224],
        "input_dtype": "float32",
        "class_names_path": "models/labels/imagenet_classes.txt",
        "preprocess": {
            "resize": [224, 224],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    }
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


@pytest.fixture
def registry_path(tmp_path: Path) -> Path:
    """Write a valid registry to a temporary directory."""
    data = [
        {
            "model_id": "mobilenetv3_small",
            "backend": "onnx",
            "manifest_path": "models/manifests/mobilenetv3_small_onnx_fp32.json",
        }
    ]
    p = tmp_path / "registry.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


# ── Manifest tests ────────────────────────────────────────────────


class TestModelManifest:
    def test_manifest_can_be_loaded(self, manifest_path):
        manifest = load_manifest(manifest_path)
        assert manifest.model_id == "mobilenetv3_small"
        assert manifest.backend == "onnx"
        assert manifest.input_shape == [1, 3, 224, 224]

    def test_artifact_path_can_be_resolved(self, manifest_path):
        manifest = load_manifest(manifest_path)
        project_root = Path("/fake/project")
        resolved = resolve_artifact_path(project_root, manifest)
        assert resolved.name == "mobilenetv3_small.onnx"
        assert "outputs" in resolved.parts
        assert "onnx" in resolved.parts

    def test_missing_required_field_raises_error(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps({"model_id": "incomplete"}), encoding="utf-8")
        with pytest.raises(ValidationError):
            load_manifest(p)

    def test_preprocess_loaded_correctly(self, manifest_path):
        manifest = load_manifest(manifest_path)
        assert manifest.preprocess is not None
        assert manifest.preprocess.resize == (224, 224)
        assert manifest.preprocess.mean == (0.485, 0.456, 0.406)

    def test_input_dtype_default_is_float32(self, manifest_path):
        manifest = load_manifest(manifest_path)
        assert manifest.input_dtype == "float32"


# ── Registry tests ────────────────────────────────────────────────


class TestModelRegistry:
    def test_registry_can_be_loaded(self, registry_path):
        registry = load_registry(registry_path)
        entry = registry.find_manifest_by_model_id("mobilenetv3_small")
        assert entry is not None
        assert entry["backend"] == "onnx"

    def test_find_nonexistent_model_returns_none(self, registry_path):
        registry = load_registry(registry_path)
        entry = registry.find_manifest_by_model_id("nonexistent")
        assert entry is None

    def test_find_with_backend_filter(self, registry_path):
        registry = load_registry(registry_path)
        entry = registry.find_manifest_by_model_id(
            "mobilenetv3_small", backend="onnx"
        )
        assert entry is not None

    def test_find_with_wrong_backend_returns_none(self, registry_path):
        registry = load_registry(registry_path)
        entry = registry.find_manifest_by_model_id(
            "mobilenetv3_small", backend="rknn"
        )
        assert entry is None

    def test_list_models_returns_all(self, registry_path):
        registry = load_registry(registry_path)
        models = registry.list_models()
        assert len(models) == 1


# ── Real repository file tests ────────────────────────────────────


class TestRealManifestFiles:
    """Smoke tests that load the actual manifest and registry files
    checked into the repository, not temporary fixtures."""

    def test_real_manifest_file_can_be_loaded(self):
        manifest = load_manifest(
            "models/manifests/mobilenetv3_small_onnx_fp32.json"
        )
        assert manifest.model_id == "mobilenetv3_small"
        assert manifest.backend == "onnx"
        assert manifest.input_shape == [1, 3, 224, 224]

    def test_real_registry_can_find_manifest(self):
        registry = load_registry("models/registry.json")
        entry = registry.find_manifest_by_model_id(
            "mobilenetv3_small", backend="onnx"
        )
        assert entry is not None
        assert entry["manifest_path"] == (
            "models/manifests/mobilenetv3_small_onnx_fp32.json"
        )

    def test_real_registry_manifest_path_resolves_to_existing_file(self):
        registry = load_registry("models/registry.json")
        entry = registry.find_manifest_by_model_id("mobilenetv3_small")
        assert entry is not None
        resolved = resolve_manifest_path(Path.cwd(), entry)
        assert resolved.exists(), f"Manifest file not found: {resolved}"
# -- From test_artifact_size.py --
"""Tests for ONNX artifact size computation."""

from pathlib import Path

import pytest

from src.benchmark.artifact import compute_onnx_artifact_size_mb


class TestComputeOnnxArtifactSize:
    """Artifact size calculation."""

    def test_missing_artifact_raises_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            compute_onnx_artifact_size_mb(tmp_path / "missing.onnx")

    @pytest.mark.skipif(
        not Path("outputs/onnx/mobilenetv3_small.onnx").exists(),
        reason="ONNX artifact not generated; run export_onnx.py first",
    )
    def test_external_data_size_is_counted(self):
        size_mb = compute_onnx_artifact_size_mb(
            "outputs/onnx/mobilenetv3_small.onnx"
        )
        # Should be significantly larger than 0.26 MB (main model file)
        # because the external data file (~9.6 MB) is included.
        assert size_mb > 9.0, (
            f"Expected > 9.0 MB (including external data), got {size_mb:.2f} MB"
        )
# -- From test_latency_stats.py --
"""Tests for benchmark statistics computation."""

from src.benchmark.report import compute_stats


class TestComputeStats:
    """Statistic calculations — no ONNX artifact needed."""

    def test_known_values(self):
        stats = compute_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        assert stats["mean"] == 3.0
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["p50"] == 3.0

    def test_p95_is_reasonable(self):
        values = list(range(1, 101))  # 1..100
        stats = compute_stats([float(v) for v in values])
        assert 90 <= stats["p95"] <= 100

    def test_empty_list_returns_zeros(self):
        stats = compute_stats([])
        assert stats["mean"] == 0.0
        assert stats["min"] == 0.0
        assert stats["max"] == 0.0

    def test_single_value(self):
        stats = compute_stats([42.0])
        assert stats["mean"] == 42.0
        assert stats["min"] == 42.0
        assert stats["max"] == 42.0
