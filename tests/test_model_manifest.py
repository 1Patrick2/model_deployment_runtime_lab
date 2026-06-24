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
