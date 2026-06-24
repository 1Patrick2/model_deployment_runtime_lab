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
