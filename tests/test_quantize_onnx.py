"""Tests for ONNX quantization."""

from pathlib import Path

import onnx
import pytest

from src.quantization.quantize_onnx import run_quantization


class TestQuantizeOnnx:
    """Quantization error handling (no real ONNX needed)."""

    def test_missing_input_raises_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Input ONNX model not found"):
            run_quantization(
                input_path=tmp_path / "nonexistent.onnx",
                output_path=tmp_path / "out.onnx",
            )

    @pytest.mark.skipif(
        not Path("outputs/onnx/mobilenetv3_small.onnx").exists(),
        reason="ONNX artifact not generated; run export_onnx.py first",
    )
    def test_quantize_and_check(self, tmp_path):
        out = tmp_path / "quantized.onnx"
        run_quantization(
            input_path="outputs/onnx/mobilenetv3_small.onnx",
            output_path=str(out),
        )
        assert out.exists()
        model = onnx.load(str(out))
        onnx.checker.check_model(model)
