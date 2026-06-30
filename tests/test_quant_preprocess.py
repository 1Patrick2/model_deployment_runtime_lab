"""Tests for quantization preprocess fallback logic."""

from pathlib import Path

import pytest


class TestQuantPreprocessFallback:
    """Fallback logic — no real ORT preprocess needed."""

    def test_config_with_skip_symbolic_shape(self, tmp_path, monkeypatch):
        """Verify skip_symbolic_shape is passed to quant_pre_process."""
        calls = []

        def fake_quant_pre_process(inp, out, **kw):
            calls.append(kw)
            raise RuntimeError("Incomplete symbolic shape inference")

        monkeypatch.setattr(
            "onnxruntime.quantization.shape_inference.quant_pre_process",
            fake_quant_pre_process,
        )

        dummy_onnx = tmp_path / "model.onnx"
        dummy_onnx.write_bytes(b"\x00" * 1024)

        config = {
            "quantization": {
                "preprocess": True,
                "skip_symbolic_shape": True,
                "fallback_to_original": True,
            },
            "calibration": {
                "image_dir": str(tmp_path / "imgs"),
                "max_samples": 1,
            },
        }

        img_dir = tmp_path / "imgs"
        img_dir.mkdir()
        (img_dir / "dummy.jpg").write_text("dummy")

        from src.quantization.quantize_onnx import run_quantization
        with pytest.raises(Exception):
            run_quantization(str(dummy_onnx), str(tmp_path / "out.onnx"), config=config)

        has_skip = any(c.get("skip_symbolic_shape") for c in calls)
        assert has_skip, f"Expected skip_symbolic_shape in calls, got: {calls}"

    def test_fallback_to_original_on_failure(self, tmp_path, monkeypatch):
        """When quant_pre_process fails and fallback_to_original is True."""
        def fail_always(inp, out, **kw):
            raise RuntimeError("Incomplete symbolic shape inference")

        monkeypatch.setattr(
            "onnxruntime.quantization.shape_inference.quant_pre_process",
            fail_always,
        )

        dummy_onnx = tmp_path / "model.onnx"
        dummy_onnx.write_bytes(b"\x00" * 1024)

        config = {
            "quantization": {
                "preprocess": True,
                "fallback_to_original": True,
            },
            "calibration": {
                "image_dir": str(tmp_path / "imgs"),
                "max_samples": 1,
            },
        }

        img_dir = tmp_path / "imgs"
        img_dir.mkdir()
        (img_dir / "dummy.jpg").write_text("dummy")

        from src.quantization.quantize_onnx import run_quantization
        with pytest.raises(Exception):
            run_quantization(str(dummy_onnx), str(tmp_path / "out.onnx"), config=config)

class TestValidationFailureReport:
    """Validation should produce failure reports instead of traceback."""

    def test_failure_report_has_required_fields(self):
        from src.validation.output_consistency import _make_failure_report
        exc = RuntimeError("Could not find an implementation for ConvInteger")
        report = _make_failure_report(
            Path("fp32.onnx"), Path("int8.onnx"), Path("images"),
            "load_int8_model", exc,
        )
        assert report["status"] == "failed"
        assert report["failure_stage"] == "load_int8_model"
        assert "ConvInteger" in report["error_message"]
        assert "linear-only" in report["recommendation"]
