"""Tests for RKNN conversion config and error handling."""

from pathlib import Path

import pytest
import yaml

from src.rknn.convert import convert_onnx_to_rknn


class TestRknnConvert:
    """Conversion error handling — no real RKNN Toolkit needed."""

    def test_missing_input_model_returns_error(self):
        config = {
            "input_model": "outputs/onnx/nonexistent.onnx",
            "output_model": "outputs/rknn/test.rknn",
        }
        report = convert_onnx_to_rknn(config)
        assert report["status"] == "error"
        assert "not found" in report["message"]

    def test_missing_rknn_toolkit_returns_clear_error(self):
        config = {
            "input_model": "models/manifests/mobilenetv3_small_onnx_fp32.json",
            "output_model": "outputs/rknn/test.rknn",
        }
        # Use a fake file path that exists
        report = convert_onnx_to_rknn(config)
        assert report["status"] == "error"
        # Should mention RKNN Toolkit, not traceback
        assert "RKNN Toolkit" in report["message"]

    def test_config_can_be_loaded(self):
        config_path = Path("configs/rknn_convert.yaml")
        assert config_path.exists()
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        assert config["input_model"] == "outputs/onnx/mobilenetv3_small.onnx"
        assert config["target_platform"] == "rk3588"
