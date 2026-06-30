"""Tests for ONNX export config and CLI."""

import json
from pathlib import Path

import yaml

from src.export.export_onnx import _load_config


class TestExportConfig:
    """Config file loading and field resolution."""

    def test_mobilenetv3_export_config_can_be_loaded(self):
        p = Path("configs/export/mobilenetv3_small_imagenet.yaml")
        assert p.exists(), "Export config file missing"
        config = _load_config(p)
        assert config["model"] == "mobilenet_v3_small"
        assert config["pretrained"] is True
        assert "output" in config
        assert "input_shape" in config
        assert "input_name" in config
        assert "output_name" in config

    def test_resnet18_export_config_can_be_loaded(self):
        p = Path("configs/export/resnet18_imagenet.yaml")
        assert p.exists()
        config = _load_config(p)
        assert config["model"] == "resnet18"
        assert config["pretrained"] is True

    def test_mobilenetv3_export_config_has_correct_output_path(self):
        p = Path("configs/export/mobilenetv3_small_imagenet.yaml")
        config = _load_config(p)
        assert "imagenet" in config["output"], (
            f"Expected imagenet in output path, got: {config['output']}"
        )
