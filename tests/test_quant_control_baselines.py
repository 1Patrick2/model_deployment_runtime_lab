"""Tests for quantization control baseline configs."""

from pathlib import Path

import yaml


class TestDynamicBaselineConfig:
    """Dynamic weight-only config should parse correctly."""

    def test_dynamic_config_loads(self):
        p = Path("configs/quant_dynamic_weight_only_mobilenetv3.yaml")
        assert p.exists()
        with open(p, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert cfg["method"] == "dynamic"
        assert "input_model" in cfg
        assert "output_model" in cfg

    def test_dynamic_validation_config_loads(self):
        p = Path("configs/validate_fp32_vs_dynamic_mobilenetv3.yaml")
        assert p.exists()
        with open(p, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert "fp32_model" in cfg
        assert "int8_model" in cfg
        assert "dynamic" in cfg["int8_model"]


class TestResNet18Configs:
    """ResNet18 control configs should parse in mdrl-runtime."""

    def test_export_config_loads_without_torch(self):
        p = Path("configs/export_resnet18_imagenet.yaml")
        assert p.exists()
        with open(p, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert cfg["model"] == "resnet18"
        assert cfg["pretrained"] is True

    def test_manifest_loads(self):
        p = Path("models/manifests/resnet18_onnx_fp32_imagenet.json")
        import json
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["model_id"] == "resnet18"
        assert data["backend"] == "onnx"

    def test_quant_configs_exist(self):
        for name in [
            "configs/quant_static_qdq_real_resnet18.yaml",
            "configs/quant_static_qdq_real_preprocessed_resnet18.yaml",
            "configs/validate_fp32_vs_qdq_real_resnet18.yaml",
            "configs/validate_fp32_vs_qdq_preprocessed_resnet18.yaml",
        ]:
            assert Path(name).exists(), f"Missing: {name}"
