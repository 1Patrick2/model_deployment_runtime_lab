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
