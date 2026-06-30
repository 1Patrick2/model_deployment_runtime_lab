"""Tests for quantisation experiment registry."""

from pathlib import Path

import yaml

from src.experiments.run_quant_experiments import _decide


class TestExperimentConfig:
    """Experiment matrix config parsing."""

    def test_matrix_config_exists(self):
        p = Path("configs/experiments/quantization_controls.yaml")
        assert p.exists()
        with open(p, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert "models" in cfg
        assert "experiments" in cfg
        assert len(cfg["experiments"]) >= 3

    def test_experiments_have_required_fields(self):
        p = Path("configs/experiments/quantization_controls.yaml")
        with open(p, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        for exp in cfg["experiments"]:
            assert "id" in exp
            assert "model" in exp
            assert "method" in exp


class TestDecisionRules:
    """Decision rule logic — no ONNX needed."""

    def test_reject_on_failed_status(self):
        d = _decide("test", {"status": "failed"})
        assert d == "invalid_artifact"

    def test_reject_low_top1(self):
        d = _decide("test", {
            "status": "ok", "top1_consistency": 0.5,
            "top5_consistency": 1.0, "mean_logits_cosine_similarity": 0.99,
        })
        assert d == "reject_consistency_failed"

    def test_reject_low_cosine(self):
        d = _decide("test", {
            "status": "ok", "top1_consistency": 0.9,
            "top5_consistency": 1.0, "mean_logits_cosine_similarity": 0.5,
        })
        assert d == "reject_distribution_shift"

    def test_size_baseline_when_slower(self):
        d = _decide("test", {
            "status": "ok", "top1_consistency": 0.97,
            "top5_consistency": 1.0, "mean_logits_cosine_similarity": 0.999,
            "fp32_mean_latency_ms": 3.8, "int8_mean_latency_ms": 4.5,
        })
        assert d == "size_optimized_baseline"

    def test_recommended_when_faster(self):
        d = _decide("test", {
            "status": "ok", "top1_consistency": 0.97,
            "top5_consistency": 1.0, "mean_logits_cosine_similarity": 0.999,
            "fp32_mean_latency_ms": 4.0, "int8_mean_latency_ms": 3.0,
        })
        assert d == "recommended_candidate"
