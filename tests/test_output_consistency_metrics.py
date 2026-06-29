"""Tests for output consistency metrics."""

import numpy as np
import pytest

from src.validation.metrics import (
    top1_consistency,
    top5_consistency,
    mean_top5_overlap,
    mean_confidence_drift,
    max_confidence_drift,
    mean_logits_cosine_similarity,
    mean_absolute_error,
    max_absolute_error,
    top_k_indices,
    softmax,
)
from src.validation.output_consistency import (
    write_validation_json,
    write_validation_markdown,
)


class TestSoftmax:
    def test_sums_to_one(self):
        probs = softmax(np.array([2.0, 1.0, 0.1]))
        assert abs(probs.sum() - 1.0) < 1e-5


class TestTopKIndices:
    def test_top1_returns_highest(self):
        logits = np.array([0.1, 0.5, 0.01, 10.0, 2.0])
        indices = top_k_indices(logits, 1)
        assert indices[0] == 3  # index 3 has value 10.0

    def test_top5_returns_five(self):
        logits = np.random.randn(1000).astype(np.float32)
        indices = top_k_indices(logits, 5)
        assert len(indices) == 5


class TestConsistencyMetrics:
    @pytest.fixture
    def identical_logits(self):
        fp32 = [np.array([0.1, 0.5, 10.0, 0.01]) for _ in range(3)]
        int8 = [np.array([0.1, 0.5, 10.0, 0.01]) for _ in range(3)]
        return fp32, int8

    @pytest.fixture
    def divergent_logits(self):
        fp32 = [np.array([10.0, 0.1, 0.01, 0.5]), np.array([0.01, 10.0, 0.1, 0.5])]
        int8 = [np.array([0.01, 10.0, 0.1, 0.5]), np.array([0.5, 0.01, 10.0, 0.1])]
        return fp32, int8

    def test_identical_has_perfect_consistency(self, identical_logits):
        fp32, int8 = identical_logits
        assert top1_consistency(fp32, int8) == 1.0
        assert top5_consistency(fp32, int8) == 1.0

    def test_cosine_similarity_of_identical(self, identical_logits):
        fp32, int8 = identical_logits
        sim = mean_logits_cosine_similarity(fp32, int8)
        assert abs(sim - 1.0) < 1e-4

    def test_identical_has_zero_drift(self, identical_logits):
        fp32, int8 = identical_logits
        assert mean_confidence_drift(fp32, int8) == 0.0
        assert max_confidence_drift(fp32, int8) == 0.0

    def test_divergent_has_lower_consistency(self, divergent_logits):
        fp32, int8 = divergent_logits
        assert top1_consistency(fp32, int8) < 1.0

    def test_metrics_with_zero_images(self):
        assert top1_consistency([], []) == 0.0


class TestValidationOutput:
    def test_json_written(self, tmp_path):
        report = {"num_images": 10, "top1_consistency": 0.9}
        p = write_validation_json(report, tmp_path / "report.json")
        assert p.exists()

    def test_markdown_written(self, tmp_path):
        report = {
            "num_images": 10,
            "top1_consistency": 0.9,
            "top5_consistency": 0.95,
            "mean_logits_cosine_similarity": 0.99,
            "mean_confidence_drift": 0.01,
            "fp32_mean_latency_ms": 2.0,
            "int8_mean_latency_ms": 1.8,
            "fp32_model_size_mb": 9.92,
            "int8_model_size_mb": 2.70,
            "size_reduction_percent": 72.88,
            "mean_top5_overlap": 0.96,
            "max_confidence_drift": 0.03,
            "mean_absolute_error": 0.05,
            "max_absolute_error": 0.12,
        }
        p = write_validation_markdown(report, tmp_path / "report.md")
        content = p.read_text(encoding="utf-8")
        assert "Quantization Validation Report" in content
        assert "FP32/INT8" in content
        assert "size_reduction_percent" in content

    def test_confidence_drift_threshold_direction_lower_is_better(self, tmp_path):
        """mean_confidence_drift: lower values should be OK, high should warn."""
        report_ok = {
            "num_images": 5,
            "top1_consistency": 0.9,
            "top5_consistency": 0.95,
            "mean_logits_cosine_similarity": 0.99,
            "mean_confidence_drift": 0.01,  # low drift = good
            "fp32_mean_latency_ms": 2.0,
            "int8_mean_latency_ms": 1.8,
            "fp32_model_size_mb": 9.92,
            "int8_model_size_mb": 2.70,
            "size_reduction_percent": 72.88,
        }
        report_bad = dict(report_ok)
        report_bad["mean_confidence_drift"] = 0.5  # high drift = bad

        p_ok = write_validation_markdown(report_ok, tmp_path / "ok.md")
        p_bad = write_validation_markdown(report_bad, tmp_path / "bad.md")

        content_ok = p_ok.read_text(encoding="utf-8")
        content_bad = p_bad.read_text(encoding="utf-8")

        # low drift should have a checkmark, high drift should have a warning
        ok_lines = [l for l in content_ok.split("\n") if "confidence" in l.lower()]
        bad_lines = [l for l in content_bad.split("\n") if "confidence" in l.lower()]

        assert ok_lines, "confidence drift line not found in markdown"
        # At least one line should exist and should differ between ok and bad
        assert any("PASS" in l for l in ok_lines), f"low drift should be PASS, got: {ok_lines}"
        assert any("WARN" in l for l in bad_lines), f"high drift should be WARN, got: {bad_lines}"

    def test_per_image_samples_table_in_markdown(self, tmp_path):
        report = {
            "num_images": 5,
            "top1_consistency": 0.9,
            "top5_consistency": 0.95,
            "mean_logits_cosine_similarity": 0.99,
            "mean_confidence_drift": 0.01,
            "fp32_mean_latency_ms": 2.0,
            "int8_mean_latency_ms": 1.8,
            "fp32_model_size_mb": 9.92,
            "int8_model_size_mb": 2.70,
            "size_reduction_percent": 72.88,
            "per_image_results": [
                {
                    "image_path": "test.jpg",
                    "fp32_top1_index": 281,
                    "int8_top1_index": 0,
                    "top1_match": False,
                    "fp32_top1_in_int8_top5": False,
                    "top5_overlap": 0.0,
                    "logits_cosine_similarity": 0.5,
                }
            ],
        }
        p = write_validation_markdown(report, tmp_path / "per_image.md")
        content = p.read_text(encoding="utf-8")
        assert "Per-image Samples" in content
        assert "test.jpg" in content
        assert "FP32 top1" in content

    def test_json_with_numpy_types_serializes_safely(self, tmp_path):
        import numpy as np
        from src.validation.output_consistency import write_validation_json

        report = {
            "num_images": np.int64(5),
            "top1_consistency": np.float32(0.9),
            "top1_match": np.bool_(True),
            "per_image_results": [
                {
                    "fp32_top1_index": np.int64(281),
                    "top1_match": np.bool_(False),
                    "top5_overlap": np.float64(0.5),
                }
            ],
        }
        p = write_validation_json(report, tmp_path / "numpy_report.json")
        import json
        data = json.loads(p.read_text(encoding="utf-8"))
        assert isinstance(data["num_images"], int)
        assert isinstance(data["top1_consistency"], float)
        assert isinstance(data["top1_match"], bool)
        assert data["per_image_results"][0]["fp32_top1_index"] == 281
