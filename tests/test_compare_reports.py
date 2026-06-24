"""Tests for benchmark comparison reports."""

import json
from pathlib import Path

import pytest

from src.quantization.compare_reports import (
    compare_reports,
    write_comparison_json,
    write_comparison_markdown,
)


@pytest.fixture
def fp32_report():
    return {
        "model": {
            "model_id": "mobilenetv3_small",
            "model_variant": "onnx_fp32_v1",
            "backend": "onnx",
            "artifact_size_mb": 10.0,
        },
        "latency_ms": {
            "inference": {"mean": 2.0, "p50": 1.9, "p95": 2.5},
            "total": {"mean": 2.5, "p50": 2.4, "p95": 3.0},
        },
    }


@pytest.fixture
def int8_report():
    return {
        "model": {
            "model_id": "mobilenetv3_small",
            "model_variant": "onnx_int8_dynamic_v1",
            "backend": "onnx",
            "artifact_size_mb": 4.0,
        },
        "latency_ms": {
            "inference": {"mean": 1.5, "p50": 1.4, "p95": 2.0},
            "total": {"mean": 2.0, "p50": 1.9, "p95": 2.5},
        },
    }


class TestCompareReports:
    """Comparison computation."""

    def test_size_reduction(self, fp32_report, int8_report):
        result = compare_reports(fp32_report, int8_report)
        cmp = result["comparison"]
        # 10 → 4 MB = 60% reduction
        assert cmp["size_reduction_percent"] == 60.0

    def test_total_speedup_ratio(self, fp32_report, int8_report):
        result = compare_reports(fp32_report, int8_report)
        cmp = result["comparison"]
        # 2.5 / 2.0 = 1.25x
        assert cmp["total_speedup_ratio"] == 1.25

    def test_inference_speedup_ratio(self, fp32_report, int8_report):
        result = compare_reports(fp32_report, int8_report)
        cmp = result["comparison"]
        # 2.0 / 1.5 = 1.3333
        assert cmp["inference_speedup_ratio"] == 1.3333


class TestCompareReportsOutput:
    """Output format tests."""

    def test_json_output(self, fp32_report, int8_report, tmp_path):
        result = compare_reports(fp32_report, int8_report)
        p = tmp_path / "compare.json"
        write_comparison_json(result, p)
        data = json.loads(p.read_text(encoding="utf-8"))
        assert data["comparison"]["size_reduction_percent"] == 60.0

    def test_markdown_contains_variant_names(self, fp32_report, int8_report, tmp_path):
        result = compare_reports(fp32_report, int8_report)
        p = tmp_path / "compare.md"
        write_comparison_markdown(result, p)
        content = p.read_text(encoding="utf-8")
        assert "onnx_fp32_v1" in content
        assert "onnx_int8_dynamic_v1" in content
        assert "Size" in content
        assert "Latency" in content or "latency" in content.lower()
