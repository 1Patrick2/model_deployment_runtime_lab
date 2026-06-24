"""Tests for benchmark report generation."""

import json
from pathlib import Path

import pytest

from src.benchmark.report import (
    build_benchmark_report,
    write_json_report,
    write_markdown_report,
)


@pytest.fixture
def sample_records():
    return [
        {"preprocess": 0.1, "inference": 2.0, "postprocess": 0.1, "total": 2.2},
        {"preprocess": 0.2, "inference": 3.0, "postprocess": 0.2, "total": 3.4},
        {"preprocess": 0.3, "inference": 4.0, "postprocess": 0.3, "total": 4.6},
    ]


@pytest.fixture
def sample_report(sample_records):
    return build_benchmark_report(
        model_id="test_model",
        model_variant="v1",
        backend="onnx",
        artifact_path="outputs/onnx/test.onnx",
        artifact_size_mb=5.0,
        input_type="dummy",
        input_value="dummy",
        warmup=5,
        repeat=3,
        latency_records=sample_records,
    )


class TestBuildBenchmarkReport:
    """Report construction."""

    def test_contains_model_section(self, sample_report):
        assert sample_report["model"]["model_id"] == "test_model"
        assert sample_report["model"]["backend"] == "onnx"

    def test_contains_benchmark_settings(self, sample_report):
        assert sample_report["benchmark"]["warmup"] == 5
        assert sample_report["benchmark"]["repeat"] == 3

    def test_contains_latency_stats(self, sample_report):
        lat = sample_report["latency_ms"]
        assert "total" in lat
        assert lat["total"]["mean"] > 0


class TestWriteReports:
    """Report output to disk."""

    def test_json_report_can_be_written(self, sample_report, tmp_path):
        p = tmp_path / "report.json"
        write_json_report(sample_report, p)
        assert p.exists()
        data = json.loads(p.read_text(encoding="utf-8"))
        assert data["model"]["model_id"] == "test_model"

    def test_markdown_report_can_be_written(self, sample_report, tmp_path):
        p = tmp_path / "report.md"
        write_markdown_report(sample_report, p)
        assert p.exists()
        content = p.read_text(encoding="utf-8")
        assert "test_model" in content
        assert "onnx" in content
        assert "Latency" in content

    def test_report_directories_are_created(self, sample_report, tmp_path):
        p = tmp_path / "sub" / "nested" / "report.json"
        write_json_report(sample_report, p)
        assert p.exists()
