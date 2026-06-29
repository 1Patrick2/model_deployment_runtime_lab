"""Tests for HTTP benchmark — no real server needed."""

import json
from pathlib import Path

import pytest
import yaml

from src.benchmark.http_benchmark import run_http_benchmark, _write_http_markdown


class TestHttpBenchmarkSummary:
    """Benchmark logic with a fake config that skips warmup/repeat."""

    def _make_fake_config(self, endpoint: str = "http://127.0.0.1:9999/infer") -> dict:
        return {
            "endpoint": endpoint,
            "request": {"input_type": "dummy", "input": "dummy", "top_k": 5},
            "benchmark": {"warmup": 0, "repeat": 0, "timeout_sec": 2},
        }

    def test_benchmark_runs_with_zero_repeat(self):
        """No real HTTP calls made when repeat=0."""
        config = self._make_fake_config()
        # This should fail because the endpoint is unreachable, but repeat=0
        # means no actual benchmark requests are sent. The health check is
        # not part of run_http_benchmark — it runs in main() only.
        report = run_http_benchmark(config)
        assert report["success_count"] == 0
        assert report["failure_count"] == 0
        assert report["repeat"] == 0


class TestHttpBenchmarkOutput:
    """Markdown report writing."""

    def test_markdown_contains_expected_fields(self, tmp_path):
        report = {
            "benchmark_type": "http_inference",
            "endpoint": "http://127.0.0.1:8001/infer",
            "input_type": "dummy",
            "backend": "onnx",
            "model_id": "test",
            "model_variant": "v1",
            "repeat": 50,
            "success_rate": 1.0,
            "client_total_ms": {
                "mean": 4.12, "p50": 3.95, "p95": 5.80, "min": 3.60, "max": 7.20,
            },
            "server_total_ms": {
                "mean": 3.40, "p50": 3.30, "p95": 4.20, "min": 3.00, "max": 5.00,
            },
            "server_inference_ms": {
                "mean": 3.10, "p50": 3.05, "p95": 3.80, "min": 2.80, "max": 4.50,
            },
        }
        p = tmp_path / "report.md"
        _write_http_markdown(report, p)
        content = p.read_text(encoding="utf-8")
        assert "HTTP Inference Benchmark Report" in content
        assert "endpoint" in content
        assert "server_total_ms" in content
        assert "Latency" in content


class TestConfigLoad:
    """Config file loading."""

    def test_dummy_config_can_be_loaded(self):
        p = Path("configs/http_benchmark_fp32_dummy.yaml")
        assert p.exists()
        with open(p, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert cfg["endpoint"] == "http://127.0.0.1:8001/infer"
        assert cfg["request"]["input_type"] == "dummy"

    def test_image_config_can_be_loaded(self):
        p = Path("configs/http_benchmark_fp32_image.yaml")
        assert p.exists()
        with open(p, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert cfg["request"]["input_type"] == "image_path"
