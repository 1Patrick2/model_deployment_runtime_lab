"""Tests for HTTP benchmark — no real server needed."""

from pathlib import Path

import pytest
import yaml

from src.benchmark.http_benchmark import (
    make_health_endpoint,
    run_http_benchmark,
    send_infer_request,
    _write_http_markdown,
)


class TestMakeHealthEndpoint:
    """Health endpoint derivation."""

    def test_from_infer_path(self):
        assert (
            make_health_endpoint("http://127.0.0.1:8001/infer")
            == "http://127.0.0.1:8001/health"
        )

    def test_from_root_path(self):
        assert (
            make_health_endpoint("http://127.0.0.1:8001")
            == "http://127.0.0.1:8001/health"
        )


class TestRunHttpBenchmarkAllSuccess:
    """All requests succeed."""

    def test_success_counts_and_latency(self, monkeypatch):
        call_count = 0

        def fake_send(endpoint, payload, timeout_sec):
            nonlocal call_count
            call_count += 1
            return 5.0, {
                "status": "ok",
                "backend": "onnx",
                "model_id": "mobilenetv3_small",
                "model_variant": "onnx_fp32_v1",
                "top_k_predictions": [
                    {"class_id": 1, "class_name": "goldfish", "score": 0.9}
                ],
                "latency_ms": {
                    "preprocess": 0.1,
                    "inference": 1.5,
                    "postprocess": 0.2,
                    "total": 1.8,
                },
            }

        monkeypatch.setattr(
            "src.benchmark.http_benchmark.send_infer_request",
            fake_send,
        )

        config = {
            "endpoint": "http://127.0.0.1:8001/infer",
            "request": {"input_type": "dummy", "input": "dummy", "top_k": 5},
            "benchmark": {"warmup": 2, "repeat": 3, "timeout_sec": 10},
        }

        report = run_http_benchmark(config)

        assert call_count == 5  # warmup 2 + repeat 3
        assert report["success_count"] == 3
        assert report["failure_count"] == 0
        assert report["success_rate"] == 1.0
        assert report["backend"] == "onnx"
        assert report["model_id"] == "mobilenetv3_small"
        assert report["server_total_ms"]["mean"] == 1.8
        assert report["client_total_ms"]["mean"] == 5.0
        assert report["sample_predictions"][0]["class_name"] == "goldfish"


class TestRunHttpBenchmarkPartialFailure:
    """Some requests fail."""

    def test_failure_counts_and_latency(self, monkeypatch):
        responses = [
            (4.0, {"status": "ok", "latency_ms": {"total": 1.0}}),
            (6.0, None),
            (5.0, {"status": "ok", "latency_ms": {"total": 2.0}}),
        ]

        def fake_send(endpoint, payload, timeout_sec):
            return responses.pop(0)

        monkeypatch.setattr(
            "src.benchmark.http_benchmark.send_infer_request",
            fake_send,
        )

        config = {
            "endpoint": "http://127.0.0.1:8001/infer",
            "request": {"input_type": "dummy", "input": "dummy", "top_k": 5},
            "benchmark": {"warmup": 0, "repeat": 3, "timeout_sec": 10},
        }

        report = run_http_benchmark(config)

        assert report["success_count"] == 2
        assert report["failure_count"] == 1
        assert report["success_rate"] == 0.6667
        assert report["client_total_ms"]["mean"] == 5.0
        assert report["server_total_ms"]["mean"] == 1.5


class TestHttpBenchmarkOutput:
    """Markdown report writing."""

    def test_markdown_contains_expected_fields(self, tmp_path):
        report = {
            "endpoint": "http://127.0.0.1:8001/infer",
            "input_type": "dummy",
            "backend": "onnx",
            "model_id": "test",
            "model_variant": "v1",
            "top_k": 5,
            "repeat": 50,
            "success_rate": 1.0,
            "client_total_ms": {
                "mean": 4.12, "p50": 3.95, "p95": 5.80, "min": 3.60, "max": 7.20,
            },
            "server_total_ms": {
                "mean": 3.40, "p50": 3.30, "p95": 4.20, "min": 3.00, "max": 5.00,
            },
        }
        p = tmp_path / "report.md"
        _write_http_markdown(report, p)
        content = p.read_text(encoding="utf-8")
        assert "HTTP Inference Benchmark Report" in content
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
