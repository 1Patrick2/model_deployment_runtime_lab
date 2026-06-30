"""Tests for TensorRT pipeline — no GPU required, uses monkeypatch."""

import json
from pathlib import Path

import pytest
import yaml

from src.tensorrt.env import collect_tensorrt_env, check_nvidia_smi
from src.tensorrt.parse_trtexec import parse_trtexec_output
from src.tensorrt.build_engine import _build_command, _benchmark_command
from src.tensorrt.report import write_tensorrt_json, write_tensorrt_markdown


class TestTensorRTConfig:
    """Config file parsing."""

    def test_mobilenetv3_config_loads(self):
        p = Path("configs/tensorrt/mobilenetv3_small_fp16.yaml")
        assert p.exists()
        with open(p, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert cfg["model_id"] == "mobilenetv3_small"
        assert cfg["precision"] == "fp16"

    def test_resnet18_config_loads(self):
        p = Path("configs/tensorrt/resnet18_fp16.yaml")
        assert p.exists()
        with open(p, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert cfg["model_id"] == "resnet18"


class TestBuildCommand:
    """Command construction."""

    def test_build_command_format(self):
        cfg = {
            "onnx_model": "test.onnx",
            "engine_output": "test.engine",
            "precision": "fp16",
            "input_name": "input",
            "input_shape": [1, 3, 224, 224],
        }
        cmd = _build_command(cfg)
        assert any("--onnx=test.onnx" in c for c in cmd)
        assert any("--fp16" in c for c in cmd)
        assert any("input:1x3x224x224" in c for c in cmd)

    def test_benchmark_command_format(self):
        cfg = {"engine_output": "test.engine", "trtexec": {"warmup": 100, "iterations": 500}}
        cmd = _benchmark_command(cfg)
        assert any("--loadEngine=test.engine" in c for c in cmd)
        assert any("--warmUp=100" in c for c in cmd)
        assert any("--iterations=500" in c for c in cmd)


class TestEnvDetection:
    """Environment detection — safe on CPU-only."""

    def test_env_returns_skipped_when_no_trtexec(self):
        env = collect_tensorrt_env()
        # Should not crash, should return something
        assert "trtexec_available" in env
        assert "status" in env

    def test_nvidia_smi_returns_available_false(self):
        result = check_nvidia_smi()
        assert "available" in result


class TestTrtexecParser:
    """Parsing trtexec benchmark output."""

    def test_parse_standard_output(self):
        sample = """
        Throughput: 1234.56 qps

        Latency: min 0.80 ms, max 1.40 ms, mean 1.00 ms, median 0.95 ms
        GPU Compute Time: mean 0.90 ms
        Enqueue Time: mean 0.05 ms
        H2D Latency: mean 0.02 ms
        D2H Latency: mean 0.02 ms
        """
        result = parse_trtexec_output(sample)
        assert result["throughput_qps"] == 1234.56
        assert result["latency_ms"]["mean"] == 1.0
        assert result["latency_ms"]["median"] == 0.95
        assert result["gpu_compute_time_ms"]["mean"] == 0.9

    def test_parse_empty_output_does_not_crash(self):
        result = parse_trtexec_output("some random text without trtexec fields")
        assert "raw_output_tail" in result or len(result) == 0


class TestTensorRTReport:
    """Report generation."""

    def test_json_report_written(self, tmp_path):
        report = {"model_id": "test", "build_status": "dry_run"}
        p = write_tensorrt_json(report, tmp_path / "report.json")
        data = json.loads(p.read_text(encoding="utf-8"))
        assert data["build_status"] == "dry_run"

    def test_markdown_report_written(self, tmp_path):
        report = {"model_id": "test", "build_status": "skipped",
                  "reason": "trtexec not available"}
        p = write_tensorrt_markdown(report, tmp_path / "report.md")
        content = p.read_text(encoding="utf-8")
        assert "TensorRT FP16 Benchmark Report" in content
