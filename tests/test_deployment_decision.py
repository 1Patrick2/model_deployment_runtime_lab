"""Consolidated tests -- merged from test_deployment_decision.py, test_deployment_advisor.py, test_compare_reports.py, test_benchmark_report.py, test_rknn_report.py, test_rknn_convert.py."""

# -- From test_deployment_decision.py --
"""Tests for deployment decision report."""

import json
from pathlib import Path

import yaml

from src.report.deployment_decision import (
    build_deployment_decision,
    write_decision_json,
    write_decision_markdown,
    load_json,
)


def _fake_report(overrides: dict = None) -> dict:
    """Build a fake full report dict."""
    data = {
        "model": {
            "model_id": "mobilenetv3_small",
            "model_variant": "onnx_fp32_v1",
            "backend": "onnx",
            "artifact_size_mb": 9.92,
        },
        "latency_ms": {
            "inference": {"mean": 1.2},
            "total": {"mean": 1.5},
        },
    }
    if overrides:
        data.update(overrides)
    return data


def _int8_report(overrides: dict = None) -> dict:
    data = {
        "model": {
            "model_id": "mobilenetv3_small",
            "model_variant": "onnx_int8_qdq_dummy_v1",
            "artifact_size_mb": 2.69,
        },
        "latency_ms": {
            "total": {"mean": 1.6},
        },
    }
    if overrides:
        data.update(overrides)
    return data


def _quant_compare() -> dict:
    return {
        "comparison": {
            "size_reduction_percent": 72.88,
            "total_speedup_ratio": 0.88,
        }
    }


def _rknn_report() -> dict:
    return {
        "status": "ok",
        "rknn_size_mb": 5.48,
        "target_platform": "rk3588",
    }


def _http_report() -> dict:
    return {
        "success_rate": 1.0,
        "client_total_ms": {"mean": 5.68},
        "server_total_ms": {"mean": 1.87},
    }


class TestBuildDeploymentDecision:
    """Report building with full evidence."""

    def _make_config(self, tmp_path, fp32, int8, quant, rknn, http) -> dict:
        def _write(name, data):
            p = tmp_path / name
            p.write_text(json.dumps(data if data else {}))
            return str(p)

        return {
            "inputs": {
                "fp32_benchmark": _write("fp32.json", fp32),
                "int8_benchmark": _write("int8.json", int8),
                "quant_compare": _write("quant.json", quant),
                "rknn_convert": _write("rknn.json", rknn),
                "http_benchmark": _write("http.json", http),
            },
        }

    def test_all_reports_present(self, tmp_path):
        config = self._make_config(
            tmp_path, _fake_report(), _int8_report(), _quant_compare(), _rknn_report(), _http_report()
        )
        report = build_deployment_decision(config)
        assert report["summary"]["pc_cpu_serving"] == "onnx_fp32"
        assert report["summary"]["size_sensitive_cpu"] == "qdq_int8"
        assert report["summary"]["rk3588_status"] == "artifact_ready_board_validation_pending"
        assert report["summary"]["http_serving_status"] == "stable_local_loopback"

    def test_recommendations_have_three_targets(self, tmp_path):
        config = self._make_config(
            tmp_path, _fake_report(), _int8_report(), _quant_compare(), _rknn_report(), _http_report()
        )
        report = build_deployment_decision(config)
        assert len(report["recommendations"]) == 3
        targets = {r["target"] for r in report["recommendations"]}
        assert "PC CPU serving" in targets
        assert "Size-sensitive CPU deployment" in targets
        assert "RK3588 deployment" in targets

    def test_risks_contain_dummy_calibration(self, tmp_path):
        config = self._make_config(
            tmp_path, _fake_report(), _int8_report(), _quant_compare(), _rknn_report(), _http_report()
        )
        report = build_deployment_decision(config)
        risks_text = " ".join(report["risks"]).lower()
        assert "dummy calibration" in risks_text

    def test_missing_report_does_not_crash(self, tmp_path):
        config = self._make_config(
            tmp_path, _fake_report(), None, None, None, None
        )
        # Remove the missing files
        import os
        for key in ["int8_benchmark", "quant_compare", "rknn_convert", "http_benchmark"]:
            p = Path(config["inputs"][key])
            if p.exists():
                os.remove(p)
        report = build_deployment_decision(config)
        assert report["summary"]["rk3588_status"] == "not_available"
        assert report["summary"]["http_serving_status"] == "not_measured"

    def test_missing_evidence_listed(self, tmp_path):
        config = self._make_config(
            tmp_path, _fake_report(), None, None, None, None
        )
        import os
        for key in ["int8_benchmark", "quant_compare", "rknn_convert", "http_benchmark"]:
            p = Path(config["inputs"][key])
            if p.exists():
                os.remove(p)
        report = build_deployment_decision(config)
        # Should not crash and should still produce recommendations
        assert len(report["recommendations"]) == 3
        # Missing evidence should be recorded
        assert "INT8 benchmark" in report["missing_evidence"]
        assert "Quant comparison" in report["missing_evidence"]
        assert "RKNN conversion" in report["missing_evidence"]
        assert "HTTP benchmark" in report["missing_evidence"]


class TestDeploymentDecisionOutput:
    """Output format tests."""

    def test_json_output(self, tmp_path):
        report = {"report_type": "deployment_decision", "recommendations": [],
                   "risks": [], "next_validation": [], "missing_evidence": [], "evidence": {},
                   "summary": {}}
        p = tmp_path / "report.json"
        write_decision_json(report, p)
        data = json.loads(p.read_text(encoding="utf-8"))
        assert data["report_type"] == "deployment_decision"

    def test_markdown_contains_sections(self, tmp_path):
        report = {
            "report_type": "deployment_decision",
            "summary": {"pc_cpu_serving": "onnx_fp32", "size_sensitive_cpu": "qdq_int8"},
            "recommendations": [
                {"target": "PC CPU", "recommendation": "onnx_fp32", "reason": "Stable"},
            ],
            "risks": ["Dummy calibration"],
            "next_validation": ["Run calibration"],
            "missing_evidence": [],
            "evidence": {
                "http": {"success_rate": 1.0, "client_total_ms_mean": 5.68, "server_total_ms_mean": 1.87},
                "rknn": {"conversion_status": "ok", "artifact_size_mb": 5.48},
            },
        }
        p = tmp_path / "report.md"
        write_decision_markdown(report, p)
        content = p.read_text(encoding="utf-8")
        assert "Deployment Decision Report" in content
        assert "Recommendation" in content
        assert "Risks" in content
        assert "Next Steps" in content
        assert "Evidence" in content


class TestConfigLoad:
    """Verify config file can be loaded."""

    def test_config_loads(self):
        p = Path("configs/deployment_decision.yaml")
        assert p.exists()
        with open(p, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert "fp32_benchmark" in cfg["inputs"]
        assert "output" in cfg
# -- From test_deployment_advisor.py --
"""Tests for rule-based deployment advisor."""

import json
from pathlib import Path

from src.advisor.deployment_advisor import (
    build_advisor_report,
    load_decision_report,
    write_advisor_markdown,
)


def _make_decision(recommendation: str = "onnx_fp32") -> dict:
    return {
        "summary": {
            "pc_cpu_serving": recommendation,
            "size_sensitive_cpu": "qdq_int8",
            "rk3588_status": "artifact_ready_board_validation_pending",
            "http_serving_status": "stable_local_loopback",
        },
        "recommendations": [
            {"target": "PC CPU serving", "recommendation": recommendation,
             "reason": "Stable ONNX Runtime path."},
            {"target": "Size-sensitive CPU deployment", "recommendation": "qdq_int8",
             "reason": "~73% smaller artifact."},
            {"target": "RK3588 deployment", "recommendation": "artifact_ready_board_validation_pending",
             "reason": "RKNN conversion succeeded."},
        ],
        "risks": [
            "QDQ INT8 uses dummy calibration; real accuracy is not validated.",
            "RKNN Lite2 board-side latency is not available without RK3588 hardware.",
        ],
        "next_validation": [
            "Run INT8 calibration with representative images.",
            "Run RKNN Lite2 on RK3588 board.",
        ],
    }


class TestBuildAdvisorReport:
    """Advisor explanation generation."""

    def test_pc_cpu_onnx_fp32(self):
        report = _make_decision("onnx_fp32")
        content = build_advisor_report(report)
        assert "ONNX FP32" in content
        assert "PC CPU" in content

    def test_pc_cpu_insufficient_evidence(self):
        report = _make_decision("insufficient_evidence")
        content = build_advisor_report(report)
        assert "insufficient" in content.lower() or "missing" in content.lower()

    def test_qdq_int8_mentions_accuracy_risk(self):
        report = _make_decision()
        content = build_advisor_report(report)
        assert "QDQ INT8" in content
        assert "accuracy" in content.lower()

    def test_rk3588_board_validation_pending(self):
        report = _make_decision()
        content = build_advisor_report(report)
        assert "board" in content.lower()
        assert "pending" in content.lower()

    def test_risks_empty_does_not_crash(self):
        report = _make_decision()
        report["risks"] = []
        content = build_advisor_report(report)
        assert "ONNX FP32" in content

    def test_missing_fields_do_not_crash(self):
        content = build_advisor_report({})
        assert len(content) > 0

    def test_next_steps_use_consecutive_numbers(self):
        report = _make_decision()
        content = build_advisor_report(report)
        assert "1. Run INT8 calibration" in content
        assert "2. Run RKNN Lite2" in content

    def test_missing_evidence_appears_in_report(self):
        report = _make_decision()
        report["missing_evidence"] = ["HTTP benchmark", "RKNN conversion"]
        content = build_advisor_report(report)
        assert "Missing Evidence" in content
        assert "HTTP benchmark" in content
        assert "RKNN conversion" in content
        assert "confidence is limited" in content


class TestAdvisorOutput:
    """Markdown output."""

    def test_markdown_can_be_written(self, tmp_path):
        report = _make_decision()
        content = build_advisor_report(report)
        p = tmp_path / "advisor.md"
        write_advisor_markdown(content, p)
        assert p.exists()
        text = p.read_text(encoding="utf-8")
        assert "Deployment Advisor" in text
# -- From test_compare_reports.py --
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
# -- From test_benchmark_report.py --
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
# -- From test_rknn_report.py --
"""Tests for RKNN conversion report output."""

from pathlib import Path

from src.rknn.report import write_rknn_conversion_json, write_rknn_conversion_markdown


class TestRknnReport:
    """Report output — no RKNN Toolkit needed."""

    def test_json_report_can_be_written(self, tmp_path):
        report = {
            "source_model": "test.onnx",
            "output_model": "test.rknn",
            "target_platform": "rk3588",
            "do_quantization": False,
            "onnx_size_mb": 5.0,
            "rknn_size_mb": 3.0,
            "status": "ok",
            "message": "Done",
        }
        p = tmp_path / "report.json"
        write_rknn_conversion_json(report, p)
        assert p.exists()

        import json
        data = json.loads(p.read_text(encoding="utf-8"))
        assert data["status"] == "ok"

    def test_markdown_report_contains_fields(self, tmp_path):
        report = {
            "source_model": "test.onnx",
            "output_model": "test.rknn",
            "target_platform": "rk3588",
            "do_quantization": False,
            "onnx_size_mb": 5.0,
            "rknn_size_mb": 3.0,
            "status": "ok",
            "message": "Done",
        }
        p = tmp_path / "report.md"
        write_rknn_conversion_markdown(report, p)
        content = p.read_text(encoding="utf-8")
        assert "rk3588" in content
        assert "RKNN" in content
        assert "5.0" in content
# -- From test_rknn_convert.py --
"""Tests for RKNN conversion config and error handling."""

from pathlib import Path

import pytest
import yaml

from src.rknn.convert import convert_onnx_to_rknn


class TestRknnConvert:
    """Conversion error handling — no real RKNN Toolkit needed."""

    def test_missing_input_model_returns_error(self):
        config = {
            "input_model": "outputs/onnx/nonexistent.onnx",
            "output_model": "outputs/rknn/test.rknn",
        }
        report = convert_onnx_to_rknn(config)
        assert report["status"] == "error"
        assert "not found" in report["message"]

    def test_missing_rknn_toolkit_returns_clear_error(self, tmp_path):
        # Create a dummy ONNX-like file that exists
        dummy_onnx = tmp_path / "model.onnx"
        dummy_onnx.write_text("not a real onnx but file exists")
        config = {
            "input_model": str(dummy_onnx),
            "output_model": str(tmp_path / "test.rknn"),
        }
        report = convert_onnx_to_rknn(config)
        assert report["status"] == "error"
        # Should mention RKNN Toolkit, not traceback
        assert "RKNN Toolkit" in report["message"]

    def test_config_can_be_loaded(self):
        config_path = Path("configs/rknn_convert.yaml")
        assert config_path.exists()
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        assert config["input_model"] == "outputs/onnx/mobilenetv3_small.onnx"
        assert config["target_platform"] == "rk3588"
