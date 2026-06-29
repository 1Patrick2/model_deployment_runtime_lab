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
