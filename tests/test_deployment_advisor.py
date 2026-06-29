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
