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
