"""RKNN conversion report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def write_rknn_conversion_json(report: Dict[str, Any], path: str | Path) -> Path:
    """Write RKNN conversion report as JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def write_rknn_conversion_markdown(report: Dict[str, Any], path: str | Path) -> Path:
    """Write RKNN conversion report as Markdown."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# RKNN Conversion Report",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| source_model | {report.get('source_model', '')} |",
        f"| output_model | {report.get('output_model', '')} |",
        f"| target_platform | {report.get('target_platform', '')} |",
        f"| do_quantization | {report.get('do_quantization', False)} |",
        f"| onnx_size_mb | {report.get('onnx_size_mb', '')} |",
        f"| rknn_size_mb | {report.get('rknn_size_mb', '')} |",
        f"| status | {report.get('status', '')} |",
        f"| message | {report.get('message', '')} |",
        "",
    ]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


__all__ = [
    "write_rknn_conversion_json",
    "write_rknn_conversion_markdown",
]
