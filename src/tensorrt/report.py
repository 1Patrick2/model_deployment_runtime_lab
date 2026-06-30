"""TensorRT report generation — JSON and Markdown output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def _safe(val: Any) -> str:
    return str(val) if val is not None else "N/A"


def write_tensorrt_json(report: Dict[str, Any], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def write_tensorrt_markdown(report: Dict[str, Any], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# TensorRT FP16 Benchmark Report",
        "",
        "## Environment",
        "",
        "| Field | Value |",
        "|---|---|",
    ]
    env = report.get("env", {})
    lines.append(f"| trtexec available | {_safe(env.get('trtexec_available'))} |")
    lines.append(f"| GPU | {_safe(env.get('gpu_name', 'N/A'))} |")
    lines.append(f"| TensorRT version | {_safe(env.get('tensorrt_version', 'N/A'))} |")

    lines += [
        "",
        "## Build",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| model_id | {_safe(report.get('model_id'))} |",
        f"| precision | {_safe(report.get('precision'))} |",
        f"| build_status | {_safe(report.get('build_status'))} |",
        f"| engine_size_mb | {_safe(report.get('engine_size_mb'))} |",
    ]

    lines += [
        "",
        "## Benchmark",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    metrics = report.get("metrics", {})
    if metrics.get("throughput_qps"):
        lines.append(f"| Throughput (qps) | {_safe(metrics['throughput_qps'])} |")
    lat = metrics.get("latency_ms", {})
    if lat:
        lines.append(f"| Mean Latency (ms) | {_safe(lat.get('mean'))} |")
        lines.append(f"| Median Latency (ms) | {_safe(lat.get('median'))} |")
        lines.append(f"| Min Latency (ms) | {_safe(lat.get('min'))} |")
        lines.append(f"| Max Latency (ms) | {_safe(lat.get('max'))} |")
    if not metrics:
        lines.append(f"| Status | {_safe(report.get('benchmark_status', 'not run'))} |")

    if report.get("build_status") == "skipped":
        lines += [
            "",
            "## Notes",
            "",
            f"- TensorRT benchmark was skipped: {report.get('reason', 'not available')}",
            "- This is expected on CPU-only systems.",
            "- To run TensorRT benchmarks, install TensorRT and ensure trtexec is in PATH.",
        ]

    lines.append("")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p
