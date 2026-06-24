"""Benchmark statistics and report generation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List


def compute_stats(values: list[float]) -> Dict[str, float]:
    """Compute summary statistics for a list of latency values.

    Returns ``{mean, min, max, p50, p95}``.
    """
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0, "p95": 0.0}

    s = sorted(values)
    n = len(s)
    mean = sum(s) / n

    def percentile(p: float) -> float:
        idx = max(0, min(n - 1, int(math.ceil(p / 100 * n) - 1)))
        return s[idx]

    return {
        "mean": round(mean, 4),
        "min": round(s[0], 4),
        "max": round(s[-1], 4),
        "p50": round(percentile(50), 4),
        "p95": round(percentile(95), 4),
    }


def build_benchmark_report(
    model_id: str,
    model_variant: str,
    backend: str,
    artifact_path: str,
    artifact_size_mb: float,
    input_type: str,
    input_value: str,
    warmup: int,
    repeat: int,
    latency_records: list[Dict[str, float]],
) -> Dict[str, Any]:
    """Aggregate latency records into a structured benchmark report."""
    stages = ["preprocess", "inference", "postprocess", "total"]
    latency_stats = {}

    for stage in stages:
        values = [r[stage] for r in latency_records if stage in r]
        latency_stats[stage] = compute_stats(values)

    return {
        "model": {
            "model_id": model_id,
            "model_variant": model_variant,
            "backend": backend,
            "artifact_path": artifact_path,
            "artifact_size_mb": round(artifact_size_mb, 2),
        },
        "input": {
            "input_type": input_type,
            "input": input_value,
        },
        "benchmark": {
            "warmup": warmup,
            "repeat": repeat,
        },
        "latency_ms": latency_stats,
    }


def write_json_report(report: Dict[str, Any], path: str | Path) -> Path:
    """Write the benchmark report as JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def write_markdown_report(report: Dict[str, Any], path: str | Path) -> Path:
    """Write the benchmark report as a human-readable Markdown file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    m = report["model"]
    inp = report["input"]
    bench = report["benchmark"]
    lat = report["latency_ms"]

    lines = [
        "# Benchmark Report",
        "",
        "## Model",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| model_id | {m['model_id']} |",
        f"| variant | {m['model_variant']} |",
        f"| backend | {m['backend']} |",
        f"| artifact size | {m['artifact_size_mb']} MB |",
        "",
        f"**Input:** type={inp['input_type']}, value={inp['input']}",
        "",
        f"**Benchmark settings:** warmup={bench['warmup']}, repeat={bench['repeat']}",
        "",
        "## Latency (ms)",
        "",
        "| Stage | Mean | P50 | P95 | Min | Max |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for stage in ["preprocess", "inference", "postprocess", "total"]:
        s = lat.get(stage, {})
        lines.append(
            f"| {stage} | {s.get('mean', '—')} | {s.get('p50', '—')} "
            f"| {s.get('p95', '—')} | {s.get('min', '—')} | {s.get('max', '—')} |"
        )

    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


__all__ = [
    "compute_stats",
    "build_benchmark_report",
    "write_json_report",
    "write_markdown_report",
]
