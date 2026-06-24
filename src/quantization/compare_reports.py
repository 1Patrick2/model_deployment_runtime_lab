"""FP32 / INT8 benchmark comparison report.

Usage
-----
.. code-block:: powershell

    python -m src.quantization.compare_reports ^
        --baseline outputs/reports/benchmark_mobilenetv3_small_onnx_fp32.json ^
        --candidate outputs/reports/benchmark_mobilenetv3_small_onnx_int8_dynamic.json ^
        --output-json outputs/reports/compare_mobilenetv3_small_fp32_vs_int8_dynamic.json ^
        --output-md outputs/reports/compare_mobilenetv3_small_fp32_vs_int8_dynamic.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def load_benchmark(path: str | Path) -> Dict[str, Any]:
    """Load a benchmark JSON report."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Benchmark report not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_reports(
    baseline: Dict[str, Any],
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare FP32 and INT8 benchmark reports.

    Returns a structured comparison dict with size and latency deltas.
    """
    b_model = baseline["model"]
    c_model = candidate["model"]
    b_lat = baseline["latency_ms"]["total"]
    c_lat = candidate["latency_ms"]["total"]
    b_inf = baseline["latency_ms"]["inference"]
    c_inf = candidate["latency_ms"]["inference"]

    fp32_size = b_model["artifact_size_mb"]
    int8_size = c_model["artifact_size_mb"]

    size_reduction_pct = round((1 - int8_size / fp32_size) * 100, 2) if fp32_size > 0 else 0.0
    total_speedup = round(b_lat["mean"] / c_lat["mean"], 4) if c_lat["mean"] > 0 else 0.0
    inference_speedup = round(b_inf["mean"] / c_inf["mean"], 4) if c_inf["mean"] > 0 else 0.0

    return {
        "baseline": {
            "model_id": b_model["model_id"],
            "model_variant": b_model["model_variant"],
            "backend": b_model["backend"],
            "artifact_size_mb": b_model["artifact_size_mb"],
            "inference_mean_ms": b_inf["mean"],
            "total_mean_ms": b_lat["mean"],
            "total_p50_ms": b_lat["p50"],
            "total_p95_ms": b_lat["p95"],
        },
        "candidate": {
            "model_id": c_model["model_id"],
            "model_variant": c_model["model_variant"],
            "backend": c_model["backend"],
            "artifact_size_mb": c_model["artifact_size_mb"],
            "inference_mean_ms": c_inf["mean"],
            "total_mean_ms": c_lat["mean"],
            "total_p50_ms": c_lat["p50"],
            "total_p95_ms": c_lat["p95"],
        },
        "comparison": {
            "size_reduction_percent": size_reduction_pct,
            "total_speedup_ratio": total_speedup,
            "inference_speedup_ratio": inference_speedup,
        },
    }


def write_comparison_json(report: Dict[str, Any], path: str | Path) -> Path:
    """Write comparison report as JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def write_comparison_markdown(report: Dict[str, Any], path: str | Path) -> Path:
    """Write comparison report as Markdown."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    b = report["baseline"]
    c = report["candidate"]
    cmp = report["comparison"]
    sr = cmp["size_reduction_percent"]
    ts = cmp["total_speedup_ratio"]
    infs = cmp["inference_speedup_ratio"]

    lines = [
        "# FP32 vs INT8 Dynamic Quantization Report",
        "",
        "| Variant | Size MB | Inference Mean | Total Mean | Total P50 | Total P95 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| {b['model_variant']} | {b['artifact_size_mb']} | {b['inference_mean_ms']} | {b['total_mean_ms']} | {b['total_p50_ms']} | {b['total_p95_ms']} |",
        f"| {c['model_variant']} | {c['artifact_size_mb']} | {c['inference_mean_ms']} | {c['total_mean_ms']} | {c['total_p50_ms']} | {c['total_p95_ms']} |",
        "",
        "## Summary",
        "",
        f"- **Size reduction:** {sr}%  ({b['artifact_size_mb']} MB → {c['artifact_size_mb']} MB)",
        f"- **Total latency speedup:** {ts}x  ({b['total_mean_ms']}ms → {c['total_mean_ms']}ms)",
        f"- **Inference speedup:** {infs}x  ({b['inference_mean_ms']}ms → {c['inference_mean_ms']}ms)",
        "",
    ]

    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare FP32 and INT8 benchmark reports"
    )
    parser.add_argument("--baseline", required=True, help="FP32 benchmark JSON path")
    parser.add_argument("--candidate", required=True, help="INT8 benchmark JSON path")
    parser.add_argument("--output-json", required=True, help="Output JSON path")
    parser.add_argument("--output-md", required=True, help="Output Markdown path")
    args = parser.parse_args()

    baseline = load_benchmark(args.baseline)
    candidate = load_benchmark(args.candidate)

    report = compare_reports(baseline, candidate)

    jp = write_comparison_json(report, args.output_json)
    mp = write_comparison_markdown(report, args.output_md)

    print(f"Comparison report written:")
    print(f"  JSON: {jp}")
    print(f"  MD:   {mp}")
    cmp = report["comparison"]
    print(f"  Size reduction: {cmp['size_reduction_percent']}%")
    print(f"  Total speedup:  {cmp['total_speedup_ratio']}x")
    print(f"  Inference speedup: {cmp['inference_speedup_ratio']}x")


if __name__ == "__main__":
    main()
