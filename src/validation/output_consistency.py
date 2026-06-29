"""FP32 vs INT8 output consistency validation.

Usage
-----
.. code-block:: powershell

    python -m src.validation.output_consistency --config configs/validate_fp32_vs_qdq_real.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import onnxruntime
import yaml

from src.benchmark.artifact import compute_onnx_artifact_size_mb
from src.runtime.image_preprocess import preprocess_image
from src.validation.metrics import (
    max_absolute_error,
    max_confidence_drift,
    mean_absolute_error,
    mean_confidence_drift,
    mean_logits_cosine_similarity,
    mean_top5_overlap,
    top1_consistency,
    top5_consistency,
)


def load_onnx_session(
    model_path: str | Path,
    providers: list[str] | None = None,
) -> onnxruntime.InferenceSession:
    """Load an ONNX model for inference."""
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"ONNX model not found: {p}")
    return onnxruntime.InferenceSession(
        str(p), providers=providers or ["CPUExecutionProvider"]
    )


def run_validation(config: dict) -> Dict[str, Any]:
    """Run FP32 vs INT8 output consistency validation.

    Reads configuration, loads both models, processes each image,
    collects logits, and computes comparison metrics.
    """
    fp32_path = Path(config["fp32_model"])
    int8_path = Path(config["int8_model"])
    image_dir = Path(config["image_dir"])
    preproc_cfg = config.get("preprocessing", {})

    target_size = tuple(preproc_cfg.get("resize", [224, 224]))
    mean = tuple(preproc_cfg.get("mean", [0.485, 0.456, 0.406]))
    std = tuple(preproc_cfg.get("std", [0.229, 0.224, 0.225]))

    # Collect images
    exts = {".jpg", ".jpeg", ".png"}
    image_paths: list[Path] = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in exts
    )
    if not image_paths:
        raise ValueError(f"No supported images found in {image_dir}")

    # Load sessions
    fp32_session = load_onnx_session(fp32_path)
    int8_session = load_onnx_session(int8_path)

    fp32_input_name = fp32_session.get_inputs()[0].name
    int8_input_name = int8_session.get_inputs()[0].name

    # Run inference on each image
    fp32_logits: List[np.ndarray] = []
    int8_logits: List[np.ndarray] = []
    fp32_latencies: list[float] = []
    int8_latencies: list[float] = []

    for img_path in image_paths:
        tensor = preprocess_image(
            str(img_path),
            target_size=target_size,
            mean=mean,
            std=std,
        )

        t0 = time.perf_counter()
        fp32_out = fp32_session.run(None, {fp32_input_name: tensor})
        fp32_latencies.append((time.perf_counter() - t0) * 1000)
        fp32_logits.append(fp32_out[0][0])

        t0 = time.perf_counter()
        int8_out = int8_session.run(None, {int8_input_name: tensor})
        int8_latencies.append((time.perf_counter() - t0) * 1000)
        int8_logits.append(int8_out[0][0])

    # Compute metrics
    result: Dict[str, Any] = {
        "num_images": len(image_paths),
        "top1_consistency": round(top1_consistency(fp32_logits, int8_logits), 4),
        "top5_consistency": round(top5_consistency(fp32_logits, int8_logits), 4),
        "mean_top5_overlap": round(mean_top5_overlap(fp32_logits, int8_logits), 4),
        "mean_confidence_drift": round(mean_confidence_drift(fp32_logits, int8_logits), 6),
        "max_confidence_drift": round(max_confidence_drift(fp32_logits, int8_logits), 6),
        "mean_logits_cosine_similarity": round(
            mean_logits_cosine_similarity(fp32_logits, int8_logits), 4
        ),
        "mean_absolute_error": round(mean_absolute_error(fp32_logits, int8_logits), 4),
        "max_absolute_error": round(max_absolute_error(fp32_logits, int8_logits), 4),
        "fp32_mean_latency_ms": round(float(np.mean(fp32_latencies)), 2),
        "int8_mean_latency_ms": round(float(np.mean(int8_latencies)), 2),
        "fp32_model_size_mb": round(compute_onnx_artifact_size_mb(fp32_path), 2),
        "int8_model_size_mb": round(compute_onnx_artifact_size_mb(int8_path), 2),
    }

    if result["fp32_model_size_mb"] > 0:
        result["size_reduction_percent"] = round(
            (1 - result["int8_model_size_mb"] / result["fp32_model_size_mb"]) * 100, 2
        )
    else:
        result["size_reduction_percent"] = 0.0

    return result


def write_validation_json(report: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def write_validation_markdown(report: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Quantization Validation Report",
        "",
        "## Summary",
        "",
        f"**Images:** {report.get('num_images', 0)}",
        "",
        "| Metric | Value | Acceptable |",
        "|---|---|---|",
    ]

    thresholds = [
        ("top1_consistency", "FP32/INT8 top-1 agreement", 0.80),
        ("top5_consistency", "FP32 top-1 in INT8 top-5", 0.90),
        ("mean_logits_cosine_similarity", "Logits cosine similarity", 0.98),
        ("mean_confidence_drift", "Avg confidence drift", 0.05),
    ]

    for key, label, threshold in thresholds:
        val = report.get(key, 0)
        ok = "✅" if val >= threshold else "⚠️"
        lines.append(f"| {label} | {val} | {ok} |")

    lines += [
        "",
        "## Detailed Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]

    for key in [
        "mean_top5_overlap", "max_confidence_drift",
        "mean_absolute_error", "max_absolute_error",
        "fp32_mean_latency_ms", "int8_mean_latency_ms",
        "fp32_model_size_mb", "int8_model_size_mb", "size_reduction_percent",
    ]:
        lines.append(f"| {key} | {report.get(key, '')} |")

    lines.append("")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantization output consistency validation")
    parser.add_argument(
        "--config",
        default="configs/validate_fp32_vs_qdq_real.yaml",
        help="Path to validation YAML config",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}")
        raise SystemExit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    print("Running FP32 vs INT8 output consistency validation...")
    print(f"  FP32:  {config.get('fp32_model')}")
    print(f"  INT8:  {config.get('int8_model')}")
    print(f"  images: {config.get('image_dir')}")
    print()

    report = run_validation(config)

    out_json = config.get("output_json", "outputs/reports/quant_validation.json")
    out_md = config.get("output_md", "outputs/reports/quant_validation.md")

    write_validation_json(report, out_json)
    write_validation_markdown(report, out_md)

    print("Quant validation report written:")
    print(f"  JSON: {out_json}")
    print(f"  MD:   {out_md}")
    print()
    print(f"  top1_consistency:               {report.get('top1_consistency', 'N/A')}")
    print(f"  top5_consistency:               {report.get('top5_consistency', 'N/A')}")
    print(f"  mean_logits_cosine_similarity:  {report.get('mean_logits_cosine_similarity', 'N/A')}")
    print(f"  mean_confidence_drift:          {report.get('mean_confidence_drift', 'N/A')}")
    print(f"  size_reduction_percent:         {report.get('size_reduction_percent', 'N/A')}%")


if __name__ == "__main__":
    main()
