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
from src.validation.metrics import (
    max_absolute_error,
    max_confidence_drift,
    mean_absolute_error,
    mean_confidence_drift,
    mean_logits_cosine_similarity,
    mean_top5_overlap,
    top_k_indices,
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

    if not fp32_path.exists():
        raise FileNotFoundError(f"FP32 ONNX model not found: {fp32_path}")
    if not int8_path.exists():
        raise FileNotFoundError(f"INT8 ONNX model not found: {int8_path}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    preproc_cfg = config.get("preprocessing", {})
    preproc_mode = preproc_cfg.get("mode", "simple")

    if preproc_mode == "imagenet_v1":
        from src.runtime.image_preprocess import preprocess_image_imagenet as pp_func
        target_size = tuple(preproc_cfg.get("center_crop", [224, 224]))
        resize_shorter = int(preproc_cfg.get("resize_shorter", 256))
    else:
        from src.runtime.image_preprocess import preprocess_image as pp_func
        target_size = tuple(preproc_cfg.get("resize", [224, 224]))
        resize_shorter = 0  # unused in simple mode

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
    per_image_results: list[dict] = []

    for img_path in image_paths:
        if preproc_mode == "imagenet_v1":
            tensor = pp_func(
                str(img_path),
                crop_size=target_size,
                resize_shorter=resize_shorter,
                mean=mean,
                std=std,
            )
        else:
            tensor = pp_func(
                str(img_path),
                target_size=target_size,
                mean=mean,
                std=std,
            )

        t0 = time.perf_counter()
        fp32_out = fp32_session.run(None, {fp32_input_name: tensor})
        fp32_latencies.append((time.perf_counter() - t0) * 1000)
        f_logits = fp32_out[0][0]
        fp32_logits.append(f_logits)

        t0 = time.perf_counter()
        int8_out = int8_session.run(None, {int8_input_name: tensor})
        int8_latencies.append((time.perf_counter() - t0) * 1000)
        i_logits = int8_out[0][0]
        int8_logits.append(i_logits)

        # Per-image diagnostics
        fp32_top1 = int(top_k_indices(f_logits, 1)[0])
        int8_top1 = int(top_k_indices(i_logits, 1)[0])
        fp32_t5 = [int(x) for x in top_k_indices(f_logits, 5)]
        int8_t5 = [int(x) for x in top_k_indices(i_logits, 5)]
        top1_match = bool(fp32_top1 == int8_top1)
        fp32_1_in_int8_5 = bool(fp32_top1 in int8_t5)
        fp32_set = set(fp32_t5)
        int8_set = set(int8_t5)
        top5_ov = round(
            len(fp32_set & int8_set) / len(fp32_set | int8_set)
            if fp32_set or int8_set
            else 1.0,
            4,
        )
        f_norm = f_logits / (np.linalg.norm(f_logits) + 1e-10)
        i_norm = i_logits / (np.linalg.norm(i_logits) + 1e-10)
        cos_sim = round(float(np.dot(f_norm, i_norm)), 4)
        per_image_results.append(
            {
                "image_path": str(img_path.name),
                "fp32_top1_index": fp32_top1,
                "int8_top1_index": int8_top1,
                "fp32_top5_indices": fp32_t5,
                "int8_top5_indices": int8_t5,
                "top1_match": top1_match,
                "fp32_top1_in_int8_top5": fp32_1_in_int8_5,
                "top5_overlap": top5_ov,
                "logits_cosine_similarity": cos_sim,
            }
        )

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

    result["per_image_results"] = per_image_results[:10]  # first 10 samples

    return result


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert NumPy types to JSON-safe Python types."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def write_validation_json(report: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(_to_jsonable(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
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
        ("top1_consistency", "FP32/INT8 top-1 agreement", 0.80, False),
        ("top5_consistency", "FP32 top-1 in INT8 top-5", 0.90, False),
        ("mean_logits_cosine_similarity", "Logits cosine similarity", 0.98, False),
        ("mean_confidence_drift", "Avg confidence drift", 0.05, True),
    ]

    for key, label, threshold, lower_is_better in thresholds:
        val = report.get(key, 0)
        if lower_is_better:
            ok = "PASS" if val <= threshold else "WARN"
        else:
            ok = "PASS" if val >= threshold else "WARN"
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
    _write_per_image_table(report, lines)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _write_per_image_table(report: dict, lines: list) -> None:
    """Append per-image sample results to the Markdown lines."""
    samples = report.get("per_image_results", [])
    if not samples:
        return

    lines += [
        "",
        "## Per-image Samples (first {})".format(len(samples)),
        "",
        "| Image | FP32 top1 | INT8 top1 | Match | FP32 top1 in INT8 top5 | Top5 overlap | Cosine sim |",
        "|---|---|---|---|---|---:|---:|",
    ]
    for s in samples:
        from src.models.imagenet_labels import label_for_class_id

        fp32_name = label_for_class_id(s["fp32_top1_index"])
        int8_name = label_for_class_id(s["int8_top1_index"])
        match = "PASS" if s["top1_match"] else "FAIL"
        in5 = "PASS" if s["fp32_top1_in_int8_top5"] else "FAIL"
        lines.append(
            f"| {s['image_path']} | {fp32_name} | {int8_name} "
            f"| {match} | {in5} | {s['top5_overlap']} | {s['logits_cosine_similarity']} |"
        )


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
