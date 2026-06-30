"""Quantisation experiment runner — runs a matrix of strategies across models.

Usage
-----
.. code-block:: powershell

    # Run a single experiment
    python -m src.experiments.run_quant_experiments --config configs/experiments/quantization_controls.yaml --experiment mobilenetv3_dynamic_linear_only

    # Run all experiments
    python -m src.experiments.run_quant_experiments --config configs/experiments/quantization_controls.yaml --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml


def _decide(exp_id: str, result: dict) -> str:
    """Apply decision rules to an experiment result.

    Returns one of: invalid_artifact, reject_consistency_failed,
    reject_distribution_shift, size_optimized_baseline, recommended_candidate.
    """
    if result.get("status") == "failed":
        return "invalid_artifact"

    t1 = result.get("top1_consistency")
    t5 = result.get("top5_consistency")
    cos = result.get("mean_logits_cosine_similarity")

    if t1 is None or t5 is None or cos is None:
        return "insufficient_data"

    if t1 < 0.8:
        return "reject_consistency_failed"
    if t5 < 0.9:
        return "reject_consistency_failed"
    if cos < 0.98:
        return "reject_distribution_shift"

    fp32_lat = result.get("fp32_mean_latency_ms")
    int8_lat = result.get("int8_mean_latency_ms")
    if fp32_lat and int8_lat and int8_lat <= fp32_lat:
        return "recommended_candidate"
    return "size_optimized_baseline"


def run_experiment(exp: dict, models: dict, datasets: dict) -> Dict[str, Any]:
    """Run a single experiment end-to-end.

    Returns a dict with metrics or a failure report.
    """
    model_cfg = models.get(exp["model"], {})
    model_id = exp["model"]
    exp_id = exp["id"]
    method = exp.get("method", "static_qdq_real")
    preproc = datasets.get("calibration", {})
    val_ds = datasets.get("validation", {})

    # Build quantization config
    quant_cfg: Dict[str, Any] = {
        "input_model": model_cfg.get("fp32", ""),
        "output_model": exp.get("output_model", ""),
    }

    if method == "dynamic":
        quant_cfg["method"] = "dynamic"
        if exp.get("op_types_to_quantize"):
            quant_cfg["op_types_to_quantize"] = exp["op_types_to_quantize"]
    else:
        cal = {
            "image_dir": preproc.get("image_dir", "samples/images/calibration_real"),
            "max_samples": int(preproc.get("max_samples", 50)),
            "preprocessing": model_cfg.get("preprocessing", {}).get("mode", "simple"),
            "resize_shorter": int(model_cfg.get("preprocessing", {}).get("resize_shorter", 256)),
            "center_crop": model_cfg.get("preprocessing", {}).get("center_crop", [224, 224]),
            "mean": model_cfg.get("preprocessing", {}).get("mean", [0.485, 0.456, 0.406]),
            "std": model_cfg.get("preprocessing", {}).get("std", [0.229, 0.224, 0.225]),
        }
        quant_cfg["calibration"] = cal
        quant_cfg["method"] = "static_qdq_real"
        if exp.get("preprocess"):
            quant_cfg["quantization"] = {
                "preprocess": True,
                "skip_symbolic_shape": exp.get("skip_symbolic_shape", True),
                "fallback_to_original": exp.get("fallback_to_original", True),
            }

    # Run quantization
    from src.quantization.quantize_onnx import run_quantization
    try:
        run_quantization(
            quant_cfg["input_model"],
            quant_cfg["output_model"],
            config=quant_cfg,
        )
    except Exception as exc:
        return {"experiment_id": exp_id, "model_id": model_id,
                "method": method, "status": "failed",
                "error": f"quantization: {exc}"}

    # Run validation
    val_cfg = {
        "fp32_model": model_cfg.get("fp32", ""),
        "int8_model": exp.get("output_model", ""),
        "image_dir": val_ds.get("image_dir", "samples/images/validation_real"),
        "preprocessing": model_cfg.get("preprocessing", {}),
    }
    from src.validation.output_consistency import run_validation
    result = run_validation(val_cfg)

    result["experiment_id"] = exp_id
    result["model_id"] = model_id
    result["method"] = method
    result["decision"] = _decide(exp_id, result)
    return result


def build_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate experiment results into a decision summary."""
    summary = {"experiments": results}
    return summary


def write_summary_json(summary: dict, path: str | Path) -> Path:
    from src.validation.output_consistency import _to_jsonable
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(_to_jsonable(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return p


def write_summary_markdown(summary: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Quantisation Experiment Summary",
        "",
        "| Experiment | Model | Method | Status | Top1 | Top5 | Cosine | Size↓ | Latency | Decision |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for exp in summary.get("experiments", []):
        sid = exp.get("experiment_id", "")
        model = exp.get("model_id", "")
        method = exp.get("method", "")
        status = exp.get("status", "ok")
        t1 = exp.get("top1_consistency", "")
        t5 = exp.get("top5_consistency", "")
        cos = exp.get("mean_logits_cosine_similarity", "")
        sr = exp.get("size_reduction_percent", "")
        # Latency comparison
        fp32_lat = exp.get("fp32_mean_latency_ms")
        int8_lat = exp.get("int8_mean_latency_ms")
        if fp32_lat and int8_lat:
            lat = f"{fp32_lat:.2f}→{int8_lat:.2f}ms"
        else:
            lat = ""
        decision = exp.get("decision", "")

        lines.append(
            f"| {sid} | {model} | {method} | {status} "
            f"| {t1} | {t5} | {cos} | {sr}% | {lat} | {decision} |"
        )

    lines.append("")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantization experiment runner")
    parser.add_argument(
        "--config",
        default="configs/experiments/quantization_controls.yaml",
        help="Path to experiment matrix YAML",
    )
    parser.add_argument("--experiment", help="Run a single experiment by ID")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}")
        raise SystemExit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    experiments: List[dict] = config.get("experiments", [])
    models = config.get("models", {})
    datasets = config.get("datasets", {})

    if args.experiment:
        experiments = [e for e in experiments if e["id"] == args.experiment]
        if not experiments:
            print(f"ERROR: experiment '{args.experiment}' not found")
            raise SystemExit(1)

    results: List[Dict[str, Any]] = []
    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"Experiment: {exp['id']}")
        print(f"{'='*60}")
        result = run_experiment(exp, models, datasets)
        results.append(result)

    summary = build_summary(results)

    out_json = "outputs/reports/quant_experiment_summary.json"
    out_md = "outputs/reports/quant_experiment_summary.md"
    write_summary_json(summary, out_json)
    write_summary_markdown(summary, out_md)

    print(f"\nExperiment summary written:")
    print(f"  JSON: {out_json}")
    print(f"  MD:   {out_md}")
    print()
    for r in results:
        d = r.get("decision", "N/A")
        s = r.get("status", "N/A")
        print(f"  {r.get('experiment_id', '?')}: {s} → {d}")


if __name__ == "__main__":
    main()
