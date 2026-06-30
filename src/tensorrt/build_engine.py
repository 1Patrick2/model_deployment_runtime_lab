"""TensorRT FP16 engine build and benchmark.

Usage
-----
.. code-block:: powershell

    python -m src.tensorrt.build_engine --config configs/tensorrt/mobilenetv3_small_fp16.yaml
    python -m src.tensorrt.build_engine --config configs/tensorrt/mobilenetv3_small_fp16.yaml --dry-run
    python -m src.tensorrt.build_engine --config configs/tensorrt/mobilenetv3_small_fp16.yaml --benchmark
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

from src.tensorrt.env import collect_tensorrt_env
from src.tensorrt.parse_trtexec import parse_trtexec_output
from src.tensorrt.report import write_tensorrt_json, write_tensorrt_markdown


def _build_command(config: dict) -> List[str]:
    """Construct the trtexec build command."""
    onnx = Path(config["onnx_model"])
    engine = Path(config["engine_output"])
    precision = config.get("precision", "fp16")
    inp = config.get("input_name", "input")
    shape = config.get("input_shape", [1, 3, 224, 224])
    shape_str = f"{inp}:{'x'.join(str(d) for d in shape)}"

    cmd = [
        "trtexec",
        f"--onnx={onnx}",
        f"--saveEngine={engine}",
        f"--{precision}",
        f"--shapes={shape_str}",
    ]
    return cmd


def _benchmark_command(config: dict) -> List[str]:
    """Construct the trtexec benchmark command."""
    engine = Path(config["engine_output"])
    trt = config.get("trtexec", {})
    cmd = [
        "trtexec",
        f"--loadEngine={engine}",
        f"--warmUp={trt.get('warmup', 200)}",
        f"--iterations={trt.get('iterations', 1000)}",
    ]
    return cmd


def run_build(config: dict, dry_run: bool = False) -> dict:
    """Build a TensorRT engine from an ONNX model.

    Returns a structured report dict.
    """
    onnx_path = Path(config["onnx_model"])
    if not onnx_path.exists() and not dry_run:
        return {
            "model_id": config.get("model_id", ""),
            "backend": "tensorrt",
            "precision": config.get("precision", "fp16"),
            "build_status": "failed",
            "error": f"ONNX model not found: {onnx_path}",
        }

    env = collect_tensorrt_env()
    if not env.get("trtexec_available") and not dry_run:
        return {
            "model_id": config.get("model_id", ""),
            "backend": "tensorrt",
            "precision": config.get("precision", "fp16"),
            "build_status": "skipped",
            "reason": env.get("reason", "trtexec not available"),
            "env": env,
        }

    cmd = _build_command(config)
    report: Dict[str, Any] = {
        "model_id": config.get("model_id", ""),
        "backend": "tensorrt",
        "precision": config.get("precision", "fp16"),
        "onnx_model": str(onnx_path),
        "engine_output": config.get("engine_output", ""),
        "env": env,
        "build_command": cmd,
    }

    if dry_run:
        report["build_status"] = "dry_run"
        report["build_output"] = " ".join(cmd)
        return report

    # Actually build
    engine_path = Path(config["engine_output"])
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        report["build_status"] = "ok" if result.returncode == 0 else "failed"
        if result.returncode != 0:
            report["build_error"] = result.stderr[-1000:]
        if engine_path.exists():
            report["engine_size_mb"] = round(
                engine_path.stat().st_size / (1024 * 1024), 2
            )
    except FileNotFoundError:
        report["build_status"] = "failed"
        report["build_error"] = "trtexec not found"
    except subprocess.TimeoutExpired:
        report["build_status"] = "failed"
        report["build_error"] = "build timed out"

    return report


def run_benchmark(config: dict, dry_run: bool = False) -> dict:
    """Benchmark a built TensorRT engine.

    Returns a structured report dict with metrics.
    """
    engine_path = Path(config["engine_output"])
    if not engine_path.exists():
        return {
            "model_id": config.get("model_id", ""),
            "backend": "tensorrt",
            "benchmark_status": "skipped",
            "reason": f"Engine not found: {engine_path}",
        }

    env = collect_tensorrt_env()
    if not env.get("trtexec_available"):
        return {
            "model_id": config.get("model_id", ""),
            "backend": "tensorrt",
            "benchmark_status": "skipped",
            "reason": env.get("reason", "trtexec not available"),
            "env": env,
        }

    cmd = _benchmark_command(config)
    report: Dict[str, Any] = {
        "model_id": config.get("model_id", ""),
        "backend": "tensorrt",
        "precision": config.get("precision", "fp16"),
        "benchmark_command": cmd,
    }

    if dry_run:
        report["benchmark_status"] = "dry_run"
        report["benchmark_output"] = " ".join(cmd)
        return report

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            report["benchmark_status"] = "ok"
            report["metrics"] = parse_trtexec_output(result.stdout)
        else:
            report["benchmark_status"] = "failed"
            report["benchmark_error"] = result.stderr[-1000:]
    except FileNotFoundError:
        report["benchmark_status"] = "failed"
        report["benchmark_error"] = "trtexec not found"
    except subprocess.TimeoutExpired:
        report["benchmark_status"] = "failed"
        report["benchmark_error"] = "benchmark timed out"

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="TensorRT FP16 engine build and benchmark")
    parser.add_argument(
        "--config",
        default="configs/tensorrt/mobilenetv3_small_fp16.yaml",
        help="Path to TensorRT config YAML",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark after build")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}")
        raise SystemExit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    # Build
    build_report = run_build(config, dry_run=args.dry_run)
    if args.benchmark and build_report.get("build_status") == "ok":
        bench_report = run_benchmark(config, dry_run=args.dry_run)
        build_report.update(bench_report)

    # Write reports
    write_tensorrt_json(build_report, config.get("report_json", "outputs/reports/tensorrt_report.json"))
    write_tensorrt_markdown(build_report, config.get("report_md", "outputs/reports/tensorrt_report.md"))

    print(f"TensorRT report written:")
    print(f"  build_status: {build_report.get('build_status', 'N/A')}")
    if "benchmark_status" in build_report:
        print(f"  benchmark_status: {build_report.get('benchmark_status', 'N/A')}")
    if "engine_size_mb" in build_report:
        print(f"  engine_size_mb: {build_report['engine_size_mb']}")


if __name__ == "__main__":
    main()
