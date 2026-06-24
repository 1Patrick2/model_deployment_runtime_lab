"""Latency benchmark — runs inference in a loop and produces a report.

Usage
-----
.. code-block:: powershell

    python -m src.benchmark.latency --config configs/benchmark.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import yaml

from src.runtime.onnx_runner import OnnxRunner
from src.server.protocol import InferenceRequest
from src.benchmark.report import (
    build_benchmark_report,
    write_json_report,
    write_markdown_report,
)


def run_benchmark(config: dict) -> dict:
    """Run one benchmark round and return the report dict."""
    manifest_path = config["manifest"]
    input_type = config.get("input_type", "dummy")
    input_value = config.get("input", "dummy")
    warmup = int(config.get("warmup", 10))
    repeat = int(config.get("repeat", 100))

    # Create and load runner
    runner = OnnxRunner(
        manifest_path=manifest_path,
        project_root=Path.cwd(),
    )
    runner.load()

    request = InferenceRequest(
        backend="onnx",
        input_type=input_type,
        input=input_value,
    )

    # Warmup
    for _ in range(warmup):
        runner.predict(request)

    # Measure
    records = []
    for _ in range(repeat):
        resp = runner.predict(request)
        if resp.latency_ms:
            records.append({
                "preprocess": resp.latency_ms.preprocess,
                "inference": resp.latency_ms.inference,
                "postprocess": resp.latency_ms.postprocess,
                "total": resp.latency_ms.total,
            })

    # Capture manifest fields before closing
    model_id = runner._manifest.model_id if runner._manifest else ""
    model_variant = runner._manifest.model_variant if runner._manifest else ""
    artifact_path = runner._manifest.artifact_path if runner._manifest else ""

    runner.close()

    # Artifact size
    artifact_size_mb = 0.0
    if artifact_path:
        ap = Path(artifact_path)
        if ap.exists():
            artifact_size_mb = os.path.getsize(ap) / (1024 * 1024)
        elif (Path.cwd() / ap).exists():
            artifact_size_mb = os.path.getsize(Path.cwd() / ap) / (1024 * 1024)

    return build_benchmark_report(
        model_id=model_id,
        model_variant=model_variant,
        backend="onnx",
        artifact_path=artifact_path,
        artifact_size_mb=artifact_size_mb,
        input_type=input_type,
        input_value=input_value,
        warmup=warmup,
        repeat=repeat,
        latency_records=records,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="ONNX latency benchmark")
    parser.add_argument(
        "--config",
        default="configs/benchmark.yaml",
        help="Path to benchmark YAML config",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"Running benchmark  (config={config_path})")
    print(f"  backend={config.get('backend')}  manifest={config.get('manifest')}")
    print(f"  warmup={config.get('warmup')}  repeat={config.get('repeat')}")
    print()

    t0 = time.perf_counter()
    report = run_benchmark(config)
    elapsed = time.perf_counter() - t0

    # Write outputs
    json_path = write_json_report(report, config.get("output_json", "outputs/reports/benchmark.json"))
    md_path = write_markdown_report(report, config.get("output_md", "outputs/reports/benchmark.md"))

    print(f"Done  ({elapsed:.1f}s)")
    print(f"  JSON: {json_path}")
    print(f"  MD:   {md_path}")

    # Summary
    t = report["latency_ms"]["total"]
    print(f"  total latency: mean={t['mean']}ms  p50={t['p50']}ms  p95={t['p95']}ms")


if __name__ == "__main__":
    main()
