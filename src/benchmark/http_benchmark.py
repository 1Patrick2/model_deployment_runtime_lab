"""HTTP inference benchmark — measures end-to-end serving latency.

Usage
-----
.. code-block:: powershell

    # Terminal 1 — start server
    conda activate mdrl-runtime
    python -m src.server.http_server --backend onnx --manifest models/manifests/mobilenetv3_small_onnx_fp32.json --port 8001

    # Terminal 2 — run benchmark
    conda activate mdrl-runtime
    python -m src.benchmark.http_benchmark --config configs/http_benchmark_fp32_dummy.yaml
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.benchmark.report import compute_stats, write_json_report, write_markdown_report


def send_infer_request(
    endpoint: str,
    payload: dict,
    timeout_sec: float = 10,
) -> tuple[float, Optional[dict]]:
    """Send a POST /infer request and return (client_latency_ms, response_dict).

    Returns ``(latency, None)`` on failure.
    """
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            client_ms = (time.perf_counter() - t0) * 1000
            body = json.loads(resp.read().decode("utf-8"))
        return client_ms, body
    except Exception:
        return (time.perf_counter() - t0) * 1000, None


def run_http_benchmark(config: dict) -> Dict[str, Any]:
    """Execute one HTTP benchmark round and return the report dict."""
    endpoint = config["endpoint"]
    req_cfg = config.get("request", {})
    bench_cfg = config.get("benchmark", {})
    warmup = int(bench_cfg.get("warmup", 5))
    repeat = int(bench_cfg.get("repeat", 50))
    timeout_sec = float(bench_cfg.get("timeout_sec", 10))

    payload = {
        "input_type": req_cfg.get("input_type", "dummy"),
        "input": req_cfg.get("input", "dummy"),
        "top_k": int(req_cfg.get("top_k", 5)),
    }

    # Warmup
    for _ in range(warmup):
        send_infer_request(endpoint, payload, timeout_sec)

    # Measure
    client_latencies: list[float] = []
    server_pre: list[float] = []
    server_inf: list[float] = []
    server_post: list[float] = []
    server_total: list[float] = []
    successes = 0
    failures = 0
    sample_prediction: Optional[dict] = None
    model_id = ""
    model_variant = ""
    backend = ""

    for _ in range(repeat):
        client_ms, resp = send_infer_request(endpoint, payload, timeout_sec)
        client_latencies.append(client_ms)

        if resp is not None and resp.get("status") == "ok":
            successes += 1
            if sample_prediction is None and resp.get("top_k_predictions"):
                sample_prediction = resp["top_k_predictions"][:3]
            model_id = resp.get("model_id", model_id)
            model_variant = resp.get("model_variant", model_variant)
            backend = resp.get("backend", backend)

            lm = resp.get("latency_ms", {})
            server_pre.append(lm.get("preprocess", 0))
            server_inf.append(lm.get("inference", 0))
            server_post.append(lm.get("postprocess", 0))
            server_total.append(lm.get("total", 0))
        else:
            failures += 1

    total = successes + failures
    success_rate = round(successes / total, 4) if total > 0 else 0.0

    report: Dict[str, Any] = {
        "benchmark_type": "http_inference",
        "endpoint": endpoint,
        "input_type": payload["input_type"],
        "input": payload["input"],
        "top_k": payload["top_k"],
        "warmup": warmup,
        "repeat": repeat,
        "success_count": successes,
        "failure_count": failures,
        "success_rate": success_rate,
        "client_total_ms": compute_stats(client_latencies),
        "model_id": model_id,
        "model_variant": model_variant,
        "backend": backend,
    }

    if server_total:
        report["server_total_ms"] = compute_stats(server_total)
    if server_pre:
        report["server_preprocess_ms"] = compute_stats(server_pre)
    if server_inf:
        report["server_inference_ms"] = compute_stats(server_inf)
    if server_post:
        report["server_postprocess_ms"] = compute_stats(server_post)
    if sample_prediction:
        report["sample_predictions"] = sample_prediction

    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="HTTP inference benchmark")
    parser.add_argument(
        "--config",
        default="configs/http_benchmark_fp32_dummy.yaml",
        help="Path to HTTP benchmark YAML config",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: config file not found: {config_path}")
        raise SystemExit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    endpoint = config.get("endpoint", "http://127.0.0.1:8001/infer")
    print(f"HTTP benchmark starting ...")
    print(f"  endpoint: {endpoint}")
    print(f"  input:    {config.get('request', {}).get('input', 'dummy')}")
    print(f"  warmup:   {config.get('benchmark', {}).get('warmup', 5)}")
    print(f"  repeat:   {config.get('benchmark', {}).get('repeat', 50)}")
    print()

    # Quick health check
    try:
        health_resp = urllib.request.urlopen(
            endpoint.replace("/infer", "/health"), timeout=5
        )
        assert health_resp.status == 200
        print("  health check: OK")
    except Exception as exc:
        print(f"  health check: FAILED  ({exc})")
        print("Hint: make sure the HTTP server is running on the expected port.")
        raise SystemExit(1)

    print()
    report = run_http_benchmark(config)

    out_json = config.get("output", {}).get(
        "json", "outputs/reports/http_benchmark.json"
    )
    out_md = config.get("output", {}).get(
        "markdown", "outputs/reports/http_benchmark.md"
    )

    write_json_report(report, out_json)
    _write_http_markdown(report, out_md)

    print(f"Done.")
    print(f"  JSON: {out_json}")
    print(f"  MD:   {out_md}")
    print(f"  success_rate: {report['success_rate']*100:.0f}%")
    print(f"  client_total_ms mean: {report['client_total_ms']['mean']} ms")
    if "server_total_ms" in report:
        print(f"  server_total_ms mean:  {report['server_total_ms']['mean']} ms")
    print(f"  samples: {report.get('sample_predictions', 'N/A')}")


def _write_http_markdown(report: Dict[str, Any], path: str | Path) -> Path:
    """Write HTTP benchmark report as Markdown."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# HTTP Inference Benchmark Report",
        "",
        "## Summary",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| endpoint | {report.get('endpoint', '')} |",
        f"| backend | {report.get('backend', '')} |",
        f"| model_id | {report.get('model_id', '')} |",
        f"| model_variant | {report.get('model_variant', '')} |",
        f"| input_type | {report.get('input_type', '')} |",
        f"| top_k | {report.get('top_k', '')} |",
        f"| repeat | {report.get('repeat', '')} |",
        f"| success_rate | {report.get('success_rate', 0)*100:.0f}% |",
        "",
        "## Latency",
        "",
        "| Metric | Mean | P50 | P95 | Min | Max |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    stages = [
        "client_total_ms",
        "server_total_ms",
        "server_preprocess_ms",
        "server_inference_ms",
        "server_postprocess_ms",
    ]
    for stage in stages:
        s = report.get(stage)
        if s is None:
            continue
        lines.append(
            f"| {stage} | {s.get('mean', '')} | {s.get('p50', '')} "
            f"| {s.get('p95', '')} | {s.get('min', '')} | {s.get('max', '')} |"
        )

    lines += [
        "",
        "## Notes",
        "",
        "- Client total latency includes HTTP request/response overhead.",
        "- Server total latency is reported by the inference server.",
        "- Difference between client_total_ms and server_total_ms approximates HTTP + serialization overhead.",
        "",
    ]

    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


if __name__ == "__main__":
    main()
