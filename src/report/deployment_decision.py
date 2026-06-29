"""Deployment decision report — aggregates all benchmark/reports into deploy/no-deploy recommendations.

Usage
-----
.. code-block:: powershell

    python -m src.report.deployment_decision --config configs/deployment_decision.yaml
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def load_json(path: str | Path) -> Optional[Dict[str, Any]]:
    """Load a JSON file, returning ``None`` on failure."""
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Evidence readers ────────────────────────────────────────────


def _read_fp32_benchmark(data: Optional[dict]) -> dict:
    if data is None:
        return {"present": False}
    m = data.get("model", {})
    return {
        "present": True,
        "model_id": m.get("model_id"),
        "artifact_size_mb": m.get("artifact_size_mb"),
        "inference_mean_ms": data.get("latency_ms", {}).get("inference", {}).get("mean"),
        "total_mean_ms": data.get("latency_ms", {}).get("total", {}).get("mean"),
    }


def _read_int8_benchmark(data: Optional[dict]) -> dict:
    if data is None:
        return {"present": False}
    m = data.get("model", {})
    return {
        "present": True,
        "model_id": m.get("model_id"),
        "artifact_size_mb": m.get("artifact_size_mb"),
        "total_mean_ms": data.get("latency_ms", {}).get("total", {}).get("mean"),
    }


def _read_quant_compare(data: Optional[dict]) -> dict:
    if data is None:
        return {"present": False}
    cmp = data.get("comparison", {})
    return {
        "present": True,
        "size_reduction_percent": cmp.get("size_reduction_percent"),
        "total_speedup_ratio": cmp.get("total_speedup_ratio"),
    }


def _read_rknn_convert(data: Optional[dict]) -> dict:
    if data is None:
        return {"present": False}
    return {
        "present": True,
        "status": data.get("status"),
        "artifact_size_mb": data.get("rknn_size_mb"),
        "target_platform": data.get("target_platform"),
    }


def _read_http_benchmark(data: Optional[dict]) -> dict:
    if data is None:
        return {"present": False}
    return {
        "present": True,
        "success_rate": data.get("success_rate"),
        "client_total_ms_mean": data.get("client_total_ms", {}).get("mean"),
        "server_total_ms_mean": data.get("server_total_ms", {}).get("mean"),
    }


# ── Decision rules ──────────────────────────────────────────────


def _decide_http_stability(http: dict) -> Optional[str]:
    if not http.get("present"):
        return None
    return "stable_local_loopback" if (http.get("success_rate", 0) or 0) >= 0.99 else "unstable"


def _decide_pc_cpu(fp32: dict, int8: dict, quant: dict) -> dict:
    if not fp32.get("present"):
        return {"recommendation": "insufficient_evidence", "reason": "FP32 benchmark missing."}
    return {
        "recommendation": "onnx_fp32",
        "reason": "Stable ONNX Runtime path with low local latency and simplest deployment path.",
    }


def _decide_size_sensitive(fp32: dict, int8: dict, quant: dict) -> dict:
    sr = quant.get("size_reduction_percent") if quant.get("present") else None
    if sr is not None and sr >= 50:
        return {
            "recommendation": "qdq_int8",
            "reason": f"~{sr:.0f}% smaller artifact with comparable CPU dummy-input latency.",
        }
    if sr is not None:
        return {
            "recommendation": "qdq_int8",
            "reason": f"~{sr:.0f}% smaller artifact; consider if size is critical.",
        }
    if int8.get("present"):
        return {
            "recommendation": "qdq_int8",
            "reason": "Smaller artifact; exact reduction unknown (comparison report missing).",
        }
    return {"recommendation": "insufficient_evidence", "reason": "INT8 data missing."}


def _decide_rk3588(rknn: dict) -> dict:
    if not rknn.get("present"):
        return {
            "recommendation": "not_available",
            "reason": "No RKNN conversion report found.",
        }
    if rknn.get("status") == "ok":
        return {
            "recommendation": "artifact_ready_board_validation_pending",
            "reason": "RKNN conversion succeeded, but no RK3588 board-side runtime latency has been measured.",
        }
    return {
        "recommendation": "conversion_failed",
        "reason": f"RKNN conversion status: {rknn.get('status')}",
    }


# ── Report builder ──────────────────────────────────────────────


def build_deployment_decision(config: dict) -> Dict[str, Any]:
    """Read all input reports and produce a deployment decision report dict."""
    inputs = config.get("inputs", {})

    fp32 = _read_fp32_benchmark(load_json(inputs.get("fp32_benchmark", "")))
    int8 = _read_int8_benchmark(load_json(inputs.get("int8_benchmark", "")))
    quant = _read_quant_compare(load_json(inputs.get("quant_compare", "")))
    rknn = _read_rknn_convert(load_json(inputs.get("rknn_convert", "")))
    http = _read_http_benchmark(load_json(inputs.get("http_benchmark", "")))

    missing = []
    for name, key in [("fp32_benchmark", "FP32 benchmark"), ("int8_benchmark", "INT8 benchmark"),
                       ("quant_compare", "Quant comparison"), ("rknn_convert", "RKNN conversion"),
                       ("http_benchmark", "HTTP benchmark")]:
        if inputs.get(name) and not Path(inputs[name]).exists():
            missing.append(key)

    http_status = _decide_http_stability(http)
    pc_cpu = _decide_pc_cpu(fp32, int8, quant)
    size_sens = _decide_size_sensitive(fp32, int8, quant)
    rk3588 = _decide_rk3588(rknn)

    risks = []
    if quant.get("present") and quant.get("size_reduction_percent"):
        risks.append("QDQ INT8 uses dummy calibration; real accuracy is not validated.")
    if rknn.get("present") and rknn.get("status") == "ok":
        risks.append("RKNN Lite2 board-side latency is not available without RK3588 hardware.")
    if http.get("present"):
        risks.append("HTTP benchmark is local-loopback only, not production network benchmark.")
    if missing:
        risks.append(f"Missing evidence: {', '.join(missing)}.")

    next_steps = []
    if quant.get("present"):
        next_steps.append("Run INT8 calibration with real representative images.")
    if rknn.get("present") and rknn.get("status") == "ok":
        next_steps.append("Run RKNN Lite2 inference on RK3588 board.")
        next_steps.append("Compare ONNX and RKNN outputs on the same image set.")
    if "HTTP benchmark" in missing:
        next_steps.append("Run HTTP benchmark to measure serving latency.")

    evidence: Dict[str, Any] = {}
    if http.get("present"):
        evidence["http"] = {
            "success_rate": http.get("success_rate"),
            "client_total_ms_mean": http.get("client_total_ms_mean"),
            "server_total_ms_mean": http.get("server_total_ms_mean"),
        }
        if http_status:
            evidence["http"]["status"] = http_status
    if rknn.get("present"):
        evidence["rknn"] = {
            "conversion_status": rknn.get("status"),
            "artifact_size_mb": rknn.get("artifact_size_mb"),
        }
    if fp32.get("present"):
        evidence["fp32"] = {"artifact_size_mb": fp32.get("artifact_size_mb")}
    if int8.get("present"):
        evidence["int8"] = {"artifact_size_mb": int8.get("artifact_size_mb")}

    return {
        "report_type": "deployment_decision",
        "summary": {
            "pc_cpu_serving": pc_cpu["recommendation"],
            "size_sensitive_cpu": size_sens["recommendation"],
            "rk3588_status": rk3588["recommendation"],
            "http_serving_status": http_status or "not_measured",
        },
        "recommendations": [
            {
                "target": "PC CPU serving",
                "recommendation": pc_cpu["recommendation"],
                "reason": pc_cpu["reason"],
            },
            {
                "target": "Size-sensitive CPU deployment",
                "recommendation": size_sens["recommendation"],
                "reason": size_sens["reason"],
            },
            {
                "target": "RK3588 deployment",
                "recommendation": rk3588["recommendation"],
                "reason": rk3588["reason"],
            },
        ],
        "risks": risks,
        "next_validation": next_steps,
        "missing_evidence": missing,
        "evidence": evidence,
    }


# ── Output writers ──────────────────────────────────────────────


def write_decision_json(report: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def write_decision_markdown(report: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Deployment Decision Report",
        "",
        "## Executive Summary",
        "",
        "| Target | Recommendation |",
        "|---|---|",
    ]
    for rec in report.get("recommendations", []):
        lines.append(f"| {rec['target']} | {rec['recommendation']} |")

    lines += [
        "",
        "## Evidence",
        "",
    ]
    ev = report.get("evidence", {})
    if "http" in ev:
        h = ev["http"]
        lines += [
            "### HTTP Serving",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| success_rate | {h.get('success_rate', 'N/A')} |",
            f"| client_total_ms mean | {h.get('client_total_ms_mean', 'N/A')} |",
            f"| server_total_ms mean | {h.get('server_total_ms_mean', 'N/A')} |",
            "",
        ]
    if "rknn" in ev:
        r = ev["rknn"]
        lines += [
            "### RKNN",
            "",
            "| Item | Value |",
            "|---|---|",
            f"| Conversion | {r.get('conversion_status', 'N/A')} |",
            f"| Artifact size | {r.get('artifact_size_mb', 'N/A')} MB |",
            "",
        ]

    lines += [
        "## Risks",
        "",
    ]
    for risk in report.get("risks", []):
        lines.append(f"- {risk}")
    lines.append("")

    lines += [
        "## Next Steps",
        "",
    ]
    for ns in report.get("next_validation", []):
        lines.append(f"1. {ns}")
    lines.append("")

    if report.get("missing_evidence"):
        lines += [
            "## Missing Evidence",
            "",
        ]
        for me in report["missing_evidence"]:
            lines.append(f"- {me}")
        lines.append("")

    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Deployment decision report")
    parser.add_argument(
        "--config",
        default="configs/deployment_decision.yaml",
        help="Path to deployment decision YAML config",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}")
        raise SystemExit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    report = build_deployment_decision(config)

    out_json = config.get("output", {}).get(
        "json", "outputs/reports/deployment_decision_report.json"
    )
    out_md = config.get("output", {}).get(
        "markdown", "outputs/reports/deployment_decision_report.md"
    )

    write_decision_json(report, out_json)
    write_decision_markdown(report, out_md)

    print("Deployment decision report written:")
    print(f"  JSON: {out_json}")
    print(f"  MD:   {out_md}")
    for rec in report.get("recommendations", []):
        print(f"  {rec['target']}: {rec['recommendation']}")
    if report.get("risks"):
        print(f"  Risks ({len(report['risks'])}):")
        for r in report["risks"]:
            print(f"    - {r}")


if __name__ == "__main__":
    main()
