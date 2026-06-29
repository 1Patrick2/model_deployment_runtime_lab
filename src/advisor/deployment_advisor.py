"""Rule-based Deployment Advisor — reads decision report and generates engineering explanations.

Usage
-----
.. code-block:: powershell

    python -m src.advisor.deployment_advisor --config configs/deployment_advisor.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_decision_report(path: str | Path) -> Optional[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Rule-based explanation builders ─────────────────────────────


def _explain_pc_cpu(summary: dict, recommendations: list) -> List[str]:
    lines: list[str] = []
    rec = summary.get("pc_cpu_serving", "")
    reason = "Stable ONNX Runtime path."
    for r in recommendations:
        if r.get("target") == "PC CPU serving":
            reason = r.get("reason", reason)

    if rec == "onnx_fp32":
        lines.append(
            "For local PC CPU serving, ONNX FP32 is the safest default path "
            "because it has a stable ONNX Runtime backend, simple deployment, "
            f"and low local latency.  ({reason})"
        )
    elif rec == "insufficient_evidence":
        lines.append("PC CPU serving recommendation is unavailable because FP32 benchmark data is missing.")
    else:
        lines.append(f"PC CPU serving: {rec}.")

    return lines


def _explain_size_sensitive(summary: dict, recommendations: list) -> List[str]:
    lines: list[str] = []
    rec = summary.get("size_sensitive_cpu", "")
    reason = "Smaller artifact."
    for r in recommendations:
        if r.get("target") == "Size-sensitive CPU deployment":
            reason = r.get("reason", reason)

    if rec == "qdq_int8":
        lines.append(
            "For size-sensitive deployment, QDQ INT8 is recommended because it "
            f"significantly reduces model size.  ({reason}) "
            "However, it currently uses dummy calibration, so real accuracy is not validated."
        )
    elif rec == "insufficient_evidence":
        lines.append("Size-sensitive CPU recommendation is unavailable because INT8 data is missing.")
    else:
        lines.append(f"Size-sensitive CPU deployment: {rec}.")

    return lines


def _explain_rk3588(summary: dict, recommendations: list) -> List[str]:
    lines: list[str] = []
    rec = summary.get("rk3588_status", "")
    reason = "RKNN artifact ready."
    for r in recommendations:
        if r.get("target") == "RK3588 deployment":
            reason = r.get("reason", reason)

    if rec == "artifact_ready_board_validation_pending":
        lines.append(
            f"For RK3588 deployment, the RKNN artifact is ready, but board-side "
            f"runtime validation is still pending.  ({reason})"
        )
    elif rec == "not_available":
        lines.append("RK3588 deployment recommendation is unavailable; no RKNN conversion report found.")
    else:
        lines.append(f"RK3588 deployment: {rec}.")

    return lines


def _explain_risks(risks: list) -> List[str]:
    if not risks:
        return []

    lines = [
        "",
        "## Key Risks",
        "",
    ]
    for risk in risks:
        lines.append(f"- {risk}")

    # Priority next step
    risk_text = " ".join(risks).lower()
    if "dummy calibration" in risk_text:
        lines.append("")
        lines.append(
            "**Priority next step:** Run INT8 calibration with representative real images, "
            "because the current INT8 decision is based on dummy calibration."
        )

    return lines


def _explain_next_steps(next_validation: list) -> List[str]:
    if not next_validation:
        return []
    result: list[str] = ["", "## Next Steps"]
    for i, ns in enumerate(next_validation, start=1):
        result.append(f"{i}. {ns}")
    return result


def _explain_missing_evidence(missing_evidence: list) -> List[str]:
    if not missing_evidence:
        return []
    lines: list[str] = [
        "",
        "## Missing Evidence",
        "",
        "Some evidence is missing, so recommendation confidence is limited.",
        "",
    ]
    for me in missing_evidence:
        lines.append(f"- {me}")
    lines.append("")
    return lines


def build_advisor_report(decision_report: Dict[str, Any]) -> str:
    """Generate a Markdown advisor report from the decision report dict."""
    summary = decision_report.get("summary", {})
    recommendations = decision_report.get("recommendations", [])
    risks = decision_report.get("risks", [])
    next_val = decision_report.get("next_validation", [])
    missing_evidence = decision_report.get("missing_evidence", [])

    lines: list[str] = [
        "# Deployment Advisor",
        "",
        "## Overall Recommendation",
        "",
    ]

    lines += _explain_pc_cpu(summary, recommendations)
    lines.append("")
    lines += _explain_size_sensitive(summary, recommendations)
    lines.append("")
    lines += _explain_rk3588(summary, recommendations)
    lines += _explain_risks(risks)
    lines += _explain_missing_evidence(missing_evidence)
    lines += _explain_next_steps(next_val)
    lines.append("")

    return "\n".join(lines)


def write_advisor_markdown(content: str, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def main() -> None:
    parser = argparse.ArgumentParser(description="Deployment advisor")
    parser.add_argument(
        "--config",
        default="configs/deployment_advisor.yaml",
        help="Path to deployment advisor YAML config",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}")
        raise SystemExit(1)

    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    report_path = config.get("input", {}).get("decision_report", "")
    data = load_decision_report(report_path)
    if data is None:
        print(f"ERROR: decision report not found: {report_path}")
        print("Hint: run deployment_decision.py first.")
        raise SystemExit(1)

    content = build_advisor_report(data)
    out_path = config.get("output", {}).get("markdown", "outputs/reports/deployment_advisor.md")
    write_advisor_markdown(content, out_path)

    print("Deployment advisor report written:")
    print(f"  {out_path}")
    print()
    # Print summary
    print(content)


if __name__ == "__main__":
    main()
