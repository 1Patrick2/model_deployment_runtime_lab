"""Parse ``trtexec`` output into structured metrics."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional


def _extract_float(text: str, pattern: str) -> Optional[float]:
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def parse_trtexec_output(text: str) -> Dict[str, Any]:
    """Parse trtexec benchmark output text into a structured dict.

    Returns a dict with fields like ``throughput_qps``, ``latency_ms``,
    ``gpu_compute_time_ms``, etc.  Unparseable fields are set to ``None``.
    The raw output tail is preserved for debugging.
    """
    result: Dict[str, Any] = {}

    # Throughput (QPS) — multiple formats
    patterns = [
        r"Throughput.*?(\d+\.?\d*)\s*(?:qps|QPS|infer/s)",
        r"\[I\](\s+)Throughput.*?(\d+\.?\d*)",
        r"throughput.*?(\d+\.?\d*)",
    ]
    for pat in patterns:
        qps = _extract_float(text, pat)
        if qps is not None:
            result["throughput_qps"] = qps
            break

    # Latency — support many formats
    latency: Dict[str, Optional[float]] = {}
    latency_pats = [
        ("min", r"(?:min(?:imum)?|minimum)\s*[:=]?\s*(\d+\.?\d*)\s*ms"),
        ("max", r"(?:max(?:imum)?|maximum)\s*[:=]?\s*(\d+\.?\d*)\s*ms"),
        ("mean", r"(?:mean|average|avg)\s*[:=]?\s*(\d+\.?\d*)\s*ms"),
        ("median", r"(?:median|med)\s*[:=]?\s*(\d+\.?\d*)\s*ms"),
    ]
    for key, pat in latency_pats:
        v = _extract_float(text, pat)
        if v is not None:
            latency[key] = v
    if latency:
        result["latency_ms"] = latency

    # GPU Compute Time
    gct = _extract_float(
        text, r"GPU Compute Time.*?mean.*?(\d+\.?\d*)\s*ms"
    )
    if gct is not None:
        result["gpu_compute_time_ms"] = {"mean": gct}

    # Enqueue Time
    et = _extract_float(
        text, r"Enqueue Time.*?mean.*?(\d+\.?\d*)\s*ms"
    )
    if et is not None:
        result["enqueue_time_ms"] = {"mean": et}

    # H2D Latency
    h2d = _extract_float(text, r"H2D.*?mean.*?(\d+\.?\d*)\s*ms")
    if h2d is not None:
        result["h2d_latency_ms"] = {"mean": h2d}

    # D2H Latency
    d2h = _extract_float(text, r"D2H.*?mean.*?(\d+\.?\d*)\s*ms")
    if d2h is not None:
        result["d2h_latency_ms"] = {"mean": d2h}

    if not result:
        result["raw_output_tail"] = text[-500:]

    return result
