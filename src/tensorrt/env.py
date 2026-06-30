"""TensorRT environment detection — trtexec and GPU availability."""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any, Dict


def find_trtexec() -> str | None:
    """Locate ``trtexec`` in PATH or return ``None``."""
    return shutil.which("trtexec")


def get_trtexec_version() -> str | None:
    """Return ``trtexec`` version string, or ``None`` if not available."""
    exe = find_trtexec()
    if exe is None:
        return None
    try:
        result = subprocess.run(
            [exe, "--version"], capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip() or result.stderr.strip() or None
    except Exception:
        return None


def check_nvidia_smi() -> Dict[str, Any]:
    """Query GPU info via ``nvidia-smi``."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return {"available": False}
        parts = [p.strip() for p in result.stdout.strip().split(", ")]
        return {
            "available": True,
            "gpu_name": parts[0] if len(parts) > 0 else None,
            "memory_total": parts[1] if len(parts) > 1 else None,
            "driver_version": parts[2] if len(parts) > 2 else None,
        }
    except Exception:
        return {"available": False}


def collect_tensorrt_env() -> Dict[str, Any]:
    """Collect TensorRT environment info.

    Returns a structured dict.  All fields are safe for CPU-only systems.
    """
    trtexec = find_trtexec()
    version = get_trtexec_version() if trtexec else None
    gpu = check_nvidia_smi()

    report: Dict[str, Any] = {
        "trtexec_available": trtexec is not None,
        "trtexec_path": trtexec,
        "tensorrt_version": version,
    }
    if gpu.get("available"):
        report["nvidia_smi_available"] = True
        report["gpu_name"] = gpu.get("gpu_name")
        report["gpu_memory_total"] = gpu.get("memory_total")
        report["driver_version"] = gpu.get("driver_version")
    else:
        report["nvidia_smi_available"] = False
    report["status"] = "ok" if trtexec else "skipped"
    if not trtexec:
        report["reason"] = "trtexec not found in PATH"
    return report
