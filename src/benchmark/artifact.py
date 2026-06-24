"""Artifact size helpers for benchmark reports.

Handles ONNX models with external data (``.onnx.data``) so that
reported sizes reflect the total disk footprint.
"""

from __future__ import annotations

from pathlib import Path

import onnx


def compute_onnx_artifact_size_mb(onnx_path: str | Path) -> float:
    """Return total ONNX artifact size in MB, including external data files.

    Reads the ONNX model header to discover external data locations
    and sums their sizes together with the main model file.
    """
    path = Path(onnx_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"ONNX artifact not found: {path}")

    total = path.stat().st_size
    seen = {path}

    model = onnx.load(str(path), load_external_data=False)

    for initializer in model.graph.initializer:
        for entry in initializer.external_data:
            if entry.key != "location":
                continue
            ext_path = (path.parent / entry.value).resolve()
            if ext_path.exists() and ext_path not in seen:
                total += ext_path.stat().st_size
                seen.add(ext_path)

    return total / (1024 * 1024)


__all__ = [
    "compute_onnx_artifact_size_mb",
]
