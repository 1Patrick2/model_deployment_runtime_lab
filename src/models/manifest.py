"""Model manifest — schema and loading utilities.

A manifest describes one model artifact (e.g. an ONNX file) together
with its preprocessing parameters so that runners and benchmarks can
work with it without hard-coded paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

from pydantic import BaseModel


class PreprocessConfig(BaseModel):
    """Image preprocessing parameters expected by the model."""

    resize: Tuple[int, int]
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]


class ModelManifest(BaseModel):
    """Metadata for a single model artifact."""

    model_id: str
    model_variant: str
    backend: str  # e.g. "onnx", "torch", "rknn"
    task: str  # e.g. "classification"
    artifact_path: str
    input_name: str
    output_name: str
    input_shape: List[int]
    input_dtype: str = "float32"
    class_names_path: Optional[str] = None
    preprocess: Optional[PreprocessConfig] = None


# ── Loading ──────────────────────────────────────────────────────


def load_manifest(path: str | Path) -> ModelManifest:
    """Load a ``ModelManifest`` from a JSON file.

    Raises ``ValidationError`` if the file content does not match the
    schema.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ModelManifest(**data)


def resolve_artifact_path(project_root: Path, manifest: ModelManifest) -> Path:
    """Resolve the model artifact path to an absolute ``Path``.

    Relative paths are treated as relative to ``project_root``.
    """
    p = Path(manifest.artifact_path)
    if p.is_absolute():
        return p
    return project_root / p


__all__ = [
    "ModelManifest",
    "PreprocessConfig",
    "load_manifest",
    "resolve_artifact_path",
]
