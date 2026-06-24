"""Model registry — lookup manifests by model_id / backend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class ModelRegistry:
    """Lookup table mapping ``(model_id, backend)`` to a manifest path."""

    def __init__(self, entries: List[Dict[str, Any]]) -> None:
        self._entries = entries

    def find_manifest_by_model_id(
        self,
        model_id: str,
        backend: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Find the first registry entry matching ``model_id``.

        If ``backend`` is given, only entries with that backend are
        considered.
        """
        for entry in self._entries:
            if entry.get("model_id") != model_id:
                continue
            if backend is not None and entry.get("backend") != backend:
                continue
            return entry
        return None

    def list_models(self, backend: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all entries, optionally filtered by ``backend``."""
        if backend is None:
            return list(self._entries)
        return [e for e in self._entries if e.get("backend") == backend]


def load_registry(path: str | Path) -> ModelRegistry:
    """Load a ``ModelRegistry`` from a JSON file.

    The file must contain a JSON array of registry entries.  Each entry
    should have at least ``model_id`` and ``backend`` keys, plus a
    ``manifest_path`` pointing to the actual manifest file.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        entries = data.get("models", [])
    elif isinstance(data, list):
        entries = data
    else:
        entries = []
    return ModelRegistry(entries)


def resolve_manifest_path(project_root: Path, registry_entry: dict) -> Path:
    """Resolve the ``manifest_path`` from a registry entry to an absolute path.

    Relative paths are treated as relative to ``project_root``.
    The entry must contain a ``manifest_path`` key.
    """
    p = Path(registry_entry["manifest_path"])
    if p.is_absolute():
        return p
    return project_root / p


__all__ = [
    "ModelRegistry",
    "load_registry",
    "resolve_manifest_path",
]
