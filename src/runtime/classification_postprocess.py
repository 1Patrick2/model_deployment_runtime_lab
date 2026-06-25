"""Top-k classification post-processing."""

from __future__ import annotations

from typing import List

import numpy as np

from src.models.imagenet_labels import label_for_class_id


class TopKPrediction:
    """A single entry in the top-k list."""

    def __init__(self, class_id: int, class_name: str, score: float) -> None:
        self.class_id = class_id
        self.class_name = class_name
        self.score = score


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(logits - np.max(logits))
    return e / e.sum()


def postprocess_top_k(
    outputs: list[np.ndarray],
    k: int = 5,
) -> List[TopKPrediction]:
    """Convert ONNX output logits to a top-k prediction list.

    Args:
        outputs: Raw ONNX outputs (list of arrays, first is logits).
        k: Number of top predictions to return.

    Returns:
        List of TopKPrediction sorted by score descending.
    """
    logits = outputs[0][0]  # (C,)
    probs = softmax(logits)
    top_indices = np.argsort(probs)[::-1][:k]

    return [
        TopKPrediction(
            class_id=int(idx),
            class_name=label_for_class_id(int(idx)),
            score=round(float(probs[idx]), 4),
        )
        for idx in top_indices
    ]


def top_k_to_dicts(predictions: List[TopKPrediction]) -> list[dict]:
    """Convert TopKPrediction list to JSON-safe dicts."""
    return [
        {"class_id": p.class_id, "class_name": p.class_name, "score": p.score}
        for p in predictions
    ]


__all__ = [
    "TopKPrediction",
    "softmax",
    "postprocess_top_k",
    "top_k_to_dicts",
]
