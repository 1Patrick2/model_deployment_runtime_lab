"""Metrics for comparing FP32 and INT8 model outputs."""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


def top_k_indices(logits: np.ndarray, k: int = 5) -> List[int]:
    """Return indices of top-k classes from logits."""
    probs = softmax(logits)
    return list(np.argsort(probs)[::-1][:k])


def top1_consistency(
    fp32_logits: List[np.ndarray],
    int8_logits: List[np.ndarray],
) -> float:
    """Fraction of images where FP32 and INT8 agree on the top-1 class."""
    matches = 0
    n = len(fp32_logits)
    for f, i in zip(fp32_logits, int8_logits):
        if top_k_indices(f, 1)[0] == top_k_indices(i, 1)[0]:
            matches += 1
    return matches / n if n > 0 else 0.0


def top5_consistency(
    fp32_logits: List[np.ndarray],
    int8_logits: List[np.ndarray],
) -> float:
    """Fraction of images where the FP32 top-1 falls within INT8 top-5."""
    matches = 0
    n = len(fp32_logits)
    for f, i in zip(fp32_logits, int8_logits):
        fp32_top1 = top_k_indices(f, 1)[0]
        int8_top5_set = set(top_k_indices(i, 5))
        if fp32_top1 in int8_top5_set:
            matches += 1
    return matches / n if n > 0 else 0.0


def mean_top5_overlap(
    fp32_logits: List[np.ndarray],
    int8_logits: List[np.ndarray],
) -> float:
    """Average Jaccard-like overlap of FP32 and INT8 top-5 sets."""
    overlaps: list[float] = []
    for f, i in zip(fp32_logits, int8_logits):
        fp32_set = set(top_k_indices(f, 5))
        int8_set = set(top_k_indices(i, 5))
        if not fp32_set and not int8_set:
            overlaps.append(1.0)
        else:
            overlap = len(fp32_set & int8_set) / len(fp32_set | int8_set)
            overlaps.append(overlap)
    return float(np.mean(overlaps)) if overlaps else 0.0


def mean_confidence_drift(
    fp32_logits: List[np.ndarray],
    int8_logits: List[np.ndarray],
) -> float:
    """Average absolute confidence difference across all images."""
    drifts: list[float] = []
    for f, i in zip(fp32_logits, int8_logits):
        fp32_probs = softmax(f)
        int8_probs = softmax(i)
        drift = float(np.mean(np.abs(fp32_probs - int8_probs)))
        drifts.append(drift)
    return float(np.mean(drifts)) if drifts else 0.0


def max_confidence_drift(
    fp32_logits: List[np.ndarray],
    int8_logits: List[np.ndarray],
) -> float:
    """Maximum per-class confidence drift across all images."""
    max_drift = 0.0
    for f, i in zip(fp32_logits, int8_logits):
        fp32_probs = softmax(f)
        int8_probs = softmax(i)
        drift = float(np.max(np.abs(fp32_probs - int8_probs)))
        max_drift = max(max_drift, drift)
    return max_drift


def mean_logits_cosine_similarity(
    fp32_logits: List[np.ndarray],
    int8_logits: List[np.ndarray],
) -> float:
    """Average cosine similarity between FP32 and INT8 logits."""
    similarities: list[float] = []
    for f, i in zip(fp32_logits, int8_logits):
        f_norm = f / (np.linalg.norm(f) + 1e-10)
        i_norm = i / (np.linalg.norm(i) + 1e-10)
        similarities.append(float(np.dot(f_norm, i_norm)))
    return float(np.mean(similarities)) if similarities else 0.0


def mean_absolute_error(
    fp32_logits: List[np.ndarray],
    int8_logits: List[np.ndarray],
) -> float:
    """Mean absolute error between FP32 and INT8 logits."""
    errors: list[float] = []
    for f, i in zip(fp32_logits, int8_logits):
        errors.append(float(np.mean(np.abs(f - i))))
    return float(np.mean(errors)) if errors else 0.0


def max_absolute_error(
    fp32_logits: List[np.ndarray],
    int8_logits: List[np.ndarray],
) -> float:
    """Maximum per-element absolute error between FP32 and INT8 logits."""
    max_err = 0.0
    for f, i in zip(fp32_logits, int8_logits):
        max_err = max(max_err, float(np.max(np.abs(f - i))))
    return max_err


__all__ = [
    "softmax",
    "top_k_indices",
    "top1_consistency",
    "top5_consistency",
    "mean_top5_overlap",
    "mean_confidence_drift",
    "max_confidence_drift",
    "mean_logits_cosine_similarity",
    "mean_absolute_error",
    "max_absolute_error",
]
