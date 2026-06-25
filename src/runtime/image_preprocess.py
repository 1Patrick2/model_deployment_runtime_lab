"""Image preprocessing for ONNX classification models."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from src.models.manifest import PreprocessConfig


def preprocess_image(
    image_path: str | Path,
    preprocess: Optional[PreprocessConfig] = None,
    target_size: tuple[int, int] = (224, 224),
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Read an image file and return a preprocessed NCHW float32 tensor.

    Uses ``preprocess`` config when given; otherwise falls back to
    the standard ImageNet parameters.
    """
    if preprocess is not None:
        target_size = preprocess.resize
        mean = preprocess.mean
        std = preprocess.std

    try:
        from PIL import Image
    except ImportError:
        raise RuntimeError("Pillow is required for image preprocessing")

    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    arr = np.transpose(arr, (2, 0, 1))[None, ...]  # HWC → NCHW
    return arr


__all__ = ["preprocess_image"]
