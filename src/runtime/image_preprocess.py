"""Image preprocessing for ONNX classification models."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

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

    For a more accurate ImageNet preprocessing pipeline (resize shorter
    side → center crop), use :func:`preprocess_image_imagenet` instead.
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


def preprocess_image_imagenet(
    image_path: str | Path,
    crop_size: Tuple[int, int] = (224, 224),
    resize_shorter: int = 256,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """ImageNet-standard preprocessing: resize shorter side → center crop → normalize.

    This matches the preprocessing used by torchvision ImageNet pretrained
    models and should be used for calibration and validation pipelines.
    """
    try:
        from PIL import Image
    except ImportError:
        raise RuntimeError("Pillow is required for image preprocessing")

    img = Image.open(image_path).convert("RGB")

    # Resize: shorter side to resize_shorter
    w, h = img.size
    ratio = resize_shorter / min(w, h)
    new_size = (int(w * ratio), int(h * ratio))
    img = img.resize(new_size, Image.BILINEAR)

    # Center crop
    new_w, new_h = new_size
    left = (new_w - crop_size[0]) // 2
    top = (new_h - crop_size[1]) // 2
    img = img.crop((left, top, left + crop_size[0], top + crop_size[1]))

    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    arr = np.transpose(arr, (2, 0, 1))[None, ...]  # HWC → NCHW
    return arr


__all__ = [
    "preprocess_image",
    "preprocess_image_imagenet",
]
