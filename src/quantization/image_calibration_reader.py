"""Real image calibration data reader for static QDQ quantization.

Reads representative images from a directory and yields preprocessed
tensors suitable for ONNX Runtime static calibration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, Optional

import numpy as np
from onnxruntime.quantization import CalibrationDataReader


def _preprocess_for_calibration(
    image_path: Path,
    target_size: tuple[int, int] = (224, 224),
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    mode: str = "simple",
) -> np.ndarray:
    """Load and preprocess a single image for calibration.

    Args:
        mode: "simple" — direct resize to *target_size*.
              "imagenet" — resize shorter side to 256, center crop 224.

    Returns an NCHW float32 tensor.
    """
    from PIL import Image

    img = Image.open(image_path).convert("RGB")

    if mode == "imagenet":
        # Resize shorter side to 256
        w, h = img.size
        ratio = 256 / min(w, h)
        new_size = (int(w * ratio), int(h * ratio))
        img = img.resize(new_size, Image.BILINEAR)
        # Center crop 224
        new_w, new_h = new_size
        left = (new_w - target_size[0]) // 2
        top = (new_h - target_size[1]) // 2
        img = img.crop((left, top, left + target_size[0], top + target_size[1]))
    else:
        img = img.resize(target_size)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    arr = np.transpose(arr, (2, 0, 1))[None, ...]  # HWC → NCHW
    return arr


class ImageCalibrationDataReader(CalibrationDataReader):
    """Yields preprocessed real images for static QDQ calibration.

    Args:
        image_dir: Directory containing JPEG/PNG images.
        input_name: ONNX model input name.
        target_size: (height, width) for resize/crop.
        mean: ImageNet mean for normalization.
        std: ImageNet std for normalization.
        max_samples: Maximum number of images to use.
        preprocess_mode: ``"simple"`` (direct resize) or ``"imagenet"``
            (resize shorter side → center crop).

    Raises:
        FileNotFoundError: If *image_dir* does not exist.
        ValueError: If *image_dir* contains no supported images.
    """

    def __init__(
        self,
        image_dir: str | Path,
        input_name: str,
        target_size: tuple[int, int] = (224, 224),
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        max_samples: int = 50,
        preprocess_mode: str = "simple",
    ) -> None:
        self._input_name = input_name
        self._target_size = target_size
        self._mean = mean
        self._std = std
        self._preprocess_mode = preprocess_mode

        image_dir = Path(image_dir)
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Calibration image directory not found: {image_dir}")

        # Collect supported images
        exts = {".jpg", ".jpeg", ".png"}
        self._image_paths: list[Path] = sorted(
            p for p in image_dir.iterdir() if p.suffix.lower() in exts
        )
        if not self._image_paths:
            raise ValueError(
                f"No supported images (jpg/jpeg/png) found in {image_dir}"
            )

        if max_samples and len(self._image_paths) > max_samples:
            self._image_paths = self._image_paths[:max_samples]

        self._iterator: Iterator[Path] = iter(self._image_paths)

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        """Return the next preprocessed image tensor, or ``None`` when exhausted."""
        try:
            path = next(self._iterator)
        except StopIteration:
            return None

        tensor = _preprocess_for_calibration(
            path,
            target_size=self._target_size,
            mean=self._mean,
            std=self._std,
            mode=self._preprocess_mode,
        )
        return {self._input_name: tensor}


__all__ = [
    "ImageCalibrationDataReader",
    "_preprocess_for_calibration",
]
