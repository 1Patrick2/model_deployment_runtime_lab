"""Real image calibration data reader for static QDQ quantization.

Reads representative images from a directory and yields preprocessed
tensors suitable for ONNX Runtime static calibration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, Optional

import numpy as np
from onnxruntime.quantization import CalibrationDataReader

from src.runtime.image_preprocess import preprocess_image, preprocess_image_imagenet


class ImageCalibrationDataReader(CalibrationDataReader):
    """Yields preprocessed real images for static QDQ calibration.

    Args:
        image_dir: Directory containing JPEG/PNG images.
        input_name: ONNX model input name.
        target_size: (height, width) for resize/crop.
        mean: ImageNet mean for normalization.
        std: ImageNet std for normalization.
        max_samples: Maximum number of images to use.
        preprocess_mode: ``"simple"`` (direct resize) or ``"imagenet_v1"``
            (resize shorter side → center crop).
        resize_shorter: Short side length for ``"imagenet_v1"`` mode.

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
        resize_shorter: int = 256,
    ) -> None:
        self._input_name = input_name
        self._target_size = target_size
        self._mean = mean
        self._std = std
        self._preprocess_mode = preprocess_mode
        self._resize_shorter = resize_shorter

        image_dir = Path(image_dir)
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Calibration image directory not found: {image_dir}")

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

        if self._preprocess_mode in ("imagenet", "imagenet_v1"):
            tensor = preprocess_image_imagenet(
                str(path),
                crop_size=self._target_size,
                resize_shorter=self._resize_shorter,
                mean=self._mean,
                std=self._std,
            )
        else:
            tensor = preprocess_image(
                str(path),
                target_size=self._target_size,
                mean=self._mean,
                std=self._std,
            )
        return {self._input_name: tensor}


__all__ = [
    "ImageCalibrationDataReader",
]
