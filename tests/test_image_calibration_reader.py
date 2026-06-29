"""Tests for real image calibration data reader."""

import numpy as np
import pytest

from src.quantization.image_calibration_reader import ImageCalibrationDataReader
from src.runtime.image_preprocess import preprocess_image_imagenet


class TestPreprocessForCalibration:
    """Image preprocessing without full ONNX pipeline."""

    def test_preprocess_imagenet_returns_correct_shape(self, tmp_path):
        img_path = tmp_path / "test.png"
        from PIL import Image
        Image.new("RGB", (224, 224), color=(128, 128, 128)).save(img_path)

        tensor = preprocess_image_imagenet(str(img_path))
        assert tensor.shape == (1, 3, 224, 224)
        assert tensor.dtype == np.float32

    def test_preprocess_imagenet_empty_image_dir(self, tmp_path):
        with pytest.raises((ValueError, FileNotFoundError)):
            ImageCalibrationDataReader(
                image_dir=tmp_path / "nonexistent",
                input_name="input",
            )


class TestImageCalibrationDataReader:
    """Calibration reader with real image fixtures."""

    def test_reader_yields_correct_tensor(self, tmp_path):
        img_dir = tmp_path / "calib"
        img_dir.mkdir()
        from PIL import Image
        Image.new("RGB", (224, 224), color=(64, 128, 192)).save(img_dir / "img1.jpg")
        Image.new("RGB", (224, 224), color=(192, 128, 64)).save(img_dir / "img2.png")

        reader = ImageCalibrationDataReader(
            image_dir=img_dir,
            input_name="input",
            max_samples=10,
        )

        data = reader.get_next()
        assert data is not None
        assert "input" in data
        assert data["input"].shape == (1, 3, 224, 224)

        data2 = reader.get_next()
        assert data2 is not None

        data3 = reader.get_next()
        assert data3 is None  # exhausted

    def test_max_samples_limits_images(self, tmp_path):
        img_dir = tmp_path / "calib2"
        img_dir.mkdir()
        from PIL import Image
        for i in range(5):
            Image.new("RGB", (224, 224), color=(i * 50, 0, 0)).save(
                img_dir / f"img{i}.jpg"
            )

        reader = ImageCalibrationDataReader(
            image_dir=img_dir,
            input_name="input",
            max_samples=3,
        )
        count = 0
        while reader.get_next() is not None:
            count += 1
        assert count == 3

    def test_no_images_raises_error(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match="No supported images"):
            ImageCalibrationDataReader(
                image_dir=empty_dir,
                input_name="input",
            )
