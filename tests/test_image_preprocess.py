"""Tests for image preprocessing."""

import numpy as np
import pytest

from src.runtime.image_preprocess import preprocess_image


class TestPreprocessImage:
    """Image preprocessing with a generated test image."""

    def test_preprocess_returns_correct_shape(self, tmp_path):
        # Create a small test image
        img_path = tmp_path / "test.png"
        from PIL import Image
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        img.save(img_path)

        tensor = preprocess_image(str(img_path), target_size=(224, 224))
        assert tensor.shape == (1, 3, 224, 224)
        assert tensor.dtype == np.float32

    def test_preprocess_accepts_jpg(self, tmp_path):
        img_path = tmp_path / "test.jpg"
        from PIL import Image
        img = Image.new("RGB", (300, 200), color=(64, 128, 192))
        img.save(img_path)

        tensor = preprocess_image(str(img_path), target_size=(224, 224))
        assert tensor.shape == (1, 3, 224, 224)

    def test_preprocess_normalization_range(self, tmp_path):
        img_path = tmp_path / "grey.png"
        from PIL import Image
        img = Image.new("RGB", (224, 224), color=(0, 0, 0))  # black
        img.save(img_path)

        tensor = preprocess_image(str(img_path), target_size=(224, 224))
        # Black pixels after normalisation should be negative (below mean)
        assert np.all(tensor < 0)

    def test_missing_file_raises_error(self):
        with pytest.raises((FileNotFoundError, RuntimeError)):
            preprocess_image("nonexistent.jpg", target_size=(224, 224))
