"""Tests for classification top-k post-processing."""

import numpy as np

from src.runtime.classification_postprocess import (
    postprocess_top_k,
    softmax,
    top_k_to_dicts,
)
from src.models.imagenet_labels import label_for_class_id


class TestSoftmax:
    def test_softmax_sums_to_one(self):
        logits = np.array([2.0, 1.0, 0.1], dtype=np.float32)
        probs = softmax(logits)
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_softmax_is_stable_large_values(self):
        logits = np.array([1000, 1010, 990], dtype=np.float32)
        probs = softmax(logits)
        assert abs(probs.sum() - 1.0) < 1e-5


class TestPostprocessTopK:
    def test_returns_correct_number(self):
        logits = np.random.randn(1, 1000).astype(np.float32)
        predictions = postprocess_top_k([logits], k=5)
        assert len(predictions) == 5

    def test_scores_are_between_0_and_1(self):
        logits = np.random.randn(1, 1000).astype(np.float32)
        predictions = postprocess_top_k([logits], k=3)
        for p in predictions:
            assert 0.0 <= p.score <= 1.0

    def test_top_prediction_has_highest_score(self):
        logits = np.zeros((1, 1000), dtype=np.float32)
        logits[0, 42] = 10.0  # class 42 is dominant
        predictions = postprocess_top_k([logits], k=3)
        assert predictions[0].class_id == 42
        assert predictions[0].score > predictions[1].score


class TestLabelForClassId:
    def test_known_class_returns_name(self):
        name = label_for_class_id(281)
        assert name == "tiger cat"

    def test_unknown_class_returns_fallback(self):
        name = label_for_class_id(9999)
        assert "class_" in name
