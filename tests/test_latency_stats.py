"""Tests for benchmark statistics computation."""

from src.benchmark.report import compute_stats


class TestComputeStats:
    """Statistic calculations — no ONNX artifact needed."""

    def test_known_values(self):
        stats = compute_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        assert stats["mean"] == 3.0
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["p50"] == 3.0

    def test_p95_is_reasonable(self):
        values = list(range(1, 101))  # 1..100
        stats = compute_stats([float(v) for v in values])
        assert 90 <= stats["p95"] <= 100

    def test_empty_list_returns_zeros(self):
        stats = compute_stats([])
        assert stats["mean"] == 0.0
        assert stats["min"] == 0.0
        assert stats["max"] == 0.0

    def test_single_value(self):
        stats = compute_stats([42.0])
        assert stats["mean"] == 42.0
        assert stats["min"] == 42.0
        assert stats["max"] == 42.0
