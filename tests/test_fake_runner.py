"""Tests for the fake runtime backend."""

import pytest

from src.runtime.fake_runner import FakeRunner
from src.server.protocol import InferenceRequest


@pytest.fixture
def runner():
    r = FakeRunner()
    r.load()
    yield r
    r.close()


class TestFakeRunnerInputHeuristics:
    """Input-based prediction heuristics."""

    def test_danger_input_returns_danger(self, runner):
        req = InferenceRequest(backend="fake", input="samples/images/danger_scene.jpg")
        resp = runner.predict(req)
        assert resp.prediction is not None
        assert resp.prediction.class_name == "danger"
        assert resp.prediction.class_id == 2

    def test_safe_input_returns_safe(self, runner):
        req = InferenceRequest(backend="fake", input="samples/images/safe_scene.jpg")
        resp = runner.predict(req)
        assert resp.prediction is not None
        assert resp.prediction.class_name == "safe"
        assert resp.prediction.class_id == 0

    def test_ordinary_input_returns_warning(self, runner):
        req = InferenceRequest(backend="fake", input="samples/images/test.jpg")
        resp = runner.predict(req)
        assert resp.prediction is not None
        assert resp.prediction.class_name == "warning"
        assert resp.prediction.class_id == 1


class TestFakeRunnerResponseShape:
    """Response structure and required fields."""

    def test_status_is_ok(self, runner):
        req = InferenceRequest(backend="fake", input="test.jpg")
        resp = runner.predict(req)
        assert resp.status == "ok"

    def test_backend_is_fake(self, runner):
        req = InferenceRequest(backend="fake", input="test.jpg")
        resp = runner.predict(req)
        assert resp.backend == "fake"

    def test_latency_ms_total_exists(self, runner):
        req = InferenceRequest(backend="fake", input="test.jpg")
        resp = runner.predict(req)
        assert resp.latency_ms is not None
        assert resp.latency_ms.total > 0

    def test_confidence_is_reasonable(self, runner):
        req = InferenceRequest(backend="fake", input="danger.jpg")
        resp = runner.predict(req)
        assert resp.prediction is not None
        assert 0.0 <= resp.prediction.confidence <= 1.0
