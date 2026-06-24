"""Fake runtime backend — returns deterministic canned predictions.

Useful for testing the serving pipeline (protocol → server → client)
without a real model.  Input-based heuristics:

- if ``input`` string contains ``"danger"`` → danger  (class_id=2)
- if ``input`` string contains ``"safe"``    → safe     (class_id=0)
- otherwise                                    → warning  (class_id=1)
"""

from __future__ import annotations

import time

from src.runtime.base_runner import BaseRunner
from src.server.protocol import (
    InferenceRequest,
    InferenceResponse,
    LatencyMs,
    Prediction,
)


class FakeRunner(BaseRunner):
    """Runner that returns deterministic predictions without a real model."""

    backend_name: str = "fake"
    model_id: str = "fake_classifier"
    model_variant: str = "fake_v1"

    def __init__(self) -> None:
        self._loaded = False

    # ── BaseRunner interface ──────────────────────────────────────

    def load(self) -> None:
        """Simulate model loading (no-op for fake runner)."""
        self._loaded = True

    def predict(self, request: InferenceRequest) -> InferenceResponse:
        """Produce a deterministic prediction based on input content."""
        t0 = time.perf_counter()

        # Simulate preprocess time
        time.sleep(0.001)
        t_pre = time.perf_counter()

        # Determine result based on input text
        inp = request.input.lower()
        if "danger" in inp:
            pred = Prediction(class_id=2, class_name="danger", confidence=0.99)
        elif "safe" in inp:
            pred = Prediction(class_id=0, class_name="safe", confidence=0.95)
        else:
            pred = Prediction(class_id=1, class_name="warning", confidence=0.87)

        # Simulate inference time
        time.sleep(0.002)
        t_inf = time.perf_counter()

        # Simulate postprocess time
        time.sleep(0.001)
        t_post = time.perf_counter()

        latency = LatencyMs(
            preprocess=round((t_pre - t0) * 1000, 2),
            inference=round((t_inf - t_pre) * 1000, 2),
            postprocess=round((t_post - t_inf) * 1000, 2),
            total=round((t_post - t0) * 1000, 2),
        )

        return InferenceResponse(
            request_id=request.request_id,
            status="ok",
            backend="fake",
            prediction=pred,
            latency_ms=latency,
            model_id=self.model_id,
            model_variant=self.model_variant,
        )

    def close(self) -> None:
        """Simulate resource cleanup (no-op for fake runner)."""
        self._loaded = False


__all__ = ["FakeRunner"]
