"""Abstract runner interface for all inference backends.

Every backend (fake, onnx, torch, rknn, …) implements this interface so
that the server layer can dispatch requests uniformly without knowing
backend details.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.server.protocol import InferenceRequest, InferenceResponse


class BaseRunner(ABC):
    """Uniform interface that every runtime backend must implement."""

    backend_name: str = ""

    @abstractmethod
    def load(self) -> None:
        """Load model artifacts into memory."""

    @abstractmethod
    def predict(self, request: InferenceRequest) -> InferenceResponse:
        """Run inference and return a response."""

    @abstractmethod
    def close(self) -> None:
        """Release resources (model, device memory, …)."""


__all__ = ["BaseRunner"]
