"""ONNX Runtime backend.

Loads a model manifest, creates an ``onnxruntime.InferenceSession``,
and executes real inference on either a dummy tensor or an image file.

Usage
-----
.. code-block:: powershell

    python -m src.runtime.onnx_runner --manifest models/manifests/mobilenetv3_small_onnx_fp32.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import onnxruntime

from src.models.manifest import load_manifest, resolve_artifact_path
from src.runtime.base_runner import BaseRunner
from src.server.protocol import (
    InferenceRequest,
    InferenceResponse,
    LatencyMs,
    Prediction,
)


# ── Helpers ──────────────────────────────────────────────────────


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _preprocess_image(image_path: str | Path, target_size: tuple[int, int],
                      mean: tuple[float, ...], std: tuple[float, ...]) -> np.ndarray:
    """Read an image file and return a preprocessed NCHW float32 tensor."""
    try:
        from PIL import Image  # defer import so runner is importable without PIL
    except ImportError:
        raise RuntimeError("Pillow is required for image_path input")

    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    # Normalise
    arr = (arr - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    # HWC → CHW + batch
    arr = np.transpose(arr, (2, 0, 1))[None, ...]
    return arr


# ── Runner ──────────────────────────────────────────────────────


class OnnxRunner(BaseRunner):
    """Runner that loads an ONNX model and executes inference."""

    backend_name: str = "onnx"

    def __init__(
        self,
        manifest_path: str | Path,
        project_root: str | Path | None = None,
        providers: list[str] | None = None,
    ) -> None:
        self._manifest_path = Path(manifest_path)
        self._project_root = Path(project_root) if project_root else Path.cwd()
        self._providers = providers or ["CPUExecutionProvider"]

        # Populated by load()
        self._session: Optional[onnxruntime.InferenceSession] = None
        self._manifest: Any = None
        self._input_name: str = ""
        self._output_name: str = ""

    # ── BaseRunner interface ──────────────────────────────────────

    def load(self) -> None:
        """Load manifest, resolve artifact, create inference session."""
        manifest = load_manifest(self._manifest_path)

        if manifest.backend != "onnx":
            raise ValueError(
                f"Manifest backend is '{manifest.backend}', expected 'onnx'"
            )

        artifact = resolve_artifact_path(self._project_root, manifest)
        if not artifact.exists():
            raise FileNotFoundError(
                f"ONNX artifact not found: {artifact}\n"
                f"Run the export script first:\n"
                f"  python -m src.export.export_onnx --model mobilenet_v3_small"
            )

        self._manifest = manifest
        self._session = onnxruntime.InferenceSession(
            str(artifact), providers=self._providers
        )
        self._input_name = manifest.input_name
        self._output_name = manifest.output_name

    def predict(self, request: InferenceRequest) -> InferenceResponse:
        """Run inference and return a response."""
        if self._session is None or self._manifest is None:
            raise RuntimeError(
                "OnnxRunner is not loaded. Call load() before predict()."
            )
        if request.backend != self.backend_name:
            raise ValueError(
                f"Request backend '{request.backend}' does not match "
                f"runner backend '{self.backend_name}'"
            )

        t0 = time.perf_counter()
        input_tensor = self._prepare_input(request)
        t_pre = time.perf_counter()

        outputs = self._session.run(
            [self._output_name],
            {self._input_name: input_tensor},
        )
        t_inf = time.perf_counter()

        pred = self._postprocess(outputs)
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
            backend="onnx",
            prediction=pred,
            latency_ms=latency,
            model_id=self._manifest.model_id,
            model_variant=self._manifest.model_variant,
        )

    def close(self) -> None:
        """Release the inference session."""
        self._session = None
        self._manifest = None

    # ── Internal helpers ──────────────────────────────────────────

    def _prepare_input(self, request: InferenceRequest) -> np.ndarray:
        """Produce the input tensor from the request."""
        if request.input_type == "dummy":
            shape = tuple(self._manifest.input_shape)
            return np.zeros(shape, dtype=np.float32)

        if request.input_type == "image_path":
            pre = self._manifest.preprocess
            if pre is None:
                raise ValueError(
                    "Manifest has no preprocess config; cannot handle image_path"
                )
            return _preprocess_image(
                request.input,
                target_size=pre.resize,
                mean=pre.mean,
                std=pre.std,
            )

        raise ValueError(f"Unsupported input_type: {request.input_type}")

    @staticmethod
    def _postprocess(outputs: list[np.ndarray]) -> Prediction:
        """Convert raw logits to a Prediction."""
        logits = outputs[0][0]  # shape (C,)
        probs = _softmax(logits)
        class_id = int(probs.argmax())
        confidence = float(probs[class_id])
        return Prediction(
            class_id=class_id,
            class_name=f"class_{class_id}",
            confidence=round(confidence, 4),
        )


# ── CLI entrypoint ──────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="ONNX Runtime runner smoke test")
    parser.add_argument(
        "--manifest",
        default="models/manifests/mobilenetv3_small_onnx_fp32.json",
        help="Path to model manifest JSON",
    )
    parser.add_argument("--input-type", default="dummy", help="dummy | image_path")
    parser.add_argument("--input", default="dummy", help="Input path or dummy marker")
    args = parser.parse_args()

    runner = OnnxRunner(manifest_path=args.manifest)
    try:
        runner.load()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    request = InferenceRequest(
        backend="onnx",
        input_type=args.input_type,
        input=args.input,
    )
    response = runner.predict(request)
    print(response.model_dump_json(indent=2))
    runner.close()


if __name__ == "__main__":
    main()
