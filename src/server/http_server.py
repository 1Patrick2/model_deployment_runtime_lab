"""Real image HTTP inference server.

Usage
-----
.. code-block:: powershell

    python -m src.server.http_server --backend onnx --manifest models/manifests/mobilenetv3_small_onnx_fp32.json

    curl.exe -X POST http://127.0.0.1:8000/infer ^
        -H "Content-Type: application/json" ^
        -d "{\"input_type\":\"dummy\",\"input\":\"dummy\",\"top_k\":5}"
"""

from __future__ import annotations

import argparse
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException

from src.runtime.classification_postprocess import postprocess_top_k
from src.runtime.image_preprocess import preprocess_image
from src.runtime.onnx_runner import OnnxRunner
from src.server.http_schema import (
    HealthResponse,
    InferRequest,
    InferResponse,
    LatencyInfo,
    MetadataResponse,
    PredictionItem,
)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Initialise runner on startup, clean up on shutdown."""
    global _runner, _manifest_info  # noqa: PLW0603
    if _runner is not None and _runner.manifest is not None:
        m = _runner.manifest
        _manifest_info = {
            "model_id": m.model_id,
            "model_variant": m.model_variant,
            "backend": m.backend,
            "task": m.task,
            "input_shape": m.input_shape,
            "input_dtype": m.input_dtype,
        }
    yield
    if _runner is not None:
        _runner.close()


app = FastAPI(
    title="Model Deployment Runtime Lab — HTTP Inference Server",
    lifespan=_lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness check."""
    return HealthResponse()


@app.get("/metadata", response_model=MetadataResponse)
def metadata() -> MetadataResponse:
    """Return model metadata from the loaded manifest."""
    if not _manifest_info:
        raise HTTPException(status_code=503, detail="Server not fully initialised")
    return MetadataResponse(**_manifest_info)


@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest) -> InferResponse:
    """Run inference on the input and return top-k predictions."""
    global _runner  # noqa: PLW0603
    if _runner is None:
        raise HTTPException(status_code=503, detail="Runner not loaded")

    t0 = time.perf_counter()

    if req.input_type == "image_path":
        input_tensor = preprocess_image(req.input, preprocess=_runner.manifest.preprocess)
    elif req.input_type == "dummy":
        input_tensor = np.zeros(tuple(_runner.manifest.input_shape), dtype=np.float32)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported input_type: {req.input_type}")
    t_pre = time.perf_counter()

    outputs = _runner._session.run(
        [_runner._output_name],
        {_runner._input_name: input_tensor},
    )
    t_inf = time.perf_counter()

    predictions = postprocess_top_k(outputs, k=req.top_k)
    t_post = time.perf_counter()

    latency = LatencyInfo(
        preprocess=round((t_pre - t0) * 1000, 2),
        inference=round((t_inf - t_pre) * 1000, 2),
        postprocess=round((t_post - t_inf) * 1000, 2),
        total=round((t_post - t0) * 1000, 2),
    )

    return InferResponse(
        status="ok",
        backend=_runner.backend_name,
        model_id=_runner.manifest.model_id,
        model_variant=_runner.manifest.model_variant,
        input_type=req.input_type,
        top_k=req.top_k,
        top_k_predictions=[
            PredictionItem(class_id=p.class_id, class_name=p.class_name, score=p.score)
            for p in predictions
        ],
        latency_ms=latency,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="HTTP inference server")
    parser.add_argument("--backend", default="onnx", help="Runtime backend")
    parser.add_argument(
        "--manifest",
        default="models/manifests/mobilenetv3_small_onnx_fp32.json",
        help="Path to model manifest JSON",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    args = parser.parse_args()

    global _runner  # noqa: PLW0603
    from src.server.zmq_server import create_runner
    _runner = create_runner(args.backend, manifest=args.manifest)

    print(f"HTTP server starting  (backend={args.backend})")
    print(f"  manifest: {args.manifest}")
    print(f"  listening on http://{args.host}:{args.port}")
    print(f"  health:   http://{args.host}:{args.port}/health")
    print(f"  metadata: http://{args.host}:{args.port}/metadata")
    print(f"  infer:    POST http://{args.host}:{args.port}/infer")
    print()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
