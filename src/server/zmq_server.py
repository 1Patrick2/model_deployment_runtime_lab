"""ZMQ REQ/REP inference server.

Usage
-----
.. code-block:: powershell

    python -m src.server.zmq_server --backend fake --host 127.0.0.1 --port 5555
    python -m src.server.zmq_server --backend onnx --manifest models/manifests/mobilenetv3_small_onnx_fp32.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import zmq

from src.runtime.base_runner import BaseRunner
from src.runtime.fake_runner import FakeRunner
from src.runtime.onnx_runner import OnnxRunner
from src.server.protocol import (
    INVALID_REQUEST,
    UNSUPPORTED_BACKEND,
    InferenceRequest,
    RUNTIME_ERROR,
    make_error_response,
)


# ── Runner factory ──────────────────────────────────────────────


def create_runner(backend_name: str, manifest: str | None = None) -> BaseRunner:
    """Create and load the appropriate runner for *backend_name*.

    Raises ``ValueError`` when the backend is unsupported or required
    arguments are missing.
    """
    if backend_name == "fake":
        runner = FakeRunner()
    elif backend_name == "onnx":
        if manifest is None:
            raise ValueError("ONNX backend requires --manifest")
        runner = OnnxRunner(manifest_path=manifest, project_root=Path.cwd())
    else:
        supported = "fake, onnx"
        raise ValueError(
            f"unsupported backend '{backend_name}'; choose from {{{supported}}}"
        )
    runner.load()
    return runner


# ── Request handler (testable without a real ZMQ socket) ────────


def handle_request_json(payload: dict, runner: BaseRunner) -> dict:
    """Parse a JSON payload, run inference, return a JSON-safe dict.

    This function is the core of the server; it is deliberately kept
    free of ZMQ so that it can be unit-tested directly.
    """
    try:
        request = InferenceRequest(**payload)
    except Exception as exc:
        err = make_error_response(
            request_id=payload.get("request_id", "unknown"),
            backend=payload.get("backend", "unknown"),
            error_type=INVALID_REQUEST,
            message=str(exc),
        )
        return err.model_dump()

    # Validate that the requested backend matches the running runner.
    if request.backend != runner.backend_name:
        err = make_error_response(
            request_id=request.request_id,
            backend=request.backend,
            error_type=UNSUPPORTED_BACKEND,
            message=(
                f"backend '{request.backend}' is not served by "
                f"current runner '{runner.backend_name}'"
            ),
        )
        return err.model_dump()

    try:
        response = runner.predict(request)
    except Exception as exc:
        err = make_error_response(
            request_id=request.request_id,
            backend=request.backend,
            error_type=RUNTIME_ERROR,
            message=str(exc),
        )
        return err.model_dump()

    return response.model_dump()


# ── Server entrypoint ────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="ZMQ inference server")
    parser.add_argument("--backend", default="fake", help="Runtime backend")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address")
    parser.add_argument("--port", type=int, default=5555, help="Bind port")
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to model manifest (required for backend=onnx)",
    )
    args = parser.parse_args()

    try:
        runner = create_runner(args.backend, manifest=args.manifest)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    address = f"tcp://{args.host}:{args.port}"

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(address)

    print(f"ZMQ server listening on {address}  (backend={args.backend})")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            raw = socket.recv()
            payload = json.loads(raw)
            result = handle_request_json(payload, runner)
            socket.send_json(result)
    except KeyboardInterrupt:
        print("\nShutting down ...")
    finally:
        runner.close()
        socket.close()
        context.term()


if __name__ == "__main__":
    main()
