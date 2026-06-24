"""ZMQ REQ/REP inference server.

Usage
-----
.. code-block:: powershell

    python -m src.server.zmq_server --backend fake --host 127.0.0.1 --port 5555
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

import zmq

from src.runtime.base_runner import BaseRunner
from src.runtime.fake_runner import FakeRunner
from src.server.protocol import (
    INVALID_REQUEST,
    UNSUPPORTED_BACKEND,
    InferenceRequest,
    RUNTIME_ERROR,
    make_error_response,
)

# ── Registry of supported backends ──────────────────────────────
# Later backends (onnx, torch, rknn) register themselves here.

_BACKENDS: dict[str, type[BaseRunner]] = {
    "fake": FakeRunner,
}


def register_backend(name: str, runner_cls: type[BaseRunner]) -> None:
    """Register a new backend so the server can dispatch to it."""
    _BACKENDS[name] = runner_cls


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


def _resolve_backend(backend_name: str) -> BaseRunner:
    """Look up the runner class, instantiate it, and call ``load()``."""
    cls = _BACKENDS.get(backend_name)
    if cls is None:
        supported = ", ".join(sorted(_BACKENDS))
        msg = f"unsupported backend '{backend_name}'; choose from {{{supported}}}"
        print(f"ERROR: {msg}", file=sys.stderr)
        sys.exit(1)
    runner = cls()
    runner.load()
    return runner


def main() -> None:
    parser = argparse.ArgumentParser(description="ZMQ inference server")
    parser.add_argument("--backend", default="fake", help="Runtime backend")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address")
    parser.add_argument("--port", type=int, default=5555, help="Bind port")
    args = parser.parse_args()

    runner = _resolve_backend(args.backend)
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
