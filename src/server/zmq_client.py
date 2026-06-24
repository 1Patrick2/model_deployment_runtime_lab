"""ZMQ REQ inference client.

Usage
-----
.. code-block:: powershell

    python -m src.server.zmq_client --input samples/images/test.jpg --backend fake
    python -m src.server.zmq_client --backend onnx --input-type dummy --input dummy
"""

from __future__ import annotations

import argparse
import json
import sys

import zmq

from src.server.protocol import InferenceRequest


def build_request_payload(
    backend: str,
    input_value: str,
    input_type: str = "image_path",
) -> dict:
    """Build an ``InferenceRequest`` payload dict.

    Extracted for unit testing without needing a ZMQ socket.
    """
    req = InferenceRequest(
        backend=backend,
        input_type=input_type,
        input=input_value,
    )
    return req.model_dump()


def main() -> None:
    parser = argparse.ArgumentParser(description="ZMQ inference client")
    parser.add_argument("--host", default="127.0.0.1", help="Server address")
    parser.add_argument("--port", type=int, default=5555, help="Server port")
    parser.add_argument(
        "--input",
        default="samples/images/test.jpg",
        help="Input path sent to the server",
    )
    parser.add_argument("--backend", default="fake", help="Backend to request")
    parser.add_argument(
        "--input-type",
        default="image_path",
        help="Input type: image_path, dummy, raw_text",
    )
    parser.add_argument("--timeout-ms", type=int, default=3000, help="Reply timeout")
    args = parser.parse_args()

    # Build request
    payload = build_request_payload(
        backend=args.backend,
        input_value=args.input,
        input_type=args.input_type,
    )
    payload_json = json.dumps(payload)

    address = f"tcp://{args.host}:{args.port}"
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, args.timeout_ms)
    socket.connect(address)

    print(f"Sending request to {address}  (backend={args.backend})")
    print(f"  input_type: {args.input_type}")
    print(f"  input: {args.input}")
    print(f"  request_id: {payload['request_id']}")
    print()

    try:
        socket.send_string(payload_json)
        reply = socket.recv_json()
    except zmq.error.Again:
        print(f"ERROR: timeout after {args.timeout_ms} ms — server did not reply")
        sys.exit(1)
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    finally:
        socket.close()
        context.term()

    print("Response:")
    print(json.dumps(reply, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
