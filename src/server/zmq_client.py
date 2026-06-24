"""ZMQ REQ inference client.

Usage
-----
.. code-block:: powershell

    python -m src.server.zmq_client --input samples/images/test.jpg --backend fake
"""

from __future__ import annotations

import argparse
import json
import sys

import zmq

from src.server.protocol import InferenceRequest


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
    parser.add_argument("--timeout-ms", type=int, default=3000, help="Reply timeout")
    args = parser.parse_args()

    # Build request
    request = InferenceRequest(
        backend=args.backend,
        input=args.input,
    )
    payload = request.model_dump_json()

    address = f"tcp://{args.host}:{args.port}"
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, args.timeout_ms)
    socket.connect(address)

    print(f"Sending request to {address}  (backend={args.backend})")
    print(f"  input: {args.input}")
    print(f"  request_id: {request.request_id}")
    print()

    try:
        socket.send_string(payload)
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
