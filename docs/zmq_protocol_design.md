# ZMQ Inference Protocol Design

## Overview

Stage 1 of Model Deployment Runtime Lab establishes a **language-agnostic,
schema-versioned** inference protocol over ZMQ REQ/REP sockets.  Every
backend (fake, onnx, torch, rknn) speaks the same protocol so that
clients do not need to know backend internals.

## Transport

| Layer | Choice |
|-------|--------|
| Socket type | ZMQ REQ/REP (synchronous request-reply) |
| Serialization | JSON (UTF-8) |
| Addressing | `tcp://HOST:PORT` (default `127.0.0.1:5555`) |

## Schema Versions

- Request: `inference_request.v1`
- Response: `inference_response.v1`

The `schema_version` field is reserved for future wire-format evolution.

---

## Request (`InferenceRequest`)

```json
{
  "request_id": "demo-001",
  "schema_version": "inference_request.v1",
  "backend": "fake",
  "model_id": "fake_classifier",
  "model_variant": "fake_v1",
  "input_type": "image_path",
  "input": "samples/images/test.jpg",
  "return_debug": false
}
```

### Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `request_id` | string | yes | auto UUID | Unique request identifier echoed in response |
| `schema_version` | string | no | `inference_request.v1` | Protocol version marker |
| `backend` | string | **yes** | — | Backend to dispatch to: `fake`, `onnx`, `torch`, `rknn` |
| `model_id` | string | no | `fake_classifier` | Logical model identifier |
| `model_variant` | string | no | `fake_v1` | Model version / tag |
| `input_type` | string | no | `image_path` | How to interpret `input` (`image_path`, `raw_text`, `tensor_meta`) |
| `input` | string | **yes** | — | Input data (path, text, or serialised metadata) |
| `return_debug` | bool | no | `false` | When true, backends may include extra debug fields |

---

## Response (`InferenceResponse`)

### Success

```json
{
  "request_id": "demo-001",
  "schema_version": "inference_response.v1",
  "status": "ok",
  "prediction": {
    "class_id": 2,
    "class_name": "danger",
    "confidence": 0.99
  },
  "latency_ms": {
    "preprocess": 1.0,
    "inference": 2.0,
    "postprocess": 1.0,
    "total": 4.0
  },
  "backend": "fake",
  "model_id": "fake_classifier",
  "model_variant": "fake_v1",
  "error_type": null,
  "message": null
}
```

### Error

```json
{
  "request_id": "demo-001",
  "schema_version": "inference_response.v1",
  "status": "error",
  "prediction": null,
  "latency_ms": null,
  "backend": "unknown",
  "model_id": null,
  "model_variant": null,
  "error_type": "UNSUPPORTED_BACKEND",
  "message": "backend 'abc' is not supported"
}
```

### Response Fields

| Field | Type | Present on | Description |
|-------|------|------------|-------------|
| `request_id` | string | always | Mirrors the request identifier |
| `schema_version` | string | always | `inference_response.v1` |
| `status` | string | always | `"ok"` or `"error"` |
| `prediction` | object or null | success | `{class_id, class_name, confidence}` |
| `latency_ms` | object or null | success | `{preprocess, inference, postprocess, total}` |
| `backend` | string | always | Backend that handled the request |
| `model_id` | string or null | success | Model that produced the prediction |
| `model_variant` | string or null | success | Model version |
| `error_type` | string or null | error | Machine-readable error category |
| `message` | string or null | error | Human-readable error description |

### Error Types

| Constant | Meaning |
|----------|---------|
| `INVALID_REQUEST` | Request JSON failed validation (missing required fields, wrong types) |
| `UNSUPPORTED_BACKEND` | The requested `backend` is not registered |
| `MODEL_NOT_FOUND` | The `model_id` / `model_variant` could not be loaded |
| `RUNTIME_ERROR` | An exception occurred during inference |
| `TIMEOUT` | The request exceeded the configured timeout |

---

## Fake Runner Heuristics

The `fake` runner simulates inference without a real model:

| Input contains | Prediction |
|----------------|-----------|
| `"danger"` | `class_id=2`, `class_name="danger"`, `confidence=0.99` |
| `"safe"` | `class_id=0`, `class_name="safe"`, `confidence=0.95` |
| anything else | `class_id=1`, `class_name="warning"`, `confidence=0.87` |

## Running

### Server

```powershell
python -m src.server.zmq_server --backend fake --host 127.0.0.1 --port 5555
```

### Client

```powershell
python -m src.server.zmq_client --input samples/images/danger_scene.jpg --backend fake
```

## Future Extensibility

The protocol is designed to be extended, not replaced:

- **New backends**: implement `BaseRunner`, register via `register_backend()`.
- **New fields**: additive only — clients and servers that do not understand a field should ignore it.
- **New `input_type` values**: `raw_text`, `tensor_meta`, `encoded_image` — each parsed by the backend's preprocessor.
- **Multiple predictions**: the `prediction` field may become a list (`List[Prediction]`) in a future schema version.
