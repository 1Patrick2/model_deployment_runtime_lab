# Model Deployment Runtime Lab — Project Summary

## Current Status (Stage 5.1)

**Real Image HTTP Inference Server — in progress.**

The project has evolved from a YOLO vision pipeline into a full **model deployment runtime lab**:

A lightweight experimental framework for model deployment, ONNX runtime, quantization,
ZMQ/HTTP inference serving, benchmark, RKNN conversion, and deployment reporting.

---

## Stages Completed

| Stage | What |
|-------|------|
| 0 | Initialize project from deployment pipeline skeleton |
| 0.5 | Environment foundation: path config, docs reframe, requirements split |
| 1 | Fake runtime + ZMQ protocol (REQ/REP, protocol schema, error handling) |
| 2.1 | Model manifest / registry (Pydantic schema, load/save, lookup) |
| 2.2 | ONNX export (MobileNetV3-small via torchvision) |
| 2.3 | ONNX Runtime runner (InferenceSession, dummy/image input, latency_ms) |
| 2.4 | ZMQ backend=onnx (server --manifest, client --input-type) |
| 2.5 | Latency benchmark (warmup, repeat, mean/p50/p95, JSON/MD reports) |
| 3 | ONNX quantization (dynamic experimental + QDQ INT8 recommended) |
| 4 | RKNN conversion (WSL, rknn-toolkit2, RK3588, 5.48 MB artifact) |
| 5.1 | **Current** — Real image HTTP inference server (FastAPI) |

---

## Architecture

```
                          ┌──────────────────────┐
                          │   Model Zoo / Export  │
                          │  (torchvision → ONNX) │
                          └──────────┬───────────┘
                                     │
                          ┌──────────▼───────────┐
                          │   Model Registry     │
                          │  (manifest + lookup) │
                          └──────────┬───────────┘
                                     │
               ┌─────────────────────┼─────────────────────┐
               │                     │                     │
   ┌───────────▼──────┐  ┌──────────▼──────┐  ┌──────────▼──────┐
   │  Benchmark       │  │  ZMQ Server     │  │  HTTP Server    │
   │  (latency/report) │  │  (REQ/REP)      │  │  (FastAPI)      │
   └──────────────────┘  └─────────────────┘  └─────────────────┘
               │                     │                     │
               └─────────────────────┼─────────────────────┘
                                     │
                          ┌──────────▼───────────┐
                          │   Runtime Backends   │
                          │  (fake / onnx / rknn)│
                          └──────────┬───────────┘
                                     │
               ┌─────────────────────┼─────────────────────┐
               │                     │                     │
   ┌───────────▼──────┐  ┌──────────▼──────┐  ┌──────────▼──────┐
   │  ONNX Runtime    │  │  QDQ INT8       │  │  RKNN Toolkit2  │
   │  (CPU)           │  │  (experimental)  │  │  (WSL, rk3588)  │
   └──────────────────┘  └─────────────────┘  └─────────────────┘
```

---

## Platform Roles

| Platform | Environment | Responsibility |
|----------|-------------|----------------|
| Windows | `mdrl-runtime` | ONNX Runtime, ZMQ, HTTP, benchmark, quantization, **pytest** |
| Windows | `mdrl-dev` | Lightweight dev / lint |
| Windows | `mdrl-train` | Optional: PyTorch export (manual torch install) |
| WSL | `rknn-env` | RKNN Toolkit2 conversion only |

---

## Key Files

| Category | Path |
|----------|------|
| Protocol | `src/server/protocol.py` |
| Runtime interface | `src/runtime/base_runner.py` |
| Fake runner | `src/runtime/fake_runner.py` |
| ONNX Runner | `src/runtime/onnx_runner.py` |
| ZMQ server | `src/server/zmq_server.py` |
| ZMQ client | `src/server/zmq_client.py` |
| HTTP server | `src/server/http_server.py` |
| HTTP schema | `src/server/http_schema.py` |
| Model manifest | `src/models/manifest.py` |
| Model registry | `src/models/registry.py` |
| ONNX export | `src/export/export_onnx.py` |
| ONNX quantization | `src/quantization/quantize_onnx.py` |
| Comparison report | `src/quantization/compare_reports.py` |
| Benchmark | `src/benchmark/latency.py`, `src/benchmark/report.py` |
| RKNN conversion | `src/rknn/convert.py` |
| Image preprocessing | `src/runtime/image_preprocess.py` |
| Classification post-process | `src/runtime/classification_postprocess.py` |
| ImageNet labels | `src/models/imagenet_labels.py` |

---

## Latest Verified

```powershell
python -m pytest tests -q
```
→ **95 passed, 2 skipped** (Windows mdrl-runtime)

**HTTP ONNX inference:**
- `/health` ok
- `/metadata` ok (mobilenetv3_small, onnx, [1,3,224,224])
- `/infer` dummy ok (~3.5 ms total)
- `/infer` image_path ok (~6.7 ms total)

**RKNN conversion (WSL):**
- RKNN Toolkit2 2.3.2 installed
- MobileNetV3-small → RK3588 conversion successful
- Output: `outputs/rknn/mobilenetv3_small_fp32.rknn` (5.48 MB)

---

## Next

Stage 5.2 → HTTP Benchmark + Deployment Decision Report
