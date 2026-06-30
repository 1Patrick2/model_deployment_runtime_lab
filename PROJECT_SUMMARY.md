# Model Deployment Runtime Lab — Project Summary

## Current Status (Stage 6B)

**All stages complete.** Project has evolved from a YOLO vision pipeline into a full **model deployment runtime lab**:

A lightweight experimental framework for model deployment, ONNX runtime, quantization,
ZMQ/HTTP inference serving, benchmark, RKNN conversion, deployment reporting, and TensorRT GPU benchmark.

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
| 5.1 | Real image HTTP inference server (FastAPI) |
| 5.2 | HTTP inference benchmark |
| 5.3 | Deployment decision report |
| 5.4 | Rule-based deployment advisor |
| 5.5 | Final documentation |
| 6A | Real image calibration, INT8 consistency validation, experiment registry |
| 6B | TensorRT backend benchmark (NVIDIA T400, ~3-4x speedup) |

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
                          │  (fake / onnx / rknn / tensorrt)│
                          └──────────┬───────────┘
                                     │
               ┌─────────────────────┼─────────────────────┐
               │                     │                     │
   ┌───────────▼──────┐  ┌──────────▼──────┐  ┌──────────▼──────┐
   │  ONNX Runtime    │  │  QDQ INT8       │  │  TensorRT       │  │  RKNN Toolkit2  │
   │  (CPU)           │  │  (experimental)  │  │  (GPU, NVIDIA)   │  │  (WSL, rk3588)  │
   └──────────────────┘  └─────────────────┘  └─────────────────┘
```

---

## Project Boundaries

- RKNN conversion is completed, but **RK3588 board-side runtime validation is pending** because RK3588 hardware is not available.
- QDQ INT8 uses dummy calibration; real accuracy is not validated.
- HTTP benchmark is local-loopback only, not production network benchmark.
- `outputs/onnx/`, `outputs/rknn/`, and `outputs/reports/` are local generated artifacts and should not be committed.

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
| TensorRT backend | `src/tensorrt/build_engine.py`, `src/tensorrt/env.py` |

---

## Latest Verified

```powershell
python -m pytest tests -q
```
→ **Latest: 164 passed, 10 skipped** (10 skipped are artifact-dependent smoke tests)

**HTTP ONNX inference:**
- `/health` ok
- `/metadata` ok (mobilenetv3_small, onnx, [1,3,224,224])
- `/infer` dummy ok (~3.5 ms total)
- `/infer` image_path ok (~6.7 ms total)

**HTTP benchmark (dummy, 50 requests):**
- success_rate: 100%
- client_total_ms mean: 5.68 ms
- server_total_ms mean: 1.87 ms

**Deployment decision report (all evidence present):**
- PC CPU serving: onnx_fp32
- Size-sensitive CPU: qdq_int8
- RK3588: artifact ready, board validation pending
- Risks: dummy calibration, no RK3588 board, local loopback only

**Deployment advisor:**
- Generates plain-English explanations from decision report
- Covers PC CPU, size-sensitive, RK3588 recommendations
- Identifies key risks and priority next steps
- No LLM, no external API

**RKNN conversion (WSL):**
- RKNN Toolkit2 2.3.2 installed
- MobileNetV3-small → RK3588 conversion successful
- Output: `outputs/rknn/mobilenetv3_small_fp32.rknn` (5.48 MB)

**TensorRT benchmark (real NVIDIA T400):**
- TensorRT 11.1.0, CUDA 12.4, Driver 552.12
- MobileNetV3-small default: 1028 qps, 0.97 ms mean latency, 10.92 MB engine
- ResNet18 default: 241 qps, 4.15 ms mean latency, 58.76 MB engine
- Precision: default / strongly typed (not FP16)

## Next

Future optional work → RK3588 board-side validation when hardware is available
