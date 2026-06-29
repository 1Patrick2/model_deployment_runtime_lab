# Model Deployment Runtime Lab

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**模型部署运行时与优化评测实验框架**

A lightweight experimental framework for model deployment, ONNX runtime, quantization, ZMQ-based inference serving, and deployment benchmarking.

## Overview

Model Deployment Runtime Lab bridges the gap between model training and production deployment. It provides a structured pipeline for:

- **Model Artifact Management** — versioned model manifests with accuracy, latency, and size tracking
- **Multi-Backend Runtime** — fake / PyTorch / ONNX Runtime / optional RKNN abstraction
- **Model Export & Optimization** — PyTorch → ONNX export, PTQ quantization, light structural pruning
- **Inference Serving** — ZMQ-based client-server protocol with per-request latency breakdown
- **Deployment Benchmarking** — accuracy eval, P50/P95 latency, throughput, model size comparison
- **Deployment Report** — automatic generation of `deployment_report.md` with deploy/don't-deploy recommendations

## Pipeline

```
Model Zoo (MobileNetV3 / ResNet18)
  → Baseline Evaluation
  → ONNX Export
  → ONNX Runtime Benchmark
  → PTQ Quantization
  → Optional Light Pruning
  → Runtime Backend Registry
  → ZMQ Inference Server
  → Benchmark Report
  → Deployment Report
```

## Windows / WSL Split

| OS | Role | Environment | Responsibilities |
|----|------|-------------|------------------|
| Windows | Main runtime | `mdrl-runtime` | ONNX Runtime, ZMQ, benchmark, quantization, report |
| Windows | Dev tools | `mdrl-dev` | pytest, linting |
| Windows | Optional training | `mdrl-train` | PyTorch baseline, pruning |
| WSL | RKNN conversion | `rknn-env` | ONNX → RKNN conversion (optional, via rknn-toolkit2) |

## Stage Roadmap

| Stage | Status | What |
|-------|--------|------|
| 0 | ✅ Complete | Initialize project from deployment pipeline skeleton |
| 0.5 | ✅ Complete | Environment foundation, path config, docs reframe |
| 1 | ✅ Complete | Fake runtime + ZMQ protocol |
| 2.1 | ✅ Complete | Model manifest / registry |
| 2.2 | ✅ Complete | ONNX export (MobileNetV3-small) |
| 2.3 | ✅ Complete | ONNX Runtime runner |
| 2.4 | ✅ Complete | ZMQ backend=onnx |
| 2.5 | ✅ Complete | Latency benchmark |
| 3 | ✅ Complete | ONNX QDQ quantization + FP32/INT8 comparison |
| 4 | ✅ Complete | RKNN conversion (WSL, rknn-toolkit2) |
| 5.1 | **Current** | Real image HTTP inference server |

## Stage 1 — Fake Runtime + ZMQ Protocol

A minimal inference serving skeleton has been built:

```
client → JSON (InferenceRequest) → ZMQ REQ/REP → fake runner → JSON (InferenceResponse)
```

### Server

```powershell
conda activate mdrl-runtime
python -m src.server.zmq_server --backend fake --host 127.0.0.1 --port 5555
```

### Client (separate terminal)

```powershell
conda activate mdrl-runtime
python -m src.server.zmq_client --input samples/images/danger_scene.jpg --backend fake
```

Expected output:

```json
{
  "request_id": "67a1b2c3",
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
  "model_variant": "fake_v1"
}
```

Input heuristics:

| Input contains | Prediction |
|----------------|------------|
| `"danger"` | danger (class_id=2, confidence 0.99) |
| `"safe"` | safe (class_id=0, confidence 0.95) |
| anything else | warning (class_id=1, confidence 0.87) |

See [docs/zmq_protocol_design.md](docs/zmq_protocol_design.md) for the full protocol specification.

## Quick Reference

```powershell
# Setup
.\setup_win.ps1
conda activate mdrl-runtime

# Verify
python verify_paths.py

# Run tests
python -m pytest tests -q

# Start fake inference server (Terminal 1)
python -m src.server.zmq_server --backend fake

# Send a request (Terminal 2)
python -m src.server.zmq_client --input samples/images/danger_scene.jpg
```

## Current Status

**Stage 5.1 — Real Image HTTP Inference Server (in progress).**

| Stage | What | Verified |
|-------|------|----------|
| 1–3 | Fake → ONNX → ZMQ → Benchmark → Quantization | Windows `mdrl-runtime`, pytest |
| 4 | RKNN conversion | WSL `rknn-env`, output 5.48 MB |
| 5.1 | **HTTP inference server** | `/health` ok, `/metadata` ok, `/infer` dummy ok (total ~3.5 ms) |

### Quantization Notes

- **Dynamic INT8** (experimental) — may fail on CPUExecutionProvider due to ConvInteger for Conv-heavy CNNs.
- **QDQ INT8** (recommended CPU-runnable path) — reduces model size by ~73% while keeping CPU dummy-input latency comparable to FP32.

### Environment Roles

| OS | Environment | What it runs |
|----|-------------|-------------|
| Windows | `mdrl-runtime` | ONNX Runtime, ZMQ, HTTP server, benchmark, quantization, **pytest** |
| Windows | `mdrl-dev` | Lightweight dev environment for independent pytest / lint |
| Windows | `mdrl-train` | Optional: PyTorch / torchvision export (CPU/CUDA, install manually) |
| WSL | `rknn-env` | **Only** RKNN Toolkit2 conversion — no pytest, no ZMQ, no ONNX Runtime |

### Real Image HTTP Server

Run real ONNX inference via HTTP:

```powershell
# Terminal 1 — start server
conda activate mdrl-runtime
python -m src.server.http_server --backend onnx --manifest models/manifests/mobilenetv3_small_onnx_fp32.json --port 8001

# Terminal 2 — request (PowerShell)
$body = @{ input_type = "dummy"; input = "dummy"; top_k = 5 } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8001/infer" -Method Post `
    -ContentType "application/json" -Body $body
```

For a real image placed in ``samples/images/real/``:

```powershell
$body = @{ input_type = "image_path"; input = "samples/images/real/cup.jpg"; top_k = 5 } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8001/infer" -Method Post `
    -ContentType "application/json" -Body $body
```

Endpoints:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check → `{"status":"ok"}` |
| GET | `/metadata` | Model metadata (id, variant, backend, shape) |
| POST | `/infer` | Inference → top-k predictions + latency breakdown |

Latest verified (Windows mdrl-runtime ONNX dummy):
```
/health: ok
/metadata: ok
/infer: ok, total latency ~3.46 ms
```

### Quick Commands

```powershell
# ONNX quantization (recommended: static QDQ with dummy calibration)
python -m src.quantization.quantize_onnx --config configs/quant_static_qdq_dummy.yaml

# ONNX quantization (experimental: dynamic — may fail on CPU due to ConvInteger)
python -m src.quantization.quantize_onnx --config configs/quant_dynamic.yaml

# FP32 benchmark
python -m src.benchmark.latency --config configs/benchmark.yaml

# QDQ INT8 benchmark
python -m src.benchmark.latency --config configs/benchmark_int8_qdq_dummy.yaml

# FP32 vs QDQ INT8 comparison
python -m src.quantization.compare_reports `
    --baseline outputs/reports/benchmark_mobilenetv3_small_onnx_fp32.json `
    --candidate outputs/reports/benchmark_mobilenetv3_small_onnx_int8_qdq_dummy.json `
    --output-json outputs/reports/compare_mobilenetv3_small_fp32_vs_int8_qdq_dummy.json `
    --output-md outputs/reports/compare_mobilenetv3_small_fp32_vs_int8_qdq_dummy.md
```

## Hard Boundaries (First Version)

This project is NOT:
- ❌ A YOLO training/detection pipeline
- ❌ A camera or video processing framework
- ❌ An RKNN board deployment toolkit
- ❌ A large model compression (QAT) framework
- ❌ A multi-node inference service (Kubernetes / Triton)

## License

Apache 2.0. See [LICENSE](LICENSE).
