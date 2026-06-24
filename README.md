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
| Windows | Main runtime | `mdrl-runtime` | ONNX Runtime, ZMQ, benchmark, report |
| Windows | Dev tools | `mdrl-dev` | pytest, linting |
| Windows | Optional training | `mdrl-train` | PyTorch baseline, pruning |
| WSL | RKNN conversion | `rknn-env` | ONNX → RKNN conversion (optional) |

## Stage Roadmap

| Stage | Status | What |
|-------|--------|------|
| 0 | ✅ Complete | Initialize project from deployment pipeline skeleton |
| 0.5 | ✅ Complete | Environment foundation, path config, docs reframe |
| 1 | ✅ Complete | Fake runtime + ZMQ protocol |
| 2 | **Current** | ONNX Runtime backend

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

**Stage 2 — ONNX Runtime Backend (in progress).**

- ✅ Stage 1: Fake runtime + ZMQ protocol, 24 tests
- ✅ Stage 2.1: Model manifest / registry with Pydantic schema
- ⏳ Stage 2.2: ONNX export (MobileNetV3-small)
- ⏳ Stage 2.3: ONNX Runtime runner
- ⏳ Stage 2.4: ZMQ backend=onnx
- ⏳ Stage 2.5: Latency benchmark

## Hard Boundaries (First Version)

This project is NOT:
- ❌ A YOLO training/detection pipeline
- ❌ A camera or video processing framework
- ❌ An RKNN board deployment toolkit
- ❌ A large model compression (QAT) framework
- ❌ A multi-node inference service (Kubernetes / Triton)

## License

Apache 2.0. See [LICENSE](LICENSE).
