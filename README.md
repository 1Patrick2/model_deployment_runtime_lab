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
| 0.5 | **Current** | Environment foundation, path config, docs reframe |
| 1 | ⏳ Planned | Fake runtime + ZMQ protocol |
| 2+ | 📋 Planned | ONNX export, runtime backend, quantization, benchmark, deployment report |

## Quick Reference

```powershell
# Setup
.\setup_win.ps1
conda activate mdrl-runtime

# Verify
python verify_paths.py

# Future — fake runtime smoke
python -m src.server.zmq_server --backend fake
```

## Current Status

**Stage 0.5 — Environment Foundation** complete.

- ✅ Project docs reframed for deployment runtime lab
- ✅ Requirements split (runtime / dev / train / wsl-rknn)
- ✅ Windows and WSL setup scripts reworked
- ✅ Path config and verification generalized

## Hard Boundaries (First Version)

This project is NOT:
- ❌ A YOLO training/detection pipeline
- ❌ A camera or video processing framework
- ❌ An RKNN board deployment toolkit
- ❌ A large model compression (QAT) framework
- ❌ A multi-node inference service (Kubernetes / Triton)

## License

Apache 2.0. See [LICENSE](LICENSE).
