# Model Deployment Runtime Lab — Quick Start

Get the environment up and running in 5 minutes.

## Prerequisites

- Windows 10/11 with Miniconda or Anaconda
- Git

## Step 1: Setup

Open PowerShell and run:

```powershell
cd path\to\model_deployment_runtime_lab
.\setup_win.ps1
```

This creates two Conda environments:
- `mdrl-runtime` — ONNX Runtime, ZMQ, benchmark
- `mdrl-dev` — pytest

## Step 2: Verify

```powershell
conda activate mdrl-runtime
python verify_paths.py
```

Expected output:
```
Model Deployment Runtime Lab - Path Verification
Project Root: ...
[OK] configs
[OK] src
[OK] README.md
[..] outputs (planned)
```

## Step 3: Run Tests

```powershell
conda activate mdrl-runtime
python -m pytest tests -q
```

All 23 tests should pass.

## Step 4: Try the Fake Inference Server

Open **two** terminals:

**Terminal 1 — Server:**

```powershell
conda activate mdrl-runtime
python -m src.server.zmq_server --backend fake
```

**Terminal 2 — Client:**

```powershell
conda activate mdrl-runtime
python -m src.server.zmq_client --input samples/images/danger_scene.jpg
```

Expected response:
```json
{
  "request_id": "...",
  "status": "ok",
  "prediction": {
    "class_id": 2,
    "class_name": "danger",
    "confidence": 0.99
  },
  "backend": "fake"
}
```

## WSL RKNN Setup (Optional)

```bash
cd /path/to/model_deployment_runtime_lab
bash setup_wsl.sh
source ~/mdrl-rknn-workdir/rknn-env/bin/activate
```

## Step 5: Export ONNX Model (Optional, requires mdrl-train)

> **Note:** PyTorch is intentionally **not** included in `requirements_win_train.txt`
> because CPU/CUDA selection depends on your device. Install manually below.

```powershell
.\setup_win.ps1 --with-train
conda activate mdrl-train

# Install torch & torchvision (CPU version — or choose CUDA from pytorch.org)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Export MobileNetV3-small to ONNX (no pretrained weights needed for pipeline test)
python -m src.export.export_onnx --model mobilenet_v3_small --output outputs/onnx/mobilenetv3_small.onnx
```

Expected output:
```
Loading mobilenet_v3_small (pretrained=False) ...
Exporting to outputs/onnx/mobilenetv3_small.onnx  (opset=17) ...
Done.  Artifact size: ~9.8 MB
```

## Step 6: Run ONNX Inference via ZMQ

```powershell
# Terminal 1 — start server
conda activate mdrl-runtime
python -m src.server.zmq_server --backend onnx --manifest models/manifests/mobilenetv3_small_onnx_fp32.json

# Terminal 2 — send request
conda activate mdrl-runtime
python -m src.server.zmq_client --backend onnx --input-type dummy --input dummy
```

Expected response:
```json
{
  "status": "ok",
  "backend": "onnx",
  "model_id": "mobilenetv3_small",
  "prediction": {"class_id": 0, "class_name": "class_0", "confidence": 0.001},
  "latency_ms": {"preprocess": 0.13, "inference": 2.54, "total": 2.8}
}
```

## Step 7: Quantization and Benchmark Comparison (Optional)

```powershell
conda activate mdrl-runtime

# Recommended: static QDQ quantization with dummy calibration
python -m src.quantization.quantize_onnx --config configs/quant_static_qdq_dummy.yaml

# Experimental: dynamic quantization (may fail on CPU due to ConvInteger)
python -m src.quantization.quantize_onnx --config configs/quant_dynamic.yaml

# FP32 benchmark
python -m src.benchmark.latency --config configs/benchmark.yaml

# QDQ INT8 benchmark
python -m src.benchmark.latency --config configs/benchmark_int8_qdq_dummy.yaml

# Comparison report
python -m src.quantization.compare_reports `
    --baseline outputs/reports/benchmark_mobilenetv3_small_onnx_fp32.json `
    --candidate outputs/reports/benchmark_mobilenetv3_small_onnx_int8_qdq_dummy.json `
    --output-json outputs/reports/compare_mobilenetv3_small_fp32_vs_int8_qdq_dummy.json `
    --output-md outputs/reports/compare_mobilenetv3_small_fp32_vs_int8_qdq_dummy.md
```

## Project Status

| Stage | Status |
|-------|--------|
| 0 | ✅ Initialize project |
| 0.5 | ✅ Environment foundation |
| 1 | ✅ Fake runtime + ZMQ protocol |
| 2.1 | ✅ Model manifest / registry |
| 2.2 | ✅ ONNX export (MobileNetV3) |
| 2.3 | ✅ ONNX Runtime backend |
| 2.4 | ✅ ZMQ backend=onnx |
| 2.5 | ✅ Latency benchmark |
| 3 | ✅ ONNX quantization |
| 4 | **Current** — RKNN conversion preparation |

### Windows vs WSL Role

| OS | Role | Responsibilities |
|----|------|------------------|
| Windows `mdrl-runtime` | Main runtime | ONNX Runtime, ZMQ, benchmark, quantization, report |
| WSL `rknn-env` | RKNN conversion | ONNX → RKNN conversion via RKNN Toolkit2 |

See [docs/zmq_protocol_design.md](docs/zmq_protocol_design.md) for protocol details.
