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

Clean checkout / no generated ONNX artifacts:
```
111 passed, 10 skipped
```
(The 10 skipped are artifact-dependent smoke tests — not failures.)

With local ONNX FP32 + QDQ INT8 artifacts generated:
```
121 passed, 0 skipped
```

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

## Step 8: ONNX → RKNN Conversion (WSL only)

```bash
# In WSL — requires rknn-env set up by setup_wsl.sh
source ~/mdrl-rknn-workdir/rknn-env/bin/activate

cd /path/to/model_deployment_runtime_lab
python -m src.rknn.convert --config configs/rknn_convert.yaml
```

Expected output:
```
RKNN conversion report written:
  JSON: outputs/reports/rknn_convert_mobilenetv3_small_fp32.json
  MD:   outputs/reports/rknn_convert_mobilenetv3_small_fp32.md
  status: ok
  message: Conversion completed successfully.
```

> **Note:** WSL `rknn-env` is for RKNN conversion only.  It does **not** run
> pytest, ZMQ, or ONNX Runtime.  The RKNN Toolkit2 wheel is installed with
> `--no-deps` after manually installing CPU PyTorch, avoiding GPU/CUDA packages.
> ONNX and protobuf are pinned (1.16.1, 4.25.4) because RKNN Toolkit2 2.3.2
> depends on the legacy `onnx.mapping` API.

## Step 9: Real Image HTTP Inference (PowerShell)

```powershell
conda activate mdrl-runtime

# Terminal 1 — start server
python -m src.server.http_server --backend onnx --manifest models/manifests/mobilenetv3_small_onnx_fp32.json --port 8001

# Terminal 2 — send dummy request
$body = @{ input_type = "dummy"; input = "dummy"; top_k = 5 } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8001/infer" -Method Post `
    -ContentType "application/json" -Body $body
```

Expected response:
```json
status            : ok
backend           : onnx
model_id          : mobilenetv3_small
model_variant     : onnx_fp32_v1
top_k_predictions : {class_id=999 class_name=hook score=0.001, ...}
latency_ms        : @{preprocess=0.16; inference=3.14; postprocess=0.17; total=3.46}
```

For a real image (place your own photo in ``samples/images/real/``):

```powershell
$body = @{ input_type = "image_path"; input = "samples/images/real/cup.jpg"; top_k = 5 } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8001/infer" -Method Post `
    -ContentType "application/json" -Body $body
```

## Step 10: HTTP Benchmark (Optional)

```powershell
# Terminal 1 — start server
conda activate mdrl-runtime
python -m src.server.http_server --backend onnx --manifest models/manifests/mobilenetv3_small_onnx_fp32.json --port 8001

# Terminal 2 — run benchmark
conda activate mdrl-runtime
python -m src.benchmark.http_benchmark --config configs/http_benchmark_fp32_dummy.yaml
```

Expected output:
```
success_rate:        100%
client_total_ms:     mean ~5.7 ms
server_total_ms:     mean ~1.9 ms
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
| 4 | ✅ RKNN conversion |
| 5.1 | ✅ HTTP inference server |
| 5.2 | ✅ HTTP benchmark |
| 5.3 | ✅ Deployment decision report |
| 5.4 | ✅ Deployment advisor |
| 5.5 | **Current** — Final documentation |

## Step 11: Deployment Decision Report

```powershell
conda activate mdrl-runtime
python -m src.report.deployment_decision --config configs/deployment_decision.yaml
```

Expected output:
```
PC CPU serving:              onnx_fp32
Size-sensitive CPU deploy:   qdq_int8
RK3588 deployment:           artifact ready, board validation pending
Risks (3): QDQ INT8 dummy calibration, RK3588 board validation pending, HTTP local loopback only
```

## Step 12: Deployment Advisor

```powershell
conda activate mdrl-runtime
python -m src.advisor.deployment_advisor --config configs/deployment_advisor.yaml
```

Generates a Markdown explanation of the deployment decision report with recommendations, risks, and next steps.

### Windows vs WSL

| OS | Environment | Role |
|----|-------------|------|
| Windows | `mdrl-runtime` | ONNX Runtime, ZMQ, benchmark, quantization, **pytest** |
| Windows | `mdrl-dev` | Lightweight dev / pytest |
| Windows | `mdrl-train` | Optional: PyTorch export (manual install) |
| WSL | `rknn-env` | **Only** RKNN Toolkit2 conversion |

See [docs/zmq_protocol_design.md](docs/zmq_protocol_design.md) for protocol details.
