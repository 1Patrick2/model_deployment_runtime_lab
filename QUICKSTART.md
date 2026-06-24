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

## Project Status

| Stage | Status |
|-------|--------|
| 0 | ✅ Initialize project |
| 0.5 | ✅ Environment foundation |
| 1 | ✅ **Fake runtime + ZMQ protocol** |
| 2 | 📋 ONNX Runtime backend |

See [docs/zmq_protocol_design.md](docs/zmq_protocol_design.md) for protocol details.
