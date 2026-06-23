# Model Deployment Runtime Lab — Quick Start

Get the environment up and running in 5 minutes.

## Prerequisites

- Windows 10/11 with Miniconda or Anaconda
- Git

## Step 1: Setup

Open PowerShell as Administrator and run:

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
✅ configs
✅ src
✅ README.md
⚠️ outputs  planned
```

## Step 3: What's Next

After Stage 0.5, Stage 1 will add:

```powershell
# Fake runtime smoke
python -m src.server.zmq_server --backend fake

# Run tests
python -m pytest tests -q
```

## WSL RKNN Setup (Optional)

```bash
cd /path/to/model_deployment_runtime_lab
bash setup_wsl.sh
source ~/mdrl-rknn-workdir/rknn-env/bin/activate
```

See [PATH_SETUP.md](PATH_SETUP.md) for detailed path configuration.
