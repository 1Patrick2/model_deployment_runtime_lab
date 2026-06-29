# Path Configuration Quick Start

## 3-Step Setup

### Step 1: Check Project Root

Open `configs/paths.yaml` and verify:

```yaml
project_root: null  # keep null for auto-detection
```

Or manually specify an absolute path:

```yaml
project_root: C:\Users\YourName\model_deployment_runtime_lab
```

### Step 2: Verify Paths

```powershell
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
[..] samples (planned)
```

### Step 3: View Full Configuration

```powershell
python verify_paths.py --show-config
```

## Key Config Files

### `configs/paths.yaml`

All project paths are defined here:

```yaml
project_root: null

configs:
  root: configs
  paths: configs/paths.yaml
  runtime: configs/runtime.yaml
  zmq: configs/zmq.yaml

artifacts:
  onnx: outputs/onnx
  quantized: outputs/quantized
  reports: outputs/reports
  runtime: outputs/runtime
```

## Using Paths in Python

```python
from src.utils.path_manager import paths

# Get a single path
onnx_dir = paths.get("artifacts.onnx")
print(onnx_dir)  # PosixPath('.../outputs/onnx')

# Ensure a directory exists
paths.ensure_dir("artifacts.onnx")

# List all paths in a section
all_artifacts = paths.get_all("artifacts")
```

## Troubleshooting

- **Auto-detection fails?** Set `project_root` to an absolute path in `configs/paths.yaml`.
- **WSL path issues?** Use `/mnt/c/Users/...` to access Windows files from WSL.
- **Outputs not in expected location?** Edit `configs/paths.yaml`.

## Verification Script

```powershell
# Basic check
python verify_paths.py

# Full config dump
python verify_paths.py --show-config
```
