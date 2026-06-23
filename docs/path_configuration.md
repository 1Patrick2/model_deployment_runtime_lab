# Path Configuration Guide

## Overview

Model Deployment Runtime Lab uses a centralized path configuration system.
All paths are defined in `configs/paths.yaml`, and the `PathManager` class
provides dot-notation access from Python code.

## Quick Start

### 1. Set Project Root (Optional)

Edit `configs/paths.yaml`:

```yaml
project_root: null  # null = auto-detect
```

Or specify an absolute path:

```yaml
project_root: C:\Users\YourName\model_deployment_runtime_lab
```

### 2. View Current Configuration

```powershell
python verify_paths.py
python verify_paths.py --show-config
```

### 3. Customize Paths

Edit `configs/paths.yaml` to change default paths.

## Path Structure

```yaml
configs/
  root: configs
  paths: configs/paths.yaml
  model: configs/model.yaml
  export: configs/export.yaml
  quant: configs/quant.yaml
  runtime: configs/runtime.yaml
  zmq: configs/zmq.yaml
  benchmark: configs/benchmark.yaml
  rknn: configs/rknn.yaml

artifacts/
  onnx: outputs/onnx
  quantized: outputs/quantized
  baseline: outputs/baseline
  reports: outputs/reports
  runtime: outputs/runtime

models/
  registry: models/registry.json
  manifests: models/manifests

data/
  samples: samples/images
  calibration: samples/calibration
```

## Usage in Python

```python
from src.utils.path_manager import paths

# Get resolved path
onnx_dir = paths.get("artifacts.onnx")         # PosixPath('.../outputs/onnx')
reports_dir = paths.get("artifacts.reports")
registry = paths.get("models.registry")

# Get as string
onnx_str = paths.get_str("artifacts.onnx")

# Ensure directory exists
paths.ensure_dir("artifacts.runtime")

# Get all config paths
all_artifacts = paths.get_all("artifacts")

# Print current configuration
paths.print_config()
```

## Running Path Verification

```powershell
# Basic verification
python verify_paths.py

# Show full configuration
python verify_paths.py --show-config
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `configs/paths.yaml not found` | Run from project root, or check `project_root` setting |
| Paths resolve incorrectly | Check for typos in `configs/paths.yaml` |
| Directory doesn't exist | Use `paths.ensure_dir()` to create it |
