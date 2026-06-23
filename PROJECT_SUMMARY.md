# Stage 0.5 Environment Foundation Summary

## Status

**Stage 0.5 — Environment Foundation — complete.**

The project has been reframed from a YOLO vision pipeline to the **Model Deployment Runtime Lab**:
a lightweight experimental framework for model deployment, ONNX runtime, quantization,
ZMQ inference serving, and deployment benchmarking.

## What Was Preserved

- Config-driven path management via `configs/paths.yaml`
- `PathManager` singleton with dot-notation path access
- `setup_win.ps1` idempotent Conda environment creation
- `setup_wsl.sh` RKNN_WORKDIR and sparse checkout logic
- `verify_paths.py` self-check entrypoint
- Windows / WSL split architecture
- Apache 2.0 license

## What Was Removed

- YOLOv8 training pipeline and dataset management
- X-AnyLabeling annotation tool integration
- Video frame extraction and deduplication scripts
- Automatic Ultralytics and Rockchip YOLOv8 clone
- YOLO-specific config files (data.yaml, train_config.yaml)
- RKNN-focused README and quickstart narrative

## What Was Added

- New project docs: README, QUICKSTART, PATH_SETUP
- Split requirements: runtime, dev, train, wsl-rknn
- Reworked setup_win.ps1 with mdrl-runtime/dev/train envs
- Generalized paths.yaml for model artifacts and benchmarks
- Updated verify_paths.py with hard-failure / warning semantics
- NOTICE.md and docs/references.md

## Next Stage

**Stage 1 — Fake Runtime + ZMQ Protocol**

- Fake inference runner
- ZMQ request/response protocol
- ZMQ server and client
- Protocol and server tests
