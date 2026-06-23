# NOTICE

This project is built upon the engineering skeleton of
[yolo-vision-pipeline-rknn](https://github.com/VaporTang/yolo-vision-pipeline-rknn)
by VaporTang, which is licensed under Apache 2.0.

The original project provides a config-driven YOLOv8 → ONNX → RKNN deployment workflow.
This project reframes and extends that skeleton into a **Model Deployment Runtime Lab**:
a lightweight experimental framework for model deployment, ONNX runtime,
quantization, ZMQ inference serving, and deployment benchmarking.

Key architectural ideas adapted from the original:
- Config-driven path management via `configs/paths.yaml`
- Windows / WSL environment split
- Automated setup scripts (`setup_win.ps1`, `setup_wsl.sh`)
- Modular export and conversion pipeline structure

Core functionality (model artifact management, runtime backend abstraction,
ZMQ protocol, benchmark and deployment reporting) is new implementation.
