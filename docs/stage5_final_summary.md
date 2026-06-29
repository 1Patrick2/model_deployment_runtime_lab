# Stage 5 Final Summary

## Project Goal

Model Deployment Runtime Lab is a lightweight model deployment runtime and evaluation framework.
It covers ONNX Runtime inference, HTTP/ZMQ serving, QDQ INT8 quantization, RKNN conversion preparation,
HTTP benchmarking, deployment decision reporting, and rule-based deployment advising.

## Pipeline Overview

```
PyTorch / torchvision model export to ONNX
  → ONNX Runtime local inference
  → ZMQ and HTTP serving
  → HTTP end-to-end benchmark
  → QDQ INT8 quantization
  → RKNN conversion for RK3588
  → Deployment decision report
  → Rule-based deployment advisor
```

## Verified Environment

| Item | Value |
|------|-------|
| OS | Windows 11 |
| Conda env | `mdrl-runtime` |
| Python | 3.13.13 |
| Test result (clean checkout) | 111 passed, 10 skipped |
| Test result (full smoke, generated artifacts) | 121 passed, 0 skipped |
| ONNX FP32 artifact | 9.92 MB (generated locally, incl. external data) |
| QDQ INT8 artifact | 2.70 MB (generated locally) |
| RKNN artifact | 5.48 MB (generated in WSL) |

> The 10 skipped tests are artifact-dependent smoke tests (ONNX checker, ORT session) that require locally generated ONNX files. They are not failures.

## Key Commands

### Run all tests

```powershell
conda activate mdrl-runtime
python -m pytest tests -q -rs
```

- Clean checkout: **111 passed, 10 skipped**
- With local ONNX FP32 + QDQ INT8 artifacts: **121 passed, 0 skipped**

### ONNX export (requires mdrl-train with torch)

```powershell
conda activate mdrl-train
python -m src.export.export_onnx --model mobilenet_v3_small --output outputs/onnx/mobilenetv3_small.onnx
```

### HTTP inference server

```powershell
conda activate mdrl-runtime
python -m src.server.http_server --backend onnx --manifest models/manifests/mobilenetv3_small_onnx_fp32.json --port 8001
```

### HTTP benchmark

```powershell
# Requires server running on :8001
python -m src.benchmark.http_benchmark --config configs/http_benchmark_fp32_dummy.yaml
```

### Quantization

```powershell
# QDQ INT8 (recommended)
python -m src.quantization.quantize_onnx --config configs/quant_static_qdq_dummy.yaml

# Dynamic INT8 (experimental, may fail on CPU)
python -m src.quantization.quantize_onnx --config configs/quant_dynamic.yaml
```

### ONNX → RKNN conversion (WSL)

```bash
source ~/mdrl-rknn-workdir/rknn-env/bin/activate
python -m src.rknn.convert --config configs/rknn_convert.yaml
```

### Deployment decision report

```powershell
python -m src.report.deployment_decision --config configs/deployment_decision.yaml
```

### Deployment advisor

```powershell
python -m src.advisor.deployment_advisor --config configs/deployment_advisor.yaml
```

## Deployment Decision

| Target | Recommendation | Reason |
|--------|---------------|--------|
| PC CPU serving | ONNX FP32 | Stable ONNX Runtime path with simple deployment and low local latency |
| Size-sensitive CPU deployment | QDQ INT8 | Smaller artifact (~73% reduction), but real calibration is still required |
| RK3588 deployment | Artifact ready, board validation pending | RKNN conversion succeeded, but no board-side latency yet |

## Known Risks

- QDQ INT8 uses dummy calibration; real accuracy is not validated.
- RKNN Lite2 board-side latency is not available without RK3588 hardware.
- HTTP benchmark is local-loopback only, not production network benchmark.

## Next Work

1. Use representative real images for INT8 calibration.
2. Run RKNN Lite2 inference on RK3588 board.
3. Compare ONNX FP32, QDQ INT8, and RKNN outputs.
