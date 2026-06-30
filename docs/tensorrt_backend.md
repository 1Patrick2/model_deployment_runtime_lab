# TensorRT Backend

## Stage 6B Objective

TensorRT is an optional GPU-accelerated backend for faster inference.
This module provides:

- **Environment detection** — check for `trtexec` and GPU availability
- **Engine build** — ONNX → TensorRT engine via `trtexec`
- **Benchmark** — `trtexec` latency / throughput measurement
- **Report** — structured JSON and Markdown reports

## Prerequisites

- NVIDIA GPU with TensorRT 8.x+ installed
- `trtexec` in PATH (comes with TensorRT)
- ONNX FP32 models exported

CPU-only users can still run all other stages (6A). TensorRT tests and dry-run
are safe on CPU-only systems and will produce "skipped" reports.

## Precision

- `precision: default` — no explicit precision flag (TensorRT's default strongly-typed path).
  Recommended for TensorRT 11.x environments where `--fp16` may not be available.
- `precision: fp16` — adds `--fp16` flag (not all `trtexec` versions support this).

## Usage

### Build TensorRT engine

```powershell
python -m src.tensorrt.build_engine --config configs/tensorrt/mobilenetv3_small_default.yaml
```

### Dry-run (print command without executing)

```powershell
python -m src.tensorrt.build_engine --config configs/tensorrt/mobilenetv3_small_default.yaml --dry-run
```

### Build + benchmark

```powershell
python -m src.tensorrt.build_engine --config configs/tensorrt/mobilenetv3_small_default.yaml --benchmark
```

### ResNet18

```powershell
python -m src.tensorrt.build_engine --config configs/tensorrt/resnet18_default.yaml --benchmark
```

## Report

Output files:

| File | Content |
|------|---------|
| `outputs/reports/tensorrt_*.json` | Structured benchmark data |
| `outputs/reports/tensorrt_*.md` | Human-readable report |

Key metrics:

- **Throughput (qps)**: inferences per second
- **Mean / median / min / max latency** (ms)
- **GPU Compute Time** (ms)
- **H2D / D2H latency** (ms)
- **Engine size** (MB)

## CPU-Only Behavior

On systems without TensorRT:

```powershell
python -m src.tensorrt.build_engine --config configs/tensorrt/mobilenetv3_small_default.yaml
```
→ Produces a report with `build_status: "skipped"` and explanation.

## Verified Results

Real benchmark completed on NVIDIA T400 4GB with TensorRT 11.1.0:

| Model | Throughput | Mean Latency | Speedup vs ONNX CPU |
|-------|:---------:|:-----------:|:-----------------:|
| MobileNetV3-small | 1028 qps | 0.97 ms | ~3.1x |
| ResNet18 | 241 qps | 4.15 ms | ~3.6x |

See `docs/results/stage6b_tensorrt_summary.md` for full results.

## Limitations

- Default precision uses TensorRT's strongly-typed path (not explicit FP16/INT8).
- Input shapes are fixed (`1x3x224x224`); dynamic shapes not supported.
- Requires NVIDIA GPU.
- Python TensorRT API is not used directly — `trtexec` CLI is the benchmark tool.
