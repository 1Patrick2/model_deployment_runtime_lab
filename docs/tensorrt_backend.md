# TensorRT FP16 Backend

## Stage 6B Objective

TensorRT FP16 is an optional GPU-accelerated backend for faster inference.
This stage provides:

- **Environment detection** — check for `trtexec` and GPU availability
- **Engine build** — ONNX → TensorRT FP16 engine via `trtexec`
- **Benchmark** — `trtexec` latency / throughput measurement
- **Report** — structured JSON and Markdown reports

## Prerequisites

- NVIDIA GPU with TensorRT 8.x+ installed
- `trtexec` in PATH (comes with TensorRT)
- ONNX FP32 models exported (see `configs/export/`)

CPU-only users can still run all other stages (6A). TensorRT tests are safe
on CPU-only systems and will produce "skipped" reports.

## Usage

### Build TensorRT FP16 engine

```powershell
python -m src.tensorrt.build_engine --config configs/tensorrt/mobilenetv3_small_fp16.yaml
```

### Dry-run (print command without executing)

```powershell
python -m src.tensorrt.build_engine --config configs/tensorrt/mobilenetv3_small_fp16.yaml --dry-run
```

### Build + benchmark

```powershell
python -m src.tensorrt.build_engine --config configs/tensorrt/mobilenetv3_small_fp16.yaml --benchmark
```

### ResNet18

```powershell
python -m src.tensorrt.build_engine --config configs/tensorrt/resnet18_fp16.yaml --benchmark
```

## Report

Output files:

| File | Content |
|------|---------|
| `outputs/reports/tensorrt_*.json` | Structured benchmark data |
| `outputs/reports/tensorrt_*.md` | Human-readable report |

Key metrics:

- **Throughput (qps)**: inferences per second
- **Mean / median / min / max latency (ms)** 
- **GPU Compute Time (ms)**
- **H2D / D2H latency (ms)**
- **Engine size (MB)**

## CPU-Only Behavior

On systems without TensorRT:

```powershell
python -m src.tensorrt.build_engine --config configs/tensorrt/mobilenetv3_small_fp16.yaml
```
→ Produces a report with `build_status: "skipped"` and explanation.

All TensorRT tests are safe on CPU-only systems (monkeypatched/mocked).

## Limitations

- Only FP16 precision is implemented (no TensorRT INT8).
- Input shapes are fixed (`1x3x224x224`); dynamic shapes not supported.
- Requires NVIDIA GPU with TensorRT installed.
- `trtexec` is used as the benchmark tool; Python TensorRT API is not used directly.
