# Stage 6B Summary — TensorRT Benchmark Results

## Environment

| Component | Detail |
|-----------|--------|
| GPU | NVIDIA T400 4GB |
| TensorRT | 11.1.0 |
| CUDA Runtime | 12.4 |
| Driver | 552.12 |
| Precision | `default` (strongly typed) |

## Results

### MobileNetV3-small

| Metric | Value |
|--------|------:|
| Engine size | 10.92 MB |
| Throughput | **1028.41 qps** |
| Mean latency | **0.971 ms** |
| Median latency | 0.939 ms |
| Min latency | 0.926 ms |
| Max latency | 2.063 ms |
| GPU Compute Time (mean) | 0.971 ms |

### ResNet18

| Metric | Value |
|--------|------:|
| Engine size | **58.76 MB** |
| Throughput | **240.74 qps** |
| Mean latency | **4.152 ms** |
| Median latency | 4.074 ms |
| Min latency | 3.912 ms |
| Max latency | 5.352 ms |
| GPU Compute Time (mean) | 4.152 ms |

### ONNX Runtime CPU Comparison (Reference)

| Model | ONNX CPU (mean) | TensorRT (mean) | Speedup |
|-------|:--------------:|:---------------:|:-------:|
| MobileNetV3-small | ~3.0 ms | **0.97 ms** | ~3.1x |
| ResNet18 | ~15.0 ms | **4.15 ms** | ~3.6x |

## Commands

```powershell
# Build + benchmark (requires NVIDIA GPU + TensorRT)
python -m src.tensorrt.build_engine --config configs/tensorrt/mobilenetv3_small_default.yaml --benchmark
python -m src.tensorrt.build_engine --config configs/tensorrt/resnet18_default.yaml --benchmark

# Dry-run (CPU-only, prints commands)
python -m src.tensorrt.build_engine --config configs/tensorrt/mobilenetv3_small_default.yaml --dry-run
```

## Notes

- These results use TensorRT 11.1 **default / strongly typed** precision.
- `--fp16` is not available in this `trtexec` version; `precision: default` is used.
- TensorRT is optional — all other stages (ONNX Runtime, quantization, serving) run on CPU.
- Dry-run and all tests are safe on CPU-only systems.
