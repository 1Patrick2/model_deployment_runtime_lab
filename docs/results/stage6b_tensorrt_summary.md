# Stage 6B Summary — TensorRT Benchmark Results

## Environment

| Component | Detail |
|-----------|--------|
| GPU | NVIDIA T400 4GB |
| TensorRT | 11.1.0 |
| CUDA | 12.4 |
| Driver | 552.12 |
| Precision | Default (strongly typed) |

## Results

| Model | Engine Size | Throughput (qps) | Mean Latency (ms) | GPU Compute (ms) |
|-------|:----------:|:----------------:|:-----------------:|:----------------:|
| MobileNetV3-small | — | **1024.52** | **0.974** | 0.974 |
| ResNet18 | 58.74 MiB | **239.02** | **4.182** | 4.182 |

## Key Findings

1. **TensorRT provides significant speedup** over ONNX Runtime CPU for both models.
2. MobileNetV3 achieves ~1000+ qps with sub-millisecond latency on T400.
3. ResNet18 achieves ~240 qps with ~4ms latency.
4. The TRT environment uses TensorRT 11.1 default strongly-typed execution.
5. `--fp16` flag is not available in this environment's `trtexec`; `precision: default` is used.

## Commands

```powershell
# Build + benchmark
python -m src.tensorrt.build_engine --config configs/tensorrt/mobilenetv3_small_default.yaml --benchmark
python -m src.tensorrt.build_engine --config configs/tensorrt/resnet18_default.yaml --benchmark
```

## Notes

- These results were obtained on a physical Windows machine with NVIDIA T400 GPU.
- The CPU ONNX Runtime baseline (Stage 2-3) provides a reference for speedup comparison.
- TensorRT is optional; all other stages run on CPU.
