# Stage 6A Summary — Quantization Validation & Experiment Registry

## Objective

Stage 6A moves the project beyond "can we generate an INT8 model?" to "is the INT8 model reliable enough to deploy?".

It establishes a **real-image calibration → FP32/INT8 output consistency validation → experiment registry → deployment decision** pipeline. Every quantized artifact is checked for runnability, output consistency, latency, and size before a deployment decision is made.

## Experiment Setup

| Component | Detail |
|-----------|--------|
| Models | MobileNetV3-small, ResNet18 (torchvision pretrained, ImageNet) |
| Calibration images | 50 real images (picsum.photos, placed locally in `samples/images/calibration_real/`) |
| Validation images | 30 images (`samples/images/validation_real/`) |
| Preprocessing | ImageNet v1: resize shorter side to 256, center crop 224, normalize |
| Quantization | ONNX Runtime static QDQ (dummy + real calibration + preprocessed), dynamic linear-only |
| Validation metrics | top1_consistency, top5_consistency, mean_logits_cosine_similarity, mean_confidence_drift, size_reduction_percent, latency |

## Metrics Definition

| Metric | What It Measures | Threshold |
|--------|-----------------|-----------|
| top1_consistency | FP32 and INT8 agree on top-1 class | >= 0.80 |
| top5_consistency | FP32 top-1 falls in INT8 top-5 | >= 0.90 |
| mean_logits_cosine_similarity | Output distribution similarity | >= 0.98 |
| mean_confidence_drift | Average per-class confidence change | <= 0.05 (lower is better) |
| size_reduction_percent | Model file size reduction | Higher is better |

## Results Table

| Model | Strategy | Top1 Cons. | Top5 Cons. | Cosine | Size↓ | Latency (FP32→INT8) | Decision |
|-------|----------|:---------:|:---------:|:-----:|:----:|:-----------------:|:--------:|
| MobileNetV3-small | Static QDQ | 0.0 | 0.0333 | 0.1053 | 72.33% | 5.54→3.61ms | Reject |
| MobileNetV3-small | Preprocessed QDQ | 0.0 | 0.0333 | 0.0912 | 74.33% | 4.55→3.16ms | Reject |
| MobileNetV3-small | Dynamic linear-only | **0.9667** | **1.0** | **0.9997** | 46.75% | 3.77→3.19ms | **Recommended** |
| ResNet18 | Static QDQ | **0.9** | **1.0** | **0.9916** | 74.84% | 14.78→8.04ms | **Recommended** |
| ResNet18 | Preprocessed QDQ | **0.9** | **1.0** | **0.9916** | 74.84% | 15.33→7.87ms | **Recommended** |

## Key Findings

1. **MobileNetV3-static QDQ fails consistently**: Neither ordinary nor preprocessed static QDQ produces usable INT8 models for MobileNetV3-small. Top-1 agreement is 0.0 and cosine similarity is ~0.1, indicating a complete output distribution collapse.

2. **MobileNetV3-dynamic-linear-only is a reliable conservative baseline**: By quantizing only MatMul/Gemm operators and skipping Conv, the dynamic INT8 path achieves 96.67% top-1 consistency and 99.97% cosine similarity with 46.75% size reduction. Latency remains comparable to FP32.

3. **ResNet18 static QDQ passes**: ResNet18 handles static QDQ well (90% top-1 consistency, 99.16% cosine similarity) while achieving 74.84% size reduction and measurable latency improvement (14.78→8.04ms).

4. **ORT quant_pre_process does not improve MobileNetV3 static QDQ**: The preprocessed path technically succeeded (external data resolved, symbolic shape bypassed), but failed to improve consistency.

5. **Conservative deployment decisions are preferred**: The framework correctly rejects unreliable INT8 artifacts and only recommends paths with evidence.

## Limitations

- **Consistency validation ≠ labeled accuracy**. The metrics measure FP32-vs-INT8 output consistency, not classification accuracy against ground truth.
- **Validation images are unlabeled smoke-test images** (picsum.photos). They do not represent the ImageNet validation distribution.
- **True accuracy requires labeled validation data** (e.g., ImageNet val subset with ground truth labels).
- **The validation set is small** (30 images). Increasing to hundreds would improve metric reliability.

## Reproduction Commands

```powershell
# 1. Export FP32 models (mdrl-train environment)
conda activate mdrl-train
python -m src.export.export_onnx --config configs/export/mobilenetv3_small_imagenet.yaml
python -m src.export.export_onnx --config configs/export/resnet18_imagenet.yaml

# 2. Download calibration/validation images
conda activate mdrl-runtime
python -m scripts.download_calibration_images --calibration-count 50 --validation-count 30

# 3. Run all quantization experiments
python -m src.experiments.run_quant_experiments --config configs/experiments/quantization_controls.yaml --all

# 4. Re-run summary with existing artifacts (faster)
python -m src.experiments.run_quant_experiments --config configs/experiments/quantization_controls.yaml --all --reuse-existing
```
