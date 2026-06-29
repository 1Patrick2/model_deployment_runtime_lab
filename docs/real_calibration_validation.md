# Real Calibration and Output Consistency Validation

## Why This Matters

The project currently supports QDQ INT8 quantization, but the calibration is done with **dummy data** (zero tensors). This means:

- The INT8 model is structurally valid — ONNX Runtime can load and run it.
- But the quantization scales are **not tuned to real image distributions**.
- Without real calibration data, we cannot trust that INT8 outputs are close to FP32 outputs.

Real calibration with representative images is a standard step in production deployment pipelines.

## Why a Pretrained Model Is Necessary

Before validating INT8 output quality, we need a **semantically meaningful FP32 baseline**.

- Random-weight models produce random logits — comparing FP32 vs INT8 on random outputs is meaningless.
- A torchvision **pretrained ImageNet model** produces interpretable top-5 predictions (e.g. "tabby cat", "coffee mug").
- This lets us measure meaningful metrics: top-1/top-5 consistency, confidence drift, logits cosine similarity.

All real calibration and validation stages use a pretrained MobileNetV3-small model exported to ONNX.

## Image Directory Convention

| Directory | Purpose |
|-----------|---------|
| `samples/images/calibration_real/` | Representative images used for QDQ INT8 calibration (20-50 images) |
| `samples/images/validation_real/` | Representative images used for FP32 vs INT8 output consistency check (20-50 images) |

**Real images are NOT committed to Git.** Only `.gitkeep` files are tracked in the repository. Place your own images locally.

### Image Requirements

- Format: JPEG or PNG
- Typical size: any (resized to 224×224 during preprocessing)
- Content: common objects — cup, keyboard, mouse, bottle, cat, dog, car, chair, phone, book, etc.
- No manual labels required for consistency validation

## Pipeline Stages

### 1. Export Pretrained ONNX

```powershell
conda activate mdrl-train
python -m src.export.export_onnx --config configs/export_mobilenetv3_small_imagenet.yaml
```

Output: `outputs/onnx/mobilenetv3_small_imagenet_fp32.onnx`

### 2. Real Image QDQ Calibration (Stage 6A.2)

```powershell
conda activate mdrl-runtime
python -m src.quantization.quantize_onnx --config configs/quant_static_qdq_real.yaml
```

Output: `outputs/onnx/mobilenetv3_small_imagenet_int8_qdq_real.onnx`

### 3. FP32 vs INT8 Output Consistency (Stage 6A.3)

```powershell
conda activate mdrl-runtime
python -m src.validation.output_consistency --config configs/validate_fp32_vs_qdq_real.yaml
```

Output: `outputs/reports/quant_validation_*.json` + `.md`

## Key Metrics

| Metric | What It Measures | Typical Threshold |
|--------|-----------------|-------------------|
| top1_consistency | FP32 and INT8 agree on top-1 class | >= 0.80 |
| top5_consistency | FP32 and INT8 top-5 sets overlap | >= 0.90 |
| mean_logits_cosine_similarity | Output distribution similarity | >= 0.98 |
| mean_confidence_drift | Average confidence change | <= 0.05 |

## Current Project Boundaries

- Real calibration and validation require **locally placed images** — not committed to Git.
- Consistency validation does **not** replace labeled accuracy evaluation.
- Full accuracy measurement requires ground-truth labels and a validation dataset like ImageNet val.

## Next After This Stage

- Stage 6A.4: Integrate quant validation results into deployment decision report and advisor.
- Stage 6B: TensorRT FP16 engine build and benchmark (optional).
