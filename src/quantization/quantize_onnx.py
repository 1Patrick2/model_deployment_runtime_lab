"""ONNX quantization script — dynamic, dummy QDQ, and real-image QDQ.

Usage
-----
.. code-block:: powershell

    # Dynamic quantization (experimental: ConvInteger may fail on CPU)
    python -m src.quantization.quantize_onnx --config configs/quant_dynamic.yaml

    # Static QDQ quantization with dummy calibration
    python -m src.quantization.quantize_onnx --config configs/quant_static_qdq_dummy.yaml

    # Static QDQ quantization with real image calibration
    python -m src.quantization.quantize_onnx --config configs/quant_static_qdq_real.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import onnx
import yaml
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)

from src.quantization.image_calibration_reader import ImageCalibrationDataReader


# ── Dummy calibration data reader ────────────────────────────────


class DummyCalibrationDataReader(CalibrationDataReader):
    """Yields zero-valued tensors for static QDQ calibration.

    Using dummy data means the calibration only serves to *make the
    pipeline runnable* — it does **not** produce accuracy-aware
    quantization scales.
    """

    def __init__(
        self,
        input_name: str,
        input_shape: list[int],
        samples: int = 8,
    ) -> None:
        self.input_name = input_name
        self.input_shape = input_shape
        self.samples = samples
        self._count = 0

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        if self._count >= self.samples:
            return None
        self._count += 1
        return {self.input_name: np.zeros(self.input_shape, dtype=np.float32)}


# ── Shape-inference helper ──────────────────────────────────────


def _relaxed_shape_infer(model: onnx.ModelProto) -> onnx.ModelProto:
    """Re-run shape inference with relaxed settings.

    Models exported via newer PyTorch dynamo backends sometimes embed
    conflicting shape info.  Clearing that info and re-inferring with
    ``strict_mode=False`` avoids spurious failures.
    """
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                _clear_value_info(attr.g)
    _clear_value_info(model.graph)
    return onnx.shape_inference.infer_shapes(
        model, strict_mode=False, check_type=False
    )


def _clear_value_info(graph) -> None:
    while graph.value_info:
        graph.value_info.pop()


# ── Quantisation runners ────────────────────────────────────────


def _run_dynamic(inp: Path, out: Path) -> None:
    """Dynamic quantization (may produce unsupported ConvInteger ops)."""
    model = onnx.load(str(inp))
    inferred = _relaxed_shape_infer(model)
    quantize_dynamic(
        model_input=inferred,
        model_output=str(out),
        weight_type=QuantType.QInt8,
    )


def _run_static_qdq(inp: Path, out: Path, config: Dict[str, Any]) -> None:
    """Static QDQ quantisation with a dummy calibration reader."""
    model = onnx.load(str(inp))
    inferred = _relaxed_shape_infer(model)

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir) / "inferred.onnx"
        onnx.save(inferred, str(tmp))

        # Read input metadata from the model
        input_meta = inferred.graph.input[0]
        shape = [
            d.dim_value if d.dim_value > 0 else 1
            for d in input_meta.type.tensor_type.shape.dim
        ]
        input_name = input_meta.name
        cal_samples = int(config.get("calibration_samples", 8))

        reader = DummyCalibrationDataReader(
            input_name=input_name,
            input_shape=shape,
            samples=cal_samples,
        )

        quantize_static(
            model_input=str(tmp),
            model_output=str(out),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
        )


def _run_static_qdq_real(inp: Path, out: Path, config: Dict[str, Any]) -> dict:
    """Static QDQ quantisation with real image calibration.

    Expects the config to have a ``calibration`` section with
    ``image_dir``, ``max_samples``, ``mean``, and ``std``.
    Optionally supports ``quantization.preprocess: true`` to run
    ORT ``quant_pre_process`` before quantisation.
    """
    quant_cfg = config.get("quantization", {})
    preprocess_enabled = bool(quant_cfg.get("preprocess", False))
    fallback_to_original = bool(quant_cfg.get("fallback_to_original", True))

    preproc_status = "disabled"
    preproc_error = None
    skip_symbolic_shape_used = False
    skip_optimization_used = False

    if preprocess_enabled:
        from onnxruntime.quantization.shape_inference import quant_pre_process
        import tempfile

        # Determine model source: preprocessed or original
        with tempfile.TemporaryDirectory() as preproc_dir:
            preproc_path = Path(preproc_dir) / "preprocessed.onnx"
            success = False

            # Attempt 1: config-specified params
            try:
                kw = {}
                if quant_cfg.get("skip_symbolic_shape"):
                    kw["skip_symbolic_shape"] = True
                if quant_cfg.get("skip_optimization"):
                    kw["skip_optimization"] = True
                quant_pre_process(str(inp), str(preproc_path), **kw)
                success = True
                preproc_status = "success"
                skip_symbolic_shape_used = kw.get("skip_symbolic_shape", False)
                skip_optimization_used = kw.get("skip_optimization", False)
            except Exception as exc:
                preproc_error = str(exc)

            # Attempt 2: try skip_symbolic_shape=True
            if not success:
                try:
                    quant_pre_process(str(inp), str(preproc_path), skip_symbolic_shape=True)
                    success = True
                    preproc_status = "success"
                    skip_symbolic_shape_used = True
                    preproc_error = None
                except Exception as exc:
                    preproc_error = str(exc)

            # Attempt 3: try skip_symbolic_shape=True + skip_optimization=True
            if not success:
                try:
                    quant_pre_process(
                        str(inp), str(preproc_path),
                        skip_symbolic_shape=True, skip_optimization=True,
                    )
                    success = True
                    preproc_status = "success"
                    skip_symbolic_shape_used = True
                    skip_optimization_used = True
                    preproc_error = None
                except Exception as exc:
                    preproc_error = str(exc)

            if success:
                model = onnx.load(str(preproc_path))
            elif fallback_to_original:
                model = onnx.load(str(inp))
                preproc_status = "fallback"
            else:
                raise RuntimeError(
                    f"ORT quant_pre_process failed after all attempts. "
                    f"Last error: {preproc_error}"
                )
    else:
        model = onnx.load(str(inp))

    inferred = _relaxed_shape_infer(model)

    cal_cfg = config.get("calibration", {})
    image_dir = cal_cfg.get("image_dir", "samples/images/calibration_real")
    max_samples = int(cal_cfg.get("max_samples", 50))
    mean = cal_cfg.get("mean", [0.485, 0.456, 0.406])
    std = cal_cfg.get("std", [0.229, 0.224, 0.225])
    preprocess_mode = cal_cfg.get("preprocessing", "simple")
    resize_shorter = int(cal_cfg.get("resize_shorter", 256))

    input_meta = inferred.graph.input[0]
    input_name = input_meta.name
    target_size = (224, 224)

    reader = ImageCalibrationDataReader(
        image_dir=image_dir,
        input_name=input_name,
        target_size=target_size,
        mean=tuple(mean),
        std=tuple(std),
        max_samples=max_samples,
        preprocess_mode=preprocess_mode,
        resize_shorter=resize_shorter,
    )

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir) / "inferred.onnx"
        onnx.save(inferred, str(tmp))

        quantize_static(
            model_input=str(tmp),
            model_output=str(out),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
        )

    return {
        "calibration_type": "real_image",
        "preprocess_enabled": preprocess_enabled,
        "preprocess_status": preproc_status,
        "preprocess_error": preproc_error,
        "skip_symbolic_shape": skip_symbolic_shape_used,
        "skip_optimization": skip_optimization_used,
        "num_calibration_images": min(max_samples, len(reader._image_paths)),
        "input_model": str(inp),
        "output_model": str(out),
    }


def run_quantization(
    input_path: str | Path,
    output_path: str | Path,
    config: Optional[Dict[str, Any]] = None,
) -> Path:
    """Run ONNX quantisation (dynamic or static QDQ).

    Raises ``FileNotFoundError`` if the input model does not exist.
    Returns the resolved output path.
    """
    if config is None:
        config = {}

    inp = Path(input_path)
    if not inp.exists():
        raise FileNotFoundError(f"Input ONNX model not found: {inp.resolve()}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Detect calibration mode
    cal = config.get("calibration", {})
    if isinstance(cal, dict) and cal.get("image_dir"):
        # Real image calibration path
        method = "static_qdq_real"
    else:
        method = config.get("method", "dynamic")

    print(f"Quantizing (method={method}) ...")
    print(f"  input:  {inp.resolve()}")
    print(f"  output: {out.resolve()}")

    if method == "static_qdq_real":
        cal_report = _run_static_qdq_real(inp, out, config)
        print(f"  calibration: {cal_report['calibration_type']}")
        print(f"  images: {cal_report['num_calibration_images']}")
    elif method == "static_qdq":
        _run_static_qdq(inp, out, config)
    else:
        _run_dynamic(inp, out)

    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"Done.  Artifact size: {size_mb:.2f} MB")

    # Write calibration report
    if method == "static_qdq_real":
        report_path = out.with_suffix(".calibration.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump({**cal_report, "artifact_size_mb": round(size_mb, 2)}, f, indent=2)
        print(f"  calibration report: {report_path}")

    return out


# ── CLI entrypoint ──────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="ONNX quantisation")
    parser.add_argument(
        "--config",
        default="configs/quant_dynamic.yaml",
        help="Path to quantisation YAML config",
    )
    parser.add_argument("--input", help="Input ONNX model path (overrides config)")
    parser.add_argument("--output", help="Output ONNX path (overrides config)")
    parser.add_argument(
        "--method",
        choices=["dynamic", "static_qdq"],
        help="Quantisation method (overrides config)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config: Dict[str, Any] = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    if args.method:
        config["method"] = args.method

    # Support both flat and nested config formats
    model_cfg = config.get("model", config)
    input_model = (
        args.input
        or model_cfg.get("input_onnx")
        or model_cfg.get("input_model")
    )
    output_model = (
        args.output
        or model_cfg.get("output_onnx")
        or model_cfg.get("output_model")
    )

    if not input_model:
        print("ERROR: input_model is required", file=sys.stderr)
        sys.exit(1)
    if not output_model:
        print("ERROR: output_model is required", file=sys.stderr)
        sys.exit(1)

    try:
        run_quantization(input_model, output_model, config=config)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
