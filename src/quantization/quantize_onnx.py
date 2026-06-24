"""ONNX quantization script — dynamic and static QDQ (dummy calibration).

Usage
-----
.. code-block:: powershell

    # Dynamic quantization (experimental: ConvInteger may fail on CPU)
    python -m src.quantization.quantize_onnx --config configs/quant_dynamic.yaml

    # Static QDQ quantization with dummy calibration (recommended)
    python -m src.quantization.quantize_onnx --config configs/quant_static_qdq_dummy.yaml
"""

from __future__ import annotations

import argparse
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

    method = config.get("method", "dynamic")
    print(f"Quantizing (method={method}) ...")
    print(f"  input:  {inp.resolve()}")
    print(f"  output: {out.resolve()}")

    if method == "static_qdq":
        _run_static_qdq(inp, out, config)
    else:
        _run_dynamic(inp, out)

    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"Done.  Artifact size: {size_mb:.2f} MB")
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

    input_model = args.input or config.get("input_model")
    output_model = args.output or config.get("output_model")

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


if __name__ == "__main__":
    main()
