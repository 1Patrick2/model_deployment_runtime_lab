"""ONNX dynamic quantization script.

Usage
-----
.. code-block:: powershell

    python -m src.quantization.quantize_onnx --config configs/quant_dynamic.yaml
    python -m src.quantization.quantize_onnx --input outputs/onnx/mobilenetv3_small.onnx --output outputs/onnx/mobilenetv3_small_int8_dynamic.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import onnx
import yaml
from onnxruntime.quantization import quantize_dynamic, QuantType


def _preprocess_for_quantization(input_path: Path) -> Path:
    """Run shape inference with relaxed options.

    The ONNX Runtime ``quantize_dynamic`` internally calls
    ``infer_shapes_path`` with default (strict) options, which can
    fail on models exported via newer PyTorch dynamo exporters.
    Preprocessing with ``strict_mode=False, check_type=False`` avoids
    this.
    """
    model = onnx.load(str(input_path), load_external_data=True)
    inferred = onnx.shape_inference.infer_shapes(
        model, strict_mode=False, check_type=False
    )
    tmp = Path(tempfile.mktemp(suffix=".onnx"))
    onnx.save(inferred, str(tmp))
    return tmp


def run_quantization(input_path: str | Path, output_path: str | Path) -> Path:
    """Run ONNX dynamic quantization.

    Raises ``FileNotFoundError`` if the input model does not exist.
    Returns the resolved output path.
    """
    inp = Path(input_path)
    if not inp.exists():
        raise FileNotFoundError(f"Input ONNX model not found: {inp.resolve()}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Quantizing (dynamic, QInt8) ...")
    print(f"  input:  {inp.resolve()}")
    print(f"  output: {out.resolve()}")

    # Load model, strip conflicting shapes, then quantize
    model = onnx.load(str(inp))
    # Remove any pre-existing inferred shape info that may conflict
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                _clear_value_info(attr.g)
    _clear_value_info(model.graph)

    # Re-infer shapes with relaxed settings for quantization compatibility
    inferred = onnx.shape_inference.infer_shapes(
        model, strict_mode=False, check_type=False
    )

    quantize_dynamic(
        model_input=inferred,
        model_output=str(out),
        weight_type=QuantType.QInt8,
    )

    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"Done.  Artifact size: {size_mb:.2f} MB")
    return out


def _clear_value_info(graph) -> None:
    """Remove existing value_info entries that may contain conflicting shape info."""
    while graph.value_info:
        graph.value_info.pop()


def main() -> None:
    parser = argparse.ArgumentParser(description="ONNX dynamic quantization")
    parser.add_argument(
        "--config",
        default="configs/quant_dynamic.yaml",
        help="Path to quantization YAML config",
    )
    parser.add_argument("--input", help="Input ONNX model path (overrides config)")
    parser.add_argument("--output", help="Output ONNX path (overrides config)")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    config = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    input_model = args.input or config.get("input_model")
    output_model = args.output or config.get("output_model")

    if not input_model:
        print("ERROR: input_model is required (set in config or via --input)", file=sys.stderr)
        sys.exit(1)
    if not output_model:
        print("ERROR: output_model is required (set in config or via --output)", file=sys.stderr)
        sys.exit(1)

    try:
        run_quantization(input_model, output_model)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
