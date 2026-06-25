"""ONNX to RKNN model conversion.

Usage
-----
.. code-block:: bash

    # Must run in the WSL rknn-env environment
    source ~/mdrl-rknn-workdir/rknn-env/bin/activate
    python -m src.rknn.convert --config configs/rknn_convert.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from src.benchmark.artifact import compute_onnx_artifact_size_mb
from src.rknn.report import (
    write_rknn_conversion_json,
    write_rknn_conversion_markdown,
)


def _import_rknn_toolkit():
    """Try to import the RKNN toolkit; return the module or None."""
    try:
        from rknn.api import RKNN  # noqa: F401

        return __import__("rknn.api", fromlist=["RKNN"])
    except ImportError:
        return None


def convert_onnx_to_rknn(config: dict) -> dict:
    """Convert an ONNX model to RKNN format.

    Returns a conversion report dict.  If the RKNN toolkit is not
    installed the report will contain ``status=error`` with a clear
    message — no traceback.
    """
    input_model = Path(config["input_model"])
    output_model = Path(config["output_model"])
    target_platform = config.get("target_platform", "rk3588")
    do_quantization = config.get("do_quantization", False)
    dataset = config.get("dataset", None)
    mean_values = config.get("mean_values", [[123.675, 116.28, 103.53]])
    std_values = config.get("std_values", [[58.395, 57.12, 57.375]])
    quantized_dtype = config.get("quantized_dtype", "asymmetric_quantized-u8")

    if not input_model.exists():
        return {
            "source_model": str(input_model),
            "output_model": str(output_model),
            "target_platform": target_platform,
            "do_quantization": do_quantization,
            "onnx_size_mb": 0.0,
            "rknn_size_mb": 0.0,
            "status": "error",
            "message": f"Input ONNX model not found: {input_model.resolve()}",
        }

    rknn_mod = _import_rknn_toolkit()
    if rknn_mod is None:
        try:
            onnx_size = compute_onnx_artifact_size_mb(input_model)
        except Exception:
            onnx_size = input_model.stat().st_size / (1024 * 1024)
        return {
            "source_model": str(input_model),
            "output_model": str(output_model),
            "target_platform": target_platform,
            "do_quantization": do_quantization,
            "onnx_size_mb": round(onnx_size, 2),
            "rknn_size_mb": 0.0,
            "status": "error",
            "message": (
                "RKNN Toolkit2 is not installed. "
                "Run this command in WSL rknn-env created by setup_wsl.sh."
            ),
        }

    output_model.parent.mkdir(parents=True, exist_ok=True)
    onnx_size = compute_onnx_artifact_size_mb(input_model)

    try:
        rknn = rknn_mod.RKNN(verbose=True)
        rknn.config(
            target_platform=target_platform,
            mean_values=mean_values,
            std_values=std_values,
            quantized_dtype=quantized_dtype,
        )

        rknn.load_onnx(model=str(input_model))
        rknn.build(do_quantization=do_quantization, dataset=dataset)
        rknn.export_rknn(str(output_model))
        rknn.release()

        rknn_size = output_model.stat().st_size / (1024 * 1024) if output_model.exists() else 0.0

        return {
            "source_model": str(input_model),
            "output_model": str(output_model),
            "target_platform": target_platform,
            "do_quantization": do_quantization,
            "onnx_size_mb": round(onnx_size, 2),
            "rknn_size_mb": round(rknn_size, 2),
            "status": "ok",
            "message": "Conversion completed successfully.",
        }

    except Exception as exc:
        return {
            "source_model": str(input_model),
            "output_model": str(output_model),
            "target_platform": target_platform,
            "do_quantization": do_quantization,
            "onnx_size_mb": round(onnx_size, 2),
            "rknn_size_mb": 0.0,
            "status": "error",
            "message": str(exc),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="ONNX to RKNN conversion")
    parser.add_argument(
        "--config",
        default="configs/rknn_convert.yaml",
        help="Path to RKNN conversion YAML config",
    )
    parser.add_argument(
        "--output-json",
        help="Conversion report JSON path (overrides config default)",
    )
    parser.add_argument(
        "--output-md",
        help="Conversion report Markdown path (overrides config default)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    report = convert_onnx_to_rknn(config)

    out_json = args.output_json or config.get(
        "output_json",
        f"outputs/reports/rknn_convert_{Path(config.get('output_model', 'model')).stem}.json",
    )
    out_md = args.output_md or config.get(
        "output_md",
        f"outputs/reports/rknn_convert_{Path(config.get('output_model', 'model')).stem}.md",
    )

    write_rknn_conversion_json(report, out_json)
    write_rknn_conversion_markdown(report, out_md)

    print(f"RKNN conversion report written:")
    print(f"  JSON: {out_json}")
    print(f"  MD:   {out_md}")
    print(f"  status: {report['status']}")
    print(f"  message: {report['message']}")


if __name__ == "__main__":
    main()
