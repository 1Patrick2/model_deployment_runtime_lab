"""Export a torchvision model to ONNX.

Usage
-----
.. code-block:: powershell

    # Use the mdrl-train environment (torch / torchvision required)
    conda activate mdrl-train

    # CLI mode
    python -m src.export.export_onnx --model mobilenet_v3_small --pretrained

    # Config-driven mode
    python -m src.export.export_onnx --config configs/export_mobilenetv3_small_imagenet.yaml
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional


def _load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a YAML export config."""
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def export_onnx(
    model_name: str,
    output_path: str | Path,
    pretrained: bool = False,
    input_shape: tuple[int, ...] = (1, 3, 224, 224),
    input_name: str = "input",
    output_name: str = "logits",
    opset_version: int = 17,
) -> Path:
    """Load a torchvision model and export it to ONNX.

    Returns the resolved output path.
    """
    import torch
    import torchvision.models as models

    model_builder = getattr(models, model_name, None)
    if model_builder is None:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Expected a torchvision.models function, e.g. "
            f"mobilenet_v3_small, resnet18, etc."
        )

    print(f"Loading {model_name} (pretrained={pretrained}) ...")
    model = model_builder(pretrained=pretrained)
    model.eval()

    dummy = torch.randn(*input_shape, requires_grad=False)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting to {output}  (opset={opset_version}) ...")
    torch.onnx.export(
        model,
        dummy,
        str(output),
        input_names=[input_name],
        output_names=[output_name],
        opset_version=opset_version,
        dynamic_axes={
            input_name: {0: "batch_size"},
            output_name: {0: "batch_size"},
        },
    )

    size_mb = os.path.getsize(output) / (1024 * 1024)
    print(f"Done.  Artifact size: {size_mb:.2f} MB")
    print(f"  path: {output.resolve()}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Export torchvision model to ONNX")
    parser.add_argument(
        "--config",
        help="Path to YAML export config (overrides individual CLI args)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="torchvision model function name (default: mobilenet_v3_small)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output ONNX path",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=None,
        help="Load pretrained weights from torchvision",
    )
    args = parser.parse_args()

    # Load config if provided
    config: Dict[str, Any] = {}
    if args.config:
        config = _load_config(args.config)

    # CLI args override config values
    model_name = args.model or config.get("model", "mobilenet_v3_small")
    output_path = args.output or config.get("output", "outputs/onnx/mobilenetv3_small.onnx")
    pretrained = args.pretrained if args.pretrained is not None else config.get("pretrained", False)
    input_shape = tuple(config.get("input_shape", [1, 3, 224, 224]))
    input_name = config.get("input_name", "input")
    output_name = config.get("output_name", "logits")
    opset_version = int(config.get("opset_version", 17))

    export_onnx(
        model_name=model_name,
        output_path=output_path,
        pretrained=pretrained,
        input_shape=input_shape,
        input_name=input_name,
        output_name=output_name,
        opset_version=opset_version,
    )


if __name__ == "__main__":
    main()
