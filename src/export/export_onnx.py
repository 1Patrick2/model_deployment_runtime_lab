"""Export a torchvision model to ONNX.

Usage
-----
.. code-block:: powershell

    # Use the mdrl-train environment (torch / torchvision required)
    conda activate mdrl-train
    python -m src.export.export_onnx --model mobilenet_v3_small
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torchvision.models as models


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
    # Resolve model builder
    model_builder = getattr(models, model_name, None)
    if model_builder is None:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Expected a torchvision.models function, e.g. "
            f"mobilenet_v3_small, resnet18, etc."
        )

    # Load model (eval mode)
    print(f"Loading {model_name} (pretrained={pretrained}) ...")
    model = model_builder(pretrained=pretrained)
    model.eval()

    # Create dummy input
    dummy = torch.randn(*input_shape, requires_grad=False)

    # Ensure output directory exists
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Export
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
        "--model",
        default="mobilenet_v3_small",
        help="torchvision model function name (default: mobilenet_v3_small)",
    )
    parser.add_argument(
        "--output",
        default="outputs/onnx/mobilenetv3_small.onnx",
        help="Output ONNX path",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Load pretrained weights from torchvision",
    )
    args = parser.parse_args()

    export_onnx(
        model_name=args.model,
        output_path=args.output,
        pretrained=args.pretrained,
    )


if __name__ == "__main__":
    main()
