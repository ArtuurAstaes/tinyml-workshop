"""
export_onnx.py — Export the trained model to ONNX with full shape inference.

Loads the full-precision baseline model and exports it to ONNX using the
legacy TorchScript-based exporter (dynamo=False), which correctly stores
weights as initializers and allows shape inference to populate all operator
attributes — including kernel_shape on Conv nodes.

Key ideas for the workshop:
  - torch.onnx.export with dynamo=False uses tracing: it runs the model once
    with a dummy input and records the computation graph.
  - Shape inference (onnx.shape_inference.infer_shapes) backfills attributes
    like kernel_shape on Conv nodes from the weight initializer's shape.
  - dynamic_axes allows the batch dimension to vary at runtime despite the
    fixed-size dummy input used during tracing.
  - Inspect the result in Netron (https://netron.app) to see all attributes.

Usage:
    python export_onnx.py
"""

import torch
import onnx
from onnx import shape_inference
from pathlib import Path

# Suppress warnings about quantization and deprecation in ONNX export
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from model import CNN


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LOAD_PATH = Path("./models/cnn.pth")
SAVE_PATH = Path("./onnx/cnn.onnx")
OPSET_VERSION = 13


def main():
    device = torch.device("cpu")

    model = CNN()
    model.load_state_dict(torch.load(LOAD_PATH, map_location=device))
    model.eval()

    # Dummy input to trace the graph — batch size 1, NCHW format
    dummy_input = torch.zeros(1, 1, 28, 28)

    SAVE_PATH.parent.mkdir(exist_ok=True)  # Ensure 'onnx' directory exists

    print("Exporting to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        SAVE_PATH,
        input_names=["image_input"],
        output_names=["class_output"],
        dynamic_axes={
            "image_input":   {0: "batch"},
            "class_output":  {0: "batch"},
        },
        opset_version=OPSET_VERSION,
        dynamo=False,   # Use legacy tracer — correctly stores weights as initializers
    )

    # Run shape inference to populate operator attributes from the weight
    # initializers. Most importantly, this adds kernel_shape to Conv nodes.
    print("Running shape inference...")
    model_onnx = onnx.load(SAVE_PATH)
    model_onnx = shape_inference.infer_shapes(model_onnx)
    onnx.save(model_onnx, SAVE_PATH)

    print(f"\nExported to '{SAVE_PATH}' (opset {OPSET_VERSION})")
    print("Open in Netron (https://netron.app) to inspect the full graph.")


if __name__ == "__main__":
    main()
