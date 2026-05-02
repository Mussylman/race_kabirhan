"""Re-export color classifier .pt → clean .onnx for DeepStream SGIE.

SGIE sometimes silently refuses old ONNXs with dynamic batch ranges
it cannot profile. This script rebuilds the ONNX from the .pt weights
with a fixed opset and an explicit, clean dynamic batch dim.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent


class SimpleColorCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def main(version: str):
    pt_path   = ROOT / "models" / f"color_classifier_{version}.pt"
    onnx_path = ROOT / "models" / f"color_classifier_{version}.onnx"
    img_size  = 128 if version == "v4" else 64

    if not pt_path.exists():
        sys.exit(f"missing: {pt_path}")

    print(f"loading {pt_path}")
    saved = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    if isinstance(saved, dict) and "model_state_dict" in saved:
        state = saved["model_state_dict"]
        print(f"  checkpoint keys: {list(saved.keys())}")
    elif hasattr(saved, "state_dict"):
        state = saved.state_dict()
    else:
        state = saved

    model = SimpleColorCNN(num_classes=5)
    model.load_state_dict(state)
    model.eval()
    print(f"  model loaded, params: {sum(p.numel() for p in model.parameters())}")

    # Dummy input with the trained image size
    dummy = torch.randn(1, 3, img_size, img_size)

    # Quick sanity: run inference
    with torch.no_grad():
        out = model(dummy)
    print(f"  forward test: input {tuple(dummy.shape)} -> output {tuple(out.shape)}")

    # Backup old onnx
    if onnx_path.exists():
        backup = onnx_path.with_suffix(".onnx.old")
        print(f"  backing up existing -> {backup.name}")
        onnx_path.rename(backup)

    print(f"exporting to {onnx_path}")
    # dynamo=False uses the legacy TorchScript exporter which keeps weights
    # INSIDE the .onnx file (no external .data). DeepStream expects this.
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    # Clean up any leftover external data file
    data_file = Path(str(onnx_path) + ".data")
    if data_file.exists():
        data_file.unlink()
    print(f"done: {onnx_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    v = sys.argv[1] if len(sys.argv) > 1 else "v4"
    main(v)
