"""Re-export color_classifier_v4 with ImageNet normalization BAKED IN as the
first layer of the graph.

This lets DeepStream's nvinfer feed raw uint8 pixels / 255.0 without the
lossy average-std approximation (previous SGIE config used
net-scale-factor=0.01735, offsets=[123.675,116.28,103.53] which smears
per-channel colours).

After export:
    - ONNX expects input in 0..1 RGB (already normalised)
    - SGIE config uses: net-scale-factor=0.00392157 (=1/255), offsets=0;0;0
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


class ColorCNNWithNorm(nn.Module):
    """Wraps SimpleColorCNN with ImageNet normalization baked into the graph."""

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    def __init__(self, base: SimpleColorCNN):
        super().__init__()
        self.base = base
        mean = torch.tensor(self.IMAGENET_MEAN).reshape(1, 3, 1, 1)
        std  = torch.tensor(self.IMAGENET_STD).reshape(1, 3, 1, 1)
        self.register_buffer("_mean", mean)
        self.register_buffer("_std", std)

    def forward(self, x):
        # x is RGB in 0..1
        x = (x - self._mean) / self._std
        return self.base(x)


def main():
    version = sys.argv[1] if len(sys.argv) > 1 else "v4"
    pt_path   = ROOT / "models" / f"color_classifier_{version}.pt"
    onnx_path = ROOT / "models" / f"color_classifier_{version}_norm.onnx"
    img_size  = 128 if version == "v4" else 64

    print(f"loading {pt_path}")
    saved = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    state = saved["model_state_dict"] if isinstance(saved, dict) and "model_state_dict" in saved else saved

    base = SimpleColorCNN(num_classes=5)
    base.load_state_dict(state)
    base.eval()
    wrapped = ColorCNNWithNorm(base).eval()

    # Sanity check: verify output matches manual normalisation
    import numpy as np
    rng = np.random.default_rng(0)
    raw = torch.from_numpy(rng.random((1, 3, img_size, img_size)).astype("float32"))
    with torch.no_grad():
        out_wrapped = wrapped(raw)
        # manual: subtract mean / divide std
        mean = torch.tensor(ColorCNNWithNorm.IMAGENET_MEAN).reshape(1, 3, 1, 1)
        std  = torch.tensor(ColorCNNWithNorm.IMAGENET_STD).reshape(1, 3, 1, 1)
        out_manual = base((raw - mean) / std)
    max_diff = (out_wrapped - out_manual).abs().max().item()
    print(f"  sanity diff (wrapped vs manual): {max_diff:.6e}")
    assert max_diff < 1e-5

    dummy = torch.randn(1, 3, img_size, img_size)
    print(f"exporting to {onnx_path}")
    torch.onnx.export(
        wrapped, dummy, str(onnx_path),
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    # Remove external data blob if torch wrote one
    for f in onnx_path.parent.glob(onnx_path.name + ".data*"):
        f.unlink()
    print(f"done: {onnx_path.stat().st_size / 1024:.1f} KB")
    print("\nSGIE config should use:")
    print(f"  onnx-file={onnx_path}")
    print("  net-scale-factor=0.00392157")
    print("  offsets=0;0;0")
    print("  model-color-format=0   # RGB")


if __name__ == "__main__":
    main()
