"""
clip_color_classifier.py — CLIP/SigLIP-based zero-shot color classifier.

Solves the main problems of the current EfficientNet classifier:
  - No training data needed (zero-shot)
  - Robust to lighting changes (pretrained on 400M+ image-text pairs)
  - Easy to add/change color classes via text prompts
  - Works as drop-in replacement for ColorClassifierInfer

Supports multiple backends:
  1. OpenAI CLIP (ViT-B/32) — widely tested, reliable
  2. SigLIP 2 ViT-B (86M params) — best accuracy/speed ratio
  3. OpenCLIP ViT-L/14 — maximum accuracy

Usage:
    classifier = CLIPColorClassifier(
        colors=["green", "red", "yellow"],
        backend="clip",  # or "siglip", "openclip"
    )
    results = classifier.classify_batch(crops_bgr)
    # -> [("green", 0.92, {"green": 0.92, "red": 0.05, "yellow": 0.03}), ...]
"""

import logging
import numpy as np
from typing import Optional

log = logging.getLogger("pipeline.clip_color")

# Prompt templates — tested for best accuracy on jockey torso crops
PROMPT_TEMPLATES = {
    "simple": "a {} colored uniform",
    "jockey": "a horse jockey wearing {} clothing",
    "torso": "torso of a person wearing {} clothes",
    "color_only": "{}",
}


class CLIPColorClassifier:
    """Zero-shot color classifier using CLIP or SigLIP vision-language models.

    This classifier requires NO training data — it uses pretrained knowledge
    of colors from massive internet-scale datasets. Color classes are defined
    as text prompts and compared against image embeddings.
    """

    def __init__(
        self,
        colors: list[str],
        backend: str = "clip",
        prompt_template: str = "jockey",
        device: str = "cuda:0",
        batch_size: int = 32,
    ):
        self.colors = colors
        self.backend = backend
        self.prompt_template = PROMPT_TEMPLATES.get(prompt_template, prompt_template)
        self.device_str = device
        self.batch_size = batch_size

        self._model = None
        self._processor = None
        self._text_features = None  # cached text embeddings

        self._load_model()

    def _load_model(self):
        """Load the vision-language model based on backend choice."""
        import torch
        self.device = torch.device(
            self.device_str if torch.cuda.is_available() else "cpu"
        )

        if self.backend == "clip":
            self._load_clip()
        elif self.backend == "siglip":
            self._load_siglip()
        elif self.backend == "openclip":
            self._load_openclip()
        else:
            raise ValueError(f"Unknown backend: {self.backend}. Use 'clip', 'siglip', or 'openclip'")

        # Pre-compute text embeddings for all color labels
        self._cache_text_features()
        log.info("CLIPColorClassifier ready: backend=%s, colors=%s, device=%s",
                 self.backend, self.colors, self.device)

    def _load_clip(self):
        """Load OpenAI CLIP model."""
        try:
            from transformers import CLIPModel, CLIPProcessor
            model_id = "openai/clip-vit-base-patch32"
            log.info("Loading CLIP: %s", model_id)
            self._model = CLIPModel.from_pretrained(model_id).to(self.device)
            self._processor = CLIPProcessor.from_pretrained(model_id)
            self._model.eval()
        except ImportError:
            raise ImportError(
                "CLIP requires: pip install transformers torch torchvision"
            )

    def _load_siglip(self):
        """Load Google SigLIP model (best accuracy/speed ratio)."""
        try:
            from transformers import AutoModel, AutoProcessor
            model_id = "google/siglip2-base-patch16-224"
            log.info("Loading SigLIP: %s", model_id)
            self._model = AutoModel.from_pretrained(model_id).to(self.device)
            self._processor = AutoProcessor.from_pretrained(model_id)
            self._model.eval()
        except ImportError:
            raise ImportError(
                "SigLIP requires: pip install transformers torch torchvision"
            )

    def _load_openclip(self):
        """Load OpenCLIP model (maximum accuracy)."""
        try:
            import open_clip
            model_name = "ViT-L-14"
            pretrained = "datacomp_xl_s13b_b90k"
            log.info("Loading OpenCLIP: %s (%s)", model_name, pretrained)
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=self.device
            )
            self._tokenizer = open_clip.get_tokenizer(model_name)
            self._model.eval()
        except ImportError:
            raise ImportError(
                "OpenCLIP requires: pip install open_clip_torch torch torchvision"
            )

    def _cache_text_features(self):
        """Pre-compute and cache text embeddings for all color labels."""
        import torch

        prompts = [self.prompt_template.format(color) for color in self.colors]
        log.info("Text prompts: %s", prompts)

        with torch.no_grad():
            if self.backend in ("clip", "siglip"):
                inputs = self._processor(
                    text=prompts,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)
                self._text_features = self._model.get_text_features(**inputs)
                self._text_features = self._text_features / self._text_features.norm(
                    dim=-1, keepdim=True
                )
            elif self.backend == "openclip":
                tokens = self._tokenizer(prompts).to(self.device)
                self._text_features = self._model.encode_text(tokens)
                self._text_features = self._text_features / self._text_features.norm(
                    dim=-1, keepdim=True
                )

    def _preprocess_crops(self, crops_bgr: list[np.ndarray]):
        """Convert BGR crops to model input tensors."""
        import torch
        import cv2
        from PIL import Image

        if self.backend in ("clip", "siglip"):
            # Convert BGR -> RGB PIL images
            pil_images = []
            for crop in crops_bgr:
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil_images.append(Image.fromarray(rgb))

            inputs = self._processor(
                images=pil_images,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            return inputs

        elif self.backend == "openclip":
            tensors = []
            for crop in crops_bgr:
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                tensors.append(self._preprocess(pil_img))
            return torch.stack(tensors).to(self.device)

    def classify_batch(
        self,
        crops: list[np.ndarray],
    ) -> list[tuple[str, float, dict]]:
        """Classify a batch of BGR crops by color.

        Args:
            crops: list of BGR numpy arrays (torso crops).

        Returns:
            list of (color_name, confidence, prob_dict) tuples.
        """
        import torch

        if not crops:
            return []

        results = [("unknown", 0.0, {})] * len(crops)

        # Filter valid crops
        valid_crops = []
        valid_indices = []
        for i, crop in enumerate(crops):
            if crop is not None and crop.size > 0:
                valid_crops.append(crop)
                valid_indices.append(i)

        if not valid_crops:
            return results

        # Process in batches
        for batch_start in range(0, len(valid_crops), self.batch_size):
            batch_crops = valid_crops[batch_start:batch_start + self.batch_size]
            batch_indices = valid_indices[batch_start:batch_start + self.batch_size]

            with torch.no_grad():
                if self.backend in ("clip", "siglip"):
                    inputs = self._preprocess_crops(batch_crops)
                    image_features = self._model.get_image_features(**inputs)
                    image_features = image_features / image_features.norm(
                        dim=-1, keepdim=True
                    )
                elif self.backend == "openclip":
                    batch_tensor = self._preprocess_crops(batch_crops)
                    image_features = self._model.encode_image(batch_tensor)
                    image_features = image_features / image_features.norm(
                        dim=-1, keepdim=True
                    )

                # Compute similarity scores
                similarity = image_features @ self._text_features.T

                # Convert to probabilities
                # Temperature scaling (100 is CLIP's default logit_scale)
                probs = torch.softmax(similarity * 100.0, dim=-1)

            for batch_idx, orig_idx in enumerate(batch_indices):
                p = probs[batch_idx]
                prob_dict = {
                    self.colors[j]: round(p[j].item(), 4)
                    for j in range(len(self.colors))
                }
                best_idx = p.argmax().item()
                color = self.colors[best_idx]
                conf = p[best_idx].item()
                results[orig_idx] = (color, conf, prob_dict)

        return results


class EnsembleColorClassifier:
    """Ensemble: combines CLIP zero-shot with HSV heuristics for robustness.

    Uses CLIP as primary classifier, HSV as fallback/tiebreaker.
    This gives the best of both worlds:
      - CLIP: semantic understanding of colors (handles shadows, lighting)
      - HSV: fast pixel-level color detection (handles clear, well-lit cases)
    """

    def __init__(
        self,
        colors: list[str],
        clip_weight: float = 0.7,
        hsv_weight: float = 0.3,
        backend: str = "clip",
        device: str = "cuda:0",
    ):
        self.colors = colors
        self.clip_weight = clip_weight
        self.hsv_weight = hsv_weight

        self._clip = CLIPColorClassifier(
            colors=colors,
            backend=backend,
            device=device,
        )

    def classify_batch(
        self,
        crops: list[np.ndarray],
    ) -> list[tuple[str, float, dict]]:
        """Classify using ensemble of CLIP + HSV."""
        clip_results = self._clip.classify_batch(crops)

        results = []
        for i, crop in enumerate(crops):
            clip_color, clip_conf, clip_probs = clip_results[i]

            if crop is None or crop.size == 0:
                results.append(("unknown", 0.0, {}))
                continue

            # HSV analysis
            hsv_probs = _hsv_color_probs(crop, self.colors)

            # Weighted ensemble
            combined_probs = {}
            for color in self.colors:
                combined_probs[color] = round(
                    self.clip_weight * clip_probs.get(color, 0.0)
                    + self.hsv_weight * hsv_probs.get(color, 0.0),
                    4,
                )

            best_color = max(combined_probs, key=combined_probs.get)
            best_conf = combined_probs[best_color]
            results.append((best_color, best_conf, combined_probs))

        return results


def _hsv_color_probs(crop_bgr: np.ndarray, colors: list[str]) -> dict[str, float]:
    """Convert HSV analysis into soft probability distribution over colors.

    Instead of hard hue ranges, uses Gaussian-like soft assignments
    for smoother transitions between color boundaries.
    """
    import cv2

    if crop_bgr is None or crop_bgr.size < 100:
        return {c: 1.0 / len(colors) for c in colors}

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Filter out low-saturation (gray/white/black) pixels
    mask = (s > 40) & (v > 50)
    if mask.sum() < 30:
        return {c: 1.0 / len(colors) for c in colors}

    h_filtered = h[mask].astype(np.float32)

    # Color center hues (in OpenCV 0-180 range) and widths
    color_hue_centers = {
        "red": 5,       # wraps around 0/180
        "yellow": 28,
        "green": 60,
        "blue": 105,
        "purple": 145,
    }
    color_hue_sigma = {
        "red": 12,
        "yellow": 12,
        "green": 20,
        "blue": 20,
        "purple": 18,
    }

    # Compute soft assignment for each pixel to each color
    scores = {}
    for color in colors:
        if color not in color_hue_centers:
            scores[color] = 0.0
            continue

        center = color_hue_centers[color]
        sigma = color_hue_sigma[color]

        if color == "red":
            # Red wraps around 0/180
            diff = np.minimum(
                np.abs(h_filtered - center),
                np.minimum(h_filtered, 180 - h_filtered)
            )
        else:
            diff = np.abs(h_filtered - center)

        # Gaussian-like score per pixel
        pixel_scores = np.exp(-0.5 * (diff / sigma) ** 2)
        scores[color] = float(pixel_scores.mean())

    # Normalize to probabilities
    total = sum(scores.values())
    if total < 1e-6:
        return {c: 1.0 / len(colors) for c in colors}

    return {c: round(scores[c] / total, 4) for c in colors}
