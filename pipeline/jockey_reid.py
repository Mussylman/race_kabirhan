"""
jockey_reid.py — Jockey Re-Identification via visual embeddings.

Instead of classifying color, this module:
  1. Takes 10-15 phone photos of each jockey as "gallery"
  2. Extracts visual embeddings (CLIP / DINOv2 / OSNet)
  3. At runtime: crop jockey from IP camera → extract embedding → match to gallery

This is fundamentally more robust than color classification because:
  - Works regardless of lighting conditions
  - Can distinguish jockeys with similar colors
  - Uses the full visual appearance (pattern, texture, body shape)
  - 15 reference photos is enough for reliable matching

Gallery structure:
    gallery/
    ├── jockey_1_Аслан/       # 10-15 phone photos
    │   ├── photo_01.jpg
    │   ├── photo_02.jpg
    │   └── ...
    ├── jockey_2_Марат/
    │   └── ...
    └── jockey_3_Болат/
        └── ...

Usage:
    reid = JockeyReID(gallery_dir="gallery/", backend="clip")

    # At runtime — pass cropped jockey images from YOLO
    results = reid.identify_batch(crops_bgr)
    # -> [("jockey_1_Аслан", 0.87, {...}), ("jockey_3_Болат", 0.92, {...}), ...]
"""

import logging
import os
import pickle
import numpy as np
from pathlib import Path
from typing import Optional

log = logging.getLogger("pipeline.jockey_reid")

# Similarity thresholds
MIN_MATCH_SIMILARITY = 0.50   # below this = "unknown jockey"
GALLERY_CACHE_FILE = "gallery_embeddings.pkl"


class JockeyReID:
    """Identify jockeys by matching against a gallery of reference photos.

    Supports multiple embedding backends:
      - "clip": OpenAI CLIP ViT-B/32 (best zero-shot, 512-dim)
      - "dinov2": Meta DINOv2 ViT-S/14 (best visual features, 384-dim)
      - "osnet": OSNet-AIN (best ReID-specific, 512-dim, needs torchreid)

    Workflow:
      1. build_gallery() — one-time: extract embeddings from phone photos
      2. identify_batch() — runtime: match camera crops to gallery
    """

    def __init__(
        self,
        gallery_dir: str = "gallery",
        backend: str = "clip",
        device: str = "cuda:0",
        min_similarity: float = MIN_MATCH_SIMILARITY,
        use_faiss: bool = True,
    ):
        self.gallery_dir = Path(gallery_dir)
        self.backend = backend
        self.device_str = device
        self.min_similarity = min_similarity
        self.use_faiss = use_faiss

        # Gallery data
        self.jockey_names: list[str] = []       # ordered list of jockey names
        self.gallery_embeddings: np.ndarray = None  # (N, dim) float32
        self.gallery_labels: np.ndarray = None      # (N,) int — index into jockey_names
        self._faiss_index = None

        # Model
        self._model = None
        self._processor = None
        self._transform = None

        self._load_model()

        # Auto-load gallery if it exists
        cache_path = self.gallery_dir / GALLERY_CACHE_FILE
        if cache_path.exists():
            self._load_gallery_cache(cache_path)
        elif self.gallery_dir.exists():
            self.build_gallery()

    # ── Model Loading ─────────────────────────────────────────────────

    def _load_model(self):
        import torch
        self.device = torch.device(
            self.device_str if torch.cuda.is_available() else "cpu"
        )

        if self.backend == "clip":
            self._load_clip()
        elif self.backend == "dinov2":
            self._load_dinov2()
        elif self.backend == "osnet":
            self._load_osnet()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        log.info("JockeyReID model loaded: backend=%s, device=%s", self.backend, self.device)

    def _load_clip(self):
        from transformers import CLIPModel, CLIPProcessor
        model_id = "openai/clip-vit-base-patch32"
        self._model = CLIPModel.from_pretrained(model_id).to(self.device)
        self._processor = CLIPProcessor.from_pretrained(model_id)
        self._model.eval()
        self._embed_dim = 512

    def _load_dinov2(self):
        import torch
        self._model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self._model = self._model.to(self.device)
        self._model.eval()
        self._embed_dim = 384

        from torchvision import transforms
        self._transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _load_osnet(self):
        """Load OSNet-AIN pretrained on person ReID datasets."""
        try:
            from torchreid.utils import FeatureExtractor
            self._model = FeatureExtractor(
                model_name="osnet_ain_x1_0",
                model_path="",  # uses default pretrained weights
                device=self.device_str,
            )
            self._embed_dim = 512
        except ImportError:
            raise ImportError(
                "OSNet requires: pip install torchreid\n"
                "  or: pip install git+https://github.com/KaiyangZhou/deep-person-reid.git"
            )

    # ── Feature Extraction ────────────────────────────────────────────

    def _extract_embeddings(self, images_bgr: list[np.ndarray]) -> np.ndarray:
        """Extract normalized embeddings from a list of BGR images.

        Returns:
            numpy array of shape (N, embed_dim), L2-normalized.
        """
        import torch
        import cv2
        from PIL import Image

        if not images_bgr:
            return np.zeros((0, self._embed_dim), dtype=np.float32)

        with torch.no_grad():
            if self.backend == "clip":
                pil_images = []
                for img in images_bgr:
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    pil_images.append(Image.fromarray(rgb))

                inputs = self._processor(
                    images=pil_images, return_tensors="pt", padding=True
                ).to(self.device)
                features = self._model.get_image_features(**inputs)

            elif self.backend == "dinov2":
                tensors = []
                for img in images_bgr:
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb)
                    tensors.append(self._transform(pil_img))
                batch = torch.stack(tensors).to(self.device)
                features = self._model(batch)

            elif self.backend == "osnet":
                # OSNet FeatureExtractor accepts file paths or PIL images
                pil_images = []
                for img in images_bgr:
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    pil_images.append(Image.fromarray(rgb))
                features = self._model(pil_images)
                if isinstance(features, np.ndarray):
                    features = torch.from_numpy(features)

        # L2 normalize
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype(np.float32)

    # ── Gallery Management ────────────────────────────────────────────

    def build_gallery(self):
        """Build gallery embeddings from phone photos in gallery_dir.

        Expected structure:
            gallery_dir/
            ├── jockey_name_1/   (10-15 .jpg photos)
            ├── jockey_name_2/
            └── ...
        """
        import cv2

        if not self.gallery_dir.exists():
            log.warning("Gallery directory not found: %s", self.gallery_dir)
            return

        jockey_dirs = sorted([
            d for d in self.gallery_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ])

        if not jockey_dirs:
            log.warning("No jockey folders found in %s", self.gallery_dir)
            return

        self.jockey_names = []
        all_embeddings = []
        all_labels = []

        for jockey_idx, jockey_dir in enumerate(jockey_dirs):
            jockey_name = jockey_dir.name
            self.jockey_names.append(jockey_name)

            # Load all photos
            photos = sorted([
                f for f in jockey_dir.iterdir()
                if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')
            ])

            if not photos:
                log.warning("No photos found for %s", jockey_name)
                continue

            images = []
            for photo_path in photos:
                img = cv2.imread(str(photo_path))
                if img is not None:
                    images.append(img)

            if not images:
                continue

            # Extract embeddings
            embeddings = self._extract_embeddings(images)
            all_embeddings.append(embeddings)
            all_labels.extend([jockey_idx] * len(embeddings))

            log.info("Gallery: %s — %d photos, %d embeddings",
                     jockey_name, len(photos), len(embeddings))

        if not all_embeddings:
            log.warning("Gallery is empty!")
            return

        self.gallery_embeddings = np.vstack(all_embeddings)
        self.gallery_labels = np.array(all_labels, dtype=np.int32)

        # Build FAISS index for fast search
        self._build_faiss_index()

        # Cache to disk
        self._save_gallery_cache(self.gallery_dir / GALLERY_CACHE_FILE)

        log.info("Gallery built: %d jockeys, %d total embeddings, dim=%d",
                 len(self.jockey_names), len(self.gallery_embeddings), self._embed_dim)

    def _build_faiss_index(self):
        """Build FAISS index for cosine similarity search."""
        if self.gallery_embeddings is None:
            return

        if self.use_faiss:
            try:
                import faiss
                # Inner product on L2-normalized vectors = cosine similarity
                self._faiss_index = faiss.IndexFlatIP(self._embed_dim)
                self._faiss_index.add(self.gallery_embeddings)
                log.info("FAISS index built: %d vectors", self._faiss_index.ntotal)
                return
            except ImportError:
                log.info("FAISS not available, using numpy fallback")

        self._faiss_index = None

    def _save_gallery_cache(self, path: Path):
        """Save gallery embeddings to disk for fast reload."""
        data = {
            'jockey_names': self.jockey_names,
            'embeddings': self.gallery_embeddings,
            'labels': self.gallery_labels,
            'backend': self.backend,
            'embed_dim': self._embed_dim,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        log.info("Gallery cached: %s", path)

    def _load_gallery_cache(self, path: Path):
        """Load gallery embeddings from cache."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        if data.get('backend') != self.backend:
            log.warning("Gallery cache backend mismatch (%s vs %s), rebuilding",
                        data.get('backend'), self.backend)
            self.build_gallery()
            return

        self.jockey_names = data['jockey_names']
        self.gallery_embeddings = data['embeddings']
        self.gallery_labels = data['labels']
        self._embed_dim = data.get('embed_dim', self.gallery_embeddings.shape[1])

        self._build_faiss_index()
        log.info("Gallery loaded from cache: %d jockeys, %d embeddings",
                 len(self.jockey_names), len(self.gallery_embeddings))

    # ── Identification ────────────────────────────────────────────────

    def identify_batch(
        self,
        crops_bgr: list[np.ndarray],
        top_k: int = 3,
    ) -> list[tuple[str, float, dict]]:
        """Identify jockeys in a batch of BGR crops.

        For each crop, finds the closest gallery match using cosine similarity.
        Uses majority voting across top-K gallery matches.

        Args:
            crops_bgr: list of BGR images (cropped jockeys from YOLO)
            top_k: number of gallery neighbors to consider

        Returns:
            list of (jockey_name, confidence, details_dict) tuples.
            If no match above threshold: ("unknown", 0.0, {})
        """
        if not crops_bgr or self.gallery_embeddings is None:
            return [("unknown", 0.0, {})] * len(crops_bgr)

        # Filter valid crops
        valid_crops = []
        valid_indices = []
        for i, crop in enumerate(crops_bgr):
            if crop is not None and crop.size > 0:
                valid_crops.append(crop)
                valid_indices.append(i)

        if not valid_crops:
            return [("unknown", 0.0, {})] * len(crops_bgr)

        # Extract query embeddings
        query_embeddings = self._extract_embeddings(valid_crops)

        # Search gallery
        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(query_embeddings, top_k)
        else:
            # Numpy fallback: cosine similarity via dot product (vectors are normalized)
            sims = query_embeddings @ self.gallery_embeddings.T  # (Q, G)
            indices = np.argsort(-sims, axis=1)[:, :top_k]
            scores = np.take_along_axis(sims, indices, axis=1)

        # Build results
        results = [("unknown", 0.0, {})] * len(crops_bgr)

        for batch_idx, orig_idx in enumerate(valid_indices):
            top_scores = scores[batch_idx]
            top_indices = indices[batch_idx]

            # Majority vote among top-K neighbors
            vote_counts: dict[int, float] = {}
            for score, gallery_idx in zip(top_scores, top_indices):
                if gallery_idx < 0:  # FAISS padding
                    continue
                label = int(self.gallery_labels[gallery_idx])
                vote_counts[label] = vote_counts.get(label, 0) + float(score)

            if not vote_counts:
                continue

            # Best match
            best_label = max(vote_counts, key=vote_counts.get)
            best_score = vote_counts[best_label] / top_k  # average similarity

            # Per-jockey similarity scores
            details = {}
            for label, total_score in sorted(vote_counts.items()):
                name = self.jockey_names[label]
                details[name] = round(total_score / top_k, 4)

            if best_score >= self.min_similarity:
                jockey_name = self.jockey_names[best_label]
                results[orig_idx] = (jockey_name, round(best_score, 4), details)
            else:
                results[orig_idx] = ("unknown", round(best_score, 4), details)

        return results

    def identify_single(self, crop_bgr: np.ndarray) -> tuple[str, float, dict]:
        """Identify a single jockey crop."""
        results = self.identify_batch([crop_bgr])
        return results[0]

    # ── Convenience ───────────────────────────────────────────────────

    def get_jockey_names(self) -> list[str]:
        """Return list of registered jockey names."""
        return list(self.jockey_names)

    def get_gallery_stats(self) -> dict:
        """Return gallery statistics."""
        if self.gallery_embeddings is None:
            return {"status": "empty"}

        stats = {
            "n_jockeys": len(self.jockey_names),
            "n_embeddings": len(self.gallery_embeddings),
            "embed_dim": self._embed_dim,
            "backend": self.backend,
            "jockeys": {},
        }
        for idx, name in enumerate(self.jockey_names):
            count = int((self.gallery_labels == idx).sum())
            stats["jockeys"][name] = count

        return stats
