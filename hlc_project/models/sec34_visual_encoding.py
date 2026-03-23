"""
sec34_visual_encoding.py
Section 3.4 — Visual Feature Encoding (ViT-B/16)

Eq. 6 : v_i = f_ViT(e_i)
Eq. 7 : ṽ_i = [v_i ‖ p_i]
Eq. 8 : V = {ṽ_1, …, ṽ_M}
"""

import math
import numpy as np
import cv2
import torch
import torch.nn as nn
from typing import List

from data_structures import DetectedElement


class ViTEncoder:
    """
    Encodes each instructional element into a spatially-augmented
    visual feature vector using a Vision Transformer (ViT-B/16).

    Pipeline per element:
        1. Crop & resize ROI to img_size × img_size
        2. Split into fixed-size patches → patch embeddings
        3. Add positional encodings → CLS token prepended
        4. Pass through Transformer blocks
        5. Extract CLS token as v_i  (Eq. 6)
        6. Concatenate spatial metadata p_i → ṽ_i  (Eq. 7)

    Parameters
    ----------
    embed_dim  : int   — Transformer hidden dimension (default 768 for ViT-B)
    patch_size : int   — Spatial patch size in pixels
    img_size   : int   — Input resolution after resize
    """

    def __init__(self, embed_dim: int = 768,
                 patch_size: int = 16,
                 img_size: int = 224):
        self.embed_dim  = embed_dim
        self.patch_size = patch_size
        self.img_size   = img_size
        self.spatial_dim = 4                         # (x, y, w, h)
        self.output_dim  = embed_dim + self.spatial_dim

        n_patches = (img_size // patch_size) ** 2
        patch_vec = 3 * patch_size * patch_size

        # Lightweight ViT (production: load timm ViT-B/16 weights)
        self._patch_embed = nn.Linear(patch_vec, embed_dim)
        self._pos_embed   = nn.Parameter(
            torch.zeros(1, n_patches + 1, embed_dim))
        enc_layer         = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, batch_first=True)
        self._transformer = nn.TransformerEncoder(enc_layer, num_layers=6)
        self._cls_token   = nn.Parameter(torch.zeros(1, 1, embed_dim))

    @torch.no_grad()
    def encode(self, element: DetectedElement,
               frame: np.ndarray) -> np.ndarray:
        """
        Encode one element.

        Returns
        -------
        ṽ_i ∈ R^(d+4)  (Eq. 7)
        """
        v_i = self._extract_patch_features(element, frame)   # Eq. 6
        p_i = np.array(
            [element.region.x, element.region.y,
             element.region.w, element.region.h],
            dtype=np.float32
        )
        return np.concatenate([v_i, p_i])                    # Eq. 7 — ṽ_i

    def encode_set(self, elements: List[DetectedElement],
                   frame: np.ndarray) -> List[np.ndarray]:
        """
        Encode all elements in a frame.

        Returns
        -------
        V = {ṽ_1, …, ṽ_M}  (Eq. 8)
        """
        return [self.encode(e, frame) for e in elements]

    # ------------------------------------------------------------------
    def _extract_patch_features(self, element: DetectedElement,
                                 frame: np.ndarray) -> np.ndarray:
        """Internal: crop ROI, patchify, run transformer, return CLS token."""
        h, w = frame.shape[:2]
        bb   = element.region
        x1   = int(bb.x * w);   y1 = int(bb.y * h)
        x2   = int((bb.x + bb.w) * w)
        y2   = int((bb.y + bb.h) * h)
        roi  = frame[max(0, y1):max(1, y2), max(0, x1):max(1, x2)]

        if roi.size == 0:
            roi = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        roi = cv2.resize(roi, (self.img_size, self.img_size))

        # Build patch tensor
        p = self.patch_size
        patches = []
        for i in range(0, self.img_size, p):
            for j in range(0, self.img_size, p):
                patch = roi[i:i+p, j:j+p].astype(np.float32) / 255.0
                patches.append(patch.flatten())

        patch_t = torch.tensor(np.stack(patches), dtype=torch.float32)
        embeds  = self._patch_embed(patch_t).unsqueeze(0)       # (1, N, D)

        cls = self._cls_token.expand(1, -1, -1)
        x   = torch.cat([cls, embeds], dim=1) + self._pos_embed
        out = self._transformer(x)                              # (1, N+1, D)
        return out[0, 0].numpy()                                # CLS → v_i


if __name__ == "__main__":
    from data_structures import BoundingBox, DetectedElement

    frame   = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    elem    = DetectedElement(
        region=BoundingBox(0.1, 0.1, 0.3, 0.2), modality="text")
    encoder = ViTEncoder(embed_dim=768)
    vec     = encoder.encode(elem, frame)
    print(f"ṽ_i shape: {vec.shape}  (expected {encoder.output_dim})")
