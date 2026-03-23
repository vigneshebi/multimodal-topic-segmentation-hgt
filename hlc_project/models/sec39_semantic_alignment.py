"""
sec39_semantic_alignment.py
Section 3.9 — Shared Semantic Space Alignment

Eq. 22 : s_i = W_{m_i} · e_i + b_{m_i}
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List

from data_structures import InstructionalUnit, MODALITIES


class SemanticSpaceAligner(nn.Module):
    """
    Projects modality-specific embeddings into a unified shared
    semantic space so that cross-modal comparison and interaction
    become possible.

    Each modality m has its own learnable linear projection:
        s_i = W_{m_i} · e_i + b_{m_i}   (Eq. 22)

    After alignment:
    - Semantically similar units (across modalities) are closer together.
    - Temporal ordering is preserved in the projected space.
    - Output serves as input to the HGT fusion stage (Section 3.10).

    Parameters
    ----------
    modality_dim : int — dimension of modality-specific embeddings e_i
    shared_dim   : int — dimension of shared semantic space d
    """

    def __init__(self, modality_dim: int = 256, shared_dim: int = 256):
        super().__init__()
        self.shared_dim = shared_dim

        # One linear projector W_{m_i}, b_{m_i} per modality
        self.projectors = nn.ModuleDict({
            m: nn.Linear(modality_dim, shared_dim)
            for m in MODALITIES
        })

    def align(self, embedding: np.ndarray, modality: str) -> np.ndarray:
        """
        Project one embedding into the shared space.

        Returns
        -------
        s_i ∈ R^d  (Eq. 22)
        """
        x    = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
        proj = self.projectors[modality]
        with torch.no_grad():
            return proj(x).squeeze(0).numpy()

    def align_all(self,
                  embeddings: List[np.ndarray],
                  units: List[InstructionalUnit]) -> List[np.ndarray]:
        """
        Align the full embedding set E → S.

        Returns
        -------
        S = {s_1, …, s_N}
        """
        assert len(embeddings) == len(units), \
            "Embeddings and units must have the same length."
        aligned = [self.align(e, u.modality)
                   for e, u in zip(embeddings, units)]
        print(f"[Aligner] Aligned {len(aligned)} embeddings "
              f"→ shared dim={self.shared_dim}")
        return aligned


if __name__ == "__main__":
    rng        = np.random.default_rng(3)
    modalities = list(MODALITIES)
    units      = [
        InstructionalUnit(
            content=rng.random(256).astype(np.float32),
            timestamp=float(i),
            modality=modalities[i % 4]
        )
        for i in range(6)
    ]
    embeddings = [u.content for u in units]   # use content as dummy e_i

    aligner = SemanticSpaceAligner(modality_dim=256, shared_dim=256)
    aligned = aligner.align_all(embeddings, units)
    for i, s in enumerate(aligned):
        print(f"  s_{i+1}: shape={s.shape}, modality={units[i].modality}")
