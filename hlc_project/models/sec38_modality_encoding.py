"""
sec38_modality_encoding.py
Section 3.8 — Modality-Specific Representation Learning

Eq. 20 : e_i = f_{m_i}(c̃_i)
Eq. 21 : E = {e_1, …, e_N}

Production model mapping:
    text      → Sentence-BERT  (SBERT)
    equation  → MathBERT
    table     → TAPAS
    diagram   → Vision Transformer (ViT)

Stub encoders (nn.Sequential) are used here for portability.
Replace each stub with the corresponding pre-trained model.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List

from data_structures import InstructionalUnit, MODALITIES


class ModalityEncoder(nn.Module):
    """
    Applies a dedicated neural encoder f_{m_i} to each instructional unit
    based on its modality label m_i.

    Each unit is encoded independently, preserving modality-specific
    semantics prior to cross-modal interaction.

    Parameters
    ----------
    input_dim  : int — dimension of the preprocessed content vector c̃_i
    hidden_dim : int — output embedding dimension d_m
    """

    def __init__(self, input_dim: int = 772, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Stub encoders — replace each with the corresponding model:
        #   "text"     → SentenceTransformer('all-MiniLM-L6-v2')
        #   "equation" → AutoModel.from_pretrained('witiko/mathberta')
        #   "table"    → TapasModel.from_pretrained('google/tapas-base')
        #   "diagram"  → ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.encoders = nn.ModuleDict({
            m: nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            for m in MODALITIES
        })

    def encode(self, unit: InstructionalUnit) -> np.ndarray:
        """
        Encode one instructional unit.

        Returns
        -------
        e_i ∈ R^{d_m}  (Eq. 20)
        """
        x   = torch.tensor(unit.content, dtype=torch.float32).unsqueeze(0)
        enc = self.encoders[unit.modality]
        with torch.no_grad():
            return enc(x).squeeze(0).numpy()

    def encode_all(self,
                   units: List[InstructionalUnit]) -> List[np.ndarray]:
        """
        Encode the full set of preprocessed units.

        Returns
        -------
        E = {e_1, …, e_N}  (Eq. 21)
        """
        embeddings = [self.encode(u) for u in units]
        print(f"[ModalityEncoder] Encoded {len(embeddings)} units "
              f"→ dim={self.hidden_dim}")
        return embeddings


if __name__ == "__main__":
    rng   = np.random.default_rng(2)
    units = [
        InstructionalUnit(
            content=rng.random(772).astype(np.float32),
            timestamp=float(i),
            modality=list(MODALITIES)[i % 4]
        )
        for i in range(6)
    ]
    encoder     = ModalityEncoder(input_dim=772, hidden_dim=256)
    embeddings  = encoder.encode_all(units)
    for i, (u, e) in enumerate(zip(units, embeddings)):
        print(f"  e_{i+1}: modality={u.modality}, shape={e.shape}")
