"""
algo2_multimodal_representation.py
Algorithm 2 — Multimodal Representation Learning  (Sections 3.7 – 3.9)

Combines:
    ModalityAwarePreprocessor  (sec37)
    ModalityEncoder            (sec38)
    SemanticSpaceAligner       (sec39)
"""

import numpy as np
from typing import List

from data_structures import InstructionalUnit
from sec37_preprocessing import ModalityAwarePreprocessor
from sec38_modality_encoding import ModalityEncoder
from sec39_semantic_alignment import SemanticSpaceAligner


class MultimodalRepresentationLearner:
    """
    Algorithm 2: MULTIMODAL_REPRESENTATION(U) → S

    Steps
    -----
    1.  E ← ∅
    2.  For each u_i ∈ U:
    3-4.    ũ_i ← P(u_i)            [preprocessing]
    5-7.    e_i ← f_{m_i}(ũ_i)      [modality-specific encoding]
    8.      E ← E ∪ {e_i}
    10. S ← ∅
    11. For each e_i ∈ E:
    12-13.  s_i ← W_{m_i}·e_i + b_{m_i}  [shared space projection]
    14.     S ← S ∪ {s_i}
    16. Return S
    """

    def __init__(self,
                 input_dim: int  = 772,
                 hidden_dim: int = 256,
                 shared_dim: int = 256):
        self.preprocessor = ModalityAwarePreprocessor()
        self.encoder      = ModalityEncoder(input_dim, hidden_dim)
        self.aligner      = SemanticSpaceAligner(hidden_dim, shared_dim)

    def learn(self, units: List[InstructionalUnit]) -> List[np.ndarray]:
        """
        Run full Algorithm 2.

        Parameters
        ----------
        units : List[InstructionalUnit]  — output of Algorithm 1

        Returns
        -------
        S = {s_1, …, s_N}  — aligned embeddings in shared semantic space
        """
        # ── Steps 1-8: modality-specific encoding ────────────────────
        embeddings: List[np.ndarray] = []           # E ← ∅
        for unit in units:                          # step 2
            u_tilde = self.preprocessor.preprocess(unit)    # step 3-4
            e_i     = self.encoder.encode(u_tilde)          # step 5-7
            embeddings.append(e_i)                          # step 8

        # ── Steps 10-14: shared space alignment ──────────────────────
        aligned: List[np.ndarray] = []              # S ← ∅
        for e_i, unit in zip(embeddings, units):    # step 11
            s_i = self.aligner.align(e_i, unit.modality)   # step 12-13
            aligned.append(s_i)                             # step 14

        print(f"[MultimodalRL] Produced {len(aligned)} aligned embeddings.")
        return aligned                              # step 16


if __name__ == "__main__":
    from data_structures import MODALITIES

    rng   = np.random.default_rng(4)
    units = [
        InstructionalUnit(
            content=rng.random(772).astype(np.float32),
            timestamp=float(i),
            modality=list(MODALITIES)[i % 4]
        )
        for i in range(8)
    ]
    learner = MultimodalRepresentationLearner()
    S       = learner.learn(units)
    print(f"\nAligned embedding set S:")
    for i, s in enumerate(S):
        print(f"  s_{i+1}: shape={s.shape}, modality={units[i].modality}")
