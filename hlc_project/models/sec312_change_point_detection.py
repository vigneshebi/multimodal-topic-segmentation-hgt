"""
sec312_change_point_detection.py
Section 3.12 — Neural Change-Point Detection

Eq. 27 : B = {b_1, …, b_K},  b_j ∈ {1, …, N-1}

A 1-D CNN classifier that operates on the similarity sequence S_sim
to detect statistically significant topic boundaries.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List


class NeuralChangePointDetector(nn.Module):
    """
    Detects topic boundaries in a lecture by analysing the similarity
    sequence S_sim produced by the Similarity Profiler (Section 3.11).

    Architecture: 1-D CNN + Sigmoid
        — Captures local patterns of semantic change and their persistence.
        — Outputs a score per position; positions below `threshold` are
          flagged as boundaries.

    Unlike fixed thresholding, this model learns to distinguish genuine
    topic transitions from transient low-similarity fluctuations.

    Parameters
    ----------
    threshold : float — boundary detection threshold (default 0.5).
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, S_sim: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        S_sim : (1, 1, L) tensor of similarity values.

        Returns
        -------
        scores : (L,) tensor, values ∈ [0,1].
                 Low score → likely boundary.
        """
        return self.net(S_sim).squeeze()

    @torch.no_grad()
    def detect(self,
               S_sim: np.ndarray,
               threshold: float = 0.5) -> List[int]:
        """
        Identify boundary indices B from the similarity sequence.

        Parameters
        ----------
        S_sim     : np.ndarray of shape (N-1,)
        threshold : float — positions with score < threshold are boundaries.

        Returns
        -------
        B = [b_1, …, b_K]  (Eq. 27)
        """
        x   = torch.tensor(S_sim, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        out = self.forward(x).numpy()           # (N-1,)

        boundaries = [int(i) for i, v in enumerate(out) if v < threshold]
        print(f"[ChangePointDetector] Detected {len(boundaries)} "
              f"boundary/ies at indices: {boundaries}")
        return boundaries


if __name__ == "__main__":
    rng = np.random.default_rng(8)

    # Synthesise a similarity sequence with drops at positions 7 and 15
    S_sim = np.ones(23, dtype=np.float32) * 0.9
    S_sim[7]  = 0.15    # simulated topic transition
    S_sim[15] = 0.12    # simulated topic transition
    S_sim    += rng.normal(0, 0.05, size=23).astype(np.float32)

    detector   = NeuralChangePointDetector()
    boundaries = detector.detect(S_sim, threshold=0.5)
    print(f"Boundaries: {boundaries}")
