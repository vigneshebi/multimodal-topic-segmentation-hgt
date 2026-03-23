"""
sec311_similarity_profiling.py
Section 3.11 — Similarity Profiling along the Lecture Timeline

Eq. 25 : sim(i, i+1) = (h_i^⊤ h_{i+1}) / (‖h_i‖ · ‖h_{i+1}‖)
Eq. 26 : S_sim = {sim(1,2), sim(2,3), …, sim(N-1,N)}
"""

import numpy as np


class SimilarityProfiler:
    """
    Computes pairwise cosine similarity between consecutive fused
    representations along the lecture timeline.

    A high sim(i, i+1) value indicates semantic continuity.
    A low  sim(i, i+1) value is indicative of a topic transition.

    The resulting similarity sequence S_sim is passed to the
    Neural Change-Point Detector (Section 3.12).
    """

    def compute(self, H: np.ndarray) -> np.ndarray:
        """
        Compute S_sim from the fused embedding matrix H.

        Parameters
        ----------
        H : np.ndarray of shape (N, d)
            Fused representations from the HGT (Section 3.10).

        Returns
        -------
        S_sim : np.ndarray of shape (N-1,)
            Cosine similarity between consecutive pairs (Eq. 25-26).
        """
        N    = H.shape[0]
        sims = np.zeros(N - 1, dtype=np.float32)

        for i in range(N - 1):
            num   = float(H[i] @ H[i + 1])                  # Eq. 25 numerator
            denom = (np.linalg.norm(H[i]) *
                     np.linalg.norm(H[i + 1])) + 1e-9       # Eq. 25 denominator
            sims[i] = num / denom

        return sims                                          # S_sim  (Eq. 26)

    def summary(self, S_sim: np.ndarray) -> None:
        """Print basic statistics of the similarity sequence."""
        print(f"[SimilarityProfiler] Length={len(S_sim)}  "
              f"min={S_sim.min():.3f}  max={S_sim.max():.3f}  "
              f"mean={S_sim.mean():.3f}  std={S_sim.std():.3f}")


if __name__ == "__main__":
    rng = np.random.default_rng(6)
    # Simulate 3 topic clusters in H
    D   = 256
    H_blocks = [
        rng.normal(loc=k, scale=0.2, size=(8, D)).astype(np.float32)
        for k in range(3)
    ]
    H = np.vstack(H_blocks)   # shape (24, 256)

    profiler = SimilarityProfiler()
    S_sim    = profiler.compute(H)
    profiler.summary(S_sim)

    print("\nSimilarity at topic transitions (indices 7→8 and 15→16):")
    print(f"  sim(7,8)   = {S_sim[7]:.4f}   ← expected drop")
    print(f"  sim(15,16) = {S_sim[15]:.4f}  ← expected drop")
