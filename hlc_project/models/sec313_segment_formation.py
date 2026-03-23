"""
sec313_segment_formation.py
Section 3.13 — Topic Segment Formation using FAISS-Assisted Semantic Grouping

Eq. 28 : S_1={u_1,…,u_{b_1}}, S_2={u_{b_1+1},…,u_{b_2}}, …
Eq. 29 : z_j = (1/|S_j|) Σ_{u_i ∈ S_j} h_i
Eq. 30 : Z = {z_1, …, z_M}
"""

import numpy as np
from typing import List

from data_structures import InstructionalUnit


class FAISSSegmentGrouper:
    """
    1. Partitions the lecture timeline into contiguous segments using the
       detected boundary indices  B = {b_1, …, b_K}  (Eq. 28).

    2. Represents each segment as an average of its constituent fused
       embeddings  z_j  (Eq. 29-30).

    3. Refines segmentation via FAISS approximate nearest-neighbour
       search (or a numpy fallback):
           - Semantically similar segments are merged.
           - Semantically distinct segments are preserved.
       This reduces fragmentation and corrects over-segmentation.

    Parameters
    ----------
    merge_threshold : float
        Cosine similarity threshold above which two segments are merged.
    """

    def __init__(self, merge_threshold: float = 0.85):
        self.merge_threshold = merge_threshold

    def segment(self,
                units: List[InstructionalUnit],
                H: np.ndarray,
                boundaries: List[int]) -> List[List[int]]:
        """
        Full segmentation pipeline.

        Parameters
        ----------
        units      : List[InstructionalUnit] — from Algorithm 1
        H          : np.ndarray (N, d)       — fused embeddings from HGT
        boundaries : List[int]               — from change-point detector

        Returns
        -------
        segments : List[List[int]]
            Each inner list contains unit indices belonging to one topic.
        """
        # ── Step 1: partition by boundary indices (Eq. 28) ───────────
        N    = len(units)
        cuts = sorted(set(boundaries)) + [N - 1]
        segments: List[List[int]] = []
        prev = 0
        for b in cuts:
            seg = list(range(prev, b + 1))
            if seg:
                segments.append(seg)
            prev = b + 1

        # ── Step 2: compute segment representations (Eq. 29-30) ──────
        Z = np.stack([H[seg].mean(axis=0) for seg in segments])  # Eq. 29-30

        # ── Step 3: FAISS refinement or numpy fallback ────────────────
        try:
            import faiss
            segments = self._faiss_merge(segments, Z)
        except ImportError:
            segments = self._numpy_merge(segments, Z)

        print(f"[Segmenter] Final topic count: {len(segments)}")
        for k, seg in enumerate(segments):
            ts = [units[i].timestamp for i in seg]
            print(f"  Segment {k+1}: units {seg[0]}–{seg[-1]}  "
                  f"({min(ts):.1f}s – {max(ts):.1f}s)")
        return segments

    # ------------------------------------------------------------------
    # Merge strategies
    # ------------------------------------------------------------------

    def _numpy_merge(self,
                     segments: List[List[int]],
                     Z: np.ndarray) -> List[List[int]]:
        """Greedy cosine-similarity merge (no FAISS dependency)."""
        Z_norm = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
        changed = True
        while changed:
            changed  = False
            used     = [False] * len(segments)
            new_segs = []
            for i in range(len(segments)):
                if used[i]:
                    continue
                grp = list(segments[i])
                for j in range(i + 1, len(segments)):
                    if used[j]:
                        continue
                    sim = float(Z_norm[i] @ Z_norm[j])
                    if sim >= self.merge_threshold:
                        grp   += segments[j]
                        used[j] = True
                        changed = True
                new_segs.append(grp)
                used[i] = True
            segments = new_segs
            # Recompute Z_norm after merge (approximate)
            Z_norm = Z_norm[:len(segments)]
        return segments

    def _faiss_merge(self,
                     segments: List[List[int]],
                     Z: np.ndarray) -> List[List[int]]:
        """FAISS-based ANN merge for scalable high-dimensional search."""
        import faiss
        d    = Z.shape[1]
        Zn   = (Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
                ).astype(np.float32)
        idx  = faiss.IndexFlatIP(d)
        idx.add(Zn)
        D, I = idx.search(Zn, k=2)        # k=2: self + nearest neighbour

        used     = [False] * len(segments)
        new_segs = []
        for i in range(len(segments)):
            if used[i]:
                continue
            grp  = list(segments[i])
            best = int(I[i, 1]) if I.shape[1] > 1 else -1
            sim  = float(D[i, 1]) if D.shape[1] > 1 else 0.0
            if best >= 0 and not used[best] and sim >= self.merge_threshold:
                grp    += segments[best]
                used[best] = True
            new_segs.append(grp)
            used[i] = True
        return new_segs


if __name__ == "__main__":
    rng = np.random.default_rng(9)
    D, N = 64, 24

    # 3 synthetic topic clusters
    H_blocks = [
        rng.normal(loc=k * 5, scale=0.5, size=(8, D)).astype(np.float32)
        for k in range(3)
    ]
    H = np.vstack(H_blocks)

    units = [
        InstructionalUnit(
            content=H[i], timestamp=float(i),
            modality=["text", "equation", "table", "diagram"][i % 4]
        )
        for i in range(N)
    ]

    boundaries = [7, 15]
    grouper    = FAISSSegmentGrouper(merge_threshold=0.90)
    segments   = grouper.segment(units, H, boundaries)
    print(f"\nSegment sizes: {[len(s) for s in segments]}")
