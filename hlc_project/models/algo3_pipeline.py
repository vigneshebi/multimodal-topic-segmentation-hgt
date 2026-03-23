"""
algo3_pipeline.py
Algorithm 3 — Multimodal Fusion and Segmentation  (Sections 3.10 – 3.13)
+ Top-Level End-to-End Pipeline

Combines:
    HeterogeneousGraphTransformer  (sec310)
    SimilarityProfiler             (sec311)
    NeuralChangePointDetector      (sec312)
    FAISSSegmentGrouper            (sec313)
"""

import numpy as np
from typing import List

from data_structures import InstructionalUnit, GraphEdge, MODALITIES
from algo1_visual_processing import VisualProcessingModule
from algo2_multimodal_representation import MultimodalRepresentationLearner
from sec36_graph_construction import InstructionalUnitGraph
from sec310_hgt_fusion import HeterogeneousGraphTransformer
from sec311_similarity_profiling import SimilarityProfiler
from sec312_change_point_detection import NeuralChangePointDetector
from sec313_segment_formation import FAISSSegmentGrouper


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm 3
# ─────────────────────────────────────────────────────────────────────────────

class MultimodalFusionAndSegmentation:
    """
    Algorithm 3: MULTIMODAL_FUSION_AND_SEGMENTATION

    Takes aligned embeddings S and the Instructional Unit Graph edges,
    and produces topic-level segment assignments.

    Steps
    -----
    3.10  HGT fusion:           H  ← HGT(S, edges)
    3.11  Similarity profiling: S_sim ← SimilarityProfiler(H)
    3.12  Change-point detect:  B    ← NeuralChangePointDetector(S_sim)
    3.13  Segment formation:    segs ← FAISSSegmentGrouper(units, H, B)
    """

    def __init__(self, shared_dim: int = 256, num_heads: int = 4,
                 merge_threshold: float = 0.85,
                 boundary_threshold: float = 0.5):
        self.hgt       = HeterogeneousGraphTransformer(shared_dim, num_heads)
        self.profiler  = SimilarityProfiler()
        self.detector  = NeuralChangePointDetector()
        self.grouper   = FAISSSegmentGrouper(merge_threshold)
        self.boundary_threshold = boundary_threshold

    def segment(self,
                units: List[InstructionalUnit],
                aligned: List[np.ndarray],
                edges: List[GraphEdge]) -> List[List[int]]:
        """
        Parameters
        ----------
        units   : instructional units (from Algorithm 1)
        aligned : shared-space embeddings (from Algorithm 2)
        edges   : graph edges (from Section 3.6)

        Returns
        -------
        segments : List[List[int]] — one list of unit indices per topic
        """
        H          = self.hgt.fuse(aligned, edges)                    # 3.10
        S_sim      = self.profiler.compute(H)                         # 3.11
        boundaries = self.detector.detect(S_sim, self.boundary_threshold)  # 3.12
        segments   = self.grouper.segment(units, H, boundaries)       # 3.13
        return segments


# ─────────────────────────────────────────────────────────────────────────────
# End-to-End Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class LectureTopicSegmentationPipeline:
    """
    Full end-to-end pipeline: video path → topic segments.

    Phase 1 — Visual Processing     (Algorithm 1)
    Phase 2 — Multimodal Repr.      (Algorithm 2)
    Phase 3 — Graph Construction    (Section 3.6)
    Phase 4 — Fusion & Segmentation (Algorithm 3)
    """

    def __init__(self,
                 sample_interval: float = 1.0,
                 embed_dim: int   = 768,
                 hidden_dim: int  = 256,
                 shared_dim: int  = 256):
        self.visual_module = VisualProcessingModule(sample_interval, embed_dim)
        self.ml_learner    = MultimodalRepresentationLearner(
                                 embed_dim + 4, hidden_dim, shared_dim)
        self.graph_builder = InstructionalUnitGraph()
        self.fusion        = MultimodalFusionAndSegmentation(shared_dim)

    def run(self, video_path: str) -> List[List[int]]:
        """
        Parameters
        ----------
        video_path : str — path to lecture video file.

        Returns
        -------
        segments : List[List[int]]
        """
        print("=" * 60)
        print("Lecture Topic Segmentation Pipeline")
        print("=" * 60)

        # Phase 1
        units = self.visual_module.process(video_path)
        # Phase 2
        aligned = self.ml_learner.learn(units)
        # Phase 3
        _, edges = self.graph_builder.build(units)
        # Phase 4
        segments = self.fusion.segment(units, aligned, edges)

        print(f"\n[Pipeline] Complete — {len(segments)} topic segment(s).")
        return segments


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test (synthetic data — no real video required)
# ─────────────────────────────────────────────────────────────────────────────

def smoke_test():
    """
    Validates Algorithms 2 & 3 end-to-end on synthetic instructional units.
    No video file is needed.
    """
    print("=" * 60)
    print("Smoke test with synthetic data (no video file required)")
    print("=" * 60)

    rng  = np.random.default_rng(42)
    mods = list(MODALITIES)
    N, D = 30, 256

    # Three synthetic topic clusters
    aligned = []
    for i in range(N):
        cluster = i // 10
        base    = np.zeros(D, dtype=np.float32)
        base[cluster * 80:(cluster + 1) * 80] = 1.0
        aligned.append(base + rng.normal(0, 0.1, D).astype(np.float32))

    units = [
        InstructionalUnit(
            content=aligned[i],
            timestamp=float(i),
            modality=mods[i % len(mods)]
        )
        for i in range(N)
    ]

    graph    = InstructionalUnitGraph(tau=2.0, delta_t=3.0, theta_s=0.6)
    _, edges = graph.build(units)

    fusion   = MultimodalFusionAndSegmentation(shared_dim=D)
    segments = fusion.segment(units, aligned, edges)

    print(f"\nResult: {len(segments)} segment(s), "
          f"sizes={[len(s) for s in segments]}")
    return segments


if __name__ == "__main__":
    smoke_test()
