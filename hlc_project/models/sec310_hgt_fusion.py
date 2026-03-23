"""
sec310_hgt_fusion.py
Section 3.10 — Multimodal Temporal Fusion using Heterogeneous Graph Transformer

Eq. 23 : h_i = σ( Σ_r Σ_{j∈N_r(i)} α^(r)_ij · W^(r) · s_j )
Eq. 24 : α^(r)_ij = softmax_k( Q_i^(r) · K_k^(r) ) for k ∈ N_r(i)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from typing import List, Dict

from data_structures import GraphEdge


class HeterogeneousGraphTransformer(nn.Module):
    """
    Relation-aware attention-based message passing over the
    Instructional Unit Graph.

    Three relation types R = {t, c, s}:
        t — temporal edges   (sequential lecture flow)
        c — cross-modal edges (complementary modality interaction)
        s — semantic edges   (long-range conceptual similarity)

    For each node u_i, the fused representation h_i is computed by
    aggregating messages from neighbours under each relation type,
    weighted by learnable attention scores α^(r)_ij.

    Parameters
    ----------
    d         : int — shared embedding dimension
    num_heads : int — number of attention heads
    """

    RELATIONS = ['t', 'c', 's']

    def __init__(self, d: int = 256, num_heads: int = 4):
        super().__init__()
        self.d   = d
        self.h   = num_heads
        self.d_k = d // num_heads

        # Relation-specific transformation matrices W^(r)  (Eq. 23)
        self.W   = nn.ModuleDict({r: nn.Linear(d, d, bias=False)
                                   for r in self.RELATIONS})
        # Query/Key projections for attention  (Eq. 24)
        self.W_Q = nn.ModuleDict({r: nn.Linear(d, d, bias=False)
                                   for r in self.RELATIONS})
        self.W_K = nn.ModuleDict({r: nn.Linear(d, d, bias=False)
                                   for r in self.RELATIONS})
        self.act = nn.ReLU()

    def forward(self,
                S: torch.Tensor,
                edges: List[GraphEdge]) -> torch.Tensor:
        """
        Parameters
        ----------
        S     : (N, d) aligned embeddings
        edges : list of GraphEdge

        Returns
        -------
        H : (N, d) context-enriched fused embeddings
        """
        N = S.shape[0]
        H = torch.zeros_like(S)

        # Group neighbour indices by relation type
        nbrs: Dict[str, Dict[int, List[int]]] = {
            r: defaultdict(list) for r in self.RELATIONS}
        for e in edges:
            nbrs[e.edge_type][e.src].append(e.dst)
            nbrs[e.edge_type][e.dst].append(e.src)   # treat as undirected

        for r in self.RELATIONS:
            for i in range(N):
                nb = nbrs[r][i]
                if not nb:
                    continue
                nb_idx = torch.tensor(nb, dtype=torch.long)

                # Compute query and keys  (Eq. 24)
                Q_i  = self.W_Q[r](S[i])               # (d,)
                K_nb = self.W_K[r](S[nb_idx])          # (|nb|, d)

                # Scaled dot-product attention
                scores = (Q_i.unsqueeze(0) * K_nb).sum(-1) / (self.d_k ** 0.5)
                alpha  = F.softmax(scores, dim=0)       # Eq. 24  α^(r)_ij

                # Weighted message aggregation  (Eq. 23)
                vals   = self.W[r](S[nb_idx])           # (|nb|, d)
                H[i]  += (alpha.unsqueeze(-1) * vals).sum(0)

        return self.act(H)                              # σ(·)  Eq. 23

    @torch.no_grad()
    def fuse(self,
             aligned: List[np.ndarray],
             edges: List[GraphEdge]) -> np.ndarray:
        """
        Convenience wrapper: numpy in → numpy out.

        Returns
        -------
        H : np.ndarray of shape (N, d)
        """
        S = torch.tensor(np.stack(aligned), dtype=torch.float32)
        H = self.forward(S, edges)
        return H.numpy()


if __name__ == "__main__":
    rng     = np.random.default_rng(5)
    N, D    = 10, 256
    aligned = [rng.random(D).astype(np.float32) for _ in range(N)]
    edges   = [GraphEdge(i, i+1, 't', 0.9) for i in range(N-1)]
    edges  += [GraphEdge(0, 5, 's', 0.75), GraphEdge(2, 7, 'c', 0.6)]

    hgt = HeterogeneousGraphTransformer(d=D, num_heads=4)
    H   = hgt.fuse(aligned, edges)
    print(f"Fused H shape: {H.shape}")
    print(f"Sample h_0: {H[0, :5]}")
