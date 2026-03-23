"""
sec36_graph_construction.py
Section 3.6 — Instructional Unit Graph Construction

Eq. 11 : G = (V, E)
Eq. 12 : E = E_t ∪ E_c ∪ E_s
Eq. 13 : E_t = {(u_i, u_{i+1}) | 1 ≤ i < N}
Eq. 14 : w^(t)_ij = exp(−|t_i − t_j| / τ)
Eq. 15 : E_c = {(u_i, u_j) | m_i ≠ m_j, |t_i − t_j| ≤ Δ_t}
Eq. 16 : w^(c)_ij = cos(c_i, c_j)
Eq. 17 : E_s = {(u_i, u_j) | cos(c_i, c_j) ≥ θ_s}
Eq. 18 : w_ij = α·w^(t) + β·w^(c) + γ·w^(s)
"""

import math
import numpy as np
from typing import List, Tuple

from data_structures import InstructionalUnit, GraphEdge


class InstructionalUnitGraph:
    """
    Builds the Instructional Unit Graph G = (V, E) where:
        V — nodes = instructional units
        E — three types of heterogeneous edges:
            E_t  temporal edges     (sequential adjacency)
            E_c  cross-modal edges  (different modality, close in time)
            E_s  semantic edges     (high cosine similarity)

    Parameters
    ----------
    tau       : temporal sensitivity scaling factor (Eq. 14)
    delta_t   : cross-modal time window Δ_t in seconds (Eq. 15)
    theta_s   : semantic similarity threshold θ_s (Eq. 17)
    alpha     : weight for temporal contribution (Eq. 18)
    beta      : weight for cross-modal contribution (Eq. 18)
    gamma     : weight for semantic contribution (Eq. 18)
    """

    def __init__(self,
                 tau: float     = 1.0,
                 delta_t: float = 5.0,
                 theta_s: float = 0.7,
                 alpha: float   = 0.33,
                 beta: float    = 0.33,
                 gamma: float   = 0.34):
        self.tau     = tau
        self.delta_t = delta_t
        self.theta_s = theta_s
        self.alpha   = alpha
        self.beta    = beta
        self.gamma   = gamma

    def build(self, units: List[InstructionalUnit]
              ) -> Tuple[List[InstructionalUnit], List[GraphEdge]]:
        """
        Construct graph G = (V, E).

        Returns
        -------
        (nodes, edges)  where nodes = units and edges = E_t ∪ E_c ∪ E_s
        """
        Et = self._temporal_edges(units)     # Eq. 13-14
        Ec = self._cross_modal_edges(units)  # Eq. 15-16
        Es = self._semantic_edges(units)     # Eq. 17
        E  = self._unify_weights(Et + Ec + Es)  # Eq. 18
        print(f"[Graph] N={len(units)} nodes | "
              f"|Et|={len(Et)}  |Ec|={len(Ec)}  |Es|={len(Es)}")
        return units, E

    # ------------------------------------------------------------------
    # Edge construction helpers
    # ------------------------------------------------------------------

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
        return float(np.dot(a, b) / denom)

    def _temporal_edges(self, units: List[InstructionalUnit]
                        ) -> List[GraphEdge]:
        """Temporal edges E_t  (Eq. 13-14)."""
        edges = []
        for i in range(len(units) - 1):
            dt = abs(units[i].timestamp - units[i + 1].timestamp)
            w  = math.exp(-dt / self.tau)                       # Eq. 14
            edges.append(GraphEdge(i, i + 1, 't', w))
        return edges

    def _cross_modal_edges(self, units: List[InstructionalUnit]
                           ) -> List[GraphEdge]:
        """Cross-modal edges E_c  (Eq. 15-16)."""
        edges = []
        for i in range(len(units)):
            for j in range(i + 1, len(units)):
                diff_mod  = units[i].modality != units[j].modality
                close_t   = abs(units[i].timestamp -
                                units[j].timestamp) <= self.delta_t
                if diff_mod and close_t:                        # Eq. 15
                    w = self._cosine(units[i].content,
                                     units[j].content)          # Eq. 16
                    edges.append(GraphEdge(i, j, 'c', w))
        return edges

    def _semantic_edges(self, units: List[InstructionalUnit]
                        ) -> List[GraphEdge]:
        """Semantic edges E_s  (Eq. 17)."""
        edges = []
        for i in range(len(units)):
            for j in range(i + 1, len(units)):
                w = self._cosine(units[i].content, units[j].content)
                if w >= self.theta_s:                           # Eq. 17
                    edges.append(GraphEdge(i, j, 's', w))
        return edges

    def _unify_weights(self, edges: List[GraphEdge]) -> List[GraphEdge]:
        """Unified edge weight  w_ij = α·w^t + β·w^c + γ·w^s  (Eq. 18)."""
        coeff = {'t': self.alpha, 'c': self.beta, 's': self.gamma}
        for e in edges:
            e.weight *= coeff[e.edge_type]
        return edges


if __name__ == "__main__":
    rng   = np.random.default_rng(7)
    units = [
        InstructionalUnit(
            content=rng.random(256).astype(np.float32),
            timestamp=float(i),
            modality=["text", "equation", "table", "diagram"][i % 4]
        )
        for i in range(8)
    ]
    graph  = InstructionalUnitGraph(tau=2.0, delta_t=3.0, theta_s=0.5)
    nodes, edges = graph.build(units)
    print(f"Total edges: {len(edges)}")
    for e in edges[:5]:
        print(f"  {e}")
