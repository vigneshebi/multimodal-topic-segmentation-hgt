"""
sec35_unit_generation.py
Section 3.5 — Visual Instructional Unit Generation

Eq.  9 : U = {u_1, …, u_N}
Eq. 10 : u_i = <c_i, t_i, m_i>
"""

import numpy as np
from typing import List

from data_structures import DetectedElement, InstructionalUnit


class InstructionalUnitGenerator:
    """
    Converts detected elements + their visual embeddings into
    Instructional Units — the minimal semantic representations
    aligned with the lecture timeline.

    Each unit u_i = <c_i, t_i, m_i>  (Eq. 10)
        c_i  — content representation (visual embedding from ViT)
        t_i  — temporal index (timestamp of source frame)
        m_i  — modality label ∈ {text, equation, table, diagram}
    """

    def generate(self,
                 elements: List[DetectedElement],
                 embeddings: List[np.ndarray],
                 timestamp: float) -> List[InstructionalUnit]:
        """
        Build instructional units for all elements in one frame.

        Parameters
        ----------
        elements   : detected elements from Section 3.3
        embeddings : spatially-augmented ViT embeddings from Section 3.4
        timestamp  : frame capture time t (seconds)

        Returns
        -------
        List[InstructionalUnit]  — subset of U
        """
        assert len(elements) == len(embeddings), \
            "Number of elements and embeddings must match."

        units: List[InstructionalUnit] = []
        for elem, emb in zip(elements, embeddings):
            unit = InstructionalUnit(
                content=emb,
                timestamp=timestamp,
                modality=elem.modality
            )
            units.append(unit)
        return units

    def generate_all(self,
                     frames_elements: List[List[DetectedElement]],
                     frames_embeddings: List[List[np.ndarray]],
                     timestamps: List[float]) -> List[InstructionalUnit]:
        """
        Generate the full set U = {u_1, …, u_N}  (Eq. 9) across
        all frames.
        """
        all_units: List[InstructionalUnit] = []
        for elems, embs, t in zip(frames_elements,
                                   frames_embeddings,
                                   timestamps):
            all_units.extend(self.generate(elems, embs, t))
        print(f"[UnitGenerator] Total instructional units N = {len(all_units)}")
        return all_units


if __name__ == "__main__":
    from data_structures import BoundingBox, DetectedElement

    rng      = np.random.default_rng(0)
    elem     = DetectedElement(BoundingBox(0.1, 0.1, 0.3, 0.2), "equation")
    emb      = rng.random(772).astype(np.float32)
    gen      = InstructionalUnitGenerator()
    units    = gen.generate([elem], [emb], timestamp=3.0)
    u        = units[0]
    print(f"u_1: modality={u.modality}, timestamp={u.timestamp}, "
          f"content.shape={u.content.shape}")
