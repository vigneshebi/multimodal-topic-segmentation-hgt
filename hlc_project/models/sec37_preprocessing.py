"""
sec37_preprocessing.py
Section 3.7 — Preprocessing and Modality Identification

Eq. 19 : ũ_i = P(u_i) = <c̃_i, t_i, m_i>
"""

import numpy as np
from typing import List

from data_structures import InstructionalUnit


class ModalityAwarePreprocessor:
    """
    Applies modality-specific normalisation to each instructional unit
    while preserving the semantic integrity of its content.

    Preprocessing rules:
        text      — minimal normalisation (L2-normalise the embedding)
        equation  — no linearisation; preserve symbolic structure
        table     — preserve row/column layout
        diagram   — preserve graphical spatial structure

    All units are temporally aligned and format-consistent after
    this stage, ready for the modality-specific encoders.
    """

    def preprocess(self, unit: InstructionalUnit) -> InstructionalUnit:
        """
        Apply P(u_i) to a single unit.

        Returns
        -------
        ũ_i = <c̃_i, t_i, m_i>  (Eq. 19)
        """
        c_norm = self._normalize(unit.content, unit.modality)
        return InstructionalUnit(
            content=c_norm,
            timestamp=unit.timestamp,
            modality=unit.modality
        )

    def preprocess_all(self,
                       units: List[InstructionalUnit]
                       ) -> List[InstructionalUnit]:
        """
        Preprocess the full set U → Ũ.
        """
        preprocessed = [self.preprocess(u) for u in units]
        print(f"[Preprocessor] Preprocessed {len(preprocessed)} units.")
        return preprocessed

    # ------------------------------------------------------------------
    # Modality-specific normalisation
    # ------------------------------------------------------------------

    def _normalize(self, content: np.ndarray, modality: str) -> np.ndarray:
        if modality == "text":
            # Minimal normalisation: L2-normalise to remove magnitude noise
            norm = np.linalg.norm(content)
            return content / (norm + 1e-9)

        elif modality == "equation":
            # Preserve symbolic form — no linearisation
            return content.copy()

        elif modality == "table":
            # Preserve row/column structure — no reshaping
            return content.copy()

        elif modality == "diagram":
            # Preserve graphical structure — no linearisation
            return content.copy()

        # Fallback
        return content.copy()


if __name__ == "__main__":
    rng  = np.random.default_rng(1)
    unit = InstructionalUnit(
        content=rng.random(772).astype(np.float32) * 10,
        timestamp=2.5,
        modality="text"
    )
    prep = ModalityAwarePreprocessor()
    u_tilde = prep.preprocess(unit)
    print(f"Original  norm : {np.linalg.norm(unit.content):.4f}")
    print(f"Processed norm : {np.linalg.norm(u_tilde.content):.4f}  (should be ≈1.0)")
    print(f"Modality preserved: {u_tilde.modality}")
