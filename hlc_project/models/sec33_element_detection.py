"""
sec33_element_detection.py
Section 3.3 — Visual Instructional Element Detection

Eq. 4 : E_t = D(f_t) = {e_1, …, e_M}
Eq. 5 : e_i = (r_i, m_i)
"""

import numpy as np
import cv2
from typing import List

from data_structures import BoundingBox, DetectedElement


class VisualElementDetector:
    """
    Applies detection function D(·) to a single video frame and returns
    a set of instructional elements E_t = {e_1, …, e_M}.

    Each element e_i is described by:
        r_i  — normalised bounding box (spatial region)
        m_i  — modality label ∈ {text, equation, table, diagram}

    Layout analysis, region segmentation, and bounding-box prediction
    are handled here. Replace the stub methods with trained models for
    production use.
    """

    def detect(self, frame: np.ndarray) -> List[DetectedElement]:
        """
        Apply D(f_t) to one frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image array (H × W × 3).

        Returns
        -------
        List[DetectedElement]  — E_t
        """
        elements: List[DetectedElement] = []
        h, w = frame.shape[:2]

        raw_boxes = self._layout_analysis(frame)
        for (x, y, bw, bh) in raw_boxes:
            modality = self._classify_modality(frame, x, y, bw, bh)
            # Normalise spatial coordinates
            bbox = BoundingBox(x / w, y / h, bw / w, bh / h)
            elements.append(DetectedElement(region=bbox, modality=modality))

        return elements                                    # E_t

    # ------------------------------------------------------------------
    # Internal helpers (replace with trained models in production)
    # ------------------------------------------------------------------

    def _layout_analysis(self, frame: np.ndarray):
        """
        Stub layout analyser.
        Returns list of raw (x, y, w, h) bounding boxes via contour detection.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 500:
                boxes.append(cv2.boundingRect(cnt))
        return boxes

    def _classify_modality(self, frame: np.ndarray,
                            x: int, y: int, w: int, h: int) -> str:
        """
        Stub modality classifier based on aspect ratio.
        Replace with a CNN/ViT-based classifier for production use.
        """
        aspect = w / (h + 1e-6)
        if aspect > 3:
            return "text"
        elif aspect > 1.5:
            return "table"
        elif aspect < 0.8:
            return "diagram"
        else:
            return "equation"


if __name__ == "__main__":
    # Demo on a blank frame
    blank = np.ones((480, 640, 3), dtype=np.uint8) * 255
    cv2.rectangle(blank, (50, 50), (400, 80), (0, 0, 0), -1)   # text-like region
    cv2.rectangle(blank, (50, 150), (200, 300), (0, 0, 0), -1) # diagram-like region

    detector  = VisualElementDetector()
    elements  = detector.detect(blank)
    print(f"Detected {len(elements)} element(s):")
    for i, e in enumerate(elements):
        print(f"  e_{i+1}: modality={e.modality}, bbox={e.region}")
