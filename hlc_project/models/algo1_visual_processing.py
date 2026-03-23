"""
algo1_visual_processing.py
Algorithm 1 — Visual Processing Module  (Sections 3.2 – 3.5)

Combines:
    VideoFrameExtractor    (sec32)
    VisualElementDetector  (sec33)
    ViTEncoder             (sec34)
    InstructionalUnitGenerator (sec35)
"""

from typing import List

from data_structures import InstructionalUnit
from sec32_frame_extraction import VideoFrameExtractor
from sec33_element_detection import VisualElementDetector
from sec34_visual_encoding import ViTEncoder
from sec35_unit_generation import InstructionalUnitGenerator


class VisualProcessingModule:
    """
    Algorithm 1: VISUAL_PROCESSING(V) → U

    Steps
    -----
    1.  Extract frames F = {f_1, …, f_T} from video V
    2.  Initialise U ← ∅
    3.  For each frame f_t:
    4.      Detect elements  E_t ← D(f_t)
    5-8.    For each e_i:  encode → generate unit u_i → U ← U ∪ {u_i}
    12. Return U
    """

    def __init__(self, sample_interval: float = 1.0,
                 embed_dim: int = 768):
        self.extractor = VideoFrameExtractor(sample_interval)
        self.detector  = VisualElementDetector()
        self.encoder   = ViTEncoder(embed_dim)
        self.unit_gen  = InstructionalUnitGenerator()

    def process(self, video_path: str) -> List[InstructionalUnit]:
        """
        Full Algorithm 1 pipeline.

        Parameters
        ----------
        video_path : str — path to the lecture video file.

        Returns
        -------
        U : List[InstructionalUnit]
        """
        # Step 1 — extract frames
        frames = self.extractor.extract_frames(video_path)
        # Step 2 — initialise unit set
        units: List[InstructionalUnit] = []

        for frame, t in frames:                            # Step 3
            elements   = self.detector.detect(frame)       # Step 4
            embeddings = self.encoder.encode_set(elements, frame)

            for elem, emb in zip(elements, embeddings):    # Steps 5-9
                unit = InstructionalUnit(
                    content=emb,
                    timestamp=t,
                    modality=elem.modality
                )
                units.append(unit)

        print(f"[VisualProcessing] Generated {len(units)} instructional units.")
        return units                                       # Step 12


if __name__ == "__main__":
    print("VisualProcessingModule ready.")
    print("Usage: module = VisualProcessingModule(); units = module.process('video.mp4')")
