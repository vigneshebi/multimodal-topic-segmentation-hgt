"""
sec32_frame_extraction.py
Section 3.2 — Video Frame Extraction

Eq. 1 : F = {f_1, f_2, …, f_T}
Eq. 2 : t_k = k · s
Eq. 3 : index(t_k) = floor(t_k · r)
"""

import math
import numpy as np
import cv2
from typing import List, Tuple


class VideoFrameExtractor:
    """
    Samples frames from a lecture video at a fixed time interval s.

    Parameters
    ----------
    sample_interval : float
        Fixed time interval s (seconds) between sampled frames.
    """

    def __init__(self, sample_interval: float = 1.0):
        self.s = sample_interval

    def extract_frames(self, video_path: str) -> List[Tuple[np.ndarray, float]]:
        """
        Extract frame sequence F = {f_1, …, f_T} from video V.

        Returns
        -------
        List of (frame_array, timestamp) pairs.
        """
        cap = cv2.VideoCapture(video_path)
        r   = cap.get(cv2.CAP_PROP_FPS)      # frame rate r (fps)
        frames: List[Tuple[np.ndarray, float]] = []

        k = 0
        while True:
            t_k       = k * self.s                        # Eq. 2
            frame_idx = int(math.floor(t_k * r))          # Eq. 3

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            frames.append((frame, t_k))
            k += 1

        cap.release()
        print(f"[FrameExtractor] Extracted {len(frames)} frames "
              f"(interval={self.s}s, fps={r})")
        return frames                                      # F


if __name__ == "__main__":
    # Quick test with a placeholder path
    extractor = VideoFrameExtractor(sample_interval=2.0)
    print("VideoFrameExtractor ready. Call extract_frames('video.mp4') to use.")
