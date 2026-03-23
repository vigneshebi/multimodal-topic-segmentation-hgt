"""
data_structures.py
Shared dataclasses used across all modules.
"""

from dataclasses import dataclass
import numpy as np

MODALITIES = {"text", "equation", "table", "diagram"}


@dataclass
class BoundingBox:
    x: float
    y: float
    w: float
    h: float


@dataclass
class DetectedElement:
    """Eq. 5: e_i = (r_i, m_i)"""
    region: BoundingBox        # r_i — spatial bounding box
    modality: str              # m_i ∈ {text, equation, table, diagram}


@dataclass
class InstructionalUnit:
    """Eq. 10: u_i = <c_i, t_i, m_i>"""
    content: np.ndarray        # c_i ∈ R^d — visual embedding
    timestamp: float           # t_i ∈ R   — temporal index
    modality: str              # m_i — modality type


@dataclass
class GraphEdge:
    src: int
    dst: int
    edge_type: str             # 't' | 'c' | 's'
    weight: float
