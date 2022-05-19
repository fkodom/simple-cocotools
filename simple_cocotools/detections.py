from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Detection:
    label: int
    # Box format: (left, top, right, bottom)
    box: np.ndarray
    mask: Optional[np.ndarray] = None
    score: float = 0.0
    area: float = 0.0

    def __post_init__(self):
        if self.mask is not None:
            self.mask = self.mask > 0.5


def box_iou(d1: Detection, d2: Detection) -> float:
    l1, t1, r1, b1 = d1.box
    l2, t2, r2, b2 = d2.box

    intersection = max(0, (min(b1, b2) - max(t1, t2)) * (min(r1, r2) - max(l1, l2)))
    union = max(0, (max(b1, b2) - min(t1, t2)) * (max(r1, r2) - min(l1, l2)))
    if union < 1e-8:
        return 0.0
    return intersection / union


def mask_iou(d1: Detection, d2: Detection, threshold: float = 0.5) -> float:
    assert d1.mask is not None
    assert d2.mask is not None
    if d1.mask.dtype != np.bool8:
        d1.mask = d1.mask > threshold
    if d2.mask.dtype != np.bool8:
        d2.mask = d2.mask > threshold

    intersection = np.count_nonzero(d1.mask & d2.mask)
    if not intersection:
        return 0.0

    union = np.count_nonzero(d1.mask | d2.mask)
    return intersection / union
