from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import linear_sum_assignment

from simple_cocotools.detections import Detection, box_iou
from simple_cocotools.utils.data import default_collate_fn

PredictionType = Dict[str, np.ndarray]


def parse_as_detections(pred: Dict[str, np.ndarray]) -> List[Detection]:
    detections = []
    for i, label in enumerate(pred["labels"]):
        detections.append(
            Detection(
                label=int(label),
                box=pred["boxes"][i],
                mask=pred["masks"][i] if "masks" in pred else None,
                score=pred["scores"][i] if "scores" in pred else 0.0,
            )
        )

    return detections


@dataclass
class Counts:
    correct: int = 0
    possible: int = 0
    predicted: int = 0

    def __add__(self, other: Counts) -> Counts:
        return Counts(
            correct=self.correct + other.correct,
            possible=self.possible + other.possible,
            predicted=self.predicted + other.predicted,
        )


def get_detection_assignments(
    pred: Sequence[Detection], true: Sequence[Detection], iou_fn: Callable = box_iou
) -> List[Tuple[Detection, Detection, float]]:
    if not (pred and true):
        return []

    ious = np.array([[iou_fn(p, t) for t in true] for p in pred])
    assigned_indices = linear_sum_assignment(ious, maximize=True)
    return [(pred[i], true[j], ious[i, j]) for i, j in zip(*assigned_indices)]


def get_counts_per_iou_threshold(
    pred: Sequence[Detection],
    true: Sequence[Detection],
    iou_thresholds: Sequence[float],
) -> List[Counts]:
    """Gets evaluation metrics for a *single* detection class across a range of
    IoU thresholds.

    NOTE: We determine "correct" detections by matching the predictions/labels
    together, maximizing the IoU between each matched pair.  A correct prediction
    will be successfully paired with one of the labels, and the IoU with that label
    is greater than the threshold value.
    """
    assignments = get_detection_assignments(pred, true)
    return [
        Counts(
            correct=sum(iou >= iou_threshold for _, _, iou in assignments),
            predicted=len(pred),
            possible=len(true),
        )
        for iou_threshold in iou_thresholds
    ]


class CocoEvaluator:
    def __init__(
        self,
        iou_thresholds: Optional[Sequence[float]] = None,
        labelmap: Optional[Dict[int, str]] = None,
    ):
        self.labelmap = labelmap
        if iou_thresholds is None:
            iou_thresholds = np.arange(0.5, 1.0, 0.05).tolist()

        self.iou_thresholds = iou_thresholds
        # 'self.counts_per_class' is a list of counts for each class label -- one
        # for each IoU threshold
        self.counts_per_class: Dict[Union[str, int], List[Counts]] = {}
        self.summary: Dict[str, float] = {}

    def _update_sample(self, pred: List[Detection], true: List[Detection]):
        pred_labels = set(p.label for p in pred)
        true_labels = set(t.label for t in true)
        labels = pred_labels.union(true_labels)

        for label in labels:
            key = self.labelmap[label] if self.labelmap else label
            assert isinstance(key, (str, int))
            if key not in self.counts_per_class:
                self.counts_per_class[key] = [Counts() for _ in self.iou_thresholds]

            _pred = [p for p in pred if p.label == label]
            _true = [t for t in true if t.label == label]

            new_counts = get_counts_per_iou_threshold(
                _pred, _true, iou_thresholds=self.iou_thresholds
            )

            self.counts_per_class[key] = [
                counts + updates
                for counts, updates in zip(self.counts_per_class[key], new_counts)
            ]

    def update(
        self,
        pred: List[Dict[str, np.ndarray]],
        true: List[Dict[str, np.ndarray]],
    ):
        for _pred, _true in zip(pred, true):
            self._update_sample(parse_as_detections(_pred), parse_as_detections(_true))

    def accumulate(self):
        class_aps = {
            key: [(m.correct / m.predicted if m.predicted else 0.0) for m in metrics]
            for key, metrics in self.counts_per_class.items()
        }
        class_ars = {
            key: [(m.correct / m.possible if m.possible else 0.0) for m in metrics]
            for key, metrics in self.counts_per_class.items()
        }
        class_map = {k: np.mean(v).item() for k, v in class_aps.items()}
        class_mar = {k: np.mean(v).item() for k, v in class_ars.items()}

        self.summary = {
            "mAP": np.mean(list(class_map.values())),
            "mAR": np.mean(list(class_mar.values())),
            "class_AP": class_map,
            "class_AR": class_mar,
        }

        return self.summary

    def summarize(self, verbose: bool = True):
        if not self.summary:
            self.accumulate()
        if verbose:
            print(json.dumps(self.summary, indent=4))
        return self.summary

    def reset(self):
        self.counts_per_class = {}
        self.summary = {}


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from simple_cocotools.utils.coco import CocoDetection2014

    ds = CocoDetection2014(split="val")
    loader = DataLoader(ds, batch_size=32, num_workers=4, collate_fn=default_collate_fn)
    evaluator = CocoEvaluator()

    for _, true in tqdm(loader):
        pred = true
        evaluator.update(pred, true)

    evaluator.summarize()
