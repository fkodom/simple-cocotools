from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import linear_sum_assignment

from simple_cocotools.detections import Detection, box_iou, mask_iou


def parse_predictions(pred: Dict[str, np.ndarray]) -> List[Detection]:
    detections = []
    for i, label in enumerate(pred["labels"]):
        # Ensure the bbox is a Numpy array, without copying data if possible.
        box = np.array(pred["boxes"][i], copy=False)
        # These values may not be present, depending on where the "predictions"
        # are coming from. (Are they from a bounding box model, which doesn't
        # have masks, or loaded directly from a dataset without confidence scores?)
        mask = np.array(pred["masks"][i], copy=False) if "masks" in pred else None
        score = float(pred["scores"][i]) if "scores" in pred else 0.0
        detections.append(Detection(label=int(label), box=box, mask=mask, score=score))

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
    iou_fn: Callable = box_iou,
) -> List[Counts]:
    """Gets evaluation metrics for a *single* detection class across a range of
    IoU thresholds.

    NOTE: We determine "correct" detections by matching the predictions/labels
    together, maximizing the IoU between each matched pair.  A correct prediction
    will be successfully paired with one of the labels, and the IoU with that label
    is greater than the threshold value.
    """
    assignments = get_detection_assignments(pred, true, iou_fn=iou_fn)
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
        # 'self.box_counts' is a list of counts for each class label -- one
        # for each IoU threshold.  Same for 'self.mask_counts'.
        self.box_counts: Dict[Union[str, int], List[Counts]] = {}
        self.mask_counts: Dict[Union[str, int], List[Counts]] = {}
        self.metrics: Dict[str, Dict] = {}

    def _update_one_sample(self, pred: List[Detection], true: List[Detection]):
        pred_labels = set(p.label for p in pred)
        true_labels = set(t.label for t in true)
        labels = pred_labels.union(true_labels)

        for label in labels:
            _pred = [p for p in pred if p.label == label]
            _true = [t for t in true if t.label == label]
            key = self.labelmap[label] if self.labelmap else label
            assert isinstance(key, (str, int))

            if key not in self.box_counts:
                self.box_counts[key] = [Counts() for _ in self.iou_thresholds]
            new_box_counts = get_counts_per_iou_threshold(
                _pred, _true, iou_thresholds=self.iou_thresholds
            )
            self.box_counts[key] = [
                counts + updates
                for counts, updates in zip(self.box_counts[key], new_box_counts)
            ]

            evaluate_masks = (
                (_true or _pred)
                and all(t.mask is not None for t in _true)
                and all(p.mask is not None for p in _pred)
            )
            if evaluate_masks:
                if key not in self.mask_counts:
                    self.mask_counts[key] = [Counts() for _ in self.iou_thresholds]
                new_mask_counts = get_counts_per_iou_threshold(
                    _pred, _true, iou_thresholds=self.iou_thresholds, iou_fn=mask_iou
                )
                self.mask_counts[key] = [
                    counts + updates
                    for counts, updates in zip(self.mask_counts[key], new_mask_counts)
                ]

    def update(
        self,
        pred: List[Dict[str, np.ndarray]],
        true: List[Dict[str, np.ndarray]],
    ):
        for _pred, _true in zip(pred, true):
            self._update_one_sample(parse_predictions(_pred), parse_predictions(_true))

    @staticmethod
    def _accumulate_metrics(
        metrics: Dict[Union[str, int], List[Counts]]
    ) -> Dict[str, Any]:
        class_aps = {
            key: [(m.correct / m.predicted if m.predicted else 0.0) for m in _metrics]
            for key, _metrics in metrics.items()
        }
        class_ars = {
            key: [(m.correct / m.possible if m.possible else 0.0) for m in _metrics]
            for key, _metrics in metrics.items()
        }
        class_map = {k: np.mean(v).item() for k, v in class_aps.items()}
        class_mar = {k: np.mean(v).item() for k, v in class_ars.items()}

        return {
            "mAP": np.mean(list(class_map.values())),
            "mAR": np.mean(list(class_mar.values())),
            "class_AP": class_map,
            "class_AR": class_mar,
        }

    def accumulate(self):
        self.metrics = {"box": self._accumulate_metrics(self.box_counts)}
        if self.mask_counts:
            self.metrics["mask"] = self._accumulate_metrics(self.mask_counts)

        return self.metrics

    def summarize(self, verbose: bool = True):
        if not self.metrics:
            self.accumulate()
        if verbose:
            print(json.dumps(self.metrics, indent=4))
        return self.metrics

    def reset(self):
        self.box_counts = {}
        self.mask_counts = {}
        self.metrics = {}
