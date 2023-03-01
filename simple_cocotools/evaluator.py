from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import linear_sum_assignment

from simple_cocotools.detections import Detection, box_iou, mask_iou

if sys.version_info >= (3, 8):
    from typing import TypedDict  # pylint: disable=no-name-in-module
else:
    from typing_extensions import TypedDict


def parse_predictions(pred: Dict[str, np.ndarray]) -> List[Detection]:
    detections = []
    for i, label in enumerate(pred["labels"]):
        # Ensure the bbox is a Numpy array, without copying data if possible.
        box = np.array(pred["boxes"][i], copy=False)
        # These values may not be present, depending on where the "predictions"
        # are coming from. (Are they from a bounding box model, which doesn't
        # have masks, or loaded directly from a dataset without confidence scores?)
        mask = np.array(pred["masks"][i], copy=False) if "masks" in pred else None
        keypoints = (
            np.array(pred["keypoints"][i], copy=False) if "keypoints" in pred else None
        )
        keypoints_scores = (
            np.array(pred["keypoints_scores"][i], copy=False)
            if "keypoints_scores" in pred
            else None
        )
        score = float(pred["scores"][i]) if "scores" in pred else 0.0
        detections.append(
            Detection(
                label=int(label),
                box=box,
                mask=mask,
                score=score,
                keypoints=keypoints,
                keypoints_scores=keypoints_scores,
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


class KeypointMetrics(TypedDict):
    distance: float


@dataclass
class PerKeypointMetrics:
    distance: float = 0.0
    num_samples: int = 0


RunningKpMetricsType = Dict[str, PerKeypointMetrics]


def _update_running_kp_metrics_single(
    metrics: RunningKpMetricsType,
    pred: Detection,
    true: Detection,
    keypoint_labelmap: Optional[Dict[int, str]] = None,
) -> RunningKpMetricsType:
    if pred.keypoints is None or true.keypoints is None:
        return metrics

    for i, (pred_kp, true_kp) in enumerate(zip(pred.keypoints, true.keypoints)):
        if pred_kp is None or true_kp is None:
            continue

        visible = true_kp[2] > 0.0
        if not visible:
            continue

        keypoint_name = keypoint_labelmap[i] if keypoint_labelmap else str(i)
        if keypoint_name not in metrics:
            metrics[keypoint_name] = PerKeypointMetrics()
        kp = metrics[keypoint_name]

        # Boxes are given in x-y coordinates: [x0, y0, x1, y1]
        box_width = max(1e-8, true.box[2] - true.box[0])
        box_height = max(1e-8, true.box[3] - true.box[1])
        dx = (pred_kp[0] - true_kp[0]) / box_width
        dy = (pred_kp[1] - true_kp[1]) / box_height

        distance = float(np.sqrt(dx**2 + dy**2))
        # Weighting factor for updating the current batch
        alpha = 1 / (kp.num_samples + 1)
        kp.distance = (1 - alpha) * kp.distance + alpha * distance
        kp.num_samples += 1

    return metrics


def update_running_kp_metrics(
    metrics: RunningKpMetricsType,
    pred: Sequence[Detection],
    true: Sequence[Detection],
    keypoint_labelmap: Optional[Dict[int, str]] = None,
) -> RunningKpMetricsType:
    # NOTE: This isn't as efficient as it could be, since we repeat the
    # 'linear_sum_assignment' calculation with box IoU.  However, for bounding boxes
    # that should be a very fast calculation, so leaving this as a future improvement.
    assignments = get_detection_assignments(pred, true, iou_fn=box_iou)
    for _pred, _true, _ in assignments:
        metrics = _update_running_kp_metrics_single(
            metrics, _pred, _true, keypoint_labelmap=keypoint_labelmap
        )
    return metrics


# Format for keypoint JSON metrics:
# {
#    "distance": 0.0,
#    "class_distance": {
#       "class_1": {
#          "distance": 0.0,
#          "keypoint_distance": {
#             "1": 0.0,
#             "2": 0.0,
#             ...
#          }
#       },
#       ...
#    }
# }


class CocoEvaluator:
    def __init__(
        self,
        iou_thresholds: Optional[Sequence[float]] = None,
        labelmap: Optional[Dict[int, str]] = None,
        keypoint_labelmap: Optional[Dict[int, str]] = None,
    ):
        self.labelmap = labelmap
        self.keypoint_labelmap = keypoint_labelmap
        if iou_thresholds is None:
            iou_thresholds = np.arange(0.5, 1.0, 0.05).tolist()

        self.iou_thresholds = iou_thresholds
        # 'self.box_counts' is a list of counts for each class label -- one
        # for each IoU threshold.  Same for 'self.mask_counts'.
        self.box_counts: Dict[str, List[Counts]] = {}
        self.mask_counts: Dict[str, List[Counts]] = {}
        self.keypoint_metrics: Dict[str, RunningKpMetricsType] = {}
        self.metrics: Dict[str, Dict] = {}

    def _update_one_sample(self, pred: List[Detection], true: List[Detection]):
        pred_labels = set(p.label for p in pred)
        true_labels = set(t.label for t in true)
        labels = pred_labels.union(true_labels)

        for label in labels:
            _pred = [p for p in pred if p.label == label]
            _true = [t for t in true if t.label == label]
            key = self.labelmap[label] if self.labelmap else str(label)
            assert isinstance(key, str)

            if key not in self.box_counts:
                self.box_counts[key] = [Counts() for _ in self.iou_thresholds]
            new_box_counts = get_counts_per_iou_threshold(
                _pred, _true, iou_thresholds=self.iou_thresholds
            )
            self.box_counts[key] = [
                counts + updates
                for counts, updates in zip(self.box_counts[key], new_box_counts)
            ]

            # Determine whether to evaluate mask mAP/mAR
            target_masks = [t.mask for t in _true if t.mask is not None]
            if len(target_masks) > 0:
                if key not in self.mask_counts:
                    self.mask_counts[key] = [Counts() for _ in self.iou_thresholds]
                new_mask_counts = get_counts_per_iou_threshold(
                    _pred, _true, iou_thresholds=self.iou_thresholds, iou_fn=mask_iou
                )
                self.mask_counts[key] = [
                    counts + updates
                    for counts, updates in zip(self.mask_counts[key], new_mask_counts)
                ]

            # Determine whether to evaluate keypoint metrics
            target_keypoints = [t.keypoints for t in _true if t.keypoints is not None]
            if len(target_keypoints) > 0:
                self.keypoint_metrics[key] = update_running_kp_metrics(
                    self.keypoint_metrics.get(key, {}),
                    _pred,
                    _true,
                    keypoint_labelmap=self.keypoint_labelmap,
                )

    def update(
        self,
        pred: List[Dict[str, np.ndarray]],
        true: List[Dict[str, np.ndarray]],
    ):
        for _pred, _true in zip(pred, true):
            self._update_one_sample(parse_predictions(_pred), parse_predictions(_true))

    @staticmethod
    def _accumulate_detection_metrics(
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

    @staticmethod
    def _accumulate_keypoint_metrics(
        metrics: Dict[str, RunningKpMetricsType]
    ) -> Dict[str, Any]:
        class_distance = {
            key: _get_class_kp_metrics(value) for key, value in metrics.items()
        }
        return {
            "class_distance": class_distance,
            "distance": np.mean(
                [v["distance"] for v in class_distance.values()]
            ).item(),
        }

    def accumulate(self):
        self.metrics = {"box": self._accumulate_detection_metrics(self.box_counts)}
        if self.mask_counts:
            self.metrics["mask"] = self._accumulate_detection_metrics(self.mask_counts)
        if self.keypoint_metrics:
            self.metrics["keypoints"] = self._accumulate_keypoint_metrics(
                self.keypoint_metrics
            )

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


def _get_class_kp_metrics(metrics: RunningKpMetricsType) -> Dict[str, Any]:
    """Returns a dictionary of keypoint metrics for a single class.

    Args:
        metrics: A dictionary of running keypoint metrics.

    Returns:
        A dictionary of keypoint metrics for a single class.
    """
    keypoint_distance = {
        key: np.mean(value.distance).item() for key, value in metrics.items()
    }
    return {
        "distance": np.mean([v for v in keypoint_distance.values()]).item(),
        "keypoint_distance": keypoint_distance,
    }
