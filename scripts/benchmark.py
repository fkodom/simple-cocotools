import json
import os
import sys
import time
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torchvision.models.detection
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from simple_cocotools.evaluator import CocoEvaluator
from simple_cocotools.utils.coco import CocoDetection2017
from simple_cocotools.utils.data import default_collate_fn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def transform_to_tensors(image: Image.Image, targets: dict[str, np.ndarray]):
    out_image = to_tensor(image)
    out_targets = {k: torch.as_tensor(v) for k, v in targets.items()}
    return out_image, out_targets


def predict(
    model: nn.Module, images: Sequence[Tensor], score_threshold: float = 0.5
) -> list[dict[str, np.ndarray]]:
    with torch.no_grad(), torch.autocast("cuda"):
        predictions = model([image.to(DEVICE) for image in images])

    out = []
    for pred in predictions:
        thresholded = pred["scores"] > score_threshold
        selected = {
            "scores": pred["scores"][thresholded].cpu().numpy(),
            "labels": pred["labels"][thresholded].cpu().numpy(),
            "boxes": pred["boxes"][thresholded].cpu().numpy(),
        }
        if "masks" in pred:
            selected["masks"] = pred["masks"][thresholded].cpu().numpy()
        out.append(selected)

    return out


def main(
    split: str = "minival",
    model: str = "maskrcnn_resnet50_fpn",
    save_path: Optional[str] = None,
    batch_size: int = 8,
    max_batches: int = sys.maxsize,
    score_threshold: float = 0.5,
) -> dict[str, Any]:
    model_fn = getattr(torchvision.models.detection, model)
    detection_model: nn.Module = model_fn(pretrained=True)
    detection_model.eval().to(DEVICE)

    dataset = CocoDetection2017(split=split, transforms=transform_to_tensors)
    dataloader = DataLoader(  # type: ignore
        dataset,  # type: ignore
        batch_size=batch_size,
        collate_fn=default_collate_fn,
    )

    labelmap = {idx: info["name"] for idx, info in dataset.coco.cats.items()}
    evaluator = CocoEvaluator(labelmap=labelmap)

    num_batches = min(len(dataloader), max_batches)
    evaluation_time: float = 0.0
    for i, (images, targets) in enumerate(tqdm(dataloader, total=num_batches)):
        if i >= num_batches:
            break
        predictions = predict(detection_model, images, score_threshold=score_threshold)
        evaluation_start = time.time()
        evaluator.update(predictions, targets)
        evaluation_time += time.time() - evaluation_start

    summarize_state = time.time()
    metrics = evaluator.summarize()
    evaluation_time += time.time() - summarize_state

    total_samples = num_batches * batch_size
    print(f"Total evaluation time: {evaluation_time:.3f} s")
    print(f"Total samples: {total_samples}")
    print(f"Evaluation samples/sec: {total_samples / evaluation_time:.3f} s")

    if save_path:
        dirname, _ = os.path.split(save_path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(metrics, f, indent=4)

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="minival")
    parser.add_argument("--model", type=str, default="maskrcnn_resnet50_fpn")
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=sys.maxsize)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    args = parser.parse_args()

    main(**vars(args))
