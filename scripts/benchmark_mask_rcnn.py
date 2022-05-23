import json
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from simple_cocotools.evaluator import CocoEvaluator
from simple_cocotools.utils.coco import CocoDetection2014
from simple_cocotools.utils.data import default_collate_fn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def transform_to_tensors(image: Image.Image, targets: Dict[str, np.ndarray]):
    out_image = to_tensor(image)
    out_targets = {k: torch.as_tensor(v) for k, v in targets.items()}
    return out_image, out_targets


def predict(
    model: nn.Module, images: Sequence[Tensor], score_threshold: float = 0.5
) -> List[Dict[str, np.ndarray]]:
    with torch.no_grad(), torch.cuda.amp.autocast():
        predictions = model([image.to(DEVICE) for image in images])

    out = []
    for pred in predictions:
        thresholded = pred["scores"] > score_threshold
        selected = {
            "scores": pred["scores"][thresholded].cpu().numpy(),
            "labels": pred["labels"][thresholded].cpu().numpy(),
            "boxes": pred["boxes"][thresholded].cpu().numpy(),
            "masks": pred["masks"][thresholded].cpu().numpy(),
        }
        out.append(selected)

    return out


def main(
    save_path: Optional[str] = None, batch_size: int = 8, score_threshold: float = 0.5
) -> Dict[str, Any]:
    model = maskrcnn_resnet50_fpn(pretrained=True).eval().to(DEVICE)

    dataset = CocoDetection2014(split="minival", transforms=transform_to_tensors)
    dataloader = DataLoader(
        dataset,  # type: ignore
        batch_size=batch_size,
        collate_fn=default_collate_fn,
    )

    labelmap = {idx: info["name"] for idx, info in dataset.coco.cats.items()}
    evaluator = CocoEvaluator(labelmap=labelmap)

    for images, targets in tqdm(dataloader):
        predictions = predict(model, images, score_threshold=score_threshold)
        evaluator.update(predictions, targets)

    metrics = evaluator.summarize()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(metrics, f, indent=4)

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    args = parser.parse_args()

    main(**vars(args))
