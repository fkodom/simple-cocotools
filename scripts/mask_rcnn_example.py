from typing import Any, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
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


def predict(model: nn.Module, images: Sequence[Tensor]) -> list[dict[str, np.ndarray]]:
    with torch.no_grad(), torch.autocast("cuda"):
        predictions = model([image.to(DEVICE) for image in images])

    out = []
    for pred in predictions:
        thresholded = pred["scores"] > 0.5
        selected = {
            "scores": pred["scores"][thresholded].cpu().numpy(),
            "labels": pred["labels"][thresholded].cpu().numpy(),
            "boxes": pred["boxes"][thresholded].cpu().numpy(),
            "masks": pred["masks"][thresholded].cpu().numpy(),
        }
        out.append(selected)

    return out


def main(max_samples: Optional[int] = None) -> dict[str, Any]:
    detection_model = maskrcnn_resnet50_fpn(pretrained=True).eval().to(DEVICE)

    dataset = CocoDetection2017(split="val", transforms=transform_to_tensors)
    dataloader = DataLoader(  # type: ignore
        dataset,  # type: ignore
        collate_fn=default_collate_fn,
    )

    labelmap = {idx: info["name"] for idx, info in dataset.coco.cats.items()}
    evaluator = CocoEvaluator(labelmap=labelmap)

    for i, (images, targets) in enumerate(tqdm(dataloader)):
        if max_samples and i >= max_samples:
            break

        predictions = predict(detection_model, images)
        evaluator.update(predictions, targets)

    return evaluator.summarize()


if __name__ == "__main__":
    main()
