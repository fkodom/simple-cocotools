from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from sweet_pipes.coco.coco_keypoints import coco_keypoints
from torch import Tensor, nn
from torch.utils.data import DataLoader2
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm

from simple_cocotools.evaluator import CocoEvaluator
from simple_cocotools.utils.data import default_collate_fn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def transform_to_tensors(
    batch: Tuple[Image.Image, Dict[str, np.ndarray]]
) -> Tuple[Tensor, Dict[str, Tensor]]:
    image, targets = batch
    out_image = to_tensor(image)
    out_targets = {k: torch.as_tensor(v) for k, v in targets.items()}
    return out_image, out_targets


def predict(model: nn.Module, images: Sequence[Tensor]) -> List[Dict[str, np.ndarray]]:
    with torch.no_grad(), torch.autocast("cuda"):
        predictions = model([image.to(DEVICE) for image in images])

    out = []
    for pred in predictions:
        thresholded = pred["scores"] > 0.5
        selected = {
            "scores": pred["scores"][thresholded].cpu().numpy(),
            "labels": pred["labels"][thresholded].cpu().numpy(),
            "boxes": pred["boxes"][thresholded].cpu().numpy(),
            "keypoints": pred["keypoints"][thresholded].cpu().numpy(),
            "keypoints_scores": pred["keypoints_scores"][thresholded].cpu().numpy(),
        }
        out.append(selected)

    return out


def main(max_samples: Optional[int] = None) -> Dict[str, Any]:
    detection_model = keypointrcnn_resnet50_fpn(pretrained=True).eval().to(DEVICE)

    pipe = coco_keypoints(split="val")
    pipe = pipe.map(transform_to_tensors)
    dataloader = DataLoader2(pipe, collate_fn=default_collate_fn)
    evaluator = CocoEvaluator()

    for i, (images, targets) in enumerate(tqdm(dataloader)):
        if max_samples and i >= max_samples:
            break

        predictions = predict(detection_model, images)
        evaluator.update(predictions, targets)

    return evaluator.summarize()


if __name__ == "__main__":
    main(max_samples=512)
