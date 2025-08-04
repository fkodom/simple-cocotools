from __future__ import annotations

import logging
import os
from abc import abstractproperty
from dataclasses import dataclass
from functools import lru_cache
from tempfile import TemporaryDirectory
from typing import Any, Callable, Literal, Optional, Tuple
from zipfile import ZipFile

import numpy as np
import wget
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass  # type: ignore
class CocoDataResource:
    split: str

    @abstractproperty
    def url(self) -> str:
        pass

    @abstractproperty
    def prefix(self) -> str:
        pass

    def download(self, root: str) -> str:
        os.makedirs(root, exist_ok=True)
        out_path = os.path.join(root, self.prefix)
        if os.path.exists(out_path):
            return out_path

        with TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "images.zip")
            wget.download(self.url, path)
            with ZipFile(path) as zipfile:
                for file in tqdm(zipfile.namelist(), desc="Extracting"):
                    if file.startswith(self.prefix):
                        zipfile.extract(file, path=root)

        return out_path


@dataclass
class CocoImages(CocoDataResource):
    @property
    def url(self) -> str:
        return f"http://images.cocodataset.org/zips/{self.split}2017.zip"

    @property
    def prefix(self) -> str:
        return f"{self.split}2017/"


@dataclass
class CocoInstances(CocoDataResource):
    @property
    def url(self) -> str:
        return "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

    @property
    def prefix(self) -> str:
        return f"annotations/instances_{self.split}2017.json"


@dataclass
class CocoKeypoints(CocoDataResource):
    @property
    def url(self) -> str:
        return "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

    @property
    def prefix(self) -> str:
        return f"annotations/person_keypoints_{self.split}2017.json"


def bbox_voc_to_coco_format(bbox: np.ndarray) -> np.ndarray:
    x, y, w, h = bbox
    return np.array([x, y, x + w, y + h])


def keypoints_to_numpy(keypoints: list[int]) -> np.ndarray:
    out = np.array(keypoints).reshape(-1, 3)
    return out.astype(np.float32)


class AnnotationsToDetectionFormat:
    def __init__(self, coco_api: COCO, mode: Literal["detection", "keypoints"]):
        self.coco_api = coco_api
        self.mode = mode

    def __call__(self, annotations: list[dict[str, Any]]) -> dict[str, np.ndarray]:
        detections = {
            "labels": np.asarray(
                [a["category_id"] for a in annotations], dtype=np.int32
            ),
            "boxes": np.asarray(
                [bbox_voc_to_coco_format(a["bbox"]) for a in annotations],
                dtype=np.float32,
            ),
            "area": np.asarray([a["area"] for a in annotations], dtype=np.float32),
            "iscrowd": np.asarray([a["iscrowd"] for a in annotations], dtype=np.uint8),
        }
        if self.mode == "detection":
            try:
                detections["masks"] = np.asarray(
                    [self.coco_api.annToMask(a) for a in annotations], dtype=np.int32
                )
            except Exception:
                _warn_once(
                    "Failed to convert annotations to masks. "
                    "This may be due to missing segmentation data in the annotations."
                )
        elif self.mode == "keypoints":
            detections["keypoints"] = (
                np.asarray([keypoints_to_numpy(a["keypoints"]) for a in annotations]),
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return detections  # type: ignore[return-value]


class _CocoDataset:
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        annFile: str,
        mode: Literal["detection", "keypoints"],
        transforms: Optional[Callable] = None,
    ) -> None:
        self.root = root
        self.coco = COCO(annFile)
        self.mode = mode
        self.transforms = transforms

        self.ids = sorted(self.coco.imgs.keys())
        self.target_transform = AnnotationsToDetectionFormat(
            coco_api=self.coco, mode=self.mode
        )

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id: int) -> list[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self.target_transform(
            [
                annotation
                for annotation in self._load_target(id)
                if int(annotation["id"]) not in getattr(self, "skip_annotation_ids", {})
            ]
        )

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.ids)


class CocoDetectionDataset(_CocoDataset):
    def __init__(
        self,
        root: str,
        annFile: str,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(
            root=root,
            annFile=annFile,
            transforms=transforms,
            mode="detection",
        )


class CocoKeypointsDataset(_CocoDataset):
    def __init__(
        self,
        root: str,
        annFile: str,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(
            root=root,
            annFile=annFile,
            transforms=transforms,
            mode="keypoints",
        )


class CocoDetection2017(CocoDetectionDataset):
    def __init__(
        self,
        split: str,
        root: str = "data/coco2017",
        transforms: Optional[Callable] = None,
    ):
        images = CocoImages(split=split)
        images.download(root=root)
        images_dir = os.path.join(root, images.prefix)

        anns = CocoInstances(split=split)
        anns.download(root=root)
        anns_file = os.path.join(root, anns.prefix)

        super().__init__(
            root=images_dir,
            annFile=anns_file,
            transforms=transforms,
        )
        self.skip_annotation_ids = [
            900100193200,
            900100259600,
            908400416500,
            900100463500,
        ]


class CocoKeypoints2017(CocoKeypointsDataset):
    def __init__(
        self,
        split: str,
        root: str = "data/coco2017",
        transforms: Optional[Callable] = None,
    ):
        images = CocoImages(split=split)
        images.download(root=root)
        images_dir = os.path.join(root, images.prefix)

        anns = CocoKeypoints(split=split)
        anns.download(root=root)
        anns_file = os.path.join(root, anns.prefix)

        super().__init__(
            root=images_dir,
            annFile=anns_file,
            transforms=transforms,
        )
        self.skip_annotation_ids = [
            900100193200,
            900100259600,
            908400416500,
            900100463500,
        ]


@lru_cache(maxsize=128)
def _warn_once(message: str) -> None:
    """Log a warning message only once."""
    logger.warning(message)
