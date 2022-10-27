from __future__ import annotations

import os
from abc import abstractproperty
from dataclasses import dataclass
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List, Optional, Tuple
from zipfile import ZipFile

import numpy as np
import wget
from PIL import Image
from pycocotools import coco
from tqdm import tqdm


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
        split = "val" if self.split == "minival" else self.split
        return f"http://images.cocodataset.org/zips/{split}2014.zip"

    @property
    def prefix(self) -> str:
        split = "val" if self.split == "minival" else self.split
        return f"{split}2014/"


@dataclass
class CocoInstances(CocoDataResource):
    @property
    def url(self) -> str:
        if self.split == "minival":
            return "https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip?dl=0"
        else:
            return (
                "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
            )

    @property
    def prefix(self) -> str:
        if self.split == "minival":
            return "instances_minival2014.json"
        else:
            return f"annotations/instances_{self.split}2014.json"


def bbox_voc_to_coco_format(bbox: np.ndarray) -> np.ndarray:
    x, y, w, h = bbox
    return np.array([x, y, x + w, y + h])


class AnnotationsToDetectionFormat:
    def __init__(self, coco_api: coco.COCO):
        self.coco_api = coco_api

    def __call__(self, annotations: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        return {
            "labels": np.array([a["category_id"] for a in annotations], dtype=np.int32),
            "boxes": np.array(
                [bbox_voc_to_coco_format(a["bbox"]) for a in annotations],
                dtype=np.float32,
            ),
            "masks": np.array(
                [self.coco_api.annToMask(a) for a in annotations], dtype=np.int32
            ),
            "area": np.array([a["area"] for a in annotations], dtype=np.float32),
            "iscrowd": np.array([a["iscrowd"] for a in annotations], dtype=np.uint8),
        }


class CocoDetection:
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
        transforms: Optional[Callable] = None,
    ) -> None:
        self.root = root
        self.transforms = transforms
        from pycocotools.coco import COCO

        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.ids)


class CocoDetection2014(CocoDetection):
    def __init__(
        self,
        split: str,
        root: str = "data/coco2014",
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
        self.target_transform = AnnotationsToDetectionFormat(coco_api=self.coco)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self.target_transform(
            [
                annotation
                for annotation in self._load_target(id)
                if int(annotation["id"]) not in self.skip_annotation_ids
            ]
        )

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
