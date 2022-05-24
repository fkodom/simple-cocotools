# simple-cocotools

[EXPERIMENTAL]

A simple, modern alternative to `pycocotools`.


## About

Why not just use [Pycocotools](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools)?

* Code is more readable and hackable.
* Metrics are more transparent and understandable.
* This project only depends on `numpy` and `scipy`. No `cython` extensions. 
* Code is more modern (type annotations, linting, etc).
* Evaluation is fast. Typically much faster than the detection model being evaluated.
    * **Bbox:** ~350 samples/second
    * **Bbox + mask:** ~100 samples/second
    * Benchmarked on COCO minival with predictions from pre-trained Faster R-CNN and Mask R-CNN models, respectively.
    * Using a Google Cloud `n1-standard-4` VM (4 vCPUs, 16 GB RAM).
    * ** Speeds are dependent on the number of detections per image (therefore dependent on the model and thresholding). **


## Install

**TODO:** Publish as a PyPI package and update instructions here.

### From Repo
```bash
pip install "simple-cocotools @ git+ssh://git@github.com/fkodom/simple-cocotools.git"
```

### For Contributors
```bash
# Install all dev dependencies (tests etc.)
pip install "simple-cocotools[all] @ git+ssh://git@github.com/fkodom/simple-cocotools.git"

# Setup pre-commit hooks
pre-commit install
```


## Usage

Expects target annotations to have the same format as model predictions. (The format used by all `torchvision` detection models.)  You probably already have code to convert annotations into this format, since it's required to train most detection models.

A minimal example:

```python
from torchvision.detection.models import fasterrcnn_resnet50_fpn
from simple_cocotools import CocoEvaluator

evaluator = CocoEvaluator()
model = fasterrcnn_resnet50_fpn(pretrained=True).eval()

for images, targets in data_loader:
    predictions = model(images)
    evaluator.update(predictions, targets)

metrics = evaluator.summarize()
```

For a more complete example, see [`scripts/faster_rcnn_example.py`](./scripts/faster_rcnn_example.py).


## Benchmarks

I benchmarked against several `torchvision` detection models, which have [mAP scores reported on the PyTorch website](https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection).

Using a default score threshold of 0.5:

Model        | Backbone          | box mAP<br>(official) | box mAP | box mAR | mask mAP<br>(official) | mask mAP | mask mAR 
-------------|-------------------|-----------------------|---------|---------|------------------------|----------|----------
Mask R-CNN   | ResNet50          | 37.9                  | 36.9    | 43.2    | 34.6                   | 34.1     | 40.0
Faster R-CNN | ResNet50          | 37.0                  | 36.3    | 42.0    | -                      | -        | -
Faster R-CNN | MobileNetV3-Large | 32.8                  | 39.9    | 35.0    | -                      | -        | -

Notice that the mAP for `MobileNetV3-Large` is artificially high, since it has a much lower mAR at that score threshold.  After tuning the score threshold, so that mAP and mAR are more balanced:  

Model        | Backbone          | Threshold | box mAP | box mAR | mask mAP | mask mAR 
-------------|-------------------|-----------|---------|---------|----------|----------
Mask R-CNN   | ResNet50          | 0.6       | 41.1    | 41.3    | 38.2     | 38.5
Faster R-CNN | ResNet50          | 0.6       | 40.8    | 40.4    | -        | -
Faster R-CNN | MobileNetV3-Large | 0.425     | 36.2    | 36.2    | -        | -

These scores are more reflective of model performance, in my opinion.  Mask R-CNN slightly outperforms Faster R-CNN, and there is a noticeable (but not horrible) gap between ResNet50 and MobileNetV3 backbones.

**Note:** PyTorch does not mention what score thresholds were used for each model benchmark. ¯\\_(ツ)_/¯
