from scripts.mask_rcnn_example import main


def test_main():
    metrics = main(max_samples=8)
    assert "box" in metrics
    box = metrics["box"]
    assert isinstance(box, dict)
    assert "mAP" in box
    assert "mAR" in box
    assert "class_AP" in box
    assert isinstance(box["class_AP"], dict)
    assert len(box["class_AP"]) > 0
    assert "class_AR" in box
    assert isinstance(box["class_AR"], dict)
    assert len(box["class_AR"]) > 0

    assert "mask" in metrics
    mask = metrics["mask"]
    assert isinstance(mask, dict)
    assert "mAP" in mask
    assert "mAR" in mask
    assert "class_AP" in mask
    assert isinstance(mask["class_AP"], dict)
    assert len(mask["class_AP"]) > 0
    assert "class_AR" in mask
    assert isinstance(mask["class_AR"], dict)
    assert len(mask["class_AR"]) > 0
