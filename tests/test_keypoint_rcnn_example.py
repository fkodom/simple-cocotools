from scripts.keypoint_rcnn_example import main


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

    # Format for the keypoints metrics:
    # {
    #     "distance": 0.0
    #     "class_distance": {
    #         "0": {
    #             "distance": 0.0,
    #             "keypoint_distance": {
    #                 "0": 0.0,
    #                 "1": 0.0,
    #                 ...
    #             }
    #         },
    #         ...
    #     }
    # }

    assert "keypoints" in metrics
    keypoints = metrics["keypoints"]
    assert isinstance(keypoints, dict)
    assert "distance" in keypoints
    assert "class_distance" in keypoints
    class_distance = keypoints["class_distance"]
    assert isinstance(class_distance, dict)
    # Check that only the "person" (1) keypoint class is present
    assert len(class_distance) == 1
    assert "1" in class_distance
    # Check that the "person" class has the expected keys for each keypoint
    class_0_distance = class_distance["1"]
    assert "distance" in class_0_distance
    assert "keypoint_distance" in class_0_distance
    keypoint_distance = class_0_distance["keypoint_distance"]
    assert isinstance(keypoint_distance, dict)
    # Pretrained human pose model has 17 keypoints.
    assert len(keypoint_distance) == 17
