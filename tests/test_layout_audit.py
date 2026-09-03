import json

import cv2
import numpy as np

from hexapod_tracker.layout_audit import audit_images, validate_layout
from hexapod_tracker.paths import CONFIG_DIR


def load_config(name):
    return json.loads((CONFIG_DIR / name).read_text(encoding="utf-8"))


def test_hexapod_1_layout_matches_consumer_configs():
    layout = load_config("hexapod-1-apriltag-layout.json")
    problems = validate_layout(
        layout,
        floor_map=load_config("floor_tag_map.json"),
        part_map=load_config("hexapod_tag_map.json"),
    )

    assert problems == []
    assert len(layout["robot_tags"]) == 37
    assert len({tag["id"] for tag in layout["robot_tags"]}) == 37
    assert len(layout["floor"]["tags"]) == 7


def test_audit_preserves_duplicate_detections_and_writes_annotation(tmp_path):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = cv2.aruco.generateImageMarker(dictionary, 7, 120)
    image = np.full((240, 400), 255, dtype=np.uint8)
    image[40:160, 40:160] = marker
    image[40:160, 240:360] = marker
    image_path = tmp_path / "duplicates.png"
    assert cv2.imwrite(str(image_path), image)
    layout = {
        "name": "test layout",
        "tag_family": "tag36h11",
        "robot_tags": [{"id": 7}],
        "floor": {"tags": []},
    }

    report = audit_images(layout, [image_path], output_dir=tmp_path / "annotations")

    assert report["detected_ids"] == [7]
    assert report["missing_ids"] == []
    assert report["unexpected_ids"] == []
    assert report["images"][0]["duplicate_ids"] == [7]
    detections = report["images"][0]["detections"]
    assert len(detections) == 2
    assert all(detection["area_px2"] > 10_000 for detection in detections)
    assert (tmp_path / "annotations/duplicates-annotated.jpg").is_file()
