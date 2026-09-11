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
    ids = [tag["id"] for tag in layout["robot_tags"]]
    assert len(ids) == len(set(ids))
    assert sum(tag["kind"] == "chassis_tag" for tag in layout["robot_tags"]) == 1
    # Every mount is either carried by a tag or declared as a gap; 37 is the
    # full complement (1 chassis + 12 lids + 24 yoke faces).
    assert len(ids) + len(layout.get("unresolved_mounts", [])) == 37
    assert len(layout["floor"]["tags"]) == 7


def _minimal_layout(**overrides):
    tags = [{"id": 0, "kind": "chassis_tag", "frame": "body", "surface": "horizontal",
             "frame_from_tag": {"euler_xyz_deg": [0, 0, 0]}}]
    next_id = 1
    for leg in range(6):
        for joint in ("hip", "knee"):
            tags.append({"id": next_id, "kind": "servo_lid", "leg": leg, "joint": joint,
                         "frame": f"L{leg}_{'coxa' if joint == 'hip' else 'femur'}",
                         "surface": "horizontal", "frame_from_tag": {"euler_xyz_deg": [0, 0, 90]}})
            next_id += 1
            for side in ("+y", "-y"):
                tags.append({"id": next_id, "kind": "yoke_face", "leg": leg, "joint": joint,
                             "frame": f"L{leg}_{'femur' if joint == 'hip' else 'tibia'}",
                             "mount_side": side,
                             "frame_from_tag": {"quaternion_xyzw": [0.5, 0.5, 0.5, -0.5],
                                                "tag_axes_in_frame": {"x": "+z", "y": "+x", "z": "+y"}}})
                next_id += 1
    layout = {"robot_tags": tags, "floor": {"tags": []}, "unresolved_mounts": []}
    layout.update(overrides)
    return layout


def test_declared_gap_is_a_note_and_undeclared_gap_is_a_problem():
    layout = _minimal_layout()
    assert validate_layout(layout) == []
    layout["robot_tags"] = [t for t in layout["robot_tags"]
                            if not (t.get("leg") == 4 and t.get("joint") == "hip" and t.get("mount_side") == "+y")]
    assert validate_layout(layout) == ["L4 hip yoke sides are ['-y']"]
    layout["unresolved_mounts"] = [{"leg": 4, "joint": "hip", "kind": "yoke_face", "mount_side": "+y",
                                    "reason": "no camera saw this face on 2026-09-11", "since": "2026-09-11"}]
    assert validate_layout(layout) == []
    # A declared gap does not excuse a different one, and needs a side for a face.
    layout["unresolved_mounts"] = [{"leg": 4, "joint": "knee", "kind": "yoke_face", "mount_side": "+y", "reason": "x"}]
    assert "L4 hip yoke sides are ['-y']" in validate_layout(layout)
    layout["unresolved_mounts"] = [{"leg": 4, "joint": "hip", "kind": "yoke_face", "reason": "x"}]
    assert any("needs a mount_side" in p for p in validate_layout(layout))


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
