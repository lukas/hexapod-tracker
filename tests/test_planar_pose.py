import math

import cv2
import numpy as np

from hexapod_tracker.planar_pose import PlanarPoseEstimator


def corners(center, yaw_degrees, size=27.0):
    half = size / 2.0
    offsets = np.asarray(
        [[-half, -half], [half, -half], [half, half], [-half, half]],
        dtype=np.float32,
    )
    yaw = math.radians(yaw_degrees)
    rotation = np.asarray(
        [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
        dtype=np.float32,
    )
    return np.asarray(center, dtype=np.float32) + offsets @ rotation.T


def test_planar_estimator_recovers_marker_and_part_pose():
    floor_map = {
        "tag_black_square_size": 27.0,
        "active_anchor_ids": [12, 13, 15],
        "coordinate_frame": {"origin": "tag 12"},
        "tags": [
            {"id": 12, "center": [0, 0, 0], "yaw_degrees": 0},
            {"id": 13, "center": [0, 600, 0], "yaw_degrees": 10},
            {"id": 15, "center": [300, 100, 0], "yaw_degrees": -20},
        ],
    }
    part_map = {
        "parts": [
            {
                "id": "test_part",
                "tag_ids": [16],
                "yaw_period_degrees": 180,
            }
        ]
    }
    homography = np.asarray(
        [[1.4, 0.15, 300], [-0.1, 1.1, 100], [0.0002, 0.0003, 1]],
        dtype=np.float32,
    )
    world_tags = {
        12: corners((0, 0), 0),
        13: corners((0, 600), 10),
        15: corners((300, 100), -20),
        16: corners((125, 240), 32),
    }
    image_tags = {
        tag_id: cv2.perspectiveTransform(points[None], homography)[0]
        for tag_id, points in world_tags.items()
    }
    payload = PlanarPoseEstimator(floor_map, part_map).estimate(
        [
            {
                "index": 0,
                "width": 1280,
                "height": 800,
                "frame_age_s": 0.01,
                "tags": image_tags,
            }
        ]
    )

    marker = payload["markers"]["16"]
    assert abs(marker["position_mm"]["x"] - 125.0) < 0.1
    assert abs(marker["position_mm"]["y"] - 240.0) < 0.1
    assert abs(marker["rotation_degrees"]["yaw"] - 32.0) < 0.01
    part = payload["parts"]["test_part"]
    assert part["status"] == "tracked"
    assert part["pose"]["position_mm"]["z"] is None
    assert abs(part["pose"]["rotation_degrees"]["yaw_axis"] - 32.0) < 0.01
    assert payload["calibration"]["cameras"]["0"]["quality"] == "good"


def test_planar_estimator_marks_camera_without_anchor_uncalibrated():
    floor_map = {
        "tag_black_square_size": 27.0,
        "active_anchor_ids": [12],
        "tags": [{"id": 12, "center": [0, 0, 0], "yaw_degrees": 0}],
    }
    part_map = {"parts": [{"id": "part", "tag_ids": [16]}]}
    payload = PlanarPoseEstimator(floor_map, part_map).estimate(
        [{"index": 3, "width": 1920, "height": 1080, "tags": {}, "frame_age_s": 0.1}]
    )

    assert payload["calibration"]["cameras"]["3"]["status"] == "uncalibrated"
    assert payload["parts"]["part"]["status"] == "not_visible"
