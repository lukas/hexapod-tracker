"""Off-robot synthetic tests for housing-marker pose estimation.

Run locally:
    uv run pytest linux_control/test_housing_pose.py
"""
from __future__ import annotations

import math
import numpy as np
from scipy.spatial.transform import Rotation

from hexapod_tracker.housing_pose import (
    HousingPoseEstimator,
    JOINT_NAMES,
    RigidTransform,
    TagMount,
    forward_frame_transforms,
)


def _angles() -> list[float]:
    values: list[float] = []
    for leg in range(6):
        values.extend([
            -18.0 + 6.0 * leg,
            -9.0 + 5.0 * leg,
            23.0 + 7.0 * leg,
        ])
    return values


def _identity_estimator() -> HousingPoseEstimator:
    mounts = {
        index: TagMount(index, frame, RigidTransform.identity())
        for index, frame in enumerate(
            ["body"] + [
                f"L{leg}_{segment}"
                for leg in range(6)
                for segment in ("coxa", "femur", "tibia")
            ]
        )
    }
    return HousingPoseEstimator(
        cameras={"camera0": RigidTransform.identity()},
        tag_mounts=mounts,
    )


def test_recovers_body_and_all_18_angles_from_frame_transforms() -> None:
    body = RigidTransform(
        np.array([0.28, -0.17, 0.64]),
        Rotation.from_euler("xyz", [7.0, -11.0, 31.0], degrees=True),
    )
    expected = _angles()
    frames = forward_frame_transforms(body, expected)
    result = _identity_estimator().estimate_frame_transforms(frames)

    assert result["ok"]
    assert result["complete"]
    assert result["unobservable_joints"] == []
    assert np.allclose(result["joint_vector_deg"], expected, atol=1e-8)
    estimated_body = RigidTransform.from_dict(
        result["body_pose"]["world_from_body"]
    )
    assert np.allclose(estimated_body.translation_m, body.translation_m)
    error_deg = math.degrees(float(
        (estimated_body.rotation.inv() * body.rotation).magnitude()
    ))
    assert error_deg < 1e-7


def test_motor_housings_recover_yaw_and_hip_but_not_knee() -> None:
    body = RigidTransform.identity()
    expected = _angles()
    all_frames = forward_frame_transforms(body, expected)
    housing_frames = {"body": all_frames["body"]}
    for leg in range(6):
        housing_frames[f"L{leg}_coxa"] = all_frames[f"L{leg}_coxa"]
        housing_frames[f"L{leg}_femur"] = all_frames[f"L{leg}_femur"]

    result = _identity_estimator().estimate_frame_transforms(housing_frames)

    assert result["ok"]
    assert not result["complete"]
    assert result["unobservable_joints"] == [
        f"L{leg}_knee" for leg in range(6)
    ]
    for leg in range(6):
        assert result["joints"][f"L{leg}_yaw"]["observable"]
        assert result["joints"][f"L{leg}_hip"]["observable"]
        assert not result["joints"][f"L{leg}_knee"]["observable"]


def test_absolute_tibia_angles_over_90_degrees_do_not_flip_yaw() -> None:
    expected = _angles()
    for leg in range(6):
        expected[leg * 3 + 2] = 105.0 + 7.0 * leg
    frames = forward_frame_transforms(RigidTransform.identity(), expected)

    result = _identity_estimator().estimate_frame_transforms(frames)

    assert result["complete"]
    assert np.allclose(result["joint_vector_deg"], expected, atol=1e-8)


def test_applies_camera_and_tag_mount_transforms() -> None:
    world_from_camera = RigidTransform(
        np.array([0.4, -0.2, 1.1]),
        Rotation.from_euler("xyz", [160.0, 4.0, -25.0], degrees=True),
    )
    body = RigidTransform(
        np.array([0.05, 0.08, 0.17]),
        Rotation.from_euler("xyz", [3.0, -2.0, 14.0], degrees=True),
    )
    expected = _angles()
    frames = forward_frame_transforms(body, expected)
    frame_from_tag = RigidTransform(
        np.array([0.011, -0.007, 0.004]),
        Rotation.from_euler("xyz", [0.0, 0.0, 90.0], degrees=True),
    )
    ordered_frames = ["body"] + [
        f"L{leg}_{segment}"
        for leg in range(6)
        for segment in ("coxa", "femur", "tibia")
    ]
    mounts = {
        tag_id: TagMount(tag_id, frame, frame_from_tag)
        for tag_id, frame in enumerate(ordered_frames)
    }
    estimator = HousingPoseEstimator(
        cameras={"cam": world_from_camera},
        tag_mounts=mounts,
    )
    camera_from_world = world_from_camera.inverse()
    detections = []
    for tag_id, frame in enumerate(ordered_frames):
        world_from_tag = frames[frame].compose(frame_from_tag)
        detections.append({
            "tag_id": tag_id,
            "camera": "cam",
            "camera_from_tag": camera_from_world.compose(world_from_tag).to_dict(),
        })

    result = estimator.estimate_detections(detections)

    assert result["complete"]
    assert np.allclose(result["joint_vector_deg"], expected, atol=1e-6)


def test_visual_minus_encoder_is_reported_not_applied() -> None:
    body = RigidTransform.identity()
    expected = _angles()
    frames = forward_frame_transforms(body, expected)
    encoder = dict(zip(JOINT_NAMES, (value - 2.5 for value in expected)))

    result = _identity_estimator().estimate_frame_transforms(
        frames, encoder_joint_deg=encoder
    )

    assert result["complete"]
    assert all(
        abs(value - 2.5) < 1e-6
        for value in result["visual_minus_encoder_deg"].values()
    )


def test_missing_body_is_a_clean_observability_failure() -> None:
    frames = forward_frame_transforms(RigidTransform.identity(), _angles())
    del frames["body"]

    result = _identity_estimator().estimate_frame_transforms(frames)

    assert not result["ok"]
    assert result["unobservable_joints"] == list(JOINT_NAMES)
