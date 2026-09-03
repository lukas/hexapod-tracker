"""Off-robot tests for calibrated AprilTag vision.

Run locally:
    uv run --extra dev pytest tests/test_apriltag_vision.py
"""
from __future__ import annotations

import json
import math
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from hexapod_tracker.apriltag_vision import (
    AprilTagPoseTracker,
    CameraCalibration,
    TagCorners,
    TemporalTagCornerTracker,
    _TagPoseSolution,
    _scale_tag_corners,
    detect_tag_corners,
    estimate_world_reference,
    marker_object_corners,
)
from hexapod_tracker.foot_tip_tracking import FootTipTracker
from hexapod_tracker.housing_pose import RigidTransform
from hexapod_tracker.paths import CONFIG_DIR
from hexapod_tracker.track import (
    FeedbackClient,
    _camera_order_after,
    _parse_camera_cycle,
    _safe_pose_assessment,
)
from hexapod_tracker import track as track_cli


def test_detects_generated_tag36h11_and_decodes_orientation() -> None:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = cv2.aruco.generateImageMarker(dictionary, 7, 240, borderBits=1)
    canvas = np.full((400, 400), 255, dtype=np.uint8)
    canvas[80:320, 80:320] = marker

    detections = detect_tag_corners(canvas)

    assert [item.tag_id for item in detections] == [7]
    assert np.allclose(detections[0].center_px, [199.5, 199.5], atol=1.0)
    assert abs(detections[0].tag_y_clockwise_from_image_up_deg) < 0.5


def test_full_resolution_tag_corners_scale_to_preview_coordinates() -> None:
    detections = [TagCorners(7, np.asarray([
        [300.0, 450.0],
        [600.0, 450.0],
        [600.0, 750.0],
        [300.0, 750.0],
    ], dtype=np.float32))]

    scaled = _scale_tag_corners(
        detections,
        source_size=(1920, 1440),
        target_size=(1280, 960),
    )

    assert scaled[0].tag_id == 7
    assert np.allclose(
        scaled[0].corners_px,
        np.asarray([
            [200.0, 300.0],
            [400.0, 300.0],
            [400.0, 500.0],
            [200.0, 500.0],
        ]),
    )


def test_scales_intrinsics_only_for_same_aspect_ratio() -> None:
    calibration = CameraCalibration.from_dict({
        "image_size_px": [1000, 500],
        "camera_matrix": [[800, 0, 500], [0, 810, 250], [0, 0, 1]],
        "distortion_coefficients": [0, 0, 0, 0, 0],
    })
    matrix, _ = calibration.for_image(2000, 1000)
    assert np.allclose(matrix, [[1600, 0, 1000], [0, 1620, 500], [0, 0, 1]])
    try:
        calibration.for_image(1920, 1080)
    except ValueError as error:
        assert "aspect ratios differ" in str(error)
    else:
        raise AssertionError("mismatched aspect ratio should fail")


def test_scales_rotated_center_crop_for_landscape_phone_video() -> None:
    calibration = CameraCalibration.from_dict({
        "image_size_px": [4284, 5712],
        "camera_matrix": [
            [3960.0, 0.0, 2142.0],
            [0.0, 3960.0, 2856.0],
            [0.0, 0.0, 1.0],
        ],
        "distortion_coefficients": [0, 0, 0, 0, 0],
        "allow_center_crop": True,
        "allow_quarter_turn": True,
    })

    matrix, _ = calibration.for_image(1920, 1080)

    assert np.allclose(matrix[0, 0], 1331.0924, atol=0.01)
    assert np.allclose(matrix[1, 1], 1331.0924, atol=0.01)
    assert np.allclose(matrix[:2, 2], [959.664, 540.0], atol=0.5)


def test_recovers_camera_pose_from_multiple_floor_tags() -> None:
    marker_size = 0.04
    camera_matrix = np.asarray([
        [900.0, 0.0, 640.0],
        [0.0, 900.0, 360.0],
        [0.0, 0.0, 1.0],
    ])
    distortion = np.zeros(5)
    world_from_camera = RigidTransform(
        np.asarray([0.12, -0.08, 1.15]),
        Rotation.from_euler("xyz", [180.0, 0.0, 8.0], degrees=True),
    )
    camera_from_world = world_from_camera.inverse()
    floor_tags = {
        12: RigidTransform.identity(),
        13: RigidTransform(
            np.asarray([0.48, 0.22, 0.0]),
            Rotation.from_euler("z", 17.0, degrees=True),
        ),
        15: RigidTransform(
            np.asarray([-0.31, 0.37, 0.0]),
            Rotation.from_euler("z", -73.0, degrees=True),
        ),
    }
    detections: list[TagCorners] = []
    for tag_id, world_from_tag in floor_tags.items():
        world_points = np.stack([
            world_from_tag.apply(point)
            for point in marker_object_corners(marker_size)
        ])
        camera_points = np.stack([
            camera_from_world.apply(point) for point in world_points
        ])
        pixels = np.column_stack([
            camera_matrix[0, 0] * camera_points[:, 0] / camera_points[:, 2]
            + camera_matrix[0, 2],
            camera_matrix[1, 1] * camera_points[:, 1] / camera_points[:, 2]
            + camera_matrix[1, 2],
        ]).astype(np.float32)
        detections.append(TagCorners(tag_id, pixels))

    reference = estimate_world_reference(
        detections,
        floor_tags,
        camera_matrix,
        distortion,
        marker_size_m=marker_size,
    )

    assert reference is not None
    assert reference.floor_tag_ids == (12, 13, 15)
    assert reference.reprojection_rms_px < 1e-3
    assert np.allclose(
        reference.world_from_camera.translation_m,
        world_from_camera.translation_m,
        atol=1e-6,
    )
    rotation_error = (
        reference.world_from_camera.rotation.inv() * world_from_camera.rotation
    ).magnitude()
    assert math.degrees(float(rotation_error)) < 1e-4


def test_as_photographed_config_maps_handwritten_zero_to_l0() -> None:
    config_path = CONFIG_DIR / "apriltag_pose_config_20260831.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    tags = config["robot_pose"]["tags"]

    assert tags["1"]["frame"] == "L0_coxa"
    assert tags["7"]["frame"] == "L0_femur"
    assert "handwritten 0" in tags["1"]["label"]
    assert set(map(int, config["floor_tags"])) == {12, 13, 15}
    assert config["marker_size_m"] == 0.027
    assert config["marker_size_verified"] is True


def test_planar_branch_uses_encoder_only_to_reject_large_hip_flip() -> None:
    tracker = AprilTagPoseTracker.from_json(
        CONFIG_DIR / "apriltag_pose_config_20260831.json"
    )
    assert tracker._branch_estimator is not None
    body_mount = tracker._branch_estimator.tag_mounts[0]
    femur_mount = tracker._branch_estimator.tag_mounts[7]
    body_from_body = RigidTransform.identity()
    body_from_femur_zero = RigidTransform(
        np.zeros(3),
        Rotation.from_euler("z", 30.0, degrees=True),
    )
    body_from_femur_flipped = RigidTransform(
        np.zeros(3),
        Rotation.from_euler("z", 30.0, degrees=True)
        * Rotation.from_euler("y", 48.0, degrees=True),
    )

    def solution(
        frame: RigidTransform,
        mount: RigidTransform,
        rms: float,
    ) -> _TagPoseSolution:
        camera_from_tag = frame.compose(mount)
        return _TagPoseSolution(
            camera_from_tag=camera_from_tag,
            reprojection_rms_px=rms,
            normal_camera=camera_from_tag.rotation.apply([0.0, 0.0, 1.0]),
        )

    body_solution = solution(body_from_body, body_mount.frame_from_tag, 0.1)
    correct = solution(body_from_femur_zero, femur_mount.frame_from_tag, 0.2)
    flipped = solution(body_from_femur_flipped, femur_mount.frame_from_tag, 0.05)
    corners = np.asarray([
        [0.0, 0.0],
        [10.0, 0.0],
        [10.0, 10.0],
        [0.0, 10.0],
    ], dtype=np.float32)
    candidate_map = {
        0: (TagCorners(0, corners), [body_solution]),
        # The physically wrong branch deliberately has the better pixel fit.
        7: (TagCorners(7, corners), [flipped, correct]),
    }

    poses, decisions = tracker._select_robot_tag_solutions(
        candidate_map,
        {"L0_yaw": 0.0, "L0_hip": 0.0},
    )

    selected = next(pose for pose in poses if pose.tag_id == 7)
    assert decisions[7]["index"] == 1
    assert decisions[7]["reason"] == "encoder_branch_disambiguation"
    assert decisions[7]["encoder_error_deg"] < 0.01
    assert decisions[7]["alternate_encoder_error_deg"] > 20.0
    assert np.allclose(
        selected.camera_from_tag.rotation.as_matrix(),
        correct.camera_from_tag.rotation.as_matrix(),
    )

    # If telemetry disappears, preserve the already selected physical branch
    # instead of snapping back to the lower-RMS mirror pose.
    _poses, temporal_decisions = tracker._select_robot_tag_solutions(
        candidate_map,
        None,
    )
    assert temporal_decisions[7]["index"] == 1
    assert temporal_decisions[7]["reason"] == "temporal_continuity"


def test_body_branch_initializes_from_upward_floor_normal() -> None:
    tracker = AprilTagPoseTracker.from_json(
        CONFIG_DIR / "apriltag_pose_config_20260831.json"
    )
    body_mount = tracker._branch_estimator.tag_mounts[0]
    corners = np.asarray([
        [0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]
    ], dtype=np.float32)

    def candidate(tilt_deg: float, rms: float) -> _TagPoseSolution:
        frame = RigidTransform(
            np.zeros(3), Rotation.from_euler("x", tilt_deg, degrees=True)
        )
        camera_from_tag = frame.compose(body_mount.frame_from_tag)
        return _TagPoseSolution(
            camera_from_tag=camera_from_tag,
            reprojection_rms_px=rms,
            normal_camera=camera_from_tag.rotation.apply([0.0, 0.0, 1.0]),
        )

    mirrored = candidate(70.0, 0.01)
    upright = candidate(5.0, 0.03)
    expected = upright.normal_camera
    poses, decisions = tracker._select_robot_tag_solutions(
        {0: (TagCorners(0, corners), [mirrored, upright])},
        None,
        preferred_body_normal_camera=expected,
    )
    assert decisions[0]["index"] == 1
    assert decisions[0]["reason"] == "floor_normal_initialization"
    assert np.allclose(
        poses[0].camera_from_tag.rotation.as_matrix(),
        upright.camera_from_tag.rotation.as_matrix(),
    )

    # A single noisy frame must not let temporal continuity latch onto the
    # physically impossible planar mirror branch for the rest of a session.
    tracker._previous_camera_from_tag[0] = mirrored.camera_from_tag
    recovered, recovered_decisions = tracker._select_robot_tag_solutions(
        {0: (TagCorners(0, corners), [mirrored, upright])},
        None,
        preferred_body_normal_camera=expected,
    )
    assert recovered_decisions[0]["index"] == 1
    assert recovered_decisions[0]["reason"] == "floor_normal_consistency"
    assert np.allclose(
        recovered[0].camera_from_tag.rotation.as_matrix(),
        upright.camera_from_tag.rotation.as_matrix(),
    )


def test_temporal_tag_tracker_bridges_a_decoder_miss_with_optical_flow() -> None:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = cv2.aruco.generateImageMarker(dictionary, 3, 180, borderBits=1)
    first = np.full((420, 520), 210, dtype=np.uint8)
    second = first.copy()
    first[120:300, 150:330] = marker
    second[126:306, 159:339] = marker
    tracker = TemporalTagCornerTracker(max_occlusion_frames=3)

    decoded = tracker.update(first, detect_tag_corners(first))
    carried = tracker.update(second, [])

    assert decoded[0].source == "detected"
    assert len(carried) == 1
    assert carried[0].tag_id == 3
    assert carried[0].source == "optical_flow"
    assert carried[0].occlusion_age_frames == 1
    assert np.allclose(
        carried[0].center_px - decoded[0].center_px, [9.0, 6.0], atol=1.0
    )


def _synthetic_red_foot_scene() -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    image = np.full((900, 900, 3), 80, dtype=np.uint8)
    body = np.asarray([450.0, 450.0])
    anchors: dict[int, np.ndarray] = {}
    for leg in range(6):
        angle = (leg + 0.5) * math.pi / 3.0
        direction = np.asarray([math.cos(angle), math.sin(angle)])
        anchor = body + 150.0 * direction
        foot = body + 340.0 * direction
        anchors[leg] = anchor
        cv2.ellipse(
            image,
            tuple(np.rint(foot).astype(int)),
            (22, 34),
            math.degrees(angle),
            0,
            360,
            (0, 0, 240),
            -1,
        )
    return image, body, anchors


def test_red_boot_detector_associates_all_six_legs_by_outward_ray() -> None:
    image, body, anchors = _synthetic_red_foot_scene()
    tracker = FootTipTracker(max_occlusion_frames=3)

    feet = tracker.update(
        image,
        body_center_px=body,
        femur_anchor_px=anchors,
        tag_scale_px=60.0,
    )

    assert [foot.leg for foot in feet] == list(range(6))
    assert all(foot.source == "color" for foot in feet)
    for foot in feet:
        radial_distance = np.linalg.norm(foot.point_px - body)
        assert 350.0 <= radial_distance <= 385.0


def test_red_boot_tracker_marks_short_missing_measurement_as_inferred() -> None:
    image, body, anchors = _synthetic_red_foot_scene()
    tracker = FootTipTracker(max_occlusion_frames=3)
    first = tracker.update(
        image,
        body_center_px=body,
        femur_anchor_px=anchors,
        tag_scale_px=60.0,
    )
    second = tracker.update(
        image,
        body_center_px=None,
        femur_anchor_px={},
        tag_scale_px=60.0,
    )

    assert len(first) == len(second) == 6
    assert all(foot.source in {"optical_flow", "prediction"} for foot in second)
    assert all(foot.occlusion_age_frames == 1 for foot in second)


def test_live_camera_cycle_is_deduplicated_and_wraps() -> None:
    cycle = _parse_camera_cycle("0, 1, 1")

    assert cycle == (0, 1)
    assert _camera_order_after(0, cycle) == (1, 0)
    assert _camera_order_after(1, cycle) == (0, 1)


def _safe_pose_fixture() -> tuple[dict, dict]:
    detections = [
        {"tag_id": tag_id, "label": f"robot {tag_id}", "source": "detected"}
        for tag_id in range(13)
    ]
    feet = [
        {"leg": leg, "source": "color", "confidence": 1.0}
        for leg in range(6)
    ]
    joints = {
        f"L{leg}_{axis}": {"value_deg": 0.0}
        for leg in range(6) for axis in ("yaw", "hip", "knee")
    }
    result = {
        "detections": detections,
        "foot_tips": feet,
        "camera_calibration_approximate": False,
        "full_pose": {
            "joints": joints,
            "calibration_disagreements": [],
            "prediction_only_joints": [],
            "walking_check": {"body_tilt_deg": 1.0},
            "zero_check": {"matches_zero": True},
        },
    }
    feedback = {
        "ok": True,
        "live_joint_count": 18,
        "roll_deg": 1.0,
        "pitch_deg": -1.0,
    }
    return result, feedback


def test_safe_pose_assessment_requires_support_only_for_motion() -> None:
    result, feedback = _safe_pose_fixture()

    unsupported = _safe_pose_assessment(
        result, feedback, operator_supported=False
    )
    supported = _safe_pose_assessment(
        result, feedback, operator_supported=True
    )

    assert unsupported["verdict"] == "safe"
    assert unsupported["safe_pose"] is True
    assert unsupported["safe_for_alignment_motion"] is False
    assert supported["safe_for_alignment_motion"] is True
    assert supported["straight_horizontal_candidate"] is True


def test_safe_pose_assessment_calls_large_tilt_unsafe() -> None:
    result, feedback = _safe_pose_fixture()
    feedback["pitch_deg"] = 22.0

    assessment = _safe_pose_assessment(
        result, feedback, operator_supported=True
    )

    assert assessment["verdict"] == "unsafe"
    assert assessment["safe_pose"] is False
    assert assessment["safe_for_alignment_motion"] is False
    assert "tilt" in assessment["unsafe_reasons"][0]


def test_safe_pose_prefers_live_imu_over_biased_visual_tilt() -> None:
    result, feedback = _safe_pose_fixture()
    result["camera_calibration_approximate"] = True
    result["full_pose"]["walking_check"]["body_tilt_deg"] = 7.2
    feedback["roll_deg"] = 0.0
    feedback["pitch_deg"] = 3.0

    assessment = _safe_pose_assessment(
        result, feedback, operator_supported=False
    )

    assert assessment["verdict"] == "safe"
    assert assessment["imu_tilt_deg"] == 3.0
    assert "visual tilt 7.2 deg disagrees" in assessment["warnings"][0]


def test_unsigned_foot_tip_knee_error_is_warning_not_unsafe() -> None:
    result, feedback = _safe_pose_fixture()
    result["full_pose"]["calibration_disagreements"] = [{
        "joint": "L2_knee",
        "visual_abs_minus_encoder_abs_deg": 27.4,
        "unsigned_visual_estimate": True,
    }]

    assessment = _safe_pose_assessment(
        result, feedback, operator_supported=False
    )

    assert assessment["verdict"] == "safe"
    assert assessment["unsafe_reasons"] == []
    assert "provisional foot-tip estimate" in assessment["warnings"][0]


def test_feedback_poll_never_blocks_camera_loop(monkeypatch) -> None:
    def slow_failure(*_args, **_kwargs):
        time.sleep(0.15)
        raise OSError("offline")

    monkeypatch.setattr(track_cli, "urlopen", slow_failure)
    client = FeedbackClient("http://robot.invalid", hz=3.0)

    start = time.perf_counter()
    angles, status = client.sample()
    elapsed = time.perf_counter() - start

    assert elapsed < 0.05
    assert angles == {}
    assert status["configured"] is True
