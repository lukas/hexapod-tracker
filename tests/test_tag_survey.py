"""Synthetic tests for handheld tag discovery and zero-pose mount learning."""
from __future__ import annotations

import base64
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from hexapod_tracker.apriltag_vision import (
    TagCorners,
    estimate_world_reference,
    marker_object_corners,
)
from hexapod_tracker.housing_pose import (
    JOINT_NAMES,
    RigidTransform,
    forward_frame_transforms,
)
from hexapod_tracker.tag_survey import (
    HandheldWorldAlignment,
    TagSurveyAccumulator,
    TagSurveyOptions,
    apply_survey_to_config,
    arkit_world_from_opencv_camera,
    learn_zero_pose_mounts,
    merge_robot_layout_into_config,
)
from hexapod_tracker.paths import CONFIG_DIR
from hexapod_tracker.zero_pose_survey import (
    _guidance,
    _quality_feedback,
    _wifi_frames,
    main as zero_pose_survey_main,
)
from hexapod_tracker.zero_pose_refinement import (
    refine_zero_pose_with_buildviz,
)


CAMERA_MATRIX = np.asarray([
    [900.0, 0.0, 640.0],
    [0.0, 900.0, 360.0],
    [0.0, 0.0, 1.0],
])


def test_full_robot_layout_has_six_legs_and_four_vertical_tags_each() -> None:
    tracker = json.loads(
        (CONFIG_DIR / "apriltag_pose_config_20260831.json").read_text()
    )
    layout = json.loads(
        (CONFIG_DIR / "hexapod-1-apriltag-layout.json").read_text()
    )

    merged = merge_robot_layout_into_config(tracker, layout)
    tags = merged["robot_pose"]["tags"]
    vertical = [spec for spec in tags.values() if spec.get("kind") == "yoke_face"]

    assert len(tags) == 37
    assert len(vertical) == 24
    assert len({spec["position"] for spec in tags.values()}) == 37
    for leg in range(6):
        leg_tags = [spec for spec in vertical if spec.get("leg") == leg]
        assert len(leg_tags) == 4
        assert {(spec["joint"], spec["mount_side"]) for spec in leg_tags} == {
            ("hip", "+y"), ("hip", "-y"),
            ("knee", "+y"), ("knee", "-y"),
        }


def _detection(
    tag_id: int,
    marker_size_m: float,
    world_from_tag: RigidTransform,
    world_from_camera: RigidTransform,
    *,
    noise_px: float = 0.0,
    seed: int = 0,
) -> TagCorners:
    camera_from_world = world_from_camera.inverse()
    world_points = np.stack([
        world_from_tag.apply(point)
        for point in marker_object_corners(marker_size_m)
    ])
    camera_points = np.stack([
        camera_from_world.apply(point) for point in world_points
    ])
    pixels = np.column_stack([
        CAMERA_MATRIX[0, 0] * camera_points[:, 0] / camera_points[:, 2]
        + CAMERA_MATRIX[0, 2],
        CAMERA_MATRIX[1, 1] * camera_points[:, 1] / camera_points[:, 2]
        + CAMERA_MATRIX[1, 2],
    ])
    if noise_px:
        pixels += np.random.default_rng(seed).normal(0.0, noise_px, pixels.shape)
    return TagCorners(tag_id, pixels.astype(np.float32))


def test_handheld_alignment_converts_opengl_camera_trajectory() -> None:
    world_from_arkit = RigidTransform(
        np.asarray([0.4, -0.2, 0.15]),
        Rotation.from_euler("xyz", [2.0, -3.0, 31.0], degrees=True),
    )
    alignment = HandheldWorldAlignment(min_observations=5)
    expected_camera_poses = []
    for index in range(5):
        arkit_world_from_gl = RigidTransform(
            np.asarray([0.03 * index, 0.02, -0.04 * index]),
            Rotation.from_euler(
                "xyz", [1.0, index * 2.0, -index * 3.0], degrees=True
            ),
        )
        world_from_cv = world_from_arkit.compose(
            arkit_world_from_opencv_camera(arkit_world_from_gl)
        )
        expected_camera_poses.append((arkit_world_from_gl, world_from_cv))
        alignment.add(world_from_cv, arkit_world_from_gl)

    consensus = alignment.consensus()

    assert consensus is not None and consensus.stable
    assert np.allclose(
        consensus.transform.translation_m, world_from_arkit.translation_m
    )
    assert math.degrees(float(
        (consensus.transform.rotation.inv() * world_from_arkit.rotation).magnitude()
    )) < 1e-6
    for arkit_pose, expected in expected_camera_poses:
        actual = alignment.world_from_camera(arkit_pose)
        assert np.allclose(actual.translation_m, expected.translation_m)
        assert math.degrees(float(
            (actual.rotation.inv() * expected.rotation).magnitude()
        )) < 1e-6


def test_handheld_alignment_uses_a_bounded_recent_window() -> None:
    alignment = HandheldWorldAlignment(min_observations=3, max_observations=4)
    arkit_pose = RigidTransform.identity()
    for offset in range(7):
        alignment.add(
            RigidTransform(
                np.asarray([float(offset) / 1000.0, 0.0, 0.0]), Rotation.identity()
            ).compose(arkit_world_from_opencv_camera(arkit_pose)),
            arkit_pose,
        )

    assert alignment.observation_count == 4
    assert np.isclose(alignment.consensus().transform.translation_m[0], 0.0045)


def test_handheld_alignment_keeps_last_good_lock_when_floor_views_split() -> None:
    alignment = HandheldWorldAlignment(
        min_observations=3,
        max_translation_spread_m=0.015,
        max_rotation_spread_deg=1.5,
        max_observations=12,
    )
    arkit_pose = RigidTransform.identity()
    initial = RigidTransform(
        np.asarray([0.20, -0.50, 0.80]), Rotation.identity()
    )
    for offset_mm in (0.0, 1.0, -1.0):
        alignment.add(
            RigidTransform(
                initial.translation_m + [offset_mm / 1000.0, 0.0, 0.0],
                initial.rotation,
            ).compose(arkit_world_from_opencv_camera(arkit_pose)),
            arkit_pose,
        )

    assert alignment.has_lock
    locked_camera = alignment.world_from_camera(arkit_pose)

    # A later view-dependent floor solve forms a second internally consistent
    # cluster. This is exactly what happened in the saved iPhone walk: the
    # diagnostic rolling window becomes ambiguous, but mapping must continue.
    shifted = RigidTransform(
        initial.translation_m + [0.045, 0.0, 0.0], initial.rotation
    )
    for offset_mm in (0.0, 1.0, -1.0):
        alignment.add(
            RigidTransform(
                shifted.translation_m + [offset_mm / 1000.0, 0.0, 0.0],
                shifted.rotation,
            ).compose(arkit_world_from_opencv_camera(arkit_pose)),
            arkit_pose,
        )

    diagnostic = alignment.consensus()
    assert diagnostic is not None
    assert diagnostic.ambiguous_cluster is True
    assert diagnostic.stable is False
    assert alignment.has_lock
    assert np.allclose(
        alignment.world_from_camera(arkit_pose).translation_m,
        locked_camera.translation_m,
    )


def test_wifi_frame_spool_decodes_packed_rgbd_and_pose(tmp_path) -> None:
    hsv_depth = np.zeros((40, 50, 3), dtype=np.uint8)
    hsv_depth[:, :, 0] = 60
    hsv_depth[:, :, 1:] = 255
    depth_bgr = cv2.cvtColor(hsv_depth, cv2.COLOR_HSV2BGR)
    rgb_bgr = np.full((40, 50, 3), (20, 100, 220), dtype=np.uint8)
    packed = np.concatenate([depth_bgr, rgb_bgr], axis=1)
    ok, encoded = cv2.imencode(".jpg", packed, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    (tmp_path / "latest.json").write_text(json.dumps({
        "sequence": 1,
        "rgbd_jpeg_base64": base64.b64encode(encoded.tobytes()).decode(),
        "camera_matrix": CAMERA_MATRIX.tolist(),
        "camera_pose_xyzw_xyz": [0, 0, 0, 1, 0.1, 0.2, 0.3],
        "max_depth_m": 3.0,
    }))

    frame = next(_wifi_frames(tmp_path, timeout_s=0.1))

    assert frame.source_label == "Record3D Wi-Fi WebRTC"
    assert frame.rgb_bgr.shape == (40, 50, 3)
    assert np.isclose(float(np.median(frame.depth_m)), 3.0 * 60 / 179, atol=0.08)
    assert np.allclose(
        frame.arkit_world_from_opengl_camera.translation_m, [0.1, 0.2, 0.3]
    )


def test_live_guidance_picks_one_target_and_coaches_camera_speed() -> None:
    progress = {
        "robot_positions": [
            {"position": "L0 knee", "state": "not_seen", "tag_id": None},
            {
                "position": "L2 knee",
                "state": "seen_needs_another_view",
                "tag_id": 10,
                "observations": 3,
            },
        ],
        "ground_tag_status": [],
        "missing_robot_positions": ["L0 knee", "L2 knee"],
        "missing_ground_tag_ids": [],
    }

    guidance = _guidance(
        "survey", progress, [104], min_observations=5, resumed=False
    )
    quality = _quality_feedback(
        guidance,
        [],
        {10},
        camera_speed_m_s=0.62,
        tracking_message="Landmark lock",
        anchor_reprojection_rms_px=None,
        depth_plane_rms_mm=None,
    )

    assert guidance["target_position"] == "L2 knee"
    assert guidance["target_tag_id"] == 10
    assert "2 more clean frames" in guidance["action"]
    assert quality["level"] == "caution"
    assert quality["headline"] == "You are moving too quickly"


def test_tag_requires_a_genuinely_different_viewpoint_when_configured() -> None:
    tag = RigidTransform(
        np.asarray([0.10, 0.08, 0.0]), Rotation.identity()
    )
    overhead = RigidTransform(
        np.asarray([0.0, 0.0, 0.75]),
        Rotation.from_euler("x", 180.0, degrees=True),
    )
    accumulator = TagSurveyAccumulator(
        robot_tags={},
        expected_ground_ids=[12],
        marker_size_m=0.027,
        options=TagSurveyOptions(
            min_observations=4,
            min_viewpoint_span_deg=8.0,
        ),
    )
    for _index in range(4):
        accumulator.observe_frame(
            [_detection(12, 0.027, tag, overhead)],
            overhead,
            CAMERA_MATRIX,
            np.zeros(5),
        )

    first = accumulator.tag_records()[0]
    assert first["observations"] == 4
    assert first["viewpoint_span_deg"] == 0.0
    assert first["stable"] is False

    side_view = RigidTransform(
        np.asarray([0.18, 0.0, 0.75]),
        Rotation.from_euler("x", 180.0, degrees=True),
    )
    for _index in range(4):
        accumulator.observe_frame(
            [_detection(12, 0.027, tag, side_view)],
            side_view,
            CAMERA_MATRIX,
            np.zeros(5),
        )

    second = accumulator.tag_records()[0]
    assert second["viewpoint_span_deg"] > 8.0
    assert second["stable"] is True


def test_slow_side_step_keeps_its_first_viewpoint_outside_rolling_window() -> None:
    tag = RigidTransform(np.asarray([0.10, 0.08, 0.0]), Rotation.identity())
    accumulator = TagSurveyAccumulator(
        robot_tags={},
        expected_ground_ids=[12],
        marker_size_m=0.027,
        options=TagSurveyOptions(
            min_observations=3,
            max_observations_per_tag=4,
            min_viewpoint_span_deg=8.0,
        ),
    )
    for x in np.linspace(0.0, 0.18, 12):
        camera = RigidTransform(
            np.asarray([x, 0.0, 0.75]),
            Rotation.from_euler("x", 180.0, degrees=True),
        )
        accumulator.observe_frame(
            [_detection(12, 0.027, tag, camera)],
            camera,
            CAMERA_MATRIX,
            np.zeros(5),
        )

    record = accumulator.tag_records()[0]
    assert record["observations"] == 4
    assert record["viewpoint_span_deg"] > 8.0
    assert record["stable"] is True


def test_world_reference_supports_mixed_floor_tag_sizes() -> None:
    world_from_camera = RigidTransform(
        np.asarray([0.02, -0.03, 0.78]),
        Rotation.from_euler("xyz", [180.0, 0.0, 6.0], degrees=True),
    )
    floor_tags = {
        12: RigidTransform(
            np.asarray([0.14, 0.08, 0.0]),
            Rotation.from_euler("z", 25.0, degrees=True),
        ),
        40: RigidTransform(
            np.asarray([-0.10, -0.05, 0.0]), Rotation.identity()
        ),
    }
    detections = [
        _detection(12, 0.027, floor_tags[12], world_from_camera),
        _detection(40, 0.070, floor_tags[40], world_from_camera),
    ]

    reference = estimate_world_reference(
        detections,
        floor_tags,
        CAMERA_MATRIX,
        np.zeros(5),
        marker_size_m=0.027,
        marker_sizes_m={12: 0.027, 40: 0.070},
    )

    assert reference is not None
    assert np.allclose(
        reference.world_from_camera.translation_m,
        world_from_camera.translation_m,
        atol=2e-6,
    )
    assert math.degrees(float(
        (
            reference.world_from_camera.rotation.inv()
            * world_from_camera.rotation
        ).magnitude()
    )) < 2e-4


def test_accuracy_gate_requires_joint_multi_tag_floor_validation() -> None:
    world_from_camera = RigidTransform(
        np.asarray([0.0, 0.0, 0.75]),
        Rotation.from_euler("x", 180.0, degrees=True),
    )
    floor_tags = {
        12: RigidTransform(np.asarray([0.10, 0.08, 0.0]), Rotation.identity()),
        40: RigidTransform(np.asarray([-0.14, -0.06, 0.0]), Rotation.identity()),
    }
    accumulator = TagSurveyAccumulator(
        robot_tags={},
        expected_ground_ids=[12, 40],
        marker_size_m=0.027,
        reference_floor_tags=floor_tags,
        options=TagSurveyOptions(min_observations=4),
    )
    for _index in range(6):
        accumulator.observe_frame(
            [
                _detection(tag_id, 0.027, transform, world_from_camera)
                for tag_id, transform in floor_tags.items()
            ],
            world_from_camera,
            CAMERA_MATRIX,
            np.zeros(5),
        )

    before_validation = accumulator.summary()
    assert before_validation["coverage_complete"] is True
    assert before_validation["complete"] is False
    for _index in range(6):
        accumulator.observe_floor_reference(
            [12, 40], reprojection_rms_px=0.4, depth_plane_rms_mm=4.0
        )

    after_validation = accumulator.summary()
    assert after_validation["quality_gate"]["passed"] is True
    assert after_validation["complete"] is True


def test_survey_records_roles_orientations_distances_and_learns_mounts() -> None:
    world_from_camera = RigidTransform(
        np.asarray([0.01, -0.02, 0.78]),
        Rotation.from_euler("xyz", [180.0, 0.0, 3.0], degrees=True),
    )
    world_from_body = RigidTransform(
        np.asarray([0.0, 0.0, 0.15]),
        Rotation.from_euler("z", 12.0, degrees=True),
    )
    zero_angles = {name: 0.0 for name in JOINT_NAMES}
    frames = forward_frame_transforms(world_from_body, zero_angles)
    known_mount = RigidTransform(
        np.asarray([0.006, -0.004, 0.008]),
        Rotation.from_euler("z", 37.0, degrees=True),
    )
    second_body_mount = RigidTransform(
        np.asarray([0.025, -0.010, 0.004]),
        Rotation.from_euler("z", -21.0, degrees=True),
    )
    tags = {
        0: (0.027, world_from_body),
        1: (0.027, frames["L0_coxa"].compose(known_mount)),
        2: (0.027, world_from_body.compose(second_body_mount)),
        12: (
            0.027,
            RigidTransform(
                np.asarray([0.16, 0.11, 0.0]),
                Rotation.from_euler("z", 28.0, degrees=True),
            ),
        ),
        40: (
            0.070,
            RigidTransform(np.asarray([-0.11, -0.08, 0.0]), Rotation.identity()),
        ),
    }
    accumulator = TagSurveyAccumulator(
        robot_tags={
            0: {"label": "body", "frame": "body"},
            1: {"label": "L0 hip", "frame": "L0_coxa"},
            2: {"label": "second body tag", "frame": "body"},
        },
        expected_ground_ids=[12],
        anchor_ids=[40],
        marker_size_m=0.027,
        marker_sizes_m={40: 0.070},
        options=TagSurveyOptions(min_observations=5),
    )
    for index in range(7):
        detections = [
            _detection(
                tag_id,
                marker_size,
                transform,
                world_from_camera,
                noise_px=0.010,
                seed=index * 10 + tag_id,
            )
            for tag_id, (marker_size, transform) in tags.items()
        ]
        accumulator.observe_frame(
            detections, world_from_camera, CAMERA_MATRIX, np.zeros(5)
        )

    summary = accumulator.summary()

    assert summary["complete"] is True
    records = {item["tag_id"]: item for item in summary["tags"]}
    assert records[0]["role"] == "robot"
    assert records[1]["robot_frame"] == "L0_coxa"
    assert records[12]["role"] == "ground"
    assert records[40]["role"] == "calibration_anchor"
    assert abs(records[12]["height_above_ground_mm"]) < 1.0
    assert records[12]["normal_error_from_world_up_deg"] < 0.2
    distance = next(
        item for item in summary["floor_tag_distances"]
        if set(item["tag_ids"]) == {12, 40}
    )
    assert np.isclose(
        distance["planar_distance_m"],
        np.linalg.norm([0.27, 0.19]),
        atol=1e-4,
    )

    config = {
        "marker_size_m": 0.027,
        "floor_tags": {},
        "robot_pose": {
            "geometry": {},
            "tags": {
                "0": {"frame": "body", "frame_from_tag": {}},
                "1": {"frame": "L0_coxa", "frame_from_tag": {}},
                "2": {"frame": "body", "frame_from_tag": {}},
            },
        },
    }
    learned, geometry_report = learn_zero_pose_mounts(config, summary)

    assert geometry_report["ok"] is True
    assert geometry_report["geometry_status"] == "partial_static_measurements"
    assert geometry_report["pose_used"]["body_anchor_tag_id"] == 0
    assert 0 not in learned
    assert np.allclose(
        learned[2].translation_m, second_body_mount.translation_m, atol=0.001
    )
    assert np.allclose(
        learned[1].translation_m, known_mount.translation_m, atol=0.001
    )
    assert math.degrees(float(
        (learned[1].rotation.inv() * known_mount.rotation).magnitude()
    )) < 0.5

    updated = apply_survey_to_config(config, summary, learned)
    assert updated["floor_tags"]["12"]["marker_size_m"] == 0.027
    assert updated["floor_tags"]["40"]["marker_size_m"] == 0.07
    assert updated["robot_pose"]["tags"]["1"]["mount_source"] == (
        "zero_pose_handheld_survey"
    )
    assert "mount_source" not in updated["robot_pose"]["tags"]["0"]


def test_mount_learning_uses_l0_tag_to_align_buildviz_body_axes() -> None:
    zero_angles = {name: 0.0 for name in JOINT_NAMES}
    world_from_body = RigidTransform(
        np.asarray([0.18, -0.07, 0.12]),
        Rotation.from_euler("xyz", [1.0, -2.0, 35.0], degrees=True),
    )
    frames = forward_frame_transforms(world_from_body, zero_angles)
    body_tag_translation = np.asarray([0.015, -0.010, 0.030])
    actual_body_mount = RigidTransform(
        body_tag_translation,
        Rotation.from_euler("z", 15.0, degrees=True),
    )
    l0_orientation_mount = RigidTransform(
        np.asarray([0.003, 0.004, 0.005]),
        Rotation.from_euler("z", 90.0, degrees=True),
    )
    femur_mount = RigidTransform(
        np.asarray([0.071, 0.026, -0.048]),
        Rotation.from_euler("xyz", [0.0, 0.0, -12.0], degrees=True),
    )
    measured = {
        0: world_from_body.compose(actual_body_mount),
        1: frames["L0_coxa"].compose(l0_orientation_mount),
        7: frames["L0_femur"].compose(femur_mount),
    }
    survey = {
        "tags": [
            {
                "tag_id": tag_id,
                "stable": True,
                "role": "robot",
                "world_from_tag": transform.to_dict(),
            }
            for tag_id, transform in measured.items()
        ],
    }
    config = {
        "robot_pose": {
            "geometry": {},
            "tags": {
                # The historical yaw is intentionally wrong.  It must not
                # define the body axes when the unchanged L0 tag is supplied.
                "0": {
                    "frame": "body",
                    "frame_from_tag": {
                        "translation_m": body_tag_translation.tolist(),
                        "euler_xyz_deg": [0.0, 0.0, 100.0],
                    },
                },
                "1": {
                    "frame": "L0_coxa",
                    "frame_from_tag": l0_orientation_mount.to_dict(),
                },
                "7": {"frame": "L0_femur", "frame_from_tag": {}},
            },
        },
    }

    learned, report = learn_zero_pose_mounts(
        config,
        survey,
        body_anchor_tag_id=0,
        orientation_anchor_tag_id=1,
    )

    assert report["ok"] is True
    assert report["pose_used"]["orientation_anchor_tag_id"] == 1
    assert report["pose_used"]["orientation_anchor_frame"] == "L0_coxa"
    recovered_world_from_body = RigidTransform.from_dict(
        report["pose_used"]["world_from_body"]
    )
    assert np.allclose(
        recovered_world_from_body.translation_m,
        world_from_body.translation_m,
        atol=1e-9,
    )
    assert math.degrees(float(
        (
            recovered_world_from_body.rotation.inv()
            * world_from_body.rotation
        ).magnitude()
    )) < 1e-7
    assert set(learned) == {0, 1, 7}
    assert np.allclose(learned[0].translation_m, body_tag_translation, atol=1e-9)
    assert math.degrees(float(
        (learned[0].rotation.inv() * actual_body_mount.rotation).magnitude()
    )) < 1e-7
    assert np.allclose(learned[7].translation_m, femur_mount.translation_m, atol=1e-9)
    assert math.degrees(float(
        (learned[7].rotation.inv() * femur_mount.rotation).magnitude()
    )) < 1e-7

    world_from_camera = RigidTransform(
        np.asarray([0.10, -0.03, 0.75]),
        Rotation.from_euler("x", 180.0, degrees=True),
    )
    accumulator = TagSurveyAccumulator(
        robot_tags={
            0: config["robot_pose"]["tags"]["0"],
            1: config["robot_pose"]["tags"]["1"],
            7: config["robot_pose"]["tags"]["7"],
        },
        marker_size_m=0.027,
        position_tag_overrides={"L0_coxa": 1},
        body_anchor_tag_id=0,
    )
    live_frames = accumulator._world_robot_frames(
        [
            _detection(tag_id, 0.027, measured[tag_id], world_from_camera)
            for tag_id in (0, 1)
        ],
        world_from_camera,
        CAMERA_MATRIX,
        np.zeros(5),
        world_from_camera.rotation.inv().apply([0.0, 0.0, 1.0]),
    )
    assert live_frames is not None
    assert math.degrees(float(
        (
            live_frames["body"].rotation.inv()
            * world_from_body.rotation
        ).magnitude()
    )) < 1e-3

    replacement_survey = {
        "tags": [
            {
                "tag_id": 50 if tag_id == 0 else tag_id,
                "stable": True,
                "role": "robot",
                "world_from_tag": transform.to_dict(),
            }
            for tag_id, transform in measured.items()
        ],
        "robot_positions": [
            {
                "state": "measured",
                "tag_id": actual_id,
                "configured_tag_id": configured_id,
                "frame": frame,
            }
            for actual_id, configured_id, frame in (
                (50, 0, "body"),
                (1, 1, "L0_coxa"),
                (7, 7, "L0_femur"),
            )
        ],
    }
    replacement_mounts, replacement_report = learn_zero_pose_mounts(
        config,
        replacement_survey,
        body_anchor_tag_id=0,
        orientation_anchor_tag_id=1,
    )
    assert replacement_report["ok"] is True
    assert replacement_report["pose_used"]["body_anchor_tag_id"] == 50
    assert (
        replacement_report["pose_used"]["body_anchor_configured_tag_id"] == 0
    )
    assert set(replacement_mounts) == {1, 7, 50}


def test_buildviz_refinement_corrects_noisy_multiview_mounts(
    tmp_path: Path,
) -> None:
    robot_specs = {
        0: {
            "frame": "body",
            "kind": "chassis_tag",
            "marker_size_m": 0.027,
            "frame_from_tag": {"euler_xyz_deg": [0.0, 0.0, 0.0]},
        },
        1: {
            "frame": "L0_coxa", "kind": "servo_lid", "joint": "hip",
            "marker_size_m": 0.027,
            "frame_from_tag": {"euler_xyz_deg": [0.0, 0.0, 90.0]},
        },
        4: {
            "frame": "L1_coxa", "kind": "servo_lid", "joint": "hip",
            "marker_size_m": 0.027,
            "frame_from_tag": {"euler_xyz_deg": [0.0, 0.0, -90.0]},
        },
        7: {
            "frame": "L0_femur", "kind": "servo_lid", "joint": "knee",
            "marker_size_m": 0.027,
            "frame_from_tag": {"euler_xyz_deg": [0.0, 0.0, 90.0]},
        },
        10: {
            "frame": "L1_femur", "kind": "servo_lid", "joint": "knee",
            "marker_size_m": 0.027,
            "frame_from_tag": {"euler_xyz_deg": [0.0, 0.0, -90.0]},
        },
        42: {
            "frame": "L0_femur", "kind": "yoke_face", "joint": "hip",
            "mount_side": "+y", "marker_size_m": 0.027,
            "frame_from_tag": {"euler_xyz_deg": [-90.0, 0.0, 0.0]},
        },
        43: {
            "frame": "L1_femur", "kind": "yoke_face", "joint": "hip",
            "mount_side": "+y", "marker_size_m": 0.027,
            "frame_from_tag": {"euler_xyz_deg": [-90.0, 0.0, 0.0]},
        },
    }
    config = {
        "marker_size_m": 0.027,
        "robot_pose": {
            "geometry": {},
            "tags": {str(tag_id): spec for tag_id, spec in robot_specs.items()},
        },
    }
    world_from_body = RigidTransform(
        np.asarray([0.0, 0.0, 0.15]),
        Rotation.from_euler("z", 20.0, degrees=True),
    )
    zero_frames = forward_frame_transforms(
        world_from_body, {name: 0.0 for name in JOINT_NAMES}
    )
    mount_points = {
        "chassis": np.asarray([0.0, 0.0, 0.0]),
        "hip_top": np.asarray([0.0, 0.005, -0.050]),
        "knee_top": np.asarray([0.078, 0.029, -0.050]),
        "hip_+y": np.asarray([0.0, 0.063, -0.074]),
    }
    true_tags: dict[int, RigidTransform] = {}
    for tag_id, spec in robot_specs.items():
        group = (
            "chassis" if spec["kind"] == "chassis_tag"
            else f"{spec['joint']}_top" if spec["kind"] == "servo_lid"
            else f"{spec['joint']}_{spec['mount_side']}"
        )
        mount = RigidTransform(
            mount_points[group],
            Rotation.from_euler(
                "xyz", spec["frame_from_tag"]["euler_xyz_deg"], degrees=True
            ),
        )
        true_tags[tag_id] = zero_frames[spec["frame"]].compose(mount)
    floor_tags = {
        104: RigidTransform(
            np.asarray([-0.24, -0.12, 0.0]), Rotation.identity()
        )
    }

    def look_at(position: np.ndarray) -> RigidTransform:
        forward = np.asarray([0.0, 0.0, 0.09]) - position
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.asarray([0.0, 0.0, 1.0]))
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        return RigidTransform(
            position,
            Rotation.from_matrix(np.column_stack([right, down, forward])),
        )

    camera_poses = [
        look_at(np.asarray([
            0.48 * math.cos(angle),
            0.48 * math.sin(angle),
            0.48,
        ]))
        for angle in np.radians([5.0, 60.0, 115.0, 180.0, 245.0])
    ]
    archive = tmp_path / "frames"
    archive.mkdir()
    cv_from_gl = RigidTransform(
        np.zeros(3), Rotation.from_euler("x", 180.0, degrees=True)
    )
    all_tags = {**true_tags, **floor_tags}
    for index, camera in enumerate(camera_poses):
        detections = [
            _detection(tag_id, 0.027, tag, camera)
            for tag_id, tag in all_tags.items()
        ]
        arkit_gl = camera.compose(cv_from_gl.inverse())
        np.savez_compressed(
            archive / f"frame-{index:02d}.npz",
            camera_matrix=CAMERA_MATRIX,
            camera_pose_xyzw_xyz=np.asarray([
                *arkit_gl.rotation.as_quat(), *arkit_gl.translation_m
            ]),
            tag_ids=np.asarray([item.tag_id for item in detections]),
            tag_corners_px=np.asarray([item.corners_px for item in detections]),
        )

    raw_transforms: dict[int, RigidTransform] = {}
    for index, (tag_id, transform) in enumerate(true_tags.items()):
        if tag_id in (0, 1):
            raw_transforms[tag_id] = transform
            continue
        offset = np.asarray([
            0.016 + index * 0.001,
            -0.012 + index * 0.0005,
            0.009,
        ])
        raw_transforms[tag_id] = RigidTransform(
            transform.translation_m + offset,
            transform.rotation * Rotation.from_euler(
                "xyz", [3.0, -2.0, 5.0], degrees=True
            ),
        )
    survey = {
        "tags": [
            {
                "tag_id": tag_id,
                "stable": True,
                "role": "robot",
                "world_from_tag": transform.to_dict(),
            }
            for tag_id, transform in raw_transforms.items()
        ],
    }

    refined, report = refine_zero_pose_with_buildviz(
        config,
        survey,
        frame_archive_dir=archive,
        floor_tags=floor_tags,
        world_from_arkit_world=RigidTransform.identity(),
        body_anchor_tag_id=0,
        orientation_anchor_tag_id=1,
    )

    assert report["ok"] is True
    assert report["frames"] == 5
    assert report["robot_tag_count"] == len(true_tags)
    assert report["final_coordinate_rms_px"] < 0.25
    assert (
        report["final_coordinate_rms_px"]
        < report["initial_coordinate_rms_px"] * 0.05
    )
    refined_by_id = {
        int(item["tag_id"]): RigidTransform.from_dict(item["world_from_tag"])
        for item in refined["tags"]
    }
    raw_error = np.mean([
        np.linalg.norm(raw_transforms[tag_id].translation_m - true.translation_m)
        for tag_id, true in true_tags.items()
    ])
    refined_error = np.mean([
        np.linalg.norm(refined_by_id[tag_id].translation_m - true.translation_m)
        for tag_id, true in true_tags.items()
    ])
    assert refined_error < raw_error * 0.1


def test_survey_rejects_two_consistent_poses_for_the_same_id() -> None:
    world_from_camera = RigidTransform(
        np.asarray([0.0, 0.0, 0.7]),
        Rotation.from_euler("x", 180.0, degrees=True),
    )
    first = RigidTransform(
        np.asarray([-0.12, 0.08, 0.0]),
        Rotation.from_euler("z", 8.0, degrees=True),
    )
    second = RigidTransform(
        np.asarray([0.15, -0.07, 0.0]),
        Rotation.from_euler("z", -13.0, degrees=True),
    )
    accumulator = TagSurveyAccumulator(
        robot_tags={},
        expected_ground_ids=[12],
        marker_size_m=0.027,
        options=TagSurveyOptions(min_observations=4),
    )
    for index in range(10):
        transform = first if index < 5 else second
        accumulator.observe_frame(
            [_detection(12, 0.027, transform, world_from_camera)],
            world_from_camera,
            CAMERA_MATRIX,
            np.zeros(5),
        )

    summary = accumulator.summary()
    record = summary["tags"][0]

    assert record["possible_duplicate_id_or_tracking_jump"] is True
    assert record["stable"] is False
    assert summary["ambiguous_tag_ids"] == [12]
    assert summary["complete"] is False
    updated = apply_survey_to_config(
        {"floor_tags": {"12": {"world_from_tag": {}}}},
        summary,
        {},
    )
    assert "12" not in updated["floor_tags"]


def test_survey_restores_only_stable_records_after_reconnect() -> None:
    transform = RigidTransform(
        np.asarray([0.12, -0.04, 0.18]),
        Rotation.from_euler("z", 17.0, degrees=True),
    )
    original = TagSurveyAccumulator(
        robot_tags={1: {"label": "L0 hip", "frame": "L0_coxa"}},
        expected_ground_ids=[104],
        marker_size_m=0.027,
        options=TagSurveyOptions(min_observations=3),
    )
    stable_record = {
        "tag_id": 1,
        "stable": True,
        "marker_size_m": 0.027,
        "world_from_tag": transform.to_dict(),
        "observations": 8,
        "used_observations": 7,
        "translation_spread_mm": 2.5,
        "rotation_spread_deg": 0.8,
        "mean_reprojection_rms_px": 0.45,
    }
    unstable_record = {
        **stable_record,
        "tag_id": 104,
        "stable": False,
    }

    restored = original.restore_stable_records(
        [stable_record, unstable_record], frames=72
    )
    summary = original.summary()

    assert restored == [1]
    assert summary["frames"] == 72
    assert summary["stable_tag_ids"] == [1]
    assert summary["missing_ground_tag_ids"] == [104]
    record = summary["tags"][0]
    assert record["observations"] == 8
    assert record["used_observations"] == 7
    assert record["translation_spread_mm"] == 2.5


def test_survey_restores_clean_same_angle_record_as_provisional_seed() -> None:
    transform = RigidTransform(
        np.asarray([0.12, -0.04, 0.18]),
        Rotation.from_euler("z", 17.0, degrees=True),
    )
    accumulator = TagSurveyAccumulator(
        robot_tags={1: {"label": "L0 hip", "frame": "L0_coxa"}},
        marker_size_m=0.027,
        options=TagSurveyOptions(
            min_observations=3,
            min_viewpoint_span_deg=8.0,
        ),
    )
    restored = accumulator.restore_stable_records([{
        "tag_id": 1,
        "stable": False,
        "viewpoint_requirement_met": False,
        "possible_duplicate_id_or_tracking_jump": False,
        "marker_size_m": 0.027,
        "world_from_tag": transform.to_dict(),
        "observations": 12,
        "used_observations": 11,
        "translation_spread_mm": 2.5,
        "rotation_spread_deg": 0.8,
        "mean_reprojection_rms_px": 0.45,
    }])

    record = accumulator.tag_records()[0]
    assert restored == [1]
    assert record["observations"] == 1
    assert record["stable"] is False
    assert record["viewpoint_requirement_met"] is False


def test_robot_completion_uses_physical_positions_and_accepts_replacement_ids() -> None:
    world_from_camera = RigidTransform(
        np.asarray([0.10, 0.05, 0.80]),
        Rotation.from_euler("x", 180.0, degrees=True),
    )
    robot_tags = {
        0: {"label": "body", "frame": "body", "photo_center_px": [0, 0]},
        1: {"label": "L0 hip", "frame": "L0_coxa", "photo_center_px": [100, 0]},
        2: {"label": "L0 knee", "frame": "L0_femur", "photo_center_px": [200, 0]},
        3: {"label": "L1 hip", "frame": "L1_coxa", "photo_center_px": [100, 100]},
        4: {"label": "L1 knee", "frame": "L1_femur", "photo_center_px": [0, 100]},
    }

    def tag_pose(photo_x: float, photo_y: float) -> RigidTransform:
        return RigidTransform(
            np.asarray([photo_x / 1000.0, photo_y / 1000.0, 0.15]),
            Rotation.identity(),
        )

    poses = {
        0: tag_pose(0, 0),
        1: tag_pose(100, 0),
        2: tag_pose(200, 0),
        4: tag_pose(0, 100),
        99: tag_pose(100, 100),
    }
    accumulator = TagSurveyAccumulator(
        robot_tags=robot_tags,
        marker_size_m=0.027,
        position_tag_overrides={"L0_coxa": 1},
        options=TagSurveyOptions(min_observations=3),
    )
    for _ in range(4):
        accumulator.observe_frame(
            [
                _detection(tag_id, 0.027, pose, world_from_camera)
                for tag_id, pose in poses.items()
            ],
            world_from_camera,
            CAMERA_MATRIX,
            np.zeros(5),
        )

    progress = accumulator.progress()
    by_position = {
        item["position"]: item for item in progress["robot_positions"]
    }

    assert progress["complete"] is True
    assert progress["missing_robot_tag_ids"] == [3]
    assert progress["missing_robot_positions"] == []
    assert by_position["L1 hip"]["tag_id"] == 99
    assert by_position["L1 hip"]["replacement"] is True
    assert by_position["L1 hip"]["match_distance_mm"] < 0.1
    assert by_position["L0 hip"]["identity_reference"] is True


def test_leg_zero_reference_is_not_silently_replaced_and_seen_is_not_unseen() -> None:
    world_from_camera = RigidTransform(
        np.asarray([0.0, 0.0, 0.70]),
        Rotation.from_euler("x", 180.0, degrees=True),
    )
    accumulator = TagSurveyAccumulator(
        robot_tags={
            0: {"label": "body", "frame": "body", "photo_center_px": [0, 0]},
            1: {"label": "L0 hip", "frame": "L0_coxa", "photo_center_px": [100, 0]},
            2: {"label": "L0 knee", "frame": "L0_femur", "photo_center_px": [0, 100]},
        },
        expected_ground_ids=[105],
        marker_size_m=0.027,
        position_tag_overrides={"L0_coxa": 1},
        options=TagSurveyOptions(min_observations=3),
    )
    poses = {
        0: RigidTransform(np.asarray([0.0, 0.0, 0.15]), Rotation.identity()),
        2: RigidTransform(np.asarray([0.0, 0.10, 0.15]), Rotation.identity()),
        98: RigidTransform(np.asarray([0.10, 0.0, 0.15]), Rotation.identity()),
        105: RigidTransform(np.asarray([-0.12, 0.08, 0.0]), Rotation.identity()),
    }
    accumulator.observe_frame(
        [
            _detection(tag_id, 0.027, pose, world_from_camera)
            for tag_id, pose in poses.items()
        ],
        world_from_camera,
        CAMERA_MATRIX,
        np.zeros(5),
    )

    progress = accumulator.progress()
    l0_hip = next(
        item for item in progress["robot_positions"]
        if item["frame"] == "L0_coxa"
    )

    assert l0_hip["tag_id"] is None
    assert l0_hip["state"] == "not_seen"
    assert "L0 hip" in progress["unseen_robot_positions"]
    assert progress["ground_tags_needing_another_view"] == [105]
    assert progress["unseen_ground_tag_ids"] == []


def test_stable_tags_correct_a_drifting_arkit_camera_pose() -> None:
    actual_camera = RigidTransform(
        np.asarray([0.01, -0.02, 0.72]),
        Rotation.from_euler("xyz", [180.0, 0.0, 4.0], degrees=True),
    )
    tags = {
        12: RigidTransform(
            np.asarray([-0.10, 0.06, 0.0]),
            Rotation.from_euler("z", 12.0, degrees=True),
        ),
        13: RigidTransform(
            np.asarray([0.11, -0.05, 0.0]),
            Rotation.from_euler("z", -18.0, degrees=True),
        ),
        14: RigidTransform(
            np.asarray([0.02, 0.14, 0.0]),
            Rotation.from_euler("z", 31.0, degrees=True),
        ),
    }
    accumulator = TagSurveyAccumulator(
        robot_tags={},
        expected_ground_ids=[12, 13, 14],
        marker_size_m=0.027,
        options=TagSurveyOptions(min_observations=4),
    )
    detections = [
        _detection(tag_id, 0.027, transform, actual_camera)
        for tag_id, transform in tags.items()
    ]
    for _ in range(5):
        accumulator.observe_frame(
            detections, actual_camera, CAMERA_MATRIX, np.zeros(5)
        )
    drifted_camera = RigidTransform(
        actual_camera.translation_m + [0.08, -0.04, 0.03],
        Rotation.from_euler("z", 6.0, degrees=True) * actual_camera.rotation,
    )

    outlier_detection = _detection(
        14,
        0.027,
        RigidTransform(
            tags[14].translation_m + [0.20, 0.0, 0.0],
            tags[14].rotation,
        ),
        actual_camera,
    )
    corrected = accumulator.estimate_world_from_camera(
        [detections[0], detections[1], outlier_detection],
        drifted_camera,
        CAMERA_MATRIX,
        np.zeros(5),
    )

    assert corrected is not None and corrected.stable
    assert corrected.input_count == 3
    assert corrected.used_count == 2
    assert np.allclose(
        corrected.transform.translation_m,
        actual_camera.translation_m,
        atol=1e-5,
    )
    assert math.degrees(float(
        (corrected.transform.rotation.inv() * actual_camera.rotation).magnitude()
    )) < 0.001


def test_live_mode_freezes_a_stable_tag_as_a_landmark() -> None:
    world_from_camera = RigidTransform(
        np.asarray([0.0, 0.0, 0.7]),
        Rotation.from_euler("xyz", [180.0, 0.0, 3.0], degrees=True),
    )
    expected = RigidTransform(
        np.asarray([0.12, 0.06, 0.0]),
        Rotation.from_euler("z", 17.0, degrees=True),
    )
    accumulator = TagSurveyAccumulator(
        robot_tags={},
        expected_ground_ids=[12],
        marker_size_m=0.027,
        options=TagSurveyOptions(
            min_observations=4,
            freeze_stable_tags=True,
        ),
    )
    for _ in range(5):
        accumulator.observe_frame(
            [_detection(12, 0.027, expected, world_from_camera)],
            world_from_camera,
            CAMERA_MATRIX,
            np.zeros(5),
        )
    for _ in range(5):
        accumulator.observe_frame(
            [_detection(12, 0.027, expected, world_from_camera)],
            RigidTransform(
                world_from_camera.translation_m + [0.15, 0.0, 0.0],
                world_from_camera.rotation,
            ),
            CAMERA_MATRIX,
            np.zeros(5),
        )

    record = accumulator.summary()["tags"][0]

    assert record["stable"] is True
    assert record["possible_duplicate_id_or_tracking_jump"] is False
    assert record["observations"] == 4


def test_zero_pose_survey_offline_guided_flow(tmp_path) -> None:
    config_path = tmp_path / "tracker.json"
    board_path = tmp_path / "board.json"
    frames_path = tmp_path / "frames"
    output_path = tmp_path / "survey.json"
    updated_path = tmp_path / "tracker-surveyed.json"
    progress_path = tmp_path / "progress.json"
    camera_preview_path = tmp_path / "camera.jpg"
    frames_path.mkdir()
    config_path.write_text(json.dumps({
        "tag_family": "tag36h11",
        "marker_size_m": 0.027,
        "camera": {
            "image_size_px": [640, 480],
            "camera_matrix": [[600, 0, 320], [0, 600, 240], [0, 0, 1]],
            "distortion_coefficients": [0, 0, 0, 0, 0],
        },
        "floor_tags": {
            "12": {"world_from_tag": {"translation_m": [0, 0, 0]}}
        },
        "robot_pose": {
            "tags": {
                "0": {"label": "body", "frame": "body", "frame_from_tag": {}}
            }
        },
    }))
    board_path.write_text(json.dumps({
        "marker_size_m": 0.070,
        "floor_tags": {
            "40": {
                "world_from_tag": {
                    "translation_m": [-0.070, 0.050, 0.0]
                }
            }
        },
    }))
    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_APRILTAG_36h11
    )

    def place(canvas, tag_id, center, size):
        marker = cv2.aruco.generateImageMarker(
            dictionary, tag_id, size, borderBits=1
        )
        x, y = center
        half = size // 2
        canvas[y - half:y - half + size, x - half:x - half + size] = marker

    rgb = np.full((480, 640), 255, dtype=np.uint8)
    place(rgb, 40, (180, 140), 140)
    place(rgb, 12, (500, 80), 54)
    place(rgb, 0, (453, 357), 90)
    depth = np.full((120, 160), 0.300, dtype=np.float32)
    confidence = np.full(depth.shape, 2, dtype=np.uint8)
    matrix = np.asarray([[600, 0, 320], [0, 600, 240], [0, 0, 1]])
    # OpenGL camera pose is identity orientation here because the OpenCV
    # camera looking down differs by the explicit 180-degree x conversion.
    camera_pose = np.asarray([0, 0, 0, 1, 0, 0, 0.300], dtype=float)
    for index in range(6):
        np.savez_compressed(
            frames_path / f"{index:02d}.npz",
            rgb=rgb,
            depth=depth,
            confidence=confidence,
            camera_matrix=matrix,
            camera_pose_xyzw_xyz=camera_pose,
        )

    result = zero_pose_survey_main([
        str(config_path),
        "--board", str(board_path),
        "--npz-dir", str(frames_path),
        "--output", str(output_path),
        "--updated-config", str(updated_path),
        "--progress-output", str(progress_path),
        "--camera-preview-output", str(camera_preview_path),
        "--anchor-frames", "4",
        "--min-observations", "3",
        "--settle-frames", "1",
    ])

    assert result == 0
    payload = json.loads(output_path.read_text())
    updated = json.loads(updated_path.read_text())
    progress = json.loads(progress_path.read_text())
    assert payload["motor_commands_sent"] is False
    assert payload["alignment"]["stable"] is True
    assert payload["survey"]["complete"] is True
    records = {item["tag_id"]: item for item in payload["survey"]["tags"]}
    assert records[0]["role"] == "robot"
    assert records[12]["role"] == "ground"
    assert records[40]["role"] == "calibration_anchor"
    assert updated["floor_tags"]["12"]["marker_size_m"] == 0.027
    assert updated["floor_tags"]["40"]["marker_size_m"] == 0.070
    assert "mount_source" not in updated["robot_pose"]["tags"]["0"]
    assert progress["status"] == "complete"
    assert progress["phase"] == "review"
    assert progress["progress"]["complete"] is True
    assert sorted(progress["detected_tag_ids"]) == [0, 12, 40]
    assert camera_preview_path.is_file()
