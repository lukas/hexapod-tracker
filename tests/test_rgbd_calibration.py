"""Off-robot tests for iPhone LiDAR-assisted AprilTag calibration."""
from __future__ import annotations

import json
import math
from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from hexapod_tracker.apriltag_vision import (
    AprilTagPoseTracker,
    TagCorners,
    marker_object_corners,
)
from hexapod_tracker.calibration_board import make_board_files
from hexapod_tracker.housing_pose import RigidTransform
from hexapod_tracker.rgbd_calibrate import main as rgbd_calibrate_main
from hexapod_tracker.rgbd_calibration import (
    RGBDCalibrationError,
    RGBDCalibrationOptions,
    average_fixed_camera_calibration,
    depth_points_for_floor_tags,
    refine_world_reference_with_depth,
)
from hexapod_tracker.track import Record3DTrackingCapture


def _synthetic_observation(*, seed: int = 2):
    marker_size_m = 0.07
    image_size = (1280, 720)
    depth_size = (256, 144)
    camera_matrix = np.asarray([
        [930.0, 0.0, 640.0],
        [0.0, 925.0, 360.0],
        [0.0, 0.0, 1.0],
    ])
    world_from_camera = RigidTransform(
        np.asarray([0.025, -0.035, 0.76]),
        Rotation.from_euler("xyz", [178.0, -5.0, 7.0], degrees=True),
    )
    floor_tags = {
        40: RigidTransform(
            np.asarray([-0.09, 0.045, 0.0]), Rotation.identity()
        ),
        41: RigidTransform(
            np.asarray([0.0, 0.045, 0.0]), Rotation.identity()
        ),
        42: RigidTransform(
            np.asarray([0.09, 0.045, 0.0]), Rotation.identity()
        ),
        43: RigidTransform(
            np.asarray([-0.09, -0.045, 0.0]), Rotation.identity()
        ),
        44: RigidTransform(
            np.asarray([0.0, -0.045, 0.0]), Rotation.identity()
        ),
        45: RigidTransform(
            np.asarray([0.09, -0.045, 0.0]), Rotation.identity()
        ),
    }
    camera_from_world = world_from_camera.inverse()
    rng = np.random.default_rng(seed)
    detections: list[TagCorners] = []
    for tag_id, world_from_tag in floor_tags.items():
        world_points = np.stack([
            world_from_tag.apply(point)
            for point in marker_object_corners(marker_size_m)
        ])
        camera_points = np.stack([
            camera_from_world.apply(point) for point in world_points
        ])
        pixels = np.column_stack([
            camera_matrix[0, 0] * camera_points[:, 0] / camera_points[:, 2]
            + camera_matrix[0, 2],
            camera_matrix[1, 1] * camera_points[:, 1] / camera_points[:, 2]
            + camera_matrix[1, 2],
        ])
        pixels += rng.normal(0.0, 0.18, pixels.shape)
        detections.append(TagCorners(tag_id, pixels.astype(np.float32)))

    depth_width, depth_height = depth_size
    rgb_width, rgb_height = image_size
    fx = camera_matrix[0, 0] * depth_width / rgb_width
    fy = camera_matrix[1, 1] * depth_height / rgb_height
    cx = camera_matrix[0, 2] * depth_width / rgb_width
    cy = camera_matrix[1, 2] * depth_height / rgb_height
    columns, rows = np.meshgrid(
        np.arange(depth_width, dtype=float),
        np.arange(depth_height, dtype=float),
    )
    rays = np.stack([
        (columns - cx) / fx,
        (rows - cy) / fy,
        np.ones_like(columns),
    ], axis=-1)
    # World z=0 transformed into the camera frame.  The scale along a ray is
    # exactly the RGB-D stream's z-depth value.
    plane_normal = camera_from_world.rotation.apply([0.0, 0.0, 1.0])
    plane_point = camera_from_world.translation_m
    depth = (plane_normal @ plane_point) / np.einsum(
        "ijk,k->ij", rays, plane_normal
    )
    depth += rng.normal(0.0, 0.0015, depth.shape)
    # Exercise robust fitting without overwhelming Apple's coarse depth map.
    outliers = rng.random(depth.shape) < 0.12
    depth[outliers] += rng.normal(0.08, 0.02, np.count_nonzero(outliers))
    confidence = np.full(depth.shape, 2, dtype=np.uint8)

    observation = refine_world_reference_with_depth(
        detections,
        floor_tags,
        camera_matrix,
        np.zeros(5),
        depth.astype(np.float32),
        image_size_px=image_size,
        confidence=confidence,
        marker_size_m=marker_size_m,
    )
    return observation, world_from_camera


def test_board_generator_writes_dimensioned_tag_map(tmp_path) -> None:
    svg_path = tmp_path / "board.svg"
    manifest_path = tmp_path / "board.json"

    manifest = make_board_files(svg_path, manifest_path)

    assert manifest["marker_size_m"] == 0.07
    assert set(map(int, manifest["floor_tags"])) == set(range(40, 44))
    assert manifest["floor_tags"]["40"]["world_from_tag"]["translation_m"] == [
        -0.044,
        0.044,
        0.0,
    ]
    assert manifest["floor_tags"]["43"]["world_from_tag"]["translation_m"] == [
        0.044,
        -0.044,
        0.0,
    ]
    assert json.loads(manifest_path.read_text()) == manifest
    svg = svg_path.read_text()
    assert 'width="188.000mm"' in svg
    assert 'height="206.000mm"' in svg
    assert svg.count("data:image/png;base64,") == 4
    assert "PRINT AT 100%" in svg


def test_rgbd_refinement_recovers_fixed_camera_pose() -> None:
    observation, expected = _synthetic_observation()

    translation_error = np.linalg.norm(
        observation.world_from_camera.translation_m - expected.translation_m
    )
    rotation_error = (
        observation.world_from_camera.rotation.inv() * expected.rotation
    ).magnitude()
    assert translation_error < 0.004
    assert math.degrees(float(rotation_error)) < 0.35
    assert observation.reprojection_rms_px < 0.5
    assert observation.depth_plane_rms_m < 0.004
    assert observation.depth_inlier_fraction > 0.75


def test_depth_sampling_enforces_confidence_gate() -> None:
    corners = TagCorners(40, np.asarray([
        [200.0, 150.0],
        [440.0, 150.0],
        [440.0, 330.0],
        [200.0, 330.0],
    ], dtype=np.float32))
    depth = np.full((120, 160), 0.7, dtype=np.float32)
    confidence = np.zeros_like(depth, dtype=np.uint8)
    matrix = np.asarray([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0, 0, 1]])

    try:
        depth_points_for_floor_tags(
            depth,
            matrix,
            [corners],
            image_size_px=(640, 480),
            confidence=confidence,
            options=RGBDCalibrationOptions(min_confidence=1),
        )
    except RGBDCalibrationError as error:
        assert "usable depth samples" in str(error)
    else:
        raise AssertionError("low-confidence depth should have been rejected")


def test_fixed_camera_consensus_rejects_moved_frame() -> None:
    base, _expected = _synthetic_observation()
    rng = np.random.default_rng(81)
    observations = [
        replace(
            base,
            world_from_camera=RigidTransform(
                base.world_from_camera.translation_m
                + rng.normal(0.0, 0.0007, 3),
                base.world_from_camera.rotation,
            ),
        )
        for _ in range(8)
    ]
    moved = replace(
        base,
        world_from_camera=RigidTransform(
            base.world_from_camera.translation_m + [0.09, 0.0, 0.0],
            base.world_from_camera.rotation,
        ),
    )

    calibration = average_fixed_camera_calibration(
        [*observations, moved], min_frames=8
    )

    assert calibration.accepted_frames == 8
    assert calibration.rejected_outlier_frames == 1
    assert calibration.translation_spread_mm < 3.0


def test_tracker_uses_saved_fixed_reference_when_board_is_out_of_view() -> None:
    fixed = {
        "translation_m": [0.11, -0.04, 0.82],
        "euler_xyz_deg": [180.0, 0.0, 4.0],
    }
    tracker = AprilTagPoseTracker({
        "tag_family": "tag36h11",
        "marker_size_m": 0.027,
        "floor_marker_size_m": 0.07,
        "camera": {
            "image_size_px": [640, 480],
            "camera_matrix": [[600, 0, 320], [0, 600, 240], [0, 0, 1]],
            "distortion_coefficients": [0, 0, 0, 0, 0],
        },
        "fixed_camera_world_reference": fixed,
    })

    result, _rendered = tracker.process_frame(
        np.full((480, 640, 3), 255, dtype=np.uint8),
        render_overlay=False,
    )

    assert tracker.marker_size_m == 0.027
    assert tracker.floor_marker_size_m == 0.07
    assert result["pose_reference"] == "floor"
    assert result["world_reference"]["source"] == "fixed_camera"
    assert result["world_reference"]["reprojection_rms_px"] is None
    assert np.allclose(
        result["world_reference"]["world_from_camera"]["translation_m"],
        fixed["translation_m"],
    )


def test_offline_cli_writes_drop_in_tracker_config(tmp_path) -> None:
    config_path = tmp_path / "tracker.json"
    board_path = tmp_path / "board.json"
    output_path = tmp_path / "calibration.json"
    updated_path = tmp_path / "tracker-calibrated.json"
    frames_path = tmp_path / "frames"
    frames_path.mkdir()
    config_path.write_text(json.dumps({
        "tag_family": "tag36h11",
        "marker_size_m": 0.027,
        "marker_size_verified": True,
        "camera": {
            "image_size_px": [640, 480],
            "camera_matrix": [[600, 0, 320], [0, 600, 240], [0, 0, 1]],
            "distortion_coefficients": [0, 0, 0, 0, 0],
        },
        "floor_tags": {},
    }))
    board_path.write_text(json.dumps({
        "marker_size_m": 0.07,
        "floor_tags": {
            "40": {
                "world_from_tag": {
                    "translation_m": [0, 0, 0],
                    "euler_xyz_deg": [0, 0, 0],
                }
            }
        },
    }))
    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_APRILTAG_36h11
    )
    marker = cv2.aruco.generateImageMarker(dictionary, 40, 160, borderBits=1)
    rgb = np.full((480, 640), 255, dtype=np.uint8)
    rgb[160:320, 240:400] = marker
    # A 70 mm tag spanning about 160 pixels at fx=600 lies 262.5 mm away.
    depth = np.full((120, 160), 0.2625, dtype=np.float32)
    confidence = np.full(depth.shape, 2, dtype=np.uint8)
    matrix = np.asarray([[600, 0, 320], [0, 600, 240], [0, 0, 1]])
    for index in range(8):
        np.savez_compressed(
            frames_path / f"{index:02d}.npz",
            rgb=rgb,
            depth=depth,
            confidence=confidence,
            camera_matrix=matrix,
        )

    return_code = rgbd_calibrate_main([
        str(config_path),
        "--board", str(board_path),
        "--npz-dir", str(frames_path),
        "--frames", "8",
        "--min-tags", "1",
        "--output", str(output_path),
        "--updated-config", str(updated_path),
    ])

    assert return_code == 0
    calibration = json.loads(output_path.read_text())
    updated = json.loads(updated_path.read_text())
    assert calibration["quality"]["accepted_frames"] == 8
    assert calibration["motor_commands_sent"] is False
    assert updated["marker_size_m"] == 0.027
    assert updated["floor_marker_size_m"] == 0.07
    assert set(updated["floor_tags"]) == {"40"}
    assert updated["camera"]["approximate"] is False
    assert "fixed_camera_world_reference" in updated


def test_record3d_tracking_capture_uses_intrinsics_from_each_frame() -> None:
    tracker = AprilTagPoseTracker({
        "camera": {
            "image_size_px": [320, 240],
            "camera_matrix": [[300, 0, 160], [0, 300, 120], [0, 0, 1]],
        }
    })

    class FakeReader:
        closed = False

        def next_frame(self):
            return SimpleNamespace(
                rgb_bgr=np.zeros((480, 640, 3), dtype=np.uint8),
                camera_matrix=np.asarray([
                    [615.0, 0.0, 318.0],
                    [0.0, 617.0, 239.0],
                    [0.0, 0.0, 1.0],
                ]),
            )

        def close(self):
            self.closed = True

    fake = FakeReader()
    capture = Record3DTrackingCapture(0, tracker, reader=fake)

    ok, frame = capture.read()

    assert ok is True
    assert frame.shape == (480, 640, 3)
    assert tracker.calibration.image_size_px == (640, 480)
    assert np.allclose(tracker.calibration.camera_matrix[:2, :], [
        [615.0, 0.0, 318.0],
        [0.0, 617.0, 239.0],
    ])
    assert capture.get(cv2.CAP_PROP_FPS) == 30.0
    capture.release()
    assert fake.closed is True
