import math

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from hexapod_tracker.planar_pose import PlanarPoseEstimator, plate_tag_entries, plate_tag_id, plate_tag_pose


def corners(center, yaw_degrees, size=27.0):
    half = size / 2.0
    offsets = np.asarray(
        [[-half, half], [half, half], [half, -half], [-half, -half]],
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


def test_planar_estimator_does_not_report_vertical_tag_projection_as_pose():
    floor_map = {
        "tag_black_square_size": 27.0,
        "active_anchor_ids": [12, 13, 15],
        "tags": [
            {"id": 12, "center": [0, 0, 0], "yaw_degrees": 0},
            {"id": 13, "center": [0, 600, 0], "yaw_degrees": 0},
            {"id": 15, "center": [300, 100, 0], "yaw_degrees": 0},
        ],
    }
    part_map = {
        "parts": [
            {"id": "leg0_hip_servo", "tag_ids": [42], "surface": "vertical"}
        ]
    }
    tags = {
        12: corners((0, 0), 0),
        13: corners((0, 600), 0),
        15: corners((300, 100), 0),
        42: corners((125, 240), 32),
    }

    payload = PlanarPoseEstimator(floor_map, part_map).estimate(
        [{"index": 0, "width": 1280, "height": 800, "tags": tags}]
    )

    part = payload["parts"]["leg0_hip_servo"]
    assert part["status"] == "projection_only"
    assert part["pose"]["rotation_degrees"]["yaw_axis"] is None
    assert part["pose"]["error_95_estimate"]["position_mm"] is None
    assert part["projection_diagnostic"]["projected_tag_edge_yaw_degrees"] == 32.0


def test_floor_rectification_recovers_robot_relative_yaw_from_horizontal_tags():
    floor_map = {
        "tag_black_square_size": 27.0,
        "active_anchor_ids": [100, 101, 102],
        "tags": [
            {"id": 100, "center": [0, 0, 0], "yaw_degrees": 0},
            {"id": 101, "center": [0, 600, 0], "yaw_degrees": 0},
            {"id": 102, "center": [600, 0, 0], "yaw_degrees": 0},
        ],
    }
    layout = {
        "robot_tags": [
            {
                "id": 0,
                "kind": "chassis_tag",
                "frame": "body",
                "surface": "horizontal",
                "frame_from_tag": {"euler_xyz_deg": [0, 0, 159.2]},
            },
            {
                "id": 1,
                "kind": "servo_lid",
                "leg": 0,
                "joint": "hip",
                "frame": "L0_coxa",
                "surface": "horizontal",
                "frame_from_tag": {"euler_xyz_deg": [0, 0, 90]},
            },
        ]
    }
    # Body is at +15 degrees. L0's zero azimuth is +30 degrees and its
    # commanded joint yaw is +12 degrees, so the coxa frame is at +57.
    world_tags = {
        100: corners((0, 0), 0),
        101: corners((0, 600), 0),
        102: corners((600, 0), 0),
        0: corners((250, 250), 15 + 159.2),
        1: corners((330, 330), 57 + 90),
    }
    homography = np.asarray(
        [[1.2, 0.1, 200], [-0.08, 1.05, 80], [0.00015, 0.0002, 1]],
        dtype=np.float32,
    )
    image_tags = {
        tag_id: cv2.perspectiveTransform(points[None], homography)[0]
        for tag_id, points in world_tags.items()
    }

    payload = PlanarPoseEstimator(floor_map, {"parts": []}, layout).estimate(
        [{"index": 2, "width": 1280, "height": 800, "tags": image_tags}]
    )

    camera_pose = payload["camera_joint_pose"]
    yaw = camera_pose["joints"]["L0_yaw"]
    assert camera_pose["joint_frame"] == "robot_abs"
    assert camera_pose["tracked_joint_count"] == 1
    assert yaw["status"] == "tracked"
    assert abs(yaw["value_deg"] - 12.0) < 0.02
    assert yaw["error_95_estimate_deg"] == 5.0
    assert yaw["camera_indices"] == [2]
    assert "paired same-camera" in yaw["uncertainty_method"]
    assert yaw["chassis_tag_id"] == 0
    assert yaw["servo_lid_tag_id"] == 1
    assert camera_pose["joints"]["L1_yaw"]["status"] == "layout_unavailable"


def test_yaw_uses_measured_azimuth_and_sign_from_the_layout():
    floor_map = {
        "tag_black_square_size": 27.0,
        "active_anchor_ids": [100, 101, 102],
        "tags": [
            {"id": 100, "center": [0, 0, 0], "yaw_degrees": 0},
            {"id": 101, "center": [0, 600, 0], "yaw_degrees": 0},
            {"id": 102, "center": [600, 0, 0], "yaw_degrees": 0},
        ],
    }
    layout = {
        "robot_tags": [
            {"id": 0, "kind": "chassis_tag", "frame": "body", "surface": "horizontal",
             "frame_from_tag": {"euler_xyz_deg": [0, 0, 10.6]}},
            {"id": 1, "kind": "servo_lid", "leg": 0, "joint": "hip", "frame": "L0_coxa",
             "surface": "horizontal", "frame_from_tag": {"euler_xyz_deg": [0, 0, 90]}},
        ],
        # As measured on 2026-09-11: legs numbered clockwise from above, leg 0 at
        # -18.5 deg (its yaw servo zero is 11.5 deg off nominal), and a positive
        # yaw command turns the leg clockwise, i.e. negative in the z-up frame.
        "leg_zero_azimuth_body_deg": {"0": -18.5, "1": -95.7},
        "joint_conventions": {"yaw_positive_seen_from_above": "clockwise", "yaw_sign_in_body_frame": -1},
    }
    # Body at +15. A commanded yaw of +12 turns leg 0 clockwise: coxa heading
    # is 15 + (-18.5) - 12 = -15.5.
    world_tags = {
        100: corners((0, 0), 0),
        101: corners((0, 600), 0),
        102: corners((600, 0), 0),
        0: corners((250, 250), 15 + 10.6),
        1: corners((330, 330), -15.5 + 90),
    }
    homography = np.asarray(
        [[1.2, 0.1, 200], [-0.08, 1.05, 80], [0.00015, 0.0002, 1]], dtype=np.float32
    )
    image_tags = {
        tag_id: cv2.perspectiveTransform(points[None], homography)[0]
        for tag_id, points in world_tags.items()
    }
    estimator = PlanarPoseEstimator(floor_map, {"parts": []}, layout)
    assert estimator.leg_zero_azimuth(0) == -18.5
    assert estimator.leg_zero_azimuth(3) == 210.0  # no measurement: historical fallback

    yaw = estimator.estimate(
        [{"index": 2, "width": 1280, "height": 800, "tags": image_tags}]
    )["camera_joint_pose"]["joints"]["L0_yaw"]
    assert yaw["status"] == "tracked"
    assert abs(yaw["value_deg"] - 12.0) < 0.02
    assert yaw["leg_zero_azimuth_body_deg"] == -18.5
    assert yaw["yaw_sign_in_body_frame"] == -1


def test_intrinsic_calibration_recovers_hip_from_body_and_femur_tags():
    camera_matrix = np.asarray(
        [[800.0, 0.0, 640.0], [0.0, 800.0, 400.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    camera_from_world = Rotation.from_euler("x", 180, degrees=True)
    camera_translation_mm = np.asarray([0.0, 0.0, 1000.0])
    floor_map = {
        "tag_black_square_size": 27.2,
        "active_anchor_ids": [100, 101, 102, 103],
        "tags": [
            {"id": 100, "center": [-300, -300, 0], "yaw_degrees": 0},
            {"id": 101, "center": [300, -300, 0], "yaw_degrees": 0},
            {"id": 102, "center": [300, 300, 0], "yaw_degrees": 0},
            {"id": 103, "center": [-300, 300, 0], "yaw_degrees": 0},
        ],
    }
    layout = {
        "tag_geometry": {"black_square_m": 0.0272},
        "robot_tags": [
            {
                "id": 0,
                "kind": "chassis_tag",
                "frame": "body",
                "surface": "horizontal",
                "frame_from_tag": {"euler_xyz_deg": [0, 0, 20]},
            },
            {
                "id": 20,
                "kind": "yoke_face",
                "leg": 0,
                "joint": "hip",
                "frame": "L0_femur",
                "frame_from_tag": {"euler_xyz_deg": [0, 0, 0]},
            },
            {
                "id": 21,
                "kind": "yoke_face",
                "leg": 0,
                "joint": "knee",
                "frame": "L0_tibia",
                "frame_from_tag": {"euler_xyz_deg": [0, 0, 0]},
            },
        ],
    }
    calibration = {
        "quality": "test",
        "image_size": {"width": 1280, "height": 800},
        "cameras": {
            "2": {
                "camera_matrix": camera_matrix.tolist(),
                "distortion_coefficients": [0, 0, 0, 0, 0],
                "floor_reprojection_rms_px": 0.0,
            }
        },
    }

    def project_floor_tag(tag):
        points = np.column_stack([corners(tag["center"][:2], 0, 27.2), np.zeros(4)])
        camera_points = camera_from_world.apply(points) + camera_translation_mm
        return np.column_stack(
            [
                camera_matrix[0, 0] * camera_points[:, 0] / camera_points[:, 2] + 640,
                camera_matrix[1, 1] * camera_points[:, 1] / camera_points[:, 2] + 400,
            ]
        ).astype(np.float32)

    tag_object = np.asarray(
        [[-0.0136, 0.0136, 0], [0.0136, 0.0136, 0], [0.0136, -0.0136, 0], [-0.0136, -0.0136, 0]],
        dtype=np.float32,
    )

    def project_robot_tag(world_from_tag, center_m):
        camera_from_tag = camera_from_world * world_from_tag
        rvec = camera_from_tag.as_rotvec().reshape(3, 1)
        tvec = (
            camera_from_world.apply(np.asarray(center_m))
            + np.asarray([0.0, 0.0, 1.0])
        ).reshape(3, 1)
        projected, _ = cv2.projectPoints(
            tag_object, rvec, tvec, camera_matrix, np.zeros(5)
        )
        return projected[:, 0].astype(np.float32)

    world_from_body = Rotation.from_euler("z", 10, degrees=True)
    body_from_tag = Rotation.from_euler("z", 20, degrees=True)
    body_from_femur = Rotation.from_euler("z", 35, degrees=True) * Rotation.from_euler(
        "y", 12, degrees=True
    )
    body_from_tibia = Rotation.from_euler("z", 35, degrees=True) * Rotation.from_euler(
        "y", -22, degrees=True
    )
    tags = {
        **{tag["id"]: project_floor_tag(tag) for tag in floor_map["tags"]},
        0: project_robot_tag(world_from_body * body_from_tag, [0.0, 0.0, 0.10]),
        20: project_robot_tag(world_from_body * body_from_femur, [0.10, 0.0, 0.08]),
        21: project_robot_tag(world_from_body * body_from_tibia, [0.18, 0.0, 0.04]),
    }

    payload = PlanarPoseEstimator(
        floor_map, {"parts": []}, layout, calibration
    ).estimate([{"index": 2, "width": 1280, "height": 800, "tags": tags}])

    hip = payload["camera_joint_pose"]["joints"]["L0_hip"]
    assert hip["status"] == "tracked"
    assert abs(hip["value_deg"] - 12.0) < 0.1
    assert hip["tag_ids"] == [20]
    knee = payload["camera_joint_pose"]["joints"]["L0_knee"]
    assert knee["status"] == "tracked"
    assert abs(knee["value_deg"] - -22.0) < 0.1
    assert knee["tag_ids"] == [21]
    assert payload["camera_joint_pose"]["tracked_knee_count"] == 1
    intrinsic = payload["calibration"]["intrinsics"]["cameras"]["2"]
    assert intrinsic["quality"] == "test"
    assert intrinsic["image_size"] == {"width": 1280, "height": 800}
    assert intrinsic["floor_reprojection_rms_px"] == 0.0


def test_fixed_camera_keeps_its_floor_calibration_while_anchors_are_hidden():
    floor_map = {
        "tag_black_square_size": 27.0,
        "active_anchor_ids": [100, 101, 102],
        "tags": [
            {"id": 100, "center": [0, 0, 0], "yaw_degrees": 0},
            {"id": 101, "center": [0, 600, 0], "yaw_degrees": 0},
            {"id": 102, "center": [600, 0, 0], "yaw_degrees": 0},
        ],
    }
    homography = np.asarray([[1.2, 0.1, 200], [-0.08, 1.05, 80], [0.00015, 0.0002, 1]], dtype=np.float32)
    def image(world):
        return {t: cv2.perspectiveTransform(c[None], homography)[0] for t, c in world.items()}
    estimator = PlanarPoseEstimator(floor_map, {"parts": []})
    first = estimator.estimate([{"index": 2, "width": 1280, "height": 800,
                                 "tags": image({100: corners((0, 0), 0), 101: corners((0, 600), 0),
                                                102: corners((600, 0), 0), 7: corners((250, 250), 20)})}])
    assert first["calibration"]["cameras"]["2"]["status"] == "calibrated"
    assert abs(first["markers"]["7"]["position_mm"]["x"] - 250) < 2
    # the robot walks over the floor tags: only the chassis tag is left in view
    second = estimator.estimate([{"index": 2, "width": 1280, "height": 800,
                                  "tags": image({7: corners((310, 250), 20)})}])
    cal = second["calibration"]["cameras"]["2"]
    assert cal["status"] == "held" and cal["anchor_ids"] == [100, 101, 102] and cal["held_for_s"] is not None
    assert second["markers"]["7"]["status"] == "tracked"
    assert abs(second["markers"]["7"]["position_mm"]["x"] - 310) < 2
    # a different image size or an expired hold is not reused
    estimator.hold_calibration_s = -1.0
    third = estimator.estimate([{"index": 2, "width": 1280, "height": 800, "tags": image({7: corners((310, 250), 20)})}])
    assert third["calibration"]["cameras"]["2"]["status"] == "uncalibrated" and "7" not in third["markers"]


def _plate_floor_map():
    """Two floor anchors plus a 50 mm tag sitting 3 mm up on a plate (own black_square_size, z in center)."""
    return {
        "tag_black_square_size": 27.0,
        "active_anchor_ids": [12, 13, 400],
        "coordinate_frame": {"origin": "tag 12"},
        "tags": [
            {"id": 12, "center": [0, 0, 0], "yaw_degrees": 0},
            {"id": 13, "center": [0, 600, 0], "yaw_degrees": 10},
            {"id": 400, "center": [300, 100, 3.0], "yaw_degrees": 90, "black_square_size": 50.0},
        ],
    }


def test_anchor_corners_use_per_tag_size_and_height():
    est = PlanarPoseEstimator(_plate_floor_map(), {"parts": []})
    assert est.anchor_size_mm(12) == 27.0
    assert est.anchor_size_mm(400) == 50.0
    small = est.anchor_corners(12)
    big = est.anchor_corners(400)
    assert abs(np.linalg.norm(small[1] - small[0]) - 27.0) < 1e-6
    assert abs(np.linalg.norm(big[1] - big[0]) - 50.0) < 1e-6
    assert est.anchor_height_mm(400) == 3.0
    assert est.anchor_corners_3d(400).shape == (4, 3)
    assert np.allclose(est.anchor_corners_3d(400)[:, 2], 3.0)
    assert np.allclose(est.anchor_corners_3d(12)[:, 2], 0.0)


def test_raised_anchor_is_flattened_along_the_camera_ray_or_excluded():
    est = PlanarPoseEstimator(_plate_floor_map(), {"parts": []})
    # without a camera position the raised tag cannot join a z = 0 homography
    assert est.anchor_corners_for_view(400, 0) is None
    assert est.homography_anchors({12: None, 13: None, 400: None}, 0) == [12, 13]
    # with one, its corners slide down the rays onto the floor: farther from the nadir by z / (h - z)
    est.camera_positions_mm[0] = np.array([0.0, 0.0, 603.0])
    flat = est.anchor_corners_for_view(400, 0)
    raised = est.anchor_corners(400)
    assert np.allclose(flat, raised * 603.0 / 600.0)
    assert est.homography_anchors({12: None, 13: None, 400: None}, 0) == [12, 13, 400]
    # a floor tag is untouched
    assert np.allclose(est.anchor_corners_for_view(12, 0), est.anchor_corners(12))


def test_homography_fit_with_a_raised_plate_tag_recovers_a_floor_marker():
    """Render the scene with a real pinhole camera (so the plate really is 3 mm up), then fit."""
    floor_map = _plate_floor_map()
    est = PlanarPoseEstimator(floor_map, {"parts": []})
    cam_pos = np.array([150.0, 250.0, 900.0])
    est.camera_positions_mm[0] = cam_pos
    K = np.array([[1200.0, 0, 640.0], [0, 1200.0, 400.0], [0, 0, 1]])
    R = np.diag([1.0, -1.0, -1.0])                       # looking straight down, image y flipped
    t = -R @ cam_pos

    def project(pts3):
        p = (R @ pts3.T).T + t
        return (K @ p.T).T[:, :2] / p[:, 2:3]

    image_tags = {tid: project(est.anchor_corners_3d(tid)) for tid in (12, 13, 400)}
    image_tags[16] = project(np.column_stack([corners((125, 240), 32), np.zeros(4)]))
    payload = est.estimate([{"index": 0, "width": 1280, "height": 800, "frame_age_s": 0.01, "tags": image_tags}])
    marker = payload["markers"]["16"]
    assert abs(marker["position_mm"]["x"] - 125.0) < 0.1
    assert abs(marker["position_mm"]["y"] - 240.0) < 0.1
    assert payload["calibration"]["cameras"]["0"]["anchor_ids"] == [12, 13, 400]


def test_plate_geometry_is_derived_from_the_map_block():
    plate = {"name": "p", "base_id": 400, "cols": 5, "rows": 4, "pitch_mm": 62.5, "black_square_size": 50.0,
             "z_mm": 3.0, "tag_yaw_in_plate_deg": 90.0}
    assert plate_tag_id(plate, 0, 0) == 400
    assert plate_tag_id(plate, 1, 0) == 404          # ids run down each column
    assert plate_tag_id(plate, 4, 3) == 419
    center, yaw = plate_tag_pose(plate, (100.0, 200.0, 0.0), 2, 1)
    assert np.allclose(center, [225.0, 262.5])
    assert yaw == 90.0
    center, yaw = plate_tag_pose(plate, (0.0, 0.0, 90.0), 1, 0)    # columns now run along +y
    assert np.allclose(center, [0.0, 62.5])
    assert yaw == 180.0 or yaw == -180.0
    entries = plate_tag_entries(plate, (0.0, 0.0, 0.0), source="test")
    assert len(entries) == 20 and entries[0]["center"][2] == 3.0 and entries[0]["black_square_size"] == 50.0
    est = PlanarPoseEstimator({"tag_black_square_size": 27.0, "active_anchor_ids": [], "tags": [], "plates": [plate]}, {"parts": []})
    assert est.plate_tags[419][1:] == (4, 3)


def test_plate_check_reports_the_rigid_residual_through_a_homography():
    plate = {"name": "p", "base_id": 400, "cols": 5, "rows": 4, "pitch_mm": 62.5, "black_square_size": 50.0,
             "z_mm": 0.0, "tag_yaw_in_plate_deg": 90.0}
    fm = {"tag_black_square_size": 27.0, "active_anchor_ids": [], "tags": [], "plates": [plate]}
    est = PlanarPoseEstimator(fm, {"parts": []})
    H = np.asarray([[1.4, 0.15, 300], [-0.1, 1.1, 100], [0.0002, 0.0003, 1]], dtype=np.float64)
    tags = {}
    for tid, (_p, c, r) in est.plate_tags.items():
        center, yaw = plate_tag_pose(plate, (-300.0, 500.0, 7.0), c, r)
        tags[tid] = cv2.perspectiveTransform(corners(center, yaw, 50.0)[None].astype(np.float64), H)[0]
    exact = est.plate_check(tags, H, None)["p"]
    assert exact["tags"] == 20 and exact["rms_mm"] < 0.01
    assert "offset_mm" not in exact                          # plate not in the map yet: shape only
    # once surveyed into the map, the check also reports where this camera puts the plate vs the map
    fixed = {**plate, "fixed": True}
    fm2 = {**fm, "plates": [fixed], "tags": plate_tag_entries(fixed, (-300.0, 500.0, 7.0), source="t"), "active_anchor_ids": list(est.plate_tags)}
    est2 = PlanarPoseEstimator(fm2, {"parts": []})
    assert est2.active_anchor_ids == sorted(est.plate_tags) or set(est2.active_anchor_ids) == set(est.plate_tags)
    # the same plate declared movable (the default) is never a floor anchor, whatever active_anchor_ids says
    loose = PlanarPoseEstimator({**fm2, "plates": [plate]}, {"parts": []})
    assert loose.active_anchor_ids == [] and 400 in loose.plate_tags
    placed = est2.plate_check(tags, H, None)["p"]
    assert placed["offset_norm_mm"] < 0.01 and abs(placed["heading_error_deg"]) < 0.01
    fm3 = {**fm2, "tags": plate_tag_entries(fixed, (-310.0, 500.0, 7.0), source="t")}
    moved = PlanarPoseEstimator(fm3, {"parts": []}).plate_check(tags, H, None)["p"]
    assert abs(moved["offset_norm_mm"] - 10.0) < 0.05
    tags[419] = tags[419] + np.array([[3.0, 0.0]] * 4)       # one tag nudged ~2 mm in the picture
    bent = est.plate_check(tags, H, None)["p"]
    assert 0.3 < bent["rms_mm"] < 3.0 and bent["max_mm"] > bent["rms_mm"]
    assert est.plate_check({400: tags[400]}, H, None) is None
