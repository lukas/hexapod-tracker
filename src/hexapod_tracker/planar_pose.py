"""Best-effort planar pose fusion for the local AprilTag camera service.

The installation has fixed tags on the floor and optional per-camera
intrinsics. This module reports floor-plane pose, robot-relative yaw from
horizontal tags, and hip pitch from relative 3-D body/femur tag orientation.
Metric tag translation remains unavailable where required calibration does
not exist. Hip and absolute tibia/knee pitch are observable from rigid femur
and tibia tags, respectively.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


# Forty stationary live samples on 2026-09-03 showed 2.1-2.3 degree standard
# deviation for relative yaw. Five degrees is therefore the provisional 95%
# repeatability floor until a larger validation dataset replaces it.
RELATIVE_YAW_ERROR_95_FLOOR_DEGREES = 5.0


def _project(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(1, -1, 2)
    return cv2.perspectiveTransform(points, homography)[0].astype(np.float64)


def _angle_difference_degrees(first: float, second: float, period: float = 360.0) -> float:
    return abs((first - second + period / 2.0) % period - period / 2.0)


def _circular_mean_degrees(values: list[float], period: float = 360.0) -> float:
    if not values:
        raise ValueError("a circular mean needs at least one value")
    scale = 2.0 * math.pi / period
    vector = sum(np.exp(1j * np.asarray(values) * scale))
    return float((math.atan2(vector.imag, vector.real) / scale) % period)


def _rounded(value: float | None, digits: int = 3) -> float | None:
    return None if value is None or not math.isfinite(value) else round(float(value), digits)


def _signed_angle_degrees(value: float) -> float:
    """Wrap an angle to the robot joint convention [-180, 180)."""
    return float((value + 180.0) % 360.0 - 180.0)


def _layout_rotation(transform: dict[str, Any]) -> Rotation:
    if "euler_xyz_deg" in transform:
        return Rotation.from_euler("xyz", transform["euler_xyz_deg"], degrees=True)
    if "quaternion_xyzw" in transform:
        quaternion = np.asarray(transform["quaternion_xyzw"], dtype=np.float64)
        return Rotation.from_quat(quaternion / np.linalg.norm(quaternion))
    if "rotation_matrix" in transform:
        return Rotation.from_matrix(transform["rotation_matrix"])
    return Rotation.identity()


@dataclass(frozen=True)
class IntrinsicCalibration:
    camera_index: int
    camera_matrix: np.ndarray
    distortion: np.ndarray
    source_size: tuple[int, int]
    quality: str
    floor_reprojection_rms_px: float | None

    def for_image(self, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        matrix = self.camera_matrix.copy()
        matrix[0, :] *= width / self.source_size[0]
        matrix[1, :] *= height / self.source_size[1]
        return matrix, self.distortion.copy()


@dataclass
class CameraCalibration:
    camera_index: int
    homography: np.ndarray
    anchor_ids: list[int]
    reprojection_rms_px: float
    world_rms_mm: float
    position_error_95_mm: float
    yaw_error_95_degrees: float
    leave_one_out_position_mm: list[float]
    leave_one_out_yaw_degrees: list[float]
    quality: str

    def as_json(self, image_size: tuple[int, int], frame_age_s: float | None) -> dict[str, Any]:
        return {
            "status": "calibrated",
            "quality": self.quality,
            "anchor_ids": self.anchor_ids,
            "image_size": {"width": image_size[0], "height": image_size[1]},
            "frame_age_s": _rounded(frame_age_s),
            "reprojection_rms_px": _rounded(self.reprojection_rms_px),
            "world_rms_mm": _rounded(self.world_rms_mm),
            "leave_one_anchor_out_position_mm": [
                _rounded(value) for value in self.leave_one_out_position_mm
            ],
            "leave_one_anchor_out_yaw_degrees": [
                _rounded(value) for value in self.leave_one_out_yaw_degrees
            ],
            "error_95_estimate": {
                "position_mm": _rounded(self.position_error_95_mm),
                "yaw_degrees": _rounded(self.yaw_error_95_degrees),
            },
            "world_to_image_homography": [
                [_rounded(value, 8) for value in row] for row in self.homography
            ],
        }


class PlanarPoseEstimator:
    """Calibrate each view from fixed tags and fuse ground-projected poses."""

    def __init__(
        self,
        floor_map: dict[str, Any],
        part_map: dict[str, Any],
        robot_layout: dict[str, Any] | None = None,
        camera_calibration: dict[str, Any] | None = None,
    ):
        self.floor_map = floor_map
        self.part_map = part_map
        self.robot_layout = robot_layout
        self.camera_calibration = camera_calibration
        self.tag_size_mm = float(floor_map["tag_black_square_size"])
        active_ids = floor_map.get("active_anchor_ids")
        if active_ids is None:
            active_ids = [tag["id"] for tag in floor_map["tags"] if "yaw_degrees" in tag]
        self.active_anchor_ids = [int(tag_id) for tag_id in active_ids]
        self.anchors = {
            int(tag["id"]): tag
            for tag in floor_map["tags"]
            if int(tag["id"]) in self.active_anchor_ids and "yaw_degrees" in tag
        }
        horizontal_tags = {
            int(tag["id"]): tag
            for tag in (robot_layout or {}).get("robot_tags", [])
            if tag.get("surface") == "horizontal"
            and "euler_xyz_deg" in tag.get("frame_from_tag", {})
        }
        chassis_tags = [
            tag for tag in horizontal_tags.values() if tag.get("kind") == "chassis_tag"
        ]
        self.chassis_tag = chassis_tags[0] if len(chassis_tags) == 1 else None
        self.leg_zero_azimuth_deg = {
            int(leg): float(value)
            for leg, value in (robot_layout or {}).get("leg_zero_azimuth_body_deg", {}).items()
        }
        conventions = (robot_layout or {}).get("joint_conventions") or {}
        self.yaw_sign = float(conventions.get("yaw_sign_in_body_frame") or 1.0)
        self.yaw_lid_by_leg = {
            int(tag["leg"]): tag
            for tag in horizontal_tags.values()
            if tag.get("kind") == "servo_lid"
            and tag.get("joint") == "hip"
            and str(tag.get("frame", "")).endswith("_coxa")
        }
        self.robot_tags = {
            int(tag["id"]): tag for tag in (robot_layout or {}).get("robot_tags", [])
        }
        self.femur_tags = {
            tag_id: tag
            for tag_id, tag in self.robot_tags.items()
            if str(tag.get("frame", "")).endswith("_femur")
        }
        self.tibia_tags = {
            tag_id: tag
            for tag_id, tag in self.robot_tags.items()
            if str(tag.get("frame", "")).endswith("_tibia")
        }
        source_size = (camera_calibration or {}).get("image_size", {})
        source_width = int(source_size.get("width", 1280))
        source_height = int(source_size.get("height", 800))
        self.intrinsic_calibrations: dict[int, IntrinsicCalibration] = {}
        for raw_index, spec in (camera_calibration or {}).get("cameras", {}).items():
            index = int(raw_index)
            camera_size = spec.get("image_size", {})
            self.intrinsic_calibrations[index] = IntrinsicCalibration(
                camera_index=index,
                camera_matrix=np.asarray(spec["camera_matrix"], dtype=np.float64),
                distortion=np.asarray(
                    spec.get("distortion_coefficients", [0.0] * 5), dtype=np.float64
                ),
                source_size=(
                    int(camera_size.get("width", source_width)),
                    int(camera_size.get("height", source_height)),
                ),
                quality=str(
                    spec.get(
                        "quality",
                        (camera_calibration or {}).get("quality", "provisional"),
                    )
                ),
                floor_reprojection_rms_px=(
                    float(spec["floor_reprojection_rms_px"])
                    if spec.get("floor_reprojection_rms_px") is not None
                    else None
                ),
            )

    def _anchor_corners(self, tag_id: int) -> np.ndarray:
        anchor = self.anchors[tag_id]
        center = np.asarray(anchor["center"][:2], dtype=np.float64)
        half = self.tag_size_mm / 2.0
        # OpenCV returns marker corners in canonical top-left, top-right,
        # bottom-right, bottom-left order.  In the tag frame (+Y toward the
        # printed top), that is (-X,+Y), (+X,+Y), (+X,-Y), (-X,-Y).
        # Keeping the same handed ordering here is essential: the former
        # bottom-left-first offsets mirrored every configured floor tag and
        # forced the homography to absorb a roughly one-tag-width residual.
        offsets = np.asarray(
            [[-half, half], [half, half], [half, -half], [-half, -half]],
            dtype=np.float64,
        )
        yaw = math.radians(float(anchor["yaw_degrees"]))
        rotation = np.asarray(
            [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
            dtype=np.float64,
        )
        return center + offsets @ rotation.T

    def _fit_homography(
        self,
        tags: dict[int, np.ndarray],
        anchor_ids: list[int],
    ) -> np.ndarray | None:
        if not anchor_ids:
            return None
        world = np.concatenate([self._anchor_corners(tag_id) for tag_id in anchor_ids])
        image = np.concatenate([tags[tag_id] for tag_id in anchor_ids])
        method = cv2.RANSAC if len(world) > 4 else 0
        homography, _mask = cv2.findHomography(
            world.astype(np.float32), image.astype(np.float32), method, 3.0
        )
        if homography is None or abs(float(np.linalg.det(homography))) < 1e-12:
            return None
        return homography / homography[2, 2]

    def _calibrate_camera(self, snapshot: dict[str, Any]) -> CameraCalibration | None:
        tags = snapshot["tags"]
        visible = [tag_id for tag_id in self.active_anchor_ids if tag_id in tags]
        homography = self._fit_homography(tags, visible)
        if homography is None:
            return None
        inverse = np.linalg.inv(homography)
        world = np.concatenate([self._anchor_corners(tag_id) for tag_id in visible])
        image = np.concatenate([tags[tag_id] for tag_id in visible])
        image_residual = _project(world, homography) - image
        world_residual = _project(image, inverse) - world
        reprojection_rms = float(np.sqrt(np.mean(np.square(image_residual))))
        world_rms = float(np.sqrt(np.mean(np.square(world_residual))))

        leave_position: list[float] = []
        leave_yaw: list[float] = []
        if len(visible) >= 2:
            for held_out in visible:
                training = [tag_id for tag_id in visible if tag_id != held_out]
                trial = self._fit_homography(tags, training)
                if trial is None:
                    continue
                estimated = _project(tags[held_out], np.linalg.inv(trial))
                expected = self._anchor_corners(held_out)
                leave_position.append(
                    float(np.linalg.norm(estimated.mean(axis=0) - expected.mean(axis=0)))
                )
                edge = estimated[1] - estimated[0]
                estimated_yaw = math.degrees(math.atan2(edge[1], edge[0]))
                leave_yaw.append(
                    _angle_difference_degrees(
                        estimated_yaw, float(self.anchors[held_out]["yaw_degrees"])
                    )
                )

        anchor_yaw_uncertainties = [
            float(self.anchors[tag_id].get("yaw_uncertainty_degrees", 0.0))
            for tag_id in visible
        ]
        position_basis = max(
            [world_rms, *leave_position] if leave_position else [world_rms, 5.0]
        )
        yaw_basis = max(
            [*leave_yaw, *anchor_yaw_uncertainties]
            if leave_yaw or anchor_yaw_uncertainties
            else [2.0]
        )
        position_error_95 = max(5.0, 2.0 * position_basis)
        yaw_error_95 = max(2.0, 2.0 * yaw_basis)
        if len(visible) >= 3 and position_error_95 <= 30.0:
            quality = "good"
        elif len(visible) >= 2:
            quality = "provisional"
        else:
            quality = "weak"
        return CameraCalibration(
            camera_index=int(snapshot["index"]),
            homography=homography,
            anchor_ids=visible,
            reprojection_rms_px=reprojection_rms,
            world_rms_mm=world_rms,
            position_error_95_mm=position_error_95,
            yaw_error_95_degrees=yaw_error_95,
            leave_one_out_position_mm=leave_position,
            leave_one_out_yaw_degrees=leave_yaw,
            quality=quality,
        )

    def _camera_marker_estimates(
        self, snapshot: dict[str, Any], calibration: CameraCalibration
    ) -> list[dict[str, Any]]:
        inverse = np.linalg.inv(calibration.homography)
        estimates: list[dict[str, Any]] = []
        for tag_id, image_corners in snapshot["tags"].items():
            if tag_id in self.active_anchor_ids:
                continue
            center_image = np.asarray(image_corners, dtype=np.float64).mean(axis=0)
            center_world = _project(center_image.reshape(1, 2), inverse)[0]
            corners_world = _project(image_corners, inverse)
            edge = corners_world[1] - corners_world[0]
            yaw = math.degrees(math.atan2(edge[1], edge[0])) % 360.0
            image_edge_lengths = [
                float(
                    np.linalg.norm(
                        np.asarray(image_corners[(index + 1) % 4])
                        - np.asarray(image_corners[index])
                    )
                )
                for index in range(4)
            ]
            mean_edge_px = float(np.mean(image_edge_lengths))
            # Subpixel-refined corners are normally substantially better than
            # one pixel. Convert a conservative 0.35 px/corner 95% bound into
            # edge-heading error. Absolute homography heading error is kept
            # separately; paired tag headings share it and largely cancel.
            heading_error_95 = math.degrees(
                math.atan2(math.sqrt(2.0) * 0.35, max(mean_edge_px, 1.0))
            )
            estimates.append(
                {
                    "tag_id": int(tag_id),
                    "camera_index": int(snapshot["index"]),
                    "position_mm": {"x": float(center_world[0]), "y": float(center_world[1]), "z": None},
                    "rotation_degrees": {"roll": None, "pitch": None, "yaw": yaw},
                    "error_95_estimate": {
                        "position_mm": calibration.position_error_95_mm,
                        "yaw_degrees": calibration.yaw_error_95_degrees,
                    },
                    "frame_age_s": snapshot.get("frame_age_s"),
                    "mean_tag_edge_px": mean_edge_px,
                    "heading_error_95_degrees": heading_error_95,
                    "method": "camera ray projected onto the calibrated floor plane",
                }
            )
        return estimates

    def _round_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        result = dict(observation)
        result["position_mm"] = {
            axis: _rounded(value) for axis, value in observation["position_mm"].items()
        }
        result["rotation_degrees"] = {
            axis: _rounded(value) for axis, value in observation["rotation_degrees"].items()
        }
        result["error_95_estimate"] = {
            key: _rounded(value) for key, value in observation["error_95_estimate"].items()
        }
        result["frame_age_s"] = _rounded(observation.get("frame_age_s"))
        return result

    def _fuse_marker(self, observations: list[dict[str, Any]]) -> dict[str, Any]:
        positions = np.asarray(
            [[item["position_mm"]["x"], item["position_mm"]["y"]] for item in observations],
            dtype=np.float64,
        )
        errors = np.asarray(
            [item["error_95_estimate"]["position_mm"] for item in observations],
            dtype=np.float64,
        )
        weights = 1.0 / np.maximum(errors, 1e-6) ** 2
        position = np.average(positions, axis=0, weights=weights)
        disagreement = np.linalg.norm(positions - position, axis=1)
        position_error = max(
            float(np.sqrt(1.0 / weights.sum())),
            float(2.0 * disagreement.max(initial=0.0)),
        )
        yaws = [item["rotation_degrees"]["yaw"] for item in observations]
        yaw = _circular_mean_degrees(yaws)
        yaw_disagreement = max(
            (_angle_difference_degrees(value, yaw) for value in yaws), default=0.0
        )
        yaw_error = max(
            min(item["error_95_estimate"]["yaw_degrees"] for item in observations),
            2.0 * yaw_disagreement,
        )
        return {
            "tag_id": int(observations[0]["tag_id"]),
            "status": "tracked",
            "position_mm": {"x": _rounded(position[0]), "y": _rounded(position[1]), "z": None},
            "rotation_degrees": {"roll": None, "pitch": None, "yaw": _rounded(yaw)},
            "error_95_estimate": {
                "position_mm": _rounded(position_error),
                "yaw_degrees": _rounded(yaw_error),
            },
            "camera_indices": sorted(item["camera_index"] for item in observations),
            "observation_count": len(observations),
            "observations": [self._round_observation(item) for item in observations],
        }

    def _part_estimate(self, part: dict[str, Any], markers: dict[int, dict[str, Any]]) -> dict[str, Any]:
        configured_ids = [int(tag_id) for tag_id in part["tag_ids"]]
        visible = [markers[tag_id] for tag_id in configured_ids if tag_id in markers]
        common = {
            "part_id": part["id"],
            "display_name": part.get("display_name", part["id"]),
            "configured_tag_ids": configured_ids,
            "observed_tag_ids": [item["tag_id"] for item in visible],
            "surface": part.get("surface"),
        }
        if not visible:
            return {**common, "status": "not_visible", "pose": None}

        positions = np.asarray(
            [[item["position_mm"]["x"], item["position_mm"]["y"]] for item in visible]
        )
        position = positions.mean(axis=0)
        offsets = np.linalg.norm(positions - position, axis=1)
        position_error = max(
            [item["error_95_estimate"]["position_mm"] for item in visible]
            + [float(offsets.max(initial=0.0))]
        )
        yaw_period = float(part.get("yaw_period_degrees", 180.0))
        tag_yaws = [item["rotation_degrees"]["yaw"] for item in visible]
        yaw = _circular_mean_degrees(tag_yaws, yaw_period)
        yaw_offsets = [_angle_difference_degrees(value, yaw, yaw_period) for value in tag_yaws]
        yaw_error = max(
            [item["error_95_estimate"]["yaw_degrees"] for item in visible]
            + [2.0 * max(yaw_offsets, default=0.0)]
        )
        cameras = sorted({camera for marker in visible for camera in marker["camera_indices"]})
        vertical = part.get("surface") == "vertical"
        return {
            **common,
            "status": "projection_only" if vertical else "tracked",
            "pose": {
                "reference": (
                    "floor intersection of camera rays through visible tag centers"
                    if vertical
                    else part.get("pose_reference", "centroid of visible tag centers")
                ),
                "position_mm": {"x": _rounded(position[0]), "y": _rounded(position[1]), "z": None},
                "rotation_degrees": {
                    "roll": None,
                    "pitch": None,
                    "yaw_axis": None if vertical else _rounded(yaw),
                    "yaw_period_degrees": yaw_period,
                },
                "error_95_estimate": {
                    "position_mm": None if vertical else _rounded(position_error),
                    "yaw_degrees": None if vertical else _rounded(yaw_error),
                },
            },
            "camera_indices": cameras,
            "quality": "diagnostic_only" if vertical else "provisional",
            "projection_diagnostic": (
                {
                    "position_mm": {
                        "x": _rounded(position[0]),
                        "y": _rounded(position[1]),
                    },
                    "calibration_only_position_error_95_mm": _rounded(position_error),
                    "projected_tag_edge_yaw_degrees": _rounded(yaw),
                    "warning": (
                        "vertical elevated tags are not on the calibrated floor "
                        "plane; these values are not physical part positions or "
                        "joint angles"
                    ),
                }
                if vertical else None
            ),
        }

    @staticmethod
    def _frame_heading(marker: dict[str, Any], tag: dict[str, Any]) -> float:
        """Return the configured parent-frame +X heading in floor-world."""
        tag_heading = float(marker["rotation_degrees"]["yaw"])
        frame_from_tag_yaw = float(tag["frame_from_tag"]["euler_xyz_deg"][2])
        return _signed_angle_degrees(tag_heading - frame_from_tag_yaw)

    def _tag_rotation_candidates(
        self,
        image_corners: np.ndarray,
        camera_matrix: np.ndarray,
        distortion: np.ndarray,
    ) -> list[tuple[Rotation, float]]:
        marker_size_m = float(
            (self.robot_layout or {}).get("tag_geometry", {}).get(
                "black_square_m", self.tag_size_mm / 1000.0
            )
        )
        half = marker_size_m / 2.0
        object_points = np.asarray(
            [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
            dtype=np.float32,
        )
        solved = cv2.solvePnPGeneric(
            object_points,
            np.asarray(image_corners, dtype=np.float32),
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not solved[0]:
            return []
        candidates: list[tuple[Rotation, float]] = []
        for rvec, tvec in zip(solved[1], solved[2], strict=True):
            projected, _ = cv2.projectPoints(
                object_points, rvec, tvec, camera_matrix, distortion
            )
            error = projected.reshape(-1, 2) - np.asarray(image_corners).reshape(-1, 2)
            rms = math.sqrt(float(np.mean(np.sum(error * error, axis=1))))
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            candidates.append((Rotation.from_matrix(rotation_matrix), rms))
        return candidates

    def _floor_normal_camera(
        self,
        snapshot: dict[str, Any],
        camera_matrix: np.ndarray,
        distortion: np.ndarray,
    ) -> np.ndarray | None:
        visible = [tag_id for tag_id in self.active_anchor_ids if tag_id in snapshot["tags"]]
        if len(visible) < 2:
            return None
        world_xy = np.concatenate([self._anchor_corners(tag_id) for tag_id in visible])
        world = np.column_stack([world_xy, np.zeros(len(world_xy), dtype=np.float64)])
        image = np.concatenate([snapshot["tags"][tag_id] for tag_id in visible])
        solved, rvec, _tvec = cv2.solvePnP(
            world.astype(np.float32),
            image.astype(np.float32),
            camera_matrix,
            distortion,
        )
        if not solved:
            return None
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        return np.asarray(rotation_matrix[:, 2], dtype=np.float64)

    @staticmethod
    def _decompose_leg_rotation(rotation: Rotation) -> tuple[float, float, float]:
        matrix = rotation.as_matrix()
        yaw = math.atan2(float(-matrix[0, 1]), float(matrix[1, 1]))
        pitch = math.atan2(float(-matrix[2, 0]), float(matrix[2, 2]))
        fitted = Rotation.from_rotvec([0.0, 0.0, yaw]) * Rotation.from_rotvec(
            [0.0, pitch, 0.0]
        )
        residual = float((fitted.inv() * rotation).magnitude())
        return math.degrees(yaw), math.degrees(pitch), math.degrees(residual)

    def leg_zero_azimuth(self, leg: int) -> float:
        """Where leg ``leg`` points at the zero pose, degrees from body +x.

        Measured per leg by hexapod-calibrate-tags when the layout carries
        ``leg_zero_azimuth_body_deg``; otherwise the historical counter-clockwise
        assumption of (leg + 0.5) * 60.
        """
        return self.leg_zero_azimuth_deg.get(int(leg), (int(leg) + 0.5) * 60.0)

    def _camera_segment_joints(
        self,
        snapshots: list[dict[str, Any]],
        segment_tags: dict[int, dict[str, Any]],
        *,
        axis: str,
        segment: str,
    ) -> dict[str, dict[str, Any]]:
        """Recover an absolute leg-plane angle from a rigid link tag."""
        observations: dict[int, list[dict[str, Any]]] = {leg: [] for leg in range(6)}
        calibrated_cameras: list[int] = []
        body_visible_cameras: list[int] = []
        chassis_id = int(self.chassis_tag["id"]) if self.chassis_tag is not None else None

        for snapshot in snapshots:
            camera_index = int(snapshot["index"])
            intrinsic = self.intrinsic_calibrations.get(camera_index)
            if intrinsic is None or chassis_id is None:
                continue
            calibrated_cameras.append(camera_index)
            if chassis_id not in snapshot["tags"]:
                continue
            camera_matrix, distortion = intrinsic.for_image(
                int(snapshot["width"]), int(snapshot["height"])
            )
            floor_normal = self._floor_normal_camera(
                snapshot, camera_matrix, distortion
            )
            if floor_normal is None:
                continue
            body_candidates = self._tag_rotation_candidates(
                snapshot["tags"][chassis_id], camera_matrix, distortion
            )
            if not body_candidates:
                continue
            body_tag_rotation, body_rms = min(
                body_candidates,
                key=lambda item: (
                    -float(np.dot(item[0].as_matrix()[:, 2], floor_normal)),
                    item[1],
                ),
            )
            body_visible_cameras.append(camera_index)
            body_from_tag = _layout_rotation(self.chassis_tag["frame_from_tag"])
            camera_from_body = body_tag_rotation * body_from_tag.inv()

            for tag_id, tag in segment_tags.items():
                if tag_id not in snapshot["tags"]:
                    continue
                leg = int(tag["leg"])
                frame_from_tag = _layout_rotation(tag["frame_from_tag"])
                candidates: list[dict[str, Any]] = []
                for tag_rotation, reprojection_rms in self._tag_rotation_candidates(
                    snapshot["tags"][tag_id], camera_matrix, distortion
                ):
                    camera_from_segment = tag_rotation * frame_from_tag.inv()
                    body_from_segment = camera_from_body.inv() * camera_from_segment
                    leg_from_segment = Rotation.from_rotvec(
                        [0.0, 0.0, -math.radians(self.leg_zero_azimuth(leg))]
                    ) * body_from_segment
                    yaw, plane_angle, residual = self._decompose_leg_rotation(
                        leg_from_segment
                    )
                    candidates.append(
                        {
                            "yaw_deg": yaw,
                            "plane_angle_deg": plane_angle,
                            "kinematic_residual_deg": residual,
                            "reprojection_rms_px": reprojection_rms,
                        }
                    )
                if not candidates:
                    continue
                chosen = min(
                    candidates,
                    key=lambda item: (
                        item["kinematic_residual_deg"], item["reprojection_rms_px"]
                    ),
                )
                if (
                    chosen["kinematic_residual_deg"] > 10.0
                    or chosen["reprojection_rms_px"] > 3.0
                ):
                    continue
                observations[leg].append(
                    {
                        **chosen,
                        "tag_id": tag_id,
                        "camera_index": camera_index,
                        "body_tag_reprojection_rms_px": body_rms,
                        "calibration_quality": intrinsic.quality,
                    }
                )

        result: dict[str, dict[str, Any]] = {}
        for leg in range(6):
            name = f"L{leg}_{axis}"
            values = observations[leg]
            if not values:
                if not calibrated_cameras:
                    reason = "no intrinsic calibration is configured for an active camera"
                    status = "calibration_unavailable"
                elif not body_visible_cameras:
                    reason = "the chassis tag is not visible with floor anchors in a calibrated camera"
                    status = "not_visible"
                else:
                    reason = (
                        f"no accepted {segment} tag pose shares a calibrated "
                        "view with the chassis tag"
                    )
                    status = "not_visible"
                result[name] = {
                    "name": name,
                    "leg": leg,
                    "axis": axis,
                    "status": status,
                    "value_deg": None,
                    "error_95_estimate_deg": None,
                    "reason": reason,
                }
                continue

            weights = np.asarray(
                [
                    1.0
                    / max(
                        1.0,
                        item["kinematic_residual_deg"] ** 2
                        + item["reprojection_rms_px"] ** 2,
                    )
                    for item in values
                ],
                dtype=np.float64,
            )
            radians = np.radians([item["plane_angle_deg"] for item in values])
            vector = np.sum(weights * np.exp(1j * radians))
            angle = _signed_angle_degrees(
                math.degrees(math.atan2(vector.imag, vector.real))
            )
            disagreement = max(
                (
                    _angle_difference_degrees(item["plane_angle_deg"], angle)
                    for item in values
                ),
                default=0.0,
            )
            error = max(
                5.0,
                2.0 * disagreement,
                2.0 * max(item["kinematic_residual_deg"] for item in values),
            )
            result[name] = {
                "name": name,
                "leg": leg,
                "axis": axis,
                "status": "tracked",
                "value_deg": _rounded(angle),
                "error_95_estimate_deg": _rounded(error),
                "tag_ids": sorted({item["tag_id"] for item in values}),
                "camera_indices": sorted({item["camera_index"] for item in values}),
                "observation_count": len(values),
                "observations": [
                    {
                        key: _rounded(value) if isinstance(value, float) else value
                        for key, value in item.items()
                    }
                    for item in values
                ],
            }
        return result

    def _camera_pitch_joints(
        self, snapshots: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        joints = self._camera_segment_joints(
            snapshots, self.femur_tags, axis="hip", segment="femur"
        )
        joints.update(
            self._camera_segment_joints(
                snapshots, self.tibia_tags, axis="knee", segment="tibia"
            )
        )
        return joints

    def _camera_joint_pose(
        self,
        markers: dict[int, dict[str, Any]],
        snapshots: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Estimate robot-relative yaw from horizontal chassis and coxa tags.

        An image-to-floor homography cannot recover an elevated tag's metric
        position, but it does rectify the direction of a line parallel to the
        floor.  The chassis and coxa lid tags are both horizontal, so their
        decoded +X edge headings can be compared in the shared floor frame.
        """
        joints: dict[str, dict[str, Any]] = {}
        body_marker = (
            markers.get(int(self.chassis_tag["id"]))
            if self.chassis_tag is not None
            else None
        )
        body_heading = (
            self._frame_heading(body_marker, self.chassis_tag)
            if body_marker is not None and self.chassis_tag is not None
            else None
        )

        for leg in range(6):
            name = f"L{leg}_yaw"
            lid = self.yaw_lid_by_leg.get(leg)
            lid_marker = markers.get(int(lid["id"])) if lid is not None else None
            common = {
                "name": name,
                "leg": leg,
                "axis": "yaw",
                "chassis_tag_id": (
                    int(self.chassis_tag["id"]) if self.chassis_tag is not None else None
                ),
                "servo_lid_tag_id": int(lid["id"]) if lid is not None else None,
            }
            if self.chassis_tag is None or lid is None:
                joints[name] = {
                    **common,
                    "status": "layout_unavailable",
                    "value_deg": None,
                    "error_95_estimate_deg": None,
                    "reason": "horizontal chassis or coxa servo-lid tag is absent from the layout",
                }
                continue
            if body_marker is None:
                joints[name] = {
                    **common,
                    "status": "not_visible",
                    "value_deg": None,
                    "error_95_estimate_deg": None,
                    "reason": "the horizontal chassis tag is not visible in a calibrated camera",
                }
                continue
            if lid_marker is None:
                joints[name] = {
                    **common,
                    "status": "not_visible",
                    "value_deg": None,
                    "error_95_estimate_deg": None,
                    "reason": "this leg's horizontal coxa servo-lid tag is not visible",
                }
                continue

            body_by_camera = {
                int(item["camera_index"]): item
                for item in body_marker.get("observations", [])
            }
            lid_by_camera = {
                int(item["camera_index"]): item
                for item in lid_marker.get("observations", [])
            }
            common_cameras = sorted(set(body_by_camera) & set(lid_by_camera))
            if not common_cameras:
                joints[name] = {
                    **common,
                    "status": "not_visible",
                    "value_deg": None,
                    "error_95_estimate_deg": None,
                    "reason": "the chassis and coxa tags do not share a calibrated camera view",
                }
                continue

            zero_azimuth = self.leg_zero_azimuth(leg)
            observations = []
            for camera_index in common_cameras:
                body_observation = body_by_camera[camera_index]
                lid_observation = lid_by_camera[camera_index]
                observation_body_heading = self._frame_heading(
                    body_observation, self.chassis_tag
                )
                observation_coxa_heading = self._frame_heading(lid_observation, lid)
                # The layout says which way the servo's positive yaw turns in the
                # body frame; the tracker reports yaw in the robot's own sense.
                observation_value = _signed_angle_degrees(
                    self.yaw_sign
                    * (
                        observation_coxa_heading
                        - observation_body_heading
                        - zero_azimuth
                    )
                )
                corner_error = math.hypot(
                    float(body_observation.get("heading_error_95_degrees", 1.0)),
                    float(lid_observation.get("heading_error_95_degrees", 1.0)),
                )
                observations.append(
                    {
                        "camera_index": camera_index,
                        "value_deg": observation_value,
                        "corner_error_95_deg": corner_error,
                        "body_heading_world_deg": observation_body_heading,
                        "coxa_heading_world_deg": observation_coxa_heading,
                    }
                )

            weights = np.asarray(
                [1.0 / max(item["corner_error_95_deg"], 0.25) ** 2 for item in observations],
                dtype=np.float64,
            )
            radians = np.radians([item["value_deg"] for item in observations])
            vector = np.sum(weights * np.exp(1j * radians))
            value = _signed_angle_degrees(
                math.degrees(math.atan2(vector.imag, vector.real))
            )
            disagreement = max(
                (
                    _angle_difference_degrees(item["value_deg"], value)
                    for item in observations
                ),
                default=0.0,
            )
            error = max(
                RELATIVE_YAW_ERROR_95_FLOOR_DEGREES,
                math.sqrt(1.0 / float(weights.sum())),
                2.0 * disagreement,
            )
            coxa_heading = _circular_mean_degrees(
                [item["coxa_heading_world_deg"] for item in observations]
            )
            joints[name] = {
                **common,
                "status": "tracked",
                "value_deg": _rounded(value),
                "error_95_estimate_deg": _rounded(error),
                "body_heading_world_deg": _rounded(body_heading),
                "coxa_heading_world_deg": _rounded(coxa_heading),
                "leg_zero_azimuth_body_deg": _rounded(zero_azimuth),
                "yaw_sign_in_body_frame": self.yaw_sign,
                "camera_indices": common_cameras,
                "uncertainty_method": (
                    "paired same-camera relative heading; conservative corner "
                    "precision plus cross-camera disagreement"
                ),
                "observations": [
                    {
                        key: _rounded(item) if isinstance(item, float) else item
                        for key, item in observation.items()
                    }
                    for observation in observations
                ],
            }

        joints.update(self._camera_pitch_joints(snapshots))
        tracked_yaw_count = sum(
            item["status"] == "tracked" and item["axis"] == "yaw"
            for item in joints.values()
        )
        tracked_hip_count = sum(
            item["status"] == "tracked" and item["axis"] == "hip"
            for item in joints.values()
        )
        tracked_knee_count = sum(
            item["status"] == "tracked" and item["axis"] == "knee"
            for item in joints.values()
        )
        tracked_count = tracked_yaw_count + tracked_hip_count + tracked_knee_count
        return {
            "status": "tracking" if tracked_count else "unavailable",
            "tracked_joint_count": tracked_count,
            "tracked_yaw_count": tracked_yaw_count,
            "tracked_hip_count": tracked_hip_count,
            "tracked_knee_count": tracked_knee_count,
            "joint_frame": "robot_abs",
            "joint_contract": "robot_abs_tibia_v2",
            "method": {
                "yaw": "floor-homography-rectified horizontal tag headings",
                "hip": "intrinsic-calibrated AprilTag PnP relative body/femur rotation",
                "knee": "intrinsic-calibrated AprilTag PnP relative body/tibia rotation",
            },
            "body_heading_world_deg": _rounded(body_heading),
            "body_tag_id": (
                int(self.chassis_tag["id"]) if self.chassis_tag is not None else None
            ),
            "joints": joints,
            "limitations": [
                "yaw assumes the chassis and coxa lid tag faces are parallel to the floor",
                "intrinsic calibration is provisional and pitch uncertainty is not statistically validated",
                "knee is the absolute tibia angle in the leg plane, not femur-relative bend",
            ],
        }

    def estimate(self, snapshots: list[dict[str, Any]]) -> dict[str, Any]:
        calibration_json: dict[str, Any] = {}
        observations_by_tag: dict[int, list[dict[str, Any]]] = {}
        for snapshot in snapshots:
            calibration = self._calibrate_camera(snapshot)
            camera_key = str(snapshot["index"])
            if calibration is None:
                calibration_json[camera_key] = {
                    "status": "uncalibrated",
                    "quality": "unavailable",
                    "anchor_ids": [
                        tag_id for tag_id in self.active_anchor_ids if tag_id in snapshot["tags"]
                    ],
                    "reason": "no fixed floor anchor with a configured orientation is visible",
                    "frame_age_s": _rounded(snapshot.get("frame_age_s")),
                }
                continue
            calibration_json[camera_key] = calibration.as_json(
                (int(snapshot["width"]), int(snapshot["height"])), snapshot.get("frame_age_s")
            )
            for observation in self._camera_marker_estimates(snapshot, calibration):
                observations_by_tag.setdefault(observation["tag_id"], []).append(observation)

        markers = {
            tag_id: self._fuse_marker(observations)
            for tag_id, observations in observations_by_tag.items()
        }
        parts = {
            part["id"]: self._part_estimate(part, markers)
            for part in self.part_map.get("parts", [])
        }
        assigned = {
            int(tag_id)
            for part in self.part_map.get("parts", [])
            for tag_id in part["tag_ids"]
        }
        return {
            "schema_version": 1,
            "generated_at_unix_s": round(time.time(), 6),
            "world_frame": self.floor_map.get("coordinate_frame", {}),
            "pose_model": {
                "name": "planar_ground_projection_v1",
                "observable_degrees_of_freedom": [
                    "x, y and yaw_axis only for markers on the floor plane",
                    "robot-relative leg yaw from paired horizontal chassis and coxa tags",
                    "robot-relative hip from intrinsic-calibrated body and femur tag orientations",
                    "robot-relative absolute tibia/knee angle from intrinsic-calibrated body and tibia tag orientations",
                ],
                "unobservable_degrees_of_freedom": [
                    "metric robot-tag translation",
                    "knee when no accepted tibia-frame tag shares a view with the chassis tag",
                ],
                "position_units": "millimeters",
                "rotation_units": "degrees",
                "warning": (
                    "Vertical tag centers are camera rays projected onto the floor. "
                    "Their part outputs are diagnostic projections, not metric 3-D "
                    "tag centers, part poses, or joint angles."
                ),
            },
            "uncertainty": {
                "field": "error_95_estimate",
                "method": (
                    "conservative bound from leave-one-anchor-out calibration and "
                    "multi-camera disagreement"
                ),
                "statistically_validated": False,
            },
            "calibration": {
                "cameras": calibration_json,
                "intrinsics": {
                    "name": (self.camera_calibration or {}).get("name"),
                    "quality": (self.camera_calibration or {}).get("quality", "unavailable"),
                    "camera_indices": sorted(self.intrinsic_calibrations),
                    "method": (self.camera_calibration or {}).get("method"),
                    "cameras": {
                        str(index): {
                            "quality": calibration.quality,
                            "image_size": {
                                "width": calibration.source_size[0],
                                "height": calibration.source_size[1],
                            },
                            "floor_reprojection_rms_px": _rounded(
                                calibration.floor_reprojection_rms_px
                            ),
                        }
                        for index, calibration in sorted(
                            self.intrinsic_calibrations.items()
                        )
                    },
                },
            },
            "camera_joint_pose": self._camera_joint_pose(markers, snapshots),
            "parts": parts,
            "markers": {str(tag_id): marker for tag_id, marker in sorted(markers.items())},
            "unassigned_tag_ids": sorted(tag_id for tag_id in markers if tag_id not in assigned),
        }
