"""Best-effort planar pose fusion for the local AprilTag camera service.

The installation has fixed tags on the floor but not yet a validated intrinsic
calibration for every camera. This module reports the observable floor-plane
degrees of freedom (x, y and a yaw axis) and explicitly leaves z, roll and
pitch unset. A future metric 6-DoF estimator can fill those fields without
breaking clients of the JSON schema.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


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

    def __init__(self, floor_map: dict[str, Any], part_map: dict[str, Any]):
        self.floor_map = floor_map
        self.part_map = part_map
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

    def _anchor_corners(self, tag_id: int) -> np.ndarray:
        anchor = self.anchors[tag_id]
        center = np.asarray(anchor["center"][:2], dtype=np.float64)
        half = self.tag_size_mm / 2.0
        offsets = np.asarray(
            [[-half, -half], [half, -half], [half, half], [-half, half]],
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
        return {
            **common,
            "status": "tracked",
            "pose": {
                "reference": part.get("pose_reference", "centroid of visible tag centers"),
                "position_mm": {"x": _rounded(position[0]), "y": _rounded(position[1]), "z": None},
                "rotation_degrees": {
                    "roll": None,
                    "pitch": None,
                    "yaw_axis": _rounded(yaw),
                    "yaw_period_degrees": yaw_period,
                },
                "error_95_estimate": {
                    "position_mm": _rounded(position_error),
                    "yaw_degrees": _rounded(yaw_error),
                },
            },
            "camera_indices": cameras,
            "quality": "provisional",
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
                "observable_degrees_of_freedom": ["x", "y", "yaw_axis"],
                "unobservable_degrees_of_freedom": ["z", "roll", "pitch"],
                "position_units": "millimeters",
                "rotation_units": "degrees",
                "warning": (
                    "Vertical tag centers are camera rays projected onto the floor. "
                    "They are not yet metric 3-D tag or joint centers."
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
            "calibration": {"cameras": calibration_json},
            "parts": parts,
            "markers": {str(tag_id): marker for tag_id, marker in sorted(markers.items())},
            "unassigned_tag_ids": sorted(tag_id for tag_id in markers if tag_id not in assigned),
        }
