"""Handheld AprilTag surveying in a board-aligned world frame.

Record3D supplies an OpenGL/ARKit camera trajectory.  A mapped AprilTag board
ties that arbitrary trajectory to the tracker's metric world frame.  This
module then aggregates tag poses from a slow walk around a stationary robot.
It contains no capture or robot I/O.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

from .apriltag_vision import TagCorners, estimate_tag_pose, marker_object_corners
from .housing_pose import (
    JOINT_NAMES,
    HexapodGeometry,
    RigidTransform,
    forward_frame_transforms,
)


OPENGL_CAMERA_FROM_OPENCV_CAMERA = RigidTransform(
    np.zeros(3), Rotation.from_euler("x", 180.0, degrees=True)
)


@dataclass(frozen=True)
class TagSurveyOptions:
    min_observations: int = 5
    max_reprojection_rms_px: float = 2.5
    max_translation_spread_m: float = 0.020
    max_rotation_spread_deg: float = 5.0
    ground_height_tolerance_m: float = 0.035
    ground_normal_tolerance_deg: float = 25.0
    max_observations_per_tag: int = 120
    min_viewpoint_span_deg: float = 0.0
    freeze_stable_tags: bool = False
    robot_slot_match_tolerance_m: float = 0.060

    def __post_init__(self) -> None:
        if self.min_observations < 2:
            raise ValueError("min_observations must be at least two")
        if self.max_observations_per_tag < self.min_observations:
            raise ValueError(
                "max_observations_per_tag cannot be below min_observations"
            )
        for name in (
            "max_reprojection_rms_px",
            "max_translation_spread_m",
            "max_rotation_spread_deg",
            "ground_height_tolerance_m",
            "ground_normal_tolerance_deg",
            "robot_slot_match_tolerance_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a positive finite number")
        if (
            not math.isfinite(float(self.min_viewpoint_span_deg))
            or self.min_viewpoint_span_deg < 0.0
            or self.min_viewpoint_span_deg >= 180.0
        ):
            raise ValueError(
                "min_viewpoint_span_deg must be finite and between 0 and 180"
            )


@dataclass(frozen=True)
class TagSurveyObservation:
    tag_id: int
    world_from_tag: RigidTransform
    marker_size_m: float
    reprojection_rms_px: float
    image_heading_deg: float
    confidence: float
    camera_position_world: np.ndarray


@dataclass(frozen=True)
class TransformConsensus:
    transform: RigidTransform
    input_count: int
    used_count: int
    translation_spread_mm: float
    rotation_spread_deg: float
    stable: bool
    ambiguous_cluster: bool
    mean_reprojection_rms_px: float | None = None


def arkit_world_from_opencv_camera(
    arkit_world_from_opengl_camera: RigidTransform,
) -> RigidTransform:
    """Convert Record3D's OpenGL camera basis to OpenCV camera coordinates."""
    return arkit_world_from_opengl_camera.compose(
        OPENGL_CAMERA_FROM_OPENCV_CAMERA
    )


def _rotation_errors_deg(center: Rotation, rotations: Rotation) -> np.ndarray:
    return np.degrees((center.inv() * rotations).magnitude())


def _transform_consensus(
    transforms: Sequence[RigidTransform],
    *,
    weights: np.ndarray | None = None,
    min_observations: int,
    max_translation_spread_m: float,
    max_rotation_spread_deg: float,
    reprojection_errors: Sequence[float] | None = None,
) -> TransformConsensus:
    if not transforms:
        raise ValueError("cannot average an empty transform sequence")
    translations = np.stack([item.translation_m for item in transforms])
    rotations = Rotation.from_quat(np.stack([
        item.rotation.as_quat() for item in transforms
    ]))
    # Use an observed medoid to avoid a quaternion mean landing between the two
    # planar IPPE branches before outlier rejection. Compute all pairwise
    # distances in NumPy: the former Python/SciPy loop starved Record3D's
    # callback thread once dozens of tags each had a full observation window.
    translation_distances = np.linalg.norm(
        translations[:, None, :] - translations[None, :, :], axis=2
    )
    quaternions = rotations.as_quat()
    quaternion_dots = np.clip(
        np.abs(quaternions @ quaternions.T), 0.0, 1.0
    )
    rotation_distances_deg = np.degrees(2.0 * np.arccos(quaternion_dots))
    scores = np.median(
        translation_distances / max_translation_spread_m
        + rotation_distances_deg / max_rotation_spread_deg,
        axis=1,
    )
    medoid_index = int(np.argmin(scores))
    medoid = transforms[medoid_index]
    keep = (
        translation_distances[medoid_index] <= max_translation_spread_m
    ) & (
        rotation_distances_deg[medoid_index] <= max_rotation_spread_deg
    )
    indexes = np.flatnonzero(keep)
    if not len(indexes):
        indexes = np.asarray([int(np.argmin(scores))])
    rejected_indexes = np.flatnonzero(~keep)
    ambiguous_cluster = False
    if len(rejected_indexes) >= min_observations:
        rejected_translation_distances = translation_distances[
            np.ix_(rejected_indexes, rejected_indexes)
        ]
        rejected_rotation_distances = rotation_distances_deg[
            np.ix_(rejected_indexes, rejected_indexes)
        ]
        compatible = (
            rejected_translation_distances <= max_translation_spread_m
        ) & (
            rejected_rotation_distances <= max_rotation_spread_deg
        )
        ambiguous_cluster = bool(np.any(
            np.count_nonzero(compatible, axis=1) >= min_observations
        ))
    kept_translations = translations[indexes]
    kept_rotations = Rotation.from_quat(rotations.as_quat()[indexes])
    kept_weights = (
        np.ones(len(indexes), dtype=float)
        if weights is None
        else np.asarray(weights, dtype=float)[indexes]
    )
    translation = np.average(
        kept_translations, axis=0, weights=kept_weights
    )
    rotation = kept_rotations.mean(weights=kept_weights)
    translation_errors = np.linalg.norm(
        kept_translations - translation, axis=1
    )
    rotation_errors = _rotation_errors_deg(rotation, kept_rotations)
    translation_spread = math.sqrt(float(np.average(
        translation_errors ** 2, weights=kept_weights
    )))
    rotation_spread = math.sqrt(float(np.average(
        rotation_errors ** 2, weights=kept_weights
    )))
    stable = (
        len(indexes) >= min_observations
        and translation_spread <= max_translation_spread_m
        and rotation_spread <= max_rotation_spread_deg
        and not ambiguous_cluster
    )
    reprojection = None
    if reprojection_errors is not None:
        reprojection = float(np.average(
            np.asarray(reprojection_errors, dtype=float)[indexes],
            weights=kept_weights,
        ))
    return TransformConsensus(
        transform=RigidTransform(translation, rotation),
        input_count=len(transforms),
        used_count=len(indexes),
        translation_spread_mm=translation_spread * 1000.0,
        rotation_spread_deg=rotation_spread,
        stable=stable,
        ambiguous_cluster=ambiguous_cluster,
        mean_reprojection_rms_px=reprojection,
    )


class HandheldWorldAlignment:
    """Estimate ``world_from_arkit_world`` from repeated board sightings.

    The rolling consensus is useful for diagnosing ARKit drift, but it must
    not also be the switch that turns surveying on and off.  Once a clean
    floor lock has been established, retain that transform as the trajectory
    fallback.  Later floor viewpoints can legitimately form a second cluster
    because tiny planar tags and LiDAR plane fits have view-dependent range
    bias.  Losing every subsequent tag observation in that situation is much
    worse than temporarily following the last good ARKit alignment.
    """

    def __init__(
        self,
        *,
        min_observations: int = 8,
        max_translation_spread_m: float = 0.025,
        max_rotation_spread_deg: float = 2.5,
        max_observations: int = 48,
    ) -> None:
        self.min_observations = min_observations
        self.max_translation_spread_m = max_translation_spread_m
        self.max_rotation_spread_deg = max_rotation_spread_deg
        if max_observations < min_observations:
            raise ValueError("max_observations cannot be below min_observations")
        self.max_observations = int(max_observations)
        self._candidates: list[RigidTransform] = []
        self._locked_transform: RigidTransform | None = None

    def add(
        self,
        world_from_opencv_camera: RigidTransform,
        arkit_world_from_opengl_camera_pose: RigidTransform,
    ) -> None:
        arkit_from_cv = arkit_world_from_opencv_camera(
            arkit_world_from_opengl_camera_pose
        )
        self._candidates.append(
            world_from_opencv_camera.compose(arkit_from_cv.inverse())
        )
        if len(self._candidates) > self.max_observations:
            del self._candidates[:-self.max_observations]
        if self._locked_transform is None:
            consensus = self.consensus()
            if consensus is not None and consensus.stable:
                self._locked_transform = consensus.transform

    @property
    def observation_count(self) -> int:
        return len(self._candidates)

    @property
    def has_lock(self) -> bool:
        """Whether a clean board lock has ever been established."""
        return self._locked_transform is not None

    @property
    def locked_transform(self) -> RigidTransform | None:
        """The persistent trajectory alignment used after initial lock."""
        return self._locked_transform

    def consensus(self) -> TransformConsensus | None:
        if not self._candidates:
            return None
        return _transform_consensus(
            self._candidates,
            min_observations=self.min_observations,
            max_translation_spread_m=self.max_translation_spread_m,
            max_rotation_spread_deg=self.max_rotation_spread_deg,
        )

    def world_from_camera(
        self,
        arkit_world_from_opengl_camera_pose: RigidTransform,
    ) -> RigidTransform:
        if self._locked_transform is None:
            raise ValueError("ARKit trajectory is not aligned to the board yet")
        return self._locked_transform.compose(arkit_world_from_opencv_camera(
            arkit_world_from_opengl_camera_pose
        ))


def _normal_error_from_up_deg(transform: RigidTransform) -> float:
    normal = transform.rotation.apply([0.0, 0.0, 1.0])
    dot = float(np.clip(np.dot(normal, [0.0, 0.0, 1.0]), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def _euler_xyz_deg(rotation: Rotation) -> list[float]:
    return [
        round(float(value), 6)
        for value in rotation.as_euler("xyz", degrees=True)
    ]


def _layout_position(spec: Mapping[str, Any], tag_id: int) -> str:
    """Give every physical mount a stable, operator-facing position name."""
    frame = str(spec.get("frame", ""))
    if frame == "body":
        return "chassis top"
    leg = spec.get("leg")
    joint = str(spec.get("joint", "mount"))
    kind = str(spec.get("kind", ""))
    if leg is not None and kind == "servo_lid":
        return f"L{int(leg)} {joint} top"
    if leg is not None and kind == "yoke_face":
        return f"L{int(leg)} {joint} {spec.get('mount_side', 'side')} side"
    return str(spec.get("label", frame or f"tag {tag_id}"))


def merge_robot_layout_into_config(
    tracker_config: Mapping[str, Any],
    robot_layout: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge the photographed 37-tag inventory into a tracker configuration.

    The layout is identity/orientation evidence.  Its null translations are not
    measurements, so new tags receive a zero placeholder until this survey
    writes a measured ``frame_from_tag`` transform.
    """
    updated = copy.deepcopy(dict(tracker_config))
    robot_pose = updated.setdefault("robot_pose", {})
    tags = robot_pose.setdefault("tags", {})
    marker_size_m = float(
        robot_layout.get("tag_geometry", {}).get(
            "black_square_m", updated.get("marker_size_m", 0.027)
        )
    )
    for raw in robot_layout.get("robot_tags", []):
        if not isinstance(raw, Mapping):
            continue
        tag_id = int(raw["id"])
        key = str(tag_id)
        existing = copy.deepcopy(tags.get(key, {}))
        kind = str(raw.get("kind", existing.get("kind", "")))
        surface = str(raw.get(
            "surface",
            "vertical" if kind == "yoke_face" else "horizontal",
        ))
        for field in ("kind", "leg", "joint", "frame", "mount_side"):
            if field in raw:
                existing[field] = copy.deepcopy(raw[field])
        existing["surface"] = surface
        existing["position"] = _layout_position(raw, tag_id)
        existing.setdefault("label", existing["position"])
        existing.setdefault("marker_size_m", marker_size_m)
        if "frame_from_tag" not in existing:
            transform = copy.deepcopy(raw.get("frame_from_tag", {}))
            if transform.get("translation_m") is None:
                transform.pop("translation_m", None)
                existing["mount_translation_placeholder"] = True
            existing["frame_from_tag"] = transform
            existing["mount_source"] = "photographed_layout_identity"
        tags[key] = existing
    robot_pose["tag_inventory"] = {
        "source": str(robot_layout.get("name", "robot AprilTag layout")),
        "expected_total": len(robot_layout.get("robot_tags", [])),
        "expected_vertical_angle_tags": sum(
            str(item.get("kind")) == "yoke_face"
            for item in robot_layout.get("robot_tags", [])
            if isinstance(item, Mapping)
        ),
    }
    return updated


class TagSurveyAccumulator:
    """Aggregate all tag IDs and orientations during a stationary-scene walk."""

    def __init__(
        self,
        *,
        robot_tags: Mapping[int, Mapping[str, Any]],
        expected_ground_ids: Sequence[int] = (),
        anchor_ids: Sequence[int] = (),
        marker_size_m: float,
        marker_sizes_m: Mapping[int, float] | None = None,
        position_tag_overrides: Mapping[str, int] | None = None,
        geometry: HexapodGeometry | Mapping[str, Any] | None = None,
        body_anchor_tag_id: int | None = None,
        reference_floor_tags: Mapping[int, RigidTransform] | None = None,
        options: TagSurveyOptions | None = None,
    ) -> None:
        marker_object_corners(marker_size_m)
        self.robot_tags = {int(key): dict(value) for key, value in robot_tags.items()}
        self.expected_robot_ids = set(self.robot_tags)
        self.expected_ground_ids = {int(value) for value in expected_ground_ids}
        self.anchor_ids = {int(value) for value in anchor_ids}
        self.marker_size_m = float(marker_size_m)
        self.marker_sizes_m = {
            int(key): float(value)
            for key, value in (marker_sizes_m or {}).items()
        }
        self.position_tag_overrides = {
            str(frame): int(tag_id)
            for frame, tag_id in (position_tag_overrides or {}).items()
        }
        self.geometry = (
            geometry if isinstance(geometry, HexapodGeometry)
            else HexapodGeometry.from_dict(geometry)
        )
        self.body_anchor_tag_id = (
            None if body_anchor_tag_id is None else int(body_anchor_tag_id)
        )
        self.reference_floor_tags = {
            int(tag_id): transform
            for tag_id, transform in (reference_floor_tags or {}).items()
        }
        for size in self.marker_sizes_m.values():
            marker_object_corners(size)
        self.options = options or TagSurveyOptions()
        self._observations: dict[int, list[TagSurveyObservation]] = {}
        self._frozen: dict[int, TransformConsensus] = {}
        self._restored_viewpoint_spans: dict[int, float] = {}
        self._replacement_specs: dict[int, dict[str, Any]] = {}
        self._multi_floor_reference_frames: list[
            tuple[tuple[int, ...], float, float]
        ] = []
        self.frames = 0

    def observe_floor_reference(
        self,
        tag_ids: Sequence[int],
        *,
        reprojection_rms_px: float,
        depth_plane_rms_mm: float,
    ) -> None:
        """Record a joint floor-grid check that cannot self-validate one tag."""
        visible = tuple(sorted({int(tag_id) for tag_id in tag_ids}))
        if len(visible) < 2:
            return
        self._multi_floor_reference_frames.append((
            visible,
            float(reprojection_rms_px),
            float(depth_plane_rms_mm),
        ))
        if len(self._multi_floor_reference_frames) > 180:
            del self._multi_floor_reference_frames[0]

    def restore_stable_records(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        frames: int = 0,
    ) -> list[int]:
        """Restore accepted landmarks and trustworthy same-angle pose seeds.

        A reconnected Record3D session has a new ARKit origin, so camera
        alignment must be acquired again. Stable tag transforms are already in
        the survey board's world frame and can safely remain frozen. A record
        that failed *only* the viewpoint-diversity gate is retained as one
        provisional observation, so reconnecting asks for a fresh side view
        without throwing away its identity and approximate pose. Ambiguous or
        otherwise noisy records remain excluded.
        """
        restored: list[int] = []
        for raw in records:
            stable = bool(raw.get("stable"))
            recoverable_viewpoint_seed = (
                raw.get("viewpoint_requirement_met") is False
                and not raw.get("possible_duplicate_id_or_tracking_jump", False)
            )
            if not stable and not recoverable_viewpoint_seed:
                continue
            try:
                tag_id = int(raw["tag_id"])
                transform = RigidTransform.from_dict(raw["world_from_tag"])
                marker_size_m = float(raw.get(
                    "marker_size_m",
                    self.marker_sizes_m.get(tag_id, self.marker_size_m),
                ))
                input_count = max(1, int(raw.get("observations", 1)))
                used_count = max(1, int(raw.get("used_observations", input_count)))
                translation_spread_mm = float(raw.get("translation_spread_mm", 0.0))
                rotation_spread_deg = float(raw.get("rotation_spread_deg", 0.0))
                mean_reprojection_rms_px = float(
                    raw.get("mean_reprojection_rms_px", 0.0)
                )
            except (KeyError, TypeError, ValueError):
                continue
            if tag_id < 0 or not math.isfinite(marker_size_m) or marker_size_m <= 0.0:
                continue
            if recoverable_viewpoint_seed and (
                used_count < self.options.min_observations
                or translation_spread_mm
                > self.options.max_translation_spread_m * 1000.0
                or rotation_spread_deg > self.options.max_rotation_spread_deg
                or mean_reprojection_rms_px
                > self.options.max_reprojection_rms_px
            ):
                continue
            self._observations[tag_id] = [TagSurveyObservation(
                tag_id=tag_id,
                world_from_tag=transform,
                marker_size_m=marker_size_m,
                reprojection_rms_px=max(0.0, mean_reprojection_rms_px),
                image_heading_deg=0.0,
                confidence=1.0,
                camera_position_world=transform.translation_m.copy(),
            )]
            if stable:
                self._frozen[tag_id] = TransformConsensus(
                    transform=transform,
                    input_count=input_count,
                    used_count=min(input_count, used_count),
                    translation_spread_mm=max(0.0, translation_spread_mm),
                    rotation_spread_deg=max(0.0, rotation_spread_deg),
                    stable=True,
                    ambiguous_cluster=False,
                    mean_reprojection_rms_px=max(
                        0.0, mean_reprojection_rms_px
                    ),
                )
                self._restored_viewpoint_spans[tag_id] = max(
                    float(raw.get(
                        "viewpoint_span_deg",
                        self.options.min_viewpoint_span_deg,
                    )),
                    self.options.min_viewpoint_span_deg,
                )
            restored.append(tag_id)
        self.frames = max(self.frames, max(0, int(frames)))
        return sorted(restored)

    @staticmethod
    def _friendly_robot_position(frame: str, label: str) -> str:
        if frame == "body":
            return "chassis"
        if frame.startswith("L") and "_" in frame:
            leg, component = frame.split("_", 1)
            component_name = {
                "coxa": "hip",
                "femur": "knee",
                "tibia": "foot",
            }.get(component, component)
            return f"{leg} {component_name}"
        return label or frame

    def _robot_slots(self) -> list[dict[str, Any]]:
        slots: list[dict[str, Any]] = []
        position_counts: dict[str, int] = {}
        for tag_id, spec in sorted(self.robot_tags.items()):
            frame = str(spec.get("frame", "unassigned"))
            label = str(spec.get("label", f"tag {tag_id}"))
            base_position = str(spec.get(
                "position",
                self._friendly_robot_position(frame, label),
            ))
            position_counts[base_position] = position_counts.get(base_position, 0) + 1
            position = base_position
            if position_counts[base_position] > 1:
                position = f"{base_position} ({label})"
            photo_center = spec.get("photo_center_px")
            slots.append({
                "position": position,
                "frame": frame,
                "configured_tag_id": tag_id,
                "declared_tag_id": self.position_tag_overrides.get(frame, tag_id),
                "identity_reference": frame in self.position_tag_overrides,
                "label": label,
                "photo_center_px": photo_center,
                "kind": spec.get("kind"),
                "surface": spec.get("surface"),
                "leg": spec.get("leg"),
                "joint": spec.get("joint"),
                "mount_side": spec.get("mount_side"),
            })
        return slots

    def _photo_layout_to_world(
        self,
        slots: Sequence[Mapping[str, Any]],
        records: Mapping[int, Mapping[str, Any]],
    ) -> np.ndarray | None:
        """Fit the old calibration photo layout to the current surveyed robot.

        The photo centers describe physical mount positions independently of tag
        orientation.  Recognized tags provide the control points; the fit then
        predicts where a replacement ID should be found.
        """
        controls: list[tuple[np.ndarray, np.ndarray]] = []
        for slot in slots:
            photo_center = slot.get("photo_center_px")
            record = records.get(int(slot["declared_tag_id"]))
            if photo_center is None or record is None:
                continue
            if int(record.get("used_observations", 0)) < self.options.min_observations:
                continue
            controls.append((
                np.asarray([*photo_center, 1.0], dtype=float),
                np.asarray(record["world_from_tag"]["translation_m"], dtype=float),
            ))
        if len(controls) < 3:
            return None
        photo_points = np.stack([item[0] for item in controls])
        world_points = np.stack([item[1] for item in controls])
        if np.linalg.matrix_rank(photo_points) < 3:
            return None
        keep = np.ones(len(controls), dtype=bool)
        transform = np.linalg.lstsq(photo_points, world_points, rcond=None)[0]
        for _ in range(3):
            residuals = np.linalg.norm(photo_points @ transform - world_points, axis=1)
            median = float(np.median(residuals[keep]))
            candidate_keep = residuals <= max(0.025, 2.5 * median)
            if int(np.count_nonzero(candidate_keep)) < 3:
                break
            keep = candidate_keep
            transform = np.linalg.lstsq(
                photo_points[keep], world_points[keep], rcond=None
            )[0]
        return transform

    def _world_robot_frames_from_records(
        self,
        records: Mapping[int, Mapping[str, Any]],
    ) -> dict[str, RigidTransform] | None:
        orientation_anchor = self._orientation_anchor_spec()
        if orientation_anchor is not None:
            tag_id, spec = orientation_anchor
            record = records.get(tag_id)
            if record is not None and record.get("stable"):
                return self._world_robot_frames_from_anchor(
                    RigidTransform.from_dict(record["world_from_tag"]),
                    spec,
                )
        body_ids = [
            tag_id for tag_id, spec in self.robot_tags.items()
            if str(spec.get("frame")) == "body"
        ]
        if self.body_anchor_tag_id is not None:
            body_ids.sort(key=lambda tag_id: tag_id != self.body_anchor_tag_id)
        for tag_id in body_ids:
            record = records.get(tag_id)
            if record is None or not record.get("stable"):
                continue
            world_from_tag = RigidTransform.from_dict(record["world_from_tag"])
            return self._world_robot_frames_from_anchor(
                world_from_tag,
                self.robot_tags[tag_id],
            )
        return None

    def _orientation_anchor_spec(self) -> tuple[int, dict[str, Any]] | None:
        """Return the unchanged L0 tag used to orient the BuildViz skeleton."""
        tag_id = self.position_tag_overrides.get("L0_coxa")
        if tag_id is None:
            return None
        spec = self.robot_tags.get(tag_id)
        if spec is None:
            spec = next((
                item for item in self.robot_tags.values()
                if str(item.get("frame")) == "L0_coxa"
            ), None)
        if spec is None:
            return None
        return tag_id, spec

    def _world_robot_frames_from_anchor(
        self,
        world_from_tag: RigidTransform,
        spec: Mapping[str, Any],
    ) -> dict[str, RigidTransform]:
        body_zero_frames = forward_frame_transforms(
            RigidTransform(np.zeros(3), Rotation.identity()),
            {name: 0.0 for name in JOINT_NAMES},
            geometry=self.geometry,
        )
        frame = str(spec.get("frame", "body"))
        body_from_frame = body_zero_frames[frame]
        frame_from_tag = RigidTransform.from_dict(spec.get("frame_from_tag", {}))
        world_from_body = world_from_tag.compose(
            body_from_frame.compose(frame_from_tag).inverse()
        )
        return forward_frame_transforms(
            world_from_body,
            {name: 0.0 for name in JOINT_NAMES},
            geometry=self.geometry,
        )

    def _robot_position_records(
        self,
        records: Mapping[int, Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        slots = self._robot_slots()
        layout_to_world = self._photo_layout_to_world(slots, records)
        positions: list[dict[str, Any]] = []
        for slot in slots:
            configured_id = int(slot["configured_tag_id"])
            declared_id = int(slot["declared_tag_id"])
            record = records.get(declared_id)
            if record is None:
                state = "not_seen"
                tag_id = None
            else:
                state = (
                    "measured" if bool(record.get("stable"))
                    else "seen_needs_another_view"
                )
                tag_id = declared_id
            expected_world = None
            if layout_to_world is not None and slot.get("photo_center_px") is not None:
                expected_world = (
                    np.asarray([*slot["photo_center_px"], 1.0], dtype=float)
                    @ layout_to_world
                )
            positions.append({
                "position": slot["position"],
                "frame": slot["frame"],
                "configured_tag_id": configured_id,
                "declared_tag_id": declared_id,
                "tag_id": tag_id,
                "replacement": tag_id is not None and tag_id != configured_id,
                "identity_reference": bool(slot["identity_reference"]),
                "kind": slot.get("kind"),
                "surface": slot.get("surface"),
                "leg": slot.get("leg"),
                "joint": slot.get("joint"),
                "mount_side": slot.get("mount_side"),
                "state": state,
                "expected_world_position_m": (
                    None if expected_world is None
                    else [round(float(value), 7) for value in expected_world]
                ),
                "observations": (
                    0 if record is None else int(record["observations"])
                ),
                "used_observations": (
                    0 if record is None else int(record["used_observations"])
                ),
                "viewpoint_span_deg": (
                    0.0 if record is None
                    else float(record.get("viewpoint_span_deg", 0.0))
                ),
                "required_viewpoint_span_deg": (
                    self.options.min_viewpoint_span_deg
                    if record is None
                    else float(record.get(
                        "required_viewpoint_span_deg",
                        self.options.min_viewpoint_span_deg,
                    ))
                ),
            })

        # The photographed layout has no metric side-tag translations.  A
        # missing/replaced yoke tag can still be localized to its physical
        # mount from the other tags surrounding the same hip or knee.
        for item in positions:
            if (
                item["state"] != "not_seen"
                or item.get("expected_world_position_m") is not None
                or item.get("kind") != "yoke_face"
            ):
                continue
            sibling_points = []
            for sibling in positions:
                if (
                    sibling is item
                    or sibling.get("leg") != item.get("leg")
                    or sibling.get("joint") != item.get("joint")
                    or sibling.get("tag_id") is None
                ):
                    continue
                sibling_record = records.get(int(sibling["tag_id"]))
                if sibling_record is not None and sibling_record.get("stable"):
                    sibling_points.append(np.asarray(
                        sibling_record["world_from_tag"]["translation_m"],
                        dtype=float,
                    ))
            if sibling_points:
                item["expected_world_position_m"] = [
                    round(float(value), 7)
                    for value in np.mean(np.stack(sibling_points), axis=0)
                ]

        # Only substitute a new ID when the configured ID was never seen.  A
        # noisy configured tag should ask for another view, not be silently
        # replaced by a nearby marker on the same link.
        empty_indexes = [
            index for index, item in enumerate(positions)
            if item["state"] == "not_seen"
            and item["expected_world_position_m"] is not None
            and not item["identity_reference"]
        ]
        configured_ids = set(self.robot_tags) | set(self.position_tag_overrides.values())
        replacement_candidates = [
            item for tag_id, item in records.items()
            if tag_id not in configured_ids
            and tag_id not in self.expected_ground_ids
            and tag_id not in self.anchor_ids
            and (
                item.get("role") == "unassigned"
                or tag_id in self._replacement_specs
            )
            and item.get("stable")
        ]
        if empty_indexes and replacement_candidates:
            world_frames = self._world_robot_frames_from_records(records)

            def match_metrics(
                position: Mapping[str, Any], candidate: Mapping[str, Any]
            ) -> tuple[float, float | None, float]:
                distance = float(np.linalg.norm(
                    np.asarray(position["expected_world_position_m"])
                    - np.asarray(candidate["world_from_tag"]["translation_m"])
                ))
                expected_normal = self._robot_normal_world(
                    self.robot_tags[int(position["configured_tag_id"])],
                    world_frames,
                )
                raw_normal = candidate.get("tag_normal_world")
                if expected_normal is None or raw_normal is None:
                    return distance, None, distance
                candidate_normal = np.asarray(raw_normal, dtype=float)
                denominator = float(
                    np.linalg.norm(expected_normal)
                    * np.linalg.norm(candidate_normal)
                )
                if denominator <= 1e-9:
                    return distance, None, distance
                angle = math.degrees(math.acos(float(np.clip(
                    np.dot(expected_normal, candidate_normal) / denominator,
                    -1.0,
                    1.0,
                ))))
                # The normal distinguishes opposing +Y/-Y yoke faces even when
                # their tag centers are only a few centimetres apart.
                cost = distance + min(angle, 90.0) / 90.0 * 0.040
                return distance, angle, cost

            metrics = [
                [
                    match_metrics(positions[index], candidate)
                    for candidate in replacement_candidates
                ]
                for index in empty_indexes
            ]
            costs = np.asarray([
                [item[2] for item in row] for row in metrics
            ])
            rows, columns = linear_sum_assignment(costs)
            for row, column in zip(rows, columns):
                distance, normal_error, _cost = metrics[int(row)][int(column)]
                if (
                    distance > self.options.robot_slot_match_tolerance_m
                    or (normal_error is not None and normal_error > 35.0)
                ):
                    continue
                index = empty_indexes[int(row)]
                candidate = replacement_candidates[int(column)]
                positions[index].update({
                    "tag_id": int(candidate["tag_id"]),
                    "replacement": True,
                    "state": "measured",
                    "match_distance_mm": round(distance * 1000.0, 2),
                    "match_normal_error_deg": (
                        None if normal_error is None else round(normal_error, 2)
                    ),
                    "observations": int(candidate["observations"]),
                    "used_observations": int(candidate["used_observations"]),
                    "viewpoint_span_deg": float(
                        candidate.get("viewpoint_span_deg", 0.0)
                    ),
                    "required_viewpoint_span_deg": float(candidate.get(
                        "required_viewpoint_span_deg",
                        self.options.min_viewpoint_span_deg,
                    )),
                })
        replacement_specs: dict[int, dict[str, Any]] = {}
        for item in positions:
            if not item.get("replacement") or item.get("tag_id") is None:
                continue
            spec = dict(self.robot_tags[int(item["configured_tag_id"])])
            spec["label"] = str(item["position"])
            spec["position"] = str(item["position"])
            replacement_specs[int(item["tag_id"])] = spec
        self._replacement_specs = replacement_specs
        return positions

    def _known_role(self, tag_id: int) -> str | None:
        if tag_id in self.anchor_ids:
            return "calibration_anchor"
        if tag_id in self.robot_tags or tag_id in self._replacement_specs:
            return "robot"
        if tag_id in self.position_tag_overrides.values():
            return "robot"
        if tag_id in self.expected_ground_ids:
            return "ground"
        return None

    def _looks_like_ground(self, transform: RigidTransform) -> bool:
        return (
            abs(float(transform.translation_m[2]))
            <= self.options.ground_height_tolerance_m
            and _normal_error_from_up_deg(transform)
            <= self.options.ground_normal_tolerance_deg
        )

    def _world_robot_frames(
        self,
        detections: Sequence[TagCorners],
        world_from_camera: RigidTransform,
        camera_matrix: np.ndarray,
        distortion: np.ndarray,
        floor_normal_camera: np.ndarray,
    ) -> dict[str, RigidTransform] | None:
        by_id = {item.tag_id: item for item in detections}
        orientation_anchor = self._orientation_anchor_spec()
        if orientation_anchor is not None:
            tag_id, spec = orientation_anchor
            world_from_tag = None
            detection = by_id.get(tag_id)
            if detection is not None:
                try:
                    pose = estimate_tag_pose(
                        detection,
                        camera_matrix,
                        distortion,
                        marker_size_m=self.marker_sizes_m.get(
                            tag_id, self.marker_size_m
                        ),
                        preferred_normal_camera=floor_normal_camera,
                    )
                except (ValueError, cv2.error):
                    pose = None
                if (
                    pose is not None
                    and pose.reprojection_rms_px
                    <= self.options.max_reprojection_rms_px
                ):
                    world_from_tag = world_from_camera.compose(
                        pose.camera_from_tag
                    )
            elif tag_id in self._observations:
                consensus = self._consensus_for(tag_id)
                if consensus.stable:
                    world_from_tag = consensus.transform
            if world_from_tag is not None:
                return self._world_robot_frames_from_anchor(
                    world_from_tag,
                    spec,
                )
        body_ids = [
            tag_id for tag_id, spec in self.robot_tags.items()
            if str(spec.get("frame")) == "body"
        ]
        if self.body_anchor_tag_id is not None:
            body_ids.sort(key=lambda tag_id: tag_id != self.body_anchor_tag_id)
        for tag_id in body_ids:
            spec = self.robot_tags[tag_id]
            world_from_tag = None
            detection = by_id.get(tag_id)
            if detection is not None:
                try:
                    pose = estimate_tag_pose(
                        detection,
                        camera_matrix,
                        distortion,
                        marker_size_m=self.marker_sizes_m.get(
                            tag_id, self.marker_size_m
                        ),
                        preferred_normal_camera=floor_normal_camera,
                    )
                except (ValueError, cv2.error):
                    pose = None
                if (
                    pose is not None
                    and pose.reprojection_rms_px
                    <= self.options.max_reprojection_rms_px
                ):
                    world_from_tag = world_from_camera.compose(
                        pose.camera_from_tag
                    )
            elif tag_id in self._observations:
                consensus = self._consensus_for(tag_id)
                if consensus.stable:
                    world_from_tag = consensus.transform
            if world_from_tag is None:
                continue
            return self._world_robot_frames_from_anchor(world_from_tag, spec)
        return None

    @staticmethod
    def _robot_normal_world(
        spec: Mapping[str, Any],
        world_frames: Mapping[str, RigidTransform] | None,
    ) -> np.ndarray | None:
        surface = str(spec.get("surface", ""))
        if surface == "horizontal" or str(spec.get("frame")) == "body":
            return np.asarray([0.0, 0.0, 1.0])
        if world_frames is None:
            return None
        frame = str(spec.get("frame", ""))
        world_from_frame = world_frames.get(frame)
        if world_from_frame is None:
            return None
        mount_side = str(spec.get("mount_side", ""))
        if mount_side == "+y":
            normal_frame = np.asarray([0.0, 1.0, 0.0])
        elif mount_side == "-y":
            normal_frame = np.asarray([0.0, -1.0, 0.0])
        else:
            return None
        return world_from_frame.rotation.apply(normal_frame)

    def observe_frame(
        self,
        detections: Sequence[TagCorners],
        world_from_camera: RigidTransform,
        camera_matrix: np.ndarray,
        distortion: np.ndarray,
    ) -> None:
        self.frames += 1
        floor_normal_camera = world_from_camera.rotation.inv().apply(
            [0.0, 0.0, 1.0]
        )
        world_frames = self._world_robot_frames(
            detections,
            world_from_camera,
            camera_matrix,
            distortion,
            floor_normal_camera,
        )
        for detection in detections:
            if detection.tag_id in self._frozen:
                if self.options.freeze_stable_tags:
                    continue
                # A restored checkpoint remains immediately usable, then starts
                # refining again as soon as the operator revisits this tag.
                self._frozen.pop(detection.tag_id, None)
                self._restored_viewpoint_spans.pop(detection.tag_id, None)
            marker_size = self.marker_sizes_m.get(
                detection.tag_id, self.marker_size_m
            )
            role = self._known_role(detection.tag_id)
            previous = self._observations.get(detection.tag_id, [])
            preferred_normal = None
            if role in ("ground", "calibration_anchor"):
                preferred_normal = floor_normal_camera
            elif role == "robot":
                normal_world = self._robot_normal_world(
                    self.robot_tags.get(
                        detection.tag_id,
                        self._replacement_specs.get(detection.tag_id, {}),
                    ),
                    world_frames,
                )
                if normal_world is not None:
                    preferred_normal = world_from_camera.rotation.inv().apply(
                        normal_world
                    )
            elif previous:
                prior_normal_world = previous[-1].world_from_tag.rotation.apply(
                    [0.0, 0.0, 1.0]
                )
                preferred_normal = world_from_camera.rotation.inv().apply(
                    prior_normal_world
                )
            try:
                pose = estimate_tag_pose(
                    detection,
                    camera_matrix,
                    distortion,
                    marker_size_m=marker_size,
                    preferred_normal_camera=preferred_normal,
                )
                world_from_tag = world_from_camera.compose(pose.camera_from_tag)
                if role is None and not previous:
                    floor_pose = estimate_tag_pose(
                        detection,
                        camera_matrix,
                        distortion,
                        marker_size_m=marker_size,
                        preferred_normal_camera=floor_normal_camera,
                    )
                    floor_world_from_tag = world_from_camera.compose(
                        floor_pose.camera_from_tag
                    )
                    if self._looks_like_ground(floor_world_from_tag):
                        pose = floor_pose
                        world_from_tag = floor_world_from_tag
            except (ValueError, cv2.error):
                continue
            if pose.reprojection_rms_px > self.options.max_reprojection_rms_px:
                continue
            observations = self._observations.setdefault(detection.tag_id, [])
            observations.append(TagSurveyObservation(
                tag_id=detection.tag_id,
                world_from_tag=world_from_tag,
                marker_size_m=marker_size,
                reprojection_rms_px=pose.reprojection_rms_px,
                image_heading_deg=detection.tag_y_clockwise_from_image_up_deg,
                confidence=detection.confidence,
                camera_position_world=world_from_camera.translation_m.copy(),
            ))
            if len(observations) > self.options.max_observations_per_tag:
                # Preserve the first clean viewpoint as the baseline while
                # rolling newer pose samples through the bounded consensus
                # window. Dropping the baseline made a slow side-step read as
                # 0 degrees forever because both ends aged out together.
                del observations[1 if len(observations) > 1 else 0]
            if (
                self.options.freeze_stable_tags
                and len(observations) >= self.options.min_observations
            ):
                consensus = self._consensus_for(detection.tag_id)
                if (
                    consensus.stable
                    and self._viewpoint_span_deg(
                        detection.tag_id, consensus.transform
                    ) >= self.options.min_viewpoint_span_deg
                ):
                    self._frozen[detection.tag_id] = consensus

    def estimate_world_from_camera(
        self,
        detections: Sequence[TagCorners],
        predicted_world_from_camera: RigidTransform,
        camera_matrix: np.ndarray,
        distortion: np.ndarray,
    ) -> TransformConsensus | None:
        """Correct camera drift from tags already stable in this stationary map."""
        candidates: list[RigidTransform] = []
        weights: list[float] = []
        errors: list[float] = []
        for detection in detections:
            if detection.tag_id not in self._observations:
                continue
            tag_consensus = self._consensus_for(detection.tag_id)
            if not tag_consensus.stable:
                continue
            world_from_tag = tag_consensus.transform
            expected_normal_camera = (
                predicted_world_from_camera.rotation.inv().apply(
                    world_from_tag.rotation.apply([0.0, 0.0, 1.0])
                )
            )
            try:
                pose = estimate_tag_pose(
                    detection,
                    camera_matrix,
                    distortion,
                    marker_size_m=self.marker_sizes_m.get(
                        detection.tag_id, self.marker_size_m
                    ),
                    preferred_normal_camera=expected_normal_camera,
                )
            except (ValueError, cv2.error):
                continue
            if pose.reprojection_rms_px > self.options.max_reprojection_rms_px:
                continue
            candidates.append(
                world_from_tag.compose(pose.camera_from_tag.inverse())
            )
            errors.append(pose.reprojection_rms_px)
            weights.append(
                detection.confidence
                / max(0.10, pose.reprojection_rms_px) ** 2
            )
        if not candidates:
            return None
        return _transform_consensus(
            candidates,
            weights=np.asarray(weights),
            min_observations=2,
            max_translation_spread_m=0.050,
            max_rotation_spread_deg=8.0,
            reprojection_errors=errors,
        )

    def _consensus_for(self, tag_id: int) -> TransformConsensus:
        if tag_id in self._frozen:
            return self._frozen[tag_id]
        observations = self._observations[tag_id]
        weights = np.asarray([
            item.confidence / max(0.10, item.reprojection_rms_px) ** 2
            for item in observations
        ])
        return _transform_consensus(
            [item.world_from_tag for item in observations],
            weights=weights,
            min_observations=self.options.min_observations,
            max_translation_spread_m=self.options.max_translation_spread_m,
            max_rotation_spread_deg=self.options.max_rotation_spread_deg,
            reprojection_errors=[
                item.reprojection_rms_px for item in observations
            ],
        )

    def _viewpoint_span_deg(
        self,
        tag_id: int,
        tag_transform: RigidTransform,
    ) -> float:
        """Largest camera-to-tag bearing change among accepted observations."""
        if tag_id in self._restored_viewpoint_spans:
            return self._restored_viewpoint_spans[tag_id]
        directions: list[np.ndarray] = []
        for observation in self._observations[tag_id]:
            direction = (
                observation.camera_position_world
                - tag_transform.translation_m
            )
            magnitude = float(np.linalg.norm(direction))
            if magnitude > 1e-6:
                directions.append(direction / magnitude)
        if len(directions) < 2:
            return 0.0
        unit = np.stack(directions)
        smallest_dot = float(np.min(np.clip(unit @ unit.T, -1.0, 1.0)))
        return math.degrees(math.acos(smallest_dot))

    def tag_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for tag_id in sorted(self._observations):
            consensus = self._consensus_for(tag_id)
            transform = consensus.transform
            viewpoint_span_deg = self._viewpoint_span_deg(tag_id, transform)
            viewpoint_requirement_met = (
                viewpoint_span_deg >= self.options.min_viewpoint_span_deg
            )
            known_role = self._known_role(tag_id)
            role = known_role
            if role is None:
                role = "ground" if self._looks_like_ground(transform) else "unassigned"
            normal = transform.rotation.apply([0.0, 0.0, 1.0])
            tag_x = transform.rotation.apply([1.0, 0.0, 0.0])
            tag_y = transform.rotation.apply([0.0, 1.0, 0.0])
            heading = math.degrees(math.atan2(float(tag_y[0]), float(tag_y[1])))
            metadata = self.robot_tags.get(
                tag_id, self._replacement_specs.get(tag_id, {})
            )
            records.append({
                "tag_id": tag_id,
                "role": role,
                "label": metadata.get("label", f"discovered tag {tag_id}"),
                "robot_frame": metadata.get("frame"),
                "kind": metadata.get("kind"),
                "surface": metadata.get("surface"),
                "leg": metadata.get("leg"),
                "joint": metadata.get("joint"),
                "mount_side": metadata.get("mount_side"),
                "marker_size_m": self._observations[tag_id][0].marker_size_m,
                "world_from_tag": transform.to_dict(),
                "euler_xyz_deg": _euler_xyz_deg(transform.rotation),
                "tag_x_world": [round(float(value), 7) for value in tag_x],
                "tag_y_world": [round(float(value), 7) for value in tag_y],
                "tag_normal_world": [round(float(value), 7) for value in normal],
                "tag_y_heading_clockwise_from_world_y_deg": round(heading, 5),
                "height_above_ground_mm": round(
                    float(transform.translation_m[2]) * 1000.0, 3
                ),
                "normal_error_from_world_up_deg": round(
                    _normal_error_from_up_deg(transform), 5
                ),
                "observations": consensus.input_count,
                "used_observations": consensus.used_count,
                "translation_spread_mm": round(
                    consensus.translation_spread_mm, 4
                ),
                "rotation_spread_deg": round(consensus.rotation_spread_deg, 5),
                "mean_reprojection_rms_px": round(
                    float(consensus.mean_reprojection_rms_px or 0.0), 5
                ),
                "viewpoint_span_deg": round(viewpoint_span_deg, 3),
                "required_viewpoint_span_deg": round(
                    self.options.min_viewpoint_span_deg, 3
                ),
                "viewpoint_requirement_met": viewpoint_requirement_met,
                "possible_duplicate_id_or_tracking_jump": (
                    consensus.ambiguous_cluster
                ),
                "stable": consensus.stable and viewpoint_requirement_met,
            })
        return records

    def _quality_gate(
        self,
        records: Mapping[int, Mapping[str, Any]],
        *,
        coverage_complete: bool,
        ambiguous_ids: Sequence[int],
    ) -> dict[str, Any]:
        reference_errors: list[tuple[float, float, float]] = []
        for tag_id, reference in self.reference_floor_tags.items():
            record = records.get(tag_id)
            if record is None or not record.get("stable"):
                continue
            measured = RigidTransform.from_dict(record["world_from_tag"])
            translation_error = measured.translation_m - reference.translation_m
            reference_errors.append((
                float(np.linalg.norm(translation_error)),
                abs(float(translation_error[2])),
                math.degrees(float(
                    (reference.rotation.inv() * measured.rotation).magnitude()
                )),
            ))
        floor_position_rms_mm = None
        floor_height_rms_mm = None
        floor_rotation_rms_deg = None
        if reference_errors:
            values = np.asarray(reference_errors, dtype=float)
            floor_position_rms_mm = math.sqrt(float(np.mean(values[:, 0] ** 2))) * 1000.0
            floor_height_rms_mm = math.sqrt(float(np.mean(values[:, 1] ** 2))) * 1000.0
            floor_rotation_rms_deg = math.sqrt(float(np.mean(values[:, 2] ** 2)))

        jointly_validated_ids = sorted({
            tag_id
            for tag_ids, _reprojection, _depth in self._multi_floor_reference_frames
            for tag_id in tag_ids
        })
        joint_reprojection = (
            None if not self._multi_floor_reference_frames else float(np.median([
                item[1] for item in self._multi_floor_reference_frames
            ]))
        )
        joint_depth = (
            None if not self._multi_floor_reference_frames else float(np.median([
                item[2] for item in self._multi_floor_reference_frames
            ]))
        )

        checks = {
            "coverage": coverage_complete,
            "no_ambiguous_ids": not ambiguous_ids,
            "mapped_floor_tags": (
                not self.reference_floor_tags
                or len(reference_errors) == len(self.reference_floor_tags)
            ),
            "joint_floor_grid_frames": (
                len(self._multi_floor_reference_frames) >= 6
            ) if self.reference_floor_tags else True,
            "joint_floor_grid_coverage": (
                set(jointly_validated_ids) >= set(self.reference_floor_tags)
            ) if self.reference_floor_tags else True,
            "joint_floor_reprojection": (
                joint_reprojection is not None and joint_reprojection <= 1.25
            ) if self.reference_floor_tags else True,
            "joint_floor_depth_plane": (
                joint_depth is not None and joint_depth <= 12.0
            ) if self.reference_floor_tags else True,
            "floor_position_rms": (
                floor_position_rms_mm is not None
                and floor_position_rms_mm <= 10.0
            ) if self.reference_floor_tags else True,
            "floor_height_rms": (
                floor_height_rms_mm is not None
                and floor_height_rms_mm <= 6.0
            ) if self.reference_floor_tags else True,
            "floor_rotation_rms": (
                floor_rotation_rms_deg is not None
                and floor_rotation_rms_deg <= 3.0
            ) if self.reference_floor_tags else True,
        }
        descriptions = {
            "coverage": "some required robot or floor mounts still need views",
            "no_ambiguous_ids": "a tag ID formed two incompatible pose clusters",
            "mapped_floor_tags": "not every mapped floor tag has a stable measurement",
            "joint_floor_grid_frames": "show pairs of floor tags together for at least six clean frames",
            "joint_floor_grid_coverage": "each floor tag must be seen together with another floor tag",
            "joint_floor_reprojection": "the measured floor grid does not fit below 1.25 px RMS",
            "joint_floor_depth_plane": "the LiDAR floor plane residual exceeds 12 mm RMS",
            "floor_position_rms": "floor-grid position residual exceeds 10 mm RMS",
            "floor_height_rms": "floor-grid height residual exceeds 6 mm RMS",
            "floor_rotation_rms": "floor-tag orientation residual exceeds 3 degrees RMS",
        }
        return {
            "passed": all(checks.values()),
            "checks": checks,
            "failing_checks": [
                descriptions[name] for name, passed in checks.items() if not passed
            ],
            "reference_floor_tag_count": len(reference_errors),
            "required_reference_floor_tag_count": len(self.reference_floor_tags),
            "joint_floor_reference_frames": len(
                self._multi_floor_reference_frames
            ),
            "jointly_validated_floor_tag_ids": jointly_validated_ids,
            "joint_floor_reprojection_rms_px": (
                None if joint_reprojection is None else round(joint_reprojection, 4)
            ),
            "joint_floor_depth_plane_rms_mm": (
                None if joint_depth is None else round(joint_depth, 3)
            ),
            "floor_position_rms_mm": (
                None if floor_position_rms_mm is None
                else round(floor_position_rms_mm, 3)
            ),
            "floor_height_rms_mm": (
                None if floor_height_rms_mm is None
                else round(floor_height_rms_mm, 3)
            ),
            "floor_rotation_rms_deg": (
                None if floor_rotation_rms_deg is None
                else round(floor_rotation_rms_deg, 4)
            ),
            "thresholds": {
                "floor_position_rms_mm": 10.0,
                "floor_height_rms_mm": 6.0,
                "floor_rotation_rms_deg": 3.0,
                "joint_floor_reprojection_rms_px": 1.25,
                "joint_floor_depth_plane_rms_mm": 12.0,
                "joint_floor_reference_frames": 6,
            },
        }

    def progress(
        self,
        record_list: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Return UI coverage, reusing a caller's consensus snapshot when given."""
        records = {
            int(item["tag_id"]): dict(item)
            for item in (self.tag_records() if record_list is None else record_list)
        }
        stable = {
            tag_id for tag_id, item in records.items() if item["stable"]
        }
        missing_robot = sorted(self.expected_robot_ids - stable)
        missing_ground = sorted(self.expected_ground_ids - stable)
        ambiguous = sorted(
            tag_id for tag_id, item in records.items()
            if item["possible_duplicate_id_or_tracking_jump"]
        )
        robot_positions = self._robot_position_records(records)
        missing_robot_positions = [
            str(item["position"]) for item in robot_positions
            if item["state"] != "measured"
        ]
        unseen_robot_positions = [
            str(item["position"]) for item in robot_positions
            if item["state"] == "not_seen"
        ]
        robot_positions_needing_another_view = [
            str(item["position"]) for item in robot_positions
            if item["state"] == "seen_needs_another_view"
        ]
        ground_tag_status = []
        for tag_id in sorted(self.expected_ground_ids):
            record = records.get(tag_id)
            state = (
                "not_seen" if record is None
                else "measured" if record["stable"]
                else "seen_needs_another_view"
            )
            ground_tag_status.append({
                "tag_id": tag_id,
                "state": state,
                "observations": 0 if record is None else record["observations"],
                "used_observations": (
                    0 if record is None else record["used_observations"]
                ),
                "viewpoint_span_deg": (
                    0.0 if record is None
                    else float(record.get("viewpoint_span_deg", 0.0))
                ),
                "required_viewpoint_span_deg": (
                    self.options.min_viewpoint_span_deg
                    if record is None
                    else float(record.get(
                        "required_viewpoint_span_deg",
                        self.options.min_viewpoint_span_deg,
                    ))
                ),
            })
        coverage_complete = not missing_robot_positions and not missing_ground
        quality_gate = self._quality_gate(
            records,
            coverage_complete=coverage_complete,
            ambiguous_ids=ambiguous,
        )
        angle_positions = [
            item for item in robot_positions if item.get("kind") == "yoke_face"
        ]
        top_positions = [
            item for item in robot_positions if item.get("kind") != "yoke_face"
        ]
        return {
            "complete": bool(quality_gate["passed"]),
            "coverage_complete": coverage_complete,
            "quality_gate": quality_gate,
            "robot_position_counts": {
                "top_and_chassis": {
                    "measured": sum(
                        item["state"] == "measured" for item in top_positions
                    ),
                    "required": len(top_positions),
                },
                "vertical_angle": {
                    "measured": sum(
                        item["state"] == "measured" for item in angle_positions
                    ),
                    "required": len(angle_positions),
                },
            },
            "expected_robot_tag_ids": sorted(self.expected_robot_ids),
            "expected_robot_positions": [
                str(item["position"]) for item in robot_positions
            ],
            "expected_ground_tag_ids": sorted(self.expected_ground_ids),
            "stable_tag_ids": sorted(stable),
            "robot_positions": robot_positions,
            "missing_robot_positions": missing_robot_positions,
            "unseen_robot_positions": unseen_robot_positions,
            "robot_positions_needing_another_view": (
                robot_positions_needing_another_view
            ),
            "ground_tag_status": ground_tag_status,
            "unseen_ground_tag_ids": [
                item["tag_id"] for item in ground_tag_status
                if item["state"] == "not_seen"
            ],
            "ground_tags_needing_another_view": [
                item["tag_id"] for item in ground_tag_status
                if item["state"] == "seen_needs_another_view"
            ],
            # Retained for schema compatibility.  UI and completion use named
            # physical positions so a replacement ID can fill a mount slot.
            "missing_robot_tag_ids": missing_robot,
            "missing_ground_tag_ids": missing_ground,
            "ambiguous_tag_ids": ambiguous,
            "discovered_unexpected_tag_ids": sorted(
                set(records) - self.expected_robot_ids
                - self.expected_ground_ids - self.anchor_ids
                - set(self._replacement_specs)
            ),
        }

    def summary(self) -> dict[str, Any]:
        records = self.tag_records()
        progress = self.progress()
        by_id = {int(item["tag_id"]): item for item in records}
        ground = [
            item for item in records
            if item["stable"] and item["role"] in ("ground", "calibration_anchor")
        ]
        floor_distances = []
        for first_index, first in enumerate(ground):
            first_position = np.asarray(
                first["world_from_tag"]["translation_m"], dtype=float
            )
            for second in ground[first_index + 1:]:
                second_position = np.asarray(
                    second["world_from_tag"]["translation_m"], dtype=float
                )
                delta = second_position - first_position
                floor_distances.append({
                    "tag_ids": [first["tag_id"], second["tag_id"]],
                    "center_distance_m": round(float(np.linalg.norm(delta)), 6),
                    "planar_distance_m": round(float(np.linalg.norm(delta[:2])), 6),
                    "delta_xyz_m": [round(float(value), 6) for value in delta],
                })
        return {
            "schema_version": 1,
            "coordinate_convention": (
                "world_from_tag; board +x printed right, +y printed up, "
                "+z out of the ground-facing-up tag surface"
            ),
            "stationary_robot_required": True,
            "zero_pose_required_for_mount_learning": True,
            "frames": self.frames,
            **progress,
            "tags": records,
            "floor_tag_distances": floor_distances,
            "stable_tag_count": sum(bool(item["stable"]) for item in records),
            "known_tag_count": len(by_id),
        }


def learn_zero_pose_mounts(
    tracker_config: Mapping[str, Any],
    survey: Mapping[str, Any],
    *,
    joint_angles_deg: Mapping[str, float] | None = None,
    body_anchor_tag_id: int | None = None,
    orientation_anchor_tag_id: int | None = None,
) -> tuple[dict[int, RigidTransform], dict[str, Any]]:
    """Learn configured robot tag mounts from one known stationary pose.

    A body-tag translation fixes the body-origin gauge.  When an unchanged L0
    tag is supplied, its configured mount and the BuildViz zero-pose frame fix
    the body-axis orientation independently of the chassis tag's old yaw.
    """
    robot_config = dict(tracker_config.get("robot_pose", {}))
    configured_tag_specs = {
        int(raw_id): dict(spec)
        for raw_id, spec in robot_config.get("tags", {}).items()
    }
    stable_survey_tags = {
        int(item["tag_id"]): item
        for item in survey.get("tags", [])
        if item.get("stable")
    }
    tag_specs: dict[int, dict[str, Any]] = {}
    position_assignments = survey.get("robot_positions", [])
    if position_assignments:
        for assignment in position_assignments:
            if assignment.get("state") != "measured":
                continue
            actual_id = assignment.get("tag_id")
            configured_id = assignment.get("configured_tag_id")
            if actual_id is None or configured_id is None:
                continue
            actual_id = int(actual_id)
            configured_id = int(configured_id)
            if actual_id not in stable_survey_tags or configured_id not in configured_tag_specs:
                continue
            spec = dict(configured_tag_specs[configured_id])
            spec["frame"] = str(assignment["frame"])
            spec["configured_tag_id"] = configured_id
            spec["replacement"] = actual_id != configured_id
            tag_specs[actual_id] = spec
    else:
        tag_specs = {
            tag_id: {**spec, "configured_tag_id": tag_id, "replacement": False}
            for tag_id, spec in configured_tag_specs.items()
            if tag_id in stable_survey_tags
            and stable_survey_tags[tag_id].get("role") == "robot"
        }
    survey_tags = {
        tag_id: stable_survey_tags[tag_id]
        for tag_id in tag_specs
    }
    available_body_ids = sorted(
        tag_id for tag_id, spec in tag_specs.items()
        if spec.get("frame") == "body" and tag_id in survey_tags
    )
    if not available_body_ids:
        return {}, {
            "ok": False,
            "error": "no stable configured body tag was surveyed",
            "identifiability": (
                "A trusted body tag or another independent body-frame datum "
                "is required before link-tag mounts can be learned."
            ),
        }
    requested_body_anchor_id = (
        None if body_anchor_tag_id is None else int(body_anchor_tag_id)
    )
    if requested_body_anchor_id is None:
        selected_anchor_id = available_body_ids[0]
    else:
        matching_body_ids = [
            tag_id for tag_id in available_body_ids
            if tag_id == requested_body_anchor_id
            or int(tag_specs[tag_id].get("configured_tag_id", tag_id))
            == requested_body_anchor_id
        ]
        if not matching_body_ids:
            return {}, {
                "ok": False,
                "error": (
                    f"body anchor position {requested_body_anchor_id} was not "
                    "stably surveyed"
                ),
                "available_body_tag_ids": available_body_ids,
                "identifiability": (
                    "The chassis position must contain a stable measured tag "
                    "to fix the body-origin translation."
                ),
            }
        selected_anchor_id = matching_body_ids[0]
    anchor_spec = tag_specs[selected_anchor_id]
    world_from_anchor_tag = RigidTransform.from_dict(
        survey_tags[selected_anchor_id]["world_from_tag"]
    )
    body_from_anchor_tag = RigidTransform.from_dict(
        anchor_spec.get("frame_from_tag", {})
    )
    angles = {name: 0.0 for name in JOINT_NAMES}
    if joint_angles_deg is not None:
        angles.update({str(key): float(value) for key, value in joint_angles_deg.items()})
    geometry = HexapodGeometry.from_dict(robot_config.get("geometry"))
    world_from_body = world_from_anchor_tag.compose(
        body_from_anchor_tag.inverse()
    )
    orientation_anchor_frame: str | None = None
    if orientation_anchor_tag_id is not None:
        orientation_anchor_tag_id = int(orientation_anchor_tag_id)
        orientation_spec = tag_specs.get(orientation_anchor_tag_id)
        orientation_record = survey_tags.get(orientation_anchor_tag_id)
        if orientation_spec is None or orientation_record is None:
            return {}, {
                "ok": False,
                "error": (
                    f"orientation anchor tag {orientation_anchor_tag_id} was "
                    "not stably surveyed"
                ),
                "identifiability": (
                    "The unchanged L0 hip tag must be measured to align the "
                    "BuildViz body axes with the photographed robot."
                ),
            }
        orientation_anchor_frame = str(orientation_spec["frame"])
        zero_frames = forward_frame_transforms(
            RigidTransform(np.zeros(3), Rotation.identity()),
            angles,
            geometry=geometry,
        )
        body_from_orientation_tag = zero_frames[
            orientation_anchor_frame
        ].compose(RigidTransform.from_dict(
            orientation_spec.get("frame_from_tag", {})
        ))
        world_from_orientation_tag = RigidTransform.from_dict(
            orientation_record["world_from_tag"]
        )
        body_rotation = (
            world_from_orientation_tag.rotation
            * body_from_orientation_tag.rotation.inv()
        )
        # Keep the trusted body tag's configured translation as the origin
        # gauge, but take body-axis orientation from the unchanged L0 mount.
        body_translation = (
            world_from_anchor_tag.translation_m
            - body_rotation.apply(body_from_anchor_tag.translation_m)
        )
        world_from_body = RigidTransform(body_translation, body_rotation)
    world_frames = forward_frame_transforms(
        world_from_body, angles, geometry=geometry
    )
    learned: dict[int, RigidTransform] = {}
    per_tag: list[dict[str, Any]] = []
    for tag_id, spec in sorted(tag_specs.items()):
        if tag_id not in survey_tags:
            continue
        frame = str(spec["frame"])
        world_from_tag = RigidTransform.from_dict(
            survey_tags[tag_id]["world_from_tag"]
        )
        frame_from_tag = world_frames[frame].inverse().compose(world_from_tag)
        # With a distinct L0 orientation anchor, the old body-tag yaw is no
        # longer the gauge and must be relearned in the corrected body frame.
        mount_updated = (
            tag_id != selected_anchor_id
            or orientation_anchor_tag_id is not None
        )
        if mount_updated:
            learned[tag_id] = frame_from_tag
        per_tag.append({
            "tag_id": tag_id,
            "configured_tag_id": int(spec.get("configured_tag_id", tag_id)),
            "replacement": bool(spec.get("replacement", False)),
            "frame": frame,
            "frame_from_tag": frame_from_tag.to_dict(),
            "euler_xyz_deg": _euler_xyz_deg(frame_from_tag.rotation),
            "body_frame_anchor": tag_id == selected_anchor_id,
            "mount_updated": mount_updated,
        })

    # Static inter-tag baselines are useful measured geometry, but link lengths
    # and mount offsets are coupled in one pose and must not be called exact.
    baselines = []
    for leg in range(6):
        leg_tags = [
            (tag_id, spec)
            for tag_id, spec in tag_specs.items()
            if str(spec.get("frame", "")).startswith(f"L{leg}_")
            and tag_id in survey_tags
        ]
        for first_index, (first_id, first_spec) in enumerate(leg_tags):
            first_position = np.asarray(
                survey_tags[first_id]["world_from_tag"]["translation_m"]
            )
            for second_id, second_spec in leg_tags[first_index + 1:]:
                second_position = np.asarray(
                    survey_tags[second_id]["world_from_tag"]["translation_m"]
                )
                baselines.append({
                    "leg": leg,
                    "tag_ids": [first_id, second_id],
                    "frames": [first_spec["frame"], second_spec["frame"]],
                    "tag_center_distance_m": round(float(np.linalg.norm(
                        second_position - first_position
                    )), 6),
                })
    report = {
        "ok": True,
        "pose_used": {
            "joint_angles_deg": angles,
            "body_anchor_tag_id": selected_anchor_id,
            "body_anchor_configured_tag_id": int(
                anchor_spec.get("configured_tag_id", selected_anchor_id)
            ),
            "body_anchor_requested_tag_id": requested_body_anchor_id,
            "body_anchor_auto_selected": body_anchor_tag_id is None,
            "orientation_anchor_tag_id": orientation_anchor_tag_id,
            "orientation_anchor_frame": orientation_anchor_frame,
            "orientation_source": (
                "unchanged_leg_zero_tag_and_buildviz_zero_pose"
                if orientation_anchor_tag_id is not None
                else "body_tag_mount"
            ),
            "world_from_body": world_from_body.to_dict(),
        },
        "learned_mounts": per_tag,
        "measured_inter_tag_baselines": baselines,
        "geometry_status": "partial_static_measurements",
        "not_identifiable_from_one_static_pose": [
            "joint-axis locations versus tag-center offsets",
            "coxa/femur/tibia link lengths independently of mount translations",
            "joint-axis direction and linkage geometry without joint excitation",
        ],
        "next_capture_for_geometry_fit": (
            "Record several stationary, encoder-known poses spanning each joint; "
            "keep the trusted body datum and the existing tibia-fixed side tags."
        ),
    }
    return learned, report


def apply_survey_to_config(
    tracker_config: Mapping[str, Any],
    survey: Mapping[str, Any],
    learned_mounts: Mapping[int, RigidTransform],
) -> dict[str, Any]:
    """Return a config containing stable floor poses and relearned mounts."""
    import copy

    updated = copy.deepcopy(dict(tracker_config))
    expected_ground_ids = {
        int(value) for value in survey.get("expected_ground_tag_ids", [])
    }
    floor_tags = {
        str(raw_id): copy.deepcopy(spec)
        for raw_id, spec in updated.get("floor_tags", {}).items()
        if int(raw_id) not in expected_ground_ids
    }
    for item in survey.get("tags", []):
        if not item.get("stable") or item.get("role") not in (
            "ground", "calibration_anchor"
        ):
            continue
        tag_id = str(int(item["tag_id"]))
        floor_tags[tag_id] = {
            "label": str(item.get("label", f"surveyed floor tag {tag_id}")),
            "marker_size_m": float(item["marker_size_m"]),
            "world_from_tag": item["world_from_tag"],
            "survey_quality": {
                "observations": int(item["used_observations"]),
                "translation_spread_mm": item["translation_spread_mm"],
                "rotation_spread_deg": item["rotation_spread_deg"],
            },
        }
    updated["floor_tags"] = floor_tags
    robot_tags = updated.setdefault("robot_pose", {}).setdefault("tags", {})
    for assignment in survey.get("robot_positions", []):
        if not assignment.get("replacement") or assignment.get("state") != "measured":
            continue
        configured_id = str(int(assignment["configured_tag_id"]))
        actual_id = str(int(assignment["tag_id"]))
        if configured_id not in robot_tags:
            continue
        replacement_spec = robot_tags.pop(configured_id)
        replacement_spec["previous_tag_id"] = int(configured_id)
        replacement_spec["label"] = (
            f"{assignment['position']} (surveyed tag {actual_id})"
        )
        robot_tags[actual_id] = replacement_spec
    for tag_id, frame_from_tag in learned_mounts.items():
        if str(tag_id) in robot_tags:
            robot_tags[str(tag_id)]["frame_from_tag"] = frame_from_tag.to_dict()
            robot_tags[str(tag_id)]["mount_source"] = "zero_pose_handheld_survey"
    updated["tag_survey"] = {
        "schema_version": survey.get("schema_version", 1),
        "complete": bool(survey.get("complete")),
        "stable_tag_ids": survey.get("stable_tag_ids", []),
        "robot_positions": survey.get("robot_positions", []),
        "stationary_robot_required": True,
    }
    return updated
