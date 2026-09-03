"""Estimate hexapod body pose and joint angles from rigid-link markers.

This module is deliberately camera-library agnostic.  An AprilTag detector (or
another tracker) supplies ``camera_from_tag`` transforms; this module applies
the configured camera extrinsics and tag mounts, fuses duplicate observations,
and solves the STS3215 kinematic chain.

Frame convention
----------------
Transforms are named ``A_from_B`` and map coordinates expressed in frame B
into frame A.  Quaternions in JSON are always ``[x, y, z, w]``.

The observed robot frames are:

``body``
    Chassis frame.  A tag on any yaw-servo housing is body-fixed.
``L{n}_coxa``
    Yaw output / coxa frame.  The hip-servo housing moves with this frame.
``L{n}_femur``
    Hip output / femur frame.  The knee-servo housing moves with this frame.
``L{n}_tibia``
    Knee output / tibia frame.  This requires a marker on the tibia; the knee
    motor housing itself is upstream of the knee output and cannot observe it.

The estimator reports the robot's measured ``absolute_tibia`` convention:
``L*_knee`` is the absolute tibia angle in the leg plane, not MuJoCo's
relative knee angle.

No robot I/O or motion is performed here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation


LEG_COUNT = 6
JOINT_NAMES = tuple(
    f"L{leg}_{axis}"
    for leg in range(LEG_COUNT)
    for axis in ("yaw", "hip", "knee")
)


def _vector3(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain three finite numbers")
    return array


def _rotation_from_dict(value: Mapping[str, Any]) -> Rotation:
    choices = [
        key for key in (
            "quaternion_xyzw",
            "rotation_matrix",
            "rotvec_rad",
            "rotvec_deg",
            "euler_xyz_deg",
        )
        if key in value
    ]
    if len(choices) > 1:
        raise ValueError(f"transform has multiple rotation fields: {choices}")
    if not choices:
        return Rotation.identity()

    key = choices[0]
    raw = np.asarray(value[key], dtype=float)
    if key == "quaternion_xyzw":
        if raw.shape != (4,) or not np.all(np.isfinite(raw)):
            raise ValueError("quaternion_xyzw must contain four finite numbers")
        norm = float(np.linalg.norm(raw))
        if norm < 1e-12:
            raise ValueError("quaternion_xyzw cannot be all zero")
        return Rotation.from_quat(raw / norm)
    if key == "rotation_matrix":
        if raw.shape != (3, 3) or not np.all(np.isfinite(raw)):
            raise ValueError("rotation_matrix must be a finite 3x3 matrix")
        return Rotation.from_matrix(raw)
    if raw.shape != (3,) or not np.all(np.isfinite(raw)):
        raise ValueError(f"{key} must contain three finite numbers")
    if key == "rotvec_rad":
        return Rotation.from_rotvec(raw)
    if key == "rotvec_deg":
        return Rotation.from_rotvec(np.radians(raw))
    return Rotation.from_euler("xyz", raw, degrees=True)


@dataclass(frozen=True)
class RigidTransform:
    """A rigid transform whose name should be read as ``parent_from_child``."""

    translation_m: np.ndarray
    rotation: Rotation

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "translation_m",
            _vector3(self.translation_m, name="translation_m"),
        )

    @classmethod
    def identity(cls) -> "RigidTransform":
        return cls(np.zeros(3), Rotation.identity())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "RigidTransform":
        if value is None:
            return cls.identity()
        if "matrix" in value:
            matrix = np.asarray(value["matrix"], dtype=float)
            if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
                raise ValueError("matrix must be a finite 4x4 matrix")
            if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
                raise ValueError("last transform-matrix row must be [0, 0, 0, 1]")
            return cls(matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]))
        return cls(
            _vector3(value.get("translation_m", [0.0, 0.0, 0.0]),
                     name="translation_m"),
            _rotation_from_dict(value),
        )

    def inverse(self) -> "RigidTransform":
        inverse_rotation = self.rotation.inv()
        return RigidTransform(
            inverse_rotation.apply(-self.translation_m),
            inverse_rotation,
        )

    def compose(self, child: "RigidTransform") -> "RigidTransform":
        """Return ``self @ child`` using transform-chain composition."""
        return RigidTransform(
            self.translation_m + self.rotation.apply(child.translation_m),
            self.rotation * child.rotation,
        )

    def apply(self, point: Sequence[float]) -> np.ndarray:
        return self.translation_m + self.rotation.apply(_vector3(point, name="point"))

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "translation_m": [round(float(v), 9) for v in self.translation_m],
            "quaternion_xyzw": [
                round(float(v), 9) for v in self.rotation.as_quat()
            ],
        }


@dataclass(frozen=True)
class HexapodGeometry:
    """STS3215 kinematic geometry in metres."""

    chassis_apothem_m: float = 0.100
    coxa_m: float = 0.0125
    femur_m: float = 0.090
    tibia_m: float = 0.150
    hip_anchor_y_m: float = -0.02565

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "HexapodGeometry":
        if not value:
            return cls()
        fields = {
            "chassis_apothem_m": cls.chassis_apothem_m,
            "coxa_m": cls.coxa_m,
            "femur_m": cls.femur_m,
            "tibia_m": cls.tibia_m,
            "hip_anchor_y_m": cls.hip_anchor_y_m,
        }
        for key in fields:
            if key in value:
                fields[key] = float(value[key])
        if any(not math.isfinite(v) for v in fields.values()):
            raise ValueError("geometry values must be finite")
        if any(fields[key] <= 0.0 for key in (
            "chassis_apothem_m", "coxa_m", "femur_m", "tibia_m"
        )):
            raise ValueError("geometry lengths must be positive")
        return cls(**fields)


@dataclass(frozen=True)
class TagMount:
    tag_id: int
    frame: str
    frame_from_tag: RigidTransform


@dataclass(frozen=True)
class TagDetection:
    tag_id: int
    camera: str
    camera_from_tag: RigidTransform
    weight: float = 1.0

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TagDetection":
        weight = float(value.get("weight", value.get("decision_margin", 1.0)))
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("detection weight must be a positive finite number")
        transform = value.get("camera_from_tag", value.get("transform"))
        if transform is None:
            raise ValueError("detection needs camera_from_tag")
        return cls(
            tag_id=int(value["tag_id"]),
            camera=str(value.get("camera", "camera0")),
            camera_from_tag=RigidTransform.from_dict(transform),
            weight=weight,
        )


@dataclass(frozen=True)
class _WeightedTransform:
    transform: RigidTransform
    weight: float
    tag_id: int | None = None


@dataclass(frozen=True)
class _FusedFrame:
    transform: RigidTransform
    count: int
    weight: float
    translation_spread_mm: float
    rotation_spread_deg: float
    tag_ids: tuple[int, ...]

    def diagnostics(self) -> dict[str, Any]:
        return {
            "observation_count": self.count,
            "tag_ids": list(self.tag_ids),
            "translation_spread_mm": round(self.translation_spread_mm, 3),
            "rotation_spread_deg": round(self.rotation_spread_deg, 3),
        }


def _fuse_transforms(values: Sequence[_WeightedTransform]) -> _FusedFrame:
    if not values:
        raise ValueError("cannot fuse an empty transform list")
    weights = np.asarray([value.weight for value in values], dtype=float)
    translations = np.stack([value.transform.translation_m for value in values])
    translation = np.average(translations, axis=0, weights=weights)
    rotations = Rotation.from_quat(
        np.stack([value.transform.rotation.as_quat() for value in values])
    )
    rotation = rotations.mean(weights=weights)

    translation_errors = np.linalg.norm(translations - translation, axis=1)
    rotation_errors = (rotation.inv() * rotations).magnitude()
    translation_spread = math.sqrt(float(np.average(
        translation_errors * translation_errors, weights=weights
    )))
    rotation_spread = math.sqrt(float(np.average(
        rotation_errors * rotation_errors, weights=weights
    )))
    return _FusedFrame(
        RigidTransform(translation, rotation),
        count=len(values),
        weight=float(np.sum(weights)),
        translation_spread_mm=translation_spread * 1000.0,
        rotation_spread_deg=math.degrees(rotation_spread),
        tag_ids=tuple(
            sorted(value.tag_id for value in values if value.tag_id is not None)
        ),
    )


def _rz(angle: float) -> Rotation:
    return Rotation.from_rotvec([0.0, 0.0, angle])


def _ry(angle: float) -> Rotation:
    return Rotation.from_rotvec([0.0, angle, 0.0])


def _decompose_zy(rotation: Rotation) -> tuple[float, float, float]:
    """Fit ``Rz(yaw) * Ry(pitch)`` and return angles + residual radians."""
    matrix = rotation.as_matrix()
    # The second column of Rz(yaw) @ Ry(pitch) is [-sin(yaw), cos(yaw), 0].
    # Using it avoids the 180-degree yaw flip that the first column develops
    # when an absolute tibia angle passes 90 degrees.
    yaw = math.atan2(float(-matrix[0, 1]), float(matrix[1, 1]))
    pitch = math.atan2(float(-matrix[2, 0]), float(matrix[2, 2]))
    fitted = _rz(yaw) * _ry(pitch)
    residual = float((fitted.inv() * rotation).magnitude())
    return yaw, pitch, residual


def _circular_mean(values: Sequence[tuple[float, float]]) -> float:
    vector = sum(weight * np.exp(1j * angle) for angle, weight in values)
    if abs(vector) < 1e-12:
        return float(values[0][0])
    return float(np.angle(vector))


def _angle_distance(a: float, b: float) -> float:
    return math.atan2(math.sin(a - b), math.cos(a - b))


def _quality(residual_deg: float | None) -> str:
    if residual_deg is None:
        return "unobservable"
    if residual_deg <= 2.0:
        return "good"
    if residual_deg <= 5.0:
        return "check_mount"
    return "poor"


def _joint_record(
    value_rad: float | None,
    *,
    residual_rad: float | None,
    sources: Iterable[str],
) -> dict[str, Any]:
    residual_deg = (
        None if residual_rad is None else math.degrees(float(residual_rad))
    )
    return {
        "observable": value_rad is not None,
        "value_deg": None if value_rad is None else round(math.degrees(value_rad), 5),
        "angular_residual_deg": (
            None if residual_deg is None else round(residual_deg, 4)
        ),
        "quality": _quality(residual_deg),
        "source_frames": list(sources),
    }


def _normalise_encoder_angles(
    value: Sequence[float] | Mapping[str, float] | None,
) -> dict[str, float]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        result = {str(key): float(angle) for key, angle in value.items()}
    else:
        if len(value) != len(JOINT_NAMES):
            raise ValueError("encoder_joint_deg list must contain 18 values")
        result = dict(zip(JOINT_NAMES, (float(angle) for angle in value)))
    unknown = sorted(set(result) - set(JOINT_NAMES))
    if unknown:
        raise ValueError(f"unknown encoder joint names: {unknown}")
    if any(not math.isfinite(angle) for angle in result.values()):
        raise ValueError("encoder angles must be finite")
    return result


class HousingPoseEstimator:
    """Fuse rigid marker observations into body and 18-joint pose estimates."""

    def __init__(
        self,
        *,
        cameras: Mapping[str, RigidTransform],
        tag_mounts: Mapping[int, TagMount],
        geometry: HexapodGeometry | None = None,
    ) -> None:
        if not cameras:
            raise ValueError("at least one camera transform is required")
        self.cameras = dict(cameras)
        self.tag_mounts = dict(tag_mounts)
        self.geometry = geometry or HexapodGeometry()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HousingPoseEstimator":
        camera_values = value.get("cameras", {
            "camera0": {"world_from_camera": {}}
        })
        cameras = {
            str(name): RigidTransform.from_dict(
                spec.get("world_from_camera", spec)
            )
            for name, spec in camera_values.items()
        }
        tag_mounts: dict[int, TagMount] = {}
        for raw_id, spec in value.get("tags", {}).items():
            tag_id = int(raw_id)
            if tag_id in tag_mounts:
                raise ValueError(f"duplicate tag id {tag_id}")
            frame = str(spec["frame"])
            _validate_frame_name(frame)
            tag_mounts[tag_id] = TagMount(
                tag_id=tag_id,
                frame=frame,
                frame_from_tag=RigidTransform.from_dict(
                    spec.get("frame_from_tag", {})
                ),
            )
        if not tag_mounts:
            raise ValueError("config must define at least one tag mount")
        return cls(
            cameras=cameras,
            tag_mounts=tag_mounts,
            geometry=HexapodGeometry.from_dict(value.get("geometry")),
        )

    def observations_from_detections(
        self,
        detections: Iterable[TagDetection | Mapping[str, Any]],
    ) -> tuple[dict[str, _FusedFrame], list[int]]:
        candidates: dict[str, list[_WeightedTransform]] = {}
        ignored: list[int] = []
        for raw in detections:
            detection = (
                raw if isinstance(raw, TagDetection) else TagDetection.from_dict(raw)
            )
            mount = self.tag_mounts.get(detection.tag_id)
            if mount is None:
                ignored.append(detection.tag_id)
                continue
            if detection.camera not in self.cameras:
                raise ValueError(
                    f"detection names unknown camera {detection.camera!r}"
                )
            world_from_tag = self.cameras[detection.camera].compose(
                detection.camera_from_tag
            )
            world_from_frame = world_from_tag.compose(
                mount.frame_from_tag.inverse()
            )
            candidates.setdefault(mount.frame, []).append(_WeightedTransform(
                transform=world_from_frame,
                weight=detection.weight,
                tag_id=detection.tag_id,
            ))
        return (
            {name: _fuse_transforms(values) for name, values in candidates.items()},
            sorted(ignored),
        )

    def estimate_detections(
        self,
        detections: Iterable[TagDetection | Mapping[str, Any]],
        *,
        encoder_joint_deg: Sequence[float] | Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        frames, ignored = self.observations_from_detections(detections)
        result = self._estimate_fused_frames(
            frames, encoder_joint_deg=encoder_joint_deg
        )
        result["ignored_unknown_tag_ids"] = ignored
        return result

    def estimate_frame_transforms(
        self,
        frames: Mapping[str, RigidTransform | Mapping[str, Any]],
        *,
        encoder_joint_deg: Sequence[float] | Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        """Estimate from tracker-provided world-frame link transforms."""
        fused: dict[str, _FusedFrame] = {}
        for name, raw_transform in frames.items():
            _validate_frame_name(name)
            transform = (
                raw_transform
                if isinstance(raw_transform, RigidTransform)
                else RigidTransform.from_dict(raw_transform)
            )
            fused[name] = _fuse_transforms([_WeightedTransform(transform, 1.0)])
        return self._estimate_fused_frames(
            fused, encoder_joint_deg=encoder_joint_deg
        )

    def _estimate_fused_frames(
        self,
        frames: Mapping[str, _FusedFrame],
        *,
        encoder_joint_deg: Sequence[float] | Mapping[str, float] | None,
    ) -> dict[str, Any]:
        encoder = _normalise_encoder_angles(encoder_joint_deg)
        frame_diagnostics = {
            name: frame.diagnostics() for name, frame in sorted(frames.items())
        }
        body_observation = frames.get("body")
        if body_observation is None:
            return {
                "ok": False,
                "complete": False,
                "error": (
                    "body frame is unobserved; configure at least one tag fixed "
                    "to the chassis (a yaw-servo housing is suitable)"
                ),
                "frame_diagnostics": frame_diagnostics,
                "unobservable_joints": list(JOINT_NAMES),
            }

        world_from_body = body_observation.transform
        body_from_world_rotation = world_from_body.rotation.inv()
        joints: dict[str, dict[str, Any]] = {}
        joint_radians: dict[str, float | None] = {}
        per_leg: list[dict[str, Any]] = []

        for leg in range(LEG_COUNT):
            azimuth = (leg + 0.5) * math.pi / 3.0
            nominal_from_body = _rz(-azimuth)
            observations: dict[str, tuple[float, float, float, _FusedFrame]] = {}
            for segment in ("coxa", "femur", "tibia"):
                frame_name = f"L{leg}_{segment}"
                observed = frames.get(frame_name)
                if observed is None:
                    continue
                body_from_segment_rotation = (
                    body_from_world_rotation * observed.transform.rotation
                )
                leg_from_segment_rotation = (
                    nominal_from_body * body_from_segment_rotation
                )
                yaw, plane_angle, residual = _decompose_zy(
                    leg_from_segment_rotation
                )
                observations[segment] = (yaw, plane_angle, residual, observed)

            yaw_candidates: list[tuple[float, float]] = []
            for _segment, (yaw, _plane, residual, observed) in observations.items():
                residual_scale = math.radians(2.0)
                weight = observed.weight / (1.0 + (residual / residual_scale) ** 2)
                yaw_candidates.append((yaw, weight))

            if yaw_candidates:
                yaw = _circular_mean(yaw_candidates)
                total_weight = sum(weight for _angle, weight in yaw_candidates)
                yaw_residual = math.sqrt(sum(
                    weight * _angle_distance(angle, yaw) ** 2
                    for angle, weight in yaw_candidates
                ) / total_weight)
                axis_residual = math.sqrt(sum(
                    weight * observations[segment][2] ** 2
                    for segment, (_angle, weight) in zip(
                        observations.keys(), yaw_candidates
                    )
                ) / total_weight)
                yaw_residual = math.hypot(yaw_residual, axis_residual)
            else:
                yaw = None
                yaw_residual = None

            femur = observations.get("femur")
            tibia = observations.get("tibia")
            hip = None if femur is None else femur[1]
            hip_residual = None if femur is None else femur[2]
            knee = None if tibia is None else tibia[1]
            knee_residual = None if tibia is None else tibia[2]

            values = {
                "yaw": (yaw, yaw_residual, tuple(
                    f"L{leg}_{segment}" for segment in observations
                )),
                "hip": (hip, hip_residual, (
                    () if femur is None else (f"L{leg}_femur",)
                )),
                "knee": (knee, knee_residual, (
                    () if tibia is None else (f"L{leg}_tibia",)
                )),
            }
            leg_records: dict[str, Any] = {"leg": leg}
            for axis, (angle, residual, sources) in values.items():
                name = f"L{leg}_{axis}"
                record = _joint_record(
                    angle, residual_rad=residual, sources=sources
                )
                if angle is not None and name in encoder:
                    delta = math.degrees(_angle_distance(
                        angle, math.radians(encoder[name])
                    ))
                    record["encoder_deg"] = round(encoder[name], 5)
                    record["visual_minus_encoder_deg"] = round(delta, 5)
                joints[name] = record
                joint_radians[name] = angle
                leg_records[axis] = record
            per_leg.append(leg_records)

        unobservable = [
            name for name in JOINT_NAMES if joint_radians[name] is None
        ]
        feet = self._foot_positions(world_from_body, joint_radians)
        zero_offsets = {
            name: joints[name]["visual_minus_encoder_deg"]
            for name in JOINT_NAMES
            if "visual_minus_encoder_deg" in joints[name]
        }
        notes: list[str] = []
        missing_knees = [f"L{leg}_knee" for leg in range(LEG_COUNT)
                         if joint_radians[f"L{leg}_knee"] is None]
        if missing_knees:
            notes.append(
                "Knee-housing tags observe the femur, not the tibia. Add one "
                "rigid tibia tag per missing knee to make those angles observable."
            )
        if zero_offsets:
            notes.append(
                "visual_minus_encoder_deg values are diagnostics only; review "
                "several stationary frames before changing servo zeros."
            )

        vector = [
            None if joint_radians[name] is None
            else round(math.degrees(float(joint_radians[name])), 5)
            for name in JOINT_NAMES
        ]
        return {
            "ok": True,
            "complete": not unobservable,
            "angle_convention": "absolute_tibia",
            "body_pose": {
                "world_from_body": world_from_body.to_dict(),
                "euler_xyz_deg": [
                    round(float(value), 5)
                    for value in world_from_body.rotation.as_euler(
                        "xyz", degrees=True
                    )
                ],
            },
            "joint_order": list(JOINT_NAMES),
            "joint_vector_deg": vector,
            "joints": joints,
            "per_leg": per_leg,
            "feet_world_m": feet,
            "visual_minus_encoder_deg": zero_offsets,
            "unobservable_joints": unobservable,
            "frame_diagnostics": frame_diagnostics,
            "notes": notes,
        }

    def _foot_positions(
        self,
        world_from_body: RigidTransform,
        joints: Mapping[str, float | None],
    ) -> list[list[float] | None]:
        geometry = self.geometry
        feet: list[list[float] | None] = []
        for leg in range(LEG_COUNT):
            yaw = joints[f"L{leg}_yaw"]
            hip = joints[f"L{leg}_hip"]
            knee = joints[f"L{leg}_knee"]
            if yaw is None or hip is None or knee is None:
                feet.append(None)
                continue
            azimuth = (leg + 0.5) * math.pi / 3.0
            leg_rotation = _rz(azimuth + yaw)
            yaw_origin = np.array([
                geometry.chassis_apothem_m * math.cos(azimuth),
                geometry.chassis_apothem_m * math.sin(azimuth),
                0.0,
            ])
            hip_origin = yaw_origin + leg_rotation.apply([
                geometry.coxa_m,
                geometry.hip_anchor_y_m,
                0.0,
            ])
            knee_origin = hip_origin + (leg_rotation * _ry(hip)).apply([
                geometry.femur_m, 0.0, 0.0
            ])
            foot_body = knee_origin + (leg_rotation * _ry(knee)).apply([
                geometry.tibia_m, 0.0, 0.0
            ])
            foot_world = world_from_body.apply(foot_body)
            feet.append([round(float(value), 7) for value in foot_world])
        return feet


def _validate_frame_name(name: str) -> None:
    if name == "body":
        return
    valid = {
        f"L{leg}_{segment}"
        for leg in range(LEG_COUNT)
        for segment in ("coxa", "femur", "tibia")
    }
    if name not in valid:
        raise ValueError(
            f"unknown frame {name!r}; expected body or L0..L5_coxa/femur/tibia"
        )


def forward_frame_transforms(
    world_from_body: RigidTransform,
    joint_angles_deg: Sequence[float] | Mapping[str, float],
    *,
    geometry: HexapodGeometry | None = None,
) -> dict[str, RigidTransform]:
    """Generate ideal tracked-frame poses; useful for tests and replay tools."""
    angles = _normalise_encoder_angles(joint_angles_deg)
    missing = [name for name in JOINT_NAMES if name not in angles]
    if missing:
        raise ValueError(f"joint angle input is missing {missing}")
    geometry = geometry or HexapodGeometry()
    result = {"body": world_from_body}
    for leg in range(LEG_COUNT):
        azimuth = (leg + 0.5) * math.pi / 3.0
        yaw = math.radians(angles[f"L{leg}_yaw"])
        hip = math.radians(angles[f"L{leg}_hip"])
        knee = math.radians(angles[f"L{leg}_knee"])
        yaw_rotation = _rz(azimuth + yaw)
        yaw_origin = np.array([
            geometry.chassis_apothem_m * math.cos(azimuth),
            geometry.chassis_apothem_m * math.sin(azimuth),
            0.0,
        ])
        body_from_coxa = RigidTransform(yaw_origin, yaw_rotation)
        hip_origin = yaw_origin + yaw_rotation.apply([
            geometry.coxa_m, geometry.hip_anchor_y_m, 0.0
        ])
        body_from_femur = RigidTransform(
            hip_origin, yaw_rotation * _ry(hip)
        )
        knee_origin = hip_origin + body_from_femur.rotation.apply([
            geometry.femur_m, 0.0, 0.0
        ])
        body_from_tibia = RigidTransform(
            knee_origin, yaw_rotation * _ry(knee)
        )
        result[f"L{leg}_coxa"] = world_from_body.compose(body_from_coxa)
        result[f"L{leg}_femur"] = world_from_body.compose(body_from_femur)
        result[f"L{leg}_tibia"] = world_from_body.compose(body_from_tibia)
    return result
