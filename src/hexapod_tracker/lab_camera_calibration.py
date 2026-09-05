"""Guided intrinsic and floor-frame calibration for fixed lab cameras.

The handheld iPhone survey and the fixed observer cameras solve different
problems.  This module owns the latter: a new camera first observes a moving,
dimensioned AprilTag board to estimate its intrinsics, then observes the
permanent floor tags while the camera is stationary to estimate
``world_from_camera``.  A previously calibrated camera that was bumped reuses
its same-mode intrinsics and only repeats the floor alignment.

Everything here is camera-only.  It does not import a robot adapter and never
sends a motor command.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import threading
import time
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .apriltag_vision import (
    CameraCalibration,
    TagCorners,
    WorldReference,
    estimate_world_reference,
    marker_object_corners,
)
from .housing_pose import RigidTransform
from .robot_lab import RobotLabPublisher


DEFAULT_INTRINSIC_VIEWS = 15
DEFAULT_EXTRINSIC_FRAMES = 12
MAX_INTRINSIC_RMS_PX = 1.0
MAX_INTRINSIC_VIEW_RMS_PX = 1.5
MAX_EXTRINSIC_REPROJECTION_RMS_PX = 1.25
MAX_EXTRINSIC_TRANSLATION_SPREAD_MM = 6.0
MAX_EXTRINSIC_ROTATION_SPREAD_DEG = 0.8
MOVED_TRANSLATION_MM = 15.0
MOVED_ROTATION_DEG = 1.5

_CAMERA_ID = re.compile(r"^[A-Za-z0-9._:@+-]{1,160}$")


class LabCameraCalibrationError(ValueError):
    """A camera-calibration observation or session is not usable."""


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _transform_map(value: Mapping[str, Any]) -> dict[int, RigidTransform]:
    return {
        int(raw_id): RigidTransform.from_dict(
            spec.get("world_from_tag", spec)
        )
        for raw_id, spec in value.items()
    }


def load_floor_layout(
    layout_path: Path,
) -> tuple[dict[int, RigidTransform], float, dict[str, Any]]:
    """Load the permanent floor map from the photographed robot layout."""
    layout = _load_json(layout_path)
    floor = dict(layout.get("floor") or {})
    raw_tags = floor.get("tags") or []
    tags = {
        int(item["id"]): RigidTransform.from_dict(item["world_from_tag"])
        for item in raw_tags
    }
    if len(tags) < 2:
        raise ValueError("lab floor layout must contain at least two tags")
    marker_size_m = float(
        (layout.get("tag_geometry") or {}).get("black_square_m", 0.0272)
    )
    marker_object_corners(marker_size_m)
    return tags, marker_size_m, floor


def load_intrinsic_board(
    manifest_path: Path,
) -> tuple[dict[int, RigidTransform], float, dict[str, Any]]:
    manifest = _load_json(manifest_path)
    tags = _transform_map(manifest.get("floor_tags") or {})
    if len(tags) < 4:
        raise ValueError("intrinsic board must contain at least four tags")
    marker_size_m = float(manifest["marker_size_m"])
    marker_object_corners(marker_size_m)
    return tags, marker_size_m, manifest


@dataclass(frozen=True)
class IntrinsicObservation:
    object_points: np.ndarray
    image_points: np.ndarray
    image_size_px: tuple[int, int]
    tag_ids: tuple[int, ...]
    # center x/y, relative diagonal, in-plane angle, two perspective ratios
    descriptor: tuple[float, float, float, float, float, float]
    sharpness: float


@dataclass(frozen=True)
class IntrinsicCalibrationResult:
    calibration: CameraCalibration
    input_views: int
    accepted_views: int
    rejected_views: int
    rms_px: float
    median_view_rms_px: float
    max_view_rms_px: float
    center_span_x: float
    center_span_y: float
    scale_ratio: float
    viewpoint_spread: float
    failing_checks: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failing_checks

    def to_dict(self) -> dict[str, Any]:
        camera = self.calibration
        return {
            "camera": {
                "image_size_px": list(camera.image_size_px),
                "camera_matrix": [
                    [round(float(value), 9) for value in row]
                    for row in camera.camera_matrix
                ],
                "distortion_coefficients": [
                    round(float(value), 12)
                    for value in camera.distortion_coefficients
                ],
                "approximate": False,
                "allow_center_crop": False,
                "allow_quarter_turn": False,
                "source": "moving dimensioned AprilTag board",
            },
            "quality": {
                "passed": self.passed,
                "input_views": self.input_views,
                "accepted_views": self.accepted_views,
                "rejected_views": self.rejected_views,
                "rms_px": round(self.rms_px, 5),
                "median_view_rms_px": round(self.median_view_rms_px, 5),
                "max_view_rms_px": round(self.max_view_rms_px, 5),
                "center_span_x": round(self.center_span_x, 5),
                "center_span_y": round(self.center_span_y, 5),
                "scale_ratio": round(self.scale_ratio, 5),
                "viewpoint_spread": round(self.viewpoint_spread, 5),
                "failing_checks": list(self.failing_checks),
            },
        }


def make_intrinsic_observation(
    detections: Sequence[TagCorners],
    board_tags: Mapping[int, RigidTransform],
    *,
    marker_size_m: float,
    image_size_px: tuple[int, int],
    sharpness: float = 100.0,
    minimum_tags: int = 3,
) -> IntrinsicObservation:
    """Build one moving-board view for OpenCV intrinsic calibration."""
    visible = [item for item in detections if item.tag_id in board_tags]
    if len(visible) < minimum_tags:
        raise LabCameraCalibrationError(
            f"see at least {minimum_tags} calibration-board tags together"
        )
    width, height = image_size_px
    if min(width, height) <= 0:
        raise ValueError("image_size_px must be positive")
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    for detection in visible:
        board_from_tag = board_tags[detection.tag_id]
        object_points.extend(
            board_from_tag.apply(point)
            for point in marker_object_corners(marker_size_m)
        )
        image_points.extend(np.asarray(detection.corners_px, dtype=float))
    pixels = np.asarray(image_points, dtype=np.float32)
    low = np.min(pixels, axis=0)
    high = np.max(pixels, axis=0)
    center = (low + high) / 2.0
    diagonal = float(np.linalg.norm(high - low)) / math.hypot(width, height)
    first = visible[0].corners_px
    top = float(np.linalg.norm(first[1] - first[0]))
    right = float(np.linalg.norm(first[2] - first[1]))
    bottom = float(np.linalg.norm(first[3] - first[2]))
    left = float(np.linalg.norm(first[0] - first[3]))
    top_edge = first[1] - first[0]
    descriptor = (
        float(center[0] / width),
        float(center[1] / height),
        diagonal,
        math.atan2(float(top_edge[1]), float(top_edge[0])),
        math.log(max(top, 1e-6) / max(bottom, 1e-6)),
        math.log(max(left, 1e-6) / max(right, 1e-6)),
    )
    return IntrinsicObservation(
        object_points=np.asarray(object_points, dtype=np.float32),
        image_points=pixels,
        image_size_px=(int(width), int(height)),
        tag_ids=tuple(sorted(item.tag_id for item in visible)),
        descriptor=descriptor,
        sharpness=float(sharpness),
    )


def intrinsic_view_is_novel(
    candidate: IntrinsicObservation,
    previous: Sequence[IntrinsicObservation],
) -> bool:
    """Reject video duplicates while retaining distinct board poses."""
    if not previous:
        return True
    value = np.asarray(candidate.descriptor, dtype=float)
    for item in previous:
        other = np.asarray(item.descriptor, dtype=float)
        angle_delta = abs((value[3] - other[3] + math.pi) % (2 * math.pi) - math.pi)
        novelty = max(
            float(np.linalg.norm(value[:2] - other[:2])) / 0.10,
            abs(math.log(max(value[2], 1e-6) / max(other[2], 1e-6))) / 0.12,
            angle_delta / math.radians(12.0),
            float(np.linalg.norm(value[4:] - other[4:])) / 0.10,
        )
        if novelty < 1.0:
            return False
    return True


def _run_intrinsic_calibration(
    observations: Sequence[IntrinsicObservation],
) -> tuple[float, np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray], np.ndarray]:
    image_sizes = {item.image_size_px for item in observations}
    if len(image_sizes) != 1:
        raise LabCameraCalibrationError(
            "camera resolution changed during intrinsic calibration"
        )
    size = observations[0].image_size_px
    rms, matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
        [item.object_points for item in observations],
        [item.image_points for item in observations],
        size,
        None,
        None,
    )
    per_view = []
    for item, rvec, tvec in zip(observations, rvecs, tvecs):
        projected, _ = cv2.projectPoints(
            item.object_points, rvec, tvec, matrix, distortion
        )
        error = projected.reshape(-1, 2) - item.image_points
        per_view.append(math.sqrt(float(np.mean(np.sum(error * error, axis=1)))))
    return (
        float(rms),
        np.asarray(matrix, dtype=float),
        np.asarray(distortion, dtype=float).reshape(-1),
        list(rvecs),
        list(tvecs),
        np.asarray(per_view, dtype=float),
    )


def calibrate_camera_intrinsics(
    observations: Sequence[IntrinsicObservation],
    *,
    minimum_views: int = 12,
) -> IntrinsicCalibrationResult:
    """Robustly estimate one lens/resolution intrinsic profile."""
    if len(observations) < minimum_views:
        raise LabCameraCalibrationError(
            f"only {len(observations)} distinct board views; need {minimum_views}"
        )
    first = _run_intrinsic_calibration(observations)
    per_view = first[-1]
    median = float(np.median(per_view))
    mad = float(np.median(np.abs(per_view - median)))
    cutoff = max(MAX_INTRINSIC_VIEW_RMS_PX, median + max(0.15, 4.0 * 1.4826 * mad))
    keep = per_view <= cutoff
    accepted = [
        item for item, include in zip(observations, keep) if bool(include)
    ]
    if len(accepted) < minimum_views:
        raise LabCameraCalibrationError(
            "too many blurry or inconsistent board views; collect cleaner angles"
        )
    rms, matrix, distortion, _rvecs, _tvecs, per_view = (
        first if len(accepted) == len(observations)
        else _run_intrinsic_calibration(accepted)
    )
    descriptors = np.asarray([item.descriptor for item in accepted], dtype=float)
    center_span_x = float(np.ptp(descriptors[:, 0]))
    center_span_y = float(np.ptp(descriptors[:, 1]))
    scale_ratio = float(
        np.max(descriptors[:, 2]) / max(1e-9, np.min(descriptors[:, 2]))
    )
    angle_span = float(np.ptp(np.unwrap(descriptors[:, 3])))
    perspective_span = float(np.linalg.norm(np.ptp(descriptors[:, 4:], axis=0)))
    viewpoint_spread = max(angle_span, perspective_span)
    width, height = accepted[0].image_size_px
    fx, fy = float(matrix[0, 0]), float(matrix[1, 1])
    cx, cy = float(matrix[0, 2]), float(matrix[1, 2])
    max_dimension = max(width, height)
    failing: list[str] = []
    if rms > MAX_INTRINSIC_RMS_PX:
        failing.append(
            f"intrinsic reprojection RMS is {rms:.2f} px; need <= {MAX_INTRINSIC_RMS_PX:.2f} px"
        )
    if float(np.max(per_view)) > MAX_INTRINSIC_VIEW_RMS_PX:
        failing.append("one or more board views still has excessive corner error")
    if center_span_x < 0.25 or center_span_y < 0.20:
        failing.append("move the board into more image corners")
    if scale_ratio < 1.35:
        failing.append("include both close and far board views")
    if viewpoint_spread < 0.16:
        failing.append("tilt and rotate the board through more angles")
    if not (0.25 * width <= cx <= 0.75 * width) or not (
        0.25 * height <= cy <= 0.75 * height
    ):
        failing.append("estimated optical center is implausible")
    if not (0.3 * max_dimension <= fx <= 5.0 * max_dimension) or not (
        0.3 * max_dimension <= fy <= 5.0 * max_dimension
    ) or not (0.7 <= fx / fy <= 1.3):
        failing.append("estimated focal lengths are implausible")
    calibration = CameraCalibration(
        image_size_px=accepted[0].image_size_px,
        camera_matrix=matrix,
        distortion_coefficients=distortion,
        approximate=False,
        allow_center_crop=False,
        allow_quarter_turn=False,
    )
    return IntrinsicCalibrationResult(
        calibration=calibration,
        input_views=len(observations),
        accepted_views=len(accepted),
        rejected_views=len(observations) - len(accepted),
        rms_px=rms,
        median_view_rms_px=float(np.median(per_view)),
        max_view_rms_px=float(np.max(per_view)),
        center_span_x=center_span_x,
        center_span_y=center_span_y,
        scale_ratio=scale_ratio,
        viewpoint_spread=viewpoint_spread,
        failing_checks=tuple(failing),
    )


@dataclass(frozen=True)
class FixedCameraCalibrationResult:
    world_from_camera: RigidTransform
    input_frames: int
    accepted_frames: int
    rejected_frames: int
    translation_spread_mm: float
    rotation_spread_deg: float
    median_reprojection_rms_px: float
    floor_tag_ids: tuple[int, ...]
    failing_checks: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failing_checks

    def to_dict(self) -> dict[str, Any]:
        return {
            "world_from_camera": self.world_from_camera.to_dict(),
            "quality": {
                "passed": self.passed,
                "input_frames": self.input_frames,
                "accepted_frames": self.accepted_frames,
                "rejected_frames": self.rejected_frames,
                "translation_spread_mm": round(self.translation_spread_mm, 4),
                "rotation_spread_deg": round(self.rotation_spread_deg, 5),
                "median_reprojection_rms_px": round(
                    self.median_reprojection_rms_px, 5
                ),
                "floor_tag_ids": list(self.floor_tag_ids),
                "failing_checks": list(self.failing_checks),
            },
        }


def make_fixed_camera_observation(
    detections: Sequence[TagCorners],
    floor_tags: Mapping[int, RigidTransform],
    calibration: CameraCalibration,
    *,
    image_size_px: tuple[int, int],
    marker_size_m: float,
    minimum_tags: int = 2,
) -> WorldReference:
    visible = [item for item in detections if item.tag_id in floor_tags]
    if len(visible) < minimum_tags:
        raise LabCameraCalibrationError(
            f"see at least {minimum_tags} permanent floor tags together"
        )
    matrix, distortion = calibration.for_image(*image_size_px)
    reference = estimate_world_reference(
        visible,
        floor_tags,
        matrix,
        distortion,
        marker_size_m=marker_size_m,
    )
    if reference is None:
        raise LabCameraCalibrationError("could not solve the floor-tag camera pose")
    return reference


def average_fixed_camera_extrinsics(
    observations: Sequence[WorldReference],
    *,
    minimum_frames: int = 10,
) -> FixedCameraCalibrationResult:
    """Average repeated floor solves and reject frames captured while moving."""
    if len(observations) < minimum_frames:
        raise LabCameraCalibrationError(
            f"only {len(observations)} floor-aligned frames; need {minimum_frames}"
        )
    translations = np.stack([
        item.world_from_camera.translation_m for item in observations
    ])
    rotations = Rotation.from_quat(np.stack([
        item.world_from_camera.rotation.as_quat() for item in observations
    ]))
    center_translation = np.median(translations, axis=0)
    center_rotation = rotations.mean()
    translation_error = np.linalg.norm(translations - center_translation, axis=1)
    rotation_error = np.degrees((center_rotation.inv() * rotations).magnitude())
    reprojection = np.asarray([
        item.reprojection_rms_px for item in observations
    ], dtype=float)
    keep = (
        (translation_error <= 0.025)
        & (rotation_error <= 2.0)
        & (reprojection <= 2.0)
    )
    if int(np.count_nonzero(keep)) < minimum_frames:
        raise LabCameraCalibrationError(
            "camera moved or floor solves disagreed; keep the mount still and retry"
        )
    accepted = [
        item for item, include in zip(observations, keep) if bool(include)
    ]
    weights = np.asarray([
        1.0 / max(0.20, item.reprojection_rms_px) ** 2 for item in accepted
    ])
    translations = np.stack([
        item.world_from_camera.translation_m for item in accepted
    ])
    rotations = Rotation.from_quat(np.stack([
        item.world_from_camera.rotation.as_quat() for item in accepted
    ]))
    translation = np.average(translations, axis=0, weights=weights)
    rotation = rotations.mean(weights=weights)
    translation_spread = math.sqrt(float(np.average(
        np.linalg.norm(translations - translation, axis=1) ** 2,
        weights=weights,
    ))) * 1000.0
    rotation_spread = math.sqrt(float(np.average(
        np.degrees((rotation.inv() * rotations).magnitude()) ** 2,
        weights=weights,
    )))
    median_reprojection = float(np.median([
        item.reprojection_rms_px for item in accepted
    ]))
    tag_ids = tuple(sorted({
        tag_id for item in accepted for tag_id in item.floor_tag_ids
    }))
    failing: list[str] = []
    if len(tag_ids) < 3:
        failing.append("observe at least three different permanent floor tags")
    if median_reprojection > MAX_EXTRINSIC_REPROJECTION_RMS_PX:
        failing.append(
            f"floor reprojection RMS is {median_reprojection:.2f} px; "
            f"need <= {MAX_EXTRINSIC_REPROJECTION_RMS_PX:.2f} px"
        )
    if translation_spread > MAX_EXTRINSIC_TRANSLATION_SPREAD_MM:
        failing.append(
            f"camera position spread is {translation_spread:.1f} mm; "
            "tighten the mount and improve the floor view"
        )
    if rotation_spread > MAX_EXTRINSIC_ROTATION_SPREAD_DEG:
        failing.append(
            f"camera angle spread is {rotation_spread:.2f} degrees; "
            "keep the camera still"
        )
    return FixedCameraCalibrationResult(
        world_from_camera=RigidTransform(translation, rotation),
        input_frames=len(observations),
        accepted_frames=len(accepted),
        rejected_frames=len(observations) - len(accepted),
        translation_spread_mm=translation_spread,
        rotation_spread_deg=rotation_spread,
        median_reprojection_rms_px=median_reprojection,
        floor_tag_ids=tag_ids,
        failing_checks=tuple(failing),
    )


def _rotation_distance_deg(first: RigidTransform, second: RigidTransform) -> float:
    return math.degrees(float((first.rotation.inv() * second.rotation).magnitude()))


class LabCameraCalibrationManager:
    """State machine used by the local web UI for fixed observer cameras."""

    def __init__(
        self,
        *,
        board_manifest_path: Path,
        board_svg_path: Path,
        floor_layout_path: Path,
        output_dir: Path,
        robot_lab: RobotLabPublisher | None = None,
        intrinsic_views: int = DEFAULT_INTRINSIC_VIEWS,
        extrinsic_frames: int = DEFAULT_EXTRINSIC_FRAMES,
    ) -> None:
        self.board_manifest_path = Path(board_manifest_path)
        self.board_svg_path = Path(board_svg_path)
        self.floor_layout_path = Path(floor_layout_path)
        self.output_dir = Path(output_dir)
        self.robot_lab = robot_lab or RobotLabPublisher.from_env()
        self.intrinsic_views = int(intrinsic_views)
        self.extrinsic_frames = int(extrinsic_frames)
        if self.intrinsic_views < 8 or self.extrinsic_frames < 8:
            raise ValueError("camera calibration needs at least eight observations")
        (
            self.board_tags,
            self.board_marker_size_m,
            self.board_manifest,
        ) = load_intrinsic_board(self.board_manifest_path)
        (
            self.floor_tags,
            self.floor_marker_size_m,
            self.floor_layout,
        ) = load_floor_layout(self.floor_layout_path)
        self.floor_map_sha256 = hashlib.sha256(
            json.dumps(
                self.floor_layout, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        self._lock = threading.RLock()
        self._intrinsic_observations: list[IntrinsicObservation] = []
        self._extrinsic_observations: list[WorldReference] = []
        self._intrinsic_result: IntrinsicCalibrationResult | None = None
        self._report_path: Path | None = None
        self._last_observation_monotonic = 0.0
        self._active_camera: dict[str, Any] = {}
        self._verification_samples: deque[tuple[float, float, float]] = deque(
            maxlen=12
        )
        self._verification_camera_id: str | None = None
        self._state = self._initial_state()

    def _initial_state(self) -> dict[str, Any]:
        return {
            "status": "idle",
            "stage": "setup",
            "mode": None,
            "camera_id": None,
            "camera_name": None,
            "message": "Select a fixed lab camera to calibrate.",
            "instruction": (
                "Use New camera for intrinsics plus placement, or Camera moved "
                "to reuse intrinsics and measure only placement."
            ),
            "accepted_frames": 0,
            "target_frames": 0,
            "rejected_frames": 0,
            "last_rejection": None,
            "quality": None,
            "report_available": False,
            "report_path": None,
            "robot_lab": {
                "status": (
                    "ready" if self.robot_lab.configured else "not_configured"
                ),
                "url": None,
                "error": (
                    None if self.robot_lab.configured
                    else self.robot_lab.credential_error
                ),
                "credential_source": self.robot_lab.credential_source,
            },
            "verification": {
                "status": "unavailable",
                "translation_delta_mm": None,
                "rotation_delta_deg": None,
                "reprojection_rms_px": None,
                "sample_count": 0,
                "message": "No active calibration profile for this camera.",
            },
        }

    @staticmethod
    def _copy(value: Any) -> Any:
        return json.loads(json.dumps(value))

    @staticmethod
    def _validate_camera_id(value: str) -> str:
        camera_id = value.strip()
        if not _CAMERA_ID.fullmatch(camera_id):
            raise ValueError(
                "camera_id must be 1-160 letters, numbers, or . _ : @ + -"
            )
        return camera_id

    @staticmethod
    def _profile_key(camera_id: str) -> str:
        return hashlib.sha256(camera_id.encode("utf-8")).hexdigest()[:24]

    def _latest_path(self, camera_id: str) -> Path:
        return self.output_dir / self._profile_key(camera_id) / "latest.json"

    def latest_profile(self, camera_id: str) -> dict[str, Any] | None:
        try:
            path = self._latest_path(self._validate_camera_id(camera_id))
            return _load_json(path) if path.is_file() else None
        except (OSError, ValueError, json.JSONDecodeError):
            return None

    def profile_for(
        self,
        camera: Mapping[str, Any],
        image_size_px: tuple[int, int],
    ) -> dict[str, Any] | None:
        raw_id = str(camera.get("stable_id") or "")
        if not raw_id:
            return None
        profile = self.latest_profile(raw_id)
        if profile is None:
            return None
        stored_size = tuple(
            int(value)
            for value in (profile.get("intrinsics") or {}).get(
                "image_size_px", []
            )
        )
        return profile if stored_size == image_size_px else None

    def public_state(self, camera: Mapping[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            current = dict(camera or self._active_camera)
            camera_id = str(current.get("stable_id") or "")
            profile = self.latest_profile(camera_id) if camera_id else None
            state = self._copy(self._state)
            state["active_camera"] = current or None
            state["current_profile"] = (
                None if profile is None else {
                    "id": profile.get("id"),
                    "camera_id": profile.get("camera_id"),
                    "camera_name": profile.get("camera_name"),
                    "created_utc": profile.get("created_utc"),
                    "mode": profile.get("mode"),
                    "image_size_px": (
                        profile.get("intrinsics") or {}
                    ).get("image_size_px"),
                    "quality": profile.get("quality"),
                }
            )
            state["defaults"] = {
                "lab_id": "hexapod-lab",
                "intrinsic_board_tag_ids": sorted(self.board_tags),
                "floor_tag_ids": sorted(self.floor_tags),
                "intrinsic_views": self.intrinsic_views,
                "extrinsic_frames": self.extrinsic_frames,
            }
            return state

    def start(
        self,
        payload: Mapping[str, Any],
        *,
        camera: Mapping[str, Any],
        image_size_px: tuple[int, int],
    ) -> dict[str, Any]:
        with self._lock:
            if self._state["status"] == "collecting":
                raise RuntimeError("a lab camera calibration is already running")
            if not camera:
                raise RuntimeError("turn on and select the camera first")
            mode = str(payload.get("mode", "new"))
            if mode not in {"new", "moved"}:
                raise ValueError("mode must be new or moved")
            camera_id = self._validate_camera_id(str(
                payload.get("camera_id") or camera.get("stable_id") or ""
            ))
            camera_name = str(
                payload.get("camera_name") or camera.get("name") or camera_id
            ).strip()[:160]
            if not camera_name:
                raise ValueError("camera_name cannot be empty")
            self._intrinsic_observations = []
            self._extrinsic_observations = []
            self._intrinsic_result = None
            self._report_path = None
            self._last_observation_monotonic = 0.0
            if mode == "moved":
                existing = self.latest_profile(camera_id)
                if existing is None:
                    raise RuntimeError(
                        "no saved intrinsic profile exists for this camera; "
                        "choose New camera"
                    )
                intrinsics = existing.get("intrinsics") or {}
                if tuple(intrinsics.get("image_size_px") or ()) != image_size_px:
                    raise RuntimeError(
                        "the resolution changed; choose New camera to recalibrate intrinsics"
                    )
                calibration = CameraCalibration.from_dict(intrinsics)
                self._intrinsic_result = IntrinsicCalibrationResult(
                    calibration=calibration,
                    input_views=0,
                    accepted_views=0,
                    rejected_views=0,
                    rms_px=float(
                        (existing.get("quality") or {}).get(
                            "intrinsic_rms_px", 0.0
                        )
                    ),
                    median_view_rms_px=0.0,
                    max_view_rms_px=0.0,
                    center_span_x=0.0,
                    center_span_y=0.0,
                    scale_ratio=0.0,
                    viewpoint_spread=0.0,
                    failing_checks=(),
                )
                stage = "extrinsics"
                message = "Intrinsic profile reused. Align this camera to the floor."
                instruction = (
                    "Do not touch the camera. Clear the robot if necessary and "
                    "make at least three permanent floor tags visible."
                )
                target = self.extrinsic_frames
            else:
                stage = "intrinsics"
                message = "Calibrating this lens and resolution."
                instruction = (
                    "Hold the printed 2x2 calibration board in view, then move it "
                    "through the center, corners, close, far, and tilted angles."
                )
                target = self.intrinsic_views
            self._state = {
                **self._initial_state(),
                "status": "collecting",
                "stage": stage,
                "mode": mode,
                "lab_id": str(payload.get("lab_id") or "hexapod-lab")[:80],
                "camera_id": camera_id,
                "camera_name": camera_name,
                "camera_kind": str(camera.get("kind") or "camera"),
                "capture_mode": {
                    "image_size_px": list(image_size_px),
                    "capture_image_size_px": camera.get("capture_image_size_px"),
                    "fps": camera.get("capture_fps"),
                    "backend": camera.get("backend"),
                },
                "message": message,
                "instruction": instruction,
                "accepted_frames": 0,
                "target_frames": target,
                "started_unix": round(time.time(), 3),
            }
            self._active_camera = dict(camera)
            return self.public_state(camera)

    def cancel(self) -> dict[str, Any]:
        with self._lock:
            if self._state["status"] == "collecting":
                self._state["status"] = "cancelled"
                self._state["message"] = "Lab camera calibration cancelled."
                self._state["instruction"] = "Start again when the camera is ready."
            self._intrinsic_observations = []
            self._extrinsic_observations = []
            return self.public_state()

    @staticmethod
    def _detections(result: Mapping[str, Any]) -> list[TagCorners]:
        detections = []
        for item in result.get("detections") or []:
            if item.get("source") != "detected":
                continue
            corners = np.asarray(item.get("corners_px"), dtype=np.float32)
            if corners.shape == (4, 2):
                detections.append(TagCorners(int(item["tag_id"]), corners))
        return detections

    def _intrinsic_guidance(self) -> str:
        if not self._intrinsic_observations:
            return "Show tags 40–43 on the printed calibration board."
        descriptors = np.asarray([
            item.descriptor for item in self._intrinsic_observations
        ])
        centers = descriptors[:, :2]
        targets = [
            (0.18, 0.18, "upper-left"),
            (0.82, 0.18, "upper-right"),
            (0.18, 0.82, "lower-left"),
            (0.82, 0.82, "lower-right"),
            (0.50, 0.50, "center"),
        ]
        x, y, label = max(
            targets,
            key=lambda target: float(np.min(np.linalg.norm(
                centers - np.asarray(target[:2]), axis=1
            ))),
        )
        del x, y
        scale_ratio = float(
            np.max(descriptors[:, 2]) / max(1e-9, np.min(descriptors[:, 2]))
        )
        if scale_ratio < 1.35:
            return "Move the board noticeably closer, then farther away."
        if float(np.linalg.norm(np.ptp(descriptors[:, 4:], axis=0))) < 0.12:
            return "Tilt the board about 25 degrees; keep all four tags sharp."
        return f"Move the board toward the {label} and change its angle."

    def _record_intrinsic(
        self,
        image: np.ndarray,
        detections: Sequence[TagCorners],
        image_size_px: tuple[int, int],
        now: float,
    ) -> None:
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if sharpness < 35.0:
            raise LabCameraCalibrationError("board image is blurry; move more slowly")
        observation = make_intrinsic_observation(
            detections,
            self.board_tags,
            marker_size_m=self.board_marker_size_m,
            image_size_px=image_size_px,
            sharpness=sharpness,
        )
        if not intrinsic_view_is_novel(observation, self._intrinsic_observations):
            raise LabCameraCalibrationError(
                "view already captured; move or tilt the board farther"
            )
        self._intrinsic_observations.append(observation)
        self._last_observation_monotonic = now
        self._state["accepted_frames"] = len(self._intrinsic_observations)
        self._state["last_rejection"] = None
        self._state["instruction"] = self._intrinsic_guidance()
        if len(self._intrinsic_observations) < self.intrinsic_views:
            return
        result = calibrate_camera_intrinsics(
            self._intrinsic_observations,
            minimum_views=max(8, min(12, self.intrinsic_views)),
        )
        self._state["quality"] = result.to_dict()["quality"]
        if not result.passed:
            self._state["message"] = "More intrinsic views are needed."
            self._state["instruction"] = result.failing_checks[0]
            return
        self._intrinsic_result = result
        self._state.update({
            "stage": "extrinsics",
            "message": "Intrinsics passed. Now align the fixed camera to the lab.",
            "instruction": (
                "Mount the camera firmly, remove the handheld board, and show "
                "at least three permanent floor tags. Do not touch the camera."
            ),
            "accepted_frames": 0,
            "target_frames": self.extrinsic_frames,
            "rejected_frames": 0,
            "last_rejection": None,
            "quality": None,
        })

    def _record_extrinsic(
        self,
        detections: Sequence[TagCorners],
        image_size_px: tuple[int, int],
        now: float,
    ) -> None:
        assert self._intrinsic_result is not None
        reference = make_fixed_camera_observation(
            detections,
            self.floor_tags,
            self._intrinsic_result.calibration,
            image_size_px=image_size_px,
            marker_size_m=self.floor_marker_size_m,
        )
        if reference.reprojection_rms_px > 2.0:
            raise LabCameraCalibrationError(
                f"floor corner error is {reference.reprojection_rms_px:.2f} px; "
                "improve focus or viewing angle"
            )
        self._extrinsic_observations.append(reference)
        self._last_observation_monotonic = now
        self._state["accepted_frames"] = len(self._extrinsic_observations)
        self._state["last_rejection"] = None
        self._state["instruction"] = (
            f"Keep the camera still. Floor tags visible now: "
            f"{', '.join('#' + str(value) for value in reference.floor_tag_ids)}."
        )
        if len(self._extrinsic_observations) < self.extrinsic_frames:
            return
        result = average_fixed_camera_extrinsics(
            self._extrinsic_observations,
            minimum_frames=max(8, min(10, self.extrinsic_frames)),
        )
        self._state["quality"] = result.to_dict()["quality"]
        if not result.passed:
            self._state["message"] = "The fixed-camera placement is not stable yet."
            self._state["instruction"] = result.failing_checks[0]
            return
        self._complete(result)

    def process_frame(
        self,
        image: np.ndarray,
        result: Mapping[str, Any],
        *,
        camera: Mapping[str, Any],
    ) -> None:
        """Consume one already-detected frame and update calibration/verification."""
        image_size = tuple(int(value) for value in result.get("image_size_px", []))
        if len(image_size) != 2:
            return
        image_size_px = (image_size[0], image_size[1])
        detections = self._detections(result)
        now = time.monotonic()
        with self._lock:
            self._active_camera = dict(camera)
            if self._state["status"] != "collecting":
                self._verify_locked(detections, camera, image_size_px)
                return
            if str(camera.get("stable_id")) != self._state["camera_id"]:
                self._state["last_rejection"] = (
                    "selected camera changed; switch back to the calibration camera"
                )
                return
            if now - self._last_observation_monotonic < 0.22:
                return
            try:
                if self._state["stage"] == "intrinsics":
                    self._record_intrinsic(image, detections, image_size_px, now)
                elif self._state["stage"] == "extrinsics":
                    self._record_extrinsic(detections, image_size_px, now)
            except (LabCameraCalibrationError, cv2.error, ValueError) as error:
                self._state["rejected_frames"] += 1
                self._state["last_rejection"] = str(error)

    def _complete(self, extrinsics: FixedCameraCalibrationResult) -> None:
        assert self._intrinsic_result is not None
        intrinsic_dict = self._intrinsic_result.to_dict()
        extrinsic_dict = extrinsics.to_dict()
        identifier = hashlib.sha256(
            f"{self._state['camera_id']}:{time.time_ns()}".encode("utf-8")
        ).hexdigest()[:24]
        intrinsic_quality = intrinsic_dict["quality"]
        extrinsic_quality = extrinsic_dict["quality"]
        quality = {
            "passed": True,
            "intrinsic_rms_px": intrinsic_quality["rms_px"],
            "extrinsic_reprojection_rms_px": extrinsic_quality[
                "median_reprojection_rms_px"
            ],
            "translation_spread_mm": extrinsic_quality[
                "translation_spread_mm"
            ],
            "rotation_spread_deg": extrinsic_quality["rotation_spread_deg"],
            "floor_tag_ids": extrinsic_quality["floor_tag_ids"],
        }
        report = {
            "schema_version": 1,
            "id": identifier,
            "kind": "lab_camera_calibration",
            "lab_id": self._state.get("lab_id", "hexapod-lab"),
            "camera_id": self._state["camera_id"],
            "camera_name": self._state["camera_name"],
            "camera_kind": self._state["camera_kind"],
            "mode": self._state["mode"],
            "source": "hexapod_tracker_fixed_camera_workflow",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "capture_mode": self._state["capture_mode"],
            "intrinsics": intrinsic_dict["camera"],
            "extrinsics": {
                "coordinate_frame": "permanent_floor_grid",
                "floor_map_sha256": self.floor_map_sha256,
                "world_from_camera": extrinsic_dict["world_from_camera"],
            },
            "quality": quality,
            "evidence": {
                "intrinsics": intrinsic_quality,
                "extrinsics": extrinsic_quality,
                "intrinsic_board_tag_ids": sorted(self.board_tags),
                "floor_map": self.floor_layout,
            },
            "movement_detection": {
                "translation_threshold_mm": MOVED_TRANSLATION_MM,
                "rotation_threshold_deg": MOVED_ROTATION_DEG,
                "minimum_direct_floor_frames": 5,
            },
            "motor_commands_sent": False,
        }
        directory = self.output_dir / self._profile_key(report["camera_id"])
        directory.mkdir(parents=True, exist_ok=True)
        version_path = directory / f"{identifier}.json"
        encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
        version_path.write_text(encoded, encoding="utf-8")
        latest = directory / "latest.json"
        temporary = directory / f".latest.{identifier}.tmp"
        try:
            temporary.write_text(encoded, encoding="utf-8")
            temporary.replace(latest)
        finally:
            temporary.unlink(missing_ok=True)
        self._report_path = version_path
        self._state.update({
            "status": "complete",
            "stage": "review",
            "message": "Camera intrinsics and floor placement passed.",
            "instruction": "Robot Lab is saving this active camera revision.",
            "accepted_frames": extrinsics.accepted_frames,
            "target_frames": self.extrinsic_frames,
            "report_available": True,
            "report_path": str(version_path),
            "quality": quality,
            "completed_unix": round(time.time(), 3),
            "robot_lab": {
                "status": (
                    "publishing" if self.robot_lab.configured
                    else "not_configured"
                ),
                "url": None,
                "error": (
                    None if self.robot_lab.configured
                    else self.robot_lab.credential_error
                ),
                "credential_source": self.robot_lab.credential_source,
            },
        })
        if self.robot_lab.configured:
            threading.Thread(
                target=self._publish_current,
                name="robot-lab-camera-publish",
                daemon=True,
            ).start()

    def result(self) -> dict[str, Any] | None:
        with self._lock:
            if self._report_path is None or not self._report_path.is_file():
                return None
            try:
                return _load_json(self._report_path)
            except (OSError, ValueError, json.JSONDecodeError):
                return None

    def _publish_current(self) -> None:
        with self._lock:
            path = self._report_path
            if path is None:
                return
            self._state["robot_lab"] = {
                "status": "publishing",
                "url": None,
                "error": None,
                "credential_source": self.robot_lab.credential_source,
            }
        try:
            published = self.robot_lab.publish_lab_camera_calibration(path)
        except (OSError, RuntimeError, ValueError) as error:
            next_state = {
                "status": "failed",
                "url": None,
                "error": str(error),
                "credential_source": self.robot_lab.credential_source,
            }
        else:
            next_state = {
                **published,
                "error": None,
                "credential_source": self.robot_lab.credential_source,
            }
        with self._lock:
            self._state["robot_lab"] = next_state

    def publish(self) -> dict[str, Any]:
        if not self.robot_lab.configured:
            raise RuntimeError(
                self.robot_lab.credential_error
                or "Robot Lab token is not available to the vision server"
            )
        self._publish_current()
        with self._lock:
            return self._copy(self._state["robot_lab"])

    def _verify_locked(
        self,
        detections: Sequence[TagCorners],
        camera: Mapping[str, Any],
        image_size_px: tuple[int, int],
    ) -> None:
        camera_id = str(camera.get("stable_id") or "")
        if not camera_id:
            return
        if self._verification_camera_id != camera_id:
            self._verification_camera_id = camera_id
            self._verification_samples.clear()
        profile = self.latest_profile(camera_id)
        if profile is None:
            self._state["verification"] = {
                "status": "uncalibrated",
                "translation_delta_mm": None,
                "rotation_delta_deg": None,
                "reprojection_rms_px": None,
                "sample_count": 0,
                "message": "This camera has no saved calibration.",
            }
            return
        try:
            calibration = CameraCalibration.from_dict(profile["intrinsics"])
            reference = make_fixed_camera_observation(
                detections,
                self.floor_tags,
                calibration,
                image_size_px=image_size_px,
                marker_size_m=self.floor_marker_size_m,
            )
            expected = RigidTransform.from_dict(
                profile["extrinsics"]["world_from_camera"]
            )
        except (KeyError, ValueError, LabCameraCalibrationError, cv2.error):
            self._state["verification"] = {
                "status": "unverified",
                "translation_delta_mm": None,
                "rotation_delta_deg": None,
                "reprojection_rms_px": None,
                "sample_count": len(self._verification_samples),
                "message": "Show at least two permanent floor tags to verify placement.",
            }
            return
        translation = float(np.linalg.norm(
            reference.world_from_camera.translation_m - expected.translation_m
        )) * 1000.0
        rotation = _rotation_distance_deg(
            expected, reference.world_from_camera
        )
        self._verification_samples.append((
            translation, rotation, reference.reprojection_rms_px
        ))
        samples = np.asarray(self._verification_samples, dtype=float)
        median = np.median(samples, axis=0)
        enough = len(samples) >= 5
        moved = enough and (
            median[0] > MOVED_TRANSLATION_MM
            or median[1] > MOVED_ROTATION_DEG
        )
        status = "moved" if moved else "valid" if enough else "checking"
        message = (
            "Camera placement changed; recalibrate its floor alignment."
            if moved else
            "Camera placement matches its active calibration."
            if enough else
            "Checking this camera against the permanent floor tags."
        )
        self._state["verification"] = {
            "status": status,
            "translation_delta_mm": round(float(median[0]), 3),
            "rotation_delta_deg": round(float(median[1]), 4),
            "reprojection_rms_px": round(float(median[2]), 4),
            "sample_count": len(samples),
            "message": message,
        }
