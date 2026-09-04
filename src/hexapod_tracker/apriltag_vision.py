"""Calibrated AprilTag detection and camera/world pose helpers.

This module is deliberately read-only with respect to the robot.  It reads
images, detects tag36h11 markers, and returns rigid transforms; it contains no
robot networking or motor-control code.

Coordinate conventions
----------------------
OpenCV camera coordinates are x right, y down, z forward.  A tag's x axis is
corner 0 -> corner 1, y is corner 3 -> corner 0 (toward its decoded top), and
z is the printed face normal.  Transforms are named ``A_from_B`` and map B
coordinates into A coordinates, matching :mod:`housing_pose`.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .foot_tip_tracking import FootTipObservation, FootTipTracker, robust_tag_scale_px
from .housing_pose import HousingPoseEstimator, RigidTransform
from .joint_contract import FRAME_ROBOT_ABS, JOINT_CONTRACT


TAG_FAMILY = "tag36h11"
DEFAULT_MARKER_SIZE_M = 37.8968e-3  # black square on the repo's printed sheet
BRANCH_DISAMBIGUATION_MARGIN_DEG = 15.0
BODY_FLOOR_NORMAL_MIN_DOT = math.cos(math.radians(45.0))
BRANCH_JOINT_ORDER = tuple(
    f"L{leg}_{axis}"
    for leg in range(6)
    for axis in ("yaw", "hip", "knee")
)


def _make_apriltag_detector() -> Any:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    parameters = cv2.aruco.DetectorParameters()
    # OpenCV's APRILTAG refinement is roughly 3-5x slower than SUBPIX on the
    # iPhone stream while producing the same decoded set here. SUBPIX keeps
    # pose corners accurate without starving the preview/UI event loop.
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    # The floor references may be much smaller than the robot in a wide shot.
    parameters.minMarkerPerimeterRate = 0.005
    return cv2.aruco.ArucoDetector(dictionary, parameters)


_APRILTAG_DETECTOR = _make_apriltag_detector()


def _finite_array(value: Any, shape: tuple[int, ...], *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite array with shape {shape}")
    return array


@dataclass(frozen=True)
class CameraCalibration:
    """Pinhole calibration at one reference image resolution."""

    image_size_px: tuple[int, int]  # width, height
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    approximate: bool = False
    allow_center_crop: bool = False
    allow_quarter_turn: bool = False

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CameraCalibration":
        size = tuple(int(v) for v in value["image_size_px"])
        if len(size) != 2 or min(size) <= 0:
            raise ValueError("camera.image_size_px must be [width, height]")
        matrix = _finite_array(
            value["camera_matrix"], (3, 3), name="camera_matrix"
        )
        distortion = np.asarray(
            value.get("distortion_coefficients", []), dtype=float
        ).reshape(-1)
        if not np.all(np.isfinite(distortion)):
            raise ValueError("distortion coefficients must be finite")
        return cls(
            image_size_px=(size[0], size[1]),
            camera_matrix=matrix,
            distortion_coefficients=distortion,
            approximate=bool(value.get("approximate", False)),
            allow_center_crop=bool(value.get("allow_center_crop", False)),
            allow_quarter_turn=bool(value.get("allow_quarter_turn", False)),
        )

    def for_image(self, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        """Scale intrinsics to a same-aspect-ratio image."""
        ref_w, ref_h = self.image_size_px
        matrix = self.camera_matrix.copy()
        target_aspect = width / height
        original_error = abs(math.log(target_aspect / (ref_w / ref_h)))
        rotated_error = abs(math.log(target_aspect / (ref_h / ref_w)))
        if self.allow_quarter_turn and rotated_error < original_error:
            old = matrix.copy()
            ref_w, ref_h = ref_h, ref_w
            matrix = np.asarray([
                [old[1, 1], 0.0, self.image_size_px[1] - 1.0 - old[1, 2]],
                [0.0, old[0, 0], old[0, 2]],
                [0.0, 0.0, 1.0],
            ])
        sx, sy = width / ref_w, height / ref_h
        if not math.isclose(sx, sy, rel_tol=0.015, abs_tol=0.0):
            if not self.allow_center_crop:
                raise ValueError(
                    f"image is {width}x{height}, but calibration is "
                    f"{ref_w}x{ref_h}; the aspect ratios differ, so a "
                    "same-lens scale is unsafe"
                )
            scale = max(sx, sy)
            crop_x = (ref_w * scale - width) / 2.0
            crop_y = (ref_h * scale - height) / 2.0
            matrix[0, :] *= scale
            matrix[1, :] *= scale
            matrix[0, 2] -= crop_x
            matrix[1, 2] -= crop_y
        else:
            matrix[0, :] *= sx
            matrix[1, :] *= sy
        matrix[2, :] = [0.0, 0.0, 1.0]
        return matrix, self.distortion_coefficients.copy()


@dataclass(frozen=True)
class TagCorners:
    tag_id: int
    corners_px: np.ndarray  # decoded corner order, shape (4, 2)
    source: str = "detected"
    occlusion_age_frames: int = 0
    confidence: float = 1.0

    @property
    def center_px(self) -> np.ndarray:
        return np.mean(self.corners_px, axis=0)

    @property
    def tag_y_clockwise_from_image_up_deg(self) -> float:
        top = (self.corners_px[0] + self.corners_px[1]) / 2.0
        bottom = (self.corners_px[2] + self.corners_px[3]) / 2.0
        y_axis = top - bottom
        return math.degrees(math.atan2(float(y_axis[0]), float(-y_axis[1])))


class TemporalTagCornerTracker:
    """Carry decoded tag corners through brief decoder occlusions."""

    def __init__(self, *, max_occlusion_frames: int = 8) -> None:
        self.max_occlusion_frames = int(max_occlusion_frames)
        if self.max_occlusion_frames < 0:
            raise ValueError("max_occlusion_frames cannot be negative")
        self._previous_gray: np.ndarray | None = None
        self._previous: dict[int, TagCorners] = {}

    def reset(self) -> None:
        self._previous_gray = None
        self._previous = {}

    @staticmethod
    def _valid_quad(old: np.ndarray, new: np.ndarray, shape: tuple[int, int]) -> bool:
        height, width = shape
        if not np.all(np.isfinite(new)):
            return False
        margin = 16.0
        if np.any(new[:, 0] < -margin) or np.any(new[:, 0] > width + margin):
            return False
        if np.any(new[:, 1] < -margin) or np.any(new[:, 1] > height + margin):
            return False
        old_area = abs(float(cv2.contourArea(old.astype(np.float32))))
        new_area = abs(float(cv2.contourArea(new.astype(np.float32))))
        if old_area < 20.0 or not 0.45 * old_area <= new_area <= 2.2 * old_area:
            return False
        return bool(cv2.isContourConvex(new.astype(np.float32)))

    def update(
        self, image: np.ndarray, detections: Sequence[TagCorners]
    ) -> list[TagCorners]:
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        current = {
            item.tag_id: TagCorners(
                item.tag_id,
                np.asarray(item.corners_px, dtype=np.float32),
                source="detected",
                occlusion_age_frames=0,
                confidence=1.0,
            )
            for item in detections
        }
        if self._previous_gray is not None:
            for tag_id, previous in self._previous.items():
                if tag_id in current or previous.occlusion_age_frames >= self.max_occlusion_frames:
                    continue
                old = previous.corners_px.astype(np.float32).reshape(-1, 1, 2)
                new, status, error = cv2.calcOpticalFlowPyrLK(
                    self._previous_gray, gray, old, None,
                    winSize=(31, 31), maxLevel=3,
                    criteria=(
                        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                        25,
                        0.01,
                    ),
                )
                if new is None or status is None or not np.all(status == 1):
                    continue
                back, back_status, _ = cv2.calcOpticalFlowPyrLK(
                    gray, self._previous_gray, new, None,
                    winSize=(31, 31), maxLevel=3,
                    criteria=(
                        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                        25,
                        0.01,
                    ),
                )
                if back is None or back_status is None or not np.all(back_status == 1):
                    continue
                new_points = new.reshape(4, 2).astype(np.float32)
                fb_error = float(np.max(np.linalg.norm(
                    back.reshape(4, 2) - old.reshape(4, 2), axis=1
                )))
                lk_error = 0.0 if error is None else float(np.max(error))
                if fb_error > 2.5 or lk_error > 45.0:
                    continue
                if not self._valid_quad(
                    previous.corners_px, new_points, gray.shape[:2]
                ):
                    continue
                age = previous.occlusion_age_frames + 1
                current[tag_id] = TagCorners(
                    tag_id,
                    new_points,
                    source="optical_flow",
                    occlusion_age_frames=age,
                    confidence=max(0.12, 0.72 ** age / (1.0 + fb_error)),
                )
        self._previous_gray = gray.copy()
        self._previous = current
        return [current[tag_id] for tag_id in sorted(current)]


@dataclass(frozen=True)
class TagPose:
    tag_id: int
    corners_px: np.ndarray
    camera_from_tag: RigidTransform
    reprojection_rms_px: float
    alternate_reprojection_rms_px: float | None


@dataclass(frozen=True)
class _TagPoseSolution:
    camera_from_tag: RigidTransform
    reprojection_rms_px: float
    normal_camera: np.ndarray


@dataclass(frozen=True)
class WorldReference:
    world_from_camera: RigidTransform
    floor_tag_ids: tuple[int, ...]
    reprojection_rms_px: float


def marker_object_corners(marker_size_m: float) -> np.ndarray:
    """Return tag corners in the ordering required by IPPE_SQUARE."""
    half = float(marker_size_m) / 2.0
    if not math.isfinite(half) or half <= 0.0:
        raise ValueError("marker_size_m must be positive")
    return np.asarray([
        [-half, +half, 0.0],
        [+half, +half, 0.0],
        [+half, -half, 0.0],
        [-half, -half, 0.0],
    ], dtype=np.float32)


def detect_tag_corners(image: np.ndarray) -> list[TagCorners]:
    """Detect tag36h11 markers and return one record per decoded ID."""
    if image is None or image.ndim not in (2, 3):
        raise ValueError("image must be a grayscale or BGR OpenCV image")
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raw_corners, raw_ids, _rejected = _APRILTAG_DETECTOR.detectMarkers(gray)
    if raw_ids is None:
        return []

    # A duplicate decoded ID is not useful for rigid tracking.  Retain the
    # larger candidate, which is normally the sharper/nearer one.
    by_id: dict[int, TagCorners] = {}
    perimeters: dict[int, float] = {}
    for raw_corner, raw_id in zip(raw_corners, np.asarray(raw_ids).reshape(-1)):
        corners = np.asarray(raw_corner, dtype=np.float32).reshape(4, 2)
        perimeter = float(sum(
            np.linalg.norm(corners[(index + 1) % 4] - corners[index])
            for index in range(4)
        ))
        tag_id = int(raw_id)
        if perimeter > perimeters.get(tag_id, -1.0):
            by_id[tag_id] = TagCorners(tag_id, corners)
            perimeters[tag_id] = perimeter
    return [by_id[tag_id] for tag_id in sorted(by_id)]


def _scale_tag_corners(
    detections: Sequence[TagCorners],
    *,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> list[TagCorners]:
    """Map decoded corners from a high-detail image to processing pixels."""
    source_width, source_height = source_size
    target_width, target_height = target_size
    if source_size == target_size:
        return list(detections)
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError("source and target image sizes must be positive")
    scale = np.asarray(
        [target_width / source_width, target_height / source_height],
        dtype=np.float32,
    )
    return [
        TagCorners(
            item.tag_id,
            np.asarray(item.corners_px, dtype=np.float32) * scale,
            source=item.source,
            occlusion_age_frames=item.occlusion_age_frames,
            confidence=item.confidence,
        )
        for item in detections
    ]


def _project_rms(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> float:
    projected, _ = cv2.projectPoints(
        object_points, rvec, tvec, camera_matrix, distortion
    )
    error = projected.reshape(-1, 2) - image_points.reshape(-1, 2)
    return math.sqrt(float(np.mean(np.sum(error * error, axis=1))))


def _solve_tag_pose_candidates(
    detection: TagCorners,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    *,
    marker_size_m: float = DEFAULT_MARKER_SIZE_M,
) -> list[_TagPoseSolution]:
    """Return both square-specific planar pose solutions."""
    object_points = marker_object_corners(marker_size_m)
    result = cv2.solvePnPGeneric(
        object_points,
        detection.corners_px,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not result[0] or not result[1]:
        raise ValueError(f"pose solve failed for tag {detection.tag_id}")
    rvecs, tvecs = result[1], result[2]
    candidates: list[_TagPoseSolution] = []
    for rvec, tvec in zip(rvecs, tvecs):
        rvec = np.asarray(rvec, dtype=float).reshape(3, 1)
        tvec = np.asarray(tvec, dtype=float).reshape(3, 1)
        rms = _project_rms(
                object_points,
                detection.corners_px,
                rvec,
                tvec,
                camera_matrix,
                distortion,
            )
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        candidates.append(_TagPoseSolution(
            camera_from_tag=RigidTransform(
                np.asarray(tvec, dtype=float).reshape(3),
                Rotation.from_matrix(rotation_matrix),
            ),
            reprojection_rms_px=float(rms),
            normal_camera=np.asarray(rotation_matrix[:, 2], dtype=float),
        ))
    return candidates


def _tag_pose_from_solution(
    detection: TagCorners,
    chosen: _TagPoseSolution,
    candidates: Sequence[_TagPoseSolution],
) -> TagPose:
    alternate_errors = [
        item.reprojection_rms_px for item in candidates if item is not chosen
    ]
    return TagPose(
        tag_id=detection.tag_id,
        corners_px=detection.corners_px,
        camera_from_tag=chosen.camera_from_tag,
        reprojection_rms_px=chosen.reprojection_rms_px,
        alternate_reprojection_rms_px=(
            None if not alternate_errors else min(alternate_errors)
        ),
    )


def estimate_tag_pose(
    detection: TagCorners,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    *,
    marker_size_m: float = DEFAULT_MARKER_SIZE_M,
    preferred_normal_camera: np.ndarray | None = None,
) -> TagPose:
    """Estimate ``camera_from_tag`` with the square-specific PnP solver."""
    candidates = _solve_tag_pose_candidates(
        detection,
        camera_matrix,
        distortion,
        marker_size_m=marker_size_m,
    )
    # IPPE's planar pair can swap on compressed or downscaled video even when
    # their reprojection errors differ by only hundredths of a pixel.  With a
    # floor frame, select the face normal that remains most upward; without
    # that physical prior preserve the ordinary lowest-RMS behavior.
    if preferred_normal_camera is None:
        ranked = sorted(candidates, key=lambda item: item.reprojection_rms_px)
    else:
        expected = np.asarray(preferred_normal_camera, dtype=float).reshape(3)
        expected /= max(1e-12, float(np.linalg.norm(expected)))
        ranked = sorted(
            candidates,
            key=lambda item: (
                -float(np.dot(item.normal_camera, expected)),
                item.reprojection_rms_px,
            ),
        )
    return _tag_pose_from_solution(
        detection,
        ranked[0],
        candidates,
    )


def estimate_world_reference(
    detections: Sequence[TagCorners],
    floor_tags: Mapping[int, RigidTransform],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    *,
    marker_size_m: float = DEFAULT_MARKER_SIZE_M,
    marker_sizes_m: Mapping[int, float] | None = None,
    previous_world_from_camera: RigidTransform | None = None,
    preferred_floor_normal_camera: np.ndarray | None = None,
) -> WorldReference | None:
    """Solve camera extrinsics from one or more mapped floor tags."""
    visible = [item for item in detections if item.tag_id in floor_tags]
    if not visible:
        return None

    def size_for(tag_id: int) -> float:
        if marker_sizes_m is None:
            return marker_size_m
        return float(marker_sizes_m.get(tag_id, marker_size_m))

    if len(visible) == 1:
        preferred_normal_camera = np.asarray([0.0, 0.0, -1.0])
        if preferred_floor_normal_camera is not None:
            preferred_normal_camera = np.asarray(
                preferred_floor_normal_camera, dtype=float
            ).reshape(3)
        elif previous_world_from_camera is not None:
            preferred_normal_camera = (
                previous_world_from_camera.rotation.inv().apply([0.0, 0.0, 1.0])
            )
        tag_pose = estimate_tag_pose(
            visible[0],
            camera_matrix,
            distortion,
            marker_size_m=size_for(visible[0].tag_id),
            preferred_normal_camera=preferred_normal_camera,
        )
        world_from_camera = floor_tags[visible[0].tag_id].compose(
            tag_pose.camera_from_tag.inverse()
        )
        return WorldReference(
            world_from_camera,
            (visible[0].tag_id,),
            tag_pose.reprojection_rms_px,
        )

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    for detection in visible:
        world_from_tag = floor_tags[detection.tag_id]
        tag_corners = marker_object_corners(size_for(detection.tag_id))
        object_points.extend(world_from_tag.apply(point) for point in tag_corners)
        image_points.extend(detection.corners_px)
    world_points = np.asarray(object_points, dtype=np.float32)
    pixels = np.asarray(image_points, dtype=np.float32)
    ok, rvec, tvec = cv2.solvePnP(
        world_points,
        pixels,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    if hasattr(cv2, "solvePnPRefineLM"):
        rvec, tvec = cv2.solvePnPRefineLM(
            world_points, pixels, camera_matrix, distortion, rvec, tvec
        )
    rms = _project_rms(
        world_points, pixels, rvec, tvec, camera_matrix, distortion
    )
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    camera_from_world = RigidTransform(
        np.asarray(tvec, dtype=float).reshape(3),
        Rotation.from_matrix(rotation_matrix),
    )
    return WorldReference(
        camera_from_world.inverse(),
        tuple(item.tag_id for item in visible),
        rms,
    )


def _read_transform_map(value: Mapping[str, Any]) -> dict[int, RigidTransform]:
    result: dict[int, RigidTransform] = {}
    for raw_id, spec in value.items():
        tag_id = int(raw_id)
        transform_value = spec.get("world_from_tag", spec)
        result[tag_id] = RigidTransform.from_dict(transform_value)
    return result


class AprilTagPoseTracker:
    """Detect tags, establish the floor frame, and estimate the hexapod pose."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        family = str(config.get("tag_family", TAG_FAMILY))
        if family != TAG_FAMILY:
            raise ValueError(f"only {TAG_FAMILY} is supported, got {family!r}")
        self.marker_size_m = float(
            config.get("marker_size_m", DEFAULT_MARKER_SIZE_M)
        )
        marker_object_corners(self.marker_size_m)  # validate now
        self.floor_marker_size_m = float(
            config.get("floor_marker_size_m", self.marker_size_m)
        )
        marker_object_corners(self.floor_marker_size_m)
        self.calibration = CameraCalibration.from_dict(config["camera"])
        self.floor_tags = _read_transform_map(config.get("floor_tags", {}))
        self.floor_marker_sizes_m = {
            int(raw_id): float(spec.get(
                "marker_size_m", self.floor_marker_size_m
            ))
            for raw_id, spec in config.get("floor_tags", {}).items()
        }
        for marker_size in self.floor_marker_sizes_m.values():
            marker_object_corners(marker_size)
        fixed_reference = config.get("fixed_camera_world_reference")
        self.fixed_world_from_camera = (
            None
            if fixed_reference is None
            else RigidTransform.from_dict(fixed_reference)
        )
        self.robot_pose_config = dict(config.get("robot_pose", {}))
        self.visual_joint_bias_deg = {
            str(name): float(value)
            for name, value in self.robot_pose_config.get(
                "visual_joint_bias_deg", {}
            ).items()
        }
        tracking_config = dict(config.get("tracking", {}))
        max_occlusion = int(tracking_config.get("max_occlusion_frames", 8))
        self.temporal_tags = TemporalTagCornerTracker(
            max_occlusion_frames=max_occlusion
        )
        self.foot_tracker = FootTipTracker(max_occlusion_frames=max_occlusion)
        self.marker_size_verified = bool(config.get("marker_size_verified", False))
        self._joint_history: dict[str, tuple[float, int]] = {}
        self._previous_floor_feet: dict[int, tuple[np.ndarray, float | None]] = {}
        self._previous_world_from_camera = self.fixed_world_from_camera
        self._previous_camera_from_tag: dict[int, RigidTransform] = {}
        self.tag_labels = {
            int(raw_id): str(spec.get("label", spec.get("frame", f"tag {raw_id}")))
            for raw_id, spec in self.robot_pose_config.get("tags", {}).items()
        }
        for raw_id, spec in config.get("floor_tags", {}).items():
            self.tag_labels[int(raw_id)] = str(
                spec.get("label", f"floor reference {raw_id}")
            )
        self.frame_by_tag = {
            int(raw_id): str(spec["frame"])
            for raw_id, spec in self.robot_pose_config.get("tags", {}).items()
        }
        branch_config = dict(self.robot_pose_config)
        branch_config["cameras"] = {
            "camera0": {"world_from_camera": RigidTransform.identity().to_dict()}
        }
        self._branch_estimator = (
            None if not self.frame_by_tag
            else HousingPoseEstimator.from_dict(branch_config)
        )

    def reset_temporal_state(self) -> None:
        self.temporal_tags.reset()
        self.foot_tracker.reset()
        self._joint_history.clear()
        self._previous_floor_feet.clear()
        self._previous_world_from_camera = self.fixed_world_from_camera
        self._previous_camera_from_tag.clear()

    @staticmethod
    def _rotation_distance_deg(
        first: RigidTransform, second: RigidTransform
    ) -> float:
        relative = first.rotation.inv() * second.rotation
        return math.degrees(float(relative.magnitude()))

    @staticmethod
    def _branch_joint_names(frame: str) -> tuple[str, ...]:
        if not frame.startswith("L") or "_" not in frame:
            return ()
        leg = int(frame[1:frame.index("_")])
        if frame.endswith("_coxa"):
            return (f"L{leg}_yaw",)
        if frame.endswith("_femur"):
            return (f"L{leg}_yaw", f"L{leg}_hip")
        if frame.endswith("_tibia"):
            return (f"L{leg}_yaw", f"L{leg}_hip", f"L{leg}_knee")
        return ()

    def _branch_encoder_error_deg(
        self,
        *,
        body_tag_id: int,
        body_solution: _TagPoseSolution,
        tag_id: int,
        solution: _TagPoseSolution,
        encoders: Mapping[str, float],
    ) -> float | None:
        if self._branch_estimator is None:
            return None
        joint_names = self._branch_joint_names(self.frame_by_tag[tag_id])
        joint_names = tuple(name for name in joint_names if name in encoders)
        if not joint_names:
            return None
        result = self._branch_estimator.estimate_detections(
            [
                {
                    "tag_id": body_tag_id,
                    "camera": "camera0",
                    "camera_from_tag": body_solution.camera_from_tag.to_dict(),
                    "weight": 1.0,
                },
                {
                    "tag_id": tag_id,
                    "camera": "camera0",
                    "camera_from_tag": solution.camera_from_tag.to_dict(),
                    "weight": 1.0,
                },
            ],
            encoder_joint_deg=encoders,
        )
        errors: list[float] = []
        for name in joint_names:
            record = result.get("joints", {}).get(name, {})
            visual = record.get("value_deg")
            if visual is None:
                continue
            error = (float(visual) - encoders[name] + 180.0) % 360.0 - 180.0
            errors.append(abs(error))
        return None if not errors else float(statistics.median(errors))

    def _fallback_branch_index(
        self, tag_id: int, candidates: Sequence[_TagPoseSolution]
    ) -> tuple[int, str]:
        previous = self._previous_camera_from_tag.get(tag_id)
        if previous is not None:
            index = min(
                range(len(candidates)),
                key=lambda item: (
                    self._rotation_distance_deg(
                        previous, candidates[item].camera_from_tag
                    ),
                    candidates[item].reprojection_rms_px,
                ),
            )
            return index, "temporal_continuity"
        return (
            min(
                range(len(candidates)),
                key=lambda item: candidates[item].reprojection_rms_px,
            ),
            "reprojection",
        )

    def _select_robot_tag_solutions(
        self,
        candidate_map: Mapping[int, tuple[TagCorners, Sequence[_TagPoseSolution]]],
        encoder_joint_deg: Sequence[float] | Mapping[str, float] | None,
        preferred_body_normal_camera: np.ndarray | None = None,
    ) -> tuple[list[TagPose], dict[int, dict[str, Any]]]:
        """Resolve planar branches without turning encoder residuals into bias."""
        encoders = self._encoder_map(
            encoder_joint_deg, BRANCH_JOINT_ORDER
        )
        body_ids = [
            tag_id for tag_id in candidate_map
            if self.frame_by_tag.get(tag_id) == "body"
        ]
        if not body_ids:
            selected: list[TagPose] = []
            decisions: dict[int, dict[str, Any]] = {}
            for tag_id, (detection, candidates) in candidate_map.items():
                index, reason = self._fallback_branch_index(tag_id, candidates)
                selected.append(_tag_pose_from_solution(
                    detection, candidates[index], candidates
                ))
                decisions[tag_id] = {"index": index, "reason": reason}
            return selected, decisions

        body_tag_id = body_ids[0]
        _body_detection, body_candidates = candidate_map[body_tag_id]
        body_encoder_errors: list[float | None] = []
        for body_solution in body_candidates:
            per_tag: list[float] = []
            for tag_id, (_detection, candidates) in candidate_map.items():
                if tag_id == body_tag_id:
                    continue
                errors = [
                    self._branch_encoder_error_deg(
                        body_tag_id=body_tag_id,
                        body_solution=body_solution,
                        tag_id=tag_id,
                        solution=solution,
                        encoders=encoders,
                    )
                    for solution in candidates
                ]
                finite = [value for value in errors if value is not None]
                if finite:
                    per_tag.append(min(finite))
            body_encoder_errors.append(
                None if not per_tag else float(statistics.median(per_tag))
            )

        finite_body = [value for value in body_encoder_errors if value is not None]
        previous_body = self._previous_camera_from_tag.get(body_tag_id)
        body_normal_dots: list[float] | None = None
        floor_body_index: int | None = None
        floor_normal_viable = False
        if preferred_body_normal_camera is not None:
            expected = np.asarray(
                preferred_body_normal_camera, dtype=float
            ).reshape(3)
            expected /= max(1e-12, float(np.linalg.norm(expected)))
            body_normal_dots = [
                float(np.dot(candidate.normal_camera, expected))
                for candidate in body_candidates
            ]
            floor_body_index = max(
                range(len(body_candidates)), key=lambda item: (
                    body_normal_dots[item],
                    -body_candidates[item].reprojection_rms_px,
                )
            )
            floor_normal_viable = (
                body_normal_dots[floor_body_index]
                    >= BODY_FLOOR_NORMAL_MIN_DOT
            )

        if previous_body is None and floor_body_index is not None:
            body_index = floor_body_index
            body_reason = "floor_normal_initialization"
        elif floor_body_index is not None and floor_normal_viable:
            # Temporal PnP continuity can otherwise latch onto the mirrored
            # planar solution after one noisy frame. The chassis tag is on the
            # top face, so one candidate being clearly upright relative to the
            # mapped floor is a stronger physical constraint than sub-pixel
            # reprojection differences. The 45-degree viability bound avoids
            # inventing an upright solution when neither branch is plausible;
            # a true physical tip also remains guarded by the independent IMU
            # safety threshold.
            body_index = floor_body_index
            body_reason = "floor_normal_consistency"
        elif (
            len(finite_body) >= 2
            and max(finite_body) - min(finite_body)
                >= BRANCH_DISAMBIGUATION_MARGIN_DEG
        ):
            body_index = min(
                range(len(body_candidates)),
                key=lambda item: float("inf")
                if body_encoder_errors[item] is None
                else body_encoder_errors[item],
            )
            body_reason = "whole_robot_encoder_consistency"
        else:
            body_index, body_reason = self._fallback_branch_index(
                body_tag_id, body_candidates
            )
        body_solution = body_candidates[body_index]

        selected = []
        decisions = {}
        for tag_id, (detection, candidates) in candidate_map.items():
            if tag_id == body_tag_id:
                index = body_index
                reason = body_reason
                encoder_errors = body_encoder_errors
            else:
                encoder_errors = [
                    self._branch_encoder_error_deg(
                        body_tag_id=body_tag_id,
                        body_solution=body_solution,
                        tag_id=tag_id,
                        solution=solution,
                        encoders=encoders,
                    )
                    for solution in candidates
                ]
                finite = [value for value in encoder_errors if value is not None]
                if (
                    len(finite) >= 2
                    and max(finite) - min(finite)
                        >= BRANCH_DISAMBIGUATION_MARGIN_DEG
                ):
                    index = min(
                        range(len(candidates)),
                        key=lambda item: float("inf")
                        if encoder_errors[item] is None
                        else encoder_errors[item],
                    )
                    reason = "encoder_branch_disambiguation"
                else:
                    index, reason = self._fallback_branch_index(tag_id, candidates)
            selected.append(_tag_pose_from_solution(
                detection, candidates[index], candidates
            ))
            self._previous_camera_from_tag[tag_id] = (
                candidates[index].camera_from_tag
            )
            decisions[tag_id] = {
                "index": index,
                "reason": reason,
                "encoder_error_deg": encoder_errors[index],
                "alternate_encoder_error_deg": next((
                    encoder_errors[item]
                    for item in range(len(encoder_errors)) if item != index
                ), None),
            }
        return selected, decisions

    @classmethod
    def from_json(cls, path: Path | str) -> "AprilTagPoseTracker":
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ValueError("tracker config must contain a JSON object")
        return cls(value)

    def process_frame(
        self,
        image: np.ndarray,
        *,
        frame_index: int = 0,
        time_s: float | None = None,
        encoder_joint_deg: Sequence[float] | Mapping[str, float] | None = None,
        render_overlay: bool = True,
        detection_gray: np.ndarray | None = None,
        tracking_gray: np.ndarray | None = None,
    ) -> tuple[dict[str, Any], np.ndarray]:
        height, width = image.shape[:2]
        if detection_gray is not None:
            detection_gray = np.asarray(detection_gray)
            if detection_gray.ndim != 2:
                raise ValueError("detection_gray must be a grayscale image")
        if tracking_gray is not None:
            tracking_gray = np.asarray(tracking_gray)
            if tracking_gray.ndim != 2 or tracking_gray.shape != image.shape[:2]:
                raise ValueError("tracking_gray must match the processing image")
        camera_matrix, distortion = self.calibration.for_image(width, height)
        detector_image = image if detection_gray is None else detection_gray
        detector_height, detector_width = detector_image.shape[:2]
        decoded_corners = _scale_tag_corners(
            detect_tag_corners(detector_image),
            source_size=(detector_width, detector_height),
            target_size=(width, height),
        )
        temporal_image = image if tracking_gray is None else tracking_gray
        corners = self.temporal_tags.update(temporal_image, decoded_corners)
        reference = estimate_world_reference(
            corners,
            self.floor_tags,
            camera_matrix,
            distortion,
            marker_size_m=self.floor_marker_size_m,
            marker_sizes_m=self.floor_marker_sizes_m,
            previous_world_from_camera=self._previous_world_from_camera,
        )
        fixed_reference_used = (
            reference is None and self.fixed_world_from_camera is not None
        )
        if fixed_reference_used:
            assert self.fixed_world_from_camera is not None
            world_from_camera = self.fixed_world_from_camera
            reference_name = "floor"
            preferred_normal_camera = world_from_camera.rotation.inv().apply(
                [0.0, 0.0, 1.0]
            )
        elif reference is None:
            world_from_camera = RigidTransform.identity()
            reference_name = "camera"
            preferred_normal_camera = None
        else:
            world_from_camera = reference.world_from_camera
            self._previous_world_from_camera = world_from_camera
            reference_name = "floor"
            preferred_normal_camera = world_from_camera.rotation.inv().apply(
                [0.0, 0.0, 1.0]
            )
        poses: list[TagPose] = []
        pose_failures: list[int] = []
        robot_candidates: dict[
            int, tuple[TagCorners, Sequence[_TagPoseSolution]]
        ] = {}
        for detection in corners:
            try:
                if detection.tag_id in self.frame_by_tag:
                    robot_candidates[detection.tag_id] = (
                        detection,
                        _solve_tag_pose_candidates(
                            detection,
                            camera_matrix,
                            distortion,
                            marker_size_m=self.marker_size_m,
                        ),
                    )
                else:
                    pose_marker_size_m = (
                        self.floor_marker_sizes_m[detection.tag_id]
                        if detection.tag_id in self.floor_tags
                        else self.marker_size_m
                    )
                    poses.append(estimate_tag_pose(
                        detection,
                        camera_matrix,
                        distortion,
                        marker_size_m=pose_marker_size_m,
                        preferred_normal_camera=preferred_normal_camera,
                    ))
            except (ValueError, cv2.error):
                pose_failures.append(detection.tag_id)
        robot_poses, branch_decisions = self._select_robot_tag_solutions(
            robot_candidates,
            encoder_joint_deg,
            preferred_body_normal_camera=preferred_normal_camera,
        )
        poses.extend(robot_poses)
        poses.sort(key=lambda item: item.tag_id)

        serialized_detections: list[dict[str, Any]] = []
        estimator_detections: list[dict[str, Any]] = []
        for pose in poses:
            corner = next(item for item in corners if item.tag_id == pose.tag_id)
            world_from_tag = world_from_camera.compose(pose.camera_from_tag)
            record = {
                "tag_id": pose.tag_id,
                "label": self.tag_labels.get(pose.tag_id, f"tag {pose.tag_id}"),
                "center_px": [round(float(v), 3) for v in corner.center_px],
                "corners_px": [
                    [round(float(v), 3) for v in point]
                    for point in corner.corners_px
                ],
                "tag_y_clockwise_from_image_up_deg": round(
                    corner.tag_y_clockwise_from_image_up_deg, 3
                ),
                "source": corner.source,
                "occlusion_age_frames": corner.occlusion_age_frames,
                "confidence": round(float(corner.confidence), 3),
                "reprojection_rms_px": round(pose.reprojection_rms_px, 4),
                "alternate_reprojection_rms_px": (
                    None if pose.alternate_reprojection_rms_px is None
                    else round(pose.alternate_reprojection_rms_px, 4)
                ),
                "camera_from_tag": pose.camera_from_tag.to_dict(),
                f"{reference_name}_from_tag": world_from_tag.to_dict(),
            }
            branch = branch_decisions.get(pose.tag_id)
            if branch is not None:
                record["pose_branch_index"] = int(branch["index"])
                record["pose_branch_reason"] = str(branch["reason"])
                for key in (
                    "encoder_error_deg", "alternate_encoder_error_deg"
                ):
                    value = branch.get(key)
                    record[f"pose_branch_{key}"] = (
                        None if value is None else round(float(value), 3)
                    )
            serialized_detections.append(record)
            estimator_detections.append({
                "tag_id": pose.tag_id,
                "camera": "camera0",
                "camera_from_tag": pose.camera_from_tag.to_dict(),
                "weight": corner.confidence
                / max(0.05, pose.reprojection_rms_px) ** 2,
            })

        robot_result: dict[str, Any] | None = None
        if self.robot_pose_config.get("tags"):
            pose_config = dict(self.robot_pose_config)
            pose_config["cameras"] = {
                "camera0": {"world_from_camera": world_from_camera.to_dict()}
            }
            robot_result = HousingPoseEstimator.from_dict(
                pose_config
            ).estimate_detections(
                estimator_detections, encoder_joint_deg=encoder_joint_deg
            )
            robot_result["pose_reference"] = reference_name

        corners_by_id = {item.tag_id: item for item in corners}
        body_centers = [
            corners_by_id[tag_id].center_px
            for tag_id, frame in self.frame_by_tag.items()
            if frame == "body" and tag_id in corners_by_id
        ]
        body_center = None if not body_centers else np.mean(body_centers, axis=0)
        femur_anchors: dict[int, np.ndarray] = {}
        for tag_id, frame in self.frame_by_tag.items():
            if not (frame.startswith("L") and frame.endswith("_femur")):
                continue
            if tag_id not in corners_by_id:
                continue
            leg = int(frame[1:frame.index("_")])
            femur_anchors[leg] = corners_by_id[tag_id].center_px
        tag_scale = robust_tag_scale_px({
            tag_id: item.corners_px for tag_id, item in corners_by_id.items()
            if tag_id in self.frame_by_tag
        })
        foot_tips = self.foot_tracker.update(
            image,
            body_center_px=body_center,
            femur_anchor_px=femur_anchors,
            tag_scale_px=40.0 if tag_scale is None else tag_scale,
            gray_image=tracking_gray,
        )
        foot_records = self._serialize_foot_tips(
            foot_tips,
            camera_matrix,
            distortion,
            reference,
            time_s=time_s,
        )
        full_pose = self._full_pose_diagnostics(
            robot_result,
            foot_tips,
            camera_matrix,
            distortion,
            world_from_camera,
            tag_scale_px=40.0 if tag_scale is None else tag_scale,
            encoder_joint_deg=encoder_joint_deg,
            corners=corners,
        )
        full_pose["walking_check"] = self._walking_check(
            robot_result, full_pose, foot_records
        )

        result: dict[str, Any] = {
            "schema_version": 2,
            "joint_frame": FRAME_ROBOT_ABS,
            "joint_contract": JOINT_CONTRACT,
            "frame_index": int(frame_index),
            "time_s": None if time_s is None else round(float(time_s), 6),
            "image_size_px": [width, height],
            "detection_image_size_px": [detector_width, detector_height],
            "native_luma_detection": detection_gray is not None,
            "tag_family": TAG_FAMILY,
            "marker_size_m": self.marker_size_m,
            "floor_marker_size_m": self.floor_marker_size_m,
            "camera_calibration_approximate": self.calibration.approximate,
            "marker_size_verified": self.marker_size_verified,
            "pose_reference": reference_name,
            "detected_tag_ids": [item.tag_id for item in decoded_corners],
            "tracked_tag_ids": [pose.tag_id for pose in poses],
            "optical_flow_tag_ids": [
                item.tag_id for item in corners if item.source == "optical_flow"
            ],
            "pose_failure_tag_ids": pose_failures,
            "detections": serialized_detections,
            "world_reference": (
                {
                    "source": "fixed_camera",
                    "floor_tag_ids": [],
                    "reprojection_rms_px": None,
                    "world_from_camera": world_from_camera.to_dict(),
                }
                if fixed_reference_used
                else None if reference is None else {
                    "source": "floor_tags",
                    "floor_tag_ids": list(reference.floor_tag_ids),
                    "reprojection_rms_px": round(
                        reference.reprojection_rms_px, 4
                    ),
                    "world_from_camera": reference.world_from_camera.to_dict(),
                }
            ),
            "hexapod_pose": robot_result,
            "foot_tips": foot_records,
            "full_pose": full_pose,
        }
        rendered = image
        if render_overlay:
            rendered = self.annotate(
                image,
                corners,
                poses,
                result,
                camera_matrix,
                distortion,
                foot_tips,
            )
        return result, rendered

    @staticmethod
    def _encoder_map(
        value: Sequence[float] | Mapping[str, float] | None,
        joint_order: Sequence[str],
    ) -> dict[str, float]:
        if value is None:
            return {}
        if isinstance(value, Mapping):
            raw = value.items()
        else:
            if len(value) != len(joint_order):
                raise ValueError("encoder_joint_deg must contain 18 values")
            raw = zip(joint_order, value)
        result: dict[str, float] = {}
        for name, angle in raw:
            if angle is None:
                continue
            number = float(angle)
            if not math.isfinite(number):
                continue
            result[str(name)] = number
        return result

    @staticmethod
    def _floor_intersection(
        point_px: np.ndarray,
        camera_matrix: np.ndarray,
        distortion: np.ndarray,
        world_from_camera: RigidTransform,
    ) -> np.ndarray | None:
        normalized = cv2.undistortPoints(
            np.asarray(point_px, dtype=np.float64).reshape(1, 1, 2),
            camera_matrix,
            distortion,
        ).reshape(2)
        camera_direction = np.asarray([normalized[0], normalized[1], 1.0])
        world_direction = world_from_camera.rotation.apply(camera_direction)
        origin = world_from_camera.translation_m
        if abs(float(world_direction[2])) < 1e-9:
            return None
        distance = -float(origin[2]) / float(world_direction[2])
        if distance <= 0.0:
            return None
        return origin + distance * world_direction

    def _serialize_foot_tips(
        self,
        observations: Sequence[FootTipObservation],
        camera_matrix: np.ndarray,
        distortion: np.ndarray,
        reference: WorldReference | None,
        *,
        time_s: float | None,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        current_legs: set[int] = set()
        for observation in observations:
            current_legs.add(observation.leg)
            record = observation.to_dict()
            floor_point = None
            if reference is not None:
                floor_point = self._floor_intersection(
                    observation.point_px,
                    camera_matrix,
                    distortion,
                    reference.world_from_camera,
                )
            if floor_point is not None:
                record["floor_intersection_world_m"] = [
                    round(float(value), 6) for value in floor_point
                ]
                previous = self._previous_floor_feet.get(observation.leg)
                if (previous is not None and time_s is not None
                        and previous[1] is not None and time_s > previous[1]):
                    speed = float(np.linalg.norm(
                        floor_point[:2] - previous[0][:2]
                    )) / (time_s - previous[1])
                    record["floor_projection_speed_m_s"] = round(speed, 5)
                self._previous_floor_feet[observation.leg] = (
                    floor_point,
                    time_s,
                )
            records.append(record)
        self._previous_floor_feet = {
            leg: value for leg, value in self._previous_floor_feet.items()
            if leg in current_legs
        }
        return records

    def _fit_knee_from_tip(
        self,
        *,
        leg: int,
        tip_px: np.ndarray,
        yaw_deg: float,
        hip_deg: float,
        world_from_body: RigidTransform,
        world_from_camera: RigidTransform,
        camera_matrix: np.ndarray,
        distortion: np.ndarray,
    ) -> tuple[float, float] | None:
        geometry = self.robot_pose_config.get("geometry", {})
        chassis = float(geometry.get("chassis_apothem_m", 0.100))
        coxa = float(geometry.get("coxa_m", 0.0125))
        femur = float(geometry.get("femur_m", 0.090))
        tibia = float(geometry.get("tibia_m", 0.150))
        hip_y = float(geometry.get("hip_anchor_y_m", -0.02565))
        azimuth = (leg + 0.5) * math.pi / 3.0
        yaw = math.radians(yaw_deg)
        hip = math.radians(hip_deg)
        leg_rotation = Rotation.from_rotvec([0.0, 0.0, azimuth + yaw])
        yaw_origin = np.asarray([
            chassis * math.cos(azimuth),
            chassis * math.sin(azimuth),
            0.0,
        ])
        hip_origin = yaw_origin + leg_rotation.apply([coxa, hip_y, 0.0])
        knee_origin = hip_origin + (
            leg_rotation * Rotation.from_rotvec([0.0, hip, 0.0])
        ).apply([femur, 0.0, 0.0])
        camera_from_world = world_from_camera.inverse()

        def projected(knee: float) -> np.ndarray | None:
            foot_body = knee_origin + (
                leg_rotation * Rotation.from_rotvec([0.0, knee, 0.0])
            ).apply([tibia, 0.0, 0.0])
            camera_point = camera_from_world.apply(
                world_from_body.apply(foot_body)
            )
            if camera_point[2] <= 1e-6:
                return None
            pixels, _ = cv2.projectPoints(
                camera_point.reshape(1, 3),
                np.zeros(3),
                np.zeros(3),
                camera_matrix,
                distortion,
            )
            return pixels.reshape(2)

        grid = np.linspace(-math.pi, math.pi, 361)
        errors: list[float] = []
        for angle in grid:
            pixel = projected(float(angle))
            errors.append(
                float("inf") if pixel is None
                else float(np.linalg.norm(pixel - tip_px))
            )
        best = int(np.argmin(errors))
        if not math.isfinite(errors[best]):
            return None
        # A small parabolic refinement is sufficient for a diagnostic signal.
        angle = float(grid[best])
        if 0 < best < len(grid) - 1:
            left, center, right = errors[best - 1:best + 2]
            denominator = left - 2.0 * center + right
            if abs(denominator) > 1e-9:
                fraction = max(-1.0, min(1.0, 0.5 * (left - right) / denominator))
                angle += fraction * float(grid[1] - grid[0])
        pixel = projected(angle)
        if pixel is None:
            return None
        return math.degrees(angle), float(np.linalg.norm(pixel - tip_px))

    def _full_pose_diagnostics(
        self,
        robot_result: Mapping[str, Any] | None,
        foot_tips: Sequence[FootTipObservation],
        camera_matrix: np.ndarray,
        distortion: np.ndarray,
        world_from_camera: RigidTransform,
        *,
        tag_scale_px: float,
        encoder_joint_deg: Sequence[float] | Mapping[str, float] | None,
        corners: Sequence[TagCorners],
    ) -> dict[str, Any]:
        if not robot_result or not robot_result.get("ok"):
            return {
                "ok": False,
                "complete": False,
                "motor_commands_sent": False,
                "error": "body pose unavailable",
            }
        joint_order = list(robot_result["joint_order"])
        encoders = self._encoder_map(encoder_joint_deg, joint_order)
        corner_by_frame = {
            self.frame_by_tag[item.tag_id]: item
            for item in corners if item.tag_id in self.frame_by_tag
        }
        tips = {item.leg: item for item in foot_tips}
        world_from_body = RigidTransform.from_dict(
            robot_result["body_pose"]["world_from_body"]
        )
        # Knee-servo lid tags move with the femur, not the tibia. A monocular
        # fit from that lid plus a colored foot tip is perspective-sensitive
        # and proved materially wrong on the real robot. Do not manufacture a
        # visual knee angle: knees are encoder-only until rigid tibia markers
        # (or another directly observed tibia orientation) exist.
        video_knees: dict[int, dict[str, Any]] = {}

        joints: dict[str, dict[str, Any]] = {}
        disagreement: list[dict[str, Any]] = []
        current_history: dict[str, tuple[float, int]] = {}
        for name in joint_order:
            leg = int(name[1:name.index("_")])
            axis = name.rsplit("_", 1)[1]
            direct = robot_result["joints"][name]
            visual_raw_value = (
                float(direct["value_deg"]) if direct["observable"] else None
            )
            visual_bias = float(self.visual_joint_bias_deg.get(name, 0.0))
            visual_value = (
                None if visual_raw_value is None else
                (visual_raw_value - visual_bias + 180.0) % 360.0 - 180.0
            )
            visual_abs_value = None
            visual_source = None
            visual_confidence = 0.0
            residual_px = None
            if visual_value is not None:
                frames = direct.get("source_frames", [])
                relevant = [corner_by_frame[frame] for frame in frames
                            if frame in corner_by_frame]
                visual_source = (
                    "apriltag_optical_flow"
                    if relevant and any(item.source != "detected" for item in relevant)
                    else "apriltag"
                )
                visual_confidence = min(
                    (item.confidence for item in relevant), default=0.7
                )
            elif axis == "knee" and leg in video_knees:
                knee_fit = video_knees[leg]
                visual_value = knee_fit["signed_deg"]
                visual_abs_value = knee_fit["absolute_deg"]
                residual_px = knee_fit["residual_px"]
                visual_confidence = knee_fit["confidence"]
                visual_source = knee_fit["source"]
            encoder_value = encoders.get(name)

            # Direct tag angles are the physical calibration check.  Knee-tip
            # fits are provisional, so a live encoder remains the primary knee.
            if axis == "knee" and encoder_value is not None:
                value = encoder_value
                source = "encoder"
                confidence = 0.95
            elif visual_value is not None:
                value = visual_value
                source = visual_source
                confidence = visual_confidence
            elif encoder_value is not None:
                value = encoder_value
                source = "encoder"
                confidence = 0.90
            else:
                previous = self._joint_history.get(name)
                if previous is None or previous[1] >= self.temporal_tags.max_occlusion_frames:
                    value = None
                    source = "unobservable"
                    confidence = 0.0
                else:
                    value = previous[0]
                    age = previous[1] + 1
                    source = "temporal_prediction"
                    confidence = max(0.05, 0.5 ** age)

            age = 0 if source != "temporal_prediction" else self._joint_history[name][1] + 1
            if value is not None:
                current_history[name] = (float(value), age)
            record: dict[str, Any] = {
                "value_deg": None if value is None else round(float(value), 4),
                "source": source,
                "confidence": round(float(confidence), 3),
                "occlusion_age_frames": age,
                "visual_deg": None if visual_value is None else round(visual_value, 4),
                "visual_absolute_deg": (
                    None if visual_abs_value is None
                    else round(float(visual_abs_value), 4)
                ),
                "visual_source": visual_source,
                "visual_confidence": round(float(visual_confidence), 3),
                "encoder_deg": None if encoder_value is None else round(encoder_value, 4),
            }
            if visual_raw_value is not None and axis != "knee":
                record["visual_raw_deg"] = round(visual_raw_value, 4)
                record["visual_bias_deg"] = round(visual_bias, 4)
            if axis == "knee" and leg in video_knees:
                record["vision_sign_ambiguous"] = bool(
                    video_knees[leg]["sign_ambiguous"]
                )
                if "alignment_deg" in video_knees[leg]:
                    record["foot_chain_alignment_deg"] = round(
                        float(video_knees[leg]["alignment_deg"]), 3
                    )
            if residual_px is not None:
                record["foot_reprojection_residual_px"] = round(residual_px, 3)
            if visual_value is not None and encoder_value is not None:
                delta = (visual_value - encoder_value + 180.0) % 360.0 - 180.0
                record["visual_minus_encoder_deg"] = round(delta, 4)
                if abs(delta) > 6.0:
                    disagreement.append({
                        "joint": name,
                        "visual_minus_encoder_deg": round(delta, 3),
                    })
            elif visual_abs_value is not None and encoder_value is not None:
                magnitude_delta = visual_abs_value - abs(encoder_value)
                record["visual_abs_minus_encoder_abs_deg"] = round(
                    magnitude_delta, 4
                )
                # This is a perspective-sensitive diagnostic, not a measured
                # knee angle. Never let it poison zero/safety calibration.
            joints[name] = record
        self._joint_history = current_history

        vector = [joints[name]["value_deg"] for name in joint_order]
        missing = [name for name in joint_order if joints[name]["value_deg"] is None]
        prediction_only = [
            name for name in joint_order
            if joints[name]["source"] == "temporal_prediction"
        ]
        zero_errors = {
            name: abs(float(joints[name]["value_deg"]))
            for name in joint_order if joints[name]["value_deg"] is not None
        }
        for name in joint_order:
            absolute = joints[name].get("visual_absolute_deg")
            if absolute is not None and name not in zero_errors:
                zero_errors[name] = float(absolute)
        out_of_zero = []
        for name, error in zero_errors.items():
            unsigned = (
                joints[name].get("vision_sign_ambiguous", False)
                and joints[name].get("value_deg") is None
            )
            tolerance = 22.0 if unsigned else 5.0
            if error > tolerance:
                out_of_zero.append({
                    "joint": name,
                    "error_deg": round(error, 3),
                    "tolerance_deg": tolerance,
                    "unsigned_visual_estimate": unsigned,
                })
        issues: list[str] = []
        if self.calibration.approximate:
            issues.append("camera intrinsics are approximate; calibrate this phone lens")
        if not self.marker_size_verified:
            issues.append("measure and verify the printed black-square tag size")
        if len([item for item in corners if item.source == "detected"
                and item.tag_id in self.frame_by_tag]) < 7:
            issues.append("fewer than seven robot tags are directly decoded")
        if len([item for item in foot_tips if item.source == "color"]) < 6:
            issues.append("one or more boot tips are inferred or hidden")
        if not encoders:
            issues.append(
                "no read-only encoder feedback; knee angles are unobservable"
            )
        if disagreement:
            issues.append("visual and encoder angles disagree on one or more joints")
        if out_of_zero:
            issues.append("one or more joints are over 5 degrees from zero")

        measured_complete = not missing and not prediction_only
        zero_observable = all(
            joints[name]["value_deg"] is not None
            or joints[name].get("visual_absolute_deg") is not None
            for name in joint_order
        )
        zero_match = zero_observable and not out_of_zero and not disagreement
        motor_assist_blockers = list(issues)
        if not zero_observable:
            motor_assist_blockers.append("the zero pose is not fully observable")
        return {
            "ok": True,
            "joint_frame": FRAME_ROBOT_ABS,
            "joint_contract": JOINT_CONTRACT,
            "complete": not missing,
            "signed_complete": not missing,
            "measured_complete": measured_complete,
            "motor_commands_sent": False,
            "joint_order": joint_order,
            "joint_vector_deg": vector,
            "joints": joints,
            "missing_joints": missing,
            "prediction_only_joints": prediction_only,
            "calibration_disagreements": disagreement,
            "zero_check": {
                "advisory_only": True,
                "automatic_motion_enabled": False,
                "ready_for_motor_assist": False,
                "motor_assist_blockers": motor_assist_blockers,
                "motor_commands_sent": False,
                "observable": zero_observable,
                "direct_tolerance_deg": 5.0,
                "unsigned_knee_tolerance_deg": 22.0,
                "matches_zero": zero_match,
                "out_of_tolerance": out_of_zero,
                "issues": issues,
                "next_action": (
                    "pose matches zero; operator may review before any set-zero action"
                    if zero_match else
                    "hand-adjust or diagnose the listed joints; do not command "
                    "motion from this report"
                ),
            },
        }

    @staticmethod
    def _walking_check(
        robot_result: Mapping[str, Any] | None,
        full_pose: Mapping[str, Any],
        foot_records: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        tilt = None
        if (robot_result and robot_result.get("ok")
                and robot_result.get("pose_reference") == "floor"):
            euler = robot_result["body_pose"].get("euler_xyz_deg")
            if euler and len(euler) == 3:
                tilt = math.hypot(float(euler[0]), float(euler[1]))
        feet = []
        for record in foot_records:
            feet.append({
                "leg": record["leg"],
                "source": record["source"],
                "confidence": record["confidence"],
                "floor_projection_speed_m_s": record.get(
                    "floor_projection_speed_m_s"
                ),
            })
        flags: list[str] = []
        if tilt is not None and tilt > 10.0:
            flags.append(f"body tilt is large ({tilt:.1f} deg)")
        if full_pose.get("calibration_disagreements"):
            flags.append("visual/encoder disagreement suggests stale zeros or tag mounts")
        direct_feet = sum(item["source"] == "color" for item in feet)
        if direct_feet < 4:
            flags.append("too few directly observed feet for reliable gait diagnosis")
        return {
            "body_tilt_deg": None if tilt is None else round(tilt, 3),
            "feet": feet,
            "flags": flags,
            "signals_available": {
                "body_motion": bool(robot_result and robot_result.get("ok")),
                "foot_trajectories": bool(feet),
                "encoder_cross_check": any(
                    record.get("encoder_deg") is not None
                    for record in full_pose.get("joints", {}).values()
                ),
            },
            "interpretation": (
                "Across a video, compare body tilt, per-leg trajectories, "
                "floor-projection speed, and visual-minus-encoder errors to "
                "find asymmetry, drag, possible slip, or calibration drift."
            ),
            "limitation": (
                "A monocular floor projection is not proof of contact; combine "
                "it with encoder/FK state before calling motion foot slip."
            ),
        }

    def annotate(
        self,
        image: np.ndarray,
        corners: Sequence[TagCorners],
        poses: Sequence[TagPose],
        result: Mapping[str, Any],
        camera_matrix: np.ndarray,
        distortion: np.ndarray,
        foot_tips: Sequence[FootTipObservation] = (),
    ) -> np.ndarray:
        output = image.copy()
        scale = max(0.55, output.shape[1] / 2600.0)
        thickness = max(1, round(scale * 2))
        pose_by_id = {pose.tag_id: pose for pose in poses}
        floor_ids = set(self.floor_tags)
        for detection in corners:
            points = np.rint(detection.corners_px).astype(int)
            if detection.source == "optical_flow":
                color = (255, 150, 35)
            else:
                color = (
                    (40, 210, 40)
                    if detection.tag_id in floor_ids else (0, 210, 255)
                )
            cv2.polylines(output, [points], True, color, thickness, cv2.LINE_AA)
            pose = pose_by_id.get(detection.tag_id)
            if pose is not None:
                rvec = pose.camera_from_tag.rotation.as_rotvec().reshape(3, 1)
                tvec = pose.camera_from_tag.translation_m.reshape(3, 1)
                cv2.drawFrameAxes(
                    output,
                    camera_matrix,
                    distortion,
                    rvec,
                    tvec,
                    (
                        self.floor_marker_sizes_m[detection.tag_id]
                        if detection.tag_id in floor_ids
                        else self.marker_size_m
                    ) * 0.7,
                    thickness,
                )
            center = detection.center_px.astype(int)
            label = self.tag_labels.get(detection.tag_id, "unmapped")
            suffix = (
                "" if detection.source == "detected"
                else f" [flow {detection.occlusion_age_frames}]"
            )
            text = f"{detection.tag_id}: {label}{suffix}"
            cv2.putText(
                output,
                text,
                (int(center[0] + 12), int(center[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                color,
                thickness,
                cv2.LINE_AA,
            )

        foot_colors = {
            "color": (40, 40, 255),
            "optical_flow": (255, 90, 255),
            "prediction": (150, 150, 150),
        }
        for foot in foot_tips:
            point = tuple(np.rint(foot.point_px).astype(int))
            color = foot_colors.get(foot.source, (255, 255, 255))
            cv2.circle(output, point, max(5, thickness * 4), color, thickness + 1,
                       cv2.LINE_AA)
            cv2.putText(
                output,
                f"L{foot.leg} tip {foot.source}",
                (point[0] + 12, point[1] + 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                color,
                thickness,
                cv2.LINE_AA,
            )

        header = (
            f"tag36h11: {len(poses)} tracked | feet: {len(foot_tips)}/6 | "
            f"pose ref: {result['pose_reference']}"
        )
        cv2.putText(
            output, header, (20, 42), cv2.FONT_HERSHEY_SIMPLEX,
            scale, (255, 255, 255), thickness + 2, cv2.LINE_AA,
        )
        cv2.putText(
            output, header, (20, 42), cv2.FONT_HERSHEY_SIMPLEX,
            scale, (20, 20, 20), thickness, cv2.LINE_AA,
        )
        full_pose = result.get("full_pose")
        if isinstance(full_pose, Mapping) and full_pose.get("ok"):
            zero = full_pose.get("zero_check", {})
            errors = zero.get("out_of_tolerance", [])
            if zero.get("matches_zero"):
                zero_text = "ZERO CHECK: MATCH (advisory only)"
                zero_color = (50, 230, 50)
            elif errors:
                details = ", ".join(
                    f"{item['joint']} {item['error_deg']:.1f}deg"
                    for item in errors[:4]
                )
                zero_text = f"ZERO CHECK: ADJUST {details}"
                zero_color = (0, 170, 255)
            else:
                zero_text = "ZERO CHECK: INCOMPLETE / OCCLUDED"
                zero_color = (0, 170, 255)
            y = 42 + max(34, round(46 * scale))
            cv2.putText(
                output, zero_text, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (20, 20, 20), thickness + 2, cv2.LINE_AA,
            )
            cv2.putText(
                output, zero_text, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, zero_color, thickness, cv2.LINE_AA,
            )
        return output
