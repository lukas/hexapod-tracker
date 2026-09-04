"""RGB-D refinement for the tracker's existing AprilTag world reference.

The ordinary tracker obtains ``world_from_camera`` from the image positions of
mapped floor tags.  This module adds a registered depth-plane constraint.  RGB
still supplies the sub-pixel tag corners and therefore x/y/yaw; depth mainly
stabilizes distance, roll, pitch, and the planar IPPE ambiguity.

The code is camera-only and read-only.  It does not import a capture backend or
connect to the robot.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .apriltag_vision import (
    DEFAULT_MARKER_SIZE_M,
    TagCorners,
    estimate_world_reference,
    marker_object_corners,
)
from .housing_pose import RigidTransform


class RGBDCalibrationError(ValueError):
    """A frame cannot provide a trustworthy RGB-D calibration observation."""


@dataclass(frozen=True)
class RGBDCalibrationOptions:
    """Quality gates and weights for an iPhone-style registered depth map."""

    min_confidence: int = 1
    min_depth_m: float = 0.20
    max_depth_m: float = 4.0
    min_depth_samples: int = 40
    max_depth_samples: int = 1200
    plane_ransac_threshold_m: float = 0.018
    min_plane_inlier_fraction: float = 0.55
    max_plane_rms_m: float = 0.018
    max_plane_normal_error_deg: float = 25.0
    rgb_sigma_px: float = 0.75
    depth_sigma_m: float = 0.012
    depth_weight: float = 1.0
    single_tag_mask_scale: float = 1.8

    def __post_init__(self) -> None:
        if self.min_confidence not in (0, 1, 2):
            raise ValueError("min_confidence must be 0, 1, or 2")
        if not 0.0 < self.min_depth_m < self.max_depth_m:
            raise ValueError("depth range must be positive and ordered")
        if self.min_depth_samples < 3:
            raise ValueError("min_depth_samples must be at least three")
        if self.max_depth_samples < self.min_depth_samples:
            raise ValueError("max_depth_samples cannot be below the minimum")
        if not 0.0 < self.min_plane_inlier_fraction <= 1.0:
            raise ValueError("min_plane_inlier_fraction must be in (0, 1]")
        for name in (
            "plane_ransac_threshold_m",
            "max_plane_rms_m",
            "max_plane_normal_error_deg",
            "rgb_sigma_px",
            "depth_sigma_m",
            "depth_weight",
            "single_tag_mask_scale",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive finite number")


@dataclass(frozen=True)
class DepthPlaneFit:
    """A robust plane in OpenCV camera coordinates: ``normal . p + d = 0``."""

    normal_camera: np.ndarray
    offset_m: float
    points_camera_m: np.ndarray
    inlier_mask: np.ndarray
    sample_count: int
    inlier_count: int
    inlier_fraction: float
    rms_m: float


@dataclass(frozen=True)
class RGBDWorldReference:
    """One accepted, depth-refined camera pose observation."""

    world_from_camera: RigidTransform
    floor_tag_ids: tuple[int, ...]
    camera_matrix: np.ndarray
    image_size_px: tuple[int, int]
    depth_size_px: tuple[int, int]
    rgb_only_reprojection_rms_px: float
    reprojection_rms_px: float
    depth_plane_rms_m: float
    depth_samples: int
    depth_inliers: int
    depth_inlier_fraction: float
    plane_normal_error_deg: float

    def quality_weight(self) -> float:
        pixel_term = max(0.25, self.reprojection_rms_px)
        depth_term = max(0.003, self.depth_plane_rms_m)
        return self.depth_inlier_fraction / (pixel_term * depth_term)

    def to_dict(self) -> dict[str, object]:
        return {
            "world_from_camera": self.world_from_camera.to_dict(),
            "floor_tag_ids": list(self.floor_tag_ids),
            "image_size_px": list(self.image_size_px),
            "depth_size_px": list(self.depth_size_px),
            "rgb_only_reprojection_rms_px": round(
                self.rgb_only_reprojection_rms_px, 5
            ),
            "reprojection_rms_px": round(self.reprojection_rms_px, 5),
            "depth_plane_rms_mm": round(self.depth_plane_rms_m * 1000.0, 4),
            "depth_samples": self.depth_samples,
            "depth_inliers": self.depth_inliers,
            "depth_inlier_fraction": round(self.depth_inlier_fraction, 5),
            "plane_normal_error_deg": round(self.plane_normal_error_deg, 5),
        }


@dataclass(frozen=True)
class RGBDSessionCalibration:
    """Robust fixed-camera consensus over several RGB-D observations."""

    world_from_camera: RigidTransform
    camera_matrix: np.ndarray
    image_size_px: tuple[int, int]
    depth_size_px: tuple[int, int]
    input_frames: int
    accepted_frames: int
    rejected_outlier_frames: int
    translation_spread_mm: float
    rotation_spread_deg: float
    median_reprojection_rms_px: float
    median_depth_plane_rms_mm: float
    floor_tag_ids: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "coordinate_convention": (
                "world_from_camera maps OpenCV camera coordinates "
                "(x right, y down, z forward) into the mapped floor frame"
            ),
            "world_from_camera": self.world_from_camera.to_dict(),
            "camera": {
                "image_size_px": list(self.image_size_px),
                "camera_matrix": [
                    [round(float(value), 9) for value in row]
                    for row in self.camera_matrix
                ],
                "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
                "approximate": False,
                "allow_center_crop": False,
                "allow_quarter_turn": False,
                "source": "registered RGB-D stream intrinsics",
            },
            "depth_size_px": list(self.depth_size_px),
            "quality": {
                "input_frames": self.input_frames,
                "accepted_frames": self.accepted_frames,
                "rejected_outlier_frames": self.rejected_outlier_frames,
                "translation_spread_mm": round(self.translation_spread_mm, 4),
                "rotation_spread_deg": round(self.rotation_spread_deg, 5),
                "median_reprojection_rms_px": round(
                    self.median_reprojection_rms_px, 5
                ),
                "median_depth_plane_rms_mm": round(
                    self.median_depth_plane_rms_mm, 4
                ),
                "floor_tag_ids": list(self.floor_tag_ids),
            },
        }


def scale_camera_matrix(
    camera_matrix: np.ndarray,
    source_size_px: tuple[int, int],
    target_size_px: tuple[int, int],
) -> np.ndarray:
    """Scale registered RGB intrinsics to a same-aspect depth map."""
    source_width, source_height = source_size_px
    target_width, target_height = target_size_px
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError("image sizes must be positive")
    sx = target_width / source_width
    sy = target_height / source_height
    if not math.isclose(sx, sy, rel_tol=0.025, abs_tol=0.0):
        raise RGBDCalibrationError(
            "RGB and depth aspect ratios differ; explicit RGB-depth "
            "registration is required instead of simple scaling"
        )
    matrix = np.asarray(camera_matrix, dtype=float).copy()
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("camera_matrix must be a finite 3x3 matrix")
    matrix[0, :] *= sx
    matrix[1, :] *= sy
    matrix[2, :] = [0.0, 0.0, 1.0]
    return matrix


def _tag_mask_at_depth_resolution(
    detections: Sequence[TagCorners],
    *,
    image_size_px: tuple[int, int],
    depth_size_px: tuple[int, int],
    single_tag_scale: float,
) -> np.ndarray:
    image_width, image_height = image_size_px
    depth_width, depth_height = depth_size_px
    scale = np.asarray(
        [depth_width / image_width, depth_height / image_height], dtype=float
    )
    polygons = [
        np.asarray(item.corners_px, dtype=float).reshape(4, 2) * scale
        for item in detections
    ]
    if not polygons:
        return np.zeros((depth_height, depth_width), dtype=np.uint8)
    if len(polygons) == 1:
        polygon = polygons[0]
        center = np.mean(polygon, axis=0)
        polygon = center + single_tag_scale * (polygon - center)
    else:
        polygon = cv2.convexHull(
            np.concatenate(polygons).astype(np.float32)
        ).reshape(-1, 2)
    polygon[:, 0] = np.clip(polygon[:, 0], 0, depth_width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, depth_height - 1)
    mask = np.zeros((depth_height, depth_width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.rint(polygon).astype(np.int32), 1)
    # Boundary depth commonly belongs to the background rather than the board.
    if np.count_nonzero(mask) > 80:
        mask = cv2.erode(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    return mask


def depth_points_for_floor_tags(
    depth_m: np.ndarray,
    camera_matrix_rgb: np.ndarray,
    detections: Sequence[TagCorners],
    *,
    image_size_px: tuple[int, int],
    confidence: np.ndarray | None = None,
    options: RGBDCalibrationOptions | None = None,
) -> np.ndarray:
    """Unproject registered depth samples on/among the visible floor tags."""
    options = options or RGBDCalibrationOptions()
    depth = np.asarray(depth_m, dtype=float)
    if depth.ndim != 2:
        raise ValueError("depth_m must be a 2-D array")
    depth_height, depth_width = depth.shape
    mask = _tag_mask_at_depth_resolution(
        detections,
        image_size_px=image_size_px,
        depth_size_px=(depth_width, depth_height),
        single_tag_scale=options.single_tag_mask_scale,
    ).astype(bool)
    valid = (
        mask
        & np.isfinite(depth)
        & (depth >= options.min_depth_m)
        & (depth <= options.max_depth_m)
    )
    if confidence is not None and np.asarray(confidence).size:
        confidence_array = np.asarray(confidence)
        if confidence_array.shape != depth.shape:
            raise ValueError("confidence must have the same shape as depth_m")
        valid &= confidence_array >= options.min_confidence
    rows, columns = np.nonzero(valid)
    if len(rows) < options.min_depth_samples:
        raise RGBDCalibrationError(
            f"only {len(rows)} usable depth samples on the calibration plane; "
            f"need {options.min_depth_samples}"
        )
    if len(rows) > options.max_depth_samples:
        indexes = np.linspace(
            0, len(rows) - 1, options.max_depth_samples, dtype=int
        )
        rows = rows[indexes]
        columns = columns[indexes]
    matrix_depth = scale_camera_matrix(
        camera_matrix_rgb,
        image_size_px,
        (depth_width, depth_height),
    )
    values = depth[rows, columns]
    x = (columns - matrix_depth[0, 2]) * values / matrix_depth[0, 0]
    y = (rows - matrix_depth[1, 2]) * values / matrix_depth[1, 1]
    return np.column_stack([x, y, values])


def fit_depth_plane(
    points_camera_m: np.ndarray,
    *,
    expected_normal_camera: np.ndarray | None = None,
    threshold_m: float = 0.018,
    iterations: int = 250,
) -> DepthPlaneFit:
    """Fit a plane using deterministic RANSAC followed by SVD refinement."""
    points = np.asarray(points_camera_m, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_camera_m must have shape (n, 3)")
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 3:
        raise RGBDCalibrationError("at least three finite depth points are required")
    expected = None
    if expected_normal_camera is not None:
        expected = np.asarray(expected_normal_camera, dtype=float).reshape(3)
        expected /= max(1e-12, float(np.linalg.norm(expected)))

    rng = np.random.default_rng(41721)
    best_mask: np.ndarray | None = None
    best_score = (-1, -float("inf"))
    for _ in range(iterations):
        sample = points[rng.choice(len(points), size=3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = float(np.linalg.norm(normal))
        if norm < 1e-9:
            continue
        normal /= norm
        offset = -float(np.dot(normal, sample[0]))
        distances = np.abs(points @ normal + offset)
        mask = distances <= threshold_m
        count = int(np.count_nonzero(mask))
        median = float(np.median(distances[mask])) if count else float("inf")
        score = (count, -median)
        if score > best_score:
            best_score = score
            best_mask = mask
    if best_mask is None or np.count_nonzero(best_mask) < 3:
        raise RGBDCalibrationError("depth RANSAC could not find a plane")

    inliers = points[best_mask]
    center = np.mean(inliers, axis=0)
    _u, _singular, vh = np.linalg.svd(inliers - center, full_matrices=False)
    normal = vh[-1]
    normal /= max(1e-12, float(np.linalg.norm(normal)))
    if expected is not None and float(np.dot(normal, expected)) < 0.0:
        normal = -normal
    offset = -float(np.dot(normal, center))
    distances = np.abs(points @ normal + offset)
    final_mask = distances <= threshold_m
    final_distances = distances[final_mask]
    if len(final_distances) < 3:
        raise RGBDCalibrationError("depth plane refinement lost its inliers")
    rms = math.sqrt(float(np.mean(final_distances * final_distances)))
    return DepthPlaneFit(
        normal_camera=normal,
        offset_m=offset,
        points_camera_m=points,
        inlier_mask=final_mask,
        sample_count=len(points),
        inlier_count=int(np.count_nonzero(final_mask)),
        inlier_fraction=float(np.mean(final_mask)),
        rms_m=rms,
    )


def _world_tag_correspondences(
    detections: Sequence[TagCorners],
    floor_tags: Mapping[int, RigidTransform],
    marker_size_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    tag_points = marker_object_corners(marker_size_m)
    world_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    for detection in detections:
        if detection.tag_id not in floor_tags:
            continue
        world_from_tag = floor_tags[detection.tag_id]
        world_points.extend(world_from_tag.apply(point) for point in tag_points)
        image_points.extend(np.asarray(detection.corners_px, dtype=float))
    return np.asarray(world_points, dtype=float), np.asarray(image_points, dtype=float)


def _reprojection_rms(
    world_points: np.ndarray,
    image_points: np.ndarray,
    camera_from_world: RigidTransform,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> float:
    projected, _ = cv2.projectPoints(
        world_points,
        camera_from_world.rotation.as_rotvec(),
        camera_from_world.translation_m,
        camera_matrix,
        distortion,
    )
    error = projected.reshape(-1, 2) - image_points
    return math.sqrt(float(np.mean(np.sum(error * error, axis=1))))


def refine_world_reference_with_depth(
    detections: Sequence[TagCorners],
    floor_tags: Mapping[int, RigidTransform],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    depth_m: np.ndarray,
    *,
    image_size_px: tuple[int, int],
    confidence: np.ndarray | None = None,
    marker_size_m: float = DEFAULT_MARKER_SIZE_M,
    previous_world_from_camera: RigidTransform | None = None,
    preferred_floor_normal_camera: np.ndarray | None = None,
    options: RGBDCalibrationOptions | None = None,
) -> RGBDWorldReference:
    """Refine the existing mapped-tag PnP solve with a LiDAR plane constraint."""
    options = options or RGBDCalibrationOptions()
    visible = [item for item in detections if item.tag_id in floor_tags]
    if not visible:
        raise RGBDCalibrationError("no mapped floor/calibration tags are visible")
    rgb_reference = estimate_world_reference(
        visible,
        floor_tags,
        camera_matrix,
        distortion,
        marker_size_m=marker_size_m,
        previous_world_from_camera=previous_world_from_camera,
        preferred_floor_normal_camera=preferred_floor_normal_camera,
    )
    if rgb_reference is None:
        raise RGBDCalibrationError("RGB AprilTag world-reference solve failed")

    camera_from_world_initial = rgb_reference.world_from_camera.inverse()
    expected_normal = camera_from_world_initial.rotation.apply([0.0, 0.0, 1.0])
    if preferred_floor_normal_camera is not None:
        expected_normal = np.asarray(
            preferred_floor_normal_camera, dtype=float
        ).reshape(3)
        expected_normal /= max(1e-12, float(np.linalg.norm(expected_normal)))
    depth_points = depth_points_for_floor_tags(
        depth_m,
        camera_matrix,
        visible,
        image_size_px=image_size_px,
        confidence=confidence,
        options=options,
    )
    plane = fit_depth_plane(
        depth_points,
        expected_normal_camera=expected_normal,
        threshold_m=options.plane_ransac_threshold_m,
    )
    normal_dot = float(np.clip(
        np.dot(plane.normal_camera, expected_normal), -1.0, 1.0
    ))
    plane_normal_error_deg = math.degrees(math.acos(normal_dot))
    if plane.inlier_fraction < options.min_plane_inlier_fraction:
        raise RGBDCalibrationError(
            f"depth plane inlier fraction {plane.inlier_fraction:.3f} is below "
            f"{options.min_plane_inlier_fraction:.3f}"
        )
    if plane.rms_m > options.max_plane_rms_m:
        raise RGBDCalibrationError(
            f"depth plane RMS {plane.rms_m * 1000.0:.1f} mm exceeds "
            f"{options.max_plane_rms_m * 1000.0:.1f} mm"
        )
    if plane_normal_error_deg > options.max_plane_normal_error_deg:
        raise RGBDCalibrationError(
            f"depth and RGB plane normals disagree by {plane_normal_error_deg:.1f} deg"
        )

    world_points, image_points = _world_tag_correspondences(
        visible, floor_tags, marker_size_m
    )
    inlier_points = plane.points_camera_m[plane.inlier_mask]
    rgb_residual_count = max(1, 2 * len(world_points))
    depth_balance = math.sqrt(rgb_residual_count / max(1, len(inlier_points)))

    initial = np.concatenate([
        camera_from_world_initial.rotation.as_rotvec(),
        camera_from_world_initial.translation_m,
    ])

    def residual(parameters: np.ndarray) -> np.ndarray:
        rotation = Rotation.from_rotvec(parameters[:3])
        translation = parameters[3:]
        projected, _ = cv2.projectPoints(
            world_points,
            parameters[:3],
            translation,
            camera_matrix,
            distortion,
        )
        pixel_error = (
            projected.reshape(-1, 2) - image_points
        ).reshape(-1) / options.rgb_sigma_px
        # All calibration tags lie in world z=0.  Under camera_from_world, its
        # normal is R[:, 2] and any point on it satisfies n.(p - t) == 0.
        normal = rotation.apply([0.0, 0.0, 1.0])
        plane_error = (inlier_points - translation) @ normal
        plane_error *= (
            options.depth_weight * depth_balance / options.depth_sigma_m
        )
        return np.concatenate([pixel_error, plane_error])

    solved = least_squares(
        residual,
        initial,
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=120,
    )
    if not solved.success or not np.all(np.isfinite(solved.x)):
        raise RGBDCalibrationError(f"joint RGB-D optimization failed: {solved.message}")
    camera_from_world = RigidTransform(
        solved.x[3:], Rotation.from_rotvec(solved.x[:3])
    )
    world_from_camera = camera_from_world.inverse()
    reprojection_rms = _reprojection_rms(
        world_points,
        image_points,
        camera_from_world,
        camera_matrix,
        distortion,
    )
    fitted_normal = camera_from_world.rotation.apply([0.0, 0.0, 1.0])
    depth_residuals = (inlier_points - camera_from_world.translation_m) @ fitted_normal
    depth_rms = math.sqrt(float(np.mean(depth_residuals * depth_residuals)))
    fitted_plane_dot = float(np.clip(
        abs(np.dot(fitted_normal, plane.normal_camera)), -1.0, 1.0
    ))
    fitted_plane_error_deg = math.degrees(math.acos(fitted_plane_dot))
    return RGBDWorldReference(
        world_from_camera=world_from_camera,
        floor_tag_ids=tuple(sorted(item.tag_id for item in visible)),
        camera_matrix=np.asarray(camera_matrix, dtype=float).copy(),
        image_size_px=image_size_px,
        depth_size_px=(int(depth_m.shape[1]), int(depth_m.shape[0])),
        rgb_only_reprojection_rms_px=rgb_reference.reprojection_rms_px,
        reprojection_rms_px=reprojection_rms,
        depth_plane_rms_m=depth_rms,
        depth_samples=plane.sample_count,
        depth_inliers=plane.inlier_count,
        depth_inlier_fraction=plane.inlier_fraction,
        plane_normal_error_deg=fitted_plane_error_deg,
    )


def average_fixed_camera_calibration(
    observations: Sequence[RGBDWorldReference],
    *,
    input_frames: int | None = None,
    min_frames: int = 8,
    max_translation_outlier_m: float = 0.025,
    max_rotation_outlier_deg: float = 2.0,
) -> RGBDSessionCalibration:
    """Robustly average repeated observations from one stationary camera."""
    if len(observations) < min_frames:
        raise RGBDCalibrationError(
            f"only {len(observations)} accepted calibration frames; need {min_frames}"
        )
    image_sizes = {item.image_size_px for item in observations}
    depth_sizes = {item.depth_size_px for item in observations}
    if len(image_sizes) != 1 or len(depth_sizes) != 1:
        raise RGBDCalibrationError("stream dimensions changed during calibration")

    translations = np.stack([
        item.world_from_camera.translation_m for item in observations
    ])
    rotations = Rotation.from_quat(np.stack([
        item.world_from_camera.rotation.as_quat() for item in observations
    ]))
    center_translation = np.median(translations, axis=0)
    center_rotation = rotations.mean()
    translation_error = np.linalg.norm(
        translations - center_translation, axis=1
    )
    rotation_error_deg = np.degrees((center_rotation.inv() * rotations).magnitude())
    keep = (
        (translation_error <= max_translation_outlier_m)
        & (rotation_error_deg <= max_rotation_outlier_deg)
    )
    if int(np.count_nonzero(keep)) < min_frames:
        raise RGBDCalibrationError(
            "camera or calibration board moved: too few mutually consistent frames"
        )
    accepted = [item for item, include in zip(observations, keep) if include]
    weights = np.asarray([item.quality_weight() for item in accepted])
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
    )))
    rotation_spread = math.sqrt(float(np.average(
        np.degrees((rotation.inv() * rotations).magnitude()) ** 2,
        weights=weights,
    )))
    matrices = np.stack([item.camera_matrix for item in accepted])
    all_tag_ids = sorted({
        tag_id for item in accepted for tag_id in item.floor_tag_ids
    })
    return RGBDSessionCalibration(
        world_from_camera=RigidTransform(translation, rotation),
        camera_matrix=np.median(matrices, axis=0),
        image_size_px=accepted[0].image_size_px,
        depth_size_px=accepted[0].depth_size_px,
        input_frames=len(observations) if input_frames is None else input_frames,
        accepted_frames=len(accepted),
        rejected_outlier_frames=len(observations) - len(accepted),
        translation_spread_mm=translation_spread * 1000.0,
        rotation_spread_deg=rotation_spread,
        median_reprojection_rms_px=float(np.median([
            item.reprojection_rms_px for item in accepted
        ])),
        median_depth_plane_rms_mm=float(np.median([
            item.depth_plane_rms_m for item in accepted
        ])) * 1000.0,
        floor_tag_ids=tuple(all_tag_ids),
    )
