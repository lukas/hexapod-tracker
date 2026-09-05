"""Offline image-first refinement for a zero-pose iPhone survey."""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from .apriltag_vision import marker_object_corners
from .housing_pose import (
    JOINT_NAMES,
    HexapodGeometry,
    RigidTransform,
    forward_frame_transforms,
)
from .tag_survey import arkit_world_from_opencv_camera


@dataclass(frozen=True)
class _ArchivedFrame:
    name: str
    captured_unix: float
    camera_matrix: np.ndarray
    arkit_world_from_camera: RigidTransform
    detections: tuple[tuple[int, np.ndarray], ...]
    image_size_px: tuple[int, int]
    depth_m: np.ndarray | None
    confidence: np.ndarray | None
    trajectory_segment: int = 0


@dataclass(frozen=True)
class _DepthConstraint:
    frame_index: int
    tag_id: int
    point_camera_m: np.ndarray
    sigma_m: float
    sample_count: int
    confidence_level: int
    incidence_cosine: float


def _transform_values(transform: RigidTransform) -> np.ndarray:
    return np.concatenate([
        transform.rotation.as_rotvec(),
        transform.translation_m,
    ])


def _transform_from_values(values: np.ndarray) -> RigidTransform:
    return RigidTransform(
        np.asarray(values[3:6], dtype=float),
        Rotation.from_rotvec(np.asarray(values[:3], dtype=float)),
    )


def _rotation_from_mount(spec: Mapping[str, Any]) -> Rotation:
    raw = spec.get("frame_from_tag", {})
    if raw.get("quaternion_xyzw") is not None:
        return Rotation.from_quat(raw["quaternion_xyzw"])
    return Rotation.from_euler(
        "xyz", raw.get("euler_xyz_deg", [0.0, 0.0, 0.0]), degrees=True
    )


def _mount_group(spec: Mapping[str, Any]) -> str:
    kind = str(spec.get("kind", ""))
    frame = str(spec.get("frame", ""))
    if kind == "chassis_tag" or frame == "body":
        return "chassis"
    joint = str(spec.get("joint", "mount"))
    if kind == "servo_lid":
        return f"{joint}_top"
    return f"{joint}_{spec.get('mount_side', 'side')}"


def _rotation_on_expected_surface(
    measured: Rotation,
    configured: Rotation,
) -> Rotation:
    """Keep measured in-plane orientation on the configured physical face."""
    normal = configured.apply([0.0, 0.0, 1.0])
    tag_x = measured.apply([1.0, 0.0, 0.0])
    tag_x = tag_x - float(tag_x @ normal) * normal
    magnitude = float(np.linalg.norm(tag_x))
    if magnitude < 1e-6:
        tag_x = configured.apply([1.0, 0.0, 0.0])
        tag_x = tag_x - float(tag_x @ normal) * normal
        magnitude = float(np.linalg.norm(tag_x))
    tag_x /= magnitude
    tag_y = np.cross(normal, tag_x)
    return Rotation.from_matrix(np.column_stack([tag_x, tag_y, normal]))


def _assigned_robot_specs(
    tracker_config: Mapping[str, Any],
    survey: Mapping[str, Any],
) -> dict[int, dict[str, Any]]:
    configured = {
        int(tag_id): dict(spec)
        for tag_id, spec in tracker_config.get("robot_pose", {}).get(
            "tags", {}
        ).items()
    }
    assignments = survey.get("robot_positions", [])
    if not assignments:
        return {
            tag_id: {**spec, "_configured_tag_id": tag_id}
            for tag_id, spec in configured.items()
        }
    result: dict[int, dict[str, Any]] = {}
    for item in assignments:
        if item.get("state") != "measured" or item.get("tag_id") is None:
            continue
        configured_id = int(item["configured_tag_id"])
        if configured_id not in configured:
            continue
        result[int(item["tag_id"])] = {
            **configured[configured_id],
            "_configured_tag_id": configured_id,
        }
    return result


def _load_frames(
    directory: Path,
    known_ids: set[int],
) -> list[_ArchivedFrame]:
    frames: list[_ArchivedFrame] = []
    for path in sorted(directory.glob("*.npz")):
        with np.load(path) as raw:
            pose = np.asarray(raw["camera_pose_xyzw_xyz"], dtype=float)
            if pose.shape != (7,):
                continue
            detections = tuple(
                (int(tag_id), np.asarray(corners, dtype=float))
                for tag_id, corners in zip(
                    raw["tag_ids"], raw["tag_corners_px"]
                )
                if int(tag_id) in known_ids
            )
            if not detections:
                continue
            arkit_gl = RigidTransform(
                pose[4:], Rotation.from_quat(pose[:4])
            )
            image_size_raw = raw.get("image_size_px")
            if image_size_raw is not None:
                image_size_values = np.asarray(
                    image_size_raw, dtype=int
                ).reshape(-1)
                image_size_px = (
                    int(image_size_values[0]), int(image_size_values[1])
                )
            elif "rgb_jpeg" in raw:
                image = cv2.imdecode(
                    np.asarray(raw["rgb_jpeg"], dtype=np.uint8),
                    cv2.IMREAD_GRAYSCALE,
                )
                if image is None:
                    continue
                image_size_px = (int(image.shape[1]), int(image.shape[0]))
            else:
                matrix = np.asarray(raw["camera_matrix"], dtype=float)
                # Synthetic and legacy archives did not record the RGB size.
                # Their principal point is centered, so this is the least
                # surprising backwards-compatible estimate. Depth is ignored
                # for such a frame unless its aspect ratio agrees below.
                image_size_px = (
                    max(1, int(round(2.0 * matrix[0, 2]))),
                    max(1, int(round(2.0 * matrix[1, 2]))),
                )
            depth = (
                np.asarray(raw["depth"], dtype=float)
                if "depth" in raw and np.asarray(raw["depth"]).size
                else None
            )
            confidence = (
                np.asarray(raw["confidence"])
                if "confidence" in raw and np.asarray(raw["confidence"]).size
                else None
            )
            if depth is not None:
                if depth.ndim != 2:
                    depth = None
                    confidence = None
                elif confidence is not None and confidence.shape != depth.shape:
                    confidence = None
            frames.append(_ArchivedFrame(
                name=path.name,
                captured_unix=float(raw.get(
                    "captured_unix", len(frames) * 5.0
                )),
                camera_matrix=np.asarray(raw["camera_matrix"], dtype=float),
                arkit_world_from_camera=arkit_world_from_opencv_camera(
                    arkit_gl
                ),
                detections=detections,
                image_size_px=image_size_px,
                depth_m=depth,
                confidence=confidence,
            ))
    segmented: list[_ArchivedFrame] = []
    segment = 0
    for index, frame in enumerate(frames):
        if index > 0:
            delta = frame.captured_unix - frames[index - 1].captured_unix
            if not 0.0 < delta <= 15.0:
                segment += 1
        segmented.append(replace(frame, trajectory_segment=segment))
    return segmented


def _select_keyframes(
    frames: Sequence[_ArchivedFrame], *, maximum_frames: int = 32
) -> list[_ArchivedFrame]:
    """Keep high-resolution views with coverage and camera-pose diversity.

    The capture loop intentionally archives often so a brief blur or missed
    decode does not lose a physical position. Bundle adjustment does not gain
    useful information from dozens of nearly identical half-second frames,
    though, and per-frame camera variables make those duplicates expensive.
    """
    if len(frames) <= maximum_frames:
        return list(frames)

    tags_by_frame = [
        {
            tag_id: float(np.mean([
                np.linalg.norm(corners[(index + 1) % 4] - corners[index])
                for index in range(4)
            ]))
            for tag_id, corners in frame.detections
        }
        for frame in frames
    ]
    available_by_tag: dict[int, int] = {}
    for tags in tags_by_frame:
        for tag_id in tags:
            available_by_tag[tag_id] = available_by_tag.get(tag_id, 0) + 1
    target_by_tag = {
        tag_id: min(3, count) for tag_id, count in available_by_tag.items()
    }

    selected: set[int] = {0, len(frames) - 1}
    for index in range(len(frames) - 1):
        if frames[index + 1].trajectory_segment != frames[index].trajectory_segment:
            selected.update((index, index + 1))

    def tag_counts() -> dict[int, int]:
        counts: dict[int, int] = {}
        for index in selected:
            for tag_id in tags_by_frame[index]:
                counts[tag_id] = counts.get(tag_id, 0) + 1
        return counts

    while len(selected) < maximum_frames:
        counts = tag_counts()
        if all(
            counts.get(tag_id, 0) >= target
            for tag_id, target in target_by_tag.items()
        ):
            break
        best_index: int | None = None
        best_score = -1.0
        for index, tags in enumerate(tags_by_frame):
            if index in selected:
                continue
            score = 0.0
            for tag_id, edge_px in tags.items():
                missing = target_by_tag[tag_id] - counts.get(tag_id, 0)
                if missing <= 0:
                    continue
                rarity = 1.0 + 8.0 / available_by_tag[tag_id]
                clarity = 1.0 + min(edge_px / 45.0, 2.0)
                score += missing * rarity * clarity
            if score > best_score:
                best_index = index
                best_score = score
        if best_index is None or best_score <= 0.0:
            break
        selected.add(best_index)

    # Spend any remaining budget on poses farthest from what is already kept.
    # Translation and orientation both matter for recovering depth and tag
    # surface normals; this does not resize or otherwise weaken source pixels.
    while len(selected) < maximum_frames:
        best_index = None
        best_score = -1.0
        for index, frame in enumerate(frames):
            if index in selected:
                continue
            nearest = min(
                np.linalg.norm(
                    frame.arkit_world_from_camera.translation_m
                    - frames[other].arkit_world_from_camera.translation_m
                ) / 0.08
                + (
                    frame.arkit_world_from_camera.rotation.inv()
                    * frames[other].arkit_world_from_camera.rotation
                ).magnitude() / math.radians(12.0)
                for other in selected
            )
            score = float(nearest) + 0.03 * len(tags_by_frame[index])
            if score > best_score:
                best_index = index
                best_score = score
        if best_index is None:
            break
        selected.add(best_index)
    return [frames[index] for index in sorted(selected)]


def _depth_constraint(
    frame: _ArchivedFrame,
    *,
    frame_index: int,
    tag_id: int,
    pixels: np.ndarray,
    world_from_tag: RigidTransform,
    world_from_camera: RigidTransform,
) -> _DepthConstraint | None:
    """Summarize registered LiDAR samples near one tag's visual center.

    Scene depth is far coarser than RGB. A single robust central point avoids
    pretending that upsampled depth can locate the four tag corners. The point
    becomes a point-to-predicted-tag-plane factor in the final bundle.
    """
    if frame.depth_m is None:
        return None
    depth = frame.depth_m
    depth_height, depth_width = depth.shape
    image_width, image_height = frame.image_size_px
    if min(depth_width, depth_height, image_width, image_height) <= 0:
        return None
    if not math.isclose(
        depth_width / depth_height,
        image_width / image_height,
        rel_tol=0.02,
        abs_tol=0.0,
    ):
        return None

    corners = np.asarray(pixels, dtype=float)
    mean_edge_px = float(np.mean([
        np.linalg.norm(corners[(index + 1) % 4] - corners[index])
        for index in range(4)
    ]))
    mean_edge_depth_px = mean_edge_px * depth_width / image_width
    if mean_edge_depth_px < 7.0:
        return None

    camera_from_tag = world_from_camera.inverse().compose(world_from_tag)
    tag_to_camera = -camera_from_tag.translation_m
    distance = float(np.linalg.norm(tag_to_camera))
    if distance <= 1e-6:
        return None
    incidence_cosine = abs(float(
        camera_from_tag.rotation.apply([0.0, 0.0, 1.0])
        @ (tag_to_camera / distance)
    ))
    if incidence_cosine < 0.55:
        return None

    scale = np.asarray([
        depth_width / image_width, depth_height / image_height
    ])
    depth_corners = corners * scale
    center = np.mean(depth_corners, axis=0)
    # Stay well inside the black square. Edge pixels frequently contain the
    # lid, carpet, or background because RGB/depth registration is coarse.
    inner = center + 0.38 * (depth_corners - center)
    mask = np.zeros(depth.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.rint(inner).astype(np.int32), 1)
    finite = (
        (mask != 0)
        & np.isfinite(depth)
        & (depth >= 0.12)
        & (depth <= 2.0)
    )
    confidence_level = 0
    if frame.confidence is not None:
        high = finite & (frame.confidence >= 2)
        medium = finite & (frame.confidence >= 1)
        if int(np.count_nonzero(high)) >= 6:
            finite = high
            confidence_level = 2
        elif int(np.count_nonzero(medium)) >= 6:
            finite = medium
            confidence_level = 1
        else:
            return None
    rows, columns = np.nonzero(finite)
    if len(rows) < 6:
        return None

    values = depth[rows, columns]
    center_depth = float(np.median(values))
    mad = float(np.median(np.abs(values - center_depth)))
    robust_sigma = max(0.004, 1.4826 * mad)
    keep = np.abs(values - center_depth) <= max(0.012, 3.0 * robust_sigma)
    rows = rows[keep]
    columns = columns[keep]
    values = values[keep]
    if len(rows) < 6:
        return None
    center_depth = float(np.median(values))
    center_column = float(np.median(columns))
    center_row = float(np.median(rows))
    matrix = frame.camera_matrix.copy()
    matrix[0, :] *= depth_width / image_width
    matrix[1, :] *= depth_height / image_height
    matrix[2, :] = [0.0, 0.0, 1.0]
    point = np.asarray([
        (center_column - matrix[0, 2]) * center_depth / matrix[0, 0],
        (center_row - matrix[1, 2]) * center_depth / matrix[1, 1],
        center_depth,
    ])
    # Six millimetres is already optimistic for registered phone scene depth
    # on a small object. Larger observed dispersion automatically weakens the
    # factor rather than letting a mixed edge pixel bend the reconstruction.
    sigma_m = min(0.025, max(0.006, 1.4826 * mad))
    return _DepthConstraint(
        frame_index=frame_index,
        tag_id=tag_id,
        point_camera_m=point,
        sigma_m=sigma_m,
        sample_count=int(len(values)),
        confidence_level=confidence_level,
        incidence_cosine=incidence_cosine,
    )


def _project(
    world_from_tag: RigidTransform,
    world_from_camera: RigidTransform,
    camera_matrix: np.ndarray,
    object_corners: np.ndarray,
) -> np.ndarray:
    world_points = (
        object_corners @ world_from_tag.rotation.as_matrix().T
        + world_from_tag.translation_m
    )
    camera_from_world = world_from_camera.inverse()
    camera_points = (
        world_points @ camera_from_world.rotation.as_matrix().T
        + camera_from_world.translation_m
    )
    depth = np.maximum(camera_points[:, 2], 1e-5)
    return np.column_stack([
        camera_matrix[0, 0] * camera_points[:, 0] / depth
        + camera_matrix[0, 2],
        camera_matrix[1, 1] * camera_points[:, 1] / depth
        + camera_matrix[1, 2],
    ])


def _write_reprojection_audit(
    *,
    frame_archive_dir: Path,
    frames: Sequence[_ArchivedFrame],
    observations: Sequence[tuple[int, int, np.ndarray]],
    cameras: Sequence[RigidTransform],
    robot_tags: Mapping[int, RigidTransform],
    floor_tags: Mapping[int, RigidTransform],
    robot_segment_motions: Sequence[RigidTransform],
    frame_segments: Sequence[int],
    object_corners: Mapping[int, np.ndarray],
    fit_observation_indices: Sequence[int],
) -> dict[str, Any]:
    """Render detected and predicted tag corners on representative photos."""
    fit_set = set(fit_observation_indices)
    per_frame: dict[int, list[tuple[int, np.ndarray, np.ndarray, bool, float]]] = {}
    for observation_index, (frame_index, tag_id, detected) in enumerate(
        observations
    ):
        canonical_tag = floor_tags.get(tag_id, robot_tags.get(tag_id))
        if canonical_tag is None:
            continue
        observed_tag = (
            canonical_tag
            if tag_id in floor_tags
            else robot_segment_motions[frame_segments[frame_index]].compose(
                canonical_tag
            )
        )
        predicted = _project(
            observed_tag,
            cameras[frame_index],
            frames[frame_index].camera_matrix,
            object_corners[tag_id],
        )
        coordinate_rms = float(np.sqrt(np.mean((predicted - detected) ** 2)))
        per_frame.setdefault(frame_index, []).append((
            tag_id,
            detected,
            predicted,
            observation_index in fit_set,
            coordinate_rms,
        ))
    candidates = sorted(per_frame)
    if not candidates:
        return {"available": False, "reason": "no projected observations"}

    frame_rms = {
        frame_index: float(np.sqrt(np.mean([
            item[4] ** 2 for item in items if item[3]
        ])))
        for frame_index, items in per_frame.items()
        if any(item[3] for item in items)
    }
    evenly_spaced = {
        candidates[int(round(value))]
        for value in np.linspace(0, len(candidates) - 1, min(8, len(candidates)))
    }
    worst = {
        frame_index
        for frame_index, _value in sorted(
            frame_rms.items(), key=lambda item: item[1], reverse=True
        )[:8]
    }
    selected = sorted(evenly_spaced | worst)
    audit_dir = frame_archive_dir.parent / "reprojection-audit-v5"
    audit_dir.mkdir(parents=True, exist_ok=True)
    written: list[dict[str, Any]] = []
    for frame_index in selected:
        path = frame_archive_dir / frames[frame_index].name
        try:
            with np.load(path, allow_pickle=False) as raw:
                if "rgb_jpeg" not in raw:
                    continue
                image = cv2.imdecode(
                    np.asarray(raw["rgb_jpeg"], dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
        except (OSError, ValueError):
            continue
        if image is None:
            continue
        for tag_id, detected, predicted, accepted, coordinate_rms in per_frame[
            frame_index
        ]:
            detected_polygon = np.rint(detected).astype(np.int32).reshape(-1, 1, 2)
            predicted_polygon = np.rint(predicted).astype(np.int32).reshape(-1, 1, 2)
            detected_color = (45, 195, 45) if accepted else (0, 135, 255)
            cv2.polylines(image, [detected_polygon], True, detected_color, 3)
            cv2.polylines(image, [predicted_polygon], True, (210, 55, 210), 2)
            for detected_point, predicted_point in zip(detected, predicted):
                cv2.line(
                    image,
                    tuple(np.rint(detected_point).astype(int)),
                    tuple(np.rint(predicted_point).astype(int)),
                    (0, 210, 255),
                    1,
                    cv2.LINE_AA,
                )
            label_at = tuple(np.rint(detected[0] + [2.0, -7.0]).astype(int))
            cv2.putText(
                image,
                f"#{tag_id} {coordinate_rms:.1f}px" + ("" if accepted else " OUT"),
                label_at,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                detected_color,
                2,
                cv2.LINE_AA,
            )
        headline = (
            "detected green | predicted magenta | connector yellow | "
            f"frame RMS {frame_rms.get(frame_index, 0.0):.2f}px"
        )
        cv2.rectangle(
            image, (0, 0), (min(image.shape[1], 1120), 42), (20, 20, 20), -1
        )
        cv2.putText(
            image,
            headline,
            (12, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )
        output_path = audit_dir / f"audit-{frame_index:03d}.jpg"
        if cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, 91]):
            written.append({
                "frame": frames[frame_index].name,
                "trajectory_segment": int(frame_segments[frame_index]),
                "coordinate_rms_px": frame_rms.get(frame_index),
                "path": str(output_path),
            })
    return {
        "available": bool(written),
        "directory": str(audit_dir),
        "overlay_frames": written,
        "legend": {
            "detected_accepted": "green",
            "detected_rejected": "orange",
            "predicted": "magenta",
            "corner_error_connector": "yellow",
        },
    }


def _record_orientation_fields(
    record: dict[str, Any],
    transform: RigidTransform,
) -> None:
    tag_x = transform.rotation.apply([1.0, 0.0, 0.0])
    tag_y = transform.rotation.apply([0.0, 1.0, 0.0])
    normal = transform.rotation.apply([0.0, 0.0, 1.0])
    heading = math.degrees(math.atan2(float(tag_y[0]), float(tag_y[1])))
    normal_error = math.degrees(math.acos(float(np.clip(
        normal @ np.asarray([0.0, 0.0, 1.0]), -1.0, 1.0
    ))))
    record.update({
        "world_from_tag": transform.to_dict(),
        "euler_xyz_deg": [
            round(float(value), 6)
            for value in transform.rotation.as_euler("xyz", degrees=True)
        ],
        "tag_x_world": [round(float(value), 7) for value in tag_x],
        "tag_y_world": [round(float(value), 7) for value in tag_y],
        "tag_normal_world": [round(float(value), 7) for value in normal],
        "tag_y_heading_clockwise_from_world_y_deg": round(heading, 5),
        "height_above_ground_mm": round(
            float(transform.translation_m[2]) * 1000.0, 3
        ),
        "normal_error_from_world_up_deg": round(normal_error, 5),
    })


def refine_zero_pose_with_buildviz(
    tracker_config: Mapping[str, Any],
    survey: Mapping[str, Any],
    *,
    frame_archive_dir: Path,
    floor_tags: Mapping[int, RigidTransform],
    world_from_arkit_world: RigidTransform | None,
    body_anchor_tag_id: int,
    orientation_anchor_tag_id: int,
    floor_marker_size_m: float | None = None,
    floor_manifest: Mapping[str, Any] | None = None,
    joint_angles_deg: Mapping[str, float] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Refine a stationary survey primarily from archived photo corners.

    The online survey stays lightweight. BuildViz supplies initialization and
    planar-branch hints, while the final joint solve is driven by decoded image
    corners. One floor tag fixes the world frame; the remaining floor tags are
    measured as coplanar landmarks instead of being forced onto a stale map.
    """
    unchanged = copy.deepcopy(dict(survey))
    if world_from_arkit_world is None:
        return unchanged, {
            "ok": False,
            "skipped_reason": "no persistent floor-to-ARKit alignment",
        }
    records = {
        int(item["tag_id"]): item
        for item in survey.get("tags", [])
        if item.get("stable") and item.get("role") == "robot"
    }
    specs = _assigned_robot_specs(tracker_config, survey)
    body_anchor_candidates = [
        tag_id for tag_id, spec in specs.items()
        if tag_id == body_anchor_tag_id
        or int(spec.get("_configured_tag_id", tag_id)) == body_anchor_tag_id
    ]
    if not body_anchor_candidates or orientation_anchor_tag_id not in records:
        return unchanged, {
            "ok": False,
            "skipped_reason": (
                "stable chassis and unchanged L0 orientation anchors are required"
            ),
        }
    specs = {tag_id: specs[tag_id] for tag_id in records if tag_id in specs}
    body_anchor_tag_id = body_anchor_candidates[0]
    if orientation_anchor_tag_id not in specs:
        return unchanged, {
            "ok": False,
            "skipped_reason": "an anchor has no robot mount assignment",
        }
    archived_frames = _load_frames(
        Path(frame_archive_dir), set(specs) | set(floor_tags)
    )
    archived_frame_count = len(archived_frames)
    frames = _select_keyframes(archived_frames)
    observations = [
        (frame_index, tag_id, corners)
        for frame_index, frame in enumerate(frames)
        for tag_id, corners in frame.detections
    ]
    robot_observations = [item for item in observations if item[1] in specs]
    if len(frames) < 3 or len(robot_observations) < 12:
        return unchanged, {
            "ok": False,
            "skipped_reason": "too few archived multi-view tag observations",
            "frames": len(frames),
            "archived_frames": archived_frame_count,
            "robot_observations": len(robot_observations),
        }

    robot_config = tracker_config.get("robot_pose", {})
    geometry = HexapodGeometry.from_dict(robot_config.get("geometry"))
    angles = {name: 0.0 for name in JOINT_NAMES}
    if joint_angles_deg is not None:
        angles.update({
            str(name): float(value)
            for name, value in joint_angles_deg.items()
        })
    body_zero = RigidTransform(np.zeros(3), Rotation.identity())
    zero_frames = forward_frame_transforms(
        body_zero, angles, geometry=geometry
    )
    body_spec = specs[body_anchor_tag_id]
    l0_spec = specs[orientation_anchor_tag_id]
    world_from_body_tag = RigidTransform.from_dict(
        records[body_anchor_tag_id]["world_from_tag"]
    )
    world_from_l0_tag = RigidTransform.from_dict(
        records[orientation_anchor_tag_id]["world_from_tag"]
    )
    body_from_body_tag = RigidTransform.from_dict(
        body_spec.get("frame_from_tag", {})
    )
    body_from_l0_tag = zero_frames[str(l0_spec["frame"])].compose(
        RigidTransform.from_dict(l0_spec.get("frame_from_tag", {}))
    )
    body_rotation = (
        world_from_l0_tag.rotation * body_from_l0_tag.rotation.inv()
    )
    body_translation = (
        world_from_body_tag.translation_m
        - body_rotation.apply(body_from_body_tag.translation_m)
    )
    world_from_body_initial = RigidTransform(body_translation, body_rotation)

    group_samples: dict[str, list[np.ndarray]] = {}
    local_rotations: dict[int, Rotation] = {}
    measured_surface_rotations: dict[int, Rotation] = {}
    body_from_world = world_from_body_initial.inverse()
    for tag_id, spec in specs.items():
        world_from_tag = RigidTransform.from_dict(records[tag_id]["world_from_tag"])
        body_from_tag = body_from_world.compose(world_from_tag)
        frame_from_tag = zero_frames[str(spec["frame"])].inverse().compose(
            body_from_tag
        )
        group_samples.setdefault(_mount_group(spec), []).append(
            frame_from_tag.translation_m
        )
        configured_rotation = _rotation_from_mount(spec)
        measured_surface_rotations[tag_id] = _rotation_on_expected_surface(
            frame_from_tag.rotation, configured_rotation
        )
        # Repeated mount geometry uses the photographed layout's known face
        # orientation. The chassis tag is the exception: its stale historical
        # yaw is exactly what the L0 orientation anchor is meant to replace.
        local_rotations[tag_id] = (
            measured_surface_rotations[tag_id]
            if tag_id == body_anchor_tag_id else configured_rotation
        )
    group_names = sorted(group_samples)
    group_index = {name: index for index, name in enumerate(group_names)}
    group_initial = {
        name: np.median(np.stack(samples), axis=0)
        for name, samples in group_samples.items()
    }
    initial_values = _transform_values(world_from_body_initial)
    for name in group_names:
        initial_values = np.concatenate([initial_values, group_initial[name]])

    marker_sizes = {
        tag_id: float(spec.get(
            "marker_size_m", tracker_config.get("marker_size_m", 0.027)
        ))
        for tag_id, spec in specs.items()
    }
    marker_sizes.update({
        tag_id: float(floor_marker_size_m or tracker_config.get(
            "marker_size_m", 0.027
        ))
        for tag_id in floor_tags
    })
    object_corners = {
        tag_id: marker_object_corners(size)
        for tag_id, size in marker_sizes.items()
    }
    camera_initial = [
        world_from_arkit_world.compose(frame.arkit_world_from_camera)
        for frame in frames
    ]
    cameras = list(camera_initial)

    def body_at(values: np.ndarray) -> RigidTransform:
        return _transform_from_values(values[:6])

    def group_at(values: np.ndarray, name: str) -> np.ndarray:
        offset = 6 + group_index[name] * 3
        return values[offset:offset + 3]

    def physical_tag(values: np.ndarray, tag_id: int) -> RigidTransform:
        spec = specs[tag_id]
        frame_from_tag = RigidTransform(
            group_at(values, _mount_group(spec)),
            local_rotations[tag_id],
        )
        return body_at(values).compose(
            zero_frames[str(spec["frame"])].compose(frame_from_tag)
        )

    def physical_residual(values: np.ndarray) -> np.ndarray:
        output: list[float] = []
        for frame_index, tag_id, pixels in observations:
            tag = (
                floor_tags[tag_id]
                if tag_id in floor_tags
                else physical_tag(values, tag_id)
            )
            output.extend((_project(
                tag,
                cameras[frame_index],
                frames[frame_index].camera_matrix,
                object_corners[tag_id],
            ) - pixels).reshape(-1))
        body = body_at(values)
        output.extend(
            (body.translation_m - world_from_body_initial.translation_m) / 0.05
        )
        output.extend((
            world_from_body_initial.rotation.inv() * body.rotation
        ).as_rotvec() / np.radians(12.0))
        for name in group_names:
            output.extend((group_at(values, name) - group_initial[name]) / 0.025)
        return np.asarray(output)

    def refine_cameras(
        modeled_tags: Mapping[int, RigidTransform],
    ) -> list[RigidTransform]:
        result: list[RigidTransform] = []
        for index, frame in enumerate(frames):
            initial = camera_initial[index]

            def residual(values: np.ndarray) -> np.ndarray:
                camera = _transform_from_values(values)
                output: list[float] = []
                for tag_id, pixels in frame.detections:
                    tag = floor_tags.get(tag_id, modeled_tags.get(tag_id))
                    if tag is None:
                        continue
                    output.extend((_project(
                        tag, camera, frame.camera_matrix, object_corners[tag_id]
                    ) - pixels).reshape(-1))
                output.extend(
                    (camera.translation_m - initial.translation_m) / 0.05
                )
                output.extend((
                    initial.rotation.inv() * camera.rotation
                ).as_rotvec() / np.radians(6.0))
                return np.asarray(output)

            solved = least_squares(
                residual,
                _transform_values(cameras[index]),
                loss="huber",
                f_scale=2.0,
                x_scale="jac",
                max_nfev=100,
            )
            result.append(_transform_from_values(solved.x))
        return result

    model_values = initial_values
    initial_pixel_residual = physical_residual(model_values)[:len(observations) * 8]
    for _round in range(2):
        solved_model = least_squares(
            physical_residual,
            model_values,
            loss="huber",
            f_scale=2.0,
            x_scale="jac",
            max_nfev=140,
        )
        model_values = solved_model.x
        physical_tags = {
            tag_id: physical_tag(model_values, tag_id) for tag_id in specs
        }
        cameras = refine_cameras(physical_tags)
    solved_model = least_squares(
        physical_residual,
        model_values,
        loss="huber",
        f_scale=2.0,
        x_scale="jac",
        max_nfev=180,
    )
    model_values = solved_model.x
    physical_tags = {
        tag_id: physical_tag(model_values, tag_id) for tag_id in specs
    }
    physical_pixel_residual = physical_residual(model_values)[:len(observations) * 8]

    by_tag: dict[int, list[tuple[int, np.ndarray]]] = {}
    for frame_index, tag_id, pixels in robot_observations:
        by_tag.setdefault(tag_id, []).append((frame_index, pixels))

    def refine_individual_tags() -> dict[int, RigidTransform]:
        result: dict[int, RigidTransform] = {}
        for tag_id, physical in physical_tags.items():
            tag_observations = by_tag.get(tag_id, [])
            if not tag_observations:
                result[tag_id] = RigidTransform.from_dict(
                    records[tag_id]["world_from_tag"]
                )
                continue
            expected_normal = physical.rotation.apply([0.0, 0.0, 1.0])
            spec = specs[tag_id]
            measured_rotation = (
                body_at(model_values).rotation
                * zero_frames[str(spec["frame"])].rotation
                * measured_surface_rotations[tag_id]
            )
            initial = RigidTransform(
                physical.translation_m, measured_rotation
            )

            def residual(values: np.ndarray) -> np.ndarray:
                tag = _transform_from_values(values)
                output: list[float] = []
                for frame_index, pixels in tag_observations:
                    output.extend((_project(
                        tag,
                        cameras[frame_index],
                        frames[frame_index].camera_matrix,
                        object_corners[tag_id],
                    ) - pixels).reshape(-1))
                output.extend(
                    (tag.translation_m - physical.translation_m) / 0.012
                )
                output.extend((
                    tag.rotation.apply([0.0, 0.0, 1.0]) - expected_normal
                ) / math.sin(math.radians(5.0)))
                rotation_sigma = 12.0 if tag_id == orientation_anchor_tag_id else 60.0
                output.extend((
                    physical.rotation.inv() * tag.rotation
                ).as_rotvec() / np.radians(rotation_sigma))
                return np.asarray(output)

            solved = least_squares(
                residual,
                _transform_values(initial),
                loss="huber",
                f_scale=2.0,
                x_scale="jac",
                max_nfev=120,
            )
            result[tag_id] = _transform_from_values(solved.x)
        return result

    refined_tags = refine_individual_tags()
    cameras = refine_cameras(refined_tags)
    refined_tags = refine_individual_tags()

    # The photographs are the final authority.  Jointly move every archived
    # camera and observed tag to minimize decoded corner error. One floor tag
    # fixes the world-coordinate gauge; the other floor tags are measured with
    # coplanarity constraints. ARKit contributes relative motion, and BuildViz
    # is deliberately only initialization and a post-fit diagnostic.
    observed_tag_ids = sorted(by_tag)
    observed_tag_index = {
        tag_id: index for index, tag_id in enumerate(observed_tag_ids)
    }
    observed_floor_ids = sorted({
        tag_id for _frame_index, tag_id, _pixels in observations
        if tag_id in floor_tags
    })
    if not observed_floor_ids:
        return unchanged, {
            "ok": False,
            "skipped_reason": "no floor tag was decoded in the archived views",
            "frames": len(frames),
            "robot_observations": len(robot_observations),
        }
    floor_anchor_id = min(
        observed_floor_ids,
        key=lambda tag_id: float(np.linalg.norm(
            floor_tags[tag_id].translation_m
        )),
    )
    # The anchor names the output coordinate frame; it is not forced to its
    # pre-fit pose. The bundle measures it like every other floor tag, then the
    # complete reconstruction is rebased so this tag becomes identity.
    movable_floor_ids = list(observed_floor_ids)
    movable_floor_index = {
        tag_id: index for index, tag_id in enumerate(movable_floor_ids)
    }
    floor_manifest = floor_manifest or {}
    raw_floor_specs = floor_manifest.get("floor_tags", {})
    default_floor_position_sigma = floor_manifest.get(
        "position_uncertainty_m"
    )
    default_floor_yaw_sigma = floor_manifest.get("yaw_uncertainty_deg")
    floor_reference_status = str(
        floor_manifest.get("reference_status", "")
    ).lower()

    def floor_prior(tag_id: int) -> dict[str, Any]:
        raw = raw_floor_specs.get(str(tag_id), {})
        position_sigma = raw.get(
            "position_uncertainty_m", default_floor_position_sigma
        )
        yaw_sigma = raw.get(
            "yaw_uncertainty_deg", default_floor_yaw_sigma
        )
        tag_reference_status = str(raw.get("reference_status", "")).lower()
        effective_status = tag_reference_status or floor_reference_status
        measured = (
            effective_status == "surveyed"
            if effective_status
            else position_sigma is not None or yaw_sigma is not None
        )
        if not measured:
            position_sigma = None
            yaw_sigma = None
        return {
            "source": "measured_ground_truth" if measured else "loose_initial_map",
            "position_sigma_m": (
                0.10 if position_sigma is None else max(0.0005, float(position_sigma))
            ),
            "height_sigma_m": (
                0.003 if position_sigma is None
                else max(0.0005, float(raw.get(
                    "height_uncertainty_m", position_sigma
                )))
            ),
            "normal_sigma_deg": max(0.25, float(raw.get(
                "normal_uncertainty_deg", 3.0
            ))),
            "rotation_sigma_deg": (
                45.0 if yaw_sigma is None else max(0.25, float(yaw_sigma))
            ),
        }

    floor_priors = {
        tag_id: floor_prior(tag_id) for tag_id in movable_floor_ids
    }

    def append_floor_prior_residual(
        output: list[float],
        tag: RigidTransform,
        initial: RigidTransform,
        tag_id: int,
    ) -> None:
        prior = floor_priors[tag_id]
        output.extend(
            (tag.translation_m[:2] - initial.translation_m[:2])
            / prior["position_sigma_m"]
        )
        output.append(
            float(tag.translation_m[2] - initial.translation_m[2])
            / prior["height_sigma_m"]
        )
        output.extend((
            tag.rotation.apply([0.0, 0.0, 1.0])
            - initial.rotation.apply([0.0, 0.0, 1.0])
        ) / math.sin(math.radians(prior["normal_sigma_deg"])))
        output.extend((
            initial.rotation.inv() * tag.rotation
        ).as_rotvec() / np.radians(prior["rotation_sigma_deg"]))

    def refine_floor_landmarks() -> dict[int, RigidTransform]:
        result: dict[int, RigidTransform] = {}
        for tag_id in movable_floor_ids:
            tag_observations = [
                (frame_index, pixels)
                for frame_index, observed_id, pixels in observations
                if observed_id == tag_id
            ]
            initial = floor_tags[tag_id]

            def residual(values: np.ndarray) -> np.ndarray:
                tag = _transform_from_values(values)
                output: list[float] = []
                for frame_index, pixels in tag_observations:
                    output.extend((_project(
                        tag,
                        cameras[frame_index],
                        frames[frame_index].camera_matrix,
                        object_corners[tag_id],
                    ) - pixels).reshape(-1))
                append_floor_prior_residual(
                    output, tag, initial, tag_id
                )
                return np.asarray(output)

            solved = least_squares(
                residual,
                _transform_values(initial),
                loss="huber",
                f_scale=2.0,
                x_scale="jac",
                max_nfev=120,
            )
            result[tag_id] = _transform_from_values(solved.x)
        return result

    # ARKit relative motion is meaningful only within one uninterrupted USB
    # session. Long archive gaps split the motion prior at reconnects.
    trajectory_edges = [
        index
        for index in range(len(frames) - 1)
        if (
            frames[index + 1].trajectory_segment
            == frames[index].trajectory_segment
        )
    ]
    if len(trajectory_edges) == len(frames) - 1:
        # A continuous capture can use the clean raw ARKit trajectory as the
        # least-biased starting point.
        cameras = list(camera_initial)
        refined_tags = refine_individual_tags()
    # Across reconnects, keep the per-frame image-aligned camera estimates
    # from the preceding rounds. One global ARKit alignment cannot place
    # multiple reset world frames into the same coordinates.
    floor_initial_tags = refine_floor_landmarks()
    camera_count = len(frames)
    selected_segment_ids = sorted({
        frame.trajectory_segment for frame in frames
    })
    segment_remap = {
        source: index for index, source in enumerate(selected_segment_ids)
    }
    frame_segments = [
        segment_remap[frame.trajectory_segment] for frame in frames
    ]
    segment_count = len(selected_segment_ids)
    robot_observations_per_segment = [0] * segment_count
    floor_observations_per_segment = [0] * segment_count
    for frame_index, tag_id, _pixels in observations:
        target = (
            floor_observations_per_segment
            if tag_id in floor_tags else robot_observations_per_segment
        )
        target[frame_segments[frame_index]] += 1
    # The best-observed session defines the canonical placement of the robot.
    # Other sessions may contain a rigid nudge of the whole robot relative to
    # the floor. Modeling that explicitly is safer than throwing away a
    # coherent group of floor observations to preserve a false static scene.
    reference_robot_segment = max(
        range(segment_count),
        key=lambda index: (
            min(
                robot_observations_per_segment[index],
                floor_observations_per_segment[index],
            ),
            robot_observations_per_segment[index]
            + floor_observations_per_segment[index],
        ),
    )
    movable_robot_segments = [
        index for index in range(segment_count)
        if index != reference_robot_segment
    ]
    movable_robot_segment_index = {
        segment: index
        for index, segment in enumerate(movable_robot_segments)
    }
    robot_variable_offset = camera_count
    floor_variable_offset = camera_count + len(observed_tag_ids)
    robot_motion_variable_offset = (
        floor_variable_offset + len(movable_floor_ids)
    )

    depth_constraints: list[_DepthConstraint] = []
    for frame_index, tag_id, pixels in observations:
        initial_tag = (
            floor_initial_tags[tag_id]
            if tag_id in floor_tags else refined_tags[tag_id]
        )
        constraint = _depth_constraint(
            frames[frame_index],
            frame_index=frame_index,
            tag_id=tag_id,
            pixels=pixels,
            world_from_tag=initial_tag,
            world_from_camera=cameras[frame_index],
        )
        if constraint is not None:
            depth_constraints.append(constraint)

    def pack_bundle() -> np.ndarray:
        values = np.concatenate([
            *[_transform_values(camera) for camera in cameras],
            *[_transform_values(refined_tags[tag_id]) for tag_id in observed_tag_ids],
            *[
                _transform_values(floor_initial_tags[tag_id])
                for tag_id in movable_floor_ids
            ],
            *[
                _transform_values(RigidTransform.identity())
                for _segment in movable_robot_segments
            ],
        ])
        return np.asarray(values, dtype=float)

    def bundle_transform(values: np.ndarray, index: int) -> RigidTransform:
        offset = index * 6
        return _transform_from_values(values[offset:offset + 6])

    def bundle_camera(values: np.ndarray, index: int) -> RigidTransform:
        return bundle_transform(values, index)

    def bundle_tag(values: np.ndarray, tag_id: int) -> RigidTransform:
        return bundle_transform(
            values, robot_variable_offset + observed_tag_index[tag_id]
        )

    def bundle_floor_tag(values: np.ndarray, tag_id: int) -> RigidTransform:
        return bundle_transform(
            values, floor_variable_offset + movable_floor_index[tag_id]
        )

    def bundle_robot_motion(values: np.ndarray, segment: int) -> RigidTransform:
        if segment == reference_robot_segment:
            return RigidTransform.identity()
        return bundle_transform(
            values,
            robot_motion_variable_offset
            + movable_robot_segment_index[segment],
        )

    def bundle_observation_tag(
        values: np.ndarray, frame_index: int, tag_id: int
    ) -> RigidTransform:
        if tag_id in floor_tags:
            return bundle_floor_tag(values, tag_id)
        return bundle_robot_motion(
            values, frame_segments[frame_index]
        ).compose(bundle_tag(values, tag_id))

    def depth_residual_m(
        values: np.ndarray, constraint: _DepthConstraint
    ) -> float:
        camera_from_tag = bundle_camera(
            values, constraint.frame_index
        ).inverse().compose(bundle_observation_tag(
            values, constraint.frame_index, constraint.tag_id
        ))
        normal = camera_from_tag.rotation.apply([0.0, 0.0, 1.0])
        return float(
            (constraint.point_camera_m - camera_from_tag.translation_m)
            @ normal
        )

    def bundle_residual(values: np.ndarray) -> np.ndarray:
        output: list[float] = []
        for frame_index, tag_id, pixels in observations:
            tag = bundle_observation_tag(values, frame_index, tag_id)
            output.extend((_project(
                tag,
                bundle_camera(values, frame_index),
                frames[frame_index].camera_matrix,
                object_corners[tag_id],
            ) - pixels).reshape(-1))
        for constraint in depth_constraints:
            output.append(
                depth_residual_m(values, constraint) / constraint.sigma_m
            )
        # This fixes only the otherwise arbitrary SE(3) gauge. Because all
        # image and relative-motion residuals are gauge invariant, the first
        # camera can stay at its ARKit-aligned pose without constraining any
        # measured robot/floor baseline. The result is rebased to the chosen
        # floor tag below.
        first_camera = bundle_camera(values, 0)
        output.extend(
            (first_camera.translation_m - camera_initial[0].translation_m)
            / 0.0001
        )
        output.extend((
            camera_initial[0].rotation.inv() * first_camera.rotation
        ).as_rotvec() / np.radians(0.01))
        for index in trajectory_edges:
            predicted = bundle_camera(values, index).inverse().compose(
                bundle_camera(values, index + 1)
            )
            measured = frames[index].arkit_world_from_camera.inverse().compose(
                frames[index + 1].arkit_world_from_camera
            )
            output.extend(
                (predicted.translation_m - measured.translation_m) / 0.035
            )
            output.extend((
                measured.rotation.inv() * predicted.rotation
            ).as_rotvec() / np.radians(4.0))
        for tag_id in movable_floor_ids:
            tag = bundle_floor_tag(values, tag_id)
            initial = floor_tags[tag_id]
            append_floor_prior_residual(output, tag, initial, tag_id)
        for segment in movable_robot_segments:
            motion = bundle_robot_motion(values, segment)
            output.extend(motion.translation_m / 0.10)
            output.extend(
                motion.rotation.as_rotvec() / np.radians(15.0)
            )
        return np.asarray(output)

    pixel_coordinate_count = len(observations) * 8
    depth_constraint_count = len(depth_constraints)
    regularizer_offset = pixel_coordinate_count + depth_constraint_count
    bundle_x0 = pack_bundle()
    bundle_initial = bundle_residual(bundle_x0)
    bundle_row_count = len(bundle_initial)
    bundle_sparsity = lil_matrix(
        (bundle_row_count, len(bundle_x0)), dtype=np.uint8
    )
    row = 0
    for frame_index, tag_id, _pixels in observations:
        bundle_sparsity[
            row:row + 8, frame_index * 6:(frame_index + 1) * 6
        ] = 1
        if tag_id in observed_tag_index:
            variable_index = robot_variable_offset + observed_tag_index[tag_id]
            bundle_sparsity[
                row:row + 8,
                variable_index * 6:(variable_index + 1) * 6,
            ] = 1
        elif tag_id in movable_floor_index:
            variable_index = floor_variable_offset + movable_floor_index[tag_id]
            bundle_sparsity[
                row:row + 8,
                variable_index * 6:(variable_index + 1) * 6,
            ] = 1
        if (
            tag_id not in floor_tags
            and frame_segments[frame_index] != reference_robot_segment
        ):
            variable_index = (
                robot_motion_variable_offset
                + movable_robot_segment_index[frame_segments[frame_index]]
            )
            bundle_sparsity[
                row:row + 8,
                variable_index * 6:(variable_index + 1) * 6,
            ] = 1
        row += 8
    for constraint in depth_constraints:
        frame_index = constraint.frame_index
        tag_id = constraint.tag_id
        bundle_sparsity[
            row, frame_index * 6:(frame_index + 1) * 6
        ] = 1
        variable_index = (
            floor_variable_offset + movable_floor_index[tag_id]
            if tag_id in floor_tags
            else robot_variable_offset + observed_tag_index[tag_id]
        )
        bundle_sparsity[
            row, variable_index * 6:(variable_index + 1) * 6
        ] = 1
        if (
            tag_id not in floor_tags
            and frame_segments[frame_index] != reference_robot_segment
        ):
            variable_index = (
                robot_motion_variable_offset
                + movable_robot_segment_index[frame_segments[frame_index]]
            )
            bundle_sparsity[
                row, variable_index * 6:(variable_index + 1) * 6
            ] = 1
        row += 1
    bundle_sparsity[row:row + 6, 0:6] = 1
    row += 6
    for index in trajectory_edges:
        bundle_sparsity[
            row:row + 6, index * 6:(index + 2) * 6
        ] = 1
        row += 6
    for tag_id in movable_floor_ids:
        variable_index = floor_variable_offset + movable_floor_index[tag_id]
        bundle_sparsity[
            row:row + 9,
            variable_index * 6:(variable_index + 1) * 6,
        ] = 1
        row += 9
    for segment in movable_robot_segments:
        variable_index = (
            robot_motion_variable_offset + movable_robot_segment_index[segment]
        )
        bundle_sparsity[
            row:row + 6,
            variable_index * 6:(variable_index + 1) * 6,
        ] = 1
        row += 6
    bundle_solution = least_squares(
        bundle_residual,
        bundle_x0,
        jac_sparsity=bundle_sparsity.tocsr(),
        loss="huber",
        f_scale=2.0,
        x_scale="jac",
        max_nfev=100,
    )
    bundle_robust = bundle_residual(bundle_solution.x)
    bundle_initial_rms = float(np.sqrt(np.mean(
        bundle_initial[:pixel_coordinate_count] ** 2
    )))
    bundle_robust_rms = float(np.sqrt(np.mean(
        bundle_robust[:pixel_coordinate_count] ** 2
    )))

    # A reconnect can divide a walk into several independently tracked ARKit
    # sessions. If the robot was nudged between them, one rigid reconstruction
    # cannot explain both robot-relative and floor-relative corners. Huber is
    # useful for finding the largest consistent mode, but reporting RMS over
    # the observations it deliberately downweighted makes a good fit look bad.
    # Finish with ordinary least squares on that consistent mode and report the
    # rejected views explicitly. This also prevents the much more numerous
    # robot corners from silently sacrificing every floor observation.
    fit_observation_indices = list(range(len(observations)))
    rejected_observation_indices: list[int] = []
    fit_depth_constraint_indices = list(range(depth_constraint_count))
    rejected_depth_constraint_indices: list[int] = []
    bundle_values = bundle_solution.x
    polish_solution = None
    polish_accepted = False
    polish_initial_rms = bundle_robust_rms
    polish_final_rms = bundle_robust_rms
    if math.isfinite(bundle_robust_rms) and bundle_robust_rms < bundle_initial_rms:
        robust_depth = bundle_robust[
            pixel_coordinate_count:regularizer_offset
        ]
        fit_depth_constraint_indices = [
            index for index, residual in enumerate(robust_depth)
            if abs(float(residual)) <= 4.0
        ]
        rejected_depth_constraint_indices = sorted(
            set(range(depth_constraint_count))
            - set(fit_depth_constraint_indices)
        )
        robust_pixels = bundle_robust[:pixel_coordinate_count].reshape(-1, 8)
        observation_rms = np.sqrt(np.mean(robust_pixels ** 2, axis=1))
        selected = {
            index
            for index, ((_frame_index, tag_id, _pixels), error) in enumerate(
                zip(observations, observation_rms)
            )
            if error <= (6.0 if tag_id in floor_tags else 8.0)
        }
        # Keep every tag and camera connected to the bundle even in a very
        # noisy capture. The quality gate below still requires two accepted
        # views of every floor tag; retaining these minima only avoids singular
        # variables and gives the report an honest failure instead of a crash.
        for tag_id in set(item[1] for item in observations):
            ranked = sorted(
                (
                    (float(observation_rms[index]), index)
                    for index, item in enumerate(observations)
                    if item[1] == tag_id
                )
            )
            selected.update(index for _error, index in ranked[:2])
        for frame_index in range(camera_count):
            ranked = sorted(
                (
                    (float(observation_rms[index]), index)
                    for index, item in enumerate(observations)
                    if item[0] == frame_index
                )
            )
            selected.update(index for _error, index in ranked[:1])
        fit_observation_indices = sorted(selected)
        rejected_observation_indices = sorted(
            set(range(len(observations))) - selected
        )

        def polish_residual(values: np.ndarray) -> np.ndarray:
            output: list[float] = []
            for observation_index in fit_observation_indices:
                frame_index, tag_id, pixels = observations[observation_index]
                tag = bundle_observation_tag(values, frame_index, tag_id)
                output.extend((_project(
                    tag,
                    bundle_camera(values, frame_index),
                    frames[frame_index].camera_matrix,
                    object_corners[tag_id],
                ) - pixels).reshape(-1))
            for constraint_index in fit_depth_constraint_indices:
                constraint = depth_constraints[constraint_index]
                output.append(
                    depth_residual_m(values, constraint) / constraint.sigma_m
                )
            # Retain the gauge, within-session ARKit motion, and floor-plane
            # regularizers from the robust solve. They are weak compared with
            # the unweighted corner residuals but keep the metric 3-D result
            # physically well conditioned.
            output.extend(bundle_residual(values)[regularizer_offset:])
            return np.asarray(output)

        polish_x0 = bundle_solution.x
        polish_initial = polish_residual(polish_x0)
        polish_pixel_count = len(fit_observation_indices) * 8
        polish_initial_rms = float(np.sqrt(np.mean(
            polish_initial[:polish_pixel_count] ** 2
        )))
        polish_sparsity = lil_matrix(
            (len(polish_initial), len(polish_x0)), dtype=np.uint8
        )
        row = 0
        for observation_index in fit_observation_indices:
            frame_index, tag_id, _pixels = observations[observation_index]
            polish_sparsity[
                row:row + 8, frame_index * 6:(frame_index + 1) * 6
            ] = 1
            variable_index = (
                floor_variable_offset + movable_floor_index[tag_id]
                if tag_id in floor_tags
                else robot_variable_offset + observed_tag_index[tag_id]
            )
            polish_sparsity[
                row:row + 8,
                variable_index * 6:(variable_index + 1) * 6,
            ] = 1
            if (
                tag_id not in floor_tags
                and frame_segments[frame_index] != reference_robot_segment
            ):
                variable_index = (
                    robot_motion_variable_offset
                    + movable_robot_segment_index[frame_segments[frame_index]]
                )
                polish_sparsity[
                    row:row + 8,
                    variable_index * 6:(variable_index + 1) * 6,
                ] = 1
            row += 8
        for constraint_index in fit_depth_constraint_indices:
            constraint = depth_constraints[constraint_index]
            frame_index = constraint.frame_index
            tag_id = constraint.tag_id
            polish_sparsity[
                row, frame_index * 6:(frame_index + 1) * 6
            ] = 1
            variable_index = (
                floor_variable_offset + movable_floor_index[tag_id]
                if tag_id in floor_tags
                else robot_variable_offset + observed_tag_index[tag_id]
            )
            polish_sparsity[
                row, variable_index * 6:(variable_index + 1) * 6
            ] = 1
            if (
                tag_id not in floor_tags
                and frame_segments[frame_index] != reference_robot_segment
            ):
                variable_index = (
                    robot_motion_variable_offset
                    + movable_robot_segment_index[frame_segments[frame_index]]
                )
                polish_sparsity[
                    row, variable_index * 6:(variable_index + 1) * 6,
                ] = 1
            row += 1
        polish_sparsity[row:row + 6, 0:6] = 1
        row += 6
        for index in trajectory_edges:
            polish_sparsity[
                row:row + 6, index * 6:(index + 2) * 6
            ] = 1
            row += 6
        for tag_id in movable_floor_ids:
            variable_index = floor_variable_offset + movable_floor_index[tag_id]
            polish_sparsity[
                row:row + 9,
                variable_index * 6:(variable_index + 1) * 6,
            ] = 1
            row += 9
        for segment in movable_robot_segments:
            variable_index = (
                robot_motion_variable_offset
                + movable_robot_segment_index[segment]
            )
            polish_sparsity[
                row:row + 6,
                variable_index * 6:(variable_index + 1) * 6,
            ] = 1
            row += 6
        polish_solution = least_squares(
            polish_residual,
            polish_x0,
            jac_sparsity=polish_sparsity.tocsr(),
            loss="linear",
            x_scale="jac",
            max_nfev=80,
        )
        polished = polish_residual(polish_solution.x)
        polish_final_rms = float(np.sqrt(np.mean(
            polished[:polish_pixel_count] ** 2
        )))
        if math.isfinite(polish_final_rms) and polish_final_rms <= polish_initial_rms:
            bundle_values = polish_solution.x
            polish_accepted = True
        else:
            fit_observation_indices = list(range(len(observations)))
            rejected_observation_indices = []
            polish_final_rms = bundle_robust_rms

    initial_rejected_observation_indices = list(rejected_observation_indices)
    initial_rejected_depth_constraint_indices = list(
        rejected_depth_constraint_indices
    )
    bundle_final_residual = bundle_residual(bundle_values)
    bundle_all_final = bundle_final_residual[:pixel_coordinate_count]
    bundle_all_final_rms = float(np.sqrt(np.mean(bundle_all_final ** 2)))
    # The linear polish can pull a view that Huber initially downweighted back
    # into excellent agreement. Reclassify against the finished geometry so a
    # 1-2 px floor view is not misleadingly reported or drawn as rejected.
    final_observation_rms = np.sqrt(np.mean(
        bundle_all_final.reshape(-1, 8) ** 2, axis=1
    ))
    fit_observation_indices = [
        index
        for index, ((_frame_index, tag_id, _pixels), error) in enumerate(
            zip(observations, final_observation_rms)
        )
        if error <= (6.0 if tag_id in floor_tags else 8.0)
    ]
    rejected_observation_indices = sorted(
        set(range(len(observations))) - set(fit_observation_indices)
    )
    final_standardized_depth = bundle_final_residual[
        pixel_coordinate_count:regularizer_offset
    ]
    fit_depth_constraint_indices = [
        index for index, residual in enumerate(final_standardized_depth)
        if abs(float(residual)) <= 4.0
    ]
    rejected_depth_constraint_indices = sorted(
        set(range(depth_constraint_count)) - set(fit_depth_constraint_indices)
    )
    robot_segment_motions = [
        bundle_robot_motion(bundle_values, segment)
        for segment in range(segment_count)
    ]
    if math.isfinite(polish_final_rms) and polish_final_rms < bundle_initial_rms:
        cameras = [
            bundle_camera(bundle_values, index)
            for index in range(camera_count)
        ]
        refined_tags.update({
            tag_id: bundle_tag(bundle_values, tag_id)
            for tag_id in observed_tag_ids
        })
        refined_floor_tags = {
            tag_id: bundle_floor_tag(bundle_values, tag_id)
            for tag_id in observed_floor_ids
        }
        alignment_rebase = refined_floor_tags[floor_anchor_id].inverse()
        cameras = [alignment_rebase.compose(camera) for camera in cameras]
        camera_initial = [
            alignment_rebase.compose(camera) for camera in camera_initial
        ]
        refined_tags = {
            tag_id: alignment_rebase.compose(tag)
            for tag_id, tag in refined_tags.items()
        }
        refined_floor_tags = {
            tag_id: alignment_rebase.compose(tag)
            for tag_id, tag in refined_floor_tags.items()
        }
        physical_tags = {
            tag_id: alignment_rebase.compose(tag)
            for tag_id, tag in physical_tags.items()
        }
        robot_segment_motions = [
            alignment_rebase.compose(motion).compose(
                alignment_rebase.inverse()
            )
            for motion in robot_segment_motions
        ]
        world_from_body_report = alignment_rebase.compose(
            body_at(model_values)
        )
    else:
        alignment_rebase = RigidTransform(np.zeros(3), Rotation.identity())
        world_from_body_report = body_at(model_values)
        refined_floor_tags = {
            tag_id: floor_tags[tag_id] for tag_id in observed_floor_ids
        }
        robot_segment_motions = [
            RigidTransform.identity() for _segment in range(segment_count)
        ]
    final_errors: list[float] = []
    floor_errors: list[float] = []
    robot_errors: list[float] = []
    per_tag_errors: dict[int, list[float]] = {}
    per_frame_errors: dict[int, list[float]] = {}
    tag_view_counts: dict[int, int] = {}
    rejected_errors: list[float] = []
    rejected_by_tag: dict[int, int] = {}
    rejected_by_frame: dict[int, int] = {}
    used_frames_by_tag: dict[int, list[int]] = {}
    fit_observation_set = set(fit_observation_indices)
    for observation_index, (frame_index, tag_id, pixels) in enumerate(observations):
        tag = refined_floor_tags.get(tag_id, refined_tags.get(tag_id))
        if tag is None:
            continue
        observed_tag = (
            tag if tag_id in floor_tags
            else robot_segment_motions[frame_segments[frame_index]].compose(tag)
        )
        errors = (_project(
            observed_tag,
            cameras[frame_index],
            frames[frame_index].camera_matrix,
            object_corners[tag_id],
        ) - pixels).reshape(-1)
        if observation_index not in fit_observation_set:
            rejected_errors.extend(errors)
            rejected_by_tag[tag_id] = rejected_by_tag.get(tag_id, 0) + 1
            rejected_by_frame[frame_index] = (
                rejected_by_frame.get(frame_index, 0) + 1
            )
            continue
        final_errors.extend(errors)
        if tag_id in floor_tags:
            floor_errors.extend(errors)
        else:
            robot_errors.extend(errors)
        per_tag_errors.setdefault(tag_id, []).extend(errors)
        per_frame_errors.setdefault(frame_index, []).extend(errors)
        tag_view_counts[tag_id] = tag_view_counts.get(tag_id, 0) + 1
        used_frames_by_tag.setdefault(tag_id, []).append(frame_index)

    refined_survey = copy.deepcopy(dict(survey))
    floor_corrections: list[dict[str, Any]] = []
    for tag_id in observed_floor_ids:
        original = floor_tags[tag_id]
        refined = refined_floor_tags[tag_id]
        floor_corrections.append({
            "tag_id": tag_id,
            "is_world_anchor": tag_id == floor_anchor_id,
            "world_from_tag": refined.to_dict(),
            "translation_mm": float(np.linalg.norm(
                refined.translation_m - original.translation_m
            ) * 1000.0),
            "rotation_deg": math.degrees(float(
                (original.rotation.inv() * refined.rotation).magnitude()
            )),
            "archived_views": tag_view_counts.get(tag_id, 0),
            "rejected_archived_views": rejected_by_tag.get(tag_id, 0),
            "coordinate_rms_px": float(np.sqrt(np.mean(
                np.square(per_tag_errors.get(tag_id, [0.0]))
            ))),
        })
    for record in refined_survey.get("tags", []):
        tag_id = int(record["tag_id"])
        if (
            tag_id not in refined_floor_tags
            or record.get("role") not in ("ground", "calibration_anchor")
        ):
            continue
        refined = refined_floor_tags[tag_id]
        record["pre_refinement_world_from_tag"] = record["world_from_tag"]
        _record_orientation_fields(record, refined)
        record["photo_bundle_refined"] = True

    stable_floor_records = [
        item for item in refined_survey.get("tags", [])
        if item.get("stable")
        and item.get("role") in ("ground", "calibration_anchor")
    ]
    floor_distances: list[dict[str, Any]] = []
    for first_index, first in enumerate(stable_floor_records):
        first_position = np.asarray(
            first["world_from_tag"]["translation_m"], dtype=float
        )
        for second in stable_floor_records[first_index + 1:]:
            second_position = np.asarray(
                second["world_from_tag"]["translation_m"], dtype=float
            )
            delta = second_position - first_position
            floor_distances.append({
                "tag_ids": [int(first["tag_id"]), int(second["tag_id"])],
                "center_distance_m": round(float(np.linalg.norm(delta)), 6),
                "planar_distance_m": round(float(np.linalg.norm(delta[:2])), 6),
                "delta_xyz_m": [round(float(value), 6) for value in delta],
            })
    if stable_floor_records:
        refined_survey["floor_tag_distances"] = floor_distances

    corrections: list[dict[str, Any]] = []
    for record in refined_survey.get("tags", []):
        tag_id = int(record["tag_id"])
        if tag_id not in refined_tags or record.get("role") != "robot":
            continue
        original = RigidTransform.from_dict(record["world_from_tag"])
        refined = refined_tags[tag_id]
        correction_mm = float(np.linalg.norm(
            refined.translation_m - original.translation_m
        ) * 1000.0)
        correction_deg = math.degrees(float(
            (original.rotation.inv() * refined.rotation).magnitude()
        ))
        shared_mount_deviation_mm = float(np.linalg.norm(
            refined.translation_m - physical_tags[tag_id].translation_m
        ) * 1000.0)
        normal_dot = float(np.clip(
            refined.rotation.apply([0.0, 0.0, 1.0])
            @ physical_tags[tag_id].rotation.apply([0.0, 0.0, 1.0]),
            -1.0,
            1.0,
        ))
        shared_surface_normal_error_deg = math.degrees(math.acos(normal_dot))
        record["pre_refinement_world_from_tag"] = record["world_from_tag"]
        _record_orientation_fields(record, refined)
        record["buildviz_refined"] = True
        record["photo_bundle_refined"] = True
        record["buildviz_correction_mm"] = round(correction_mm, 3)
        record["buildviz_correction_deg"] = round(correction_deg, 4)
        if tag_id not in by_tag:
            continue
        view_directions = []
        for frame_index in used_frames_by_tag.get(tag_id, []):
            canonical_camera = robot_segment_motions[
                frame_segments[frame_index]
            ].inverse().compose(cameras[frame_index])
            direction = (
                canonical_camera.translation_m - refined.translation_m
            )
            magnitude = float(np.linalg.norm(direction))
            if magnitude > 1e-9:
                view_directions.append(direction / magnitude)
        viewpoint_span_deg = max((
            math.degrees(math.acos(float(np.clip(
                first @ second, -1.0, 1.0
            ))))
            for index, first in enumerate(view_directions)
            for second in view_directions[index + 1:]
        ), default=0.0)
        corrections.append({
            "tag_id": tag_id,
            "world_from_tag": refined.to_dict(),
            "translation_mm": correction_mm,
            "rotation_deg": correction_deg,
            "shared_mount_deviation_mm": shared_mount_deviation_mm,
            "shared_surface_normal_error_deg": shared_surface_normal_error_deg,
            "archived_views": tag_view_counts.get(tag_id, 0),
            "rejected_archived_views": rejected_by_tag.get(tag_id, 0),
            "viewpoint_span_deg": viewpoint_span_deg,
            "coordinate_rms_px": float(np.sqrt(np.mean(
                np.square(per_tag_errors.get(tag_id, [0.0]))
            ))),
        })

    def rms(values: Sequence[float] | np.ndarray) -> float:
        array = np.asarray(values, dtype=float)
        return float(np.sqrt(np.mean(array ** 2))) if array.size else 0.0

    physical_rms = rms(physical_pixel_residual)
    final_rms = rms(final_errors)
    floor_rms = rms(floor_errors)
    robot_rms = rms(robot_errors)
    rejected_rms = rms(rejected_errors)
    final_depth_errors_mm = np.asarray([
        depth_residual_m(bundle_values, constraint) * 1000.0
        for constraint in depth_constraints
    ])
    used_depth_errors_mm = np.asarray([
        final_depth_errors_mm[index]
        for index in fit_depth_constraint_indices
    ])
    depth_available = any(frame.depth_m is not None for frame in frames)
    depth_median_absolute_mm = (
        float(np.median(np.abs(used_depth_errors_mm)))
        if used_depth_errors_mm.size else None
    )
    depth_p90_absolute_mm = (
        float(np.percentile(np.abs(used_depth_errors_mm), 90.0))
        if used_depth_errors_mm.size else None
    )
    minimum_depth_constraints = max(12, len(frames) // 4)
    depth_quality_passed = (
        not depth_available
        or (
            len(fit_depth_constraint_indices) >= minimum_depth_constraints
            and depth_median_absolute_mm is not None
            and depth_median_absolute_mm <= 12.0
            and depth_p90_absolute_mm is not None
            and depth_p90_absolute_mm <= 25.0
        )
    )
    max_mount_deviation_mm = max((
        item["shared_mount_deviation_mm"] for item in corrections
    ), default=0.0)
    for item in corrections:
        issues: list[str] = []
        if item["archived_views"] < 2:
            issues.append("insufficient_archived_views")
        if item["viewpoint_span_deg"] < 8.0:
            issues.append("weak_viewpoint_geometry")
        if item["coordinate_rms_px"] > 3.0:
            issues.append("image_fit_outlier")
        if item["shared_mount_deviation_mm"] > 15.0:
            issues.append("buildviz_or_mount_position_mismatch")
        if item["shared_surface_normal_error_deg"] > 12.0:
            issues.append("configured_face_orientation_mismatch")
        item["diagnostic_flags"] = issues
    camera_translation_corrections_mm = [
        float(np.linalg.norm(
            camera.translation_m - initial.translation_m
        ) * 1000.0)
        for camera, initial in zip(cameras, camera_initial)
    ]
    camera_rotation_corrections_deg = [
        math.degrees(float(
            (initial.rotation.inv() * camera.rotation).magnitude()
        ))
        for camera, initial in zip(cameras, camera_initial)
    ]
    per_camera: list[dict[str, Any]] = []
    for index, (camera, initial) in enumerate(zip(cameras, camera_initial)):
        frame_tag_ids = [
            int(tag_id) for tag_id, _pixels in frames[index].detections
        ]
        per_camera.append({
            "frame": frames[index].name,
            "trajectory_segment": frame_segments[index],
            "robot_reference_segment": (
                frame_segments[index] == reference_robot_segment
            ),
            "world_from_camera": camera.to_dict(),
            "coordinate_rms_px": rms(per_frame_errors.get(index, [])),
            "used_tag_observations": len(per_frame_errors.get(index, [])) // 8,
            "rejected_tag_observations": rejected_by_frame.get(index, 0),
            "translation_from_arkit_mm": float(np.linalg.norm(
                camera.translation_m - initial.translation_m
            ) * 1000.0),
            "rotation_from_arkit_deg": math.degrees(float(
                (initial.rotation.inv() * camera.rotation).magnitude()
            )),
            "tag_ids": frame_tag_ids,
            "floor_tag_ids": [
                tag_id for tag_id in frame_tag_ids if tag_id in floor_tags
            ],
        })
    image_outlier_ids = [
        int(item["tag_id"]) for item in corrections
        if "image_fit_outlier" in item["diagnostic_flags"]
    ]
    geometry_mismatch_ids = [
        int(item["tag_id"]) for item in corrections
        if any(flag in item["diagnostic_flags"] for flag in (
            "buildviz_or_mount_position_mismatch",
            "configured_face_orientation_mismatch",
        ))
    ]
    sparse_view_ids = [
        int(item["tag_id"]) for item in corrections
        if "insufficient_archived_views" in item["diagnostic_flags"]
    ]
    weak_viewpoint_ids = [
        int(item["tag_id"]) for item in corrections
        if "weak_viewpoint_geometry" in item["diagnostic_flags"]
    ]
    floor_view_coverage = all(
        tag_view_counts.get(tag_id, 0) >= 2
        for tag_id in observed_floor_ids
    )
    floor_observation_indices = [
        index for index, item in enumerate(observations)
        if item[1] in floor_tags
    ]
    rejected_observation_set = set(rejected_observation_indices)
    rejected_floor_count = sum(
        index in rejected_observation_set for index in floor_observation_indices
    )
    rejected_floor_fraction = (
        rejected_floor_count / len(floor_observation_indices)
        if floor_observation_indices else 0.0
    )
    floor_rejection_by_segment: list[dict[str, Any]] = []
    for segment in range(segment_count):
        segment_indices = [
            index for index in floor_observation_indices
            if frame_segments[observations[index][0]] == segment
        ]
        rejected = sum(
            index in rejected_observation_set for index in segment_indices
        )
        floor_rejection_by_segment.append({
            "segment": segment,
            "observations": len(segment_indices),
            "rejected_observations": rejected,
            "rejected_fraction": (
                rejected / len(segment_indices) if segment_indices else 0.0
            ),
        })
    systematic_floor_consistency_passed = (
        rejected_floor_fraction <= 0.20
        and all(
            item["observations"] < 6 or item["rejected_fraction"] <= 0.50
            for item in floor_rejection_by_segment
        )
    )
    robot_segment_motion_report = []
    for segment, motion in enumerate(robot_segment_motions):
        robot_segment_motion_report.append({
            "segment": segment,
            "reference": segment == reference_robot_segment,
            "robot_observations": robot_observations_per_segment[segment],
            "floor_observations": floor_observations_per_segment[segment],
            "translation_mm": float(np.linalg.norm(
                motion.translation_m
            ) * 1000.0),
            "rotation_deg": math.degrees(float(
                motion.rotation.magnitude()
            )),
            "world_from_reference_robot": motion.to_dict(),
        })
    scene_motion_passed = all(
        item["translation_mm"] <= 120.0 and item["rotation_deg"] <= 12.0
        for item in robot_segment_motion_report
    )
    quality_passed = (
        final_rms <= 3.0
        and floor_rms <= 3.0
        and robot_rms <= 3.5
        and floor_view_coverage
        and systematic_floor_consistency_passed
        and depth_quality_passed
        and scene_motion_passed
    )
    visual_audit = _write_reprojection_audit(
        frame_archive_dir=Path(frame_archive_dir),
        frames=frames,
        observations=observations,
        cameras=cameras,
        robot_tags=refined_tags,
        floor_tags=refined_floor_tags,
        robot_segment_motions=robot_segment_motions,
        frame_segments=frame_segments,
        object_corners=object_corners,
        fit_observation_indices=fit_observation_indices,
    )
    report = {
        "ok": quality_passed,
        "method": (
            "image-first joint bundle adjustment of archived full-resolution "
            "corners; one output floor origin plus coplanar floor landmarks "
            "and surveyed priors when configured; "
            "ARKit relative motion within uninterrupted sessions; a robust "
            "consistent-mode selection followed by unweighted least squares; "
            "confidence-filtered LiDAR point-to-tag-plane factors; rigid "
            "whole-robot motion between reconnect segments; "
            "BuildViz used only for initialization and as a post-fit diagnostic"
        ),
        "frames": len(frames),
        "archived_frames": archived_frame_count,
        "keyframe_selection": {
            "used": len(frames) < archived_frame_count,
            "selected_frames": len(frames),
            "archived_frames": archived_frame_count,
            "source_resolution_preserved": True,
            "selection_basis": (
                "tag coverage, apparent tag size, reconnect boundaries, and "
                "ARKit translation/rotation diversity"
            ),
        },
        "corner_observations": len(observations),
        "used_corner_observations": len(fit_observation_indices),
        "rejected_corner_observations": len(rejected_observation_indices),
        "robot_tag_count": len(by_tag),
        "initial_coordinate_rms_px": rms(initial_pixel_residual),
        "physical_model_coordinate_rms_px": physical_rms,
        "pre_bundle_coordinate_rms_px": bundle_initial_rms,
        "robust_all_observation_coordinate_rms_px": bundle_robust_rms,
        "all_observation_coordinate_rms_px": bundle_all_final_rms,
        "final_coordinate_rms_px": final_rms,
        "floor_coordinate_rms_px": floor_rms,
        "robot_coordinate_rms_px": robot_rms,
        "rejected_observation_coordinate_rms_px": rejected_rms,
        "floor_anchor_tag_id": floor_anchor_id,
        "quality_gate": {
            "passed": quality_passed,
            "basis": (
                "full-resolution image reprojection, LiDAR range, surveyed "
                "floor consistency, and bounded reconnect movement; BuildViz "
                "disagreement is diagnostic"
            ),
            "final_coordinate_rms_px_max": 3.0,
            "floor_coordinate_rms_px_max": 3.0,
            "robot_coordinate_rms_px_max": 3.5,
            "at_least_two_views_per_floor_tag": floor_view_coverage,
            "systematic_floor_consistency": systematic_floor_consistency_passed,
            "lidar_depth_consistency": depth_quality_passed,
            "scene_motion_bounded": scene_motion_passed,
        },
        "bundle_optimizer": {
            "success": bool(bundle_solution.success),
            "message": str(bundle_solution.message),
            "function_evaluations": int(bundle_solution.nfev),
            "trajectory_segments": camera_count - len(trajectory_edges),
            "relative_motion_edges": len(trajectory_edges),
            "inlier_polish": {
                "accepted": polish_accepted,
                "success": (
                    None if polish_solution is None
                    else bool(polish_solution.success)
                ),
                "message": (
                    None if polish_solution is None
                    else str(polish_solution.message)
                ),
                "function_evaluations": (
                    0 if polish_solution is None else int(polish_solution.nfev)
                ),
                "initial_coordinate_rms_px": polish_initial_rms,
                "final_coordinate_rms_px": polish_final_rms,
            },
        },
        "lidar_depth": {
            "available": depth_available,
            "constraints": depth_constraint_count,
            "used_constraints": len(fit_depth_constraint_indices),
            "rejected_constraints": len(rejected_depth_constraint_indices),
            "minimum_constraints": minimum_depth_constraints,
            "median_absolute_error_mm": depth_median_absolute_mm,
            "p90_absolute_error_mm": depth_p90_absolute_mm,
            "rms_error_mm": (
                rms(used_depth_errors_mm) if used_depth_errors_mm.size else None
            ),
            "quality_passed": depth_quality_passed,
            "role": (
                "metric range/plane constraint; RGB corners remain the "
                "lateral and orientation measurement"
            ),
        },
        "scene_motion": {
            "reference_segment": reference_robot_segment,
            "segments": robot_segment_motion_report,
            "floor_rejection_fraction": rejected_floor_fraction,
            "floor_rejection_by_segment": floor_rejection_by_segment,
            "systematic_floor_consistency_passed": (
                systematic_floor_consistency_passed
            ),
        },
        "floor_reference": {
            "status": floor_reference_status or "unspecified",
            "uses_measured_ground_truth": any(
                prior["source"] == "measured_ground_truth"
                for prior in floor_priors.values()
            ),
            "per_tag": {
                str(tag_id): prior
                for tag_id, prior in sorted(floor_priors.items())
            },
        },
        "visual_reprojection_audit": visual_audit,
        "outlier_filter": {
            "robot_observation_rms_px_max": 8.0,
            "floor_observation_rms_px_max": 6.0,
            "initial_robust_rejected_observations": len(
                initial_rejected_observation_indices
            ),
            "recovered_after_polish": len(
                set(initial_rejected_observation_indices)
                - set(rejected_observation_indices)
            ),
            "initial_robust_rejected_depth_constraints": len(
                initial_rejected_depth_constraint_indices
            ),
            "rejected_by_tag_id": {
                str(tag_id): count
                for tag_id, count in sorted(rejected_by_tag.items())
            },
        },
        "world_from_body": world_from_body_report.to_dict(),
        "alignment_rebase": {
            "translation_mm": float(np.linalg.norm(
                alignment_rebase.translation_m
            ) * 1000.0),
            "rotation_deg": math.degrees(float(
                alignment_rebase.rotation.magnitude()
            )),
            "new_world_from_previous_world": alignment_rebase.to_dict(),
        },
        "buildviz_geometry_m": {
            "chassis_apothem_m": geometry.chassis_apothem_m,
            "coxa_m": geometry.coxa_m,
            "femur_m": geometry.femur_m,
            "tibia_m": geometry.tibia_m,
            "hip_anchor_y_m": geometry.hip_anchor_y_m,
        },
        "geometry_source": (
            "hexapod_walker/prototype_sts3215 BuildViz and parametric CAD"
        ),
        "shared_mount_translation_m": {
            name: [round(float(value), 9) for value in group_at(model_values, name)]
            for name in group_names
        },
        "median_tag_correction_mm": float(np.median([
            item["translation_mm"] for item in corrections
        ])),
        "max_tag_correction_mm": max(
            (item["translation_mm"] for item in corrections), default=0.0
        ),
        "median_shared_mount_deviation_mm": float(np.median([
            item["shared_mount_deviation_mm"] for item in corrections
        ])),
        "max_shared_mount_deviation_mm": max_mount_deviation_mm,
        "camera_correction_from_arkit": {
            "median_translation_mm": float(np.median(
                camera_translation_corrections_mm
            )),
            "max_translation_mm": max(camera_translation_corrections_mm),
            "median_rotation_deg": float(np.median(
                camera_rotation_corrections_deg
            )),
            "max_rotation_deg": max(camera_rotation_corrections_deg),
        },
        "floor_tag_corrections": floor_corrections,
        "per_camera": per_camera,
        "diagnostics": {
            "image_fit_outlier_tag_ids": image_outlier_ids,
            "buildviz_or_mount_mismatch_tag_ids": geometry_mismatch_ids,
            "insufficient_archived_view_tag_ids": sparse_view_ids,
            "weak_viewpoint_geometry_tag_ids": weak_viewpoint_ids,
            "worst_image_frames": [
                item["frame"] for item in sorted(
                    per_camera,
                    key=lambda item: item["coordinate_rms_px"],
                    reverse=True,
                )[:5]
                if item["coordinate_rms_px"] > 3.0
            ],
            "floor_tags_moved_over_10mm": [
                int(item["tag_id"]) for item in floor_corrections
                if not item["is_world_anchor"] and item["translation_mm"] > 10.0
            ],
        },
        "per_tag": corrections,
    }
    if not quality_passed:
        failing_reasons: list[str] = []
        if not floor_view_coverage:
            failing_reasons.append(
                "fewer than two consistent image views remain for a floor tag"
            )
        if not systematic_floor_consistency_passed:
            failing_reasons.append(
                "too many floor observations disagree systematically with the fit"
            )
        if not depth_quality_passed:
            failing_reasons.append(
                "too few trustworthy LiDAR samples or excessive LiDAR range error"
            )
        if not scene_motion_passed:
            failing_reasons.append(
                "robot movement between reconnects exceeds the safe compensation limit"
            )
        if final_rms > 3.0 or floor_rms > 3.0 or robot_rms > 3.5:
            failing_reasons.append(
                "joint photo reprojection residual exceeds the image-fit accuracy gate"
            )
        report["skipped_reason"] = "; ".join(failing_reasons)
    return refined_survey, report
