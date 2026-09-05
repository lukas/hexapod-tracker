"""Offline, BuildViz-constrained refinement for a zero-pose iPhone survey."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares
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
    camera_matrix: np.ndarray
    arkit_world_from_camera: RigidTransform
    detections: tuple[tuple[int, np.ndarray], ...]


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
            frames.append(_ArchivedFrame(
                name=path.name,
                camera_matrix=np.asarray(raw["camera_matrix"], dtype=float),
                arkit_world_from_camera=arkit_world_from_opencv_camera(
                    arkit_gl
                ),
                detections=detections,
            ))
    return frames


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
    joint_angles_deg: Mapping[str, float] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Refine a stationary survey using photos and the verified CAD skeleton.

    The online survey stays lightweight.  This final pass jointly learns the
    repeated physical tag-mount offsets, refines archived camera poses, and
    then permits small per-tag corrections for real placement differences.
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
    frames = _load_frames(
        Path(frame_archive_dir), set(specs) | set(floor_tags)
    )
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
    final_errors: list[float] = []
    per_tag_errors: dict[int, list[float]] = {}
    for frame_index, tag_id, pixels in observations:
        tag = floor_tags.get(tag_id, refined_tags.get(tag_id))
        if tag is None:
            continue
        errors = (_project(
            tag,
            cameras[frame_index],
            frames[frame_index].camera_matrix,
            object_corners[tag_id],
        ) - pixels).reshape(-1)
        final_errors.extend(errors)
        per_tag_errors.setdefault(tag_id, []).extend(errors)

    refined_survey = copy.deepcopy(dict(survey))
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
        record["pre_refinement_world_from_tag"] = record["world_from_tag"]
        _record_orientation_fields(record, refined)
        record["buildviz_refined"] = True
        record["buildviz_correction_mm"] = round(correction_mm, 3)
        record["buildviz_correction_deg"] = round(correction_deg, 4)
        if tag_id not in by_tag:
            continue
        corrections.append({
            "tag_id": tag_id,
            "translation_mm": correction_mm,
            "rotation_deg": correction_deg,
            "shared_mount_deviation_mm": shared_mount_deviation_mm,
            "coordinate_rms_px": float(np.sqrt(np.mean(
                np.square(per_tag_errors.get(tag_id, [0.0]))
            ))),
        })

    def rms(values: Sequence[float] | np.ndarray) -> float:
        array = np.asarray(values, dtype=float)
        return float(np.sqrt(np.mean(array ** 2))) if array.size else 0.0

    physical_rms = rms(physical_pixel_residual)
    final_rms = rms(final_errors)
    max_mount_deviation_mm = max((
        item["shared_mount_deviation_mm"] for item in corrections
    ), default=0.0)
    quality_passed = (
        physical_rms <= 15.0
        and final_rms <= 5.0
        and max_mount_deviation_mm <= 20.0
    )
    report = {
        "ok": quality_passed,
        "method": (
            "BuildViz CAD skeleton, shared six-leg mount geometry, archived "
            "full-resolution corner reprojection, and bounded per-tag offsets"
        ),
        "frames": len(frames),
        "corner_observations": len(observations),
        "robot_tag_count": len(by_tag),
        "initial_coordinate_rms_px": rms(initial_pixel_residual),
        "physical_model_coordinate_rms_px": physical_rms,
        "final_coordinate_rms_px": final_rms,
        "quality_gate": {
            "passed": quality_passed,
            "physical_model_coordinate_rms_px_max": 15.0,
            "final_coordinate_rms_px_max": 5.0,
            "max_shared_mount_deviation_mm": 20.0,
        },
        "world_from_body": body_at(model_values).to_dict(),
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
        "per_tag": corrections,
    }
    if not quality_passed:
        report["skipped_reason"] = (
            "photo-to-CAD residual exceeds the calibration accuracy gate"
        )
    return refined_survey, report
