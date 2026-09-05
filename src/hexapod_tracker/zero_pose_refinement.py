"""Offline image-first refinement for a zero-pose iPhone survey."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

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
                output.extend(
                    (tag.translation_m[:2] - initial.translation_m[:2]) / 0.10
                )
                output.append(float(tag.translation_m[2]) / 0.003)
                output.extend((
                    tag.rotation.apply([0.0, 0.0, 1.0])
                    - np.asarray([0.0, 0.0, 1.0])
                ) / math.sin(math.radians(3.0)))
                output.extend((
                    initial.rotation.inv() * tag.rotation
                ).as_rotvec() / np.radians(45.0))
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

    # Restart the final image solve from the actual ARKit trajectory. The
    # preceding physical-model rounds are useful tag initializers, but their
    # camera corrections can bake a wrong CAD or nominal floor map into the
    # final minimum.
    cameras = list(camera_initial)
    refined_tags = refine_individual_tags()
    floor_initial_tags = refine_floor_landmarks()
    camera_count = len(frames)
    robot_variable_offset = camera_count
    floor_variable_offset = camera_count + len(observed_tag_ids)

    def pack_bundle() -> np.ndarray:
        values = np.concatenate([
            *[_transform_values(camera) for camera in cameras],
            *[_transform_values(refined_tags[tag_id]) for tag_id in observed_tag_ids],
            *[
                _transform_values(floor_initial_tags[tag_id])
                for tag_id in movable_floor_ids
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

    def bundle_residual(values: np.ndarray) -> np.ndarray:
        output: list[float] = []
        for frame_index, tag_id, pixels in observations:
            tag = (
                bundle_floor_tag(values, tag_id)
                if tag_id in floor_tags
                else bundle_tag(values, tag_id)
            )
            output.extend((_project(
                tag,
                bundle_camera(values, frame_index),
                frames[frame_index].camera_matrix,
                object_corners[tag_id],
            ) - pixels).reshape(-1))
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
        for index in range(camera_count - 1):
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
            output.extend(
                (tag.translation_m[:2] - initial.translation_m[:2]) / 0.10
            )
            output.append(float(tag.translation_m[2]) / 0.003)
            output.extend((
                tag.rotation.apply([0.0, 0.0, 1.0])
                - np.asarray([0.0, 0.0, 1.0])
            ) / math.sin(math.radians(3.0)))
            output.extend((
                initial.rotation.inv() * tag.rotation
            ).as_rotvec() / np.radians(45.0))
        return np.asarray(output)

    pixel_coordinate_count = len(observations) * 8
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
        row += 8
    bundle_sparsity[row:row + 6, 0:6] = 1
    row += 6
    for index in range(camera_count - 1):
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
    bundle_solution = least_squares(
        bundle_residual,
        bundle_x0,
        jac_sparsity=bundle_sparsity.tocsr(),
        loss="huber",
        f_scale=2.0,
        x_scale="jac",
        max_nfev=300,
    )
    bundle_final = bundle_residual(bundle_solution.x)
    bundle_initial_rms = float(np.sqrt(np.mean(
        bundle_initial[:pixel_coordinate_count] ** 2
    )))
    bundle_final_rms = float(np.sqrt(np.mean(
        bundle_final[:pixel_coordinate_count] ** 2
    )))
    if math.isfinite(bundle_final_rms) and bundle_final_rms < bundle_initial_rms:
        cameras = [
            bundle_camera(bundle_solution.x, index)
            for index in range(camera_count)
        ]
        refined_tags.update({
            tag_id: bundle_tag(bundle_solution.x, tag_id)
            for tag_id in observed_tag_ids
        })
        refined_floor_tags = {
            tag_id: bundle_floor_tag(bundle_solution.x, tag_id)
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
        world_from_body_report = alignment_rebase.compose(
            body_at(model_values)
        )
    else:
        alignment_rebase = RigidTransform(np.zeros(3), Rotation.identity())
        world_from_body_report = body_at(model_values)
        refined_floor_tags = {
            tag_id: floor_tags[tag_id] for tag_id in observed_floor_ids
        }
    final_errors: list[float] = []
    floor_errors: list[float] = []
    robot_errors: list[float] = []
    per_tag_errors: dict[int, list[float]] = {}
    per_frame_errors: dict[int, list[float]] = {}
    tag_view_counts: dict[int, int] = {}
    for frame_index, tag_id, pixels in observations:
        tag = refined_floor_tags.get(tag_id, refined_tags.get(tag_id))
        if tag is None:
            continue
        errors = (_project(
            tag,
            cameras[frame_index],
            frames[frame_index].camera_matrix,
            object_corners[tag_id],
        ) - pixels).reshape(-1)
        final_errors.extend(errors)
        if tag_id in floor_tags:
            floor_errors.extend(errors)
        else:
            robot_errors.extend(errors)
        per_tag_errors.setdefault(tag_id, []).extend(errors)
        per_frame_errors.setdefault(frame_index, []).extend(errors)
        tag_view_counts[tag_id] = tag_view_counts.get(tag_id, 0) + 1

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
        for frame_index, _pixels in by_tag[tag_id]:
            direction = (
                cameras[frame_index].translation_m - refined.translation_m
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
            "archived_views": len(by_tag.get(tag_id, [])),
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
            "world_from_camera": camera.to_dict(),
            "coordinate_rms_px": rms(per_frame_errors.get(index, [])),
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
    quality_passed = (
        final_rms <= 3.0
        and floor_rms <= 3.0
        and robot_rms <= 3.5
    )
    report = {
        "ok": quality_passed,
        "method": (
            "image-first joint bundle adjustment of archived full-resolution "
            "corners; one fixed floor origin plus measured coplanar floor tags; "
            "ARKit relative motion; BuildViz used only for initialization "
            "and as a post-fit diagnostic"
        ),
        "frames": len(frames),
        "corner_observations": len(observations),
        "robot_tag_count": len(by_tag),
        "initial_coordinate_rms_px": rms(initial_pixel_residual),
        "physical_model_coordinate_rms_px": physical_rms,
        "pre_bundle_coordinate_rms_px": bundle_initial_rms,
        "final_coordinate_rms_px": final_rms,
        "floor_coordinate_rms_px": floor_rms,
        "robot_coordinate_rms_px": robot_rms,
        "floor_anchor_tag_id": floor_anchor_id,
        "quality_gate": {
            "passed": quality_passed,
            "basis": "image reprojection only; BuildViz disagreement is diagnostic",
            "final_coordinate_rms_px_max": 3.0,
            "floor_coordinate_rms_px_max": 3.0,
            "robot_coordinate_rms_px_max": 3.5,
        },
        "bundle_optimizer": {
            "success": bool(bundle_solution.success),
            "message": str(bundle_solution.message),
            "function_evaluations": int(bundle_solution.nfev),
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
        report["skipped_reason"] = (
            "joint photo reprojection residual exceeds the image-fit accuracy gate"
        )
    return refined_survey, report
