"""Audit AprilTag layout files against still-image evidence.

The audit intentionally separates facts a detector can establish (IDs and
canonical corner order) from physical mount assignments recorded by a human.
It emits annotated images and JSON so those assignments can be reviewed rather
than silently inferred from tag proximity.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .paths import CONFIG_DIR


AXIS_VECTORS = {
    "+x": np.asarray([1.0, 0.0, 0.0]),
    "-x": np.asarray([-1.0, 0.0, 0.0]),
    "+y": np.asarray([0.0, 1.0, 0.0]),
    "-y": np.asarray([0.0, -1.0, 0.0]),
    "+z": np.asarray([0.0, 0.0, 1.0]),
    "-z": np.asarray([0.0, 0.0, -1.0]),
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ids_with_duplicates(values: Iterable[int]) -> list[int]:
    return sorted(value for value, count in Counter(values).items() if count > 1)


def _expected_yoke_pairs(layout: dict[str, Any]) -> dict[tuple[int, str], list[int]]:
    pairs: dict[tuple[int, str], dict[str, int]] = {}
    for tag in layout["robot_tags"]:
        if tag.get("kind") != "yoke_face":
            continue
        key = (int(tag["leg"]), str(tag["joint"]))
        pairs.setdefault(key, {})[str(tag["mount_side"])] = int(tag["id"])
    return {
        key: [faces["+y"], faces["-y"]]
        for key, faces in pairs.items()
        if set(faces) == {"+y", "-y"}
    }


def validate_layout(
    layout: dict[str, Any],
    floor_map: dict[str, Any] | None = None,
    part_map: dict[str, Any] | None = None,
) -> list[str]:
    """Return machine-readable-layout consistency problems."""

    issues: list[str] = []
    robot_tags = layout.get("robot_tags", [])
    floor_tags = layout.get("floor", {}).get("tags", [])
    robot_ids = [int(tag["id"]) for tag in robot_tags]
    floor_ids = [int(tag["id"]) for tag in floor_tags]
    for duplicate in _ids_with_duplicates(robot_ids):
        issues.append(f"duplicate robot tag ID {duplicate}")
    for duplicate in _ids_with_duplicates(floor_ids):
        issues.append(f"duplicate floor tag ID {duplicate}")
    overlap = sorted(set(robot_ids) & set(floor_ids))
    if overlap:
        issues.append(f"robot/floor ID overlap: {overlap}")
    if layout.get("unresolved_mounts"):
        issues.append("layout has unresolved mounts")
    if layout.get("id_collisions"):
        issues.append("layout declares ID collisions")

    chassis = [tag for tag in robot_tags if tag.get("kind") == "chassis_tag"]
    if len(chassis) != 1:
        issues.append(f"expected one chassis tag, found {len(chassis)}")
    for leg in range(6):
        for joint in ("hip", "knee"):
            lids = [
                tag
                for tag in robot_tags
                if tag.get("kind") == "servo_lid"
                and tag.get("leg") == leg
                and tag.get("joint") == joint
            ]
            if len(lids) != 1:
                issues.append(f"L{leg} {joint} has {len(lids)} servo-lid tags")
            faces = [
                tag
                for tag in robot_tags
                if tag.get("kind") == "yoke_face"
                and tag.get("leg") == leg
                and tag.get("joint") == joint
            ]
            sides = [str(tag.get("mount_side")) for tag in faces]
            if sorted(sides) != ["+y", "-y"]:
                issues.append(f"L{leg} {joint} yoke sides are {sorted(sides)}")

    for tag in robot_tags:
        transform = tag.get("frame_from_tag", {})
        quaternion = transform.get("quaternion_xyzw")
        axes = transform.get("tag_axes_in_frame")
        if quaternion is None:
            if "euler_xyz_deg" not in transform:
                issues.append(f"tag {tag['id']} has no orientation")
            continue
        norm = math.sqrt(sum(float(value) ** 2 for value in quaternion))
        if abs(norm - 1.0) > 1e-5:
            issues.append(f"tag {tag['id']} quaternion norm is {norm:.6f}")
        if not axes:
            issues.append(f"tag {tag['id']} has a quaternion but no readable axes")
            continue
        try:
            declared = np.column_stack([AXIS_VECTORS[axes[name]] for name in "xyz"])
        except KeyError as exc:
            issues.append(f"tag {tag['id']} has invalid axis token {exc.args[0]!r}")
            continue
        calculated = Rotation.from_quat(quaternion).as_matrix()
        if not np.allclose(declared, calculated, atol=1e-5):
            issues.append(f"tag {tag['id']} quaternion disagrees with readable axes")

    if floor_map is not None:
        active = {int(value) for value in floor_map.get("active_anchor_ids", [])}
        if active != set(floor_ids):
            issues.append(
                f"floor map IDs {sorted(active)} differ from layout {sorted(floor_ids)}"
            )
        by_id = {int(tag["id"]): tag for tag in floor_map.get("tags", [])}
        for tag in floor_tags:
            tag_id = int(tag["id"])
            if tag_id not in by_id:
                continue
            meters = tag["world_from_tag"]["translation_m"]
            millimeters = by_id[tag_id]["center"]
            if not np.allclose(np.asarray(meters) * 1000.0, millimeters, atol=1e-6):
                issues.append(f"floor tag {tag_id} position differs between configs")
            if abs(float(tag["yaw_deg"]) - float(by_id[tag_id]["yaw_degrees"])) > 1e-6:
                issues.append(f"floor tag {tag_id} yaw differs between configs")

    if part_map is not None:
        expected = _expected_yoke_pairs(layout)
        actual: dict[tuple[int, str], list[int]] = {}
        for part in part_map.get("parts", []):
            part_id = str(part["id"])
            try:
                prefix, joint, _suffix = part_id.split("_", 2)
                key = (int(prefix.removeprefix("leg")), joint)
            except (TypeError, ValueError):
                issues.append(f"cannot parse part ID {part_id!r}")
                continue
            actual[key] = [int(value) for value in part["tag_ids"]]
        if actual != expected:
            issues.append("planar part-map yoke pairs differ from the physical layout")
    return issues


def detect_image(path: Path) -> tuple[np.ndarray, list[dict[str, Any]]]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not read image: {path}")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    corners, ids, _rejected = detector.detectMarkers(image)
    detections: list[dict[str, Any]] = []
    if ids is None:
        return image, detections
    for marker_id, marker_corners in zip(ids.reshape(-1), corners):
        points = marker_corners.reshape(4, 2).astype(np.float64)
        center = points.mean(axis=0)
        area = abs(float(cv2.contourArea(points.astype(np.float32))))
        detections.append(
            {
                "id": int(marker_id),
                "center_px": [round(float(value), 3) for value in center],
                "corners_px": [
                    [round(float(value), 3) for value in point] for point in points
                ],
                "area_px2": round(area, 3),
                "tag_x_vector_px": [
                    round(float(value), 3) for value in points[1] - points[0]
                ],
                "tag_y_vector_px": [
                    round(float(value), 3) for value in points[0] - points[3]
                ],
            }
        )
    return image, sorted(detections, key=lambda item: (item["id"], item["center_px"]))


def annotate_image(
    image: np.ndarray,
    detections: list[dict[str, Any]],
    floor_ids: set[int],
) -> np.ndarray:
    annotated = image.copy()
    scale = max(0.7, min(image.shape[:2]) / 1400.0)
    thickness = max(2, round(scale * 3))
    for detection in detections:
        points = np.rint(detection["corners_px"]).astype(np.int32)
        center = np.rint(detection["center_px"]).astype(np.int32)
        color = (255, 180, 0) if detection["id"] in floor_ids else (255, 255, 255)
        cv2.polylines(annotated, [points], True, color, thickness, cv2.LINE_AA)
        # OpenCV canonical tag axes: +X is corner 0 -> 1; +Y is corner 3 -> 0.
        cv2.arrowedLine(
            annotated, tuple(points[0]), tuple(points[1]), (0, 0, 255), thickness,
            cv2.LINE_AA, tipLength=0.2,
        )
        cv2.arrowedLine(
            annotated, tuple(points[3]), tuple(points[0]), (0, 255, 0), thickness,
            cv2.LINE_AA, tipLength=0.2,
        )
        label_at = (int(center[0] + 6 * scale), int(center[1] - 6 * scale))
        cv2.putText(
            annotated, str(detection["id"]), label_at, cv2.FONT_HERSHEY_SIMPLEX,
            scale, (0, 0, 0), thickness + 3, cv2.LINE_AA,
        )
        cv2.putText(
            annotated, str(detection["id"]), label_at, cv2.FONT_HERSHEY_SIMPLEX,
            scale, (0, 255, 255), thickness, cv2.LINE_AA,
        )
    return annotated


def _unit(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length <= 1e-9:
        raise ValueError("cannot normalize a zero-length vector")
    return vector / length


def _angle_degrees(vector: np.ndarray) -> float:
    return math.degrees(math.atan2(float(vector[1]), float(vector[0])))


def _angle_difference(first: float, second: float) -> float:
    return (first - second + 180.0) % 360.0 - 180.0


def _detection_index(image_record: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = {}
    for detection in image_record["detections"]:
        result.setdefault(int(detection["id"]), []).append(detection)
    return result


def audit_side_orientations(
    layout: dict[str, Any], image_records: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    """Check vertical-tag axes from projected leg and lid directions.

    This is a discrete orientation check for the photographed straight-leg
    pose.  The hip-to-knee centerline supplies link +X.  The direction from a
    side marker toward its corresponding lid supplies link +Z.  The recorded
    mount side supplies the outward-normal sign.  Requiring agreement across
    these independent image measurements catches wrong IDs, joints, faces, and
    quarter-turn rotations without pretending to recover metric 3-D geometry.
    """

    lids = {
        (int(tag["leg"]), str(tag["joint"])): int(tag["id"])
        for tag in layout["robot_tags"]
        if tag.get("kind") == "servo_lid"
    }
    side_tags = {
        int(tag["id"]): tag
        for tag in layout["robot_tags"]
        if tag.get("kind") == "yoke_face"
    }
    samples: dict[int, list[dict[str, Any]]] = {tag_id: [] for tag_id in side_tags}
    for image_record in image_records:
        by_id = _detection_index(image_record)
        for tag_id, tag in side_tags.items():
            hip_id = lids[(int(tag["leg"]), "hip")]
            knee_id = lids[(int(tag["leg"]), "knee")]
            required = (tag_id, hip_id, knee_id)
            if any(len(by_id.get(required_id, [])) != 1 for required_id in required):
                continue
            detection = by_id[tag_id][0]
            hip_center = np.asarray(by_id[hip_id][0]["center_px"], dtype=np.float64)
            knee_center = np.asarray(by_id[knee_id][0]["center_px"], dtype=np.float64)
            link_x = _unit(knee_center - hip_center)
            marker_center = np.asarray(detection["center_px"], dtype=np.float64)
            lid_center = hip_center if tag["joint"] == "hip" else knee_center
            toward_lid = _unit(lid_center - marker_center)
            image_axes = {
                "x": _unit(np.asarray(detection["tag_x_vector_px"], dtype=np.float64)),
                "y": _unit(np.asarray(detection["tag_y_vector_px"], dtype=np.float64)),
            }
            link_axis = max(image_axes, key=lambda name: abs(float(image_axes[name] @ link_x)))
            vertical_axis = "y" if link_axis == "x" else "x"
            predicted = {
                link_axis: ("+" if image_axes[link_axis] @ link_x > 0 else "-") + "x",
                vertical_axis: (
                    "+" if image_axes[vertical_axis] @ toward_lid > 0 else "-"
                ) + "z",
                "z": str(tag["mount_side"]),
            }
            declared = tag["frame_from_tag"]["tag_axes_in_frame"]
            link_score = abs(float(image_axes[link_axis] @ link_x))
            vertical_score = abs(float(image_axes[vertical_axis] @ toward_lid))
            samples[tag_id].append(
                {
                    "image": Path(image_record["path"]).name,
                    "predicted_tag_axes_in_frame": predicted,
                    "matches_layout": predicted == declared,
                    "link_axis_cosine": round(link_score, 4),
                    "vertical_axis_cosine": round(vertical_score, 4),
                }
            )
    results: list[dict[str, Any]] = []
    mismatches: list[int] = []
    no_evidence: list[int] = []
    for tag_id, tag in sorted(side_tags.items()):
        tag_samples = samples[tag_id]
        confirmed = bool(tag_samples) and all(
            sample["matches_layout"]
            and sample["link_axis_cosine"] >= 0.9
            and sample["vertical_axis_cosine"] >= 0.55
            for sample in tag_samples
        )
        if not tag_samples:
            status = "no_evidence"
            no_evidence.append(tag_id)
        elif confirmed:
            status = "confirmed"
        else:
            status = "mismatch"
            mismatches.append(tag_id)
        results.append(
            {
                "id": tag_id,
                "frame": tag["frame"],
                "mount_side": tag["mount_side"],
                "declared_tag_axes_in_frame": tag["frame_from_tag"]["tag_axes_in_frame"],
                "status": status,
                "samples": tag_samples,
            }
        )
    return {
        "method": "straight-leg projected centerline and side-tag-to-lid direction",
        "confirmed_count": sum(result["status"] == "confirmed" for result in results),
        "mismatch_ids": mismatches,
        "no_evidence_ids": no_evidence,
        "tags": results,
    }


def _floor_corners(tag: dict[str, Any], tag_size_mm: float) -> np.ndarray:
    half = tag_size_mm / 2.0
    offsets = np.asarray(
        [[-half, half], [half, half], [half, -half], [-half, -half]],
        dtype=np.float64,
    )
    yaw = math.radians(float(tag["yaw_degrees"]))
    rotation = np.asarray(
        [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
        dtype=np.float64,
    )
    return np.asarray(tag["center"][:2], dtype=np.float64) + offsets @ rotation.T


def _project(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(
        np.asarray(points, dtype=np.float32).reshape(1, -1, 2), homography
    )[0].astype(np.float64)


def audit_horizontal_orientations(
    layout: dict[str, Any],
    floor_map: dict[str, Any],
    image_records: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Check chassis and lid yaw after rectifying an overhead floor view."""

    horizontal = {
        int(tag["id"]): tag
        for tag in layout["robot_tags"]
        if tag.get("surface") == "horizontal"
    }
    chassis_id = next(
        int(tag["id"]) for tag in horizontal.values() if tag["kind"] == "chassis_tag"
    )
    lids = {
        (int(tag["leg"]), str(tag["joint"])): int(tag["id"])
        for tag in horizontal.values()
        if tag["kind"] == "servo_lid"
    }
    floor_by_id = {int(tag["id"]): tag for tag in floor_map["tags"]}
    samples: dict[int, list[dict[str, Any]]] = {tag_id: [] for tag_id in horizontal}
    for image_record in image_records:
        by_id = _detection_index(image_record)
        visible_floor = [
            tag_id for tag_id in floor_by_id if len(by_id.get(tag_id, [])) == 1
        ]
        if len(visible_floor) < 4 or len(by_id.get(chassis_id, [])) != 1:
            continue
        image_points = np.concatenate(
            [np.asarray(by_id[tag_id][0]["corners_px"]) for tag_id in visible_floor]
        )
        world_points = np.concatenate(
            [
                _floor_corners(floor_by_id[tag_id], float(floor_map["tag_black_square_size"]))
                for tag_id in visible_floor
            ]
        )
        homography, _mask = cv2.findHomography(
            image_points.astype(np.float32), world_points.astype(np.float32), cv2.RANSAC, 3.0
        )
        if homography is None:
            continue
        centers = {
            tag_id: _project(np.asarray([detections[0]["center_px"]]), homography)[0]
            for tag_id, detections in by_id.items()
            if len(detections) == 1
        }
        if any(lid_id not in centers for lid_id in lids.values()):
            continue
        body_center = centers[chassis_id]
        body_yaw_candidates = []
        for leg in range(6):
            observed = _angle_degrees(centers[lids[(leg, "knee")]] - body_center)
            body_yaw_candidates.append(observed - (leg + 0.5) * 60.0)
        vector = sum(np.exp(1j * np.radians(body_yaw_candidates)))
        body_yaw = math.degrees(math.atan2(float(vector.imag), float(vector.real)))

        for tag_id, tag in horizontal.items():
            if len(by_id.get(tag_id, [])) != 1:
                continue
            corners = _project(
                np.asarray(by_id[tag_id][0]["corners_px"], dtype=np.float64), homography
            )
            tag_yaw = _angle_degrees(corners[1] - corners[0])
            declared_yaw = float(tag["frame_from_tag"]["euler_xyz_deg"][2])
            if tag["kind"] == "chassis_tag":
                measured_yaw = _angle_difference(tag_yaw, body_yaw)
                error = abs(_angle_difference(measured_yaw, declared_yaw))
                matches = error <= 5.0
            else:
                leg = int(tag["leg"])
                hip_center = centers[lids[(leg, "hip")]]
                if tag["joint"] == "hip":
                    frame_x = hip_center - body_center
                else:
                    frame_x = centers[lids[(leg, "knee")]] - hip_center
                measured_yaw = _angle_difference(tag_yaw, _angle_degrees(frame_x))
                measured_yaw = round(measured_yaw / 90.0) * 90.0
                error = abs(_angle_difference(measured_yaw, declared_yaw))
                matches = error < 1e-6
            samples[tag_id].append(
                {
                    "image": Path(image_record["path"]).name,
                    "measured_euler_z_deg": round(measured_yaw, 3),
                    "declared_euler_z_deg": declared_yaw,
                    "matches_layout": matches,
                    "angular_error_deg": round(error, 3),
                    "floor_anchor_count": len(visible_floor),
                }
            )
    results: list[dict[str, Any]] = []
    mismatches: list[int] = []
    no_evidence: list[int] = []
    for tag_id, tag in sorted(horizontal.items()):
        tag_samples = samples[tag_id]
        if not tag_samples:
            status = "no_evidence"
            no_evidence.append(tag_id)
        elif all(sample["matches_layout"] for sample in tag_samples):
            status = "confirmed"
        else:
            status = "mismatch"
            mismatches.append(tag_id)
        results.append(
            {
                "id": tag_id,
                "frame": tag["frame"],
                "status": status,
                "samples": tag_samples,
            }
        )
    return {
        "method": "floor-homography rectification plus photographed leg azimuths",
        "confirmed_count": sum(result["status"] == "confirmed" for result in results),
        "mismatch_ids": mismatches,
        "no_evidence_ids": no_evidence,
        "tags": results,
    }


def audit_floor_orientations(
    floor_map: dict[str, Any], image_records: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    """Re-measure floor-tag yaw using only the configured grid centers."""

    floor_by_id = {int(tag["id"]): tag for tag in floor_map["tags"]}
    samples: dict[int, list[dict[str, Any]]] = {tag_id: [] for tag_id in floor_by_id}
    views: list[dict[str, Any]] = []
    for image_record in image_records:
        by_id = _detection_index(image_record)
        visible = [
            tag_id for tag_id in floor_by_id if len(by_id.get(tag_id, [])) == 1
        ]
        if len(visible) < 4:
            continue
        image_centers = np.asarray(
            [by_id[tag_id][0]["center_px"] for tag_id in visible], dtype=np.float32
        )
        world_centers = np.asarray(
            [floor_by_id[tag_id]["center"][:2] for tag_id in visible], dtype=np.float32
        )
        # All configured centers are deliberate grid correspondences. A direct
        # least-squares fit is deterministic and keeps every anchor in the
        # residual instead of allowing RANSAC to hide a misplaced floor tag.
        homography, _mask = cv2.findHomography(image_centers, world_centers, 0)
        if homography is None:
            continue
        projected_centers = _project(image_centers, homography)
        center_rms = float(
            np.sqrt(np.mean(np.sum(np.square(projected_centers - world_centers), axis=1)))
        )
        views.append(
            {
                "image": Path(image_record["path"]).name,
                "anchor_count": len(visible),
                "center_fit_rms_mm": round(center_rms, 4),
            }
        )
        for tag_id in visible:
            detection = by_id[tag_id][0]
            corners = _project(
                np.asarray(detection["corners_px"], dtype=np.float64), homography
            )
            measured = _angle_degrees(corners[1] - corners[0])
            configured = float(floor_by_id[tag_id]["yaw_degrees"])
            error = abs(_angle_difference(measured, configured))
            tolerance = float(
                floor_by_id[tag_id].get("yaw_uncertainty_degrees", 1.5)
            )
            samples[tag_id].append(
                {
                    "image": Path(image_record["path"]).name,
                    "measured_yaw_deg": round(measured, 3),
                    "configured_yaw_deg": configured,
                    "angular_error_deg": round(error, 3),
                    "matches_layout": error <= tolerance,
                }
            )
    results: list[dict[str, Any]] = []
    mismatches: list[int] = []
    no_evidence: list[int] = []
    for tag_id in sorted(floor_by_id):
        tag_samples = samples[tag_id]
        if not tag_samples:
            status = "no_evidence"
            no_evidence.append(tag_id)
        elif all(sample["matches_layout"] for sample in tag_samples):
            status = "confirmed"
        else:
            status = "mismatch"
            mismatches.append(tag_id)
        results.append({"id": tag_id, "status": status, "samples": tag_samples})
    return {
        "method": "center-only grid homography; tag corners withheld from the fit",
        "confirmed_count": sum(result["status"] == "confirmed" for result in results),
        "mismatch_ids": mismatches,
        "no_evidence_ids": no_evidence,
        "views": views,
        "tags": results,
    }


def audit_images(
    layout: dict[str, Any],
    image_paths: Iterable[Path],
    floor_map: dict[str, Any] | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    robot_ids = {int(tag["id"]) for tag in layout["robot_tags"]}
    floor_ids = {int(tag["id"]) for tag in layout["floor"]["tags"]}
    expected_ids = robot_ids | floor_ids
    seen_ids: set[int] = set()
    images: list[dict[str, Any]] = []
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    for path in image_paths:
        image, detections = detect_image(path)
        ids = [int(item["id"]) for item in detections]
        seen_ids.update(ids)
        record = {
            "path": str(path),
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "detections": detections,
            "duplicate_ids": _ids_with_duplicates(ids),
        }
        if output_dir is not None:
            output_path = output_dir / f"{path.stem}-annotated.jpg"
            if not cv2.imwrite(
                str(output_path), annotate_image(image, detections, floor_ids)
            ):
                raise ValueError(f"could not write annotation: {output_path}")
            record["annotation"] = str(output_path)
        images.append(record)
    report = {
        "schema_version": 1,
        "tag_family": layout["tag_family"],
        "layout_name": layout["name"],
        "expected_ids": sorted(expected_ids),
        "detected_ids": sorted(seen_ids),
        "missing_ids": sorted(expected_ids - seen_ids),
        "unexpected_ids": sorted(seen_ids - expected_ids),
        "images": images,
    }
    report["side_orientation_audit"] = audit_side_orientations(layout, images)
    if floor_map is not None:
        report["horizontal_orientation_audit"] = audit_horizontal_orientations(
            layout, floor_map, images
        )
        report["floor_orientation_audit"] = audit_floor_orientations(floor_map, images)
    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "images", nargs="+", type=Path, help="still images to scan at full resolution"
    )
    parser.add_argument(
        "--layout",
        type=Path,
        default=CONFIG_DIR / "hexapod-1-apriltag-layout.json",
    )
    parser.add_argument("--floor-map", type=Path, default=CONFIG_DIR / "floor_tag_map.json")
    parser.add_argument("--part-map", type=Path, default=CONFIG_DIR / "hexapod_tag_map.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--require-all-layout-ids",
        action="store_true",
        help="exit nonzero unless the image set covers every configured ID",
    )
    parser.add_argument(
        "--require-all-orientations",
        action="store_true",
        help="exit nonzero unless every robot and floor orientation is confirmed",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    layout = _load_json(args.layout)
    problems = validate_layout(
        layout,
        floor_map=_load_json(args.floor_map),
        part_map=_load_json(args.part_map),
    )
    floor_map = _load_json(args.floor_map)
    report = audit_images(layout, args.images, floor_map, args.output_dir)
    report["layout_validation"] = {"ok": not problems, "issues": problems}
    rendered = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if problems:
        return 2
    if args.require_all_layout_ids and (
        report["missing_ids"] or report["unexpected_ids"]
    ):
        return 1
    if args.require_all_orientations:
        for key in (
            "side_orientation_audit",
            "horizontal_orientation_audit",
            "floor_orientation_audit",
        ):
            audit = report[key]
            if audit["mismatch_ids"] or audit["no_evidence_ids"]:
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
