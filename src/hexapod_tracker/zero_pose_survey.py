#!/usr/bin/env python3
"""Guide a handheld iPhone survey around a stationary zero-pose hexapod."""
from __future__ import annotations

import argparse
import base64
from collections.abc import Iterator, Sequence
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np

from .apriltag_vision import TagCorners, detect_tag_corners
from .housing_pose import RigidTransform
from .rgbd_calibrate import (
    RGBDFrame,
    Record3DReader,
    _calibration_target,
    _load_json,
    _npz_frames,
)
from .rgbd_calibration import (
    RGBDCalibrationError,
    RGBDCalibrationOptions,
    refine_world_reference_with_depth,
)
from .tag_survey import (
    HandheldWorldAlignment,
    TagSurveyAccumulator,
    TagSurveyOptions,
    apply_survey_to_config,
    arkit_world_from_opencv_camera,
    learn_zero_pose_mounts,
    merge_robot_layout_into_config,
)


def _parse_ids(value: str) -> tuple[int, ...]:
    try:
        result = tuple(sorted({
            int(part.strip()) for part in value.split(",") if part.strip()
        }))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "tag IDs must be comma-separated integers"
        ) from error
    if any(tag_id < 0 for tag_id in result):
        raise argparse.ArgumentTypeError("tag IDs cannot be negative")
    return result


def _write_progress(path: Path | None, payload: dict[str, Any]) -> None:
    """Atomically publish a compact live snapshot for the local web UI."""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    temporary.replace(path)


def _camera_preview(
    image: np.ndarray,
    detections: Sequence[TagCorners],
) -> np.ndarray:
    """Return a clean, labelled camera frame for the browser walkthrough."""
    output = image.copy()
    for detection in detections:
        corners = np.rint(detection.corners_px).astype(np.int32)
        cv2.polylines(output, [corners], True, (45, 214, 171), 3, cv2.LINE_AA)
        center = tuple(np.rint(detection.center_px).astype(int))
        label = f"#{detection.tag_id}"
        (text_width, text_height), _baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )
        label_origin = (center[0] + 8, center[1] - 8)
        cv2.rectangle(
            output,
            (label_origin[0] - 4, label_origin[1] - text_height - 5),
            (label_origin[0] + text_width + 4, label_origin[1] + 5),
            (22, 35, 44),
            -1,
        )
        cv2.putText(
            output,
            label,
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (245, 255, 251),
            2,
            cv2.LINE_AA,
        )
    return output


def _guidance(
    phase: str,
    progress: dict[str, Any],
    anchor_ids: Sequence[int],
    *,
    min_observations: int,
    resumed: bool,
) -> dict[str, Any]:
    """Choose one concrete target and one physical action for the operator."""
    if phase == "anchor":
        ids = ", ".join(f"#{tag_id}" for tag_id in sorted(anchor_ids))
        mapped = len(anchor_ids) > 1
        return {
            "headline": (
                f"Point down at any mapped floor tag ({ids})"
                if mapped else f"Point down at floor origin {ids}"
            ),
            "detail": (
                "Keep at least one whole square visible—two or more is even "
                "better—and hold the phone still until the metric floor grid locks."
                if mapped else
                "Keep the entire black square visible, fill roughly a quarter "
                "of the image, and hold the phone still until the origin locks."
            ),
            "action": "Hold still over the mapped floor tags.",
            "target_kind": "floor",
            "target_tag_id": int(sorted(anchor_ids)[0]),
            "target_position": None,
            "target_state": "relock" if resumed else "not_seen",
            "remaining_targets": 0,
        }

    positions = list(progress.get("robot_positions", []))
    needs_view = next((
        item for item in positions
        if item.get("state") == "seen_needs_another_view"
    ), None)
    if needs_view is not None:
        observed = int(needs_view.get("observations", 0))
        remaining = max(1, min_observations - observed)
        tag_id = needs_view.get("tag_id")
        position = str(needs_view["position"])
        is_side = needs_view.get("kind") == "yoke_face"
        return {
            "headline": f"Finish {position} — keep tag #{tag_id} in view",
            "detail": (
                f"It has {observed} clean view{'s' if observed != 1 else ''}. "
                "Lower the phone and aim squarely at the vertical face; take a "
                "small arc around that joint, then hold for about one second."
                if is_side else
                "Take one small step sideways (20–30 cm), aim squarely at it, "
                "then hold for about one second."
            ),
            "action": f"Move sideways, then hold steady for {remaining} more clean view{'s' if remaining != 1 else ''}.",
            "target_kind": "robot",
            "target_tag_id": tag_id,
            "target_position": position,
            "target_state": "seen_needs_another_view",
            "remaining_targets": len(progress.get("missing_robot_positions", [])),
        }

    unseen = next((
        item for item in positions if item.get("state") == "not_seen"
    ), None)
    if unseen is not None:
        position = str(unseen["position"])
        is_side = unseen.get("kind") == "yoke_face"
        return {
            "headline": f"Find the tag mounted at {position}",
            "detail": (
                "Move to the named side of that joint and lower the phone until "
                "the vertical square is nearly face-on. Keep a floor tag near "
                "the edge of frame when possible."
                if is_side else
                "Circle slowly around the robot until that physical mount is "
                "unobstructed. The tag ID may be new; keep the whole square in "
                "frame until this highlighted row turns green."
            ),
            "action": "Walk slowly around the robot; stop as soon as the tag is outlined.",
            "target_kind": "robot",
            "target_tag_id": None,
            "target_position": position,
            "target_state": "not_seen",
            "remaining_targets": len(progress.get("missing_robot_positions", [])),
        }

    ground_status = list(progress.get("ground_tag_status", []))
    ground_needs_view = next((
        item for item in ground_status
        if item.get("state") == "seen_needs_another_view"
    ), None)
    if ground_needs_view is not None:
        tag_id = int(ground_needs_view["tag_id"])
        return {
            "headline": f"Finish floor tag #{tag_id}",
            "detail": (
                "Point the phone down, keep all four corners visible, step a "
                "little to one side, and hold until the floor card turns green."
            ),
            "action": "Take a second oblique view, then hold still for one second.",
            "target_kind": "floor",
            "target_tag_id": tag_id,
            "target_position": None,
            "target_state": "seen_needs_another_view",
            "remaining_targets": len(progress.get("missing_ground_tag_ids", [])),
        }

    ground_unseen = next((
        item for item in ground_status if item.get("state") == "not_seen"
    ), None)
    if ground_unseen is not None:
        tag_id = int(ground_unseen["tag_id"])
        return {
            "headline": f"Find floor tag #{tag_id}",
            "detail": (
                "Walk toward that printed floor tag and point down. Keep the "
                "whole square visible until its card turns green."
            ),
            "action": "Point down at the floor tag and hold for one second.",
            "target_kind": "floor",
            "target_tag_id": tag_id,
            "target_position": None,
            "target_state": "not_seen",
            "remaining_targets": len(progress.get("missing_ground_tag_ids", [])),
        }

    quality_gate = progress.get("quality_gate") or {}
    if quality_gate and not quality_gate.get("passed"):
        return {
            "headline": "Coverage is complete—tighten the floor lock",
            "detail": "; ".join(quality_gate.get("failing_checks", [])) or (
                "The global geometry check has not passed yet."
            ),
            "action": (
                "Frame two or more floor tags together, hold still, then take "
                "one slow arc with a floor tag kept in view."
            ),
            "target_kind": "floor",
            "target_tag_id": None,
            "target_position": None,
            "target_state": "quality_check",
            "remaining_targets": 0,
        }

    return {
        "headline": "Everything is recorded — hold still",
        "detail": "Keep the phone steady while the last measurements are checked.",
        "action": "Do not move the robot or phone.",
        "target_kind": None,
        "target_tag_id": None,
        "target_position": None,
        "target_state": "complete",
        "remaining_targets": 0,
    }


def _quality_feedback(
    guidance: dict[str, Any],
    records: Sequence[dict[str, Any]],
    detected_ids: set[int],
    *,
    camera_speed_m_s: float | None,
    tracking_message: str,
    anchor_reprojection_rms_px: float | None,
    depth_plane_rms_mm: float | None,
) -> dict[str, Any]:
    target_id = guidance.get("target_tag_id")
    target_record = next((
        item for item in records if int(item["tag_id"]) == target_id
    ), None)
    reprojection = anchor_reprojection_rms_px
    translation_spread = None
    rotation_spread = None
    if target_record is not None:
        reprojection = float(target_record.get("mean_reprojection_rms_px", 0.0))
        translation_spread = float(target_record.get("translation_spread_mm", 0.0))
        rotation_spread = float(target_record.get("rotation_spread_deg", 0.0))
    target_visible = target_id in detected_ids if target_id is not None else None

    level = "good"
    headline = "Good capture — keep moving smoothly"
    suggestion = "Use a slow arc and pause for one second whenever a tag is outlined."
    lower_message = tracking_message.lower()
    if "drift warning" in lower_message:
        level = "poor"
        headline = "Tracking drift is too high"
        suggestion = "Walk back to the floor origin tag and hold it centered until tracking recovers."
    elif "rejected" in lower_message:
        level = "poor"
        headline = "This angle was rejected"
        suggestion = "Hold steadier, move slightly closer, and keep all four tag corners visible."
    elif not detected_ids:
        level = "caution"
        headline = "No tag is visible right now"
        suggestion = "Slow down and tilt the phone toward the highlighted target; avoid motion blur."
    elif camera_speed_m_s is not None and camera_speed_m_s > 0.25:
        level = "caution"
        headline = "You are moving too quickly"
        suggestion = "Walk at about half this speed, then pause for one second on the target."
    elif target_visible is False:
        level = "caution"
        headline = "The target is outside the camera view"
        suggestion = "Continue the slow arc toward the highlighted target and tilt the phone to keep the square face-on."
    elif reprojection is not None and reprojection > 1.0:
        level = "caution"
        headline = "Corner fit is noisy"
        suggestion = "Move a little closer, reduce glare, and hold until the error drops below about 1.0 px."
    elif translation_spread is not None and translation_spread > 8.0:
        level = "caution"
        headline = "Position estimates are still spread out"
        suggestion = "Hold still for one second, then take one clean view from 20–30 cm to the side."
    elif guidance.get("target_state") == "seen_needs_another_view":
        headline = "Target visible — add a different angle"
        suggestion = "Step 20–30 cm sideways without rushing, aim at the same tag, and hold still."

    return {
        "level": level,
        "headline": headline,
        "suggestion": suggestion,
        "visible_tag_count": len(detected_ids),
        "target_visible": target_visible,
        "reprojection_rms_px": (
            None if reprojection is None else round(reprojection, 3)
        ),
        "depth_plane_rms_mm": (
            None if depth_plane_rms_mm is None else round(depth_plane_rms_mm, 2)
        ),
        "translation_spread_mm": (
            None if translation_spread is None else round(translation_spread, 2)
        ),
        "rotation_spread_deg": (
            None if rotation_spread is None else round(rotation_spread, 3)
        ),
        "camera_speed_m_s": (
            None if camera_speed_m_s is None else round(camera_speed_m_s, 3)
        ),
    }


def _wifi_frames(directory: Path, *, timeout_s: float = 8.0) -> Iterator[RGBDFrame]:
    """Consume browser-relayed Record3D WebRTC frames from an atomic spool."""
    latest_path = directory / "latest.json"
    last_sequence = -1
    last_frame_time = time.monotonic()
    while True:
        try:
            payload = json.loads(latest_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, ValueError):
            payload = None
        if isinstance(payload, dict):
            sequence = int(payload.get("sequence", -1))
            if sequence > last_sequence:
                encoded = str(payload.get("rgbd_jpeg_base64", ""))
                if "," in encoded:
                    encoded = encoded.split(",", 1)[1]
                packed = cv2.imdecode(
                    np.frombuffer(base64.b64decode(encoded), dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
                if packed is None or packed.shape[1] < 4:
                    raise ValueError("Wi-Fi frame did not contain a decodable RGB-D image")
                half = packed.shape[1] // 2
                depth_bgr = packed[:, :half]
                rgb_bgr = packed[:, packed.shape[1] - half:]
                hsv = cv2.cvtColor(depth_bgr, cv2.COLOR_BGR2HSV)
                max_depth_m = float(payload.get("max_depth_m", 3.0))
                depth_m = hsv[:, :, 0].astype(np.float32) * (
                    max_depth_m / 179.0
                )
                # The hue wheel wraps maximum-range red to zero. Treat highly
                # saturated near-red values as the configured far/invalid depth.
                red = (hsv[:, :, 0] <= 2) & (hsv[:, :, 1] >= 160)
                depth_m[red] = max_depth_m
                matrix = np.asarray(payload["camera_matrix"], dtype=float).reshape(3, 3)
                pose = np.asarray(
                    payload["camera_pose_xyzw_xyz"], dtype=float
                ).reshape(7)
                last_sequence = sequence
                last_frame_time = time.monotonic()
                yield RGBDFrame(
                    rgb_bgr=rgb_bgr,
                    depth_m=depth_m,
                    confidence=None,
                    camera_matrix=matrix,
                    source_label="Record3D Wi-Fi WebRTC",
                    arkit_world_from_opengl_camera=RigidTransform.from_dict({
                        "quaternion_xyzw": pose[:4],
                        "translation_m": pose[4:],
                    }),
                )
                continue
        if time.monotonic() - last_frame_time >= timeout_s:
            raise TimeoutError(
                "Record3D Wi-Fi stream stopped; reconnect the phone and continue calibration"
            )
        time.sleep(0.04)


def _tag_size_map(
    tracker_config: dict[str, Any],
    board_manifest: dict[str, Any],
    default_marker_size_m: float,
) -> dict[int, float]:
    result: dict[int, float] = {}
    floor_default = float(tracker_config.get(
        "floor_marker_size_m", default_marker_size_m
    ))
    for raw_id, spec in tracker_config.get("floor_tags", {}).items():
        result[int(raw_id)] = float(spec.get("marker_size_m", floor_default))
    for raw_id, spec in tracker_config.get("robot_pose", {}).get("tags", {}).items():
        result[int(raw_id)] = float(spec.get("marker_size_m", default_marker_size_m))
    board_default = float(board_manifest["marker_size_m"])
    for raw_id, spec in board_manifest.get("floor_tags", {}).items():
        result[int(raw_id)] = float(spec.get("marker_size_m", board_default))
    return result


def _rotation_error_deg(first: RigidTransform, second: RigidTransform) -> float:
    return math.degrees(float((first.rotation.inv() * second.rotation).magnitude()))


def _alignment_record(alignment: HandheldWorldAlignment) -> dict[str, Any] | None:
    consensus = alignment.consensus()
    if consensus is None:
        return None
    return {
        "world_from_arkit_world": consensus.transform.to_dict(),
        "observations": consensus.input_count,
        "used_observations": consensus.used_count,
        "translation_spread_mm": round(consensus.translation_spread_mm, 4),
        "rotation_spread_deg": round(consensus.rotation_spread_deg, 5),
        "stable": consensus.stable,
        "ambiguous_cluster": consensus.ambiguous_cluster,
    }


def _annotate(
    image: np.ndarray,
    detections: Sequence[TagCorners],
    *,
    phase: str,
    anchor_ids: set[int],
    robot_ids: set[int],
    ground_ids: set[int],
    stable_ids: set[int],
    progress: dict[str, Any],
    records: Sequence[dict[str, Any]],
    camera_path: Sequence[np.ndarray],
    world_from_camera: RigidTransform | None,
    alignment_count: int,
    anchor_frames: int,
    message: str,
) -> np.ndarray:
    image_height, image_width = image.shape[:2]
    panel_width = 600
    output_height = max(720, image_height)
    output = np.full(
        (output_height, image_width + panel_width, 3),
        (18, 20, 24),
        dtype=np.uint8,
    )
    output[:image_height, :image_width] = image
    position_by_tag = {
        int(item["tag_id"]): item
        for item in progress["robot_positions"]
        if item.get("tag_id") is not None
    }
    ground_status = {
        int(item["tag_id"]): item for item in progress["ground_tag_status"]
    }
    for detection in detections:
        tag_id = detection.tag_id
        if tag_id in anchor_ids:
            color = (30, 220, 30)
            tag_label = f"{tag_id} ANCHOR"
        elif tag_id in position_by_tag:
            position = position_by_tag[tag_id]
            if position["state"] == "measured":
                color = (70, 220, 110)
                state_label = "OK"
            else:
                color = (40, 210, 255)
                state_label = "VIEW"
            tag_label = f"{tag_id} {str(position['position']).upper()} {state_label}"
        elif tag_id in ground_status:
            state = ground_status[tag_id]["state"]
            color = (255, 190, 45) if state == "measured" else (40, 210, 255)
            tag_label = f"{tag_id} FLOOR {'OK' if state == 'measured' else 'VIEW'}"
        elif tag_id in stable_ids:
            color = (255, 170, 40)
            tag_label = f"{tag_id} EXTRA"
        elif tag_id in robot_ids:
            color = (230, 80, 230)
            tag_label = f"{tag_id} ROBOT"
        elif tag_id in ground_ids:
            color = (230, 210, 30)
            tag_label = f"{tag_id} FLOOR"
        else:
            color = (0, 160, 255)
            tag_label = str(tag_id)
        corners = np.rint(detection.corners_px).astype(np.int32)
        cv2.polylines(output, [corners], True, color, 3, cv2.LINE_AA)
        center = tuple(np.rint(detection.center_px).astype(int))
        cv2.putText(
            output,
            tag_label,
            (center[0] + 7, center[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )
    panel_height = min(118, image_height)
    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (image_width, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.72, output, 0.28, 0.0, output)
    if phase == "anchor":
        title = "STEP 1 OF 2  LOCK THE WORLD ORIGIN"
        counts = f"Good anchor frames: {alignment_count}/{anchor_frames}"
        instruction = "NEXT: Keep the floor anchor centered and hold the phone still."
    else:
        measured_robot = sum(
            item["state"] == "measured" for item in progress["robot_positions"]
        )
        measured_floor = sum(
            item["state"] == "measured" for item in progress["ground_tag_status"]
        )
        title = "STEP 2 OF 2  MAP EVERY PHYSICAL TAG POSITION"
        counts = (
            f"Recorded: {measured_robot}/{len(progress['robot_positions'])} robot positions"
            f"  |  {measured_floor}/{len(progress['ground_tag_status'])} floor tags"
        )
        if "drift" in message.lower() or "rejected" in message.lower():
            instruction = "NEXT: Point at the floor anchor and hold still to restore tracking."
        elif progress["unseen_robot_positions"]:
            names = ", ".join(progress["unseen_robot_positions"][:4])
            instruction = f"NEXT: Walk until these positions enter view: {names}"
        elif progress["robot_positions_needing_another_view"]:
            names = ", ".join(progress["robot_positions_needing_another_view"][:4])
            instruction = f"NEXT: Get a slower, closer view of: {names}"
        elif progress["unseen_ground_tag_ids"]:
            instruction = f"NEXT: Find floor tags {progress['unseen_ground_tag_ids']}"
        elif progress["ground_tags_needing_another_view"]:
            instruction = (
                "NEXT: Revisit floor tags "
                f"{progress['ground_tags_needing_another_view']} from another angle"
            )
        else:
            instruction = "All positions recorded. Hold still while the survey finishes."
    in_view = ", ".join(str(item.tag_id) for item in detections) or "none"
    lines = [
        title,
        counts + f"  |  In view: {in_view}",
        instruction,
    ]
    for index, line in enumerate(lines):
        cv2.putText(
            output,
            str(line)[:120],
            (14, 29 + index * 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (240, 240, 240),
            2 if index == 0 else 1,
            cv2.LINE_AA,
        )
    _draw_scan_dashboard(
        output,
        image_width,
        progress=progress,
        records=records,
        camera_path=camera_path,
        world_from_camera=world_from_camera,
        message=message,
    )
    return output


def _draw_scan_dashboard(
    output: np.ndarray,
    left: int,
    *,
    progress: dict[str, Any],
    records: Sequence[dict[str, Any]],
    camera_path: Sequence[np.ndarray],
    world_from_camera: RigidTransform | None,
    message: str,
) -> None:
    """Draw a compact isometric map and a physical-position checklist."""
    height, width = output.shape[:2]
    right = width
    cv2.rectangle(output, (left, 0), (right, height), (15, 18, 23), -1)
    cv2.putText(
        output, "LIVE 3D SURVEY MAP", (left + 18, 29),
        cv2.FONT_HERSHEY_SIMPLEX, 0.70, (245, 245, 245), 2, cv2.LINE_AA,
    )
    references = [
        item for item in progress["robot_positions"] if item["identity_reference"]
    ]
    if references:
        reference = references[0]
        reference_state = (
            "LOCKED" if reference["state"] == "measured" else "NEEDS VIEW"
        )
        cv2.putText(
            output,
            f"LEG NUMBER REFERENCE: {reference['position']} = tag {reference['declared_tag_id']}  {reference_state}",
            (left + 18, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
            (80, 220, 125) if reference_state == "LOCKED" else (40, 210, 255),
            1, cv2.LINE_AA,
        )

    map_left, map_top = left + 16, 66
    map_right, map_bottom = right - 16, min(432, height - 250)
    cv2.rectangle(
        output, (map_left, map_top), (map_right, map_bottom),
        (25, 30, 38), -1,
    )
    usable_records = [
        item for item in records if int(item.get("used_observations", 0)) > 0
    ]
    points = [
        np.asarray(item["world_from_tag"]["translation_m"], dtype=float)
        for item in usable_records
    ]
    points.extend(np.asarray(item, dtype=float) for item in camera_path[-200:])
    if world_from_camera is not None:
        points.append(world_from_camera.translation_m)
    for item in progress["robot_positions"]:
        if item.get("expected_world_position_m") is not None:
            points.append(np.asarray(item["expected_world_position_m"], dtype=float))
    if not points:
        cv2.putText(
            output, "Waiting for mapped tags...", (map_left + 20, map_top + 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 180, 195), 1, cv2.LINE_AA,
        )
    else:
        point_array = np.stack(points)
        low = np.min(point_array[:, :2], axis=0) - 0.08
        high = np.max(point_array[:, :2], axis=0) + 0.08

        def raw_project(point: np.ndarray) -> np.ndarray:
            x, y, z = point
            return np.asarray([x - y, 0.48 * (x + y) - 0.90 * z])

        bounds = np.stack([
            raw_project(np.asarray([x, y, z]))
            for x in (low[0], high[0])
            for y in (low[1], high[1])
            for z in (0.0, max(0.35, float(np.max(point_array[:, 2]))))
        ])
        raw_low = np.min(bounds, axis=0)
        raw_high = np.max(bounds, axis=0)
        raw_size = np.maximum(raw_high - raw_low, 1e-6)
        scale = min(
            (map_right - map_left - 36) / raw_size[0],
            (map_bottom - map_top - 36) / raw_size[1],
        )

        def project(point: np.ndarray) -> tuple[int, int]:
            raw = raw_project(point)
            px = map_left + 18 + (raw[0] - raw_low[0]) * scale
            py = map_bottom - 18 - (raw[1] - raw_low[1]) * scale
            return int(round(px)), int(round(py))

        step = 0.10
        grid_x = np.arange(math.floor(low[0] / step) * step, high[0] + step, step)
        grid_y = np.arange(math.floor(low[1] / step) * step, high[1] + step, step)
        for x in grid_x:
            cv2.line(
                output, project(np.asarray([x, low[1], 0.0])),
                project(np.asarray([x, high[1], 0.0])),
                (45, 52, 62), 1, cv2.LINE_AA,
            )
        for y in grid_y:
            cv2.line(
                output, project(np.asarray([low[0], y, 0.0])),
                project(np.asarray([high[0], y, 0.0])),
                (45, 52, 62), 1, cv2.LINE_AA,
            )

        by_id = {int(item["tag_id"]): item for item in usable_records}
        position_by_tag = {
            int(item["tag_id"]): item
            for item in progress["robot_positions"] if item.get("tag_id") is not None
        }
        frame_points: dict[str, np.ndarray] = {}
        for position in progress["robot_positions"]:
            tag_id = position.get("tag_id")
            record = None if tag_id is None else by_id.get(int(tag_id))
            point = (
                np.asarray(record["world_from_tag"]["translation_m"], dtype=float)
                if record is not None
                else None if position.get("expected_world_position_m") is None
                else np.asarray(position["expected_world_position_m"], dtype=float)
            )
            if point is not None:
                frame_points[str(position["frame"])] = point
        body_point = frame_points.get("body")
        for leg in range(6):
            hip = frame_points.get(f"L{leg}_coxa")
            knee = frame_points.get(f"L{leg}_femur")
            if body_point is not None and hip is not None:
                cv2.line(output, project(body_point), project(hip), (80, 88, 102), 2, cv2.LINE_AA)
            if hip is not None and knee is not None:
                cv2.line(output, project(hip), project(knee), (80, 88, 102), 2, cv2.LINE_AA)

        if len(camera_path) >= 2:
            path_points = np.asarray([project(np.asarray(item)) for item in camera_path[-200:]])
            cv2.polylines(output, [path_points], False, (210, 95, 230), 2, cv2.LINE_AA)
        if world_from_camera is not None:
            camera_point = project(world_from_camera.translation_m)
            cv2.drawMarker(
                output, camera_point, (235, 120, 245), cv2.MARKER_TRIANGLE_UP,
                14, 2, cv2.LINE_AA,
            )

        for record in usable_records:
            tag_id = int(record["tag_id"])
            point = np.asarray(record["world_from_tag"]["translation_m"], dtype=float)
            center = project(point)
            position = position_by_tag.get(tag_id)
            if position is not None:
                color = (
                    (70, 220, 110) if position["state"] == "measured"
                    else (40, 210, 255)
                )
            elif record["role"] in ("ground", "calibration_anchor"):
                color = (
                    (255, 190, 45) if record.get("stable")
                    else (40, 210, 255)
                )
            else:
                color = (0, 150, 255)
            radius = 6 if record.get("stable") else 4
            cv2.circle(output, center, radius, color, -1, cv2.LINE_AA)
            important = position is not None or record["role"] in (
                "ground", "calibration_anchor"
            )
            if important:
                tag_y = np.asarray(
                    record.get("tag_y_world", [0.0, 0.0, 0.0]), dtype=float
                )
                if np.linalg.norm(tag_y) > 0.0:
                    arrow = project(point + 0.045 * tag_y)
                    cv2.arrowedLine(
                        output, center, arrow, color, 1, cv2.LINE_AA, tipLength=0.35
                    )
                cv2.putText(
                    output, str(tag_id), (center[0] + 7, center[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA,
                )
        for position in progress["robot_positions"]:
            if position["state"] == "measured" or position.get("expected_world_position_m") is None:
                continue
            center = project(np.asarray(position["expected_world_position_m"], dtype=float))
            cv2.drawMarker(output, center, (40, 210, 255), cv2.MARKER_TILTED_CROSS, 12, 2, cv2.LINE_AA)

    checklist_top = map_bottom + 26
    cv2.putText(
        output, "ROBOT POSITIONS", (left + 18, checklist_top),
        cv2.FONT_HERSHEY_SIMPLEX, 0.53, (225, 230, 238), 1, cv2.LINE_AA,
    )
    positions = progress["robot_positions"]
    rows_per_column = 7
    for index, item in enumerate(positions):
        column = index // rows_per_column
        row = index % rows_per_column
        x = left + 18 + column * 290
        y = checklist_top + 24 + row * 21
        state = item["state"]
        symbol = "OK" if state == "measured" else "VIEW" if state == "seen_needs_another_view" else "FIND"
        color = (70, 220, 110) if state == "measured" else (40, 210, 255)
        tag = "--" if item.get("tag_id") is None else str(item["tag_id"])
        replacement = " NEW" if item.get("replacement") else ""
        cv2.putText(
            output, f"{symbol:4} {item['position']}: tag {tag}{replacement}", (x, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.39, color, 1, cv2.LINE_AA,
        )
    floor_y = min(height - 52, checklist_top + 180)
    floor_text = "   ".join(
        f"{item['tag_id']}:{'OK' if item['state'] == 'measured' else 'VIEW' if item['state'] == 'seen_needs_another_view' else 'FIND'}"
        for item in progress["ground_tag_status"]
    )
    cv2.putText(
        output, "FLOOR  " + floor_text, (left + 18, floor_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 205, 225), 1, cv2.LINE_AA,
    )
    cv2.putText(
        output, ("TRACKING: " + message)[:92], (left + 18, min(height - 20, floor_y + 27)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
        (60, 100, 255) if "warning" in message.lower() or "rejected" in message.lower() else (150, 165, 185),
        1, cv2.LINE_AA,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tracker_config", type=Path)
    parser.add_argument("--board", type=Path, required=True)
    parser.add_argument(
        "--robot-layout",
        type=Path,
        help="full photographed robot-tag inventory, including vertical angle tags",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--npz-dir", type=Path)
    source.add_argument("--record3d-device", type=int, default=0)
    source.add_argument(
        "--wifi-frame-dir",
        type=Path,
        help="consume Record3D WebRTC RGB-D frames relayed by the calibration webpage",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updated-config", type=Path)
    parser.add_argument("--preview-output", type=Path)
    parser.add_argument(
        "--camera-preview-output",
        type=Path,
        help="periodically write the clean labelled camera frame for a web UI",
    )
    parser.add_argument(
        "--progress-output",
        type=Path,
        help="periodically write an atomic machine-readable progress snapshot",
    )
    parser.add_argument(
        "--resume-progress",
        type=Path,
        help="restore stable tag landmarks from an earlier progress snapshot",
    )
    parser.add_argument("--anchor-frames", type=int, default=8)
    parser.add_argument("--min-observations", type=int, default=8)
    parser.add_argument(
        "--expected-floor-ids",
        type=_parse_ids,
        help="comma-separated IDs; defaults to floor IDs in the input config",
    )
    parser.add_argument(
        "--survey-marker-size-mm",
        type=float,
        help="size for unknown tags; defaults to tracker marker_size_m",
    )
    parser.add_argument(
        "--joint-angles-json",
        type=Path,
        help="known stationary pose as a joint-name-to-degrees object; default all zero",
    )
    parser.add_argument(
        "--body-anchor-tag-id",
        type=int,
        help=(
            "trusted chassis tag whose existing mount defines the body frame; "
            "defaults to the lowest stable configured body tag"
        ),
    )
    parser.add_argument(
        "--leg-zero-anchor-tag-id",
        type=int,
        help=(
            "tag physically on the L0 hip; defaults to the configured L0 hip "
            "tag. Specify a new ID explicitly only after replacing that tag"
        ),
    )
    parser.add_argument("--settle-frames", type=int, default=12)
    parser.add_argument("--max-seconds", type=float, default=300.0)
    parser.add_argument("--no-preview", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.anchor_frames < 4:
        raise SystemExit("--anchor-frames must be at least 4")
    if args.min_observations < 2:
        raise SystemExit("--min-observations must be at least 2")
    if args.settle_frames < 1 or args.max_seconds <= 0.0:
        raise SystemExit("settle frames and max seconds must be positive")

    tracker_config = _load_json(args.tracker_config)
    if args.robot_layout is not None:
        tracker_config = merge_robot_layout_into_config(
            tracker_config, _load_json(args.robot_layout)
        )
    board_manifest = _load_json(args.board)
    anchors, anchor_marker_size_m = _calibration_target(
        tracker_config, board_manifest
    )
    default_marker_size_m = float(tracker_config.get("marker_size_m", 0.027))
    unknown_marker_size_m = (
        default_marker_size_m
        if args.survey_marker_size_mm is None
        else float(args.survey_marker_size_mm) / 1000.0
    )
    if not math.isfinite(unknown_marker_size_m) or unknown_marker_size_m <= 0.0:
        raise SystemExit("--survey-marker-size-mm must be positive")
    robot_tags = {
        int(raw_id): spec
        for raw_id, spec in tracker_config.get("robot_pose", {}).get("tags", {}).items()
    }
    configured_l0_ids = sorted(
        tag_id for tag_id, spec in robot_tags.items()
        if str(spec.get("frame")) == "L0_coxa"
    )
    configured_l0_id = configured_l0_ids[0] if configured_l0_ids else None
    leg_zero_anchor_id = (
        configured_l0_id
        if args.leg_zero_anchor_tag_id is None
        else int(args.leg_zero_anchor_tag_id)
    )
    position_tag_overrides = (
        {} if leg_zero_anchor_id is None
        else {"L0_coxa": leg_zero_anchor_id}
    )
    configured_floor_ids = {
        int(raw_id) for raw_id in tracker_config.get("floor_tags", {})
    }
    expected_ground_ids = (
        configured_floor_ids
        if args.expected_floor_ids is None
        else set(args.expected_floor_ids)
    )
    marker_sizes = _tag_size_map(
        tracker_config, board_manifest, default_marker_size_m
    )
    survey = TagSurveyAccumulator(
        robot_tags=robot_tags,
        expected_ground_ids=sorted(expected_ground_ids),
        anchor_ids=sorted(anchors),
        marker_size_m=unknown_marker_size_m,
        marker_sizes_m=marker_sizes,
        position_tag_overrides=position_tag_overrides,
        geometry=tracker_config.get("robot_pose", {}).get("geometry"),
        body_anchor_tag_id=args.body_anchor_tag_id,
        reference_floor_tags=anchors if len(anchors) >= 2 else {},
        options=TagSurveyOptions(
            min_observations=args.min_observations,
            max_reprojection_rms_px=(
                1.5 if args.robot_layout is not None else 2.5
            ),
            max_translation_spread_m=(
                0.010 if args.robot_layout is not None else 0.020
            ),
            max_rotation_spread_deg=(
                2.0 if args.robot_layout is not None else 5.0
            ),
            freeze_stable_tags=False,
        ),
    )
    resume_snapshot: dict[str, Any] = {}
    if args.resume_progress is not None and args.resume_progress.is_file():
        resume_snapshot = _load_json(args.resume_progress)
    restored_ids = survey.restore_stable_records(
        resume_snapshot.get("records", []),
        frames=int(resume_snapshot.get("frame_sequence", 0)),
    )
    resumed = bool(restored_ids)
    small_single_tag_anchor = (
        len(anchors) == 1 and anchor_marker_size_m <= 0.035
    )
    alignment = HandheldWorldAlignment(
        min_observations=args.anchor_frames,
        max_translation_spread_m=(
            0.050 if small_single_tag_anchor else 0.015
        ),
        max_rotation_spread_deg=(
            5.0 if small_single_tag_anchor else 1.5
        ),
    )
    rgbd_options = RGBDCalibrationOptions(
        min_confidence=1,
        min_depth_samples=(24 if small_single_tag_anchor else 40),
    )
    previous_anchor_pose: RigidTransform | None = None
    reader: Record3DReader | None = None
    start = time.monotonic()
    last_status_time = -float("inf")
    completed_frames = 0
    last_preview: np.ndarray | None = None
    camera_path = [
        np.asarray(item, dtype=float).reshape(3)
        for item in resume_snapshot.get("camera_path_m", [])[-240:]
    ]
    phase = "anchor"
    last_message = (
        f"Saved {len(restored_ids)} stable tags; re-lock the floor origin"
        if resumed else "Find the calibration board"
    )
    last_camera_matrix: np.ndarray | None = None
    last_quality: dict[str, Any] | None = None
    camera_speed_m_s: float | None = None
    previous_camera_sample: tuple[np.ndarray, float] | None = None
    connection_mode = "wifi" if args.wifi_frame_dir is not None else "usb"
    initial_progress = survey.progress()
    initial_guidance = _guidance(
        phase,
        initial_progress,
        sorted(anchors),
        min_observations=args.min_observations,
        resumed=resumed,
    )
    _write_progress(args.progress_output, {
        "calibration_model_version": 2,
        "status": "connecting",
        "phase": "connect",
        "message": (
            "Waiting for Wi-Fi frames from this webpage."
            if connection_mode == "wifi"
            else "Waiting for the Record3D USB stream from the iPhone."
        ),
        "instruction": (
            "Connect the phone's Record3D Wi-Fi stream in this webpage."
            if connection_mode == "wifi"
            else "In Record3D, select USB and tap the red stream button."
        ),
        "guidance": initial_guidance,
        "connection_mode": connection_mode,
        "resumed": resumed,
        "restored_tag_ids": restored_ids,
        "anchor_ids": sorted(anchors),
        "alignment_count": 0,
        "anchor_frames": args.anchor_frames,
        "detected_tag_ids": [],
        "frame_sequence": int(resume_snapshot.get("frame_sequence", 0)),
        "elapsed_s": 0.0,
        "progress": initial_progress,
        "records": survey.tag_records(),
        "camera_path_m": [item.tolist() for item in camera_path],
        "camera_position_m": None,
    })
    connection_error: str | None = None
    try:
        if args.wifi_frame_dir is not None:
            frames: Iterator[RGBDFrame] = _wifi_frames(args.wifi_frame_dir)
        elif args.npz_dir is not None:
            frames = _npz_frames(args.npz_dir)
        else:
            reader = Record3DReader(args.record3d_device)

            def live_frames() -> Iterator[RGBDFrame]:
                assert reader is not None
                while True:
                    yield reader.next_frame()

            frames = live_frames()
        for frame_index, frame in enumerate(frames):
            current_world_from_camera: RigidTransform | None = None
            anchor_reprojection_rms_px: float | None = None
            depth_plane_rms_mm: float | None = None
            if frame.arkit_world_from_opengl_camera is None:
                raise RuntimeError(
                    f"{frame.source_label} has no ARKit camera pose; handheld "
                    "survey requires camera_pose_xyzw_xyz"
                )
            last_camera_matrix = frame.camera_matrix
            detections = detect_tag_corners(frame.rgb_bgr)
            anchor_detections = [
                item for item in detections if item.tag_id in anchors
            ]
            direct_anchor_pose = None
            if anchor_detections:
                try:
                    height, width = frame.rgb_bgr.shape[:2]
                    preferred_floor_normal_camera = None
                    if small_single_tag_anchor:
                        arkit_from_camera = arkit_world_from_opencv_camera(
                            frame.arkit_world_from_opengl_camera
                        )
                        preferred_floor_normal_camera = (
                            arkit_from_camera.rotation.inv().apply(
                                [0.0, 1.0, 0.0]
                            )
                        )
                    direct_anchor_pose = refine_world_reference_with_depth(
                        anchor_detections,
                        anchors,
                        frame.camera_matrix,
                        np.zeros(5),
                        frame.depth_m,
                        image_size_px=(width, height),
                        confidence=frame.confidence,
                        marker_size_m=anchor_marker_size_m,
                        previous_world_from_camera=previous_anchor_pose,
                        preferred_floor_normal_camera=(
                            preferred_floor_normal_camera
                        ),
                        options=rgbd_options,
                    )
                    previous_anchor_pose = direct_anchor_pose.world_from_camera
                    anchor_reprojection_rms_px = float(
                        direct_anchor_pose.reprojection_rms_px
                    )
                    depth_plane_rms_mm = float(
                        direct_anchor_pose.depth_plane_rms_m * 1000.0
                    )
                    survey.observe_floor_reference(
                        direct_anchor_pose.floor_tag_ids,
                        reprojection_rms_px=anchor_reprojection_rms_px,
                        depth_plane_rms_mm=depth_plane_rms_mm,
                    )
                    if phase == "anchor":
                        alignment.add(
                            direct_anchor_pose.world_from_camera,
                            frame.arkit_world_from_opengl_camera,
                        )
                    last_message = (
                        f"Board residual {direct_anchor_pose.reprojection_rms_px:.2f}px, "
                        f"depth {direct_anchor_pose.depth_plane_rms_m * 1000.0:.1f}mm"
                    )
                except (RGBDCalibrationError, ValueError, cv2.error) as error:
                    last_message = f"Board rejected: {error}"

            alignment_consensus = alignment.consensus()
            if alignment_consensus is not None and alignment_consensus.stable:
                phase = "survey"
                predicted_world_from_camera = alignment.world_from_camera(
                    frame.arkit_world_from_opengl_camera
                )
                world_from_camera = predicted_world_from_camera
                if direct_anchor_pose is not None:
                    world_from_camera = direct_anchor_pose.world_from_camera
                    direct_correction_m = float(np.linalg.norm(
                        world_from_camera.translation_m
                        - predicted_world_from_camera.translation_m
                    ))
                    last_message = (
                        f"Mapped-floor lock {len(anchor_detections)} tag(s); "
                        f"corrected {direct_correction_m * 1000.0:.0f}mm"
                    )
                else:
                    landmark_reference = survey.estimate_world_from_camera(
                        detections,
                        predicted_world_from_camera,
                        frame.camera_matrix,
                        np.zeros(5),
                    )
                    if (
                        landmark_reference is not None
                        and landmark_reference.stable
                        and landmark_reference.used_count >= 2
                    ):
                        landmark_drift_m = float(np.linalg.norm(
                            landmark_reference.transform.translation_m
                            - predicted_world_from_camera.translation_m
                        ))
                        landmark_drift_deg = _rotation_error_deg(
                            landmark_reference.transform,
                            predicted_world_from_camera,
                        )
                        if landmark_drift_m <= 0.080 and landmark_drift_deg <= 8.0:
                            world_from_camera = landmark_reference.transform
                            last_message = (
                                f"Landmark lock {landmark_reference.used_count} tags; "
                                f"corrected {landmark_drift_m * 1000.0:.0f}mm"
                            )
                current_world_from_camera = world_from_camera
                sample_time = time.monotonic()
                if previous_camera_sample is not None:
                    distance = float(np.linalg.norm(
                        world_from_camera.translation_m - previous_camera_sample[0]
                    ))
                    delta_time = sample_time - previous_camera_sample[1]
                    if delta_time > 1e-3:
                        instant_speed = distance / delta_time
                        camera_speed_m_s = (
                            instant_speed if camera_speed_m_s is None
                            else 0.72 * camera_speed_m_s + 0.28 * instant_speed
                        )
                previous_camera_sample = (
                    world_from_camera.translation_m.copy(), sample_time
                )
                survey.observe_frame(
                    detections,
                    world_from_camera,
                    frame.camera_matrix,
                    np.zeros(5),
                )

                if (
                    not camera_path
                    or np.linalg.norm(
                        world_from_camera.translation_m - camera_path[-1]
                    ) >= 0.015
                ):
                    camera_path.append(world_from_camera.translation_m.copy())

            progress = survey.progress()
            stable_ids = set(progress["stable_tag_ids"])
            if phase == "survey" and progress["complete"]:
                completed_frames += 1
                last_message = (
                    f"All expected tags found; holding {completed_frames}/"
                    f"{args.settle_frames} frames"
                )
            else:
                completed_frames = 0
            last_preview = _annotate(
                frame.rgb_bgr,
                detections,
                phase=phase,
                anchor_ids=set(anchors),
                robot_ids=set(robot_tags),
                ground_ids=set(expected_ground_ids),
                stable_ids=stable_ids,
                progress=progress,
                records=survey.tag_records(),
                camera_path=camera_path,
                world_from_camera=current_world_from_camera,
                alignment_count=alignment.observation_count,
                anchor_frames=args.anchor_frames,
                message=last_message,
            )
            preview = not args.no_preview and args.npz_dir is None
            if preview:
                cv2.imshow("Hexapod zero-pose tag survey", last_preview)
                if cv2.waitKey(1) & 0xFF in (27, ord("q"), ord("Q")):
                    break
            now = time.monotonic()
            if now - last_status_time >= 0.75:
                measured_robot = sum(
                    item["state"] == "measured"
                    for item in progress["robot_positions"]
                )
                measured_floor = sum(
                    item["state"] == "measured"
                    for item in progress["ground_tag_status"]
                )
                needs_view = [
                    f"{item['position']}(tag {item['tag_id']})"
                    for item in progress["robot_positions"]
                    if item["state"] == "seen_needs_another_view"
                ] + [
                    f"floor tag {tag_id}"
                    for tag_id in progress["ground_tags_needing_another_view"]
                ]
                find = progress["unseen_robot_positions"] + [
                    f"floor tag {tag_id}"
                    for tag_id in progress["unseen_ground_tag_ids"]
                ]
                if phase == "anchor":
                    print(
                        f"ORIGIN LOCK | point at floor tag(s) {sorted(anchors)} "
                        f"| good frames {alignment.observation_count}/{args.anchor_frames} "
                        f"| {last_message}",
                        flush=True,
                    )
                else:
                    view_text = ", ".join(needs_view[:4]) or "none"
                    if len(needs_view) > 4:
                        view_text += f" +{len(needs_view) - 4} more"
                    find_text = ", ".join(find[:4]) or "none"
                    if len(find) > 4:
                        find_text += f" +{len(find) - 4} more"
                    print(
                        f"MAPPING | robot {measured_robot}/{len(progress['robot_positions'])} "
                        f"| floor {measured_floor}/{len(progress['ground_tag_status'])} "
                        f"| show again: {view_text} | find: {find_text} "
                        f"| {last_message}",
                        flush=True,
                    )
                if args.preview_output is not None:
                    args.preview_output.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(args.preview_output), last_preview)
                if args.camera_preview_output is not None:
                    args.camera_preview_output.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(
                        str(args.camera_preview_output),
                        _camera_preview(frame.rgb_bgr, detections),
                    )
                guidance = _guidance(
                    phase,
                    progress,
                    sorted(anchors),
                    min_observations=args.min_observations,
                    resumed=resumed,
                )
                records = survey.tag_records()
                quality = _quality_feedback(
                    guidance,
                    records,
                    {item.tag_id for item in detections},
                    camera_speed_m_s=camera_speed_m_s,
                    tracking_message=last_message,
                    anchor_reprojection_rms_px=anchor_reprojection_rms_px,
                    depth_plane_rms_mm=depth_plane_rms_mm,
                )
                last_quality = quality
                _write_progress(args.progress_output, {
                    "calibration_model_version": 2,
                    "status": "locking_origin" if phase == "anchor" else (
                        "finishing" if progress["complete"] else "scanning"
                    ),
                    "phase": phase,
                    "message": last_message,
                    "instruction": guidance["headline"],
                    "guidance": guidance,
                    "quality": quality,
                    "connection_mode": connection_mode,
                    "resumed": resumed,
                    "restored_tag_ids": restored_ids,
                    "anchor_ids": sorted(anchors),
                    "alignment_count": alignment.observation_count,
                    "anchor_frames": args.anchor_frames,
                    "detected_tag_ids": sorted(item.tag_id for item in detections),
                    "frame_sequence": int(resume_snapshot.get("frame_sequence", 0)) + frame_index + 1,
                    "elapsed_s": round(now - start, 2),
                    "progress": progress,
                    "records": records,
                    "camera_path_m": [item.tolist() for item in camera_path[-240:]],
                    "camera_position_m": (
                        None if current_world_from_camera is None
                        else current_world_from_camera.translation_m.tolist()
                    ),
                })
                last_status_time = now
            if completed_frames >= args.settle_frames:
                break
            if now - start >= args.max_seconds:
                last_message = "time limit reached; saving partial survey"
                break
    except KeyboardInterrupt:
        last_message = "operator stopped; saving partial survey"
    except (OSError, RuntimeError, TimeoutError) as error:
        connection_error = str(error)
        last_message = connection_error
    finally:
        if reader is not None:
            reader.close()
        if not args.no_preview and args.npz_dir is None:
            cv2.destroyAllWindows()

    survey_payload = survey.summary()
    joint_angles = None
    if args.joint_angles_json is not None:
        joint_angles = _load_json(args.joint_angles_json)
    learned_mounts, geometry_report = learn_zero_pose_mounts(
        tracker_config,
        survey_payload,
        joint_angles_deg=joint_angles,
        body_anchor_tag_id=args.body_anchor_tag_id,
    )
    payload = {
        "schema_version": 2,
        "calibration_model_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_tracker_config": str(args.tracker_config),
        "source_board_manifest": str(args.board),
        "anchor_mode": (
            "small_single_tag_bootstrap"
            if small_single_tag_anchor
            else "multi_tag_board"
        ),
        "motor_commands_sent": False,
        "leg_zero_reference": {
            "frame": "L0_coxa",
            "configured_tag_id": configured_l0_id,
            "declared_tag_id": leg_zero_anchor_id,
            "explicitly_overridden": args.leg_zero_anchor_tag_id is not None,
        },
        "alignment": _alignment_record(alignment),
        "survey": survey_payload,
        "mount_learning": geometry_report,
        "last_camera_matrix": (
            None if last_camera_matrix is None else last_camera_matrix.tolist()
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.updated_config is not None:
        updated = apply_survey_to_config(
            tracker_config, survey_payload, learned_mounts
        )
        args.updated_config.parent.mkdir(parents=True, exist_ok=True)
        args.updated_config.write_text(
            json.dumps(updated, indent=2) + "\n", encoding="utf-8"
        )
    if args.preview_output is not None and last_preview is not None:
        args.preview_output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.preview_output), last_preview):
            raise OSError(f"could not write {args.preview_output}")
    final_ok = bool(survey_payload["complete"]) and (
        not robot_tags or bool(geometry_report.get("ok"))
    )
    final_progress = survey.progress()
    final_guidance = _guidance(
        "survey",
        final_progress,
        sorted(anchors),
        min_observations=args.min_observations,
        resumed=resumed,
    )
    _write_progress(args.progress_output, {
        "calibration_model_version": 2,
        "status": (
            "connection_lost" if connection_error is not None
            else "complete" if final_ok else "incomplete"
        ),
        "phase": "connect" if connection_error is not None else "review",
        "message": (
            f"Connection lost. {survey_payload['stable_tag_count']} stable tags are saved."
            if connection_error is not None
            else "Survey complete. Review the geometry and tag assignments."
            if final_ok
            else "Partial survey saved. Continue this calibration when ready."
        ),
        "instruction": (
            "Reconnect the phone, then continue this calibration."
            if connection_error is not None
            else "Review the 3D survey map."
            if final_ok
            else final_guidance["headline"]
        ),
        "guidance": final_guidance,
        "quality": last_quality,
        "connection_mode": connection_mode,
        "resumed": resumed,
        "restored_tag_ids": restored_ids,
        "anchor_ids": sorted(anchors),
        "alignment_count": alignment.observation_count,
        "anchor_frames": args.anchor_frames,
        "detected_tag_ids": sorted(
            int(item["tag_id"]) for item in survey_payload["tags"]
        ),
        "frame_sequence": survey_payload.get("frames", 0),
        "elapsed_s": round(time.monotonic() - start, 2),
        "progress": final_progress,
        "records": survey_payload["tags"],
        "camera_path_m": [item.tolist() for item in camera_path[-240:]],
        "camera_position_m": None,
        "result_path": str(args.output),
        "mount_learning": geometry_report,
        "connection_error": connection_error,
    })
    print(f"wrote survey: {args.output}")
    if args.updated_config is not None:
        print(f"wrote updated config: {args.updated_config}")
    if connection_error is not None:
        print(f"connection lost; stable observations saved: {connection_error}")
        return 4
    if not survey_payload["complete"]:
        print(
            "survey incomplete; robot positions still needing work: "
            f"{survey_payload['missing_robot_positions'] or 'none'}; floor tags: "
            f"{survey_payload['missing_ground_tag_ids'] or 'none'}"
        )
        return 2
    if robot_tags and not geometry_report.get("ok"):
        print(
            "tag survey completed, but robot mount learning failed: "
            f"{geometry_report.get('error', 'unknown error')}"
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
