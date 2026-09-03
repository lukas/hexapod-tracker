#!/usr/bin/env python3
"""Measure floor-referenced gait displacement from offline AprilTag poses."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _best_floor_homography(
    detections: dict[int, dict[str, Any]],
    floor_corners: dict[int, np.ndarray],
    *,
    maximum_rms_m: float = 0.008,
) -> tuple[np.ndarray, list[int], float] | None:
    """Fit the largest self-consistent subset of visible mapped floor tags.

    The garage currently contains two physical tag-13 prints. The pose tracker
    emits one decode for that ID, which may be the unmapped spare. Trying all
    3-tag then 2-tag subsets prevents that spare from changing the world frame
    while retaining the two unambiguous references.
    """
    visible = sorted(tag_id for tag_id in floor_corners if tag_id in detections)
    for subset_size in range(len(visible), 1, -1):
        acceptable: list[tuple[np.ndarray, list[int], float]] = []
        for subset in itertools.combinations(visible, subset_size):
            image_points = np.concatenate([
                np.asarray(detections[tag_id]["corners_px"], dtype=np.float32)
                for tag_id in subset
            ])
            world_points = np.concatenate([
                floor_corners[tag_id].astype(np.float32) for tag_id in subset
            ])
            homography, _ = cv2.findHomography(image_points, world_points, 0)
            if homography is None:
                continue
            projected = cv2.perspectiveTransform(
                image_points.reshape(-1, 1, 2), homography
            ).reshape(-1, 2)
            rms_m = float(np.sqrt(np.mean(np.sum(
                (projected - world_points) ** 2, axis=1
            ))))
            if rms_m <= maximum_rms_m:
                acceptable.append((homography, list(subset), rms_m))
        if acceptable:
            return min(acceptable, key=lambda item: item[2])
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pose-jsonl", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    telemetry = _csv(args.run_dir / "telemetry.csv")
    telemetry_unix = np.asarray([
        float(row["receipt_unix_s"]) for row in telemetry
    ])
    timestamps = _csv(args.run_dir / "iphone_raw_timestamps.csv")
    source_frame = np.asarray([float(row["frame"]) for row in timestamps])
    source_unix = np.asarray([float(row["unix_s"]) for row in timestamps])
    capture = cv2.VideoCapture(str(args.run_dir / "iphone_raw.mp4"))
    raw_fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    capture.release()
    config = json.loads(args.config.read_text())
    half = float(config["marker_size_m"]) / 2.0
    local_corners = np.asarray([
        [-half, +half, 0.0], [+half, +half, 0.0],
        [+half, -half, 0.0], [-half, -half, 0.0],
    ])
    floor_corners: dict[int, np.ndarray] = {}
    for raw_id, item in config["floor_tags"].items():
        transform = item["world_from_tag"]
        rotation = Rotation.from_euler(
            "xyz", transform.get("euler_xyz_deg", [0.0, 0.0, 0.0]),
            degrees=True,
        )
        translation = np.asarray(transform["translation_m"], dtype=float)
        floor_corners[int(raw_id)] = (
            rotation.apply(local_corners) + translation
        )[:, :2]

    groups: dict[str, list[list[float]]] = defaultdict(list)
    with args.pose_jsonl.open() as stream:
        for line in stream:
            pose = json.loads(line)
            detections = {
                int(item["tag_id"]): item for item in pose.get("detections", [])
                if item.get("source") == "detected"
            }
            if 0 not in detections:
                continue
            direct_tags = sum(
                item.get("source") == "detected"
                and not str(item.get("label", "")).lower().startswith("floor")
                for item in pose.get("detections", [])
            )
            if direct_tags < 6:
                continue
            fit = _best_floor_homography(detections, floor_corners)
            if fit is None:
                continue
            homography, selected_floor, floor_rms_m = fit
            body_center = np.asarray(
                detections[0]["center_px"], dtype=np.float32
            ).reshape(1, 1, 2)
            body_floor_xy = cv2.perspectiveTransform(
                body_center, homography
            ).reshape(2)
            unix_s = float(np.interp(
                float(pose["time_s"]) * raw_fps, source_frame, source_unix
            ))
            index = int(np.argmin(np.abs(telemetry_unix - unix_s)))
            if abs(float(telemetry_unix[index]) - unix_s) > 0.6:
                continue
            phase = telemetry[index]["phase"]
            if not (phase.startswith("gait_") and phase.rsplit("_", 1)[-1]
                    in {"forward", "backward"}):
                continue
            groups[phase].append([
                unix_s, *body_floor_xy, direct_tags, len(selected_floor),
                floor_rms_m,
            ])

    phases: dict[str, dict[str, Any]] = {}
    for phase, values in sorted(groups.items()):
        array = np.asarray(values, dtype=float)
        edge = max(2, round(len(array) * 0.15))
        start = np.median(array[:edge, 1:3], axis=0)
        end = np.median(array[-edge:, 1:3], axis=0)
        start_t = float(np.median(array[:edge, 0]))
        end_t = float(np.median(array[-edge:, 0]))
        duration_s = max(0.0, end_t - start_t)
        delta = end - start
        phases[phase] = {
            "usable_vision_frames": len(array),
            "measured_duration_s": round(duration_s, 4),
            "floor_projected_body_delta_xy_m": np.round(delta, 5).tolist(),
            "horizontal_distance_m": round(float(np.linalg.norm(delta)), 5),
            "mean_direct_robot_tags": round(float(np.mean(array[:, 3])), 2),
            "mean_direct_floor_tags": round(float(np.mean(array[:, 4])), 2),
            "median_floor_homography_rms_mm": round(
                float(np.median(array[:, 5])) * 1000.0, 3
            ),
        }

    # Chassis-tag yaw still has a mount-offset uncertainty. Derive a pragmatic
    # locomotion axis from the baseline gait's forward and reverse travel so
    # all other gaits can be compared without pretending that yaw is calibrated.
    reference = None
    if "gait_0_forward" in phases and "gait_0_backward" in phases:
        forward = np.asarray(
            phases["gait_0_forward"]["floor_projected_body_delta_xy_m"]
        )
        backward = np.asarray(
            phases["gait_0_backward"]["floor_projected_body_delta_xy_m"]
        )
        candidate = forward - backward
        norm = float(np.linalg.norm(candidate))
        if norm > 1e-6:
            reference = candidate / norm
    if reference is not None:
        for phase, record in phases.items():
            delta = np.asarray(record["floor_projected_body_delta_xy_m"])
            record["baseline_axis_progress_m"] = round(
                float(np.dot(delta, reference)), 5
            )
            record["baseline_axis_lateral_m"] = round(
                float(np.dot(delta, np.asarray([-reference[1], reference[0]]))), 5
            )
            duration_s = float(record["measured_duration_s"])
            direction_sign = -1.0 if phase.endswith("_backward") else 1.0
            record["commanded_axis_speed_mm_s"] = (
                None if duration_s <= 0.0 else round(
                    direction_sign
                    * float(record["baseline_axis_progress_m"])
                    / duration_s * 1000.0,
                    3,
                )
            )
    report = {
        "visual_knees_used": False,
        "measurement": "planar homography from 2+ mapped floor tags to chassis tag center",
        "scale_caution": (
            "relative motion is robust to phone movement and avoids planar-PnP "
            "branch flips; the chassis tag is above the floor, so oblique-view "
            "parallax keeps absolute distance approximate"
        ),
        "time_alignment": "raw frame index -> iphone_raw_timestamps.csv",
        "baseline_forward_axis_world_xy": (
            None if reference is None else np.round(reference, 6).tolist()
        ),
        "phases": phases,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
