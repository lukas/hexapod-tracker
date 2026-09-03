#!/usr/bin/env python3
"""Render an LLM-friendly gait video with synchronized telemetry overlays."""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .joint_contract import (
    FRAME_ROBOT_ABS,
    JOINT_CONTRACT,
    require_robot_abs_joint_frame,
)


GAITS = {
    0: "TRIPOD DRAG",
    1: "NO-SLIP TRIPOD",
    2: "NO-SLIP RIPPLE",
    3: "NO-SLIP WAVE",
    4: "SE2 TETRAPOD",
    5: "SE2 WAVE",
    6: "SE2 CPG ROBUST120",
    7: "NO-SLIP CLAMP-FIT",
    8: "MIDDLE-TUCK QUAD",
}


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _nearest_index(values: list[float], query: float) -> int:
    index = bisect.bisect_left(values, query)
    if index <= 0:
        return 0
    if index >= len(values):
        return len(values) - 1
    return index if values[index] - query < query - values[index - 1] else index - 1


def _load_pose(path: Path | None) -> tuple[list[float], list[dict[str, Any]]]:
    if path is None or not path.exists():
        return [], []
    records = []
    with path.open() as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))
    return [float(item["time_s"]) for item in records], records


def _debounced_temperatures(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    previous = [None] * 18
    last_good = [None] * 18
    for row in rows:
        raw = json.loads(row["joint_temperatures_c"])
        filtered: list[float | None] = []
        glitches: list[dict[str, Any]] = []
        for joint, value in enumerate(raw):
            value = None if value is None else float(value)
            confirmed = (
                value is not None and value >= 55.0
                and previous[joint] is not None and previous[joint] >= 55.0
            )
            if value is not None and value >= 55.0 and not confirmed:
                glitches.append({"joint": joint, "raw_c": value})
                filtered.append(last_good[joint])
            else:
                filtered.append(value)
                if value is not None and value < 55.0:
                    last_good[joint] = value
            previous[joint] = value
        values = [value for value in filtered if value is not None]
        output.append({
            "max_c": max(values, default=None),
            "glitches": glitches,
        })
    return output


def _box(frame: np.ndarray, x: int, y: int, w: int, h: int,
         alpha: float = 0.72) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (8, 12, 12), -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, frame)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (70, 90, 85), 1)


def _text(frame: np.ndarray, text: str, x: int, y: int, *,
          scale: float = 0.56, color: tuple[int, int, int] = (235, 240, 236),
          thickness: int = 1) -> None:
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def _joint_text(value: Any) -> str:
    """Keep the overlay alive across brief missing encoder samples."""
    if value is None:
        return "   n/a "
    try:
        return f"{float(value):+7.1f}"
    except (TypeError, ValueError):
        return "   n/a "


def _phase_title(row: dict[str, str]) -> tuple[str, str]:
    phase = row["phase"]
    gait_text = "ROBOT SETUP / PARK"
    if row.get("gait", "").strip():
        gait = int(row["gait"])
        gait_text = f"GAIT {gait} - {GAITS.get(gait, 'UNKNOWN')}"
    state = row.get("direction", "").strip().upper()
    if not state:
        state = "SETTLE" if phase.endswith("_settle") else phase.replace("_", " ").upper()
    return gait_text, state


def _draw_pose(frame: np.ndarray, pose: dict[str, Any] | None,
               sx: float, sy: float) -> tuple[int, int, float | None]:
    if not pose:
        return 0, 0, None
    direct_tags = 0
    for tag in pose.get("detections", []):
        points = np.asarray(tag.get("corners_px") or [], dtype=float)
        if points.shape != (4, 2):
            continue
        points[:, 0] *= sx
        points[:, 1] *= sy
        polygon = np.rint(points).astype(np.int32)
        inferred = tag.get("source") != "detected"
        color = (90, 180, 255) if not inferred else (120, 120, 170)
        cv2.polylines(frame, [polygon], True, color, 2, cv2.LINE_AA)
        center = np.mean(points, axis=0).astype(int)
        _text(frame, f"#{tag.get('tag_id')}", int(center[0] + 4),
              int(center[1] - 5), scale=0.42, color=color, thickness=1)
        direct_tags += int(not inferred)
    direct_feet = 0
    for foot in pose.get("foot_tips", []):
        point = foot.get("point_px")
        if not point:
            continue
        x, y = round(float(point[0]) * sx), round(float(point[1]) * sy)
        direct = foot.get("source") == "color"
        color = (80, 245, 190) if direct else (190, 120, 220)
        cv2.drawMarker(frame, (x, y), color, cv2.MARKER_CROSS, 22, 2,
                       cv2.LINE_AA)
        _text(frame, f"L{foot.get('leg')}", x + 9, y - 9, scale=0.45,
              color=color, thickness=1)
        direct_feet += int(direct)
    tilt = ((pose.get("full_pose") or {}).get("walking_check") or {}).get(
        "body_tilt_deg"
    )
    return direct_tags, direct_feet, None if tilt is None else float(tilt)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--pose-jsonl", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clips-dir", type=Path)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    video = args.video or args.run_dir / "iphone_raw.mp4"
    telemetry = _csv_rows(args.run_dir / "telemetry.csv")
    timestamps = _csv_rows(args.run_dir / "iphone_raw_timestamps.csv")
    if not telemetry or not timestamps:
        raise RuntimeError("run has no telemetry or video timestamps")
    config = json.loads((args.run_dir / "config.json").read_text())
    require_robot_abs_joint_frame(
        config, source=str(args.run_dir / "config.json"))
    telemetry_times = [float(row["receipt_unix_s"]) for row in telemetry]
    source_frames = np.asarray([float(row["frame"]) for row in timestamps])
    source_unix = np.asarray([float(row["unix_s"]) for row in timestamps])
    raw_capture = cv2.VideoCapture(str(args.run_dir / "iphone_raw.mp4"))
    raw_writer_fps = float(raw_capture.get(cv2.CAP_PROP_FPS)) or 30.0
    raw_capture.release()
    filtered_temps = _debounced_temperatures(telemetry)
    pose_times, poses = _load_pose(args.pose_jsonl)

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise OSError(f"could not open {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 8.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
        "-r", f"{fps:.6f}", "-i", "-", "-an", "-c:v", "libx264",
        "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-g", str(max(1, round(fps))), str(args.output),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    frames = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            time_s = frames / fps
            # The recorder retains exact per-frame wall timestamps. Continuity
            # Camera delivered ~24 fps in this run while the legacy MP4 header
            # said 30 fps, so video PTS alone would shift gait labels by tens
            # of seconds. Map this derived video's PTS back through the raw
            # source frame index and timestamp sidecar.
            source_frame = time_s * raw_writer_fps
            unix_s = float(np.interp(source_frame, source_frames, source_unix))
            ti = _nearest_index(telemetry_times, unix_s)
            row = telemetry[ti]
            pose = None
            if pose_times:
                pi = _nearest_index(pose_times, time_s)
                if abs(pose_times[pi] - time_s) <= max(0.3, 1.5 / fps):
                    pose = poses[pi]
            pose_width, pose_height = width, height
            if pose and pose.get("image_size_px"):
                pose_width, pose_height = pose["image_size_px"]
            tags, feet, visual_tilt = _draw_pose(
                frame, pose, width / pose_width, height / pose_height
            )

            gait_title, state_title = _phase_title(row)
            _box(frame, 18, 18, min(650, width - 36), 205)
            _text(frame, gait_title, 38, 54, scale=0.82,
                  color=(80, 230, 255), thickness=2)
            _text(frame, state_title, 38, 87, scale=0.69,
                  color=(120, 255, 175), thickness=2)
            elapsed = float(row["elapsed_s"])
            _text(frame, f"run {elapsed:6.1f}s  video {time_s:6.1f}s  phase {row['phase']}",
                  38, 116, scale=0.48)
            roll = float(row.get("body_roll_deg") or row.get("roll_deg") or 0)
            pitch = float(row.get("body_pitch_deg") or row.get("pitch_deg") or 0)
            tilt = max(abs(roll), abs(pitch))
            _text(frame, f"IMU roll {roll:+5.1f}  pitch {pitch:+5.1f}  peak-axis tilt {tilt:4.1f} deg",
                  38, 145, scale=0.52)
            _text(frame, f"current max {float(row['max_joint_current_a']):.2f} A  bus-sum {float(row['bus_current_a']):.2f} A  voltage {float(row['min_voltage_v']):.1f} V",
                  38, 173, scale=0.50)
            temp = filtered_temps[ti]
            temp_text = "temperature unavailable" if temp["max_c"] is None \
                else f"temperature {temp['max_c']:.0f} C (debounced)"
            if temp["glitches"]:
                detail = ", ".join(
                    f"J{x['joint']} raw {x['raw_c']:.0f}C" for x in temp["glitches"]
                )
                temp_text += f"  |  REJECTED BYTE GLITCH: {detail}"
            _text(frame, temp_text, 38, 201, scale=0.47,
                  color=(80, 190, 255) if temp["glitches"] else (230, 235, 230))

            panel_w = min(340, width - 36)
            panel_x = width - panel_w - 18
            _box(frame, panel_x, 18, panel_w, 286)
            _text(frame, "DIRECT ENCODERS (deg)", panel_x + 18, 49,
                  scale=0.55, color=(80, 230, 255), thickness=2)
            _text(frame, "leg       yaw       hip      knee", panel_x + 18, 75,
                  scale=0.44, color=(180, 190, 185))
            joints = json.loads(row["joint_degrees"])
            for leg in range(6):
                values = joints[3 * leg:3 * leg + 3]
                while len(values) < 3:
                    values.append(None)
                line = (
                    f"L{leg}    {_joint_text(values[0])}  "
                    f"{_joint_text(values[1])}  {_joint_text(values[2])}"
                )
                _text(frame, line, panel_x + 18, 103 + leg * 27,
                      scale=0.49, color=(235, 240, 236))
            _text(frame, "Knees: encoder only; visual knee discarded",
                  panel_x + 18, 275, scale=0.40, color=(100, 220, 255))
            vision = (
                f"VISION diagnostic: {tags} direct tags, {feet}/6 direct feet"
                + ("" if visual_tilt is None else f", tilt {visual_tilt:.1f} deg")
            )
            _text(frame, vision, 30, height - 28, scale=0.48,
                  color=(90, 240, 205), thickness=1)
            if encoder.stdin is None:
                raise BrokenPipeError("ffmpeg stdin unavailable")
            encoder.stdin.write(frame.tobytes())
            frames += 1
            if args.max_frames is not None and frames >= args.max_frames:
                break
    finally:
        capture.release()
        if encoder.stdin is not None:
            encoder.stdin.close()
        code = encoder.wait()
        if code:
            raise RuntimeError(f"ffmpeg exited {code}")

    clips: list[dict[str, Any]] = []
    if args.clips_dir is not None and args.max_frames is None:
        args.clips_dir.mkdir(parents=True, exist_ok=True)
        for gait in range(9):
            matching = [
                row for row in telemetry
                if row.get("gait", "").strip() == str(gait)
            ]
            if not matching:
                continue
            start_frame = float(np.interp(
                float(matching[0]["receipt_unix_s"]), source_unix, source_frames
            ))
            end_frame = float(np.interp(
                float(matching[-1]["receipt_unix_s"]), source_unix, source_frames
            ))
            start = max(0.0, start_frame / raw_writer_fps - 0.5)
            end = end_frame / raw_writer_fps + 0.8
            clip = args.clips_dir / f"gait_{gait}_{GAITS[gait].lower().replace(' ', '_')}.mp4"
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
                "-i", str(args.output), "-c", "copy", str(clip),
            ], check=True)
            clips.append({
                "gait": gait, "name": GAITS[gait], "start_s": round(start, 3),
                "end_s": round(end, 3), "video": str(clip),
            })
    manifest = {
        "source_video": str(video),
        "annotated_video": str(args.output),
        "telemetry_csv": str(args.run_dir / "telemetry.csv"),
        "events_csv": str(args.run_dir / "events.csv"),
        "apriltag_pose_jsonl": None if args.pose_jsonl is None else str(args.pose_jsonl),
        "apriltag_motion_json": str(args.run_dir / "apriltag_motion.json"),
        "mujoco_comparison_json": str(args.run_dir / "mujoco_comparison.json"),
        "joint_frame": FRAME_ROBOT_ABS,
        "joint_contract": JOINT_CONTRACT,
        "visual_knees_used": False,
        "temperature_filter": "same servo must be >=55C on consecutive samples",
        "time_alignment": "raw frame index -> iphone_raw_timestamps.csv wall clock",
        "frames": frames,
        "fps": fps,
        "clips": clips,
    }
    (args.output.parent / "llm_review_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
