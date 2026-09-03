#!/usr/bin/env python3
"""Detect AprilTags in a photo, video, or live camera and estimate pose.

The tool never connects to the robot.  It can save raw camera media, annotated
media, and per-frame JSON/JSONL pose records.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import threading
import time
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

import cv2

from .apriltag_vision import AprilTagPoseTracker
from .housing_pose import JOINT_NAMES


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


class FeedbackClient:
    """Rate-limited, read-only client for the robot's bulk feedback route."""

    def __init__(self, base_url: str, *, hz: float = 3.0, timeout_s: float = 0.6):
        self.url = base_url.rstrip("/") + "/api/feedback"
        self.minimum_interval_s = 1.0 / hz
        self.timeout_s = timeout_s
        self.last_poll = -float("inf")
        self.last_angles: dict[str, float] = {}
        self._lock = threading.Lock()
        self._poll_in_flight = False
        self.status: dict[str, Any] = {
            "configured": True,
            "ok": False,
            "endpoint": self.url,
            "error": "not polled yet",
        }

    def sample(self) -> tuple[dict[str, float], dict[str, Any]]:
        now = time.monotonic()
        with self._lock:
            should_poll = (
                not self._poll_in_flight
                and now - self.last_poll >= self.minimum_interval_s
            )
            if should_poll:
                self.last_poll = now
                self._poll_in_flight = True
            angles = dict(self.last_angles)
            status = dict(self.status)
        if should_poll:
            threading.Thread(target=self._poll, daemon=True).start()
        return angles, status

    def _poll(self) -> None:
        try:
            request = Request(self.url, headers={"Accept": "application/json"})
            with urlopen(request, timeout=self.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not payload.get("ok"):
                raise OSError(str(payload.get("error", "feedback not ok")))
            joints = payload.get("joints", [])
            angles = {
                name: float(item["deg"])
                for name, item in zip(JOINT_NAMES, joints)
                if isinstance(item, dict) and item.get("deg") is not None
            }
            status = {
                "configured": True,
                "ok": True,
                "endpoint": self.url,
                "sample_time_unix": payload.get("t_unix"),
                "live_joint_count": len(angles),
                "roll_deg": payload.get("roll_deg"),
                "pitch_deg": payload.get("pitch_deg"),
            }
        except (OSError, URLError, ValueError, json.JSONDecodeError) as error:
            angles = None
            status = {
                "configured": True,
                "ok": False,
                "endpoint": self.url,
                "using_cached_joint_count": len(self.last_angles),
                "error": str(error),
            }
        with self._lock:
            if angles is not None:
                self.last_angles = angles
            self.status = status
            self._poll_in_flight = False


class VideoDiagnosticAccumulator:
    """Compact cross-frame evidence for calibration and gait triage."""

    def __init__(self) -> None:
        self.frames = 0
        self.decoded_tags = 0
        self.foot_direct = [0] * 6
        self.foot_inferred = [0] * 6
        self.foot_speeds: list[list[float]] = [[] for _ in range(6)]
        self.disagreement_count: dict[str, int] = {}
        self.disagreement_max: dict[str, float] = {}
        self.zero_error_count: dict[str, int] = {}
        self.max_body_tilt_deg = 0.0
        self.safety_verdict_counts = {
            "safe": 0,
            "unsafe": 0,
            "unverified": 0,
        }

    def update(self, result: dict[str, Any]) -> None:
        self.frames += 1
        self.decoded_tags += len(result.get("detected_tag_ids", []))
        for foot in result.get("foot_tips", []):
            leg = int(foot["leg"])
            if foot.get("source") == "color":
                self.foot_direct[leg] += 1
            else:
                self.foot_inferred[leg] += 1
            speed = foot.get("floor_projection_speed_m_s")
            if speed is not None:
                self.foot_speeds[leg].append(float(speed))
        full = result.get("full_pose") or {}
        for item in full.get("calibration_disagreements", []):
            name = str(item["joint"])
            raw_error = item.get(
                "visual_minus_encoder_deg",
                item.get("visual_abs_minus_encoder_abs_deg", 0.0),
            )
            error = abs(float(raw_error))
            self.disagreement_count[name] = self.disagreement_count.get(name, 0) + 1
            self.disagreement_max[name] = max(
                error, self.disagreement_max.get(name, 0.0)
            )
        for item in full.get("zero_check", {}).get("out_of_tolerance", []):
            name = str(item["joint"])
            self.zero_error_count[name] = self.zero_error_count.get(name, 0) + 1
        tilt = full.get("walking_check", {}).get("body_tilt_deg")
        if tilt is not None:
            self.max_body_tilt_deg = max(self.max_body_tilt_deg, float(tilt))
        verdict = result.get("safety_assessment", {}).get("verdict")
        if verdict in self.safety_verdict_counts:
            self.safety_verdict_counts[verdict] += 1

    def summary(self) -> dict[str, Any]:
        frames = max(1, self.frames)
        feet = []
        for leg in range(6):
            speeds = sorted(self.foot_speeds[leg])
            percentile = None
            if speeds:
                percentile = speeds[min(len(speeds) - 1, round(0.95 * (len(speeds) - 1)))]
            feet.append({
                "leg": leg,
                "direct_visible_fraction": round(self.foot_direct[leg] / frames, 3),
                "inferred_fraction": round(self.foot_inferred[leg] / frames, 3),
                "floor_projection_speed_p95_m_s": (
                    None if percentile is None else round(percentile, 4)
                ),
            })
        persistent_disagreements = [
            {
                "joint": name,
                "frame_fraction": round(count / frames, 3),
                "max_abs_deg": round(self.disagreement_max[name], 3),
            }
            for name, count in sorted(self.disagreement_count.items())
            if count / frames >= 0.2
        ]
        persistent_zero_errors = [
            {"joint": name, "frame_fraction": round(count / frames, 3)}
            for name, count in sorted(self.zero_error_count.items())
            if count / frames >= 0.2
        ]
        return {
            "frames": self.frames,
            "mean_decoded_tags_per_frame": round(self.decoded_tags / frames, 2),
            "max_body_tilt_deg": round(self.max_body_tilt_deg, 3),
            "safety_verdict_fractions": {
                name: round(count / frames, 3)
                for name, count in self.safety_verdict_counts.items()
            },
            "feet": feet,
            "persistent_visual_encoder_disagreements": persistent_disagreements,
            "persistent_zero_pose_errors": persistent_zero_errors,
            "notes": [
                "High direct-visible fractions make per-leg trajectory comparisons reliable.",
                "Floor-projection speed is a slip candidate signal, not proof of contact.",
                "Persistent visual/encoder disagreement is stronger evidence "
                "of a zero or mount problem than one frame.",
            ],
        }


def _safe_pose_assessment(
    result: dict[str, Any],
    feedback: dict[str, Any],
    *,
    operator_supported: bool,
) -> dict[str, Any]:
    full = result.get("full_pose") or {}
    detections = result.get("detections", [])
    direct_robot_tags = [
        item for item in detections
        if item.get("source") == "detected"
        and not str(item.get("label", "")).lower().startswith("floor")
    ]
    direct_feet = [
        item for item in result.get("foot_tips", [])
        if item.get("source") == "color"
    ]
    body_tilt = full.get("walking_check", {}).get("body_tilt_deg")
    imu_roll = feedback.get("roll_deg") if feedback.get("ok") else None
    imu_pitch = feedback.get("pitch_deg") if feedback.get("ok") else None
    imu_tilt = None
    if imu_roll is not None and imu_pitch is not None:
        imu_tilt = (float(imu_roll) ** 2 + float(imu_pitch) ** 2) ** 0.5

    unsafe: list[str] = []
    unknown: list[str] = []
    warnings: list[str] = []
    # Prefer the robot's calibrated IMU for safety. The monocular visual tilt
    # remains a useful calibration diagnostic, but approximate phone
    # intrinsics and a slightly tilted chassis tag can bias it by several
    # degrees without making the physical pose unsafe.
    if imu_tilt is not None:
        if imu_tilt > 15.0:
            unsafe.append(f"IMU tilt is {imu_tilt:.1f} deg")
        if body_tilt is not None and abs(float(body_tilt) - imu_tilt) > 3.0:
            warnings.append(
                f"visual tilt {float(body_tilt):.1f} deg disagrees with "
                f"IMU tilt {imu_tilt:.1f} deg"
            )
    elif body_tilt is not None and float(body_tilt) > 15.0:
        if result.get("camera_calibration_approximate"):
            unknown.append(
                f"visual tilt is {float(body_tilt):.1f} deg but phone "
                "calibration is approximate"
            )
        else:
            unsafe.append(f"visual body tilt is {float(body_tilt):.1f} deg")
    if body_tilt is None and imu_tilt is None:
        unknown.append("neither floor-referenced body tilt nor IMU tilt is available")

    joint_limits = {"yaw": 75.0, "hip": 75.0, "knee": 150.0}
    for name, record in full.get("joints", {}).items():
        value = record.get("value_deg")
        if value is None:
            continue
        axis = name.rsplit("_", 1)[1]
        if abs(float(value)) > joint_limits[axis]:
            unsafe.append(
                f"{name} is outside the broad safe envelope ({float(value):.1f} deg)"
            )
    for disagreement in full.get("calibration_disagreements", []):
        delta = disagreement.get(
            "visual_minus_encoder_deg",
            disagreement.get("visual_abs_minus_encoder_abs_deg"),
        )
        if delta is None or abs(float(delta)) <= 15.0:
            continue
        message = (
            f"{disagreement['joint']} visual/encoder mismatch is "
            f"{float(delta):+.1f} deg"
        )
        if disagreement.get("unsigned_visual_estimate"):
            warnings.append(message + " (provisional foot-tip estimate)")
        else:
            # A disagreement diagnoses a vision/tag-mount calibration problem;
            # it does not prove that the encoder-reported physical pose is
            # unsafe. Keep automatic alignment blocked via UNVERIFIED.
            unknown.append(message)

    if len(direct_robot_tags) < 7:
        unknown.append(
            f"only {len(direct_robot_tags)} robot tags are directly decoded"
        )
    if len(direct_feet) < 4:
        unknown.append(f"only {len(direct_feet)} feet are directly visible")
    if not feedback.get("ok") or feedback.get("live_joint_count", 0) < 18:
        unknown.append("18/18 read-only encoder feedback is unavailable")
    if result.get("camera_calibration_approximate"):
        warnings.append("phone lens calibration is still approximate")
    if full.get("prediction_only_joints"):
        warnings.append("one or more joints use short-term prediction")
    if any(
        record.get("visual_source") == "foot_tip_projection_magnitude"
        for name, record in full.get("joints", {}).items()
        if name.endswith("_knee")
    ):
        warnings.append(
            "knee vision is unobservable with lid tags; using encoders"
        )

    if unsafe:
        verdict = "unsafe"
    elif unknown:
        verdict = "unverified"
    else:
        verdict = "safe"
    zero_check = full.get("zero_check", {})
    primary_tilt = imu_tilt if imu_tilt is not None else body_tilt
    straight_horizontal = bool(
        zero_check.get("matches_zero")
        and (primary_tilt is None or float(primary_tilt) <= 5.0)
        and len(direct_feet) == 6
    )
    alignment_blockers = list(unsafe) + list(unknown)
    if not operator_supported:
        alignment_blockers.append(
            "operator has not confirmed the chassis is supported with legs free"
        )
    if result.get("camera_calibration_approximate"):
        alignment_blockers.append("phone lens calibration is approximate")
    if len(direct_robot_tags) < 13 or len(direct_feet) < 6:
        alignment_blockers.append(
            "automatic alignment requires all 13 robot tags and all 6 feet"
        )
    return {
        "verdict": verdict,
        "safe_pose": verdict == "safe",
        "straight_horizontal_candidate": straight_horizontal,
        "safe_for_alignment_motion": not alignment_blockers,
        "operator_supported": operator_supported,
        "direct_robot_tag_count": len(direct_robot_tags),
        "direct_foot_count": len(direct_feet),
        "body_tilt_deg": body_tilt,
        "imu_tilt_deg": None if imu_tilt is None else round(imu_tilt, 3),
        "unsafe_reasons": unsafe,
        "unknown_reasons": unknown,
        "warnings": warnings,
        "alignment_motion_blockers": alignment_blockers,
        "motor_commands_sent": False,
    }


def _draw_safety_assessment(image: Any, assessment: dict[str, Any]) -> None:
    verdict = assessment["verdict"]
    colors = {
        "safe": (40, 220, 40),
        "unsafe": (20, 20, 255),
        "unverified": (0, 180, 255),
    }
    scale = max(0.55, image.shape[1] / 2600.0)
    thickness = max(1, round(scale * 2))
    text = f"POSE SAFETY: {verdict.upper()}"
    cv2.putText(
        image, text, (20, 125), cv2.FONT_HERSHEY_SIMPLEX,
        scale, colors[verdict], thickness + 1, cv2.LINE_AA,
    )
    reasons = assessment["unsafe_reasons"] or assessment["unknown_reasons"]
    if reasons:
        cv2.putText(
            image, reasons[0][:90], (20, 160), cv2.FONT_HERSHEY_SIMPLEX,
            scale * 0.85, colors[verdict], thickness, cv2.LINE_AA,
        )


def _process_one(
    tracker: AprilTagPoseTracker,
    frame: Any,
    *,
    frame_index: int = 0,
    time_s: float | None = None,
    feedback: FeedbackClient | None = None,
    operator_supported: bool = False,
) -> tuple[dict[str, Any], Any]:
    encoder, feedback_status = ({}, {"configured": False})
    if feedback is not None:
        encoder, feedback_status = feedback.sample()
    result, annotated = tracker.process_frame(
        frame,
        frame_index=frame_index,
        time_s=time_s,
        encoder_joint_deg=encoder or None,
    )
    result["encoder_feedback"] = feedback_status
    result["safety_assessment"] = _safe_pose_assessment(
        result,
        feedback_status,
        operator_supported=operator_supported,
    )
    _draw_safety_assessment(annotated, result["safety_assessment"])
    return result, annotated


def _write_json(path: Path | None, value: dict[str, Any]) -> None:
    rendered = json.dumps(value, indent=2) + "\n"
    if path is None:
        print(rendered, end="")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def _write_image(path: Path | None, image: Any) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"could not write image {path}")


def _process_image(
    tracker: AprilTagPoseTracker,
    image_path: Path,
    *,
    pose_output: Path | None,
    annotated_output: Path | None,
    feedback: FeedbackClient | None,
    summary_output: Path | None,
    operator_supported: bool,
) -> int:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"could not read image {image_path}")
    result, annotated = _process_one(
        tracker,
        image,
        feedback=feedback,
        operator_supported=operator_supported,
    )
    _write_json(pose_output, result)
    if summary_output is not None:
        diagnostics = VideoDiagnosticAccumulator()
        diagnostics.update(result)
        _write_json(summary_output, diagnostics.summary())
    _write_image(annotated_output, annotated)
    hexapod_pose = result.get("hexapod_pose")
    return 0 if not hexapod_pose or hexapod_pose.get("ok", True) else 2


def _video_writer(path: Path, fps: float, size: tuple[int, int]) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
    )
    if not writer.isOpened():
        raise OSError(f"could not open video writer {path}")
    return writer


def _parse_camera_cycle(value: str) -> tuple[int, ...]:
    try:
        indexes = tuple(dict.fromkeys(
            int(item.strip()) for item in value.split(",") if item.strip()
        ))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "camera cycle must be comma-separated integer indexes"
        ) from error
    if not indexes or any(index < 0 for index in indexes):
        raise argparse.ArgumentTypeError(
            "camera cycle needs one or more non-negative indexes"
        )
    return indexes


def _camera_order_after(current: int, cycle: tuple[int, ...]) -> tuple[int, ...]:
    indexes = cycle if current in cycle else cycle + (current,)
    position = indexes.index(current)
    return indexes[position + 1:] + indexes[:position + 1]


def _switch_camera(
    capture: Any,
    current: int,
    cycle: tuple[int, ...],
) -> tuple[Any, int]:
    capture.release()
    failures: list[int] = []
    for index in _camera_order_after(current, cycle):
        candidate = cv2.VideoCapture(index)
        if candidate.isOpened():
            ok, _frame = candidate.read()
            if ok:
                return candidate, index
        failures.append(index)
        candidate.release()
    raise OSError(f"could not open any camera in cycle {failures}")


def _fit_writer_size(image: Any, size: tuple[int, int]) -> Any:
    if (image.shape[1], image.shape[0]) == size:
        return image
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _resize_for_processing(frame: Any, maximum_width: int) -> Any:
    if maximum_width <= 0 or frame.shape[1] <= maximum_width:
        return frame
    scale = maximum_width / frame.shape[1]
    size = (maximum_width, max(1, round(frame.shape[0] * scale)))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def _process_capture(
    tracker: AprilTagPoseTracker,
    capture: Any,
    *,
    pose_output: Path | None,
    annotated_output: Path | None,
    raw_output: Path | None,
    duration_s: float | None,
    max_frames: int | None,
    camera_mode: bool,
    feedback: FeedbackClient | None,
    preview: bool,
    summary_output: Path | None,
    camera_index: int | None = None,
    camera_cycle: tuple[int, ...] = (0, 1),
    processing_width: int = 1280,
    frame_step: int = 1,
    operator_supported: bool = False,
) -> int:
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math_is_finite_positive(fps):
        fps = 30.0
    annotated_writer = None
    raw_writer = None
    json_handle = None
    start = time.monotonic()
    source_frame_index = 0
    processed_frames = 0
    last_result: dict[str, Any] | None = None
    diagnostics = VideoDiagnosticAccumulator()
    writer_size: tuple[int, int] | None = None
    active_camera_index = camera_index
    camera_history = [] if camera_index is None else [camera_index]
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            # Offline surveys only need pose samples at roughly 10 Hz. Keep
            # the original source-frame index/time in JSONL while avoiding
            # expensive AprilTag detection on intermediate frames.
            if not camera_mode and source_frame_index % frame_step != 0:
                source_frame_index += 1
                if max_frames is not None and source_frame_index >= max_frames:
                    break
                continue
            if processed_frames == 0:
                height, width = frame.shape[:2]
                writer_size = (width, height)
                if annotated_output is not None:
                    annotated_writer = _video_writer(
                        annotated_output,
                        fps if camera_mode else fps / frame_step,
                        (width, height),
                    )
                if raw_output is not None:
                    raw_writer = _video_writer(raw_output, fps, (width, height))
                if pose_output is not None:
                    pose_output.parent.mkdir(parents=True, exist_ok=True)
                    json_handle = pose_output.open("w", encoding="utf-8")

            if camera_mode:
                time_s = time.monotonic() - start
            else:
                time_s = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            processing_frame = _resize_for_processing(frame, processing_width)
            result, annotated = _process_one(
                tracker,
                processing_frame,
                frame_index=source_frame_index,
                time_s=time_s,
                feedback=feedback,
                operator_supported=operator_supported,
            )
            if active_camera_index is not None:
                result["camera_index"] = active_camera_index
                result["capture_image_size_px"] = [frame.shape[1], frame.shape[0]]
                cv2.putText(
                    annotated,
                    f"CAMERA {active_camera_index} | C: switch | Q/Esc: stop",
                    (20, annotated.shape[0] - 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.55, annotated.shape[1] / 2600.0),
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            last_result = result
            diagnostics.update(result)
            if raw_writer is not None:
                raw_writer.write(_fit_writer_size(frame, writer_size))
            if annotated_writer is not None:
                annotated_writer.write(_fit_writer_size(annotated, writer_size))
            if json_handle is not None:
                json_handle.write(json.dumps(result, separators=(",", ":")) + "\n")
            if preview:
                cv2.imshow(
                    "Hexapod visual checkup (C camera, Q/Esc stop)", annotated
                )
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    source_frame_index += 1
                    processed_frames += 1
                    break
                if (key in (ord("c"), ord("C")) and camera_mode
                        and active_camera_index is not None):
                    capture, active_camera_index = _switch_camera(
                        capture, active_camera_index, camera_cycle
                    )
                    camera_history.append(active_camera_index)
                    tracker.reset_temporal_state()
                    print(f"switched to camera {active_camera_index}", flush=True)

            source_frame_index += 1
            processed_frames += 1
            if max_frames is not None and source_frame_index >= max_frames:
                break
            if duration_s is not None and time.monotonic() - start >= duration_s:
                break
    finally:
        capture.release()
        if annotated_writer is not None:
            annotated_writer.release()
        if raw_writer is not None:
            raw_writer.release()
        if json_handle is not None:
            json_handle.close()
        if preview:
            cv2.destroyAllWindows()

    if processed_frames == 0 or last_result is None:
        raise OSError("capture produced no frames")
    summary = {
        "frames": processed_frames,
        "source_frames_read": source_frame_index,
        "frame_step": frame_step,
        "pose_output": None if pose_output is None else str(pose_output),
        "annotated_output": (
            None if annotated_output is None else str(annotated_output)
        ),
        "raw_output": None if raw_output is None else str(raw_output),
        "last_detected_tag_ids": last_result["detected_tag_ids"],
        "camera_history": camera_history,
        "diagnostic_summary": diagnostics.summary(),
    }
    if summary_output is not None:
        _write_json(summary_output, summary)
    print(json.dumps(summary, indent=2))
    return 0


def math_is_finite_positive(value: float) -> bool:
    return value > 0.0 and value < float("inf")


def _capture_still(
    tracker: AprilTagPoseTracker,
    camera_index: int,
    *,
    pose_output: Path | None,
    annotated_output: Path | None,
    raw_output: Path | None,
    feedback: FeedbackClient | None,
    summary_output: Path | None,
    operator_supported: bool,
) -> int:
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise OSError(f"could not open camera {camera_index}")
    try:
        # Let auto-exposure settle without sleeping or retaining stale frames.
        frame = None
        for _ in range(12):
            ok, candidate = capture.read()
            if ok:
                frame = candidate
        if frame is None:
            raise OSError(f"camera {camera_index} produced no frame")
    finally:
        capture.release()
    result, annotated = _process_one(
        tracker,
        frame,
        feedback=feedback,
        operator_supported=operator_supported,
    )
    _write_json(pose_output, result)
    if summary_output is not None:
        diagnostics = VideoDiagnosticAccumulator()
        diagnostics.update(result)
        _write_json(summary_output, diagnostics.summary())
    _write_image(raw_output, frame)
    _write_image(annotated_output, annotated)
    hexapod_pose = result.get("hexapod_pose")
    return 0 if not hexapod_pose or hexapod_pose.get("ok", True) else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="camera/tag-layout JSON")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="input image or video")
    source.add_argument("--camera", type=int, help="OpenCV camera index")
    parser.add_argument("--pose-output", type=Path,
                        help="JSON for an image, JSONL for video")
    parser.add_argument("--annotated-output", type=Path,
                        help="annotated image or MP4")
    parser.add_argument("--summary-output", type=Path,
                        help="write a compact cross-frame diagnostic JSON")
    parser.add_argument("--raw-output", type=Path,
                        help="save raw camera photo or MP4 (camera mode only)")
    parser.add_argument("--duration", type=float,
                        help="camera recording duration; omitted means one photo")
    parser.add_argument("--max-frames", type=int,
                        help="optional video frame limit for testing")
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help=("process every Nth frame of an input video while preserving "
              "source timestamps/indexes (default: 1)"),
    )
    parser.add_argument(
        "--robot-url",
        help=("optional robot base URL, e.g. http://hexapod.local:8080; "
              "only GET /api/feedback is used and no motor command is sent"),
    )
    parser.add_argument("--feedback-hz", type=float, default=3.0,
                        help="read-only robot feedback rate (default: 3 Hz)")
    parser.add_argument(
        "--preview",
        action="store_true",
        help=("show the annotated live/video checkup; press C to switch "
              "camera or Q/Esc to stop"),
    )
    parser.add_argument(
        "--camera-cycle",
        type=_parse_camera_cycle,
        default=(0, 1),
        metavar="INDEXES",
        help="camera indexes cycled by C in preview (default: 0,1)",
    )
    parser.add_argument(
        "--processing-width",
        type=int,
        default=1280,
        help="downscale video/live processing to this width; 0 disables",
    )
    parser.add_argument(
        "--robot-supported",
        action="store_true",
        help=("operator assertion that chassis is supported and every leg is "
              "free; does not enable or send motion"),
    )
    args = parser.parse_args(argv)
    if args.duration is not None and args.duration <= 0.0:
        parser.error("--duration must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    if args.frame_step <= 0:
        parser.error("--frame-step must be positive")
    if args.camera is not None and args.frame_step != 1:
        parser.error("--frame-step is only supported with --input video")
    if args.feedback_hz <= 0.0:
        parser.error("--feedback-hz must be positive")
    if args.processing_width != 0 and args.processing_width < 320:
        parser.error("--processing-width must be 0 or at least 320")
    if args.input is not None and args.raw_output is not None:
        parser.error("--raw-output is only for --camera capture")

    tracker = AprilTagPoseTracker.from_json(args.config)
    feedback = (
        None if args.robot_url is None
        else FeedbackClient(args.robot_url, hz=args.feedback_hz)
    )
    if args.input is not None and args.input.suffix.lower() in _IMAGE_SUFFIXES:
        return _process_image(
            tracker,
            args.input,
            pose_output=args.pose_output,
            annotated_output=args.annotated_output,
            feedback=feedback,
            summary_output=args.summary_output,
            operator_supported=args.robot_supported,
        )
    if args.input is not None:
        capture = cv2.VideoCapture(str(args.input))
        if not capture.isOpened():
            raise OSError(f"could not open video {args.input}")
        return _process_capture(
            tracker,
            capture,
            pose_output=args.pose_output,
            annotated_output=args.annotated_output,
            raw_output=None,
            duration_s=None,
            max_frames=args.max_frames,
            camera_mode=False,
            feedback=feedback,
            preview=args.preview,
            summary_output=args.summary_output,
            camera_index=None,
            camera_cycle=args.camera_cycle,
            processing_width=args.processing_width,
            frame_step=args.frame_step,
            operator_supported=args.robot_supported,
        )
    if args.duration is None and not args.preview:
        return _capture_still(
            tracker,
            args.camera,
            pose_output=args.pose_output,
            annotated_output=args.annotated_output,
            raw_output=args.raw_output,
            feedback=feedback,
            summary_output=args.summary_output,
            operator_supported=args.robot_supported,
        )

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise OSError(f"could not open camera {args.camera}")
    return _process_capture(
        tracker,
        capture,
        pose_output=args.pose_output,
        annotated_output=args.annotated_output,
        raw_output=args.raw_output,
        duration_s=args.duration,
        max_frames=args.max_frames,
        camera_mode=True,
        feedback=feedback,
        preview=args.preview,
        summary_output=args.summary_output,
        camera_index=args.camera,
        camera_cycle=args.camera_cycle,
        processing_width=args.processing_width,
        frame_step=1,
        operator_supported=args.robot_supported,
    )


if __name__ == "__main__":
    raise SystemExit(main())
