#!/usr/bin/env python3
"""AprilTag camera, calibration, and optional survey routes for a web hub.

The service owns the OpenCV camera, publishes a latest-frame MJPEG stream,
and exposes compact JSON state for the browser UI.  Visual calibration is a
stationary multi-frame comparison between AprilTag-derived joint angles and
read-only encoder feedback. Calibration remains observation-only. Survey
routes stay unavailable unless a consuming robot project injects its own
guarded motion adapter.
"""
from __future__ import annotations

from collections import Counter, deque
import json
import math
from http import HTTPStatus
from pathlib import Path
import statistics
import sys
import threading
import time
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

import cv2

from .apriltag_vision import AprilTagPoseTracker
from .joint_contract import FRAME_ROBOT_ABS, JOINT_CONTRACT
from .paths import CONFIG_DIR, DEFAULT_REPORT_DIR, WEB_DIST_DIR
from .zero_survey_web import ZeroPoseSurveyManager
from .track import (
    FeedbackClient,
    _resize_for_processing,
    _safe_pose_assessment,
)


DEFAULT_CONFIG = CONFIG_DIR / "apriltag_pose_config_20260831.json"
DEFAULT_UI_DIR = WEB_DIST_DIR
DEFAULT_STABLE_FRAMES = 12
DEFAULT_CALIBRATION_FRAMES = 45


def _direct_ids(result: Mapping[str, Any], allowed: set[int]) -> set[int]:
    return {
        int(item["tag_id"])
        for item in result.get("detections", [])
        if item.get("source") == "detected" and int(item["tag_id"]) in allowed
    }


def _direct_signed_joint_names(result: Mapping[str, Any]) -> set[str]:
    full_pose = result.get("full_pose") or {}
    return {
        str(name)
        for name, record in full_pose.get("joints", {}).items()
        if record.get("visual_source") == "apriltag"
        and record.get("visual_deg") is not None
        and record.get("encoder_deg") is not None
    }


def _frame_calibration_facts(
    result: Mapping[str, Any],
    *,
    robot_tag_ids: set[int],
    floor_tag_ids: set[int],
) -> dict[str, Any]:
    direct_robot = _direct_ids(result, robot_tag_ids)
    direct_floor = _direct_ids(result, floor_tag_ids)
    direct_feet = {
        int(item["leg"])
        for item in result.get("foot_tips", [])
        if item.get("source") == "color"
    }
    signed_joints = _direct_signed_joint_names(result)
    feedback = result.get("encoder_feedback") or {}
    safety = result.get("safety_assessment") or {}
    blockers: list[str] = []
    missing_robot = sorted(robot_tag_ids - direct_robot)
    if missing_robot:
        blockers.append(
            "Show every robot lid tag "
            f"({len(direct_robot)}/{len(robot_tag_ids)} direct; missing "
            + ", ".join(str(value) for value in missing_robot)
            + ")"
        )
    if len(direct_floor) < min(2, len(floor_tag_ids)):
        blockers.append(
            f"Show at least two floor tags ({len(direct_floor)}/"
            f"{len(floor_tag_ids)} direct)"
        )
    if not feedback.get("ok") or int(feedback.get("live_joint_count", 0)) < 18:
        blockers.append("Connect read-only encoder feedback (18/18 joints)")
    if len(signed_joints) < 12:
        blockers.append(
            f"Need 12 direct yaw/hip comparisons ({len(signed_joints)}/12)"
        )
    if safety.get("verdict") == "unsafe":
        reasons = safety.get("unsafe_reasons") or ["pose safety check failed"]
        blockers.append(str(reasons[0]))
    if result.get("pose_reference") != "floor":
        blockers.append("Hold the camera so the pose is floor-referenced")
    warnings: list[str] = []
    if result.get("camera_calibration_approximate"):
        warnings.append(
            "iPhone lens intrinsics are approximate; offsets are provisional"
        )
    warnings.append(
        "Knees are not observed by the lid tags; this capture calibrates "
        "the 12 signed yaw/hip joints only"
    )
    return {
        "eligible": not blockers,
        "blockers": blockers,
        "warnings": warnings,
        "direct_robot_tag_ids": sorted(direct_robot),
        "direct_floor_tag_ids": sorted(direct_floor),
        "direct_foot_legs": sorted(direct_feet),
        "direct_signed_joint_names": sorted(signed_joints),
    }


def _joint_motion_span_deg(results: Sequence[Mapping[str, Any]]) -> float | None:
    values: dict[str, list[float]] = {}
    for result in results:
        joints = (result.get("full_pose") or {}).get("joints", {})
        for name, record in joints.items():
            encoder = record.get("encoder_deg")
            visual = record.get("visual_deg")
            if encoder is not None:
                values.setdefault(f"encoder:{name}", []).append(float(encoder))
            # When live feedback exists it is the independent evidence that
            # the physical robot is stationary. Visual jitter is precisely
            # what this calibration is intended to measure, so it must not be
            # misclassified as robot motion.
            elif visual is not None and record.get("visual_source") == "apriltag":
                values.setdefault(f"visual:{name}", []).append(float(visual))
    spans = [max(items) - min(items) for items in values.values() if len(items) >= 2]
    return None if not spans else max(spans)


def assess_visual_calibration_readiness(
    history: Sequence[Mapping[str, Any]],
    *,
    robot_tag_ids: set[int],
    floor_tag_ids: set[int],
    stable_frames: int = DEFAULT_STABLE_FRAMES,
    maximum_motion_deg: float = 2.0,
) -> dict[str, Any]:
    """Return UI-ready evidence that a stationary capture may begin."""
    if not history:
        return {
            "ready": False,
            "status": "waiting_for_camera",
            "headline": "Waiting for camera",
            "blockers": ["No processed camera frame is available"],
            "warnings": [],
            "stable_frames": 0,
            "required_stable_frames": stable_frames,
            "progress": 0.0,
            "maximum_joint_motion_deg": None,
            "scope": "none",
        }

    latest = history[-1]
    latest_facts = _frame_calibration_facts(
        latest, robot_tag_ids=robot_tag_ids, floor_tag_ids=floor_tag_ids
    )
    eligible_tail: list[Mapping[str, Any]] = []
    for result in reversed(history):
        facts = _frame_calibration_facts(
            result, robot_tag_ids=robot_tag_ids, floor_tag_ids=floor_tag_ids
        )
        if not facts["eligible"]:
            break
        eligible_tail.append(result)
        if len(eligible_tail) >= stable_frames:
            break
    eligible_tail.reverse()
    motion_span = _joint_motion_span_deg(eligible_tail)
    stationary = (
        len(eligible_tail) >= stable_frames
        and motion_span is not None
        and motion_span <= maximum_motion_deg
    )
    blockers = list(latest_facts["blockers"])
    if latest_facts["eligible"] and len(eligible_tail) < stable_frames:
        blockers.append(
            f"Hold the robot still ({len(eligible_tail)}/{stable_frames} frames)"
        )
    elif latest_facts["eligible"] and not stationary:
        blockers.append(
            "Hold still; observed joint motion is "
            f"{motion_span:.1f}° (limit {maximum_motion_deg:.1f}°)"
        )
    ready = latest_facts["eligible"] and stationary
    provisional = bool(latest.get("camera_calibration_approximate"))
    scope = "lid_joints"
    if ready:
        status = "ready_provisional" if provisional else "ready"
        headline = (
            "Ready for provisional capture" if provisional
            else "Ready for visual calibration"
        )
    elif latest_facts["eligible"]:
        status = "hold_still"
        headline = "Hold still"
    else:
        status = "not_ready"
        headline = "Not ready"
    return {
        "ready": ready,
        "status": status,
        "headline": headline,
        "blockers": blockers,
        "warnings": latest_facts["warnings"],
        "stable_frames": len(eligible_tail),
        "required_stable_frames": stable_frames,
        "progress": round(min(1.0, len(eligible_tail) / stable_frames), 3),
        "maximum_joint_motion_deg": (
            None if motion_span is None else round(motion_span, 3)
        ),
        "scope": scope,
        "camera_calibration_provisional": provisional,
        "coverage": {
            "robot_tags": len(latest_facts["direct_robot_tag_ids"]),
            "robot_tags_required": len(robot_tag_ids),
            "floor_tags": len(latest_facts["direct_floor_tag_ids"]),
            "floor_tags_available": len(floor_tag_ids),
            "feet": len(latest_facts["direct_foot_legs"]),
            "signed_joints": len(latest_facts["direct_signed_joint_names"]),
        },
    }


def _circular_median_deg(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot calculate an empty circular median")
    center = float(statistics.median(values))
    wrapped = [((value - center + 180.0) % 360.0) - 180.0 for value in values]
    return ((center + float(statistics.median(wrapped)) + 180.0) % 360.0) - 180.0


def build_visual_calibration_report(
    samples: Sequence[Mapping[str, Any]],
    *,
    camera_index: int,
    config_path: Path,
) -> dict[str, Any]:
    """Aggregate stable visual/encoder comparisons without applying them."""
    if not samples:
        raise ValueError("visual calibration requires at least one sample")
    joint_names = sorted({
        str(name)
        for sample in samples
        for name in (sample.get("full_pose") or {}).get("joints", {})
    })
    joints: list[dict[str, Any]] = []
    for name in joint_names:
        signed: list[float] = []
        confidences: list[float] = []
        for sample in samples:
            record = (sample.get("full_pose") or {}).get("joints", {}).get(name, {})
            delta = record.get("visual_minus_encoder_deg")
            if delta is not None and record.get("visual_source") == "apriltag":
                signed.append(float(delta))
                confidences.append(float(record.get("visual_confidence", 0.0)))
            # Foot-tip foreshortening is not a direct knee observation. Keep
            # knee records explicitly unobservable until a marker is attached
            # to the moving tibia/yoke.
        if signed:
            median = _circular_median_deg(signed)
            deviations = [abs(((value - median + 180.0) % 360.0) - 180.0)
                          for value in signed]
            mad = float(statistics.median(deviations))
            joints.append({
                "joint": name,
                "observable": True,
                "signed": True,
                "sample_count": len(signed),
                "visual_minus_encoder_deg": round(median, 3),
                "median_absolute_deviation_deg": round(mad, 3),
                "median_confidence": round(
                    float(statistics.median(confidences)), 3
                ),
                "quality": "good" if mad <= 1.5 else "unstable",
                "interpretation": (
                    "visual angle minus current logical encoder angle; review "
                    "this signed delta before changing any zero"
                ),
            })
        else:
            joints.append({
                "joint": name,
                "observable": False,
                "signed": False,
                "sample_count": 0,
                "quality": "unobservable",
                "interpretation": "no direct visual/encoder comparison",
            })
    approximate = any(
        bool(sample.get("camera_calibration_approximate")) for sample in samples
    )
    signed_good = sum(
        item["signed"] and item["quality"] == "good" for item in joints
    )
    return {
        "schema_version": 1,
        "kind": "advisory_visual_encoder_calibration",
        "created_unix": round(time.time(), 3),
        "camera_index": camera_index,
        "config_path": str(config_path),
        "sample_count": len(samples),
        "camera_calibration_approximate": approximate,
        "quality": "provisional" if approximate else (
            "good" if signed_good >= 12 else "incomplete"
        ),
        "signed_joint_count": sum(bool(item["signed"]) for item in joints),
        "good_signed_joint_count": signed_good,
        "joints": joints,
        "advisory_only": True,
        "configuration_changed": False,
        "servo_zeros_changed": False,
        "motor_commands_sent": False,
        "next_action": (
            "Review repeated captures and physically verify tag mounts before "
            "using any signed delta to change a servo zero."
        ),
    }


def updated_visual_bias_config(
    config: Mapping[str, Any], report: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, float]]:
    """Return config with stable signed visual residuals accumulated as bias."""
    updated = json.loads(json.dumps(config))
    robot_pose = updated.setdefault("robot_pose", {})
    existing = {
        str(name): float(value)
        for name, value in robot_pose.get("visual_joint_bias_deg", {}).items()
    }
    previous = dict(sorted(existing.items()))
    applied: dict[str, float] = {}
    for joint in report.get("joints", []):
        if not joint.get("signed") or joint.get("quality") != "good":
            continue
        name = str(joint["joint"])
        delta = float(joint["visual_minus_encoder_deg"])
        existing[name] = round(existing.get(name, 0.0) + delta, 6)
        applied[name] = round(delta, 6)
    if len(applied) != 12:
        raise ValueError(
            f"visual calibration needs 12 good signed yaw/hip joints; got {len(applied)}"
        )
    robot_pose["visual_joint_bias_deg"] = dict(sorted(existing.items()))
    updated["visual_calibration"] = {
        "applied_unix": round(time.time(), 3),
        "kind": "visual_joint_bias",
        "source_report": report.get("report_path"),
        "joint_count": len(applied),
        "camera_calibration_approximate": bool(
            report.get("camera_calibration_approximate")
        ),
        "previous_visual_joint_bias_deg": previous,
        "servo_zeros_changed": False,
        "motor_commands_sent": False,
    }
    return updated, applied


def _load_latest_calibration_report(
    report_dir: Path,
) -> dict[str, Any] | None:
    for path in sorted(
        report_dir.glob("visual_calibration_*.json"), reverse=True
    ):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(report, dict):
            report["report_path"] = str(path)
            return report
    return None


class UnavailableSurveyManager:
    """Read-only default used when no robot-specific survey adapter is supplied."""

    def __init__(self, *, robot_url: str | None, vision_runtime: Any) -> None:
        del robot_url, vision_runtime

    def shutdown(self) -> None:
        pass

    def public_state(self) -> dict[str, Any]:
        return {
            "status": "unavailable",
            "error": "no robot motion adapter is installed",
        }

    def start(self, _payload: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("no robot motion adapter is installed")

    def stop(self) -> dict[str, Any]:
        return self.public_state()


class VisionRuntime:
    """Latest-frame camera worker shared by HTTP request threads."""

    def __init__(
        self,
        config_path: Path,
        *,
        camera_index: int = 0,
        camera_cycle: Sequence[int] = (0, 1),
        processing_width: int = 1280,
        target_fps: float = 10.0,
        opencv_threads: int = 4,
        robot_url: str | None = None,
        feedback_hz: float = 3.0,
        report_dir: Path = DEFAULT_REPORT_DIR,
        calibration_frames: int = DEFAULT_CALIBRATION_FRAMES,
        capture_backend: str = "auto",
        capture_width: int = 1920,
        capture_height: int = 1440,
        capture_fps: float = 30.0,
        capture_factory: Any | None = None,
        survey_factory: Any | None = None,
        zero_survey_factory: Any | None = None,
    ) -> None:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.config_path = config_path
        self.tracker = AprilTagPoseTracker(config)
        self.robot_tag_ids = {
            int(value) for value in config.get("robot_pose", {}).get("tags", {})
        }
        self.floor_tag_ids = {int(value) for value in config.get("floor_tags", {})}
        self.feedback = (
            None if robot_url is None
            else FeedbackClient(robot_url, hz=feedback_hz)
        )
        indexes = list(dict.fromkeys(int(value) for value in camera_cycle))
        if camera_index not in indexes:
            indexes.append(camera_index)
        self.camera_indexes = indexes
        self._camera_devices: list[dict[str, Any]] = []
        self._camera_scan_error: str | None = None
        self._camera_scan_unix: float | None = None
        self._camera_discovery_exact = False
        self.processing_width = processing_width
        self.target_fps = float(target_fps)
        if self.target_fps <= 0.0:
            raise ValueError("target_fps must be positive")
        if opencv_threads <= 0:
            raise ValueError("opencv_threads must be positive")
        cv2.setNumThreads(opencv_threads)
        self.report_dir = report_dir
        self.calibration_frames = calibration_frames
        if capture_backend not in {"auto", "avfoundation", "opencv"}:
            raise ValueError(
                "capture_backend must be auto, avfoundation, or opencv"
            )
        if min(capture_width, capture_height) <= 0:
            raise ValueError("capture dimensions must be positive")
        if capture_fps <= 0.0:
            raise ValueError("capture_fps must be positive")
        self.capture_backend = capture_backend
        self.capture_width = int(capture_width)
        self.capture_height = int(capture_height)
        self.capture_fps = float(capture_fps)
        self.capture_factory = capture_factory
        self._lock = threading.RLock()
        self._frame_ready = threading.Condition(self._lock)
        self._shutdown = threading.Event()
        self._camera_enabled = threading.Event()
        self._thread: threading.Thread | None = None
        self._requested_camera_index = camera_index
        self._active_camera_index: int | None = None
        self._camera_status = "off"
        self._camera_error: str | None = None
        self._last_open_error: str | None = None
        self._capture_details: dict[str, Any] = {}
        self._latest_result: dict[str, Any] | None = None
        self._latest_jpeg: bytes | None = None
        self._frame_sequence = 0
        self._frame_times: deque[float] = deque(maxlen=40)
        self._history: deque[dict[str, Any]] = deque(maxlen=90)
        self._started_monotonic = time.monotonic()
        self._latest_report = _load_latest_calibration_report(report_dir)
        self._calibration: dict[str, Any] = {
            "status": "idle",
            "accepted_frames": 0,
            "target_frames": calibration_frames,
            "rejected_frames": 0,
            "report_available": self._latest_report is not None,
        }
        self._calibration_samples: list[dict[str, Any]] = []
        self._calibration_rejections: Counter[str] = Counter()
        self.motion_control_available = survey_factory is not None
        self.gait_survey = (survey_factory or UnavailableSurveyManager)(
            robot_url=robot_url,
            vision_runtime=self,
        )
        self.zero_survey = (zero_survey_factory or ZeroPoseSurveyManager)(
            config_path=config_path,
        )
        self.refresh_camera_devices()

    def start(self) -> None:
        """Start the dormant worker; camera capture remains opt-in."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._shutdown.clear()
        self._thread = threading.Thread(
            target=self._camera_loop, name="vision-camera", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Shut down the worker when the shared web server exits."""
        self.gait_survey.shutdown()
        self.zero_survey.shutdown()
        self._shutdown.set()
        self._camera_enabled.clear()
        with self._frame_ready:
            self._frame_ready.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def enable_camera(self, index: int | None = None) -> dict[str, Any]:
        if index is not None:
            self.switch_camera(index)
        self.start()
        with self._lock:
            self._camera_status = "starting"
            self._camera_error = None
            self._capture_details = {}
        self._camera_enabled.set()
        return self.public_state()

    def disable_camera(self) -> dict[str, Any]:
        self._camera_enabled.clear()
        with self._frame_ready:
            self._active_camera_index = None
            self._camera_status = "off"
            self._camera_error = None
            self._capture_details = {}
            self._latest_result = None
            self._latest_jpeg = None
            self._frame_times.clear()
            self._history.clear()
            if self._calibration["status"] == "collecting":
                self._calibration["status"] = "cancelled"
            self._calibration_samples = []
            self._frame_ready.notify_all()
        return self.public_state()

    def switch_camera(self, index: int) -> dict[str, Any]:
        index = int(index)
        if index < 0 or index > 12:
            raise ValueError("camera index must be between 0 and 12")
        self.refresh_camera_devices()
        with self._lock:
            detected_indexes = {
                int(device["index"])
                for device in self._camera_devices
                if device.get("available", True)
            }
            if self._camera_discovery_exact and index not in detected_indexes:
                count = len(detected_indexes)
                noun = "camera" if count == 1 else "cameras"
                raise ValueError(
                    f"camera {index} is not currently available; macOS reports "
                    f"{count} {noun}. Bring the iPhone near the Mac, lock it, "
                    "then rescan cameras."
                )
            if index not in self.camera_indexes:
                self.camera_indexes.append(index)
                self.camera_indexes.sort()
            self._requested_camera_index = index
            self._camera_status = (
                "switching" if self._camera_enabled.is_set() else "off"
            )
            self._camera_error = None
            self._capture_details = {}
            self._history.clear()
        return self.public_state()

    def refresh_camera_devices(self) -> dict[str, Any]:
        """Rescan camera names without opening or enabling a camera."""
        try:
            if self.capture_factory is not None:
                devices = [
                    {
                        "index": index,
                        "name": f"Camera {index}",
                        "kind": "configured",
                        "available": True,
                    }
                    for index in self.camera_indexes
                ]
                exact = False
            elif sys.platform == "darwin" and self.capture_backend in {
                "auto", "avfoundation"
            }:
                from .avfoundation_capture import AVFoundationYuvCapture

                devices = AVFoundationYuvCapture.device_descriptors()
                exact = True
            else:
                devices = [
                    {
                        "index": index,
                        "name": f"Camera {index}",
                        "kind": "configured",
                        "available": True,
                    }
                    for index in self.camera_indexes
                ]
                exact = False
            error = None
        except Exception as caught:  # discovery must not take down vision
            devices = []
            error = f"camera discovery failed: {caught}"
            exact = False
        with self._lock:
            self._camera_devices = devices
            self._camera_scan_error = error
            self._camera_scan_unix = round(time.time(), 3)
            self._camera_discovery_exact = exact
            if devices:
                discovered_indexes = [int(item["index"]) for item in devices]
                self.camera_indexes = discovered_indexes
                if (
                    not self._camera_enabled.is_set()
                    and self._requested_camera_index not in discovered_indexes
                ):
                    self._requested_camera_index = discovered_indexes[0]
        return self.public_state()

    def start_calibration(self) -> dict[str, Any]:
        with self._lock:
            readiness = self._readiness_locked()
            if not readiness["ready"]:
                raise RuntimeError("; ".join(readiness["blockers"]))
            self._calibration_samples = []
            self._calibration_rejections.clear()
            self._calibration = {
                "status": "collecting",
                "accepted_frames": 0,
                "target_frames": self.calibration_frames,
                "rejected_frames": 0,
                "report_available": self._latest_report is not None,
                "started_unix": round(time.time(), 3),
            }
            return dict(self._calibration)

    def cancel_calibration(self) -> dict[str, Any]:
        with self._lock:
            if self._calibration["status"] == "collecting":
                self._calibration["status"] = "cancelled"
            self._calibration_samples = []
            return dict(self._calibration)

    def latest_report(self) -> dict[str, Any] | None:
        with self._lock:
            return None if self._latest_report is None else dict(self._latest_report)

    def apply_latest_calibration(self) -> dict[str, Any]:
        with self._lock:
            if self._latest_report is None:
                raise RuntimeError("no completed visual calibration report")
            if self._latest_report.get("configuration_changed"):
                raise RuntimeError("this visual calibration is already applied")
            config = json.loads(self.config_path.read_text(encoding="utf-8"))
            updated, applied = updated_visual_bias_config(
                config, self._latest_report
            )
            report_path = self._latest_report.get("report_path")
            if report_path:
                try:
                    source_report = str(
                        Path(report_path).relative_to(self.config_path.parent)
                    )
                except ValueError:
                    source_report = str(report_path)
                updated["visual_calibration"]["source_report"] = source_report
            temporary = self.config_path.with_suffix(
                self.config_path.suffix + ".tmp"
            )
            temporary.write_text(
                json.dumps(updated, indent=2) + "\n", encoding="utf-8"
            )
            temporary.replace(self.config_path)
            self.tracker.visual_joint_bias_deg = {
                str(name): float(value)
                for name, value in updated["robot_pose"][
                    "visual_joint_bias_deg"
                ].items()
            }
            self.tracker.reset_temporal_state()
            self._history.clear()
            self._latest_report["configuration_changed"] = True
            self._latest_report["applied_visual_bias_delta_deg"] = applied
            self._latest_report["servo_zeros_changed"] = False
            self._latest_report["motor_commands_sent"] = False
            if report_path:
                Path(report_path).write_text(
                    json.dumps(self._latest_report, indent=2) + "\n",
                    encoding="utf-8",
                )
            return {
                "ok": True,
                "joint_count": len(applied),
                "applied_visual_bias_delta_deg": applied,
                "config_path": str(self.config_path),
                "camera_calibration_provisional": bool(
                    self._latest_report.get("camera_calibration_approximate")
                ),
                "configuration_changed": True,
                "servo_zeros_changed": False,
                "motor_commands_sent": False,
            }

    def latest_jpeg(self) -> tuple[int, bytes | None]:
        with self._lock:
            return self._frame_sequence, self._latest_jpeg

    def wait_for_jpeg(
        self, after_sequence: int, timeout_s: float = 2.0
    ) -> tuple[int, bytes | None]:
        with self._frame_ready:
            self._frame_ready.wait_for(
                lambda: self._shutdown.is_set()
                or self._frame_sequence > after_sequence,
                timeout=timeout_s,
            )
            return self._frame_sequence, self._latest_jpeg

    @property
    def stopped(self) -> bool:
        return self._shutdown.is_set()

    def _readiness_locked(self) -> dict[str, Any]:
        if not self._camera_enabled.is_set():
            return {
                "ready": False,
                "status": "camera_off",
                "headline": "Camera is off",
                "blockers": ["Turn on the camera to begin pose checks"],
                "warnings": [],
                "stable_frames": 0,
                "required_stable_frames": DEFAULT_STABLE_FRAMES,
                "progress": 0.0,
                "maximum_joint_motion_deg": None,
                "scope": "none",
            }
        return assess_visual_calibration_readiness(
            list(self._history),
            robot_tag_ids=self.robot_tag_ids,
            floor_tag_ids=self.floor_tag_ids,
        )

    def public_state(self) -> dict[str, Any]:
        with self._lock:
            latest = self._latest_result or {}
            direct_robot = _direct_ids(latest, self.robot_tag_ids)
            direct_floor = _direct_ids(latest, self.floor_tag_ids)
            direct_feet = [
                int(item["leg"])
                for item in latest.get("foot_tips", [])
                if item.get("source") == "color"
            ]
            fps = 0.0
            if len(self._frame_times) >= 2:
                elapsed = self._frame_times[-1] - self._frame_times[0]
                if elapsed > 0:
                    fps = (len(self._frame_times) - 1) / elapsed
            full_pose = latest.get("full_pose") or {}
            joints = [
                {"joint": name, **record}
                for name, record in full_pose.get("joints", {}).items()
            ]
            return {
                "ok": self._latest_result is not None,
                "service": "hexapod-vision",
                "joint_frame": FRAME_ROBOT_ABS,
                "joint_contract": JOINT_CONTRACT,
                "uptime_s": round(time.monotonic() - self._started_monotonic, 2),
                "camera": {
                    "enabled": self._camera_enabled.is_set(),
                    "active_index": self._active_camera_index,
                    "requested_index": self._requested_camera_index,
                    "indexes": list(self.camera_indexes),
                    "status": self._camera_status,
                    "error": self._camera_error,
                    "backend": self._capture_details.get("backend"),
                    "pixel_format": self._capture_details.get("pixel_format"),
                    "native_luma": bool(
                        self._capture_details.get("native_luma", False)
                    ),
                    "capture_fps": self._capture_details.get("capture_fps"),
                    "devices": [dict(item) for item in self._camera_devices],
                    "scan_error": self._camera_scan_error,
                    "scan_unix": self._camera_scan_unix,
                    "discovery_exact": self._camera_discovery_exact,
                },
                "performance": {
                    "fps": round(fps, 1),
                    "frame_sequence": self._frame_sequence,
                    "frame_age_ms": (
                        None if not self._frame_times else round(
                            1000.0 * (time.monotonic() - self._frame_times[-1]), 1
                        )
                    ),
                    "processing_width": self.processing_width,
                    "target_fps": self.target_fps,
                    "image_size_px": latest.get("image_size_px"),
                    "capture_image_size_px": latest.get("capture_image_size_px"),
                    "detection_image_size_px": latest.get(
                        "detection_image_size_px"
                    ),
                },
                "coverage": {
                    "robot_tags": len(direct_robot),
                    "robot_tag_ids": sorted(direct_robot),
                    "robot_tags_required": len(self.robot_tag_ids),
                    "floor_tags": len(direct_floor),
                    "floor_tag_ids": sorted(direct_floor),
                    "floor_tags_available": len(self.floor_tag_ids),
                    "feet": len(direct_feet),
                    "foot_legs": sorted(direct_feet),
                },
                "readiness": self._readiness_locked(),
                "calibration": dict(self._calibration),
                "pose": {
                    "image_size_px": latest.get("image_size_px"),
                    "tags": latest.get("detections", []),
                    "feet": latest.get("foot_tips", []),
                    "joints": joints,
                    "safety": latest.get("safety_assessment"),
                    "zero_check": full_pose.get("zero_check"),
                    "body_tilt_deg": (
                        full_pose.get("walking_check") or {}
                    ).get("body_tilt_deg"),
                    "pose_reference": latest.get("pose_reference"),
                },
                # The vision worker already obtains this rate-limited,
                # read-only sample for encoder branch selection.  Exposing it
                # lets sysid capture camera pose and IMU roll/pitch on the
                # same vision-frame timeline without adding another robot
                # bus poller during motion.
                "feedback": latest.get("encoder_feedback"),
                "survey": self.gait_survey.public_state(),
                "zero_survey": self.zero_survey.public_state(),
                "read_only": not self.motion_control_available,
                "motion_control_scope": (
                    "acknowledged_guarded_gait_survey"
                    if self.motion_control_available else "none"
                ),
            }

    @staticmethod
    def _validated_capture(candidate: Any) -> Any | None:
        if not candidate.isOpened():
            candidate.release()
            return None
        ok, _frame = candidate.read()
        if not ok:
            candidate.release()
            return None
        return candidate

    def _opencv_capture(self, index: int) -> Any:
        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        candidate = cv2.VideoCapture(index, backend)
        candidate.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.capture_width))
        candidate.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.capture_height))
        candidate.set(cv2.CAP_PROP_FPS, self.capture_fps)
        candidate.set(cv2.CAP_PROP_BUFFERSIZE, 1.0)
        return candidate

    def _open_camera(self, index: int) -> Any | None:
        if self.capture_factory is not None:
            return self._validated_capture(self.capture_factory(index))

        if self.capture_backend in {"auto", "avfoundation"}:
            from .avfoundation_capture import AVFoundationYuvCapture

            preferred = [(self.capture_width, self.capture_height)]
            for fallback in ((1920, 1080), (1280, 720)):
                if fallback not in preferred:
                    preferred.append(fallback)
            native = AVFoundationYuvCapture(
                index,
                preferred_sizes=preferred,
                fps=self.capture_fps,
                processing_width=self.processing_width,
            )
            opened = self._validated_capture(native)
            if opened is not None:
                self._last_open_error = None
                return opened
            self._last_open_error = native.last_error
            if self.capture_backend == "avfoundation":
                return None

        opened = self._validated_capture(self._opencv_capture(index))
        if opened is None and self._last_open_error is None:
            self._last_open_error = "OpenCV produced no frame"
        return opened

    @staticmethod
    def _capture_frame_details(
        capture: Any, frame: Any
    ) -> tuple[Any, Any | None, Any | None, dict[str, Any]]:
        if getattr(capture, "provides_native_luma", False):
            processed = frame
            detection_gray = getattr(capture, "detection_gray", None)
            tracking_gray = getattr(capture, "tracking_gray", None)
            details = dict(capture.capture_info())
            return processed, detection_gray, tracking_gray, details

        processed = frame
        backend_name = "opencv"
        try:
            raw_backend = capture.getBackendName()
            if raw_backend:
                backend_name = f"opencv-{str(raw_backend).lower()}"
        except (AttributeError, cv2.error):
            pass
        details = {
            "backend": backend_name,
            "pixel_format": "BGR",
            "native_luma": False,
            "capture_fps": None,
            "capture_image_size_px": [frame.shape[1], frame.shape[0]],
            "detection_image_size_px": None,
        }
        return processed, None, None, details

    def _set_camera_failure(self, message: str) -> None:
        with self._lock:
            self._camera_status = "error"
            self._camera_error = message
            self._active_camera_index = None
            self._capture_details = {}

    def _record_calibration_frame_locked(self, result: dict[str, Any]) -> None:
        if self._calibration["status"] != "collecting":
            return
        facts = _frame_calibration_facts(
            result,
            robot_tag_ids=self.robot_tag_ids,
            floor_tag_ids=self.floor_tag_ids,
        )
        if not facts["eligible"]:
            reason = facts["blockers"][0] if facts["blockers"] else "frame rejected"
            self._calibration_rejections[reason] += 1
            self._calibration["rejected_frames"] += 1
            self._calibration["last_rejection"] = reason
            return
        self._calibration_samples.append(result)
        self._calibration["accepted_frames"] = len(self._calibration_samples)
        self._calibration["progress"] = round(
            len(self._calibration_samples) / self.calibration_frames, 3
        )
        if len(self._calibration_samples) < self.calibration_frames:
            return
        report = build_visual_calibration_report(
            self._calibration_samples,
            camera_index=int(self._active_camera_index or 0),
            config_path=self.config_path,
        )
        report["rejected_frames"] = int(self._calibration["rejected_frames"])
        report["rejection_reasons"] = dict(self._calibration_rejections)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.report_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.report_dir / f"visual_calibration_{stamp}.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        report["report_path"] = str(report_path)
        self._latest_report = report
        self._calibration = {
            **self._calibration,
            "status": "complete",
            "progress": 1.0,
            "completed_unix": round(time.time(), 3),
            "report_available": True,
            "report_path": str(report_path),
            "quality": report["quality"],
        }
        self._calibration_samples = []

    def _camera_loop(self) -> None:
        capture = None
        active: int | None = None
        frame_index = 0
        start = time.monotonic()
        try:
            while not self._shutdown.is_set():
                iteration_started = time.monotonic()
                if not self._camera_enabled.is_set():
                    if capture is not None:
                        capture.release()
                        capture = None
                    active = None
                    with self._lock:
                        self._active_camera_index = None
                        self._camera_status = "off"
                        self._capture_details = {}
                    self._shutdown.wait(0.1)
                    continue
                with self._lock:
                    requested = self._requested_camera_index
                if capture is None or requested != active:
                    previous = active
                    if capture is not None:
                        capture.release()
                    capture = self._open_camera(requested)
                    if capture is None:
                        message = self._last_open_error or (
                            f"camera {requested} opened but produced no frames"
                        )
                        if previous is not None and previous != requested:
                            fallback = self._open_camera(previous)
                            if fallback is not None:
                                capture = fallback
                                active = previous
                                with self._lock:
                                    self._requested_camera_index = previous
                                    self._active_camera_index = previous
                                    self._camera_status = "running"
                                    self._camera_error = (
                                        message + f"; returned to camera {previous}"
                                    )
                                continue
                        self._set_camera_failure(message)
                        active = None
                        self._shutdown.wait(0.5)
                        continue
                    active = requested
                    frame_index = 0
                    start = time.monotonic()
                    self.tracker.reset_temporal_state()
                    with self._lock:
                        self._active_camera_index = active
                        self._camera_status = "running"
                        self._camera_error = None
                        self._history.clear()

                ok, frame = capture.read()
                if not ok:
                    capture.release()
                    capture = None
                    self._set_camera_failure(
                        f"camera {active} stopped producing frames; retrying"
                    )
                    self._shutdown.wait(0.15)
                    continue
                (
                    processed,
                    detection_gray,
                    tracking_gray,
                    capture_details,
                ) = self._capture_frame_details(capture, frame)
                if not capture_details["native_luma"]:
                    processed = _resize_for_processing(
                        processed, self.processing_width
                    )
                    capture_details["detection_image_size_px"] = [
                        processed.shape[1], processed.shape[0]
                    ]
                encoder, feedback_status = ({}, {"configured": False})
                if self.feedback is not None:
                    encoder, feedback_status = self.feedback.sample()
                try:
                    result, _ = self.tracker.process_frame(
                        processed,
                        frame_index=frame_index,
                        time_s=time.monotonic() - start,
                        encoder_joint_deg=encoder or None,
                        render_overlay=False,
                        detection_gray=detection_gray,
                        tracking_gray=tracking_gray,
                    )
                except (ValueError, cv2.error) as error:
                    with self._lock:
                        self._camera_error = f"frame processing failed: {error}"
                    continue
                result["encoder_feedback"] = feedback_status
                result["safety_assessment"] = _safe_pose_assessment(
                    result, feedback_status, operator_supported=False
                )
                result["camera_index"] = active
                result["capture_backend"] = capture_details["backend"]
                result["capture_pixel_format"] = capture_details["pixel_format"]
                result["capture_image_size_px"] = capture_details[
                    "capture_image_size_px"
                ]
                encoded_ok, jpeg = cv2.imencode(
                    ".jpg", processed, [cv2.IMWRITE_JPEG_QUALITY, 82]
                )
                if not encoded_ok:
                    continue
                now = time.monotonic()
                with self._frame_ready:
                    # A switch can be requested while the old camera's final
                    # frame is being processed.  Never publish that stale
                    # frame as "running" for the new request.
                    if self._requested_camera_index != active:
                        self._camera_status = "switching"
                        continue
                    self._latest_result = result
                    self._latest_jpeg = jpeg.tobytes()
                    self._capture_details = capture_details
                    self._frame_sequence += 1
                    self._frame_times.append(now)
                    self._history.append(result)
                    self._record_calibration_frame_locked(result)
                    self._camera_status = "running"
                    self._frame_ready.notify_all()
                frame_index += 1
                remaining = 1.0 / self.target_fps - (
                    time.monotonic() - iteration_started
                )
                if remaining > 0.0:
                    self._shutdown.wait(remaining)
        finally:
            if capture is not None:
                capture.release()
            with self._lock:
                self._camera_status = "stopped"


def wrap_handler_with_vision(
    base_handler: type,
    runtime: VisionRuntime,
    ui_dir: Path = DEFAULT_UI_DIR,
) -> type:
    """Add local `/vision` and `/api/vision/*` routes to another handler."""

    class Handler(base_handler):
        def _vision_send_bytes(
            self,
            code: int,
            body: bytes,
            content_type: str,
            *,
            cache: str = "no-store",
        ) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            self.end_headers()
            try:
                self.wfile.write(body)
            except OSError:
                pass

        def _vision_json(self, code: int, payload: Any) -> None:
            self._vision_send_bytes(
                code,
                json.dumps(payload, separators=(",", ":")).encode("utf-8"),
                "application/json; charset=utf-8",
            )

        def _vision_read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            value = json.loads(raw.decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError("request body must be an object")
            return value

        def _vision_stream_mjpeg(self) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            sequence = -1
            try:
                while not runtime.stopped:
                    next_sequence, jpeg = runtime.wait_for_jpeg(sequence)
                    if jpeg is None or next_sequence == sequence:
                        continue
                    sequence = next_sequence
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                        + jpeg
                        + b"\r\n"
                    )
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

        def _vision_serve_static(self, path: str) -> None:
            if not ui_dir.is_dir():
                message = (
                    "Vision UI has not been built. Run `make web-build` "
                    "from the hexapod-tracker repository."
                )
                self._vision_send_bytes(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    message.encode("utf-8"),
                    "text/plain; charset=utf-8",
                )
                return
            relative = (
                "index.html"
                if path in {"/vision", "/vision/"}
                else path.removeprefix("/vision/")
            )
            candidate = (ui_dir / relative).resolve()
            root = ui_dir.resolve()
            if root not in candidate.parents and candidate != root:
                self._vision_json(
                    HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"}
                )
                return
            if not candidate.is_file():
                candidate = ui_dir / "index.html"
            suffix_types = {
                ".html": "text/html; charset=utf-8",
                ".js": "application/javascript; charset=utf-8",
                ".css": "text/css; charset=utf-8",
                ".svg": "image/svg+xml",
                ".png": "image/png",
            }
            content_type = suffix_types.get(
                candidate.suffix.lower(), "application/octet-stream"
            )
            cache = "max-age=31536000, immutable" if "/assets/" in path else "no-cache"
            self._vision_send_bytes(
                HTTPStatus.OK, candidate.read_bytes(), content_type, cache=cache
            )

        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            if path == "/api/vision/health":
                self._vision_json(HTTPStatus.OK, {
                    "ok": True,
                    "service": "hexapod-vision",
                    "joint_frame": FRAME_ROBOT_ABS,
                    "joint_contract": JOINT_CONTRACT,
                    "read_only": not runtime.motion_control_available,
                    "motion_control_scope": (
                        "acknowledged_guarded_gait_survey"
                        if runtime.motion_control_available else "none"
                    ),
                })
            elif path == "/api/vision/state":
                self._vision_json(HTTPStatus.OK, runtime.public_state())
            elif path == "/api/vision/frame.mjpg":
                self._vision_stream_mjpeg()
            elif path == "/api/vision/frame.jpg":
                _sequence, jpeg = runtime.latest_jpeg()
                if jpeg is None:
                    self._vision_json(
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        {"ok": False, "error": "no frame available"},
                    )
                else:
                    self._vision_send_bytes(HTTPStatus.OK, jpeg, "image/jpeg")
            elif path == "/api/vision/calibration/report":
                report = runtime.latest_report()
                if report is None:
                    self._vision_json(
                        HTTPStatus.NOT_FOUND,
                        {"ok": False, "error": "no calibration report yet"},
                    )
                else:
                    self._vision_json(HTTPStatus.OK, report)
            elif path == "/api/vision/zero-survey/frame.jpg":
                jpeg = runtime.zero_survey.latest_camera_jpeg()
                if jpeg is None:
                    self._vision_json(
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        {"ok": False, "error": "no Record3D frame available"},
                    )
                else:
                    self._vision_send_bytes(HTTPStatus.OK, jpeg, "image/jpeg")
            elif path == "/api/vision/zero-survey/result":
                result = runtime.zero_survey.result()
                if result is None:
                    self._vision_json(
                        HTTPStatus.NOT_FOUND,
                        {"ok": False, "error": "no zero-pose survey result yet"},
                    )
                else:
                    self._vision_json(HTTPStatus.OK, result)
            elif path.startswith("/api/vision/"):
                self._vision_json(
                    HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"}
                )
            elif path in {"/vision", "/vision/"} or path.startswith(
                "/vision/"
            ):
                self._vision_serve_static(path)
            else:
                super().do_GET()

        def do_POST(self) -> None:
            path = urlsplit(self.path).path
            if not path.startswith("/api/vision/"):
                super().do_POST()
                return
            try:
                if path == "/api/vision/camera":
                    body = self._vision_read_json()
                    self._vision_json(
                        HTTPStatus.ACCEPTED,
                        runtime.switch_camera(int(body["index"])),
                    )
                elif path == "/api/vision/cameras/rescan":
                    self._vision_read_json()
                    self._vision_json(
                        HTTPStatus.OK, runtime.refresh_camera_devices()
                    )
                elif path == "/api/vision/camera/start":
                    body = self._vision_read_json()
                    index = body.get("index")
                    self._vision_json(
                        HTTPStatus.ACCEPTED,
                        runtime.enable_camera(
                            None if index is None else int(index)
                        ),
                    )
                elif path == "/api/vision/camera/stop":
                    self._vision_read_json()
                    self._vision_json(
                        HTTPStatus.OK, runtime.disable_camera()
                    )
                elif path == "/api/vision/calibration/start":
                    self._vision_read_json()
                    self._vision_json(
                        HTTPStatus.ACCEPTED, runtime.start_calibration()
                    )
                elif path == "/api/vision/calibration/cancel":
                    self._vision_read_json()
                    self._vision_json(
                        HTTPStatus.OK, runtime.cancel_calibration()
                    )
                elif path == "/api/vision/calibration/apply":
                    self._vision_read_json()
                    self._vision_json(
                        HTTPStatus.OK, runtime.apply_latest_calibration()
                    )
                elif path == "/api/vision/survey/start":
                    body = self._vision_read_json()
                    self._vision_json(
                        HTTPStatus.ACCEPTED,
                        runtime.gait_survey.start(body),
                    )
                elif path == "/api/vision/survey/stop":
                    self._vision_read_json()
                    self._vision_json(
                        HTTPStatus.ACCEPTED,
                        runtime.gait_survey.stop(),
                    )
                elif path == "/api/vision/zero-survey/start":
                    body = self._vision_read_json()
                    if runtime.public_state()["camera"]["enabled"]:
                        runtime.disable_camera()
                    self._vision_json(
                        HTTPStatus.ACCEPTED,
                        runtime.zero_survey.start(body),
                    )
                elif path == "/api/vision/zero-survey/stop":
                    self._vision_read_json()
                    self._vision_json(
                        HTTPStatus.ACCEPTED,
                        runtime.zero_survey.stop(),
                    )
                elif path == "/api/vision/zero-survey/save":
                    body = self._vision_read_json()
                    self._vision_json(
                        HTTPStatus.OK,
                        runtime.zero_survey.save_reviewed_config(body),
                    )
                elif path == "/api/vision/zero-survey/publish":
                    self._vision_read_json()
                    self._vision_json(
                        HTTPStatus.OK,
                        runtime.zero_survey.publish_to_robot_lab(),
                    )
                else:
                    self._vision_json(
                        HTTPStatus.NOT_FOUND,
                        {"ok": False, "error": "not found"},
                    )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                self._vision_json(
                    HTTPStatus.BAD_REQUEST,
                    {"ok": False, "error": str(error)},
                )
            except RuntimeError as error:
                self._vision_json(
                    HTTPStatus.CONFLICT,
                    {"ok": False, "error": str(error)},
                )

    return Handler
