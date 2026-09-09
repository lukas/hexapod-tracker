#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "opencv-contrib-python>=4.8,<6",
#   "numpy>=1.24",
# ]
# ///
"""Serve multiple annotated AprilTag camera feeds to a local web browser.

This is a camera-only diagnostic: it never connects to or moves the robot.
Each capture explicitly requests MJPG input and reports whether the selected
OpenCV backend accepted that request.  Browser output is always an MJPEG HTTP
stream, independently of the USB input format selected by macOS.
"""

from __future__ import annotations

import argparse
import json
import signal
import socket
import struct
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

import cv2
import numpy as np

from .housing_pose import JOINT_NAMES
from .joint_contract import FRAME_ROBOT_ABS, JOINT_CONTRACT
from .paths import CONFIG_DIR
from .planar_pose import PlanarPoseEstimator


# Robot Lab sets an intent; this server decides how to observe under it.
#
# TRACK keeps every camera on and never switches, because a camera that adds
# nothing while the robot stands still may be the only one holding it as it
# walks out of another view, and re-opening a camera costs about two seconds
# of blindness on that view.
# SURVEY permits coverage-driven arbitration, which is only safe while the
# scene is static.
# Identifies this process run. A browser holds one long-lived
# multipart/x-mixed-replace connection per camera, and those never recover on
# their own once the server they came from is gone: the page keeps showing the
# last frame it received while the JSON polling continues to look healthy.
# Publishing an id lets the page notice a restart and re-attach.
SERVER_RUN_ID = uuid.uuid4().hex

OBSERVATION_MODE_TRACK = "track"
OBSERVATION_MODE_SURVEY = "survey"
OBSERVATION_MODES = (OBSERVATION_MODE_TRACK, OBSERVATION_MODE_SURVEY)

# A feed older than this is reported unhealthy. Well above the ~0.1 s seen on
# a healthy 10 fps camera, and well below the multi-second gaps that marked
# the starved ones.
STALE_FRAME_AGE_S = 2.0


def decode_fourcc(value: float) -> str | None:
    """Convert OpenCV's numeric FourCC to readable text when available."""
    number = int(value)
    if number <= 0 or number == 0xFFFFFFFF:
        return None
    chars = bytes((number >> (8 * i)) & 0xFF for i in range(4))
    if not all(32 <= char < 127 for char in chars):
        return None
    return chars.decode("ascii")


def make_tag_detector() -> cv2.aruco.ArucoDetector:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(dictionary, parameters)


def enhance_for_tag_detection(gray: np.ndarray) -> np.ndarray:
    """Upscale and sharpen small tags without changing the displayed frame."""
    enlarged = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(enlarged, (0, 0), 1.1)
    return cv2.addWeighted(enlarged, 1.8, blurred, -0.8, 0)


def detect_tag_corners(
    gray: np.ndarray,
    detector: cv2.aruco.ArucoDetector,
) -> dict[int, np.ndarray]:
    """Fuse native-resolution and 2x detections, preferring native corners."""
    detections: dict[int, np.ndarray] = {}
    for image, scale in ((gray, 1.0), (enhance_for_tag_detection(gray), 2.0)):
        corners, ids, _rejected = detector.detectMarkers(image)
        if ids is None:
            continue
        for corner, raw_id in zip(corners, ids.flatten(), strict=True):
            tag_id = int(raw_id)
            if tag_id not in detections:
                detections[tag_id] = corner[0].astype(np.float32) / scale

    return detections


def detect_tags(
    gray: np.ndarray,
    detector: cv2.aruco.ArucoDetector,
) -> tuple[list[np.ndarray], np.ndarray | None]:
    """Return OpenCV-compatible corners and IDs for the fused detections."""
    detections = detect_tag_corners(gray, detector)

    if not detections:
        return [], None
    tag_ids = sorted(detections)
    corners = [detections[tag_id][None, :, :] for tag_id in tag_ids]
    ids = np.asarray(tag_ids, dtype=np.int32).reshape(-1, 1)
    return corners, ids


def annotate_tags(
    frame: np.ndarray,
    detector: cv2.aruco.ArucoDetector,
    camera_index: int,
) -> tuple[np.ndarray, list[int]]:
    """Draw tag corners and IDs; return the annotated frame and sorted IDs."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids = detect_tags(gray, detector)
    tag_ids: list[int] = []
    if ids is not None:
        tag_ids = sorted(int(tag_id) for tag_id in ids.flatten())
        cv2.aruco.drawDetectedMarkers(frame, corners, ids, (0, 255, 0))
    label = f"camera {camera_index} | tags: {tag_ids if tag_ids else 'none'}"
    cv2.rectangle(frame, (0, 0), (min(frame.shape[1], 760), 42), (0, 0, 0), -1)
    cv2.putText(
        frame,
        label,
        (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 255, 0) if tag_ids else (0, 190, 255),
        2,
        cv2.LINE_AA,
    )
    return frame, tag_ids


def annotate_tag_corners(
    frame: np.ndarray,
    detections: dict[int, np.ndarray],
    camera_index: int,
) -> tuple[np.ndarray, list[int]]:
    """Annotate an already-detected set so pose and display use identical corners."""
    tag_ids = sorted(detections)
    if tag_ids:
        corners = [detections[tag_id][None, :, :] for tag_id in tag_ids]
        ids = np.asarray(tag_ids, dtype=np.int32).reshape(-1, 1)
        cv2.aruco.drawDetectedMarkers(frame, corners, ids, (0, 255, 0))
    label = f"camera {camera_index} | tags: {tag_ids if tag_ids else 'none'}"
    cv2.rectangle(frame, (0, 0), (min(frame.shape[1], 760), 42), (0, 0, 0), -1)
    cv2.putText(
        frame,
        label,
        (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 255, 0) if tag_ids else (0, 190, 255),
        2,
        cv2.LINE_AA,
    )
    return frame, tag_ids


def favicon_ico(size: int = 32) -> bytes:
    """Build a tag-like favicon as a PNG wrapped in an ICO container.

    Drawn rather than shipped as a binary asset so it survives the
    source-checkout layout this package assumes. ICO with a PNG payload is
    what browsers, including Safari, expect at /favicon.ico.
    """

    cells = 8
    glyph = np.zeros((cells, cells), dtype=np.uint8)
    glyph[1:-1, 1:-1] = 255
    # An off-centre interior keeps it recognisable as a tag rather than a
    # checkerboard once scaled down to 16 px.
    for row, column in ((2, 2), (2, 4), (3, 5), (4, 2), (5, 4), (5, 5)):
        glyph[row, column] = 0
    image = cv2.resize(glyph, (size, size), interpolation=cv2.INTER_NEAREST)
    rgba = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    rgba[:, :, 3] = 255
    ok, encoded = cv2.imencode(".png", rgba)
    if not ok:
        raise RuntimeError("OpenCV could not encode the favicon PNG")
    payload = encoded.tobytes()
    header = struct.pack("<HHH", 0, 1, 1)
    entry = struct.pack(
        "<BBBBHHII",
        size if size < 256 else 0,
        size if size < 256 else 0,
        0,
        0,
        1,
        32,
        len(payload),
        6 + 16,
    )
    return header + entry + payload


_FAVICON_CACHE: bytes | None = None


def favicon_bytes() -> bytes:
    global _FAVICON_CACHE
    if _FAVICON_CACHE is None:
        _FAVICON_CACHE = favicon_ico()
    return _FAVICON_CACHE


def downscale_preview(frame: np.ndarray, max_width: int) -> np.ndarray:
    """Shrink a frame for the browser preview, never enlarging it.

    Only the MJPEG preview needs to be small. Tag detection runs on the
    full-resolution luma plane and pose uses full-frame corners, so this costs
    no accuracy -- and at full size three 1280x720 feeds pushed about 46 Mbps,
    which Safari cannot decode smoothly.
    """

    if max_width <= 0 or frame.shape[1] <= max_width:
        return frame
    scale = max_width / frame.shape[1]
    height = max(1, round(frame.shape[0] * scale))
    return cv2.resize(frame, (max_width, height), interpolation=cv2.INTER_AREA)


def placeholder_jpeg(index: int, message: str) -> bytes:
    frame = np.zeros((400, 640, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        f"camera {index}",
        (22, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        message[:72],
        (22, 102),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 190, 255),
        2,
        cv2.LINE_AA,
    )
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("OpenCV could not encode placeholder JPEG")
    return encoded.tobytes()


@dataclass
class CameraStatus:
    index: int
    device_name: str | None = None
    requested_stable_id: str | None = None
    backend: str = "AVFOUNDATION"
    state: str = "starting"
    requested_fourcc: str = "MJPG"
    mjpg_request_accepted: bool | None = None
    reported_fourcc: str | None = None
    requested_width: int = 1280
    requested_height: int = 800
    requested_fps: float = 30.0
    output_fps: float = 10.0
    rotation_degrees: int = 0
    reported_width: int = 0
    reported_height: int = 0
    reported_fps: float = 0.0
    native_capture_width: int | None = None
    native_capture_height: int | None = None
    native_luma_available: bool = False
    measured_fps: float = 0.0
    frames: int = 0
    consecutive_failures: int = 0
    reconnects: int = 0
    tag_ids: list[int] = field(default_factory=list)
    last_frame_age_s: float | None = None
    error: str | None = None


class CameraWorker:
    def __init__(
        self,
        index: int,
        width: int,
        height: int,
        fps: float,
        output_fps: float,
        jpeg_quality: int,
        rotate_180: bool = False,
        native_avfoundation: bool = False,
        stable_id: str | None = None,
        preview_max_width: int = 0,
    ):
        self.index = index
        self.stable_id = stable_id
        # The browser preview is the only consumer that needs to be small.
        # Detection runs on the full-resolution luma plane and pose uses
        # full-frame corners, so shrinking this costs no accuracy.
        self.preview_max_width = int(preview_max_width)
        self.width = width
        self.height = height
        self.fps = fps
        self.output_fps = output_fps
        self.jpeg_quality = jpeg_quality
        self.rotate_180 = rotate_180
        self.native_avfoundation = native_avfoundation
        device_name = None
        if native_avfoundation:
            try:
                from .avfoundation_capture import AVFoundationYuvCapture

                descriptors = AVFoundationYuvCapture.device_descriptors()
                if stable_id:
                    matches = (
                        str(item["name"])
                        for item in descriptors
                        if str(item["stable_id"]) == stable_id
                    )
                else:
                    matches = (
                        str(item["name"])
                        for item in descriptors
                        if int(item["index"]) == index
                    )
                device_name = next(matches, None)
            except Exception:
                device_name = None
        self.status = CameraStatus(
            index=index,
            device_name=device_name,
            requested_stable_id=stable_id,
            backend="AVFOUNDATION_NATIVE" if native_avfoundation else "AVFOUNDATION",
            requested_fourcc="420v" if native_avfoundation else "MJPG",
            requested_width=width,
            requested_height=height,
            requested_fps=fps,
            output_fps=output_fps,
            rotation_degrees=180 if rotate_180 else 0,
        )
        self._jpeg = placeholder_jpeg(index, "waiting for frames")
        self._raw_jpeg = self._jpeg
        self._native_planes: tuple[np.ndarray, np.ndarray] | None = None
        self._tag_corners: dict[int, np.ndarray] = {}
        self._last_frame_at: float | None = None
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"camera-{index}", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)

    def snapshot(self) -> tuple[bytes, dict[str, Any]]:
        with self._condition:
            status = asdict(self.status)
            if self._last_frame_at is not None:
                status["last_frame_age_s"] = round(time.monotonic() - self._last_frame_at, 3)
            return self._jpeg, status

    def raw_snapshot(self) -> bytes:
        with self._condition:
            return self._raw_jpeg

    def native_luma_snapshot(self) -> tuple[bytes, int, int] | None:
        """Encode the unscaled native luminance plane as a lossless PNG."""
        with self._condition:
            planes = self._native_planes
        if planes is None:
            return None
        y = planes[0]
        ok, encoded = cv2.imencode(
            ".png", y, [cv2.IMWRITE_PNG_COMPRESSION, 3]
        )
        if not ok:
            return None
        return encoded.tobytes(), int(y.shape[1]), int(y.shape[0])

    def native_nv12_snapshot(self) -> tuple[bytes, int, int] | None:
        """Pack the exact unscaled AVFoundation Y and UV planes as NV12."""
        with self._condition:
            planes = self._native_planes
        if planes is None:
            return None
        y, uv = planes
        height, width = y.shape
        return y.tobytes() + uv.tobytes(), int(width), int(height)

    def pose_snapshot(self) -> dict[str, Any]:
        """Return one coherent set of corners and capture metadata for fusion."""
        with self._condition:
            age = None
            if self._last_frame_at is not None:
                age = time.monotonic() - self._last_frame_at
            return {
                "index": self.index,
                "width": self.status.reported_width or self.width,
                "height": self.status.reported_height or self.height,
                "frame_age_s": age,
                "tags": {
                    tag_id: corners.copy() for tag_id, corners in self._tag_corners.items()
                },
            }

    def wait_for_frame(
        self,
        previous_frames: int,
        timeout: float = 1.0,
        raw: bool = False,
    ) -> tuple[bytes, int]:
        with self._condition:
            if self.status.frames == previous_frames:
                self._condition.wait(timeout=timeout)
            jpeg = self._raw_jpeg if raw else self._jpeg
            return jpeg, self.status.frames

    def _publish(
        self,
        raw_jpeg: bytes,
        jpeg: bytes,
        tag_ids: list[int],
        tag_corners: dict[int, np.ndarray],
        native_planes: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        with self._condition:
            self._raw_jpeg = raw_jpeg
            self._jpeg = jpeg
            self.status.frames += 1
            self.status.tag_ids = tag_ids
            self._tag_corners = {
                tag_id: corners.copy() for tag_id, corners in tag_corners.items()
            }
            self._native_planes = native_planes
            self.status.native_luma_available = native_planes is not None
            if native_planes is not None:
                self.status.native_capture_width = int(native_planes[0].shape[1])
                self.status.native_capture_height = int(native_planes[0].shape[0])
            self.status.consecutive_failures = 0
            self.status.state = "streaming"
            self.status.error = None
            self._last_frame_at = time.monotonic()
            self._condition.notify_all()

    def _set_waiting(self, message: str, state: str = "waiting") -> None:
        with self._condition:
            self.status.state = state
            self.status.error = message
            self._jpeg = placeholder_jpeg(self.index, message)
            self._condition.notify_all()

    def _open(self) -> Any | None:
        if self.native_avfoundation:
            from .avfoundation_capture import AVFoundationYuvCapture

            cap = AVFoundationYuvCapture(
                self.index,
                stable_id=self.stable_id,
                preferred_sizes=((1920, 1440), (1920, 1080), (1280, 720)),
                fps=self.fps,
                processing_width=self.width,
            )
            if not cap.isOpened():
                self._set_waiting("could not open native AVFoundation device", "open_failed")
                cap.release()
                return None
            self.status.mjpg_request_accepted = None
            self.status.reported_fourcc = "420v"
            self.status.reported_fps = self.fps
            self.status.state = "opened"
            return cap

        cap = cv2.VideoCapture(self.index, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            self._set_waiting("could not open device", "open_failed")
            cap.release()
            return None

        mjpg = cv2.VideoWriter_fourcc(*"MJPG")
        self.status.mjpg_request_accepted = bool(cap.set(cv2.CAP_PROP_FOURCC, mjpg))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.status.reported_fourcc = decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))
        self.status.reported_width = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.status.reported_height = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.status.reported_fps = round(cap.get(cv2.CAP_PROP_FPS), 3)
        self.status.state = "opened"
        return cap

    def _run(self) -> None:
        detector = make_tag_detector()
        while not self._stop.is_set():
            cap = self._open()
            if cap is None:
                self._stop.wait(1.0)
                self.status.reconnects += 1
                continue

            sample_started = time.monotonic()
            sample_frames = 0
            next_output_at = 0.0
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    self.status.consecutive_failures += 1
                    if self.status.consecutive_failures >= 3:
                        # Keep the backend's own reason. A mistyped --device-id
                        # otherwise reads as a flaky camera rather than a
                        # camera that was never there.
                        reason = getattr(cap, "last_error", None)
                        self._set_waiting(
                            f"capture stalled; reconnecting ({reason})"
                            if reason
                            else "capture stalled; reconnecting",
                            "stalled",
                        )
                        break
                    continue

                self.status.reported_width = int(frame.shape[1])
                self.status.reported_height = int(frame.shape[0])

                sample_frames += 1
                now = time.monotonic()
                if now < next_output_at:
                    continue
                next_output_at = now + 1.0 / self.output_fps

                if self.rotate_180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)

                raw_ok, raw_encoded = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95]
                )
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                tag_corners = detect_tag_corners(gray, detector)
                annotated, tag_ids = annotate_tag_corners(frame, tag_corners, self.index)
                # Downscale after annotating so the overlay keeps its
                # proportions, and only for the browser copy: /snapshot and
                # /raw-stream stay at full processing resolution.
                preview = downscale_preview(annotated, self.preview_max_width)
                ok, encoded = cv2.imencode(
                    ".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                )
                if raw_ok and ok:
                    native_planes = (
                        cap.native_planes()
                        if self.native_avfoundation
                        and hasattr(cap, "native_planes")
                        else None
                    )
                    self._publish(
                        raw_encoded.tobytes(),
                        encoded.tobytes(),
                        tag_ids,
                        tag_corners,
                        native_planes,
                    )

                elapsed = time.monotonic() - sample_started
                if elapsed >= 2.0:
                    self.status.measured_fps = round(sample_frames / elapsed, 2)
                    sample_started = time.monotonic()
                    sample_frames = 0

            cap.release()
            if not self._stop.is_set():
                self.status.reconnects += 1
                self._stop.wait(1.0)


INDEX_HTML = """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hexapod AprilTag cameras</title>
<link rel="icon" href="/favicon.ico" sizes="any">
<style>
  :root { color-scheme: dark; font-family: system-ui, sans-serif; }
  body { margin: 20px; background: #101214; color: #eef1f3; }
  header { display:flex; align-items:baseline; gap:16px; flex-wrap:wrap; }
  h1 { font-size: 22px; margin: 0 0 14px; }
  h2 { font-size:16px; margin:0; }
  #summary { color:#aeb8bf; }
  a { color:#8bc7ff; }
  nav { display:flex; gap:8px; margin:0 0 16px; }
  nav button { padding:8px 14px; color:#c8d1d8; background:#1b1f22; border:1px solid #30363b; border-radius:8px; cursor:pointer; }
  nav button[aria-selected="true"] { color:#fff; background:#245c8a; border-color:#3984bd; }
  #calibration { margin:0 0 16px; padding:14px 16px; background:#1b1f22; border:1px solid #30363b; border-radius:10px; }
  #calibration-head { display:flex; justify-content:space-between; gap:16px; margin-bottom:9px; }
  #calibration-message { margin-top:9px; color:#d6dde2; }
  #calibration-meta { margin-top:5px; color:#aeb8bf; font:12px ui-monospace,monospace; }
  .progress { height:10px; overflow:hidden; border-radius:999px; background:#30363b; }
  #calibration-progress { width:0; height:100%; background:#67db83; transition:width .2s ease; }
  #calibration[data-state="moving"] #calibration-progress,
  #calibration[data-state="holding"] #calibration-progress { background:#ffba5a; }
  #calibration[data-state="complete"] { border-color:#397f49; }
  #cameras { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; }
  article { background:#1b1f22; border:1px solid #30363b; border-radius:10px; overflow:hidden; }
  article h2 { padding:12px 14px 0; }
  img { width:100%; aspect-ratio:16/10; object-fit:contain; background:#000; display:block; }
  .meta { padding:10px 12px 12px; font:13px ui-monospace,monospace; white-space:pre-wrap; }
  .ok { color:#67db83; } .bad { color:#ffba5a; }
  #pose-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; }
  #pose-grid article { padding:14px; }
  #pose-grid article h2 { padding:0; margin-bottom:10px; }
  #pose-grid .wide { grid-column:1 / -1; }
  .pose-copy { color:#aeb8bf; font-size:13px; line-height:1.45; }
  .pose-values { font:13px ui-monospace,monospace; white-space:pre-wrap; line-height:1.55; }
  table { width:100%; border-collapse:collapse; font:13px ui-monospace,monospace; }
  th, td { padding:7px 9px; text-align:right; border-bottom:1px solid #30363b; }
  th:first-child, td:first-child { text-align:left; }
  [hidden] { display:none !important; }
  @media (max-width:850px) { #cameras, #pose-grid { grid-template-columns:1fr; } #pose-grid .wide { grid-column:auto; } }
</style>
<header><h1>Hexapod tracker</h1><span id="summary">starting…</span></header>
<nav aria-label="Tracker views">
  <button type="button" data-tab="cameras" aria-selected="true">Cameras</button>
  <button type="button" data-tab="pose" aria-selected="false">Pose</button>
</nav>
<section id="camera-panel">
  <section id="calibration" data-state="waiting">
    <div id="calibration-head"><strong>Stereo calibration</strong><span id="calibration-count">checking…</span></div>
    <div class="progress"><div id="calibration-progress"></div></div>
    <div id="calibration-message">Checking automatic capture…</div>
    <div id="calibration-meta"></div>
  </section>
  <main id="cameras"></main>
</section>
<section id="pose-panel" hidden>
  <div id="pose-grid">
    <article>
      <h2>Camera-estimated joint pose</h2>
      <div id="camera-status" class="pose-copy">waiting for camera estimates…</div>
      <table>
        <thead><tr><th>Leg</th><th>Yaw</th><th>Hip</th><th>Knee</th><th>Visible tags</th></tr></thead>
        <tbody id="camera-joint-rows"></tbody>
      </table>
    </article>
    <article>
      <h2>Calibrated IMU</h2>
      <div id="imu-pose" class="pose-values">waiting…</div>
    </article>
    <article class="wide">
      <h2>Robot motor pose</h2>
      <div id="motor-status" class="pose-copy">waiting for read-only feedback…</div>
      <table>
        <thead><tr><th>Leg</th><th>Yaw</th><th>Hip</th><th>Knee</th></tr></thead>
        <tbody id="joint-rows"></tbody>
      </table>
    </article>
    <article class="wide">
      <h2>Source contract</h2>
      <div id="fusion-status" class="pose-copy"></div>
      <div class="pose-copy"><a href="/api/pose-state">combined JSON API</a> · read-only; no motor commands</div>
    </article>
  </div>
</section>
<script>
let activeTab = 'cameras';
document.querySelectorAll('nav button').forEach(button => {
  button.addEventListener('click', () => {
    activeTab = button.dataset.tab;
    document.querySelectorAll('nav button').forEach(item => {
      item.setAttribute('aria-selected', String(item === button));
    });
    document.getElementById('camera-panel').hidden = activeTab !== 'cameras';
    document.getElementById('pose-panel').hidden = activeTab !== 'pose';
    if (activeTab === 'pose') updatePose();
  });
});
let serverRunId = null;
function attachStream(img, index) {
  // The query string only forces a new connection; the server ignores it.
  img.src = `/stream/${index}.mjpg?run=${Date.now()}`;
}
function ensureCameraCard(c) {
  let article = document.getElementById(`camera-${c.index}`);
  if (!article) {
    article = document.createElement('article');
    article.id = `camera-${c.index}`;
    article.dataset.index = String(c.index);
    const img = document.createElement('img');
    img.alt = `Camera ${c.index} — AprilTag annotated`;
    // A dropped multipart stream stays frozen on its last frame forever
    // unless something re-requests it.
    // Back off rather than reconnecting on a fixed timer: a browser allows
    // only a handful of connections per host, and each camera holds one open
    // for as long as it streams, so a tight retry loop can starve the page
    // itself of connections.
    img.addEventListener('error', () => {
      if (img.dataset.retrying === '1') return;
      img.dataset.retrying = '1';
      const attempt = Number(img.dataset.attempts || '0') + 1;
      img.dataset.attempts = String(attempt);
      setTimeout(() => {
        img.dataset.retrying = '0';
        attachStream(img, c.index);
      }, Math.min(1500 * attempt, 10000));
    });
    img.addEventListener('load', () => { img.dataset.attempts = '0'; });
    attachStream(img, c.index);
    const meta = document.createElement('div');
    meta.className = 'meta';
    meta.id = `meta-${c.index}`;
    article.append(img, meta);
    document.getElementById('cameras').append(article);
  }
  return article.querySelector('.meta');
}
function describe(c) {
  const accepted = c.mjpg_request_accepted === true ? 'accepted' :
    c.mjpg_request_accepted === false ? 'rejected by AVFoundation' : 'pending';
  const input = c.backend === 'AVFOUNDATION_NATIVE'
    ? `native input: ${c.reported_fourcc || '420v'}`
    : `USB input MJPG request: ${accepted}; reported: ${c.reported_fourcc || 'unavailable'}`;
  const nativeCapture = c.native_capture_width
    ? `; native source ${c.native_capture_width}x${c.native_capture_height}`
    : '';
  const nativeExport = c.native_luma_available
    ? `\nfull native frame: /native-frame/${c.index}.nv12 · lossless luma: /native-luma/${c.index}.png`
    : '';
  return `camera ${c.index}${c.device_name ? ` · ${c.device_name}` : ''}: ${c.state}\n` +
    `${input}${nativeCapture}\n` +
    `mode: ${c.reported_width}x${c.reported_height}; rotation: ${c.rotation_degrees || 0}°; camera measured ${c.measured_fps} fps; browser ${c.output_fps} fps\n` +
    `frames: ${c.frames}; age: ${c.last_frame_age_s ?? 'n/a'} s; reconnects: ${c.reconnects}\n` +
    `tag36h11 IDs: ${c.tag_ids.length ? c.tag_ids.join(', ') : 'none'}${nativeExport}${c.error ? `\n${c.error}` : ''}`;
}
function degrees(value) {
  return value == null ? '—' : `${Number(value).toFixed(2)}°`;
}
function cameraDegrees(joint) {
  if (!joint || joint.status !== 'tracked' || joint.value_deg == null) return '—';
  const uncertainty = joint.error_95_estimate_deg == null
    ? '' : ` ± ${Number(joint.error_95_estimate_deg).toFixed(1)}°`;
  return `${Number(joint.value_deg).toFixed(2)}°${uncertainty}`;
}
function renderPose(state) {
  const camera = state.camera_pose || {};
  const calibrations = Object.entries(camera.calibration?.cameras || {});
  const intrinsicCameras = camera.calibration?.intrinsics?.cameras || {};
  const calibrationLine = calibrations.length
    ? calibrations.map(([index, value]) => {
        const lens = intrinsicCameras[index];
        const lensText = lens
          ? `${lens.quality} lens${lens.floor_reprojection_rms_px == null ? '' : `, ${Number(lens.floor_reprojection_rms_px).toFixed(2)} px fit`}`
          : 'no lens profile';
        return `camera ${index}: floor ${value.status} (${value.quality}) · ${lensText}`;
      }).join('\\n')
    : 'No camera calibration status.';
  const cameraJoints = camera.camera_joint_pose || {};
  const trackedYaw = Number(cameraJoints.tracked_yaw_count || 0);
  const trackedHip = Number(cameraJoints.tracked_hip_count || 0);
  const trackedKnee = Number(cameraJoints.tracked_knee_count || 0);
  const cameraStatus = document.getElementById('camera-status');
  const intrinsicQuality = camera.calibration?.intrinsics?.quality || 'unavailable';
  cameraStatus.textContent = `${calibrationLine}\nlens set: ${intrinsicQuality}; provisional means single-plane, not precision calibrated\n${trackedYaw}/6 yaw · ${trackedHip}/6 hip · ${trackedKnee}/6 knee from rigid link-tag orientations`;
  cameraStatus.className = `pose-copy ${trackedYaw || trackedHip || trackedKnee ? 'ok' : 'bad'}`;
  const parts = Object.values(camera.parts || {});
  const cameraRows = document.getElementById('camera-joint-rows');
  cameraRows.replaceChildren();
  for (let leg = 0; leg < 6; leg++) {
    const row = document.createElement('tr');
    const yaw = cameraJoints.joints?.[`L${leg}_yaw`];
    const hip = cameraJoints.joints?.[`L${leg}_hip`];
    const knee = cameraJoints.joints?.[`L${leg}_knee`];
    const observed = parts
      .filter(part => part.part_id?.startsWith(`leg${leg}_`))
      .flatMap(part => part.observed_tag_ids || []);
    const visibleTags = [
      yaw?.status === 'tracked' ? yaw.servo_lid_tag_id : null,
      ...(hip?.status === 'tracked' ? hip.tag_ids || [] : []),
      ...(knee?.status === 'tracked' ? knee.tag_ids || [] : []),
      ...observed,
    ]
      .filter((value, index, values) => value != null && values.indexOf(value) === index);
    const values = [`L${leg}`, cameraDegrees(yaw), cameraDegrees(hip), cameraDegrees(knee), visibleTags.length ? visibleTags.join(', ') : '—'];
    values.forEach(value => {
      const cell = document.createElement('td');
      cell.textContent = value;
      row.append(cell);
    });
    cameraRows.append(row);
  }

  const motor = state.motor_feedback || {};
  const age = motor.sample_age_s == null ? 'unknown age' : `${Number(motor.sample_age_s).toFixed(2)} s old`;
  const motorStatus = document.getElementById('motor-status');
  motorStatus.textContent = motor.ok
    ? `${motor.live_joint_count}/18 live angles · ${age} · ${motor.joint_frame}`
    : `Telemetry unavailable: ${motor.error || (motor.configured ? 'waiting for feedback' : 'server has no --robot-url')}`;
  motorStatus.className = `pose-copy ${motor.ok ? 'ok' : 'bad'}`;
  const byName = Object.fromEntries((motor.joints || []).map(joint => [joint.name, joint]));
  const rows = document.getElementById('joint-rows');
  rows.replaceChildren();
  for (let leg = 0; leg < 6; leg++) {
    const row = document.createElement('tr');
    const values = [`L${leg}`, ...['yaw', 'hip', 'knee'].map(axis => degrees(byName[`L${leg}_${axis}`]?.degrees))];
    values.forEach(value => {
      const cell = document.createElement('td');
      cell.textContent = value;
      row.append(cell);
    });
    rows.append(row);
  }

  const imu = state.imu || {};
  const calibrated = imu.body_frame_calibrated === true;
  const gyro = Array.isArray(imu.gyro_dps) ? imu.gyro_dps.map(value => Number(value).toFixed(2)).join(', ') : '—';
  const imuElement = document.getElementById('imu-pose');
  imuElement.textContent = `${calibrated ? 'body frame calibrated' : 'body frame NOT calibrated'}\nbody roll: ${degrees(imu.body_roll_deg)}\nbody pitch: ${degrees(imu.body_pitch_deg)}\nsensor roll: ${degrees(imu.sensor_roll_deg)}\nsensor pitch: ${degrees(imu.sensor_pitch_deg)}\ngyro x/y/z: ${gyro} °/s`;
  imuElement.className = `pose-values ${calibrated ? 'ok' : 'bad'}`;
  document.getElementById('fusion-status').textContent = state.fusion?.reason || 'Camera and robot sources are shown separately.';
}
async function updatePose() {
  if (activeTab !== 'pose') return;
  try {
    const response = await fetch('/api/pose-state', {cache:'no-store'});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    renderPose(await response.json());
  } catch (error) {
    document.getElementById('motor-status').textContent = `pose status error: ${error}`;
    document.getElementById('motor-status').className = 'pose-copy bad';
  }
}
async function update() {
  try {
    const [status, calibration] = await Promise.all([
      fetch('/status.json', {cache:'no-store'}).then(r => r.json()),
      fetch('/calibration-status.json', {cache:'no-store'}).then(r => r.json()),
    ]);
    if (serverRunId !== null && status.server_run_id !== serverRunId) {
      // The server restarted, so every open stream belongs to a dead process.
      // Dropping the cards makes ensureCameraCard rebuild them.
      document.querySelectorAll('#cameras article').forEach(a => a.remove());
    }
    serverRunId = status.server_run_id;
    const active = new Set(status.cameras.map(c => String(c.index)));
    document.querySelectorAll('#cameras article').forEach(article => {
      if (!active.has(article.dataset.index)) article.remove();
    });
    let live = 0, tags = 0;
    for (const c of status.cameras) {
      const el = ensureCameraCard(c);
      el.textContent = describe(c);
      el.className = `meta ${c.state === 'streaming' ? 'ok' : 'bad'}`;
      if (c.state === 'streaming') live++;
      tags += c.tag_ids.length;
    }
    document.getElementById('summary').textContent = `${live}/${status.cameras.length} live · ${tags} tags detected`;
    const saved = calibration.saved || 0;
    const target = calibration.target || 0;
    const percent = target ? Math.min(100, 100 * saved / target) : 0;
    const panel = document.getElementById('calibration');
    panel.dataset.state = calibration.state || 'waiting';
    document.getElementById('calibration-count').textContent = `${saved} / ${target} pairs`;
    document.getElementById('calibration-progress').style.width = `${percent}%`;
    document.getElementById('calibration-message').textContent = calibration.message || 'Waiting for calibration capture.';
    const motion = calibration.motion_px == null ? '—' : `${calibration.motion_px.toFixed(1)} px`;
    document.getElementById('calibration-meta').textContent = calibration.state === 'complete'
      ? 'automatic collection finished · dataset ready for validation'
      : `common board tags: ${calibration.common_tags || 0} · motion: ${motion}`;
  } catch (e) {
    document.getElementById('summary').textContent = `status error: ${e}`;
  }
}
update(); setInterval(update, 1000); setInterval(updatePose, 1000);
</script>
</html>
"""


class StreamHandler(BaseHTTPRequestHandler):
    server: "CameraHTTPServer"

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        path = urlparse(self.path).path
        if path != "/api/mode":
            self.send_error(HTTPStatus.NOT_FOUND, "unknown endpoint")
            return
        # Deliberately the only writable route on this server. It selects how
        # to observe and can never move a robot, which matters because :8766
        # is reverse-tunnelled off this machine.
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            requested = json.loads(raw or b"{}")
        except json.JSONDecodeError as error:
            self._send_json(
                {"ok": False, "error": f"invalid JSON: {error}"},
                HTTPStatus.BAD_REQUEST,
            )
            return
        if not isinstance(requested, dict) or "mode" not in requested:
            self._send_json(
                {"ok": False, "error": 'expected a JSON object with a "mode" key'},
                HTTPStatus.BAD_REQUEST,
            )
            return
        try:
            mode = self.server.set_observation_mode(requested["mode"])
        except ValueError as error:
            self._send_json(
                {"ok": False, "error": str(error)},
                HTTPStatus.BAD_REQUEST,
            )
            return
        self._send_json({"ok": True, "observation_mode": mode})

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        path = urlparse(self.path).path
        if path == "/":
            body = INDEX_HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path in ("/favicon.ico", "/favicon.png"):
            body = favicon_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/x-icon")
            self.send_header("Cache-Control", "max-age=86400")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/status.json":
            self._send_json({
                "server_run_id": SERVER_RUN_ID,
                "observation_mode": self.server.observation_mode,
                "cameras": [worker.snapshot()[1] for worker in self.server.workers],
            })
            return
        if path in ("/api/cameras/health", "/api/cameras/health.json"):
            self._send_json(self.server.camera_health())
            return
        if path == "/api/mode":
            self._send_json({
                "observation_mode": self.server.observation_mode,
                "available_modes": list(OBSERVATION_MODES),
            })
            return
        if path in ("/api/poses", "/api/poses.json"):
            body = json.dumps(self.server.pose_status()).encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path in ("/api/pose-state", "/api/pose-state.json"):
            body = json.dumps(self.server.combined_pose_status()).encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/calibration-status.json":
            body = json.dumps(self.server.calibration_status()).encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path.startswith("/snapshot/") and path.endswith(".jpg"):
            try:
                index = int(path.removeprefix("/snapshot/").removesuffix(".jpg"))
                worker = next(item for item in self.server.workers if item.index == index)
            except (ValueError, StopIteration):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            body = worker.raw_snapshot()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path.startswith("/native-luma/") and path.endswith(".png"):
            try:
                index = int(path.removeprefix("/native-luma/").removesuffix(".png"))
                worker = next(item for item in self.server.workers if item.index == index)
            except (ValueError, StopIteration):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            snapshot = worker.native_luma_snapshot()
            if snapshot is None:
                self.send_error(HTTPStatus.NOT_FOUND, "native luma unavailable")
                return
            body, width, height = snapshot
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/png")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Frame-Width", str(width))
            self.send_header("X-Frame-Height", str(height))
            self.send_header("X-Pixel-Format", "Y8-video-range-from-NV12")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path.startswith("/native-frame/") and path.endswith(".nv12"):
            try:
                index = int(path.removeprefix("/native-frame/").removesuffix(".nv12"))
                worker = next(item for item in self.server.workers if item.index == index)
            except (ValueError, StopIteration):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            snapshot = worker.native_nv12_snapshot()
            if snapshot is None:
                self.send_error(HTTPStatus.NOT_FOUND, "native NV12 unavailable")
                return
            body, width, height = snapshot
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Disposition", f'attachment; filename="camera-{index}.nv12"')
            self.send_header("X-Frame-Width", str(width))
            self.send_header("X-Frame-Height", str(height))
            self.send_header("X-Pixel-Format", "NV12-video-range")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path.startswith("/stream/") and path.endswith(".mjpg"):
            try:
                index = int(path.removeprefix("/stream/").removesuffix(".mjpg"))
                worker = next(item for item in self.server.workers if item.index == index)
            except (ValueError, StopIteration):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._stream(worker)
            return
        if path.startswith("/raw-stream/") and path.endswith(".mjpg"):
            try:
                index = int(path.removeprefix("/raw-stream/").removesuffix(".mjpg"))
                worker = next(item for item in self.server.workers if item.index == index)
            except (ValueError, StopIteration):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._stream(worker, raw=True)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def _stream(self, worker: CameraWorker, raw: bool = False) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.end_headers()
        previous_frames = -1
        try:
            while True:
                jpeg, previous_frames = worker.wait_for_frame(previous_frames, raw=raw)
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            return


class CameraHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    @staticmethod
    def _address_family(host: str) -> int:
        """Pick the socket family from the requested host.

        macOS resolves ``localhost`` to ``::1`` as well as ``127.0.0.1``, and
        Safari tries the IPv6 answer. An IPv4-only listener refuses that
        connection, which shows up as a browser that loads nothing at all
        while curl and Chrome quietly fall back to IPv4.
        """

        try:
            infos = socket.getaddrinfo(
                host or None, None, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE
            )
        except socket.gaierror:
            return socket.AF_INET
        families = {info[0] for info in infos}
        if socket.AF_INET in families:
            return socket.AF_INET
        return socket.AF_INET6 if socket.AF_INET6 in families else socket.AF_INET

    def __init__(
        self,
        address: tuple[str, int],
        workers: list[CameraWorker],
        calibration_directory: Path | None = None,
        calibration_target: int = 12,
        pose_estimator: PlanarPoseEstimator | None = None,
        feedback_client: Any | None = None,
        floor_anchor_ids: Sequence[int] = (),
    ):
        self.workers = workers
        self.calibration_directory = calibration_directory
        self.calibration_target = calibration_target
        self.pose_estimator = pose_estimator
        self.feedback_client = feedback_client
        self.floor_anchor_ids = {int(value) for value in floor_anchor_ids}
        self.address_family = self._address_family(address[0])
        if self.address_family == socket.AF_INET6:
            # Dual-stack so an IPv6 bind still answers IPv4 clients.
            self.allow_reuse_address = True
        # Robot Lab asks for an intent; this server decides how to observe.
        # It is only ever an input: nothing here queries the robot or the Lab
        # for task state, which keeps the observation-only boundary intact.
        # A restart deliberately returns to TRACK, the mode that never turns a
        # camera off.
        self._observation_mode = OBSERVATION_MODE_TRACK
        self._observation_mode_lock = threading.Lock()
        super().__init__(address, StreamHandler)

    def server_bind(self) -> None:
        if self.address_family == socket.AF_INET6:
            try:
                self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
            except OSError:
                pass
        super().server_bind()

    @property
    def observation_mode(self) -> str:
        with self._observation_mode_lock:
            return self._observation_mode

    def set_observation_mode(self, mode: str) -> str:
        candidate = str(mode).strip().lower()
        if candidate not in OBSERVATION_MODES:
            raise ValueError(
                f"unknown mode {mode!r}; expected one of "
                f"{', '.join(sorted(OBSERVATION_MODES))}"
            )
        with self._observation_mode_lock:
            self._observation_mode = candidate
        return candidate

    def camera_health(self) -> dict[str, Any]:
        """Report which cameras are working and what each one adds.

        The Lab needs two different things here.  "Is this camera working"
        is per-camera and local.  "Is this camera worth keeping on" is a
        comparison: a feed can be perfectly healthy and still contribute
        nothing that another camera does not already see.  Both are reported
        so a caller can distinguish a broken camera from a redundant one.
        """

        snapshots = [worker.snapshot()[1] for worker in self.workers]
        tag_sets = {
            int(item["index"]): {int(tag) for tag in item.get("tag_ids") or ()}
            for item in snapshots
        }
        cameras: list[dict[str, Any]] = []
        for item in snapshots:
            slot = int(item["index"])
            tags = tag_sets[slot]
            others: set[int] = set()
            for other_slot, other_tags in tag_sets.items():
                if other_slot != slot:
                    others |= other_tags
            age = item.get("last_frame_age_s")
            reasons: list[str] = []
            if item.get("state") != "streaming":
                reasons.append(f"state is {item.get('state')}")
            if not item.get("frames"):
                reasons.append("no frames delivered yet")
            if age is None:
                reasons.append("no frame timestamp")
            elif age > STALE_FRAME_AGE_S:
                reasons.append(f"last frame {age:.1f}s old")
            if item.get("error"):
                reasons.append(str(item["error"]))
            cameras.append({
                "slot": slot,
                "device_name": item.get("device_name"),
                "stable_id": item.get("requested_stable_id"),
                "healthy": not reasons,
                "reasons": reasons,
                "state": item.get("state"),
                "measured_fps": item.get("measured_fps"),
                "frames": item.get("frames"),
                "reconnects": item.get("reconnects"),
                "last_frame_age_s": age,
                "capture_size_px": [
                    item.get("native_capture_width"),
                    item.get("native_capture_height"),
                ],
                "tags_seen": len(tags),
                "unique_tags": sorted(tags - others),
                "floor_anchors_seen": sorted(tags & self.floor_anchor_ids),
                # A healthy feed adding no unique tag is a candidate to drop or
                # re-aim, never evidence that it is broken.
                "redundant": bool(tags) and not (tags - others),
            })
        union = set().union(*tag_sets.values()) if tag_sets else set()
        healthy = [item for item in cameras if item["healthy"]]
        return {
            "schema_version": 1,
            "generated_at_unix_s": round(time.time(), 6),
            "observation_mode": self.observation_mode,
            "cameras_total": len(cameras),
            "cameras_healthy": len(healthy),
            "union_tags_seen": len(union),
            "floor_anchors_seen": sorted(union & self.floor_anchor_ids),
            "floor_anchors_missing": sorted(self.floor_anchor_ids - union),
            "cameras": cameras,
        }

    def pose_status(self) -> dict[str, Any]:
        if self.pose_estimator is None:
            return {
                "schema_version": 1,
                "error": "pose estimation is disabled",
            }
        return self.pose_estimator.estimate(
            [worker.pose_snapshot() for worker in self.workers]
        )

    def combined_pose_status(self) -> dict[str, Any]:
        """Return camera pose plus read-only encoder and calibrated IMU state."""
        if self.feedback_client is None:
            angles: dict[str, float] = {}
            feedback: dict[str, Any] = {
                "configured": False,
                "ok": False,
                "error": "restart with --robot-url to read robot telemetry",
            }
        else:
            angles, feedback = self.feedback_client.sample()

        sample_time = feedback.get("sample_time_unix")
        sample_age = None
        if isinstance(sample_time, (int, float)):
            sample_age = max(0.0, time.time() - float(sample_time))
        joints = [
            {
                "name": name,
                "leg": int(name[1]),
                "axis": name.split("_", 1)[1],
                "degrees": angles.get(name),
            }
            for name in JOINT_NAMES
        ]
        return {
            "schema_version": 1,
            "generated_at_unix_s": round(time.time(), 6),
            "read_only": True,
            "fusion": {
                "status": "sources_presented_separately",
                "reason": (
                    "provisional camera yaw, hip, and absolute tibia/knee angles "
                    "are presented beside, not fused with, encoders"
                ),
            },
            "camera_pose": self.pose_status(),
            "motor_feedback": {
                "configured": bool(feedback.get("configured")),
                "ok": bool(feedback.get("ok")),
                "endpoint": feedback.get("endpoint"),
                "error": feedback.get("error"),
                "sample_time_unix": sample_time,
                "sample_age_s": (
                    None if sample_age is None else round(sample_age, 3)
                ),
                "live_joint_count": int(
                    feedback.get("live_joint_count", len(angles))
                ),
                "joint_frame": FRAME_ROBOT_ABS,
                "joint_contract": JOINT_CONTRACT,
                "joints": joints,
            },
            "imu": {
                "source": "robot GET /api/feedback with apply_calib=True",
                "body_frame_calibrated": bool(
                    feedback.get("body_frame_calibrated", False)
                ),
                "body_roll_deg": feedback.get("body_roll_deg"),
                "body_pitch_deg": feedback.get("body_pitch_deg"),
                "rear_pose_pitch_reference_deg": feedback.get(
                    "rear_pose_pitch_reference_deg"
                ),
                "sensor_roll_deg": feedback.get("roll_deg"),
                "sensor_pitch_deg": feedback.get("pitch_deg"),
                "gyro_dps": feedback.get("gyro_dps"),
            },
        }

    def calibration_status(self) -> dict[str, Any]:
        if self.calibration_directory is None:
            return {
                "state": "disabled",
                "saved": 0,
                "target": 0,
                "common_tags": 0,
                "motion_px": None,
                "message": "Automatic calibration capture is not running.",
            }
        status_path = self.calibration_directory / "capture_status.json"
        try:
            return json.loads(status_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            saved = len(list(self.calibration_directory.glob("pair_*_camera0.jpg")))
            return {
                "state": "waiting",
                "saved": saved,
                "target": self.calibration_target,
                "common_tags": 0,
                "motion_px": None,
                "message": "Move to a new pose, then hold the board steady.",
            }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--output-fps", type=float, default=10.0)
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=70,
        help=(
            "quality of the browser MJPEG preview only; /snapshot and "
            "/raw-stream stay at 95"
        ),
    )
    parser.add_argument(
        "--preview-max-width",
        type=int,
        default=640,
        help=(
            "downscale only the browser MJPEG preview to this width (0 keeps "
            "full size). Detection, pose and /snapshot are unaffected. At full "
            "size three feeds pushed ~46 Mbps, which Safari cannot decode "
            "smoothly; the default is ~10 Mbps"
        ),
    )
    parser.add_argument(
        "--capture-profile",
        help="named camera setup from --capture-profiles-file",
    )
    parser.add_argument(
        "--capture-profiles-file",
        type=Path,
        default=CONFIG_DIR / "camera_capture_profiles.json",
        help="saved capture modes, backends, rotations, and calibration pairing",
    )
    parser.add_argument(
        "--robot-url",
        help=(
            "optional robot HTTP base URL; only read-only GET /api/feedback "
            "is used"
        ),
    )
    parser.add_argument(
        "--feedback-hz",
        type=float,
        default=3.0,
        help="read-only robot feedback rate (default: 3 Hz)",
    )
    parser.add_argument("--calibration-directory", type=Path)
    parser.add_argument("--calibration-target", type=int, default=12)
    parser.add_argument(
        "--floor-map",
        type=Path,
        default=CONFIG_DIR / "floor_tag_map.json",
        help="fixed floor-anchor geometry used by /api/poses",
    )
    parser.add_argument(
        "--part-map",
        type=Path,
        default=CONFIG_DIR / "hexapod_tag_map.json",
        help="tag-to-part grouping used by /api/poses",
    )
    parser.add_argument(
        "--robot-tag-layout",
        type=Path,
        default=CONFIG_DIR / "hexapod-1-apriltag-layout.json",
        help="chassis and link tag orientations used for camera joints",
    )
    parser.add_argument(
        "--camera-calibration",
        type=Path,
        default=CONFIG_DIR / "camera_intrinsics.json",
        help="per-camera intrinsic calibration used for 3-D joint orientation",
    )
    parser.add_argument(
        "--camera-mode",
        action="append",
        default=[],
        metavar="INDEX:WIDTH:HEIGHT:FPS",
        help="override capture mode for one camera; may be repeated",
    )
    parser.add_argument(
        "--rotate-180",
        type=int,
        nargs="+",
        default=[],
        metavar="INDEX",
        help="rotate selected camera frames 180 degrees before detection and display",
    )
    parser.add_argument(
        "--native-avfoundation",
        type=int,
        nargs="+",
        default=[],
        metavar="INDEX",
        help="capture selected AVFoundation device indices through native 420v/NV12",
    )
    parser.add_argument(
        "--device-id",
        action="append",
        default=[],
        metavar="INDEX:STABLE_ID",
        help=(
            "pin one slot to a specific camera by its AVFoundation stable id "
            "(uniqueID), so replugging cannot silently reassign it; may be "
            "repeated. Requires that slot to be in --native-avfoundation. "
            "Read the ids from /status.json or camera_descriptors()"
        ),
    )
    return parser.parse_args()


def parse_device_ids(values: list[str]) -> dict[int, str]:
    """Parse ``INDEX:STABLE_ID`` pairs pinning slots to specific cameras."""

    pinned: dict[int, str] = {}
    for value in values:
        raw_index, separator, stable_id = value.partition(":")
        if not separator or not stable_id.strip():
            raise SystemExit(
                f"--device-id expects INDEX:STABLE_ID, got {value!r}"
            )
        try:
            index = int(raw_index)
        except ValueError:
            raise SystemExit(
                f"--device-id expects an integer slot index, got {raw_index!r}"
            ) from None
        if index in pinned:
            raise SystemExit(f"--device-id repeats slot {index}")
        pinned[index] = stable_id.strip()
    return pinned


def parse_camera_modes(values: list[str]) -> dict[int, tuple[int, int, float]]:
    modes: dict[int, tuple[int, int, float]] = {}
    for value in values:
        try:
            raw_index, raw_width, raw_height, raw_fps = value.split(":")
            index = int(raw_index)
            width = int(raw_width)
            height = int(raw_height)
            fps = float(raw_fps)
        except (ValueError, TypeError) as error:
            raise SystemExit(
                f"invalid --camera-mode {value!r}; expected INDEX:WIDTH:HEIGHT:FPS"
            ) from error
        if index < 0 or width <= 0 or height <= 0 or fps <= 0:
            raise SystemExit(f"invalid non-positive --camera-mode value: {value!r}")
        modes[index] = (width, height, fps)
    return modes


def load_capture_profile(path: Path, name: str) -> dict[str, Any]:
    """Load and validate one saved camera setup."""
    try:
        document = json.loads(path.read_text())
        profile = document["profiles"][name]
    except FileNotFoundError as error:
        raise SystemExit(f"capture profile file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise SystemExit(f"invalid capture profile JSON in {path}: {error}") from error
    except KeyError as error:
        choices = sorted(document.get("profiles", {})) if "document" in locals() else []
        suffix = f"; available: {', '.join(choices)}" if choices else ""
        raise SystemExit(f"unknown capture profile {name!r}{suffix}") from error

    try:
        indices = [int(index) for index in profile["indices"]]
        raw_modes = profile["camera_modes"]
        modes = {
            int(index): (
                int(spec["width"]),
                int(spec["height"]),
                float(spec["fps"]),
            )
            for index, spec in raw_modes.items()
        }
        rotate_180 = [int(index) for index in profile.get("rotate_180", [])]
        native = [int(index) for index in profile.get("native_avfoundation", [])]
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"invalid capture profile {name!r} in {path}") from error

    if not indices or set(modes) != set(indices):
        raise SystemExit(
            f"capture profile {name!r} must define exactly one camera mode per index"
        )
    if any(min(width, height, fps) <= 0 for width, height, fps in modes.values()):
        raise SystemExit(f"capture profile {name!r} contains a non-positive mode")
    if not set(rotate_180).issubset(indices) or not set(native).issubset(indices):
        raise SystemExit(f"capture profile {name!r} references an unknown camera index")

    result = dict(profile)
    result["indices"] = indices
    result["camera_modes"] = modes
    result["rotate_180"] = rotate_180
    result["native_avfoundation"] = native
    return result


def main() -> None:
    args = parse_args()
    if args.feedback_hz <= 0.0:
        raise SystemExit("--feedback-hz must be positive")
    profile = None
    if args.capture_profile:
        profile = load_capture_profile(args.capture_profiles_file, args.capture_profile)
        args.indices = profile["indices"]
        args.rotate_180 = profile["rotate_180"]
        args.native_avfoundation = profile["native_avfoundation"]
        args.output_fps = float(profile.get("output_fps", args.output_fps))
        args.jpeg_quality = int(profile.get("jpeg_quality", args.jpeg_quality))
        camera_modes = profile["camera_modes"]
        if profile.get("camera_calibration"):
            calibration_path = Path(str(profile["camera_calibration"]))
            if not calibration_path.is_absolute():
                calibration_path = args.capture_profiles_file.parent / calibration_path
            args.camera_calibration = calibration_path
    else:
        camera_modes = parse_camera_modes(args.camera_mode)
    floor_map = json.loads(args.floor_map.read_text())
    pose_estimator = PlanarPoseEstimator(
        floor_map,
        json.loads(args.part_map.read_text()),
        json.loads(args.robot_tag_layout.read_text()),
        json.loads(args.camera_calibration.read_text()),
    )
    pinned_device_ids = parse_device_ids(args.device_id)
    unknown_slots = sorted(set(pinned_device_ids) - set(args.indices))
    if unknown_slots:
        raise SystemExit(
            f"--device-id names slots not in --indices: {unknown_slots}"
        )
    # Only the native adapter can address a device by identity; OpenCV's
    # VideoCapture takes an index and nothing else, so silently ignoring a pin
    # there would hand back whichever camera happened to occupy the slot.
    unpinnable = sorted(set(pinned_device_ids) - set(args.native_avfoundation))
    if unpinnable:
        raise SystemExit(
            "--device-id requires --native-avfoundation for the same slots; "
            f"missing: {unpinnable}"
        )

    workers = [
        CameraWorker(
            index,
            camera_modes.get(index, (args.width, args.height, args.fps))[0],
            camera_modes.get(index, (args.width, args.height, args.fps))[1],
            camera_modes.get(index, (args.width, args.height, args.fps))[2],
            args.output_fps,
            args.jpeg_quality,
            rotate_180=index in args.rotate_180,
            native_avfoundation=index in args.native_avfoundation,
            stable_id=pinned_device_ids.get(index),
            preview_max_width=args.preview_max_width,
        )
        for index in args.indices
    ]
    for worker in workers:
        worker.start()
        time.sleep(0.4)

    feedback_client = None
    if args.robot_url:
        from .track import FeedbackClient

        feedback_client = FeedbackClient(args.robot_url, hz=args.feedback_hz)
    server = CameraHTTPServer(
        (args.host, args.port),
        workers,
        calibration_directory=args.calibration_directory,
        calibration_target=args.calibration_target,
        pose_estimator=pose_estimator,
        feedback_client=feedback_client,
        floor_anchor_ids=floor_map.get("active_anchor_ids") or (),
    )
    stop_requested = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        if not stop_requested.is_set():
            stop_requested.set()
            threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    print(f"AprilTag camera viewer: http://{args.host}:{args.port}", flush=True)
    if profile is not None:
        print(
            f"Capture profile: {args.capture_profile} — "
            f"{profile.get('description', 'saved camera setup')}",
            flush=True,
        )
    print("Input format requested: MJPG; see /status.json for backend acceptance", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
        for worker in workers:
            worker.stop()


if __name__ == "__main__":
    main()
