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
import threading
import time
from dataclasses import asdict, dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np

from .housing_pose import JOINT_NAMES
from .joint_contract import FRAME_ROBOT_ABS, JOINT_CONTRACT
from .paths import CONFIG_DIR
from .planar_pose import PlanarPoseEstimator


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
    ):
        self.index = index
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
                device_name = next(
                    (
                        str(item["name"])
                        for item in descriptors
                        if int(item["index"]) == index
                    ),
                    None,
                )
            except Exception:
                device_name = None
        self.status = CameraStatus(
            index=index,
            device_name=device_name,
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
                        self._set_waiting("capture stalled; reconnecting", "stalled")
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
                ok, encoded = cv2.imencode(
                    ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
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
function ensureCameraCard(c) {
  let article = document.getElementById(`camera-${c.index}`);
  if (!article) {
    article = document.createElement('article');
    article.id = `camera-${c.index}`;
    article.dataset.index = String(c.index);
    const img = document.createElement('img');
    img.src = `/stream/${c.index}.mjpg`;
    img.alt = `Camera ${c.index} — AprilTag annotated`;
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
        if path == "/status.json":
            body = json.dumps({"cameras": [worker.snapshot()[1] for worker in self.server.workers]}).encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
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

    def __init__(
        self,
        address: tuple[str, int],
        workers: list[CameraWorker],
        calibration_directory: Path | None = None,
        calibration_target: int = 12,
        pose_estimator: PlanarPoseEstimator | None = None,
        feedback_client: Any | None = None,
    ):
        self.workers = workers
        self.calibration_directory = calibration_directory
        self.calibration_target = calibration_target
        self.pose_estimator = pose_estimator
        self.feedback_client = feedback_client
        super().__init__(address, StreamHandler)

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
    parser.add_argument("--jpeg-quality", type=int, default=82)
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
    return parser.parse_args()


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
    pose_estimator = PlanarPoseEstimator(
        json.loads(args.floor_map.read_text()),
        json.loads(args.part_map.read_text()),
        json.loads(args.robot_tag_layout.read_text()),
        json.loads(args.camera_calibration.read_text()),
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
