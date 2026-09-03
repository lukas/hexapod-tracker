#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "opencv-contrib-python>=4.8,<6",
#   "numpy>=1.24",
# ]
# ///
"""Serve two annotated AprilTag camera feeds to a local web browser.

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
    backend: str = "AVFOUNDATION"
    state: str = "starting"
    requested_fourcc: str = "MJPG"
    mjpg_request_accepted: bool | None = None
    reported_fourcc: str | None = None
    requested_width: int = 1280
    requested_height: int = 800
    requested_fps: float = 30.0
    output_fps: float = 10.0
    reported_width: int = 0
    reported_height: int = 0
    reported_fps: float = 0.0
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
    ):
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self.output_fps = output_fps
        self.jpeg_quality = jpeg_quality
        self.status = CameraStatus(
            index=index,
            requested_width=width,
            requested_height=height,
            requested_fps=fps,
            output_fps=output_fps,
        )
        self._jpeg = placeholder_jpeg(index, "waiting for frames")
        self._raw_jpeg = self._jpeg
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
    ) -> None:
        with self._condition:
            self._raw_jpeg = raw_jpeg
            self._jpeg = jpeg
            self.status.frames += 1
            self.status.tag_ids = tag_ids
            self._tag_corners = {
                tag_id: corners.copy() for tag_id, corners in tag_corners.items()
            }
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

    def _open(self) -> cv2.VideoCapture | None:
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

                sample_frames += 1
                now = time.monotonic()
                if now < next_output_at:
                    continue
                next_output_at = now + 1.0 / self.output_fps

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
                    self._publish(
                        raw_encoded.tobytes(), encoded.tobytes(), tag_ids, tag_corners
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
  #summary { color:#aeb8bf; }
  a { color:#8bc7ff; }
  #calibration { margin:0 0 16px; padding:14px 16px; background:#1b1f22; border:1px solid #30363b; border-radius:10px; }
  #calibration-head { display:flex; justify-content:space-between; gap:16px; margin-bottom:9px; }
  #calibration-message { margin-top:9px; color:#d6dde2; }
  #calibration-meta { margin-top:5px; color:#aeb8bf; font:12px ui-monospace,monospace; }
  .progress { height:10px; overflow:hidden; border-radius:999px; background:#30363b; }
  #calibration-progress { width:0; height:100%; background:#67db83; transition:width .2s ease; }
  #calibration[data-state="moving"] #calibration-progress,
  #calibration[data-state="holding"] #calibration-progress { background:#ffba5a; }
  #calibration[data-state="complete"] { border-color:#397f49; }
  main { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; }
  article { background:#1b1f22; border:1px solid #30363b; border-radius:10px; overflow:hidden; }
  img { width:100%; aspect-ratio:16/10; object-fit:contain; background:#000; display:block; }
  .meta { padding:10px 12px 12px; font:13px ui-monospace,monospace; white-space:pre-wrap; }
  .ok { color:#67db83; } .bad { color:#ffba5a; }
  @media (max-width:850px) { main { grid-template-columns:1fr; } }
</style>
<header><h1>Hexapod AprilTag cameras</h1><span id="summary">starting…</span><a href="/api/poses">pose API</a></header>
<section id="calibration" data-state="waiting">
  <div id="calibration-head"><strong>Stereo calibration</strong><span id="calibration-count">checking…</span></div>
  <div class="progress"><div id="calibration-progress"></div></div>
  <div id="calibration-message">Checking automatic capture…</div>
  <div id="calibration-meta"></div>
</section>
<main id="cameras"></main>
<script>
const cameraNames = {
  0: 'Camera 0 — OV9281',
  1: 'Camera 1 — OV9281',
  2: 'MacBook camera',
  3: 'Continuity camera',
};
function ensureCameraCard(c) {
  let article = document.getElementById(`camera-${c.index}`);
  if (!article) {
    article = document.createElement('article');
    article.id = `camera-${c.index}`;
    article.dataset.index = String(c.index);
    const img = document.createElement('img');
    const raw = c.index >= 2;
    img.src = `${raw ? '/raw-stream/' : '/stream/'}${c.index}.mjpg`;
    img.alt = `${cameraNames[c.index] || `Camera ${c.index}`} — ${raw ? 'raw high quality' : 'AprilTag annotated'}`;
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
  return `camera ${c.index}: ${c.state}\n` +
    `USB input MJPG request: ${accepted}; reported: ${c.reported_fourcc || 'unavailable'}\n` +
    `mode: ${c.reported_width}x${c.reported_height}; camera measured ${c.measured_fps} fps; browser ${c.output_fps} fps\n` +
    `frames: ${c.frames}; age: ${c.last_frame_age_s ?? 'n/a'} s; reconnects: ${c.reconnects}\n` +
    `tag36h11 IDs: ${c.tag_ids.length ? c.tag_ids.join(', ') : 'none'}${c.error ? `\n${c.error}` : ''}`;
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
update(); setInterval(update, 1000);
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
    ):
        self.workers = workers
        self.calibration_directory = calibration_directory
        self.calibration_target = calibration_target
        self.pose_estimator = pose_estimator
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
        "--camera-mode",
        action="append",
        default=[],
        metavar="INDEX:WIDTH:HEIGHT:FPS",
        help="override capture mode for one camera; may be repeated",
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


def main() -> None:
    args = parse_args()
    camera_modes = parse_camera_modes(args.camera_mode)
    pose_estimator = PlanarPoseEstimator(
        json.loads(args.floor_map.read_text()),
        json.loads(args.part_map.read_text()),
    )
    workers = [
        CameraWorker(
            index,
            camera_modes.get(index, (args.width, args.height, args.fps))[0],
            camera_modes.get(index, (args.width, args.height, args.fps))[1],
            camera_modes.get(index, (args.width, args.height, args.fps))[2],
            args.output_fps,
            args.jpeg_quality,
        )
        for index in args.indices
    ]
    for worker in workers:
        worker.start()
        time.sleep(0.4)

    server = CameraHTTPServer(
        (args.host, args.port),
        workers,
        calibration_directory=args.calibration_directory,
        calibration_target=args.calibration_target,
        pose_estimator=pose_estimator,
    )
    stop_requested = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        if not stop_requested.is_set():
            stop_requested.set()
            threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    print(f"AprilTag camera viewer: http://{args.host}:{args.port}", flush=True)
    print("Input format requested: MJPG; see /status.json for backend acceptance", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
        for worker in workers:
            worker.stop()


if __name__ == "__main__":
    main()
