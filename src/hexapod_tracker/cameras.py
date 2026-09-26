"""Cameras without a daemon.

Three things live here:

* **The registry** -- one JSON file per machine (``~/.hexapod/cameras.json``,
  override with ``HEXAPOD_CAMERAS_FILE``) that says, per camera, what it is
  and what we know about it: its role (``top``, ``side``, ...), the capture
  size and frame rate to open it at, its intrinsics, and its saved floor fit
  (homography from the floor AprilTags, with the numbers that say how good
  the fit was), and optionally ``uvc``: UVC controls to push through
  ``uvc-util`` every time the camera is opened (``{"auto-focus": false,
  "focus-abs": 496}`` pins a ceiling camera's lens on the floor; the firmware
  default is autofocus, which hunts and refocuses on whoever stands under
  it). Cameras are keyed by their AVFoundation ``uniqueID`` (the
  "stable id"), never by a slot number. That id follows the USB port, not the
  device, so every entry also carries ``device_name`` and ``adopt`` moves an
  entry to a new id when a camera is replugged elsewhere.

* **A per-run capture** -- :class:`Rig` opens exactly the cameras a job asks
  for, by role, and releases them when the job ends. It detects tags on the
  full-resolution luma and publishes two documents in the same shapes the old
  HTTP server served (``/api/detections.json`` and ``/api/poses``), so
  everything that parsed those keeps working.

* **A session** -- ``hexapod-cameras session --out DIR --roles top,side --hz 20 --aux-hz 2``
  is the process a run starts. One *primary* camera (``--primary``, default ``top``) paces it:
  a state per primary frame up to ``--hz``; the other cameras are polled without waiting,
  recorded, and detected ``--aux-hz`` times a second. For the life of the run it writes, into
  ``DIR``: ``state.json`` (atomically replaced; sequence number, detections,
  poses), ``latest_<role>.jpg``, ``vision.jsonl`` (one line per state), and
  ``<role>.mp4`` with ``<role>_timestamps.csv`` (the capture time of every
  frame). It ends when its stdin closes, when ``DIR/STOP`` appears, on
  SIGTERM, or after ``--seconds``. A dead parent therefore never leaves a
  camera claimed, which is what the always-on server kept doing.

The calibration commands (``list``, ``assign``, ``adopt``, ``calibrate floor``,
``calibrate intrinsics``, ``check``, ``snapshot``, ``record``, ``show``,
``import-intrinsics``) write to the registry. Nothing here binds a port.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import select
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import cv2
import numpy as np

from .camera_server import (
    detect_tag_corners_with_duplicates,
    make_tag_detector,
)
from .paths import CONFIG_DIR
from .planar_pose import CameraCalibration, PlanarPoseEstimator

REGISTRY_ENV = "HEXAPOD_CAMERAS_FILE"
DEFAULT_REGISTRY = Path("~/.hexapod/cameras.json")
SCHEMA_VERSION = 1
DEFAULT_ROLES = ("top",)
DEFAULT_PREFERRED_SIZES = ((1920, 1440), (1920, 1080), (1280, 720))
DEFAULT_PROCESSING_WIDTH = 1280
UVC_UTIL_ENV = "HEXAPOD_UVC_UTIL"
DEFAULT_UVC_UTIL = Path.home() / ".hexapod" / "bin" / "uvc-util"     # built from github.com/jtfrey/uvc-util
LAB_MODEL_ENV = "HEXAPOD_LAB_MODEL_FILE"
DEFAULT_LAB_MODEL = Path("~/.hexapod/lab_model.json")     # solved camera poses (robot-lab lab_model.py); positions
                                                          # let a raised anchor (the 50 mm plate) join a floor homography


def camera_positions_from_lab_model(path: Path | None = None) -> dict[str, np.ndarray]:
    """stable_id (and studio-<role> name) -> [x, y, z] mm of every solved camera in the lab model; {} if absent."""
    p = Path(os.environ.get(LAB_MODEL_ENV) or (path or DEFAULT_LAB_MODEL)).expanduser()
    try:
        model = json.loads(p.read_text())
    except (OSError, ValueError):
        return {}
    out: dict[str, np.ndarray] = {}
    for cam in model.get("cameras") or []:
        pos = cam.get("position_mm")
        if not cam.get("solved") or not pos or len(pos) < 3:
            continue
        xyz = np.asarray(pos[:3], dtype=np.float64)
        for key in (cam.get("stable_id"), cam.get("name")):
            if key:
                out[str(key)] = xyz
    return out
MIN_ANCHORS_TO_REFIT = 3     # fewer visible floor anchors than this: keep the saved floor fit
FFMPEG = os.environ.get("HEXAPOD_FFMPEG", "ffmpeg")


# --------------------------------------------------------------------------- registry


def registry_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path).expanduser()
    return Path(os.environ.get(REGISTRY_ENV, str(DEFAULT_REGISTRY))).expanduser()


def empty_registry() -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "cameras": {}, "configs": {}}


def load_registry(path: str | Path | None = None) -> dict[str, Any]:
    p = registry_path(path)
    if not p.exists():
        return empty_registry()
    doc = json.loads(p.read_text())
    doc.setdefault("schema_version", SCHEMA_VERSION)
    doc.setdefault("cameras", {})
    doc.setdefault("configs", {})
    return doc


def save_registry(doc: dict[str, Any], path: str | Path | None = None) -> Path:
    """Write atomically: a half-written registry must never be read by a run."""
    p = registry_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=False) + "\n")
    os.replace(tmp, p)
    return p


def new_entry(device_name: str, *, role: str | None = None) -> dict[str, Any]:
    return {
        "device_name": device_name,
        "role": role,
        "capture_size": None,          # [w, h] to pin; None lets the camera pick from DEFAULT_PREFERRED_SIZES
        "fps": 30.0,
        "processing_width": DEFAULT_PROCESSING_WIDTH,
        "rotate_180": False,
        "rotate": 0,                   # 0 / 90 / 180 / 270 degrees clockwise applied to every frame (a camera mounted sideways)
        "intrinsics": None,
        "floor": None,
        "notes": "",
    }


def entries_by_role(doc: dict[str, Any]) -> dict[str, str]:
    """role -> stable id (the first entry claiming a role wins; ``assign`` keeps roles unique)."""
    out: dict[str, str] = {}
    for stable_id, entry in doc.get("cameras", {}).items():
        role = entry.get("role")
        if role and role not in out:
            out[role] = stable_id
    return out


def resolve(doc: dict[str, Any], spec: str) -> str:
    """A role name, a stable id, or a device name that matches exactly one entry -> stable id."""
    cameras = doc.get("cameras", {})
    if spec in cameras:
        return spec
    roles = entries_by_role(doc)
    if spec in roles:
        return roles[spec]
    by_name = [sid for sid, e in cameras.items() if str(e.get("device_name", "")).strip() == spec.strip()]
    if len(by_name) == 1:
        return by_name[0]
    if len(by_name) > 1:
        raise KeyError(f"{spec!r} names {len(by_name)} cameras; use the stable id")
    raise KeyError(f"no camera called {spec!r} in the registry (roles: {sorted(roles)})")


def assign(doc: dict[str, Any], stable_id: str, *, device_name: str | None = None, role: str | None = None,
           capture_size: tuple[int, int] | None = None, fps: float | None = None,
           rotate_180: bool | None = None, notes: str | None = None, rotate: int | None = None) -> dict[str, Any]:
    cameras = doc.setdefault("cameras", {})
    entry = cameras.get(stable_id) or new_entry(device_name or "")
    if device_name:
        entry["device_name"] = device_name
    if role is not None:
        for other_id, other in cameras.items():
            if other_id != stable_id and other.get("role") == role:
                other["role"] = None          # a role names one camera
        entry["role"] = role or None
    if capture_size is not None:
        entry["capture_size"] = [int(capture_size[0]), int(capture_size[1])]
    if fps is not None:
        entry["fps"] = float(fps)
    if rotate_180 is not None:
        entry["rotate_180"] = bool(rotate_180)
    if rotate is not None:
        if int(rotate) % 90:
            raise ValueError("rotate must be 0, 90, 180 or 270")
        entry["rotate"] = int(rotate) % 360
    if notes is not None:
        entry["notes"] = notes
    cameras[stable_id] = entry
    return doc


def adopt(doc: dict[str, Any], old_id: str, new_id: str) -> dict[str, Any]:
    """Move an entry to the id a replugged camera now has. Its calibration comes along."""
    cameras = doc.setdefault("cameras", {})
    if old_id not in cameras:
        raise KeyError(f"no entry {old_id!r} to adopt from")
    if new_id in cameras:
        raise KeyError(f"{new_id!r} already has an entry; delete one first")
    cameras[new_id] = cameras.pop(old_id)
    cameras[new_id].setdefault("notes", "")
    cameras[new_id]["adopted_from"] = old_id
    return doc


def import_intrinsics(doc: dict[str, Any], intrinsics_doc: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Bring entries of an identity-keyed intrinsics file (``camera_intrinsics_*.json``) into the registry."""
    imported: list[str] = []
    for key, spec in (intrinsics_doc.get("cameras") or {}).items():
        if not isinstance(spec, dict):
            continue
        stable_id = str(spec.get("stable_id") or "").strip()
        if not stable_id:
            continue
        entry = doc.setdefault("cameras", {}).get(stable_id) or new_entry(str(spec.get("device_name") or key))
        entry["intrinsics"] = {k: v for k, v in spec.items() if k not in ("stable_id", "device_name")}
        if spec.get("capture_size") and not entry.get("capture_size"):
            entry["capture_size"] = [int(v) for v in spec["capture_size"]]
        doc["cameras"][stable_id] = entry
        imported.append(stable_id)
    return doc, imported


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


# --------------------------------------------------------------------------- frames


@dataclass
class Frame:
    bgr: np.ndarray
    gray: np.ndarray                 # the image tags are detected on (full-resolution luma when the camera gives it)
    captured_unix: float
    seq: int

    @property
    def width(self) -> int:
        return int(self.bgr.shape[1])

    @property
    def height(self) -> int:
        return int(self.bgr.shape[0])


@dataclass
class Observation:
    role: str
    index: int
    frame: Frame | None
    tags: dict[int, np.ndarray] = field(default_factory=dict)   # id -> 4x2 corners in frame (bgr) coordinates
    duplicates: list[int] = field(default_factory=list)
    detect_size: tuple[int, int] = (0, 0)
    detect_seq: int = 0


def default_capture_factory(slot: int, stable_id: str, entry: dict[str, Any]) -> Any:
    from .avfoundation_capture import AVFoundationYuvCapture

    size = entry.get("capture_size")
    preferred = ((int(size[0]), int(size[1])),) if size else DEFAULT_PREFERRED_SIZES
    return AVFoundationYuvCapture(
        slot,
        stable_id=stable_id,
        preferred_sizes=preferred,
        fps=float(entry.get("fps") or 30.0),
        processing_width=int(entry.get("processing_width") or DEFAULT_PROCESSING_WIDTH),
    )



# --------------------------------------------------------------------------- UVC controls (focus etc.)

def uvc_location_id(stable_id: str) -> str | None:
    """AVFoundation's uniqueID for a UVC camera is ``locationID << 32 | vendor << 16 | product``
    (``0x110000032e40362`` -> location ``0x01100000``), which is how uvc-util selects a device."""
    try:
        return f"0x{int(str(stable_id), 16) >> 32:08x}"
    except ValueError:
        return None


def apply_uvc_controls(stable_id: str, controls: dict[str, Any], *, runner: Callable[..., Any] = subprocess.run,
                       util: str | os.PathLike[str] | None = None, log: Callable[[str], None] = print,
                       sleep: Callable[[float], None] = time.sleep) -> dict[str, str]:
    """Push the registry entry's ``uvc`` controls to the camera through uvc-util.

    Reads each control first and only sets what differs; ``auto-focus`` goes first so
    the firmware does not move the lens after we placed it; waits a second for the lens
    to travel only when ``focus-abs`` actually changed. Never raises: without the tool
    the camera still opens, just with whatever the firmware chose. Returns what changed."""
    if not controls:
        return {}
    location = uvc_location_id(stable_id)
    if location is None:
        log(f"uvc: {stable_id!r} is not a UVC unique id; controls not applied")
        return {}
    tool = Path(util or os.environ.get(UVC_UTIL_ENV) or DEFAULT_UVC_UTIL).expanduser()
    if not tool.exists():
        log(f"uvc: {tool} not found; controls {sorted(controls)} not applied")
        return {}
    changed: dict[str, str] = {}
    for name in sorted(controls, key=lambda k: (k != "auto-focus", k)):
        want = controls[name]
        want_s = ("true" if want else "false") if isinstance(want, bool) else str(want)
        try:
            got = runner([str(tool), "-L", location, "-o", name], capture_output=True, text=True, timeout=10)
            if got.returncode == 0 and (got.stdout or "").strip() == want_s:
                continue
            res = runner([str(tool), "-L", location, "-s", f"{name}={want_s}"], capture_output=True, text=True, timeout=10)
            if res.returncode != 0:
                log(f"uvc: could not set {name}={want_s} on {location}: {(res.stderr or res.stdout or '').strip()[:200]}")
                continue
            changed[name] = want_s
        except (OSError, subprocess.SubprocessError) as e:
            log(f"uvc: {name} on {location}: {e}")
    if "focus-abs" in changed:
        sleep(1.0)
    return changed


def frame_rotation(entry: dict[str, Any]):
    """OpenCV rotate code for a registry entry: ``rotate`` (0/90/180/270, clockwise) or the older ``rotate_180``."""
    deg = int(entry.get("rotate") or 0) % 360
    if entry.get("rotate_180") and deg == 0:
        deg = 180
    return {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}[deg]


class Camera:
    """One opened camera: frames plus tag detection, released on close."""

    def __init__(self, role: str, index: int, stable_id: str, entry: dict[str, Any], *,
                 capture_factory: Callable[[int, str, dict[str, Any]], Any] = default_capture_factory,
                 clock: Callable[[], float] = time.time, log: Callable[[str], None] = print):
        self.role, self.index, self.stable_id, self.entry = role, index, stable_id, entry
        self._factory = capture_factory
        self._clock = clock
        self.capture: Any = None
        self.detector = make_tag_detector()
        self.seq = 0
        self.detect_seq = 0
        self.last: Frame | None = None
        self.error: str | None = None
        self.uvc_changed: dict[str, str] = {}
        self._log = log
        self._fused_ids: set[int] = set()      # what the last full (native + 2x) pass found, and when
        self._fused_t = -math.inf
        self._native_misses = 0                # consecutive fast detections that had to fall back: 3+ and we skip the native pass

    def open(self) -> None:
        self.capture = self._factory(self.index, self.stable_id, self.entry)
        if not self.capture.isOpened():
            self.error = getattr(self.capture, "last_error", None) or "camera did not open"
            raise RuntimeError(f"{self.role} ({self.entry.get('device_name')} {self.stable_id}): {self.error}")
        if self.entry.get("uvc"):
            self.uvc_changed = apply_uvc_controls(self.stable_id, dict(self.entry["uvc"]), log=self._log)

    def grab(self, wait: bool = True) -> Frame | None:
        """Next frame. ``wait=False`` polls: None when no new frame has arrived yet, and that is
        not an error (captures without a non-blocking read just block, as before)."""
        if wait:
            ok, bgr = self.capture.read()
        else:
            try:
                ok, bgr = self.capture.read(wait=False)
            except TypeError:
                ok, bgr = self.capture.read()
        if not ok or bgr is None:
            if wait:
                self.error = getattr(self.capture, "last_error", None) or "no frame"
            return None
        rot = frame_rotation(self.entry)
        if rot is not None:
            bgr = cv2.rotate(bgr, rot)
        gray = getattr(self.capture, "detection_gray", None)
        if gray is None or getattr(gray, "ndim", 0) != 2 or max(gray.shape) < max(bgr.shape[:2]):
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        elif rot is not None:
            gray = cv2.rotate(gray, rot)
        self.seq += 1
        self.error = None
        self.last = Frame(bgr, gray, self._clock(), self.seq)
        return self.last

    def detect(self, frame: Frame, *, fast: bool = False, full_every_s: float = 1.0) -> Observation:
        """Tags in the frame. ``fast`` runs the native pass only (a quarter of the cost; the primary
        camera of a session) and falls back to the full native + 2x pass when the native pass lost an
        id the last full pass had (a 22 px chassis tag from the ceiling decodes only upscaled), and
        once every ``full_every_s`` anyway so newly visible small tags are picked up."""
        if fast and self._native_misses >= 3 and self._clock() - self._fused_t < full_every_s:
            # this camera's tags need the 2x pass (the ceiling's 22 px chassis tag): do not pay for a
            # native pass that will come up short; try native again once a second
            corners, duplicates = detect_tag_corners_with_duplicates(frame.gray, self.detector)
            self._fused_ids = set(corners)
        else:
            corners, duplicates = detect_tag_corners_with_duplicates(frame.gray, self.detector, enhance=not fast)
            if fast:
                now = self._clock()
                missing = bool(self._fused_ids - set(corners))
                if now - self._fused_t >= full_every_s or missing:
                    native_ids = set(corners)
                    corners, duplicates = detect_tag_corners_with_duplicates(frame.gray, self.detector)
                    self._fused_ids, self._fused_t = set(corners), now
                    self._native_misses = self._native_misses + 1 if (self._fused_ids - native_ids) else 0
            else:
                self._fused_ids, self._fused_t = set(corners), self._clock()
        scale = frame.width / frame.gray.shape[1]
        if scale != 1.0:
            corners = {tid: c * scale for tid, c in corners.items()}
        self.detect_seq += 1
        return Observation(self.role, self.index, frame,
                           {int(t): np.asarray(c, dtype=np.float64).reshape(4, 2) for t, c in corners.items()},
                           duplicates, (frame.gray.shape[1], frame.gray.shape[0]), self.detect_seq)

    def release(self) -> None:
        if self.capture is not None:
            try:
                self.capture.release()
            finally:
                self.capture = None

    def info(self) -> dict[str, Any]:
        cap = self.capture.capture_info() if self.capture is not None and hasattr(self.capture, "capture_info") else {}
        return {"role": self.role, "index": self.index, "stable_id": self.stable_id,
                "device_name": self.entry.get("device_name"), "capture": cap}


# --------------------------------------------------------------------------- the rig


def load_configs(doc: dict[str, Any], configs: Path = CONFIG_DIR) -> dict[str, Any]:
    """Floor map, part map and robot layout named by the registry (defaults: the tracker's configs/)."""
    names = doc.get("configs") or {}
    paths = {
        "floor_map": Path(names.get("floor_map") or configs / "floor_tag_map.json").expanduser(),
        "part_map": Path(names.get("part_map") or configs / "hexapod_tag_map.json").expanduser(),
        "robot_layout": Path(names.get("robot_layout") or configs / "hexapod-1-apriltag-layout.json").expanduser(),
    }
    out: dict[str, Any] = {"paths": {k: str(v) for k, v in paths.items()}}
    for key, p in paths.items():
        out[key] = json.loads(p.read_text()) if p.exists() else None
    if out["floor_map"] is None:
        raise FileNotFoundError(f"floor map not found: {paths['floor_map']}")
    if out["part_map"] is None:
        out["part_map"] = {"parts": []}
    return out


def calibration_for_slots(cameras: Sequence[Camera]) -> dict[str, Any]:
    """The slot-keyed ``camera_calibration`` PlanarPoseEstimator wants, from registry intrinsics."""
    slots: dict[str, Any] = {}
    for cam in cameras:
        intr = cam.entry.get("intrinsics")
        if intr:
            slots[str(cam.index)] = {**intr, "stable_id": cam.stable_id, "device_name": cam.entry.get("device_name")}
    return {"schema_version": 1, "quality": "provisional", "cameras": slots}


def floor_fit_to_json(cal: CameraCalibration, image_size: tuple[int, int], *, floor_map_path: str,
                      floor_map_digest: str, frames: int, clock: Callable[[], float] = time.time) -> dict[str, Any]:
    return {
        "homography": np.asarray(cal.homography, dtype=float).round(9).tolist(),   # floor mm -> image px
        "image_size": [int(image_size[0]), int(image_size[1])],
        "anchor_ids": [int(t) for t in cal.anchor_ids],
        "reprojection_rms_px": round(float(cal.reprojection_rms_px), 3),
        "world_rms_mm": round(float(cal.world_rms_mm), 3),
        "leave_one_anchor_out_position_mm": [round(float(v), 2) for v in cal.leave_one_out_position_mm],
        "leave_one_anchor_out_yaw_degrees": [round(float(v), 3) for v in cal.leave_one_out_yaw_degrees],
        "position_error_95_mm": round(float(cal.position_error_95_mm), 2),
        "yaw_error_95_degrees": round(float(cal.yaw_error_95_degrees), 3),
        "quality": cal.quality,
        "frames": int(frames),
        "floor_map": floor_map_path,
        "floor_map_digest": floor_map_digest,
        "fitted_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(clock())),
    }


def floor_fit_from_json(index: int, saved: dict[str, Any]) -> CameraCalibration:
    return CameraCalibration(
        camera_index=int(index),
        homography=np.asarray(saved["homography"], dtype=np.float64),
        anchor_ids=[int(t) for t in saved.get("anchor_ids", [])],
        reprojection_rms_px=float(saved.get("reprojection_rms_px", 0.0)),
        world_rms_mm=float(saved.get("world_rms_mm", 0.0)),
        position_error_95_mm=float(saved.get("position_error_95_mm", 30.0)),
        yaw_error_95_degrees=float(saved.get("yaw_error_95_degrees", 5.0)),
        leave_one_out_position_mm=[float(v) for v in saved.get("leave_one_anchor_out_position_mm", [])],
        leave_one_out_yaw_degrees=[float(v) for v in saved.get("leave_one_anchor_out_yaw_degrees", [])],
        quality=str(saved.get("quality", "provisional")),
    )


def median_tags(observations: Iterable[Observation], *, need: int) -> dict[int, np.ndarray]:
    """Per tag, the median corners over the observations that saw it at least ``need`` times."""
    seen: dict[int, list[np.ndarray]] = {}
    for obs in observations:
        for tid, corners in obs.tags.items():
            seen.setdefault(tid, []).append(corners)
    return {tid: np.median(np.stack(c), axis=0) for tid, c in seen.items() if len(c) >= need}


class Rig:
    """The cameras one job needs, opened by role, released on exit."""

    def __init__(self, doc: dict[str, Any], roles: Sequence[str] = DEFAULT_ROLES, *, configs: Path = CONFIG_DIR,
                 capture_factory: Callable[[int, str, dict[str, Any]], Any] = default_capture_factory,
                 clock: Callable[[], float] = time.time, log: Callable[[str], None] = print):
        self.doc = doc
        self.roles = list(roles)
        self.log = log
        self.clock = clock
        by_role = entries_by_role(doc)
        missing = [r for r in self.roles if r not in by_role]
        if missing:
            raise KeyError(f"no camera has role {missing}; run `hexapod-cameras list` then `assign`")
        self.cameras = [Camera(role, i, by_role[role], doc["cameras"][by_role[role]], capture_factory=capture_factory,
                               clock=clock, log=log) for i, role in enumerate(self.roles)]
        self.configs = load_configs(doc, configs)
        self.estimator = PlanarPoseEstimator(self.configs["floor_map"], self.configs["part_map"],
                                             self.configs["robot_layout"], calibration_for_slots(self.cameras))
        self.estimator.hold_calibration_s = 1e9          # a run is shorter than any sensible hold
        positions = camera_positions_from_lab_model()
        for cam in self.cameras:
            pos = positions.get(cam.stable_id)
            if pos is None:
                pos = positions.get(f"studio-{cam.role}")
            if pos is not None:
                self.estimator.camera_positions_mm[cam.index] = pos
        self.state_seq = 0
        self.opened = False

    # -- lifecycle
    def open(self) -> "Rig":
        opened: list[Camera] = []
        try:
            for cam in self.cameras:
                cam.open()
                opened.append(cam)
                saved = cam.entry.get("floor")
                if saved:
                    self.estimator._held[cam.index] = (floor_fit_from_json(cam.index, saved), time.monotonic(),
                                                       tuple(int(v) for v in saved["image_size"]))
        except Exception:
            for cam in opened:
                cam.release()
            raise
        self.opened = True
        return self

    def close(self) -> None:
        for cam in self.cameras:
            cam.release()
        self.opened = False

    def __enter__(self) -> "Rig":
        return self.open()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def camera(self, role: str) -> Camera:
        for cam in self.cameras:
            if cam.role == role:
                return cam
        raise KeyError(role)

    # -- observing
    def observe(self, detect: bool = True) -> list[Observation]:
        """Grab one frame per camera; run tag detection on it unless ``detect`` is
        False (a video-only frame between state writes carries no tags)."""
        out: list[Observation] = []
        for cam in self.cameras:
            frame = cam.grab()
            if frame is None:
                out.append(Observation(cam.role, cam.index, None))
            elif detect:
                out.append(cam.detect(frame))
            else:
                out.append(Observation(cam.role, cam.index, frame, detect_seq=cam.detect_seq))
        return out

    def pose_snapshots(self, observations: Sequence[Observation]) -> list[dict[str, Any]]:
        snaps = []
        for obs in observations:
            if obs.frame is None:
                continue
            saved = self.camera(obs.role).entry.get("floor")
            width, height = obs.frame.width, obs.frame.height
            if saved and tuple(saved.get("image_size", ())) != (width, height):
                self.log(f"{obs.role}: saved floor fit is for {saved.get('image_size')} px but frames are "
                         f"{width}x{height}; refit with `hexapod-cameras calibrate floor --role {obs.role}`")
                saved = None
            tags = {tid: c.copy() for tid, c in obs.tags.items()}
            if saved:
                # The estimator refits from whatever anchors it sees, and a one- or two-anchor
                # fit is worse than the saved one (the robot covers most anchors mid-walk).
                # Hide the anchors then, so it falls back to the saved fit it was seeded with.
                visible = [t for t in self.estimator.active_anchor_ids if t in tags]
                if len(visible) < MIN_ANCHORS_TO_REFIT:
                    for t in visible:
                        tags.pop(t, None)
            snaps.append({"index": obs.index, "width": width, "height": height,
                          "frame_age_s": max(0.0, self.clock() - obs.frame.captured_unix), "tags": tags})
        return snaps

    def detections_doc(self, observations: Sequence[Observation]) -> dict[str, Any]:
        """Same shape as the old server's ``/api/detections.json``, plus ``role``."""
        cams = []
        for obs in observations:
            cam = self.camera(obs.role)
            f = obs.frame
            cams.append({
                "index": obs.index, "role": obs.role, "stable_id": cam.stable_id,
                "state": "streaming" if f is not None else "no_frame",
                "width": f.width if f else None, "height": f.height if f else None,
                "detect_width": obs.detect_size[0], "detect_height": obs.detect_size[1],
                "detect_seq": obs.detect_seq, "duplicate_ids": list(obs.duplicates),
                "frame_age_s": (max(0.0, self.clock() - f.captured_unix) if f else None),
                "captured_unix": f.captured_unix if f else None,
                "tags": {str(t): np.asarray(c, dtype=float).reshape(4, 2).round(3).tolist() for t, c in obs.tags.items()},
            })
        return {"cameras": cams, "roles": {cam.role: cam.index for cam in self.cameras}}

    def poses_doc(self, observations: Sequence[Observation]) -> dict[str, Any]:
        """The old server's ``/api/poses`` document."""
        return self.estimator.estimate(self.pose_snapshots(observations))

    def state(self, observations: Sequence[Observation]) -> dict[str, Any]:
        self.state_seq += 1
        det = self.detections_doc(observations)
        poses = self.poses_doc(observations)
        return {
            "schema_version": SCHEMA_VERSION, "seq": self.state_seq, "generated_at_unix_s": self.clock(),
            "roles": det["roles"], "cameras": det["cameras"], "poses": poses,
            # run_hw's vision sidecar keys uniqueness on performance.frame_sequence
            "performance": {"frame_sequence": self.state_seq},
        }

    # -- calibration
    def fit_floor(self, role: str, *, frames: int = 15, interval_s: float = 0.2,
                  sleep: Callable[[float], None] = time.sleep) -> tuple[dict[str, Any], dict[str, Any]]:
        """Median the tag corners over ``frames`` frames and fit the floor homography for one camera.

        Returns (saved-fit json, report). Raises RuntimeError when fewer than
        three surveyed floor anchors were seen: a two-anchor fit is not a
        calibration worth saving."""
        cam = self.camera(role)
        obs_list: list[Observation] = []
        size: tuple[int, int] | None = None
        for _ in range(frames):
            frame = cam.grab()
            if frame is not None:
                obs_list.append(cam.detect(frame))
                size = (frame.width, frame.height)
            sleep(interval_s)
        if size is None:
            raise RuntimeError(f"{role}: no frames")
        tags = median_tags(obs_list, need=max(1, (len(obs_list) + 1) // 2))
        anchors = self.estimator.homography_anchors(tags, cam.index)
        raised_skipped = [t for t in self.estimator.active_anchor_ids if t in tags and t not in anchors]
        if raised_skipped:
            self.log(f"{role}: raised anchors {raised_skipped} seen but this camera has no solved position in the lab "
                     f"model; run the camera survey (lab_model.py) first to use them in the floor fit")
        if len(anchors) < 3:
            raise RuntimeError(f"{role}: only floor anchors {anchors} seen (need 3+ of {self.estimator.active_anchor_ids})")
        cal = self.estimator._calibrate_camera({"index": cam.index, "width": size[0], "height": size[1], "tags": tags})
        if cal is None:
            raise RuntimeError(f"{role}: homography did not fit")
        floor_path = Path(self.configs["paths"]["floor_map"])
        saved = floor_fit_to_json(cal, size, floor_map_path=str(floor_path), floor_map_digest=file_digest(floor_path),
                                  frames=len(obs_list), clock=self.clock)
        plate = self.estimator.plate_check(tags, np.asarray(saved["homography"], dtype=np.float64), cam.index)
        if plate:
            saved["plate_check"] = plate
        report = {"role": role, "stable_id": cam.stable_id, "anchors_seen": anchors, "frames_used": len(obs_list),
                  "raised_anchors_skipped": raised_skipped, "plate_check": plate,
                  "robot_tags_seen": sorted(t for t in tags if t not in self.estimator.anchors),
                  "quality": cal.quality, "reprojection_rms_px": saved["reprojection_rms_px"],
                  "leave_one_anchor_out_position_mm": saved["leave_one_anchor_out_position_mm"]}
        return saved, report

    def check_floor(self, role: str, *, frames: int = 5, interval_s: float = 0.2, drift_px: float = 6.0,
                    sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
        """How far today's anchors sit from where the saved fit puts them (a bumped camera shows up here)."""
        cam = self.camera(role)
        saved = cam.entry.get("floor")
        obs_list: list[Observation] = []
        for _ in range(frames):
            frame = cam.grab()
            if frame is not None:
                obs_list.append(cam.detect(frame))
            sleep(interval_s)
        tags = median_tags(obs_list, need=max(1, (len(obs_list) + 1) // 2))
        anchors = self.estimator.homography_anchors(tags, cam.index)
        report: dict[str, Any] = {"role": role, "stable_id": cam.stable_id, "frames": len(obs_list),
                                  "anchors_seen": anchors, "robot_tags_seen": sorted(t for t in tags if t not in self.estimator.anchors),
                                  "saved_fit": bool(saved), "ok": False}
        if not saved:
            report["reason"] = "no saved floor fit; run `calibrate floor`"
            return report
        if not anchors:
            report["reason"] = "no floor anchor visible"
            return report
        H = np.asarray(saved["homography"], dtype=np.float64)
        world = np.concatenate([self.estimator.anchor_corners_for_view(t, cam.index) for t in anchors])
        image = np.concatenate([tags[t] for t in anchors])
        pts = cv2.perspectiveTransform(world.reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)
        residual = np.linalg.norm(pts - image, axis=1)
        report.update({"drift_rms_px": round(float(np.sqrt(np.mean(residual ** 2))), 2),
                       "drift_max_px": round(float(residual.max()), 2), "drift_limit_px": drift_px})
        report["ok"] = report["drift_rms_px"] <= drift_px
        if not report["ok"]:
            report["reason"] = "anchors moved in the picture: the camera was bumped or the floor map changed; refit"
        report["plate_check"] = self.estimator.plate_check(tags, H, cam.index)
        return report


# --------------------------------------------------------------------------- video


# H.264 encoder for the session videos.  2026-09-23: three parallel software x264 encodes (one of them 4K
# portrait) were the slow job that coalesced video down to 5-10 fps; the Mac's hardware encoder
# (VideoToolbox, present in ffmpeg 9) costs the CPU almost nothing.  Override with HEXAPOD_VIDEO_CODEC.
VIDEO_CODEC = os.environ.get("HEXAPOD_VIDEO_CODEC") or ("h264_videotoolbox" if sys.platform == "darwin" else "libx264")


def _encoder_args(codec: str) -> list[str]:
    if codec == "libx264":
        return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"]
    if codec.endswith("_videotoolbox"):
        return ["-c:v", codec, "-q:v", "60", "-realtime", "1"]
    return ["-c:v", codec]


def _rotation_filter(entry: dict[str, Any]) -> list[str]:
    """ffmpeg -vf equivalent of frame_rotation(entry): the registry's clockwise ``rotate`` degrees."""
    deg = int(entry.get("rotate") or 0) % 360
    if entry.get("rotate_180") and deg == 0:
        deg = 180
    return {0: [], 90: ["-vf", "transpose=1"], 180: ["-vf", "transpose=1,transpose=1"], 270: ["-vf", "transpose=2"]}[deg]


class VideoRecorder:
    """Frames -> H.264 mp4 through an ffmpeg pipe, with a CSV of capture times per frame."""

    def __init__(self, path: Path, fps: float, size: tuple[int, int], *, ffmpeg: str = FFMPEG,
                 popen: Callable[..., Any] = subprocess.Popen, codec: str = VIDEO_CODEC):
        self.path = Path(path)
        self.fps = float(fps)
        self.size = size
        self.frames = 0
        self._ts = open(self.path.with_name(self.path.stem + "_timestamps.csv"), "w", newline="")
        self._csv = csv.writer(self._ts)
        self._csv.writerow(["frame", "captured_unix", "seq"])
        self.proc = popen([ffmpeg, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", f"{size[0]}x{size[1]}", "-r", f"{self.fps:g}", "-i", "-",
                           *_encoder_args(codec), "-pix_fmt", "yuv420p",
                           "-movflags", "+faststart", str(self.path)], stdin=subprocess.PIPE)

    def write(self, frame: Frame) -> None:
        bgr = frame.bgr
        if (bgr.shape[1], bgr.shape[0]) != self.size:
            bgr = cv2.resize(bgr, self.size)
        self.proc.stdin.write(np.ascontiguousarray(bgr).tobytes())
        self._csv.writerow([self.frames, f"{frame.captured_unix:.6f}", frame.seq])
        self.frames += 1

    def close(self) -> None:
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=30)
        finally:
            self._ts.close()


class NativeVideoRecorder:
    """Every frame the camera delivers -> H.264 mp4, fed as raw NV12 planes (no cv2 work in Python).

    2026-09-23 (Lukas: "when tracking the hexapod for an experiment I need one video to be like
    20 Hz"): the loop-gated ``VideoRecorder`` can never beat the detection loop (12-17 Hz with the
    top camera pacing, 6 Hz with the ceiling), and coalescing dropped it to 5-10 fps.  This one is
    fed straight from the capture callback's planes at the camera's own rate; rotation happens in
    ffmpeg (``transpose``), so the loop thread only pays for a pipe write.
    """

    def __init__(self, path: Path, fps: float, size: tuple[int, int], *, entry: dict[str, Any] | None = None,
                 ffmpeg: str = FFMPEG, popen: Callable[..., Any] = subprocess.Popen, codec: str = VIDEO_CODEC):
        self.path = Path(path)
        self.fps = float(fps)
        self.size = size                    # native (unrotated) width, height
        self.frames = 0
        self.native = True
        self._ts = open(self.path.with_name(self.path.stem + "_timestamps.csv"), "w", newline="")
        self._csv = csv.writer(self._ts)
        self._csv.writerow(["frame", "captured_unix", "seq"])
        self.proc = popen([ffmpeg, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "nv12",
                           "-s", f"{size[0]}x{size[1]}", "-r", f"{self.fps:g}", "-i", "-",
                           *_rotation_filter(entry or {}), *_encoder_args(codec), "-pix_fmt", "yuv420p",
                           "-movflags", "+faststart", str(self.path)], stdin=subprocess.PIPE)

    def write_planes(self, y: np.ndarray, uv: np.ndarray, captured_unix: float, seq: int) -> None:
        self.proc.stdin.write(np.ascontiguousarray(y).tobytes())
        self.proc.stdin.write(np.ascontiguousarray(uv).tobytes())
        self._csv.writerow([self.frames, f"{captured_unix:.6f}", seq])
        self.frames += 1

    def close(self) -> None:
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=30)
        finally:
            self._ts.close()


class NativeRecordThread:
    """Feeds a ``NativeVideoRecorder`` from ``capture.wait_frame`` on its own thread.

    Waits in half-second slices so ``stop()`` is seen promptly; a slow write just means the next
    wait returns a newer frame (the CSV's seq column shows the skip).  The recorder is created on
    the first frame, once the native size is known.
    """

    def __init__(self, cam: Camera, path: Path, *, recorder_factory: Callable[..., Any] = NativeVideoRecorder,
                 log: Callable[[str], None] = print):
        self.cam, self.path, self._factory, self._log = cam, Path(path), recorder_factory, log
        self.recorder: Any = None
        self.error: str | None = None
        self.started_unix: float | None = None
        self.last_unix: float | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"native-record-{cam.role}", daemon=True)

    @staticmethod
    def supported(cam: Camera) -> bool:
        return callable(getattr(cam.capture, "wait_frame", None))

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        seq = 0
        fps = float(self.cam.entry.get("fps") or getattr(self.cam.capture, "fps", 0) or 30.0)
        try:
            while not self._stop.is_set():
                got = self.cam.capture.wait_frame(seq, 0.5)
                if got is None:
                    continue
                seq, t, (y, uv) = got
                if self.recorder is None:
                    self.recorder = self._factory(self.path, fps, (int(y.shape[1]), int(y.shape[0])), entry=self.cam.entry)
                    self.started_unix = t
                self.recorder.write_planes(y, uv, t, seq)
                self.last_unix = t
        except Exception as e:  # noqa: BLE001 -- a broken pipe must not take the session down
            self.error = str(e)
            self._log(f"native recording of {self.cam.role} stopped: {e}")

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        self._thread.join(timeout=5.0)
        out: dict[str, Any] = {"path": str(self.path), "native": True, "frames": 0, "fps": None, "error": self.error}
        if self.recorder is not None:
            try:
                self.recorder.close()
            except Exception as e:  # noqa: BLE001
                out["error"] = out["error"] or str(e)
            out["frames"] = self.recorder.frames
            if self.started_unix is not None and self.last_unix is not None and self.last_unix > self.started_unix:
                out["fps"] = round((self.recorder.frames - 1) / (self.last_unix - self.started_unix), 2)
        return out


class Worker:
    """One background thread running coalesced jobs: ``submit(key, fn)`` keeps only the
    newest job per key, so a slow job (a 4K aux detection, a video frame going into ffmpeg)
    never queues up behind itself; it just skips frames. ``inline=True`` runs jobs at once
    on the caller's thread (tests, and anything that wants determinism)."""

    def __init__(self, name: str, *, inline: bool = False, log: Callable[[str], None] = print):
        self.name, self.inline, self.log = name, inline, log
        self._jobs: dict[Any, Callable[[], None]] = {}
        self._cv = threading.Condition()
        self._stop = False
        self.ran = 0
        self._thread: threading.Thread | None = None
        if not inline:
            self._thread = threading.Thread(target=self._run, name=f"cameras-{name}", daemon=True)
            self._thread.start()

    def submit(self, key: Any, fn: Callable[[], None]) -> None:
        if self.inline:
            self._call(fn)
            return
        with self._cv:
            self._jobs[key] = fn
            self._cv.notify()

    def _call(self, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception as e:  # noqa: BLE001  -- one bad frame must not end the session
            self.log(f"{self.name}: {type(e).__name__}: {e}")
        self.ran += 1

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._jobs and not self._stop:
                    self._cv.wait()
                if not self._jobs:
                    return
                key = next(iter(self._jobs))
                fn = self._jobs.pop(key)
            self._call(fn)

    def close(self) -> None:
        """Finish what is queued, then stop."""
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=30)



# --------------------------------------------------------------------------- session


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _stdin_closed(stdin: Any) -> bool:
    """True once the parent's write end has gone away (EOF on our stdin)."""
    if stdin is None:
        return False
    try:
        ready, _, _ = select.select([stdin], [], [], 0)
    except (ValueError, OSError):
        return True
    if not ready:
        return False
    try:
        return stdin.read(1) == ""
    except (ValueError, OSError):
        return True


def pick_primary(roles: Sequence[str], primary: str | None = None) -> str:
    """The camera a session paces itself on: the one asked for, else ``top``, else the first."""
    if primary and primary in roles:
        return primary
    return "top" if "top" in roles else roles[0]


def run_session(doc: dict[str, Any], out_dir: Path, *, roles: Sequence[str] = DEFAULT_ROLES, hz: float = 20.0,
                aux_hz: float = 2.0, primary: str | None = None, jpeg_hz: float = 1.0,
                video: bool = True, video_fps: float = 10.0, seconds: float | None = None, stdin: Any = None,
                stop_file: bool = True, background: bool = True, capture_factory: Callable[..., Any] = default_capture_factory,
                clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep,
                recorder_factory: Callable[..., Any] = VideoRecorder, configs: Path = CONFIG_DIR,
                log: Callable[[str], None] = print, native_record: Sequence[str] = (),
                native_recorder_factory: Callable[..., Any] = NativeVideoRecorder) -> dict[str, Any]:
    """Own the cameras for one run; write state.json / latest_<role>.jpg / vision.jsonl / <role>.mp4.

    One camera, the *primary*, sets the pace: every state waits for its next frame (a
    blocking grab), detects on it natively (no 2x pass) and writes state.json + a
    vision.jsonl line, up to ``hz`` times a second. The other cameras are *auxiliary*:
    polled without waiting, recorded, and detected only ``aux_hz`` times a second each,
    staggered so two of them never detect in the same state. Their entry in the state
    carries the tags of their last *detected* frame with that frame's age, so a consumer
    can tell a fresh detection from one that is half a second old. Video frames are
    written at most ``video_fps`` per second per camera (capped by ``hz``), the
    ``latest_<role>.jpg`` previews at ``jpeg_hz``. The primary can be changed while running:
    write ``{"primary": "<role>"}`` to ``DIR/control.json`` (the loop watches its mtime) and the
    next state is paced by that camera; the old primary becomes auxiliary at once. Every switch
    is listed in session.json ``primary_switches``. Aux detection, video frames and previews
    run on two background workers (``background=False`` runs them inline), so the loop thread
    only ever waits for the primary's frame and detects on it. Before 2026-09-20 every camera was
    grabbed in turn (each grab waiting for that camera's next frame) and detected with
    the 2x pass at every state, which held three cameras to 4 states a second.

    ``native_record``: roles (``"primary"``, ``"all"`` or role names) whose video is written at the camera's own frame
    rate from the capture thread (``NativeRecordThread``), independent of the state loop and of
    ``video_fps``; a camera whose capture cannot do that (no ``wait_frame``) falls back to the gated
    path with a log line."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stop_path = out_dir / "STOP"
    if stop_path.exists():
        stop_path.unlink()
    stopping = {"why": None}

    def on_signal(signum: int, _frame: Any) -> None:
        stopping["why"] = f"signal {signum}"

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, on_signal)
        except ValueError:            # not the main thread (tests)
            pass

    recorders: dict[str, Any] = {}
    primary_role = pick_primary(list(roles), primary)
    summary: dict[str, Any] = {"out_dir": str(out_dir), "roles": list(roles), "primary": primary_role, "hz": hz,
                               "aux_hz": aux_hz, "states": 0, "frames": {}, "detections": {}, "video": {},
                               "started_unix": clock()}
    period = 1.0 / max(0.5, hz)
    aux_period = 1.0 / max(0.1, aux_hz)
    video_period = 1.0 / max(1.0, video_fps)
    jpeg_period = 1.0 / max(0.1, jpeg_hz)
    with Rig(doc, roles, configs=configs, capture_factory=capture_factory, clock=clock, log=log) as rig:
        write_atomic(out_dir / "session.json", json.dumps({"pid": os.getpid(), "roles": list(roles), "primary": primary_role,
                                                            "hz": hz, "aux_hz": aux_hz, "video_fps": video_fps,
                                                            "cameras": [c.info() for c in rig.cameras],
                                                            "started_unix": summary["started_unix"]}, indent=1))
        prim = rig.camera(primary_role)
        aux = [c for c in rig.cameras if c is not prim]
        native_roles: set[str] = set()
        for r in native_record:
            if r == "all":
                native_roles.update(c.role for c in rig.cameras)
            else:
                native_roles.add(primary_role if r == "primary" else r)
        native_threads: dict[str, NativeRecordThread] = {}
        for role in sorted(native_roles):
            if role not in [c.role for c in rig.cameras]:
                log(f"native-record: {role!r} is not one of {[c.role for c in rig.cameras]}; ignored")
                continue
            cam = rig.camera(role)
            if not video:
                break
            if not NativeRecordThread.supported(cam):
                log(f"native-record: {role}'s capture has no wait_frame; recording it at the gated {video_fps:g} fps instead")
                native_roles.discard(role)
                continue
            native_threads[role] = NativeRecordThread(cam, out_dir / f"{role}.mp4", recorder_factory=native_recorder_factory, log=log)
            native_threads[role].start()
        native_roles = set(native_threads)
        detect_worker = Worker("aux-detect", inline=not background, log=log)
        io_worker = Worker("video", inline=not background, log=log)
        t0 = clock()
        next_state = t0
        next_detect = {c.role: t0 + aux_period * i / max(1, len(aux)) for i, c in enumerate(aux)}   # staggered
        last_obs: dict[str, Observation] = {c.role: Observation(c.role, c.index, None) for c in rig.cameras}
        newest: dict[str, Frame] = {}
        last_video: dict[str, float] = {}
        last_jpeg: dict[str, float] = {}
        control_path = out_dir / "control.json"
        control_seen: list[float | None] = [None]
        summary["primary_switches"] = []

        def apply_control(now: float) -> None:
            """Re-point the pacing at another camera when control.json names one."""
            nonlocal prim, aux
            try:
                m = control_path.stat().st_mtime
            except OSError:
                return
            if m == control_seen[0]:
                return
            control_seen[0] = m
            try:
                want = json.loads(control_path.read_text()).get("primary")
            except (OSError, ValueError):
                return
            if not want or want == prim.role:
                return
            if want not in [c.role for c in rig.cameras]:
                log(f"control.json asks for primary {want!r}, not one of {[c.role for c in rig.cameras]}; ignored")
                return
            old = prim
            prim = rig.camera(want)
            aux = [c for c in rig.cameras if c is not prim]
            next_detect[old.role] = now
            next_detect.pop(prim.role, None)
            summary["primary_switches"].append({"t": now, "from": old.role, "to": prim.role})
            log(f"primary: {old.role} -> {prim.role}")

        def record(cam: Camera, frame: Frame, now: float) -> None:
            summary["frames"][cam.role] = summary["frames"].get(cam.role, 0) + 1
            newest[cam.role] = frame
            # a frame 3/4 of a video period after the last one counts: 20 fps frames gated at 15 fps
            # then keep every frame instead of every other one (10 fps)
            if video and cam.role not in native_roles and now - last_video.get(cam.role, -math.inf) >= 0.75 * video_period - 1e-6:
                rec = recorders.get(cam.role)
                if rec is None:
                    rec = recorders[cam.role] = recorder_factory(out_dir / f"{cam.role}.mp4", video_fps, (frame.width, frame.height))
                io_worker.submit(("video", cam.role), lambda rec=rec, frame=frame: rec.write(frame))
                last_video[cam.role] = now

        def detect_aux(cam: Camera, frame: Frame) -> None:
            last_obs[cam.role] = cam.detect(frame)
            summary["detections"][cam.role] = summary["detections"].get(cam.role, 0) + 1

        def write_jpeg(role: str, frame: Frame) -> None:
            ok, jpg = cv2.imencode(".jpg", frame.bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if ok:
                tmp = out_dir / f"latest_{role}.jpg.tmp"
                tmp.write_bytes(jpg.tobytes())
                os.replace(tmp, out_dir / f"latest_{role}.jpg")

        with open(out_dir / "vision.jsonl", "a") as jsonl:
            while stopping["why"] is None:
                now = clock()
                if seconds is not None and now - t0 >= seconds:
                    stopping["why"] = "seconds"
                    break
                if stop_file and stop_path.exists():
                    stopping["why"] = "STOP file"
                    break
                if _stdin_closed(stdin):
                    stopping["why"] = "stdin closed"
                    break
                apply_control(now)
                before = now
                frame = prim.grab()                       # blocks for the primary's next frame: this is the pace
                now = clock()
                if frame is None:
                    sleep(0.05)
                    continue
                record(prim, frame, now)
                if now < next_state - 0.25 * period:
                    # Too early for a state: this frame is video only. A frame up to a quarter
                    # period early still counts, so a camera running at about ``hz`` gives a
                    # state per frame; sleeping instead would land every state on a later frame
                    # (30 fps frames + a 50 ms sleep = 15 Hz).
                    if now - before < 1e-3:               # the frame was already waiting (or a fake): do not spin
                        sleep(next_state - now)
                    continue
                last_obs[prim.role] = prim.detect(frame, fast=True)
                summary["detections"][prim.role] = summary["detections"].get(prim.role, 0) + 1
                for cam in aux:
                    # Poll an aux camera only when something wants its frame: converting a
                    # frame nobody uses costs the loop thread 2-8 ms each.
                    video_due = video and cam.role not in native_roles and now - last_video.get(cam.role, -math.inf) >= video_period - 1e-6
                    if (not video_due and now < next_detect[cam.role]
                            and now - last_jpeg.get(cam.role, -math.inf) < jpeg_period - 1e-6):
                        continue
                    f = cam.grab(wait=False)
                    if f is None:
                        continue
                    record(cam, f, now)
                    if now >= next_detect[cam.role]:
                        detect_worker.submit(("detect", cam.role), lambda cam=cam, f=f: detect_aux(cam, f))
                        next_detect[cam.role] = now + aux_period
                observations = [last_obs[c.role] for c in rig.cameras]
                state = rig.state(observations)
                state["primary"] = prim.role
                write_atomic(out_dir / "state.json", json.dumps(state))
                for role, f in newest.items():
                    if now - last_jpeg.get(role, -math.inf) >= jpeg_period - 1e-6:
                        io_worker.submit(("jpeg", role), lambda role=role, f=f: write_jpeg(role, f))
                        last_jpeg[role] = now
                markers = state["poses"].get("markers") or {}
                jsonl.write(json.dumps({
                    "seq": state["seq"], "capture_unix": state["generated_at_unix_s"],
                    "tags": {c["role"]: sorted(int(t) for t in c["tags"]) for c in state["cameras"]},
                    "detect_age_s": {c["role"]: (round(c["frame_age_s"], 3) if c.get("frame_age_s") is not None else None)
                                     for c in state["cameras"]},
                    "markers": {k: {"status": m.get("status"), "position_mm": m.get("position_mm"),
                                    "yaw": (m.get("rotation_degrees") or {}).get("yaw")}
                                for k, m in markers.items() if m.get("status") == "tracked"},
                }) + "\n")
                jsonl.flush()
                summary["states"] = state["seq"]
                next_state += period
                if next_state < now:                      # fell behind: do not try to catch up in a burst
                    next_state = now + period
        summary["primary"] = prim.role         # the current one; the switches list has the history
        for role, th in native_threads.items():
            summary["video"][role] = th.stop()  # joins the thread, closes the pipe (30 s wait), measured fps
        detect_worker.close()
        io_worker.close()                      # every queued frame is in ffmpeg before the pipes close
        for role, rec in recorders.items():
            rec.close()
            summary["video"][role] = {"path": str(rec.path), "frames": rec.frames, "fps": rec.fps}
    summary["stopped"] = stopping["why"]
    summary["ended_unix"] = clock()
    write_atomic(out_dir / "session.json", json.dumps({**json.loads((out_dir / "session.json").read_text()),
                                                        **summary}, indent=1))
    return summary


# --------------------------------------------------------------------------- CLI


def _discover() -> list[dict[str, Any]]:
    from .avfoundation_capture import AVFoundationYuvCapture
    from .rig import usb_controller

    devices = AVFoundationYuvCapture.device_descriptors()
    return [{**d, "usb_controller": usb_controller(str(d.get("stable_id", "")))} for d in devices]


def cmd_list(args: argparse.Namespace, doc: dict[str, Any]) -> int:
    devices = _discover()
    cameras = doc.get("cameras", {})
    rows = []
    seen_ids = set()
    for d in devices:
        sid = str(d.get("stable_id", ""))
        seen_ids.add(sid)
        entry = cameras.get(sid)
        row = {"stable_id": sid, "name": d.get("name"), "kind": d.get("kind"), "usb_controller": d.get("usb_controller"),
               "available": d.get("available", True), "registered": entry is not None,
               "role": entry.get("role") if entry else None,
               "intrinsics": bool(entry and entry.get("intrinsics")), "floor": (entry or {}).get("floor", {}) and
               {k: (entry or {})["floor"].get(k) for k in ("quality", "reprojection_rms_px", "fitted_at")}}
        if entry is None:
            twins = [k for k, e in cameras.items() if e.get("device_name") == d.get("name") and k not in seen_ids]
            if twins:
                row["note"] = (f"unregistered, but {twins[0]} has the same device name: if this is that camera on a new "
                               f"port run `hexapod-cameras adopt {twins[0]} {sid}`")
        rows.append(row)
    for sid, entry in cameras.items():
        if sid not in seen_ids:
            rows.append({"stable_id": sid, "name": entry.get("device_name"), "registered": True, "role": entry.get("role"),
                         "attached": False})
    if args.json:
        print(json.dumps(rows, indent=1))
    else:
        for r in rows:
            flags = " ".join(f for f, on in (("intrinsics", r.get("intrinsics")), ("floor", bool(r.get("floor"))),
                                             ("DETACHED", r.get("attached") is False)) if on)
            print(f"{r['stable_id']:22s} {str(r.get('name')):28s} role={str(r.get('role')):6s} "
                  f"usb={str(r.get('usb_controller')):6s} {flags}")
            if r.get("note"):
                print(f"    {r['note']}")
    return 0


def cmd_assign(args: argparse.Namespace, doc: dict[str, Any]) -> int:
    devices = {str(d.get("stable_id")): d for d in (_discover() if not args.offline else [])}
    try:
        sid = resolve(doc, args.camera)
    except KeyError:
        matches = [s for s, d in devices.items() if s == args.camera or d.get("name") == args.camera]
        if len(matches) != 1:
            print(f"{args.camera!r} is neither a registered camera nor exactly one attached device", file=sys.stderr)
            return 2
        sid = matches[0]
    name = devices.get(sid, {}).get("name") or doc.get("cameras", {}).get(sid, {}).get("device_name") or ""
    size = tuple(int(v) for v in args.capture_size.lower().split("x")) if args.capture_size else None
    assign(doc, sid, device_name=name, role=args.role, capture_size=size, fps=args.fps, rotate_180=args.rotate_180,
           notes=args.notes, rotate=args.rotate)
    save_registry(doc, args.registry)
    print(json.dumps({sid: doc["cameras"][sid]}, indent=1))
    return 0


def cmd_adopt(args: argparse.Namespace, doc: dict[str, Any]) -> int:
    adopt(doc, args.old_id, args.new_id)
    save_registry(doc, args.registry)
    print(f"{args.old_id} -> {args.new_id}: entry moved, calibration kept")
    return 0


def cmd_show(args: argparse.Namespace, doc: dict[str, Any]) -> int:
    print(json.dumps(doc, indent=1))
    return 0


def cmd_import_intrinsics(args: argparse.Namespace, doc: dict[str, Any]) -> int:
    doc, imported = import_intrinsics(doc, json.loads(Path(args.file).expanduser().read_text()))
    save_registry(doc, args.registry)
    print(f"imported intrinsics for {imported or 'no cameras'} into {registry_path(args.registry)}")
    return 0


def cmd_snapshot(args: argparse.Namespace, doc: dict[str, Any]) -> int:
    with Rig(doc, [args.role]) as rig:
        cam = rig.camera(args.role)
        frame = None
        for _ in range(10):
            frame = cam.grab()
            if frame is not None:
                break
        if frame is None:
            print(f"{args.role}: no frame ({cam.error})", file=sys.stderr)
            return 1
        obs = cam.detect(frame)
        cv2.imwrite(args.out, frame.bgr)
        print(json.dumps({"out": args.out, "size": [frame.width, frame.height], "tags": sorted(obs.tags),
                          "duplicates": obs.duplicates}))
    return 0


def cmd_record(args: argparse.Namespace, doc: dict[str, Any]) -> int:
    out = Path(args.out)
    summary = run_session(doc, out.parent if out.suffix else out, roles=[args.role], hz=args.hz, video=True,
                          video_fps=args.fps, seconds=args.seconds, stdin=None, stop_file=True)
    print(json.dumps(summary, indent=1))
    return 0


def cmd_calibrate_floor(args: argparse.Namespace, doc: dict[str, Any]) -> int:
    with Rig(doc, [args.role]) as rig:
        try:
            saved, report = rig.fit_floor(args.role, frames=args.frames, interval_s=args.interval)
        except RuntimeError as exc:
            print(f"refusing to save: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(report, indent=1))
        if args.dry_run:
            return 0
        cam = rig.camera(args.role)
        doc["cameras"][cam.stable_id]["floor"] = saved
        save_registry(doc, args.registry)
        print(f"saved floor fit for {args.role} ({cam.stable_id}) to {registry_path(args.registry)}")
    return 0


def cmd_calibrate_intrinsics(args: argparse.Namespace, doc: dict[str, Any]) -> int:
    from . import fit_intrinsics as fi

    with Rig(doc, [args.role]) as rig:
        cam = rig.camera(args.role)

        def grab_gray() -> np.ndarray:
            frame = None
            for _ in range(5):
                frame = cam.grab()
                if frame is not None:
                    break
            if frame is None:
                raise fi.FitError(f"{args.role}: no frame ({cam.error})")
            return frame.gray

        try:
            observations, size, seen = fi.collect_observations_from(grab_gray, args.frames, args.interval, rig.estimator)
            fit = fi.fit_focal_length(observations, size, sorted(seen))
            centers = np.array([rig.estimator.anchors[t]["center"][:2] for t in fit.anchors], dtype=np.float64)
            fi.check_conditioning(fit, centers)
        except fi.FitError as exc:
            print(f"refusing to save: {exc}", file=sys.stderr)
            return 1
        cap = cam.capture.capture_info() if hasattr(cam.capture, "capture_info") else {}
        capture_size = tuple(cap.get("capture_image_size_px") or ()) or None
        entry = fi.intrinsics_entry(fit, size, stable_id=cam.stable_id, device_name=str(cam.entry.get("device_name") or ""),
                                    capture_size=capture_size)
        print(json.dumps({k: entry[k] for k in ("image_size", "camera_matrix", "floor_reprojection_rms_px",
                                                  "horizontal_fov_deg", "quality")}, indent=1))
        if args.dry_run:
            return 0
        doc["cameras"][cam.stable_id]["intrinsics"] = {k: v for k, v in entry.items() if k not in ("stable_id", "device_name")}
        save_registry(doc, args.registry)
        print(f"saved intrinsics for {args.role} ({cam.stable_id})")
    return 0


def cmd_check(args: argparse.Namespace, doc: dict[str, Any]) -> int:
    with Rig(doc, [args.role]) as rig:
        report = rig.check_floor(args.role, frames=args.frames, drift_px=args.drift_px)
        if args.out:
            cam = rig.camera(args.role)
            if cam.last is not None:
                from .camera_server import annotate_tag_corners

                obs = cam.detect(cam.last)
                img = cam.last.bgr.copy()
                annotate_tag_corners(img, obs.tags, cam.index)      # draws in place
                cv2.imwrite(args.out, img)
                report["image"] = args.out
        print(json.dumps(report, indent=1))
        return 0 if report.get("ok") else 3


def cmd_session(args: argparse.Namespace, doc: dict[str, Any]) -> int:
    roles = [r.strip() for r in args.roles.split(",") if r.strip()]
    native = [r.strip() for r in (args.native_record or "").split(",") if r.strip()]
    summary = run_session(doc, Path(args.out), roles=roles, hz=args.hz, aux_hz=args.aux_hz, primary=args.primary,
                          video=not args.no_video, video_fps=args.fps, native_record=native,
                          seconds=args.seconds, stdin=(None if args.no_stdin else sys.stdin), stop_file=True)
    print(json.dumps(summary, indent=1))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hexapod-cameras", description=__doc__.split("\n\n")[0])
    p.add_argument("--registry", default=None, help=f"registry file (default ${REGISTRY_ENV} or {DEFAULT_REGISTRY})")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("list", help="attached cameras and what the registry knows about them")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_list)

    s = sub.add_parser("assign", help="give a camera a role and capture settings")
    s.add_argument("camera", help="stable id, device name, or existing role")
    s.add_argument("--role", help="top, side, ... (one camera per role)")
    s.add_argument("--capture-size", help="WxH to pin, e.g. 1920x1080")
    s.add_argument("--fps", type=float)
    s.add_argument("--rotate-180", action="store_true", default=None)
    s.add_argument("--rotate", type=int, default=None, choices=(0, 90, 180, 270),
                   help="rotate every frame clockwise by this many degrees (a camera mounted sideways); replaces --rotate-180")
    s.add_argument("--notes")
    s.add_argument("--offline", action="store_true", help="do not enumerate devices (edit the registry only)")
    s.set_defaults(fn=cmd_assign)

    s = sub.add_parser("adopt", help="move an entry to the id a replugged camera has now")
    s.add_argument("old_id")
    s.add_argument("new_id")
    s.set_defaults(fn=cmd_adopt)

    s = sub.add_parser("show", help="print the registry")
    s.set_defaults(fn=cmd_show)

    s = sub.add_parser("import-intrinsics", help="bring an identity-keyed camera_intrinsics_*.json into the registry")
    s.add_argument("file")
    s.set_defaults(fn=cmd_import_intrinsics)

    s = sub.add_parser("snapshot", help="one frame from a role to a JPEG")
    s.add_argument("--role", default="top")
    s.add_argument("out")
    s.set_defaults(fn=cmd_snapshot)

    s = sub.add_parser("record", help="record a role to <dir>/<role>.mp4 for --seconds")
    s.add_argument("--role", default="top")
    s.add_argument("--seconds", type=float, default=10.0)
    s.add_argument("--fps", type=float, default=10.0)
    s.add_argument("--hz", type=float, default=2.0, help="state.json rate")
    s.add_argument("out", help="output directory")
    s.set_defaults(fn=cmd_record)

    cal = sub.add_parser("calibrate", help="fit and save calibration for a role")
    cs = cal.add_subparsers(dest="what", required=True)
    s = cs.add_parser("floor", help="homography from the surveyed floor tags")
    s.add_argument("--role", default="top")
    s.add_argument("--frames", type=int, default=15)
    s.add_argument("--interval", type=float, default=0.2)
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=cmd_calibrate_floor)
    s = cs.add_parser("intrinsics", help="focal length from the floor anchors (hexapod-fit-intrinsics, in-process)")
    s.add_argument("--role", default="top")
    s.add_argument("--frames", type=int, default=40)
    s.add_argument("--interval", type=float, default=0.5)
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=cmd_calibrate_intrinsics)

    s = sub.add_parser("check", help="anchors vs the saved floor fit; exit 3 if the camera moved")
    s.add_argument("--role", default="top")
    s.add_argument("--frames", type=int, default=5)
    s.add_argument("--drift-px", type=float, default=6.0)
    s.add_argument("--out", help="annotated JPEG")
    s.set_defaults(fn=cmd_check)

    s = sub.add_parser("session", help="own the cameras for one run and write state/video into --out")
    s.add_argument("--out", required=True)
    s.add_argument("--roles", default=",".join(DEFAULT_ROLES))
    s.add_argument("--hz", type=float, default=20.0, help="state.json / vision.jsonl rate, paced by the primary camera's frames")
    s.add_argument("--aux-hz", type=float, default=2.0, help="tag detection rate of every camera but the primary")
    s.add_argument("--primary", default=None, help="the camera that sets the pace (default: top if present, else the first role)")
    s.add_argument("--fps", type=float, default=10.0, help="video frame rate (at most; capped by --hz)")
    s.add_argument("--native-record", default="", metavar="ROLES",
                   help="comma list of roles, 'primary' or 'all', recorded at the camera's own frame rate, "
                        "independent of --hz/--fps; with 'all' the state loop never converts a frame just for video "
                        "(2026-09-23: three gated 15 fps streams cost the loop 10 Hz)")
    s.add_argument("--seconds", type=float, default=None)
    s.add_argument("--no-video", action="store_true")
    s.add_argument("--no-stdin", action="store_true", help="do not stop when stdin closes")
    s.set_defaults(fn=cmd_session)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    doc = load_registry(args.registry)
    try:
        return int(args.fn(args, doc))
    except (KeyError, FileNotFoundError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
