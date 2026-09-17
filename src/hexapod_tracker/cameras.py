"""Cameras without a daemon.

Three things live here:

* **The registry** -- one JSON file per machine (``~/.hexapod/cameras.json``,
  override with ``HEXAPOD_CAMERAS_FILE``) that says, per camera, what it is
  and what we know about it: its role (``top``, ``side``, ...), the capture
  size and frame rate to open it at, its intrinsics, and its saved floor fit
  (homography from the floor AprilTags, with the numbers that say how good
  the fit was). Cameras are keyed by their AVFoundation ``uniqueID`` (the
  "stable id"), never by a slot number. That id follows the USB port, not the
  device, so every entry also carries ``device_name`` and ``adopt`` moves an
  entry to a new id when a camera is replugged elsewhere.

* **A per-run capture** -- :class:`Rig` opens exactly the cameras a job asks
  for, by role, and releases them when the job ends. It detects tags on the
  full-resolution luma and publishes two documents in the same shapes the old
  HTTP server served (``/api/detections.json`` and ``/api/poses``), so
  everything that parsed those keeps working.

* **A session** -- ``hexapod-cameras session --out DIR --roles top --video``
  is the process a run starts. For the life of the run it writes, into
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
           rotate_180: bool | None = None, notes: str | None = None) -> dict[str, Any]:
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


class Camera:
    """One opened camera: frames plus tag detection, released on close."""

    def __init__(self, role: str, index: int, stable_id: str, entry: dict[str, Any], *,
                 capture_factory: Callable[[int, str, dict[str, Any]], Any] = default_capture_factory,
                 clock: Callable[[], float] = time.time):
        self.role, self.index, self.stable_id, self.entry = role, index, stable_id, entry
        self._factory = capture_factory
        self._clock = clock
        self.capture: Any = None
        self.detector = make_tag_detector()
        self.seq = 0
        self.detect_seq = 0
        self.last: Frame | None = None
        self.error: str | None = None

    def open(self) -> None:
        self.capture = self._factory(self.index, self.stable_id, self.entry)
        if not self.capture.isOpened():
            self.error = getattr(self.capture, "last_error", None) or "camera did not open"
            raise RuntimeError(f"{self.role} ({self.entry.get('device_name')} {self.stable_id}): {self.error}")

    def grab(self) -> Frame | None:
        ok, bgr = self.capture.read()
        if not ok or bgr is None:
            self.error = getattr(self.capture, "last_error", None) or "no frame"
            return None
        if self.entry.get("rotate_180"):
            bgr = cv2.rotate(bgr, cv2.ROTATE_180)
        gray = getattr(self.capture, "detection_gray", None)
        if gray is None or getattr(gray, "ndim", 0) != 2 or gray.shape[1] < bgr.shape[1]:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        elif self.entry.get("rotate_180"):
            gray = cv2.rotate(gray, cv2.ROTATE_180)
        self.seq += 1
        self.error = None
        self.last = Frame(bgr, gray, self._clock(), self.seq)
        return self.last

    def detect(self, frame: Frame) -> Observation:
        corners, duplicates = detect_tag_corners_with_duplicates(frame.gray, self.detector)
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
                               clock=clock) for i, role in enumerate(self.roles)]
        self.configs = load_configs(doc, configs)
        self.estimator = PlanarPoseEstimator(self.configs["floor_map"], self.configs["part_map"],
                                             self.configs["robot_layout"], calibration_for_slots(self.cameras))
        self.estimator.hold_calibration_s = 1e9          # a run is shorter than any sensible hold
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
        anchors = [t for t in self.estimator.active_anchor_ids if t in tags]
        if len(anchors) < 3:
            raise RuntimeError(f"{role}: only floor anchors {anchors} seen (need 3+ of {self.estimator.active_anchor_ids})")
        cal = self.estimator._calibrate_camera({"index": cam.index, "width": size[0], "height": size[1], "tags": tags})
        if cal is None:
            raise RuntimeError(f"{role}: homography did not fit")
        floor_path = Path(self.configs["paths"]["floor_map"])
        saved = floor_fit_to_json(cal, size, floor_map_path=str(floor_path), floor_map_digest=file_digest(floor_path),
                                  frames=len(obs_list), clock=self.clock)
        report = {"role": role, "stable_id": cam.stable_id, "anchors_seen": anchors, "frames_used": len(obs_list),
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
        anchors = [t for t in self.estimator.active_anchor_ids if t in tags]
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
        world = np.concatenate([self.estimator.anchor_corners(t) for t in anchors])
        image = np.concatenate([tags[t] for t in anchors])
        pts = cv2.perspectiveTransform(world.reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)
        residual = np.linalg.norm(pts - image, axis=1)
        report.update({"drift_rms_px": round(float(np.sqrt(np.mean(residual ** 2))), 2),
                       "drift_max_px": round(float(residual.max()), 2), "drift_limit_px": drift_px})
        report["ok"] = report["drift_rms_px"] <= drift_px
        if not report["ok"]:
            report["reason"] = "anchors moved in the picture: the camera was bumped or the floor map changed; refit"
        return report


# --------------------------------------------------------------------------- video


class VideoRecorder:
    """Frames -> H.264 mp4 through an ffmpeg pipe, with a CSV of capture times per frame."""

    def __init__(self, path: Path, fps: float, size: tuple[int, int], *, ffmpeg: str = FFMPEG,
                 popen: Callable[..., Any] = subprocess.Popen):
        self.path = Path(path)
        self.fps = float(fps)
        self.size = size
        self.frames = 0
        self._ts = open(self.path.with_name(self.path.stem + "_timestamps.csv"), "w", newline="")
        self._csv = csv.writer(self._ts)
        self._csv.writerow(["frame", "captured_unix", "seq"])
        self.proc = popen([ffmpeg, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", f"{size[0]}x{size[1]}", "-r", f"{self.fps:g}", "-i", "-",
                           "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
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


def run_session(doc: dict[str, Any], out_dir: Path, *, roles: Sequence[str] = DEFAULT_ROLES, hz: float = 5.0,
                video: bool = True, video_fps: float = 10.0, seconds: float | None = None, stdin: Any = None,
                stop_file: bool = True, capture_factory: Callable[..., Any] = default_capture_factory,
                clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep,
                recorder_factory: Callable[..., Any] = VideoRecorder, configs: Path = CONFIG_DIR,
                log: Callable[[str], None] = print) -> dict[str, Any]:
    """Own the cameras for one run; write state.json / latest_<role>.jpg / vision.jsonl / <role>.mp4."""
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
    summary: dict[str, Any] = {"out_dir": str(out_dir), "roles": list(roles), "states": 0, "frames": {}, "video": {},
                               "started_unix": clock()}
    period = 1.0 / max(0.5, hz)
    with Rig(doc, roles, configs=configs, capture_factory=capture_factory, clock=clock, log=log) as rig:
        write_atomic(out_dir / "session.json", json.dumps({"pid": os.getpid(), "roles": list(roles),
                                                            "cameras": [c.info() for c in rig.cameras],
                                                            "started_unix": summary["started_unix"]}, indent=1))
        t0 = clock()
        next_state = t0
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
                # Detect tags only when a state is due: full-frame detection on every
                # camera every frame held a two-camera session to ~7 fps (2026-09-17);
                # the frames in between are video only.
                state_due = now >= next_state
                observations = rig.observe(detect=state_due)
                for obs in observations:
                    if obs.frame is None:
                        continue
                    summary["frames"][obs.role] = summary["frames"].get(obs.role, 0) + 1
                    if video:
                        rec = recorders.get(obs.role)
                        if rec is None:
                            rec = recorders[obs.role] = recorder_factory(out_dir / f"{obs.role}.mp4", video_fps,
                                                                        (obs.frame.width, obs.frame.height))
                        rec.write(obs.frame)
                if state_due:
                    next_state = now + period
                    state = rig.state(observations)
                    write_atomic(out_dir / "state.json", json.dumps(state))
                    for obs in observations:
                        if obs.frame is not None:
                            ok, jpg = cv2.imencode(".jpg", obs.frame.bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                            if ok:
                                tmp = out_dir / f"latest_{obs.role}.jpg.tmp"
                                tmp.write_bytes(jpg.tobytes())
                                os.replace(tmp, out_dir / f"latest_{obs.role}.jpg")
                    markers = state["poses"].get("markers") or {}
                    jsonl.write(json.dumps({
                        "seq": state["seq"], "capture_unix": state["generated_at_unix_s"],
                        "tags": {c["role"]: sorted(int(t) for t in c["tags"]) for c in state["cameras"]},
                        "markers": {k: {"status": m.get("status"), "position_mm": m.get("position_mm"),
                                        "yaw": (m.get("rotation_degrees") or {}).get("yaw")}
                                    for k, m in markers.items() if m.get("status") == "tracked"},
                    }) + "\n")
                    jsonl.flush()
                    summary["states"] = state["seq"]
                # pace to the video rate; state writes are gated above
                spent = clock() - now
                sleep(max(0.0, 1.0 / max(1.0, video_fps if video else hz) - spent))
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
           notes=args.notes)
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
    summary = run_session(doc, Path(args.out), roles=roles, hz=args.hz, video=not args.no_video, video_fps=args.fps,
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
    s.add_argument("--hz", type=float, default=5.0, help="state.json / vision.jsonl rate")
    s.add_argument("--fps", type=float, default=10.0, help="video frame rate")
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
