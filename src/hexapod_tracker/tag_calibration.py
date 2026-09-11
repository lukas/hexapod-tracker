"""Calibrate the hexapod's AprilTag layout by moving the robot and watching.

    hexapod-calibrate-tags [--out DIR] [--write] [--no-claude]
    hexapod-calibrate-tags --replay DIR [--assign-from REPORT ...] [--write]

The tracker needs a small layout: for every tag, which link it is on (body,
L<n>_coxa / _femur / _tibia), whether it is a horizontal servo lid or a
vertical yoke face, which side a yoke face is on, and the rotation from tag
axes to link axes. Translations are never read, so they stay null.

The program runs in stages, each of which writes what it saw into ``--out``:

1. **Stability.** Two looks at the resting robot a few seconds apart, then a
   small hip lift of one leg and back. If a floor tag or the chassis tag moved
   the robot is not resting flat or a camera is moving (or refocusing), and
   nothing measured afterwards would mean anything, so the run stops.
2. **Motion pass.** One leg at a time, lifted clear of the floor: a hip move
   carries femur + tibia tags, an antisymmetric yaw swing adds the coxa, a
   knee move carries the tibia only. Tags that never move are body or floor.
   Because the moves go through the servo joint indices, leg numbering in the
   layout is the servo numbering by construction. The same swing also
   records which way a positive yaw turns the leg as seen from above.
3. **Geometry** (`derive_layout`, a pure function of the zero-pose corners).
   Everything happens in the top camera's lid plane: a focal length is fitted
   so the flat tags agree on one normal, corners are back-projected onto that
   plane, and directions in the plane are exact up to that fit. The leg's
   direction is hip lid -> knee lid; lid rotation is the angle from that
   direction to the tag's +x, snapped to 90 degrees; a yoke face's side is
   which side of the leg axis it sits on, and its +x is whichever chord lies
   along the leg (horizontal) or, if neither does, the vertical chord with its
   higher end farther from the camera's foot. Body +x is the circular mean of
   every leg's direction rotated back by its nominal azimuth; the legs'
   measured azimuths and the chassis tag rotation follow from it.
4. **Assembly.** Faces nobody saw are carried from the previous layout with
   ``verified: false``; mounts with no tag at all are declared in
   ``unresolved_mounts`` so the validator can tell a known gap from a mistake.
   The report lists the diff against the previous layout in plain words.
5. **Claude** (optional) reads annotated crops with the ids drawn on, as a
   cross-check on placement and a census of blank servo faces. It cannot
   decode tags and never decides anything on its own.

``--write`` installs the layout and tag map into configs/ (old files backed
up). It refuses when the validator reports problems unless ``--force`` is
given; declared gaps are notes, not problems.

To confirm faces the cameras could not see, turn the robot (by hand or with
the gait), run another pass, and merge with ``--assign-from`` pointing at each
earlier report.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import math
import os
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .camera_server import detect_tag_corners_with_duplicates, make_tag_detector
from .layout_audit import AXIS_VECTORS, declared_gap_notes, validate_layout
from .paths import CONFIG_DIR

TAG_M = 0.0272
LEGS = range(6)
LINKS = ("coxa", "femur", "tibia")
JOINTS = ("hip", "knee")
# Lifted-leg test angles (robot_abs degrees). Negative hip lifts the femur;
# negative knee lifts the tibia, so the foot never touches the floor.
HIP_LIFT = -30.0
YAW_SWING = 18.0
KNEE_LIFT = -20.0
STABILITY_LIFT = -15.0
MOVE_S = 2.0
SETTLE_S = 1.2
TORQUE = 500
FRAMES_PER_STATE = 3
# Motion evidence: a tag "moved" if its centre shifted more than this many
# tag-edges, or its in-plane angle changed more than MOVE_DEG.
MOVE_EDGE_FRAC = 0.25
MOVE_PX_FLOOR = 6.0
MOVE_DEG = 6.0
MOVE_SCALE_FRAC = 0.08
# The commanded joint must travel at least this far for a step to count.
MIN_TRAVEL_DEG = 8.0


def nominal_azimuth_deg(leg: int) -> float:
    """Where leg ``leg`` points at the zero pose, in the tracker's body frame.

    The body frame is right-handed with +z up and +x forward, between legs 0
    and 5. Seen from above the legs are numbered clockwise, so leg 0 sits at
    -30 degrees, leg 1 at -90 and so on. The gait code's own frame puts leg i
    at +(i+0.5)*60 and calls a clockwise-from-above yaw positive: both are
    right-handed about a z axis that points DOWN. That frame (x forward, y
    right, z down) is this one rotated 180 degrees about x; it is not a
    reflection, so pitch signs about the leg's y axis are unaffected.
    """
    return float(((-(leg + 0.5) * 60.0 + 180.0) % 360.0) - 180.0)


class Log:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, msg: str) -> None:
        print(msg, flush=True)
        self.lines.append(msg)


# ----------------------------------------------------------------- capture

class Cameras:
    """Frames and tag corners from the camera server.

    Corners come from the server's own detector via ``/api/detections.json``
    when the running server has it (it detects on the full sensor and scales
    into snapshot coordinates); otherwise they are detected here on the
    snapshot JPEG. Either way an observation is a majority vote over a few
    frames, and a camera that gives no frame is skipped for that state.
    """

    def __init__(self, base: str, indices: list[int]):
        self.base = base.rstrip("/")
        self.indices = indices
        self.detector = make_tag_detector()
        self.size: dict[int, tuple[int, int]] = {}
        self.server_detections: Optional[bool] = None   # unknown until first tried
        self.duplicate_ids: dict[int, set[int]] = {}    # camera -> ids decoded twice in one frame

    def snapshot(self, i: int) -> np.ndarray:
        with urllib.request.urlopen(f"{self.base}/snapshot/{i}.jpg", timeout=6) as r:
            data = np.frombuffer(r.read(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"camera {i}: bad JPEG")
        self.size[i] = (img.shape[1], img.shape[0])
        return img

    def _server_corners(self) -> Optional[dict[int, dict[str, Any]]]:
        if self.server_detections is False:
            return None
        try:
            with urllib.request.urlopen(f"{self.base}/api/detections.json", timeout=4) as r:
                doc = json.loads(r.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                self.server_detections = False
                print("camera server has no /api/detections.json; detecting on snapshots", flush=True)
            return None
        except (urllib.error.URLError, OSError, ValueError):
            return None
        self.server_detections = True
        return {int(c["index"]): c for c in doc.get("cameras", [])}

    def _note_duplicates(self, cam: int, ids: list) -> None:
        new = set(int(t) for t in ids) - self.duplicate_ids.setdefault(cam, set())
        if new:
            self.duplicate_ids[cam] |= new
            print(f"camera {cam}: ids {sorted(new)} decoded at two places in one frame (a spare tag in view?); "
                  f"those ids are ignored while it lasts. Move spare tags out of the frame.", flush=True)

    def observe(self, frames: int = FRAMES_PER_STATE) -> dict[int, dict[str, Any]]:
        """Per camera: ``{"tags": {id: 4x2 corners}, "image": last frame, "size": (w, h)}``."""
        out: dict[int, dict[str, Any]] = {}
        seen: dict[int, dict[int, list[np.ndarray]]] = {i: {} for i in self.indices}
        images: dict[int, np.ndarray] = {}
        for i in self.indices:
            try:
                images[i] = self.snapshot(i)
            except (urllib.error.URLError, RuntimeError, OSError) as exc:
                print(f"camera {i}: no frame ({exc}); skipping this camera for this state", flush=True)
        last_seq: dict[int, Any] = {}
        for k in range(frames):
            server = self._server_corners()
            for i in list(images):
                used_server = False
                if server and i in server and server[i].get("tags") is not None:
                    c = server[i]
                    fresh = c.get("detect_seq") != last_seq.get(i) and (c.get("frame_age_s") or 0.0) < 2.0
                    if fresh:
                        last_seq[i] = c.get("detect_seq")
                        self._note_duplicates(i, c.get("duplicate_ids") or [])
                        w = float(c.get("width") or images[i].shape[1])
                        scale = images[i].shape[1] / w
                        for tid, corners in c["tags"].items():
                            seen[i].setdefault(int(tid), []).append(np.asarray(corners, dtype=np.float64) * scale)
                        used_server = True
                if not used_server:
                    if k > 0:
                        try:
                            images[i] = self.snapshot(i)
                        except (urllib.error.URLError, RuntimeError, OSError):
                            continue
                    gray = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
                    corners_by_id, duplicates = detect_tag_corners_with_duplicates(gray, self.detector)
                    self._note_duplicates(i, duplicates)
                    for tid, corners in corners_by_id.items():
                        seen[i].setdefault(tid, []).append(np.asarray(corners, dtype=np.float64))
            time.sleep(0.15)
        need = max(1, (frames + 1) // 2)
        for i, img in images.items():
            tags = {tid: np.median(np.stack(c), axis=0) for tid, c in seen[i].items() if len(c) >= need}
            out[i] = {"tags": tags, "image": img, "size": self.size[i]}
        return out


def tags_only(obs: dict[int, dict[str, Any]]) -> dict[int, dict[int, np.ndarray]]:
    return {c: o["tags"] for c, o in obs.items()}


def save_tags(path: Path, obs: dict[int, dict[str, Any]]) -> None:
    path.write_text(json.dumps({"cameras": {
        str(c): {"size": list(o["size"]), "tags": {str(t): crn.tolist() for t, crn in o["tags"].items()}}
        for c, o in obs.items()}}, indent=0))


def load_tags(path: Path) -> tuple[dict[int, dict[int, np.ndarray]], dict[int, tuple[int, int]]]:
    """Read a saved zero observation; also understands the first tool's flat format."""
    doc = json.loads(path.read_text())
    tags: dict[int, dict[int, np.ndarray]] = {}
    sizes: dict[int, tuple[int, int]] = {}
    if "cameras" in doc:
        for c, o in doc["cameras"].items():
            tags[int(c)] = {int(t): np.asarray(v, dtype=np.float64) for t, v in o["tags"].items()}
            sizes[int(c)] = tuple(int(v) for v in o["size"])
    else:
        for c, o in doc.items():
            tags[int(c)] = {int(t): np.asarray(v, dtype=np.float64) for t, v in o.items()}
            raw = path.parent / f"zero_cam{c}_raw.jpg"
            img = cv2.imread(str(raw)) if raw.exists() else None
            if img is not None:
                sizes[int(c)] = (img.shape[1], img.shape[0])
    return tags, sizes


# ------------------------------------------------------------------- robot

class Robot:
    def __init__(self, base: str, dry: bool = False):
        self.base = base.rstrip("/")
        self.dry = dry

    def _json(self, path: str, body: Optional[dict] = None, method: str = "GET") -> dict:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"} if data else {})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())

    def pose(self) -> list[float]:
        """Present joint angles from /api/feedback. Never GET /api/pose here: that
        read releases servo torque (observed 2026-09-11), which aborts a glide in
        progress and lets an unsupported leg sag."""
        last: Optional[list[float]] = None
        for _ in range(4):
            joints = self._json("/api/feedback").get("joints") or []
            if len(joints) == 18 and all(isinstance(j, dict) and j.get("deg") is not None for j in joints):
                last = [float(j["deg"]) for j in joints]
                return last
            time.sleep(0.2)                         # one servo missed its slot; read again
        if last is None:
            raise RuntimeError("feedback never returned 18 joint angles")
        return last

    def move(self, q: list[float], *, seconds: float = MOVE_S, torque: int = TORQUE) -> dict:
        """Command a pose and wait until every joint is there or has stopped."""
        if self.dry:
            return {"ok": True, "dry": True, "short": []}
        res = self._json("/api/pose", {"q_deg": [round(v, 2) for v in q], "seconds": seconds,
                                       "torque": torque, "label": "tag-calibration"}, "POST")
        if not res.get("ok"):
            raise RuntimeError(f"pose refused: {res}")
        # The servo profile caps joint speed near 30 deg/s, so a long glide outlasts
        # `seconds`. Wait until every joint is within 3 deg or has stopped moving.
        deadline = time.monotonic() + max(seconds, 1.0) + 10.0
        last = None
        time.sleep(seconds)
        while time.monotonic() < deadline:
            present = self.pose()
            if max(abs(present[j] - q[j]) for j in range(18)) < 3.0:
                break
            if last is not None and max(abs(present[j] - last[j]) for j in range(18)) < 0.5:
                break                                   # stopped short: blocked or at a limit
            last = present
            time.sleep(0.4)
        time.sleep(SETTLE_S)
        present = self.pose()
        return {**{k: v for k, v in res.items() if k in ("ok", "label", "worst_delta_deg", "peak_a")},
                "reached": [round(v, 1) for v in present],
                "short": [(j, round(present[j] - q[j], 1)) for j in range(18) if abs(present[j] - q[j]) > 5.0]}

    def relax(self) -> None:
        if self.dry:
            return
        req = urllib.request.Request(self.base + "/cmd", data=b"RELAX", method="POST")
        with urllib.request.urlopen(req, timeout=10) as r:
            r.read()


# ------------------------------------------------------------- motion logic

def center(corners: np.ndarray) -> np.ndarray:
    return corners.mean(axis=0)


def edge_px(corners: np.ndarray) -> float:
    return float(np.mean([np.linalg.norm(corners[k] - corners[(k + 1) % 4]) for k in range(4)]))


def angle_img(corners: np.ndarray) -> float:
    v = corners[1] - corners[0]
    return math.degrees(math.atan2(v[1], v[0]))


def turn_deg(ca: np.ndarray, cb: np.ndarray) -> float:
    """In-plane turn of a tag between two states, degrees in image coordinates.

    Image y points down, so a positive value is clockwise on screen, which for
    a camera looking down at the robot is clockwise as seen from above."""
    return (angle_img(cb) - angle_img(ca) + 180.0) % 360.0 - 180.0


def moved(a: dict[int, np.ndarray], b: dict[int, np.ndarray]) -> tuple[set[int], set[int]]:
    """Tags seen whole in both a and b that moved / stayed, by centre shift or in-plane turn.

    A tag whose apparent size changed a lot between the two states was
    partly hidden in one of them; its corners are not evidence either way."""
    mv, st = set(), set()
    for tid in a.keys() & b.keys():
        ca, cb = a[tid], b[tid]
        ratio = edge_px(cb) / (edge_px(ca) + 1e-9)
        if not 0.75 < ratio < 1.33:
            continue
        shift = float(np.linalg.norm(center(ca) - center(cb)))
        turn = abs(turn_deg(ca, cb))
        thr = max(MOVE_PX_FLOOR, MOVE_EDGE_FRAC * edge_px(ca))
        # A lid lifted toward a top camera may barely shift or turn but grows;
        # autofocus and frame noise change a median-of-3 size by well under this.
        grew = abs(ratio - 1.0) > MOVE_SCALE_FRAC and edge_px(ca) > 12.0
        (mv if shift > thr or turn > MOVE_DEG or grew else st).add(tid)
    return mv, st


def moved_antisymmetric(base: dict[int, np.ndarray], plus: dict[int, np.ndarray],
                        minus: dict[int, np.ndarray]) -> tuple[set[int], set[int]]:
    """For a +/- joint swing about a vertical axis: a tag on the moving link turns one
    way then the other. Occlusion flicker does not."""
    mv, st = set(), set()
    for tid in base.keys() & plus.keys() & minus.keys():
        cb, cp, cm = base[tid], plus[tid], minus[tid]
        if not (0.75 < edge_px(cp) / (edge_px(cb) + 1e-9) < 1.33 and 0.75 < edge_px(cm) / (edge_px(cb) + 1e-9) < 1.33):
            continue
        tp, tm = turn_deg(cb, cp), turn_deg(cb, cm)
        sp = float(np.linalg.norm(center(cp) - center(cb))); sm = float(np.linalg.norm(center(cm) - center(cb)))
        thr = max(MOVE_PX_FLOOR, MOVE_EDGE_FRAC * edge_px(cb))
        turned = abs(tp) > MOVE_DEG and abs(tm) > MOVE_DEG and (tp > 0) != (tm > 0)
        shifted = sp > thr and sm > thr and float(np.dot(center(cp) - center(cb), center(cm) - center(cb))) < 0
        (mv if turned or shifted else st).add(tid)
    return mv, st


def merge_votes(per_camera: list[tuple[set[int], set[int]]]) -> tuple[set[int], set[int], set[int]]:
    """Union over cameras; a tag both moved and stayed in different cameras is a conflict."""
    mv = set().union(*(m for m, _ in per_camera)) if per_camera else set()
    st = set().union(*(s for _, s in per_camera)) if per_camera else set()
    # Motion seen anywhere is evidence; "stayed" can be a small move under the
    # threshold in a far camera. Moved wins, and the disagreement is reported.
    return mv, st - mv, mv & st


def partition_leg(zero: dict, lifted: dict, yawed: list[dict], knee: dict) -> dict[str, Any]:
    """Assign tags of one leg to coxa/femur/tibia from the three moves."""
    cams = zero.keys() & lifted.keys() & knee.keys()
    for y in yawed:
        cams &= y.keys()
    hip_mv, _, hip_conf = merge_votes([moved(zero[c]["tags"], lifted[c]["tags"]) for c in cams])
    if len(yawed) >= 2:
        yaw_votes = [moved_antisymmetric(lifted[c]["tags"], yawed[0][c]["tags"], yawed[1][c]["tags"]) for c in cams]
    else:
        yaw_votes = [moved(lifted[c]["tags"], yawed[0][c]["tags"]) for c in cams]
    yaw_mv, _, yaw_conf = merge_votes(yaw_votes)
    knee_mv, _, knee_conf = merge_votes([moved(lifted[c]["tags"], knee[c]["tags"]) for c in cams])
    tibia = knee_mv
    femur = hip_mv - tibia
    coxa = yaw_mv - femur - tibia
    # A tag that turned with the knee but not with the hip is contradictory.
    contradictions = sorted((tibia - hip_mv) | (femur & coxa))
    return {"coxa": sorted(coxa), "femur": sorted(femur), "tibia": sorted(tibia),
            "conflicts": sorted(hip_conf | yaw_conf | knee_conf), "contradictions": contradictions,
            "moved_hip": sorted(hip_mv), "moved_yaw": sorted(yaw_mv), "moved_knee": sorted(knee_mv)}


def yaw_sense_from_swing(lifted: dict, plus: dict, coxa_tags: list[int], top: int) -> Optional[str]:
    """Which way the coxa turned, seen from above, for a positive yaw command."""
    if top not in lifted or top not in plus:
        return None
    turns = [turn_deg(lifted[top]["tags"][t], plus[top]["tags"][t])
             for t in coxa_tags if t in lifted[top]["tags"] and t in plus[top]["tags"]]
    turns = [t for t in turns if abs(t) > MOVE_DEG]
    if not turns:
        return None
    return "clockwise" if float(np.median(turns)) > 0 else "counterclockwise"


# ---------------------------------------------------------------- geometry

def approx_intrinsics(width: int, height: int, device: str = "") -> np.ndarray:
    """Good enough for 90-degree decisions and normals; not for metrology."""
    if "OV9281" in device:
        f = 948.0 * width / 1280.0
    else:
        f = 0.85 * width
    return np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1.0]])


def fit_focal(horizontal: list[np.ndarray], width: int, height: int) -> tuple[float, float]:
    """Pick the focal length that makes all horizontal tags' normals parallel.

    Every lid and the chassis tag lie flat at the zero pose, so their IPPE
    normals must agree; a wrong focal length splays them. Returns (f, mean
    normal disagreement in degrees at that f)."""
    best = (0.85 * width, 90.0)
    if len(horizontal) < 3:
        return best
    for f in np.linspace(0.45 * width, 2.2 * width, 71):
        K = np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1.0]])
        up = consensus_up(horizontal, K)
        if up is None:
            continue
        normals = [tag_pose(c, K, up)[0][:, 2] for c in horizontal]
        n = np.stack(normals)
        spread = float(np.degrees(np.mean(np.arccos(np.clip(n @ up, -1, 1)))))
        if spread < best[1]:
            best = (float(f), spread)
    return best


def tag_poses(corners: np.ndarray, K: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Both IPPE-square solutions for one tag: (tag axes in camera, translation m).

    Small tags have a real two-fold ambiguity; callers pick the solution that
    agrees with a known direction (the shared 'up' of every flat tag)."""
    h = TAG_M / 2.0
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)
    ok, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, corners.astype(np.float64), K, None,
                                                 flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        raise RuntimeError("PnP failed")
    return [(cv2.Rodrigues(r)[0], t.reshape(3)) for r, t in zip(rvecs, tvecs)]


def tag_pose(corners: np.ndarray, K: np.ndarray, up: Optional[np.ndarray] = None,
             want: str = "parallel") -> tuple[np.ndarray, np.ndarray]:
    """One IPPE solution. With `up`, the one whose normal is most parallel
    (lids) or most perpendicular (yoke faces) to it."""
    sols = tag_poses(corners, K)
    if up is None or len(sols) == 1:
        return sols[0]
    dots = [abs(float(np.dot(R[:, 2], up))) for R, _ in sols]
    idx = int(np.argmax(dots)) if want == "parallel" else int(np.argmin(dots))
    return sols[idx]


def consensus_up(flat: list[np.ndarray], K: np.ndarray) -> Optional[np.ndarray]:
    """Shared normal of the flat tags, choosing each tag's IPPE branch iteratively."""
    if len(flat) < 2:
        return None
    sols = [tag_poses(c, K) for c in flat]
    up = np.mean([s[0][0][:, 2] for s in sols], axis=0)
    for _ in range(4):
        picked = [max(s, key=lambda rt: float(np.dot(rt[0][:, 2], up)))[0][:, 2] for s in sols]
        up = np.mean(picked, axis=0); up /= np.linalg.norm(up) + 1e-9
    return up


def center(corners: np.ndarray) -> np.ndarray:
    return corners.mean(axis=0)


def edge_px(corners: np.ndarray) -> float:
    return float(np.mean([np.linalg.norm(corners[k] - corners[(k + 1) % 4]) for k in range(4)]))


def x_axis_img(corners: np.ndarray) -> np.ndarray:
    v = corners[1] - corners[0]
    return v / (np.linalg.norm(v) + 1e-9)


def angle_img(corners: np.ndarray) -> float:
    v = corners[1] - corners[0]
    return math.degrees(math.atan2(v[1], v[0]))


# ------------------------------------------------------------------- robot

class Robot:
    def __init__(self, base: str, dry: bool = False):
        self.base = base.rstrip("/")
        self.dry = dry

    def _json(self, path: str, body: Optional[dict] = None, method: str = "GET") -> dict:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"} if data else {})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())

    def pose(self) -> list[float]:
        """Present joint angles from /api/feedback. Never GET /api/pose here: that
        read releases servo torque (observed 2026-09-11), which aborts a glide in
        progress and lets an unsupported leg sag."""
        last: Optional[list[float]] = None
        for _ in range(4):
            joints = self._json("/api/feedback").get("joints") or []
            if len(joints) == 18 and all(isinstance(j, dict) and j.get("deg") is not None for j in joints):
                last = [float(j["deg"]) for j in joints]
                return last
            time.sleep(0.2)                         # one servo missed its slot; read again
        if last is None:
            raise RuntimeError("feedback never returned 18 joint angles")
        return last

    def feedback(self) -> dict:
        return self._json("/api/feedback")

    def move(self, q: list[float], *, seconds: float = MOVE_S, torque: int = TORQUE) -> dict:
        """Command a pose and confirm every joint got within 5 deg of it."""
        if self.dry:
            return {"ok": True, "dry": True}
        res = self._json("/api/pose", {"q_deg": [round(v, 2) for v in q], "seconds": seconds,
                                       "torque": torque, "label": "relayout"}, "POST")
        if not res.get("ok"):
            raise RuntimeError(f"pose refused: {res}")
        # The servo profile caps joint speed near 30 deg/s, so a long glide outlasts
        # `seconds`. Wait until every joint is within 3 deg or has stopped moving.
        deadline = time.monotonic() + max(seconds, 1.0) + 10.0
        last = None
        time.sleep(seconds)
        while time.monotonic() < deadline:
            present = self.pose()
            if max(abs(present[j] - q[j]) for j in range(18)) < 3.0:
                break
            if last is not None and max(abs(present[j] - last[j]) for j in range(18)) < 0.5:
                break                                   # stopped short: blocked or at a limit
            last = present
            time.sleep(0.4)
        time.sleep(SETTLE_S)
        present = self.pose()
        res["reached"] = [round(v, 1) for v in present]
        res["short"] = [(j, round(present[j] - q[j], 1)) for j in range(18) if abs(present[j] - q[j]) > 5.0]
        return res

    def relax(self) -> None:
        if self.dry:
            return
        req = urllib.request.Request(self.base + "/cmd", data=b"RELAX", method="POST")
        with urllib.request.urlopen(req, timeout=10) as r:
            r.read()


# ------------------------------------------------------------- motion logic

def turn_deg(ca: np.ndarray, cb: np.ndarray) -> float:
    return (angle_img(cb) - angle_img(ca) + 180.0) % 360.0 - 180.0


def moved(a: dict[int, np.ndarray], b: dict[int, np.ndarray]) -> tuple[set[int], set[int]]:
    """Tags seen whole in both a and b that moved / stayed, by centre shift or in-plane turn.

    A tag whose apparent size changed a lot between the two states was
    partly hidden in one of them; its corners are not evidence either way."""
    mv, st = set(), set()
    for tid in a.keys() & b.keys():
        ca, cb = a[tid], b[tid]
        ratio = edge_px(cb) / (edge_px(ca) + 1e-9)
        if not 0.75 < ratio < 1.33:
            continue
        shift = float(np.linalg.norm(center(ca) - center(cb)))
        turn = abs(turn_deg(ca, cb))
        thr = max(MOVE_PX_FLOOR, MOVE_EDGE_FRAC * edge_px(ca))
        # A lid lifted toward a top camera may barely shift or turn but grows;
        # autofocus and frame noise change a median-of-3 size by well under this.
        grew = abs(ratio - 1.0) > MOVE_SCALE_FRAC and edge_px(ca) > 12.0
        (mv if shift > thr or turn > MOVE_DEG or grew else st).add(tid)
    return mv, st


def moved_antisymmetric(base: dict[int, np.ndarray], plus: dict[int, np.ndarray],
                        minus: dict[int, np.ndarray]) -> tuple[set[int], set[int]]:
    """For a +/- joint swing about a vertical axis: a tag on the moving link turns one
    way then the other. Occlusion flicker does not."""
    mv, st = set(), set()
    for tid in base.keys() & plus.keys() & minus.keys():
        cb, cp, cm = base[tid], plus[tid], minus[tid]
        if not (0.75 < edge_px(cp) / (edge_px(cb) + 1e-9) < 1.33 and 0.75 < edge_px(cm) / (edge_px(cb) + 1e-9) < 1.33):
            continue
        tp, tm = turn_deg(cb, cp), turn_deg(cb, cm)
        sp = float(np.linalg.norm(center(cp) - center(cb))); sm = float(np.linalg.norm(center(cm) - center(cb)))
        thr = max(MOVE_PX_FLOOR, MOVE_EDGE_FRAC * edge_px(cb))
        turned = abs(tp) > MOVE_DEG and abs(tm) > MOVE_DEG and (tp > 0) != (tm > 0)
        shifted = sp > thr and sm > thr and float(np.dot(center(cp) - center(cb), center(cm) - center(cb))) < 0
        (mv if turned or shifted else st).add(tid)
    return mv, st


def merge_votes(per_camera: list[tuple[set[int], set[int]]]) -> tuple[set[int], set[int], set[int]]:
    """Union over cameras; a tag both moved and stayed in different cameras is a conflict."""
    mv = set().union(*(m for m, _ in per_camera)) if per_camera else set()
    st = set().union(*(s for _, s in per_camera)) if per_camera else set()
    # Motion seen anywhere is evidence; "stayed" can be a small move under the
    # threshold in a far camera. Moved wins, and the disagreement is reported.
    return mv, st - mv, mv & st


def partition_leg(zero: dict, lifted: dict, yawed: list[dict], knee: dict) -> dict[str, Any]:
    """Assign tags of one leg to coxa/femur/tibia from the three moves."""
    cams = zero.keys()
    hip_mv, _, hip_conf = merge_votes([moved(zero[c]["tags"], lifted[c]["tags"]) for c in cams])
    if len(yawed) >= 2:
        yaw_votes = [moved_antisymmetric(lifted[c]["tags"], yawed[0][c]["tags"], yawed[1][c]["tags"]) for c in cams]
    else:
        yaw_votes = [moved(lifted[c]["tags"], yawed[0][c]["tags"]) for c in cams]
    yaw_mv, _, yaw_conf = merge_votes(yaw_votes)
    knee_mv, _, knee_conf = merge_votes([moved(lifted[c]["tags"], knee[c]["tags"]) for c in cams])
    tibia = knee_mv
    femur = hip_mv - tibia
    coxa = yaw_mv - femur - tibia
    # A tag that turned with the knee but not with the hip is contradictory.
    contradictions = sorted((tibia - hip_mv) | (femur & coxa))
    return {"coxa": sorted(coxa), "femur": sorted(femur), "tibia": sorted(tibia),
            "conflicts": sorted(hip_conf | yaw_conf | knee_conf), "contradictions": contradictions,
            "moved_hip": sorted(hip_mv), "moved_yaw": sorted(yaw_mv), "moved_knee": sorted(knee_mv)}


# ---------------------------------------------------------------- geometry

def rot_to_tokens(R: np.ndarray) -> dict[str, str]:
    """Columns of R are tag axes in the link frame; snap each to a signed axis."""
    out = {}
    for k, name in enumerate("xyz"):
        v = R[:, k]
        j = int(np.argmax(np.abs(v)))
        out[name] = ("+" if v[j] > 0 else "-") + "xyz"[j]
    return out


def tokens_to_rot(tokens: dict[str, str]) -> np.ndarray:
    return np.column_stack([AXIS_VECTORS[tokens[n]] for n in "xyz"])


def quat_xyzw(R: np.ndarray) -> list[float]:
    q = Rotation.from_matrix(R).as_quat()
    return [round(float(v), 7) for v in q]


def as_rotation(fft: Optional[dict]) -> Optional[Rotation]:
    if not fft:
        return None
    if fft.get("quaternion_xyzw") is not None:
        return Rotation.from_quat(fft["quaternion_xyzw"])
    if fft.get("euler_xyz_deg") is not None:
        return Rotation.from_euler("xyz", fft["euler_xyz_deg"], degrees=True)
    return None


def same_rotation(a: Optional[dict], b: Optional[dict], tol_deg: float = 1.0) -> bool:
    ra, rb = as_rotation(a), as_rotation(b)
    if ra is None or rb is None:
        return ra is rb
    return float(np.degrees((ra.inv() * rb).magnitude())) < tol_deg


def snap90(deg: float) -> tuple[float, float]:
    s = round(deg / 90.0) * 90.0
    s = ((s + 180.0) % 360.0) - 180.0
    if s == -180.0:
        s = 180.0
    return s, ((deg - s + 180.0) % 360.0) - 180.0


def wrap_deg(deg: float) -> float:
    return ((deg + 180.0) % 360.0) - 180.0


def circular_mean_deg(values: list[float]) -> float:
    r = np.radians(values)
    return math.degrees(math.atan2(float(np.mean(np.sin(r))), float(np.mean(np.cos(r)))))


class LidPlane:
    """The top camera's view of the horizontal plane the servo lids lie in.

    Depth from a 27 mm tag is too noisy to use, so every position is the tag
    centre's ray intersected with the shared plane of the flat tags (normal
    ``up`` from the IPPE consensus, pointing from the scene toward the
    camera). Directions in that plane are exact up to the focal-length fit;
    depths cancel. The camera's foot on the plane is at ``-up``.
    """

    def __init__(self, K: np.ndarray, up: np.ndarray):
        self.K_inv = np.linalg.inv(K)
        self.up = up / (np.linalg.norm(up) + 1e-9)
        self.foot = -self.up

    def point(self, pix: np.ndarray) -> np.ndarray:
        ray = self.K_inv @ np.array([pix[0], pix[1], 1.0])
        return -ray / float(np.dot(self.up, ray))

    def in_plane(self, v: np.ndarray) -> np.ndarray:
        v = v - self.up * float(np.dot(v, self.up))
        return v / (np.linalg.norm(v) + 1e-9)

    def angle(self, v_from: np.ndarray, v_to: np.ndarray) -> float:
        """Rotation about +up taking v_from to v_to, degrees (right-handed, z up)."""
        return math.degrees(math.atan2(float(np.dot(np.cross(v_from, v_to), self.up)), float(np.dot(v_from, v_to))))

    def rotate(self, v: np.ndarray, deg: float) -> np.ndarray:
        r = math.radians(deg)
        return self.in_plane(v * math.cos(r) + np.cross(self.up, v) * math.sin(r))

    def tag_x(self, corners: np.ndarray) -> np.ndarray:
        """Tag +x direction on the plane: back-project the corner-0 -> corner-1 chord."""
        return self.in_plane(self.point(corners[1]) - self.point(corners[0]))

    def squareness(self, corners: np.ndarray) -> float:
        """0 for a flat tag (its back-projected quad is square); larger for a vertical face."""
        P = [self.point(corners[k]) for k in range(4)]
        a, b = np.linalg.norm(P[1] - P[0]), np.linalg.norm(P[3] - P[0])
        return abs(math.log((a + 1e-9) / (b + 1e-9)))

    def height_order(self, corners: np.ndarray, k0: int, k1: int) -> bool:
        """True if corner k1 is higher than corner k0 (farther from the camera's foot)."""
        P0, P1 = self.point(corners[k0]), self.point(corners[k1])
        return bool(np.linalg.norm(P1 - self.foot) > np.linalg.norm(P0 - self.foot))


def derive_layout(zero: dict[int, dict[int, np.ndarray]], sizes: dict[int, tuple[int, int]],
                  link_of: dict[int, tuple[int, str]], old_layout: dict, floor_ids: set[int], top: int,
                  *, device: Optional[dict[int, str]] = None, log: Optional[Callable[[str], None]] = None) -> dict[str, Any]:
    """Tag orientations, leg azimuths and the chassis rotation from one zero-pose observation.

    Pure: no I/O. ``zero`` maps camera -> tag id -> 4x2 corners in that camera's
    image; ``link_of`` maps tag id -> (leg, link) from the motion pass. Returns
    a dict with ``tags`` (layout entries without carried faces), ``notes`` per
    tag, ``azimuth_deg`` per leg, ``focal_fit`` and the ids seen.
    """
    log = log or (lambda msg: None)
    old_by_id = {int(t["id"]): t for t in old_layout.get("robot_tags", [])}
    horiz_old = {tid for tid, t in old_by_id.items() if t.get("surface") == "horizontal"} | set(floor_ids)
    device = device or {}
    K: dict[int, np.ndarray] = {}
    focal_fit: dict[int, dict[str, float]] = {}
    for c, tags in zero.items():
        w, h = sizes[c]
        K[c] = approx_intrinsics(w, h, device.get(c, ""))
        flat = [crn for tid, crn in tags.items() if tid in horiz_old]
        f, spread = fit_focal(flat, w, h)
        if len(flat) >= 3:
            K[c] = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])
            focal_fit[c] = {"f_px": round(f, 1), "normal_spread_deg": round(spread, 2), "tags": len(flat)}
            log(f"cam {c}: fitted focal {f:.0f} px from {len(flat)} flat tags (normal spread {spread:.2f} deg)")
    seen_ids = set().union(*(set(t) for t in zero.values())) if zero else set()
    result: dict[str, Any] = {"tags": [], "notes": {}, "azimuth_deg": {}, "azimuth_residual_deg": {},
                              "axis_quality": {}, "body_x_estimates": {}, "focal_fit": focal_fit, "seen_ids": sorted(seen_ids),
                              "top_camera": top, "chassis_tag_seen": 0 in zero.get(top, {})}
    if top not in zero:
        result["error"] = f"top camera {top} gave no frame"
        return result
    tops = zero[top]
    up = consensus_up([crn for tid, crn in tops.items() if tid in horiz_old], K[top])
    if up is None:
        result["error"] = "top camera has fewer than two known flat tags; no plane to work in"
        return result
    plane = LidPlane(K[top], up)
    notes: dict[int, dict[str, Any]] = result["notes"]

    def observed_in(tid: int) -> list[int]:
        return sorted(c for c, tags in zero.items() if tid in tags)

    # ---- which tag is which on each leg
    per_leg: dict[int, dict[str, list[int]]] = {}
    for tid, (leg, link) in link_of.items():
        per_leg.setdefault(leg, {"coxa": [], "femur": [], "tibia": []})[link].append(tid)
    kind_of: dict[int, str] = {}
    lid_of: dict[tuple[int, str], int] = {}
    axis_of: dict[int, tuple[np.ndarray, np.ndarray]] = {}     # leg -> (point on axis, distal unit)
    body_pt = plane.point(center(tops[0])) if 0 in tops else None
    for leg, links in sorted(per_leg.items()):
        # Motion already says: coxa tags are the hip lid, tibia tags are knee yokes,
        # femur tags are the knee lid plus the two hip yokes.
        hip_lid = links["coxa"][0] if links["coxa"] else None
        if len(links["coxa"]) > 1:
            seen_c = [t for t in links["coxa"] if t in tops]
            hip_lid = seen_c[0] if seen_c else hip_lid
            notes.setdefault(hip_lid, {})["warning"] = f"several coxa tags {links['coxa']}; took the first seen"
        for t in links["coxa"]:
            kind_of[t] = "servo_lid" if t == hip_lid else "unplaced"
        for t in links["tibia"]:
            kind_of[t] = "yoke_face"
        femur_seen = [t for t in links["femur"] if t in tops]
        knee_lid = None
        old_femur_lids = [t for t in links["femur"] if old_by_id.get(t, {}).get("kind") == "servo_lid"]
        if femur_seen:
            # The knee lid is the femur tag that back-projects as a square. A tag that
            # does not (log chord ratio above 0.25) is a yoke face; if none is square the
            # knee lid was not in view and stays unknown rather than being faked by a face.
            sq = {t: round(plane.squareness(tops[t]), 2) for t in femur_seen}
            best = min(femur_seen, key=sq.get)
            if sq[best] <= 0.25:
                knee_lid = best
                notes.setdefault(knee_lid, {})["chord_log_ratio_by_tag"] = sq
            else:
                knee_lid = old_femur_lids[0] if old_femur_lids else None
                log(f"leg {leg}: no femur tag projects as a square {sq}; knee lid "
                    + (f"taken from the previous layout ({knee_lid})" if knee_lid is not None else "unknown"))
        elif len(links["femur"]) == 1 and not old_by_id.get(links["femur"][0], {}).get("kind") == "yoke_face":
            knee_lid = links["femur"][0]
        elif links["femur"]:
            knee_lid = old_femur_lids[0] if old_femur_lids else None
        for t in links["femur"]:
            kind_of[t] = "servo_lid" if t == knee_lid else "yoke_face"
        if hip_lid is not None:
            lid_of[(leg, "hip")] = hip_lid
        if knee_lid is not None:
            lid_of[(leg, "knee")] = knee_lid
        # Leg direction on the plane: hip lid -> knee lid; else chassis tag -> a lid.
        pts = [plane.point(center(tops[t])) for t in (hip_lid, knee_lid) if t is not None and t in tops]
        if len(pts) == 2:
            axis_of[leg] = (pts[0], plane.in_plane(pts[1] - pts[0]))
            result["axis_quality"][leg] = "two_lids"
        elif pts and body_pt is not None:
            axis_of[leg] = (pts[0], plane.in_plane(pts[0] - body_pt))
            result["axis_quality"][leg] = "chassis_fallback"
            notes.setdefault(hip_lid if hip_lid in tops else knee_lid, {})["axis_from_chassis_tag"] = (
                "only one lid seen; leg direction taken from the chassis tag centre, which is off the body centre")

    # ---- body +x from every leg, then each leg's measured azimuth
    if axis_of:
        # Body +x from the legs whose direction is hip lid -> knee lid; a leg whose
        # direction had to come from the off-centre chassis tag only joins when
        # fewer than two proper ones exist.
        good = [leg for leg, q in result["axis_quality"].items() if q == "two_lids"]
        use = good if len(good) >= 2 else list(axis_of)
        ests = {leg: plane.rotate(d, -nominal_azimuth_deg(leg)) for leg, (_, d) in axis_of.items()}
        ref = ests[use[0]]
        angles = {leg: plane.angle(ref, v) for leg, v in ests.items()}
        mean = circular_mean_deg([angles[leg] for leg in use])
        body_x = plane.rotate(ref, mean)
        for leg, (_, d) in sorted(axis_of.items()):
            az = plane.angle(body_x, d)
            result["azimuth_deg"][leg] = round(az, 1)
            result["azimuth_residual_deg"][leg] = round(wrap_deg(az - nominal_azimuth_deg(leg)), 1)
            result["body_x_estimates"][leg] = round(wrap_deg(angles[leg] - mean), 1)
        spread = max(abs(result["body_x_estimates"][leg]) for leg in use)
        log(f"body +x from {len(axis_of)} legs; per-leg disagreement up to {spread:.1f} deg; "
            f"azimuths {result['azimuth_deg']} (nominal residuals {result['azimuth_residual_deg']})")
        if spread > 15.0:
            result["warning"] = (f"legs disagree about body +x by {spread:.0f} deg: some yaw servo is far from "
                                 f"its zero or a lid is not on its leg's centreline")
    else:
        body_x = None

    # ---- lids
    new_tags: list[dict[str, Any]] = []
    for tid, (leg, link) in sorted(link_of.items()):
        frame = f"L{leg}_{link}"
        entry: dict[str, Any] = {"id": tid, "leg": leg, "frame": frame}
        n = notes.setdefault(tid, {})
        kind = kind_of.get(tid, "yoke_face")
        if kind == "unplaced":
            n["unresolved"] = "second tag on the coxa; the layout has one hip lid per leg"
            new_tags.append(entry)
            continue
        if kind == "servo_lid":
            entry.update(kind="servo_lid", joint="hip" if link == "coxa" else "knee", surface="horizontal")
            if tid in tops and leg in axis_of:
                # frame_from_tag euler z: rotation about +z taking link +x to tag +x.
                raw = plane.angle(axis_of[leg][1], plane.tag_x(tops[tid]))
                snapped, resid = snap90(raw)
                entry["frame_from_tag"] = {"translation_m": None, "euler_xyz_deg": [0.0, 0.0, snapped]}
                n["measured_z_deg"] = round(raw, 1)
                n["snap_residual_deg"] = round(resid, 1)
                old = old_by_id.get(tid, {}).get("frame_from_tag", {}).get("euler_xyz_deg")
                if old is not None:
                    n["old_z_deg"] = old[2]
                log(f"lid {tid} L{leg} {entry['joint']}: euler z {raw:+.1f} -> {snapped:+.0f}"
                    + (f" (old layout {old[2]:+.0f})" if old is not None else ""))
            elif tid in old_by_id:
                entry["frame_from_tag"] = old_by_id[tid]["frame_from_tag"]
                n["orientation_from_old_layout"] = True
            entry["observed_in"] = observed_in(tid)
            new_tags.append(entry)
            continue
        # ---- yoke faces
        entry.update(kind="yoke_face", joint="hip" if link == "femur" else "knee")
        if link == "coxa":
            n["warning"] = "vertical tag on the coxa: the layout has no place for it"
        geo = None
        if tid in tops and leg in axis_of:
            a, d = axis_of[leg]
            y_link = np.cross(plane.up, d)                 # z cross x = y (right-handed link frame)
            c = plane.point(center(tops[tid]))
            off = c - a
            off -= d * float(np.dot(off, d))
            edge_plane = float(np.linalg.norm(plane.point(tops[tid][1]) - plane.point(tops[tid][0])))
            side_val = float(np.dot(off, y_link)) / (edge_plane + 1e-9)
            if abs(side_val) >= 0.15:
                side = "+y" if side_val > 0 else "-y"
                # Tag +x is either along the leg (+-x) or vertical (+-z); on a square
                # mounting exactly one of the tag's two chords is horizontal. The chord
                # more parallel to the leg is the horizontal one. A vertical chord's
                # higher end back-projects farther from the camera's foot on the plane.
                P = [plane.point(tops[tid][k]) for k in range(4)]
                ch_x, ch_y = P[1] - P[0], P[3] - P[0]
                ax = abs(float(np.dot(plane.in_plane(ch_x), d))); ay = abs(float(np.dot(plane.in_plane(ch_y), d)))
                if ax >= ay:
                    tx = "+x" if float(np.dot(ch_x, d)) > 0 else "-x"
                else:
                    tx = "+z" if plane.height_order(tops[tid], 0, 1) else "-z"
                tokens = {"x": tx, "z": side}
                yv = np.cross(AXIS_VECTORS[side], AXIS_VECTORS[tx]); j = int(np.argmax(np.abs(yv)))
                tokens["y"] = ("+" if yv[j] > 0 else "-") + "xyz"[j]
                geo = {"side": side, "side_margin": round(abs(side_val), 2), "tokens": tokens,
                       "quaternion_xyzw": quat_xyzw(tokens_to_rot(tokens)),
                       "chords": {"x_along_leg": round(ax, 2), "y_along_leg": round(ay, 2)}, "camera": top}
            else:
                n["side_undecided"] = round(side_val, 2)
        if geo:
            entry["mount_side"] = geo["side"]
            entry["frame_from_tag"] = {"translation_m": None, "quaternion_xyzw": geo["quaternion_xyzw"],
                                       "tag_axes_in_frame": geo["tokens"]}
            n["yoke_geometry"] = geo
        elif tid in old_by_id and old_by_id[tid].get("kind") == "yoke_face":
            entry["mount_side"] = old_by_id[tid]["mount_side"]
            entry["frame_from_tag"] = old_by_id[tid]["frame_from_tag"]
            n["orientation_from_old_layout"] = True
        else:
            n["unresolved"] = "top camera did not see this face together with its leg's lids"
        entry["observed_in"] = observed_in(tid)
        new_tags.append(entry)

    # ---- chassis tag
    if 0 in seen_ids:
        entry = {"id": 0, "kind": "chassis_tag", "frame": "body", "surface": "horizontal"}
        if 0 in tops and body_x is not None:
            raw = plane.angle(body_x, plane.tag_x(tops[0]))
            entry["frame_from_tag"] = {"translation_m": None, "euler_xyz_deg": [0.0, 0.0, round(raw, 1)]}
            entry["measured_from_legs"] = sorted(leg for leg in axis_of if result["axis_quality"].get(leg) == "two_lids")
            notes[0] = {"measured_z_deg": round(raw, 1), "legs_used": sorted(axis_of),
                        "old_z_deg": old_by_id[0]["frame_from_tag"]["euler_xyz_deg"][2] if 0 in old_by_id else None}
            log(f"chassis tag 0: euler z {raw:+.1f} from {len(axis_of)} legs"
                + (f" (old layout {notes[0]['old_z_deg']:+.1f})" if notes[0]["old_z_deg"] is not None else ""))
        elif 0 in old_by_id:
            entry["frame_from_tag"] = old_by_id[0]["frame_from_tag"]
            notes[0] = {"orientation_from_old_layout": True}
        entry["observed_in"] = observed_in(0)
        new_tags.insert(0, entry)
    result["tags"] = new_tags
    result["never_moved_unassigned"] = sorted(seen_ids - set(link_of) - set(floor_ids) - {0})
    return result


# ---------------------------------------------------------------- assembly

def assemble_layout(old_layout: dict, old_map: dict, floor: dict, derived: dict[str, Any], *,
                    yaw_sense: Optional[str], yaw_sense_source: str, moved: bool, cameras: list[int],
                    out_dir: str, today: Optional[str] = None) -> dict[str, Any]:
    """Merge the derived tags with what the old layout still has to offer.

    Faces nobody saw are carried with ``verified: false``; mounts with no tag
    at all become declared ``unresolved_mounts``. Returns the layout, the tag
    map, the diff and the validator's verdict."""
    today = today or dt.date.today().isoformat()
    old_by_id = {int(t["id"]): t for t in old_layout.get("robot_tags", [])}
    new_tags = [dict(t) for t in derived["tags"]]
    # Body +x is a mean over the legs in view, so the chassis rotation from a pass
    # that saw more legs with both lids is the better one; keep it.
    for e in new_tags:
        if e.get("kind") == "chassis_tag" and e["id"] in old_by_id:
            prior = old_by_id[e["id"]]
            if len(prior.get("measured_from_legs") or []) > len(e.get("measured_from_legs") or []):
                e["frame_from_tag"] = prior["frame_from_tag"]
                e["measured_from_legs"] = prior["measured_from_legs"]
                e["note"] = (f"rotation kept from the pass that saw {len(prior['measured_from_legs'])} legs with both "
                             f"lids; this pass saw {len(derived.get('axis_quality', {}))}")
    unresolved_tags = [e["id"] for e in new_tags
                       if "frame_from_tag" not in e or (e.get("kind") == "yoke_face" and "mount_side" not in e)]
    new_tags = [e for e in new_tags if e["id"] not in unresolved_tags]
    for e in new_tags:
        e["verified"] = True
    have_faces = {(t["leg"], t["joint"], t.get("mount_side")) for t in new_tags if t["kind"] == "yoke_face"}
    have_lids = {(t["leg"], t["joint"]) for t in new_tags if t["kind"] == "servo_lid"}
    carried: list[int] = []
    for tid, t in old_by_id.items():
        if any(e["id"] == tid for e in new_tags):
            continue
        if t.get("kind") == "chassis_tag":
            # Body +x is defined from the legs and their nominal azimuths, the same way
            # every run, so an unseen chassis tag keeps its previous rotation.
            if not any(e["kind"] == "chassis_tag" for e in new_tags):
                e = dict(t); e["verified"] = False; new_tags.insert(0, e); carried.append(tid)
            continue
        if t["kind"] == "yoke_face" and (int(t["leg"]), t["joint"], t.get("mount_side")) not in have_faces:
            e = dict(t); e["verified"] = False; new_tags.append(e)
            have_faces.add((int(t["leg"]), t["joint"], t.get("mount_side"))); carried.append(tid)
        elif t["kind"] == "servo_lid" and (int(t["leg"]), t["joint"]) not in have_lids:
            e = dict(t); e["verified"] = False; new_tags.append(e)
            have_lids.add((int(t["leg"]), t["joint"])); carried.append(tid)
    gaps: list[dict[str, Any]] = []
    for leg in LEGS:
        for joint in JOINTS:
            if (leg, joint) not in have_lids:
                gaps.append({"leg": leg, "joint": joint, "kind": "servo_lid",
                             "reason": f"no camera saw a lid tag on this servo during the {today} pass", "since": today})
            for side in ("+y", "-y"):
                if (leg, joint, side) not in have_faces:
                    gaps.append({"leg": leg, "joint": joint, "kind": "yoke_face", "mount_side": side,
                                 "reason": f"no camera saw a tag on this face during the {today} pass "
                                           f"and the previous layout had none", "since": today})

    diff: dict[str, list] = {"unchanged": [], "changed": [], "new": [], "gone_or_unseen": [],
                             "carried_unverified": carried, "unresolved_tags": unresolved_tags,
                             "never_moved_unassigned": derived.get("never_moved_unassigned", [])}
    for e in new_tags:
        tid = e["id"]
        if tid not in old_by_id:
            diff["new"].append(tid); continue
        o = old_by_id[tid]
        same = (o["frame"] == e["frame"] and o.get("mount_side") == e.get("mount_side")
                and same_rotation(o.get("frame_from_tag"), e.get("frame_from_tag")))
        (diff["unchanged"] if same else diff["changed"]).append(tid)
    for tid in old_by_id:
        if not any(e["id"] == tid for e in new_tags):
            diff["gone_or_unseen"].append(tid)

    layout = dict(old_layout)
    layout["captured"] = today
    layout["robot_tags"] = new_tags
    layout["unresolved_mounts"] = gaps
    layout["body_frame"] = {
        "axes": "+z up, +x forward between legs 0 and 5, +y = z cross x",
        "leg_numbering_from_above": "clockwise",
        "nominal_leg_azimuth_deg": {str(leg): nominal_azimuth_deg(leg) for leg in LEGS},
        "note": ("The gait code's own frame puts leg i at +(i+0.5)*60 deg and counts a clockwise-from-above yaw "
                 "as positive: it is right-handed with z DOWN (x forward, y right), i.e. this frame rotated 180 deg "
                 "about x, not a reflection. Azimuths below were measured at the commanded zero pose, so each yaw "
                 "servo's zero offset is absorbed into its leg's azimuth."),
    }
    # A leg this pass could not see, or saw with only one lid (direction from the
    # off-centre chassis tag), keeps the azimuth an earlier pass measured properly.
    prior = {int(k): float(v) for k, v in (old_layout.get("leg_zero_azimuth_body_deg") or {}).items()}
    prior_measured = {int(v) for v in (old_layout.get("leg_zero_azimuth_measured") or [])}
    quality = derived.get("axis_quality", {})
    azimuths = {}
    for leg in LEGS:
        if leg in derived["azimuth_deg"] and (quality.get(leg) == "two_lids" or leg not in prior_measured):
            azimuths[str(leg)] = derived["azimuth_deg"][leg]
        else:
            azimuths[str(leg)] = prior.get(leg, nominal_azimuth_deg(leg))
    layout["leg_zero_azimuth_body_deg"] = azimuths
    layout["leg_zero_azimuth_measured"] = sorted(
        {leg for leg in derived["azimuth_deg"] if quality.get(leg) == "two_lids"} | (prior_measured & set(prior)))
    yaw_sign = None
    if yaw_sense == "clockwise":
        yaw_sign = -1
    elif yaw_sense == "counterclockwise":
        yaw_sign = 1
    layout["joint_conventions"] = {
        "yaw_positive_seen_from_above": yaw_sense,
        "yaw_sign_in_body_frame": yaw_sign,
        "source": yaw_sense_source,
        "note": "robot yaw = yaw_sign_in_body_frame * (coxa heading - body heading - leg zero azimuth)",
    }
    layout["evidence"] = {"method": "hexapod-calibrate-tags: motion partition + lid-plane geometry + Claude census",
                          "cameras": cameras, "top_camera": derived.get("top_camera"), "moved": moved,
                          "decoded_unique_ids": derived.get("seen_ids", []), "focal_fit_px": derived.get("focal_fit", {}),
                          "output_dir": out_dir}
    layout["limitations"] = [
        "Robot-tag translations are null; the tracker never reads them.",
        "Tags with verified=false were carried from the previous layout unseen; turn the robot and rerun to confirm them.",
        "Lid rotations are snapped to 90 deg; residuals are in the report.",
        "Leg azimuths include each yaw servo's zero offset at the commanded zero pose.",
    ]
    tag_map = dict(old_map)
    parts = []
    for leg in LEGS:
        for joint in JOINTS:
            faces = {t.get("mount_side"): t["id"] for t in new_tags
                     if t["kind"] == "yoke_face" and t["leg"] == leg and t["joint"] == joint}
            if set(faces) == {"+y", "-y"}:
                parts.append({"id": f"leg{leg}_{joint}_servo", "display_name": f"Leg {leg} {joint} servo assembly",
                              "tag_ids": [faces["+y"], faces["-y"]],
                              "pose_reference": "centroid of the two opposing yoke-face tag centers",
                              "surface": "vertical", "yaw_period_degrees": 180.0})
    tag_map["parts"] = parts
    problems = validate_layout(layout, floor, tag_map)
    return {"layout": layout, "tag_map": tag_map, "diff": diff, "problems": problems,
            "declared_gaps": declared_gap_notes(layout)}


# ------------------------------------------------------------------ claude

def annotate(img: np.ndarray, tags: dict[int, np.ndarray], color=(0, 220, 255), labels: Optional[dict] = None) -> np.ndarray:
    out = img.copy()
    for tid, c in tags.items():
        cv2.polylines(out, [c.astype(np.int32).reshape(-1, 1, 2)], True, color, 2)
        cx, cy = center(c).astype(int)
        text = str(tid) if not labels or tid not in labels else f"{tid} {labels[tid]}"
        cv2.putText(out, text, (int(cx) + 6, int(cy) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
        cv2.putText(out, text, (int(cx) + 6, int(cy) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return out


def crop_around(img: np.ndarray, pts: list[np.ndarray], pad: int = 160, max_w: int = 1400) -> np.ndarray:
    if not pts:
        return img
    p = np.stack(pts)
    x0, y0 = np.maximum(p.min(0) - pad, 0).astype(int)
    x1, y1 = np.minimum(p.max(0) + pad, [img.shape[1], img.shape[0]]).astype(int)
    crop = img[y0:y1, x0:x1]
    if crop.shape[1] > max_w:
        s = max_w / crop.shape[1]
        crop = cv2.resize(crop, None, fx=s, fy=s)
    return crop


def ask_claude(images: list[np.ndarray], prompt: str, *, model: str = "claude-sonnet-5",
               api_key: Optional[str] = None) -> dict:
    api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return {"error": "no ANTHROPIC_API_KEY"}
    content = []
    for k, im in enumerate(images):
        ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 82])
        content.append({"type": "text", "text": f"image {k + 1}"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                    "data": base64.b64encode(buf.tobytes()).decode()}})
    content.append({"type": "text", "text": prompt})
    # The model's hidden reasoning counts against max_tokens; 900 returned empty text.
    body = {"model": model, "max_tokens": 3000, "messages": [{"role": "user", "content": content}]}
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=json.dumps(body).encode(),
                                 headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                                          "content-type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            doc = json.loads(r.read().decode())
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}
    text = " ".join(p.get("text", "") for p in doc.get("content", []) if p.get("type") == "text").strip()
    usage = doc.get("usage", {})
    cost = usage.get("input_tokens", 0) * 3e-6 + usage.get("output_tokens", 0) * 15e-6
    start, end = text.find("{"), text.rfind("}")
    parsed = None
    if start >= 0 and end > start:
        try:
            parsed = json.loads(text[start:end + 1])
        except ValueError:
            parsed = None
    return {"text": text, "json": parsed, "cost_usd": round(cost, 4), "stop_reason": doc.get("stop_reason")}


LEG_PROMPT = (
    "This is a cheap 3D-printed hexapod lying flat on the floor, seen from above. Image 1 is the resting "
    "pose: every AprilTag has its decoded id drawn next to it; the tags that MOVE with leg {leg} are drawn in "
    "green with a star, everything else in yellow. Image 2 is the same view with leg {leg} lifted at "
    "the hip. Servos are red boxes; each has a lid on top and two side faces that can carry a tag. Going "
    "outward from the body, leg {leg} has a hip servo, a femur, a knee servo, and a black tibia rod with a red "
    "foot. Looking along the leg from the body toward the foot, 'left' and 'right' are the side faces.\n\n"
    "For every drawn id in this list: {ids}, say where it sits: on the lifted leg's hip servo lid, hip "
    "servo left face, hip servo right face, knee servo lid, knee left face, knee right face, or 'not on this "
    "leg' (another leg, the body, or the floor). Also list any face of the lifted leg's two servos (lid, "
    "left, right) that has NO tag on it. Answer as JSON only: "
    '{{"tags": {{"<id>": "<placement>"}}, "blank_faces": ["hip left", ...], "notes": "<one sentence>"}}'
)


def claude_census(leg: int, ids: list[int], zero_top: dict, lifted_top: dict, out: Path) -> dict:
    """Annotated before/after crops of one leg to Claude; returns its answer and cost."""
    pts = [center(lifted_top["tags"][t]) for t in ids if t in lifted_top["tags"]] \
        or [center(zero_top["tags"][t]) for t in ids if t in zero_top["tags"]]
    if not pts or zero_top.get("image") is None or lifted_top.get("image") is None:
        return {"error": "no image"}
    labels = {t: "*" for t in ids}
    z_img = annotate(annotate(zero_top["image"], {k: v for k, v in zero_top["tags"].items() if k not in ids}),
                     {k: v for k, v in zero_top["tags"].items() if k in ids}, color=(0, 255, 0), labels=labels)
    imgs = [crop_around(z_img, pts), crop_around(annotate(lifted_top["image"], lifted_top["tags"]), pts)]
    for k, im in enumerate(imgs):
        cv2.imwrite(str(out / f"leg{leg}_lifted_{k}.jpg"), im)
    return ask_claude(imgs, LEG_PROMPT.format(leg=leg, ids=ids))


# ------------------------------------------------------------------ stages

def load_configs(config_dir: Path) -> tuple[dict, dict, dict]:
    layout = json.loads((config_dir / "hexapod-1-apriltag-layout.json").read_text())
    tag_map = json.loads((config_dir / "hexapod_tag_map.json").read_text())
    floor = json.loads((config_dir / "floor_tag_map.json").read_text())
    return layout, tag_map, floor


def check_stability(robot: Robot, cams: Cameras, anchors: set[int], log: Callable[[str], None],
                    *, test_leg: int = 0) -> dict[str, Any]:
    """Is the robot resting flat and are the cameras still?

    Two zero observations a few seconds apart catch a camera that is still
    refocusing or being moved; a small hip lift of one leg and back catches a
    robot that rocks on its belly. Anchors are the floor tags and the chassis
    tag: if any of them moves, stop."""
    first = cams.observe()
    time.sleep(3.0)
    second = cams.observe()
    drift = sorted(anchors & set().union(*(moved(first[c]["tags"], second[c]["tags"])[0]
                                          for c in first.keys() & second.keys())))
    q = [0.0] * 18
    q[3 * test_leg + 1] = STABILITY_LIFT
    robot.move(q)
    lifted = cams.observe()
    robot.move([0.0] * 18)
    back = cams.observe()
    rocked = sorted(anchors & set().union(*(moved(second[c]["tags"], lifted[c]["tags"])[0]
                                           for c in second.keys() & lifted.keys())))
    settled = sorted(anchors & set().union(*(moved(second[c]["tags"], back[c]["tags"])[0]
                                            for c in second.keys() & back.keys())))
    anchors_seen = {c: sorted(anchors & set(o["tags"])) for c, o in second.items()}
    ok = not drift and not rocked and not settled
    verdict = ("stable" if ok else
               f"camera or robot drifting: anchors {drift} moved between two looks" if drift else
               f"robot rocks when leg {test_leg} lifts: anchors {rocked} moved" if rocked else
               f"robot did not settle back: anchors {settled} moved")
    log(f"stability: {verdict}; anchors seen per camera {anchors_seen}")
    return {"ok": ok, "verdict": verdict, "drift": drift, "rocked": rocked, "settled": settled,
            "anchors_seen": anchors_seen}


def motion_pass(robot: Robot, cams: Cameras, out: Path, log: Callable[[str], None], *, top: int,
                anchors: set[int], legs: list[int], claude: bool) -> dict[str, Any]:
    """Lift, swing and bend every leg in turn; return per-leg link partitions."""
    leg_reports: dict[int, dict[str, Any]] = {}
    claude_total = 0.0
    yaw_senses: dict[int, Optional[str]] = {}
    for leg in legs:
        j0, j1, j2 = 3 * leg, 3 * leg + 1, 3 * leg + 2
        # Fresh baseline for this leg: command zero (torque held) and look again, so a
        # joint that sagged while limp between legs is not read as motion.
        robot.move([0.0] * 18)
        zero_leg = cams.observe()
        q = [0.0] * 18
        q[j1] = HIP_LIFT
        shorts = []
        r = robot.move(q); shorts.append(("hip", r.get("short"))); lifted = cams.observe()
        yawed = []
        for yaw in (YAW_SWING, -YAW_SWING):
            q[j0] = yaw
            r = robot.move(q); shorts.append((f"yaw{yaw:+.0f}", r.get("short"))); yawed.append(cams.observe())
        q[j0] = 0.0
        q[j2] = KNEE_LIFT
        r = robot.move(q); shorts.append(("knee", r.get("short"))); knee = cams.observe()
        robot.move([0.0] * 18)
        part = partition_leg(zero_leg, lifted, yawed, knee)
        part["joints_short_of_target"] = [(s_, sh) for s_, sh in shorts if sh]
        if part["joints_short_of_target"]:
            log(f"leg {leg}: joints short of target {part['joints_short_of_target']}")
        body_moved = sorted(anchors & (set(part["moved_hip"]) | set(part["moved_yaw"]) | set(part["moved_knee"])))

        def travelled(step: str, j: int, target: float) -> bool:
            for s_, sh in shorts:
                if s_ == step and sh:
                    for jj, dlt in sh:
                        if jj == j and abs(target - dlt) < MIN_TRAVEL_DEG:
                            return False
            return True
        weak = [st for st, j, tgt in (("hip", j1, HIP_LIFT), (f"yaw{YAW_SWING:+.0f}", j0, YAW_SWING),
                                      (f"yaw{-YAW_SWING:+.0f}", j0, -YAW_SWING), ("knee", j2, KNEE_LIFT))
                if not travelled(st, j, tgt)]
        if body_moved or weak:
            part["discarded"] = {"body_or_floor_moved": body_moved, "joint_barely_moved": weak}
            log(f"leg {leg}: DISCARDED (body/floor moved {body_moved}; joint barely moved on {weak})")
            part["coxa"], part["femur"], part["tibia"] = [], [], []
        else:
            yaw_senses[leg] = yaw_sense_from_swing(lifted, yawed[0], part["coxa"], top)
            part["yaw_positive_seen_from_above"] = yaw_senses[leg]
        leg_reports[leg] = part
        log(f"leg {leg}: coxa {part['coxa']} femur {part['femur']} tibia {part['tibia']}"
            + (f" conflicts {part['conflicts']}" if part['conflicts'] else "")
            + (f" contradictions {part['contradictions']}" if part['contradictions'] else "")
            + (f"; +yaw turns {yaw_senses[leg]} from above" if yaw_senses.get(leg) else ""))
        if claude and top in zero_leg and top in lifted:
            ids = sorted(set(part["coxa"]) | set(part["femur"]) | set(part["tibia"]))
            ans = claude_census(leg, ids, zero_leg[top], lifted[top], out) if ids else {"error": "nothing moved"}
            claude_total += ans.get("cost_usd", 0.0)
            part["claude"] = ans
            log(f"leg {leg} claude: {str(ans.get('json') or ans.get('text') or ans.get('error'))[:300]}")
    robot.relax()
    senses = [s for s in yaw_senses.values() if s]
    sense = max(set(senses), key=senses.count) if senses else None
    return {"legs": leg_reports, "claude_cost_usd": round(claude_total, 3),
            "yaw_sense": sense, "yaw_sense_votes": {str(k): v for k, v in yaw_senses.items()}}


def votes_from_reports(reports: list[dict]) -> dict[int, list[tuple[int, str]]]:
    votes: dict[int, list[tuple[int, str]]] = {}
    for rep in reports:
        for leg, part in (rep.get("legs") or {}).items():
            for link in LINKS:
                for tid in part.get(link, []):
                    votes.setdefault(int(tid), []).append((int(leg), link))
    return votes


def votes_from_layout(layout: dict) -> dict[int, list[tuple[int, str]]]:
    votes: dict[int, list[tuple[int, str]]] = {}
    for t in layout.get("robot_tags", []):
        if t.get("kind") != "chassis_tag" and "leg" in t:
            votes[int(t["id"])] = [(int(t["leg"]), str(t["frame"]).split("_")[1])]
    return votes


def resolve_links(votes: dict[int, list[tuple[int, str]]], log: Callable[[str], None],
                  old_layout: Optional[dict] = None) -> dict[int, tuple[int, str]]:
    """Majority link per tag across passes; a tie goes to what the previous layout said."""
    prior = votes_from_layout(old_layout) if old_layout else {}
    link_of: dict[int, tuple[int, str]] = {}
    for tid, vs in votes.items():
        counts = {v: vs.count(v) for v in set(vs)}
        top = max(counts.values())
        tied = sorted(v for v, n in counts.items() if n == top)
        best = tied[0]
        if len(tied) > 1 and prior.get(tid) and prior[tid][0] in tied:
            best = prior[tid][0]
        if len(counts) > 1:
            log(f"tag {tid}: seen on more than one link {sorted(counts.items())}; taking {best}"
                + (" (previous layout breaks the tie)" if len(tied) > 1 else ""))
        link_of[tid] = best
    return link_of


def write_report(out: Path, report: dict[str, Any], assembled: dict[str, Any]) -> None:
    (out / "layout.json").write_text(json.dumps(assembled["layout"], indent=1))
    (out / "hexapod_tag_map.json").write_text(json.dumps(assembled["tag_map"], indent=1))
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    diff = assembled["diff"]
    layout = assembled["layout"]
    md = [f"# Tag calibration {report['generated']}", "",
          f"moved robot: {report['moved']}; cameras {report['cameras']}; top camera {report['top_camera']}", "",
          "## what changed against the previous layout",
          f"- unchanged: {diff['unchanged']}", f"- changed: {diff['changed']}", f"- new ids: {diff['new']}",
          f"- gone or unseen: {diff['gone_or_unseen']}", f"- carried unverified: {diff['carried_unverified']}",
          f"- seen but not placed: {diff['unresolved_tags']}",
          f"- never moved, unassigned (floor? body?): {diff['never_moved_unassigned']}", "",
          "## conventions measured",
          f"- positive yaw seen from above: {layout['joint_conventions']['yaw_positive_seen_from_above']} "
          f"({layout['joint_conventions']['source']})",
          f"- leg azimuths in the body frame (z up, x forward): {layout['leg_zero_azimuth_body_deg']}",
          f"- residuals from nominal -(i+0.5)*60: {report.get('azimuth_residual_deg')}",
          f"- chassis tag euler z: {next((t['frame_from_tag']['euler_xyz_deg'][2] for t in layout['robot_tags'] if t['kind'] == 'chassis_tag'), None)}",
          "", "## mounts with no tag (declared gaps)", *([f"- {g}" for g in assembled["declared_gaps"]] or ["- none"]),
          "", "## validation", *([f"- {p}" for p in assembled["problems"]] or ["- clean"]),
          "", "## stability", f"- {report.get('stability', {}).get('verdict', 'not checked')}",
          "", "## per-leg motion"]
    for leg, r in (report.get("legs") or {}).items():
        md.append(f"- leg {leg}: coxa {r.get('coxa')} femur {r.get('femur')} tibia {r.get('tibia')}; "
                  f"conflicts {r.get('conflicts')}; claude: {json.dumps((r.get('claude') or {}).get('json'))[:400]}")
    md += ["", "## per-tag notes", *(f"- {k}: {json.dumps(v, default=str)[:300]}" for k, v in sorted(report["notes"].items(), key=lambda kv: int(kv[0])))]
    (out / "report.md").write_text("\n".join(md) + "\n")


def install(out: Path, config_dir: Path, log: Callable[[str], None]) -> None:
    bak = config_dir / f"backup-{dt.date.today().isoformat()}"
    bak.mkdir(exist_ok=True)
    for name in ("hexapod-1-apriltag-layout.json", "hexapod_tag_map.json"):
        if (config_dir / name).exists() and not (bak / name).exists():
            shutil.copy2(config_dir / name, bak / name)
    shutil.copy2(out / "layout.json", config_dir / "hexapod-1-apriltag-layout.json")
    shutil.copy2(out / "hexapod_tag_map.json", config_dir / "hexapod_tag_map.json")
    log(f"wrote configs; previous copies in {bak}")


# -------------------------------------------------------------------- main

def default_out() -> Path:
    return (Path.home() / "Library" / "Application Support" / "Hexapod Lab" / "v2"
            / f"tag-calibration-{dt.datetime.now():%Y%m%d-%H%M%S}")


def run(args) -> int:
    out = Path(args.out).expanduser() if args.out else default_out()
    out.mkdir(parents=True, exist_ok=True)
    log = Log()
    config_dir = Path(args.config_dir)
    old_layout, old_map, floor = load_configs(config_dir)
    floor_ids = {int(t["id"]) for t in floor["tags"]}
    anchors = floor_ids | {0}
    top = int(args.top_camera)
    cameras = [int(i) for i in args.cameras.split(",")]
    legs = [int(i) for i in args.legs.split(",")] if args.legs else list(LEGS)
    moving = not (args.no_move or args.replay)
    report: dict[str, Any] = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "moved": moving,
                              "cameras": cameras, "top_camera": top, "out": str(out)}
    votes: dict[int, list[tuple[int, str]]] = {}
    yaw_sense, yaw_source = None, "unknown"

    if args.replay:
        rep_dir = Path(args.replay).expanduser()
        zero_tags, sizes = load_tags(rep_dir / "zero_tags.json")
        images = {c: cv2.imread(str(rep_dir / f"zero_cam{c}_raw.jpg")) for c in zero_tags}
        zero = {c: {"tags": t, "image": images.get(c), "size": sizes.get(c)} for c, t in zero_tags.items()}
        cameras = sorted(zero); report["cameras"] = cameras
        log(f"replay: zero pose from {rep_dir} (cameras {cameras})")
        if not args.assign_from and (rep_dir / "report.json").exists():
            args.assign_from = [str(rep_dir / "report.json")]
    else:
        cams = Cameras(args.camera_url, cameras)
        robot = Robot(args.robot_url, dry=not moving)
        if moving:
            q0 = robot.pose()
            if max(abs(v) for v in q0) > 3.0:
                log(f"robot not at zero (max |q| = {max(abs(v) for v in q0):.1f} deg); moving to zero first")
            robot.move([0.0] * 18)
            if not args.skip_stability:
                report["stability"] = check_stability(robot, cams, anchors, log)
                if not report["stability"]["ok"]:
                    (out / "report.json").write_text(json.dumps({**report, "log": log.lines}, indent=1))
                    log("stopping: nothing measured on a moving stage would mean anything")
                    return 3
        zero = cams.observe()
        for c, o in zero.items():
            cv2.imwrite(str(out / f"zero_cam{c}.jpg"), annotate(o["image"], o["tags"]))
            cv2.imwrite(str(out / f"zero_cam{c}_raw.jpg"), o["image"])
            log(f"zero: cam {c} sees {sorted(o['tags'])}")
        if moving:
            motion = motion_pass(robot, cams, out, log, top=top, anchors=anchors, legs=legs, claude=not args.no_claude)
            report.update(legs=motion["legs"], claude_cost_usd=motion["claude_cost_usd"],
                          yaw_sense_votes=motion["yaw_sense_votes"])
            votes = votes_from_reports([motion])
            if motion["yaw_sense"]:
                yaw_sense, yaw_source = motion["yaw_sense"], f"measured by the {report['generated'][:10]} motion pass"
    save_tags(out / "zero_tags.json", zero)
    if top not in zero:
        log(f"top camera {top} gave no frame; cannot do geometry")
        return 3

    prior_reports = []
    for path in args.assign_from or []:
        rep = json.loads(Path(path).expanduser().read_text())
        prior_reports.append(rep)
        if not yaw_sense and rep.get("yaw_sense"):
            yaw_sense, yaw_source = rep["yaw_sense"], f"from earlier report {path}"
        log(f"link assignment merged from {path}")
    if prior_reports:
        for tid, vs in votes_from_reports(prior_reports).items():
            votes.setdefault(tid, []).extend(vs)
        report.setdefault("legs", {})
        for rep in prior_reports:
            for leg, part in (rep.get("legs") or {}).items():
                report["legs"].setdefault(leg, part)
    if not votes:
        votes = votes_from_layout(old_layout)
        log("no motion evidence: link assignment taken from the old layout; geometry re-derived from the zero pose")
    if args.yaw_sense:
        yaw_sense, yaw_source = args.yaw_sense, "given on the command line (--yaw-sense)"
    if not yaw_sense:
        old = (old_layout.get("joint_conventions") or {}).get("yaw_positive_seen_from_above")
        if old:
            yaw_sense, yaw_source = old, "carried from the previous layout"
        else:
            log("WARNING: yaw sense not measured, not given (--yaw-sense) and not in the previous layout; "
                "the layout will carry yaw_sign_in_body_frame = null and planar_pose will assume +1, which is "
                "wrong for this robot's clockwise legs")
    link_of = resolve_links(votes, log, old_layout)

    derived = derive_layout(tags_only(zero), {c: o["size"] for c, o in zero.items()}, link_of, old_layout,
                            floor_ids, top, device={0: "OV9281"}, log=log)
    if derived.get("error"):
        log(f"geometry: {derived['error']}")
        (out / "report.json").write_text(json.dumps({**report, "log": log.lines}, indent=1))
        return 3
    if derived.get("warning"):
        log(f"WARNING {derived['warning']}")
    assembled = assemble_layout(old_layout, old_map, floor, derived, yaw_sense=yaw_sense, yaw_sense_source=yaw_source,
                                moved=moving, cameras=cameras, out_dir=str(out))
    if not args.replay and any(cams.duplicate_ids.values()):
        report["duplicate_ids_seen"] = {str(c): sorted(v) for c, v in cams.duplicate_ids.items() if v}
        log(f"WARNING duplicate tag ids were in view: {report['duplicate_ids_seen']}; spare tags near the robot "
            f"make the detector's choice arbitrary; move them away and rerun")
    report.update(diff=assembled["diff"], notes={str(k): v for k, v in derived["notes"].items()},
                  azimuth_deg={str(k): v for k, v in derived["azimuth_deg"].items()},
                  azimuth_residual_deg={str(k): v for k, v in derived["azimuth_residual_deg"].items()},
                  body_x_disagreement_deg={str(k): v for k, v in derived["body_x_estimates"].items()},
                  focal_fit=derived["focal_fit"], yaw_sense=yaw_sense, yaw_sense_source=yaw_source,
                  declared_gaps=assembled["declared_gaps"], validation_problems=assembled["problems"],
                  geometry_warning=derived.get("warning"), log=log.lines)
    write_report(out, report, assembled)
    log(f"diff: {json.dumps(assembled['diff'])}")
    log(f"declared gaps: {len(assembled['declared_gaps'])}; validation: {assembled['problems'] or 'clean'}")
    log(f"outputs in {out}")

    if args.write:
        if assembled["problems"] and not args.force:
            log("NOT writing: validation problems (use --force to install anyway)")
            return 2
        if assembled["problems"]:
            log(f"writing despite validation problems (--force): {assembled['problems']}")
        install(out, config_dir, log)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=None, help="output directory (default: a timestamped dir under Hexapod Lab/v2)")
    ap.add_argument("--camera-url", default="http://127.0.0.1:8766")
    ap.add_argument("--robot-url", default="http://192.168.4.39:8080")
    ap.add_argument("--cameras", default="0,1,2")
    ap.add_argument("--top-camera", default="2")
    ap.add_argument("--config-dir", default=str(CONFIG_DIR))
    ap.add_argument("--legs", default=None, help="comma-separated legs to move (default all)")
    ap.add_argument("--no-move", action="store_true", help="look at the current pose only; no robot motion")
    ap.add_argument("--skip-stability", action="store_true", help="skip the stability check before moving")
    ap.add_argument("--replay", default=None, metavar="DIR",
                    help="re-derive geometry from an earlier run's zero_tags.json instead of the cameras")
    ap.add_argument("--assign-from", action="append", default=None, metavar="REPORT",
                    help="report.json of an earlier pass; repeat to merge several passes' link assignments")
    ap.add_argument("--yaw-sense", choices=("clockwise", "counterclockwise"), default=None,
                    help="which way a positive yaw turns a leg seen from above, if not measured in this run")
    ap.add_argument("--no-claude", action="store_true")
    ap.add_argument("--write", action="store_true", help="install into configs/ if validation passes")
    ap.add_argument("--force", action="store_true", help="with --write: install despite validation problems")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
