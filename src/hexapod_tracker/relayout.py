"""Re-derive the hexapod's AprilTag layout by moving the robot and watching.

The layout the tracker needs is small: for every tag, which link it is on
(body, L<n>_coxa / _femur / _tibia), whether it is a horizontal lid or a
vertical yoke face, which side a yoke face is on, and the rotation from tag
axes to link axes. No translations (planar_pose.py never reads them).

How this tool gets each of those:

* **Link** — by motion, not by looking. One leg at a time, lifted clear of
  the floor: a hip move carries femur + tibia tags, a yaw move adds the coxa,
  a knee move carries the tibia only. Tags that never move are body or floor.
  Because the moves are made through the servo joint indices, leg numbering
  in the layout is the servo numbering by construction.
* **Lid vs yoke** — the tag normal (IPPE PnP with approximate intrinsics)
  compared with the chassis tag's normal in the same frame: parallel is a
  lid, perpendicular is a yoke face.
* **Lid rotation** — the angle from the tag's +x (corner 0 -> 1) to the
  link's distal direction in the top camera at the zero pose, where all
  three links of a leg are collinear with the body centre.
* **Yoke side and rotation** — in a side camera at the zero pose, the link
  frame is built from the lid normal (up) and the lid-to-lid direction
  (distal); the yoke normal's sign along link +y gives the side and the tag
  +x picks one of four 90-degree mountings.
* **Claude** — cannot decode tag ids, so every image it sees has the ids
  drawn on. It answers discrete questions per leg (top / left / right / not
  this leg, and which servo faces are blank) as a cross-check and a
  missing-tag census. Disagreements are reported, never auto-resolved.

    hexapod-relayout --out DIR [--no-move] [--write] [--no-claude]

``--no-move`` analyses the current pose only (useful to prove the geometry
against tags that did not change). ``--write`` installs the new layout and
tag map into configs/ (old files backed up) after validate_layout passes.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import math
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .camera_server import detect_tag_corners, make_tag_detector
from .layout_audit import AXIS_VECTORS, validate_layout
from .paths import CONFIG_DIR

TAG_M = 0.0272
LEGS = range(6)
LINKS = ("coxa", "femur", "tibia")
# Lifted-leg test angles (robot_abs degrees). Negative hip lifts the femur;
# negative knee lifts the tibia, so the foot never touches the floor.
HIP_LIFT = -30.0
YAW_SWING = 18.0
KNEE_LIFT = -20.0
MOVE_S = 2.0
SETTLE_S = 1.2
TORQUE = 500
FRAMES_PER_STATE = 3
# Motion evidence: a tag "moved" if its centre shifted more than this many
# tag-edges, or its in-plane angle changed more than MOVE_DEG.
MOVE_EDGE_FRAC = 0.25
MOVE_PX_FLOOR = 6.0
MOVE_DEG = 6.0


# ----------------------------------------------------------------- capture

class Cameras:
    def __init__(self, base: str, indices: list[int]):
        self.base = base.rstrip("/")
        self.indices = indices
        self.detector = make_tag_detector()
        self.size: dict[int, tuple[int, int]] = {}

    def snapshot(self, i: int) -> np.ndarray:
        with urllib.request.urlopen(f"{self.base}/snapshot/{i}.jpg", timeout=6) as r:
            data = np.frombuffer(r.read(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"camera {i}: bad JPEG")
        self.size[i] = (img.shape[1], img.shape[0])
        return img

    def observe(self, frames: int = FRAMES_PER_STATE) -> dict[int, dict[str, Any]]:
        """Per camera: majority-voted tag corners over a few frames and the last image."""
        out = {}
        for i in self.indices:
            seen: dict[int, list[np.ndarray]] = {}
            img = None
            for _ in range(frames):
                try:
                    img = self.snapshot(i)
                except (urllib.error.URLError, RuntimeError, OSError) as exc:
                    print(f"camera {i}: no frame ({exc}); skipping this camera for this state", flush=True)
                    time.sleep(0.5)
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                for tid, corners in detect_tag_corners(gray, self.detector).items():
                    seen.setdefault(tid, []).append(corners)
                time.sleep(0.15)
            need = max(1, (frames + 1) // 2)
            tags = {tid: np.median(np.stack(c), axis=0) for tid, c in seen.items() if len(c) >= need}
            if img is None:
                continue
            out[i] = {"tags": tags, "image": img}
        return out


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


def lid_yaw_3d(lid: np.ndarray, distal_cam: np.ndarray, K: np.ndarray, up: Optional[np.ndarray] = None) -> float:
    """Euler z of frame_from_tag for a lid from its IPPE pose: angle from tag +x to the
    link's distal direction, measured about the tag normal (+z, up)."""
    R, _ = tag_pose(lid, K, up)
    n = R[:, 2]
    d = distal_cam - n * float(np.dot(distal_cam, n))
    d /= np.linalg.norm(d) + 1e-9
    x = R[:, 0]
    return math.degrees(math.atan2(float(np.dot(np.cross(x, d), n)), float(np.dot(x, d))))


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
        (mv if shift > thr or turn > MOVE_DEG else st).add(tid)
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


def screen_angle(v_from: np.ndarray, v_to: np.ndarray) -> float:
    """Rotation about +z (toward the viewer) taking v_from to v_to, degrees, CCW on screen positive.

    Image v points down, so CCW on screen is a negative atan2 in image coordinates."""
    a = math.atan2(v_from[1], v_from[0])
    b = math.atan2(v_to[1], v_to[0])
    return -math.degrees((b - a + math.pi) % (2 * math.pi) - math.pi)


def lid_yaw(tag_corners: np.ndarray, distal_img: np.ndarray, sign: float = 1.0) -> float:
    """Euler z of frame_from_tag for a lid: angle from tag +x to link +x about +z."""
    return sign * screen_angle(x_axis_img(tag_corners), distal_img)


def yoke_geometry(yoke: np.ndarray, lids: list[np.ndarray], K: np.ndarray,
                  body_center_tvec: Optional[np.ndarray] = None,
                  up_hint: Optional[np.ndarray] = None) -> Optional[dict[str, Any]]:
    """Side and axis tokens of a yoke face from one camera at the zero pose.

    lids: corners of lid tags on this leg (proximal first) visible in the same frame."""
    if len(lids) < 1 or (len(lids) < 2 and body_center_tvec is None):
        return None
    poses = [tag_pose(c, K, up_hint) for c in lids]
    pts = [p[1] for p in poses]
    if body_center_tvec is not None:
        pts = [body_center_tvec] + pts
    d = pts[-1] - pts[0]
    d = d / (np.linalg.norm(d) + 1e-9)
    up = np.mean([p[0][:, 2] for p in poses], axis=0) if up_hint is None else up_hint.copy()
    R_t, t_t = tag_pose(yoke, K, up, want="perpendicular")
    up -= d * float(np.dot(up, d))
    up = up / (np.linalg.norm(up) + 1e-9)
    y = np.cross(up, d)  # z cross x = y
    n = R_t[:, 2]
    side_val = float(np.dot(n, y))
    side = "+y" if side_val > 0 else "-y"
    # Tag axes expressed in the link frame (x=d, y=y, z=up).
    M = np.column_stack([d, y, up])          # link axes in camera
    R_link_from_tag = M.T @ R_t              # tag axes in link frame
    tokens = rot_to_tokens(R_link_from_tag)
    # Snap: normal must be the side axis; x and y snapped from the matrix.
    tokens["z"] = side
    R_snap = tokens_to_rot(tokens)
    if abs(np.linalg.det(R_snap) - 1.0) > 1e-6 or not np.allclose(R_snap.T @ R_snap, np.eye(3)):
        # x or y snapped onto the same axis as z; rebuild y = z cross x.
        x = AXIS_VECTORS[tokens["x"]]
        z = AXIS_VECTORS[tokens["z"]]
        if abs(np.dot(x, z)) > 0.5:
            return None
        yv = np.cross(z, x)
        tokens["y"] = ("+" if yv[int(np.argmax(np.abs(yv)))] > 0 else "-") + "xyz"[int(np.argmax(np.abs(yv)))]
        R_snap = tokens_to_rot(tokens)
    residual = float(np.degrees(np.arccos(np.clip((np.trace(R_snap.T @ R_link_from_tag) - 1) / 2, -1, 1))))
    return {"side": side, "side_margin": round(abs(side_val), 3), "tokens": tokens,
            "quaternion_xyzw": quat_xyzw(R_snap), "snap_residual_deg": round(residual, 1)}


def yoke_from_top(yoke: np.ndarray, axis_point: np.ndarray, distal_img: np.ndarray,
                  principal: np.ndarray, sign: float = 1.0) -> Optional[dict[str, Any]]:
    """Side and axis tokens of a yoke face from the top camera alone.

    The face centre sits off the leg axis on its own side (+y is CCW from
    distal as seen from above, sign-calibrated like the lids). A vertical
    tag +x axis projects radially away from the principal point (up) or
    toward it (down); a horizontal one projects along +/- distal."""
    c = center(yoke)
    off = c - axis_point
    perp = off - distal_img * float(np.dot(off, distal_img))
    # CCW-on-screen from distal (image y down): rotate distal by -90 deg in image coords
    y_img = np.array([distal_img[1], -distal_img[0]]) * sign
    side_val = float(np.dot(perp, y_img)) / (edge_px(yoke) + 1e-9)
    if abs(side_val) < 0.15:
        return None
    side = "+y" if side_val > 0 else "-y"
    x_img = x_axis_img(yoke)
    along = float(np.dot(x_img, distal_img))
    radial = c - principal
    radial = radial / (np.linalg.norm(radial) + 1e-9)
    up = float(np.dot(x_img, radial))
    if abs(along) >= abs(up):
        tx = "+x" if along > 0 else "-x"
    else:
        tx = "+z" if up > 0 else "-z"
    tokens = {"x": tx, "z": side}
    yv = np.cross(AXIS_VECTORS[side], AXIS_VECTORS[tx])
    j = int(np.argmax(np.abs(yv)))
    tokens["y"] = ("+" if yv[j] > 0 else "-") + "xyz"[j]
    R = tokens_to_rot(tokens)
    return {"side": side, "side_margin": round(abs(side_val), 2), "tokens": tokens,
            "quaternion_xyzw": quat_xyzw(R), "method": "top_camera_2d",
            "x_along": round(along, 2), "x_up": round(up, 2)}


def is_horizontal(corners: np.ndarray, up: Optional[np.ndarray], K: np.ndarray) -> Optional[bool]:
    """Either IPPE branch's normal parallel to the shared up -> horizontal (lid)."""
    if up is None:
        return None
    d = max(abs(float(np.dot(R[:, 2], up))) for R, _ in tag_poses(corners, K))
    if d > 0.82:
        return True
    if d < 0.6:
        return False
    return None


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
    body = {"model": model, "max_tokens": 900, "messages": [{"role": "user", "content": content}]}
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
    return {"text": text, "json": parsed, "cost_usd": round(cost, 4)}


LEG_PROMPT = (
    "This is a cheap 3D-printed hexapod lying flat on the floor, seen from above. Image 1 is the resting "
    "pose: every AprilTag has its decoded id drawn next to it; the tags that MOVE with leg {leg} are drawn in "
    "green with a star, everything else in yellow. Image 2 is the same view with leg {leg} lifted 45 degrees at "
    "the hip. Servos are red boxes; each has a lid on top and two side faces that can carry a tag. Going "
    "outward from the body, leg {leg} has a hip servo, a femur, a knee servo, and a black tibia rod with a red "
    "foot. Looking along the leg from the body toward the foot, 'left' and 'right' are the side faces.\n\n"
    "For every drawn id in this list: {ids}, say where it sits: on the lifted leg's hip servo lid, hip "
    "servo left face, hip servo right face, knee servo lid, knee left face, knee right face, or 'not on this "
    "leg' (another leg, the body, or the floor). Also list any face of the lifted leg's two servos (lid, "
    "left, right) that has NO tag on it. Answer as JSON only: "
    '{{"tags": {{"<id>": "<placement>"}}, "blank_faces": ["hip left", ...], "notes": "<one sentence>"}}'
)


# -------------------------------------------------------------------- main

def load_old(config_dir: Path) -> tuple[dict, dict, dict]:
    layout = json.loads((config_dir / "hexapod-1-apriltag-layout.json").read_text())
    tag_map = json.loads((config_dir / "hexapod_tag_map.json").read_text())
    floor = json.loads((config_dir / "floor_tag_map.json").read_text())
    return layout, tag_map, floor


def run(args) -> int:
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    cams = Cameras(args.camera_url, [int(i) for i in args.cameras.split(",")])
    robot = Robot(args.robot_url, dry=args.no_move)
    old_layout, old_map, floor = load_old(Path(args.config_dir))
    old_by_id = {int(t["id"]): t for t in old_layout["robot_tags"]}
    floor_ids = {int(t["id"]) for t in floor["tags"]}
    top = int(args.top_camera)

    # Zero-pose observation (the reference for everything).
    q0 = robot.pose()
    if max(abs(v) for v in q0) > 3.0 and not args.no_move:
        log(f"robot not at zero (max |q| = {max(abs(v) for v in q0):.1f} deg); moving to zero first")
        robot.move([0.0] * 18)
    zero = cams.observe()
    for c, o in zero.items():
        cv2.imwrite(str(out / f"zero_cam{c}.jpg"), annotate(o["image"], o["tags"]))
        cv2.imwrite(str(out / f"zero_cam{c}_raw.jpg"), o["image"])
        log(f"zero: cam {c} sees {sorted(o['tags'])}")
    (out / "zero_tags.json").write_text(json.dumps(
        {str(c): {str(t): crn.tolist() for t, crn in o["tags"].items()} for c, o in zero.items()}))
    K = {c: approx_intrinsics(*cams.size[c], "OV9281" if c == 0 else "") for c in cams.indices}
    horiz_old = {int(t["id"]) for t in old_layout["robot_tags"] if t.get("surface") == "horizontal"}
    focal_fit: dict[int, dict[str, float]] = {}
    for c in cams.indices:
        flat = [crn for tid, crn in zero[c]["tags"].items() if tid in horiz_old]
        f, spread = fit_focal(flat, *cams.size[c])
        if len(flat) >= 3:
            K[c] = np.array([[f, 0, cams.size[c][0] / 2.0], [0, f, cams.size[c][1] / 2.0], [0, 0, 1.0]])
            focal_fit[c] = {"f_px": round(f, 1), "normal_spread_deg": round(spread, 2), "tags": len(flat)}
            log(f"cam {c}: fitted focal {f:.0f} px from {len(flat)} flat tags (normal spread {spread:.2f} deg)")
    up_of = {c: consensus_up([crn for tid, crn in zero[c]["tags"].items() if tid in horiz_old], K[c]) for c in cams.indices}

    # ---- motion: assign tags to links, one leg at a time
    assign: dict[int, dict[str, Any]] = {}
    leg_reports: dict[int, dict[str, Any]] = {}
    claude_total = 0.0
    if not args.no_move:
        for leg in LEGS:
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
            # If the body or the floor "moved", the robot rocked or slid: nothing from
            # this leg's states can be trusted.
            anchors = floor_ids | {0}
            body_moved = sorted(anchors & (set(part["moved_hip"]) | set(part["moved_yaw"]) | set(part["moved_knee"])))
            # The commanded joint must have travelled at least 8 deg for its step to mean
            # anything (a cable or a neighbour can stop a leg early; that is fine as long
            # as it moved). Other joints drifting is reported, not fatal.
            def travelled(step: str, j: int, target: float) -> bool:
                for s_, sh in shorts:
                    if s_ == step and sh:
                        for jj, dlt in sh:
                            if jj == j and abs(target - dlt) < 8.0:
                                return False
                return True
            weak = [st for st, j, tgt in (("hip", j1, HIP_LIFT), ("yaw+18", j0, YAW_SWING), ("yaw-18", j0, -YAW_SWING), ("knee", j2, KNEE_LIFT))
                    if not travelled(st, j, tgt)]
            if body_moved or weak:
                part["discarded"] = {"body_or_floor_moved": body_moved, "joint_barely_moved": weak}
                log(f"leg {leg}: DISCARDED (body/floor moved {body_moved}; joint barely moved on {weak})")
                part["coxa"], part["femur"], part["tibia"] = [], [], []
            leg_reports[leg] = part
            log(f"leg {leg}: coxa {part['coxa']} femur {part['femur']} tibia {part['tibia']}"
                + (f" conflicts {part['conflicts']}" if part['conflicts'] else "")
                + (f" contradictions {part['contradictions']}" if part['contradictions'] else ""))
            for link in LINKS:
                for tid in part[link]:
                    assign.setdefault(tid, {"votes": []})["votes"].append((leg, link))
            # Claude cross-check on the lifted state.
            if not args.no_claude:
                ids = sorted(set(part["coxa"]) | set(part["femur"]) | set(part["tibia"]))
                imgs = []
                o0, o1 = zero[top], lifted[top]
                pts = [center(o1["tags"][t]) for t in ids if t in o1["tags"]] or [center(o0["tags"][t]) for t in ids if t in o0["tags"]]
                if pts:
                    labels = {t: "*" for t in ids}
                    z_img = annotate(annotate(o0["image"], {k: v for k, v in o0["tags"].items() if k not in ids}),
                                     {k: v for k, v in o0["tags"].items() if k in ids}, color=(0, 255, 0), labels=labels)
                    imgs.append(crop_around(z_img, pts))
                    imgs.append(crop_around(annotate(o1["image"], o1["tags"]), pts))
                ans = ask_claude(imgs, LEG_PROMPT.format(leg=leg, ids=ids)) if imgs else {"error": "no image"}
                claude_total += ans.get("cost_usd", 0.0)
                leg_reports[leg]["claude"] = ans
                for k, im in enumerate(imgs):
                    cv2.imwrite(str(out / f"leg{leg}_lifted_{k}.jpg"), im)
                log(f"leg {leg} claude: {str(ans.get('json') or ans.get('text') or ans.get('error'))[:300]}")
        robot.relax()
    elif args.assign_from:
        prev = json.loads(Path(args.assign_from).expanduser().read_text())
        for leg, part in prev["legs"].items():
            leg_reports[int(leg)] = part
            for link in LINKS:
                for tid in part.get(link, []):
                    assign.setdefault(tid, {"votes": []})["votes"].append((int(leg), link))
        log(f"no-move: link assignment taken from {args.assign_from}; geometry re-derived from the zero pose")
    else:
        # No motion: trust the old layout for link assignment; geometry is still re-derived.
        for tid, t in old_by_id.items():
            if t["kind"] != "chassis_tag":
                assign[tid] = {"votes": [(int(t["leg"]), t["frame"].split("_")[1])]}
        log("no-move: link assignment taken from the old layout; geometry re-derived from the zero pose")

    # ---- resolve link per tag
    link_of: dict[int, tuple[int, str]] = {}
    for tid, a in assign.items():
        votes = a["votes"]
        best = max(set(votes), key=votes.count)
        if len(set(votes)) > 1:
            log(f"tag {tid}: seen on more than one link {sorted(set(votes))}; taking {best}")
        link_of[tid] = best

    seen_ids = set().union(*(set(o["tags"]) for o in zero.values()))
    body_tag = 0 if 0 in seen_ids else None
    never_moved = seen_ids - set(link_of) - floor_ids - ({0} if body_tag is not None else set())

    # ---- geometry at the zero pose, all in the top camera's lid plane
    # Depth from a 27 mm tag is too noisy to use, so every position is the
    # tag centre's ray intersected with the shared horizontal plane of the
    # flat tags (normal `up` from the IPPE consensus). Directions in that
    # plane are exact up to the focal-length fit; depths cancel.
    tops = zero[top]["tags"]
    up = up_of.get(top)
    if up is None:
        log("top camera has no consensus up (fewer than two flat tags); aborting geometry")
        return 3
    Kt_inv = np.linalg.inv(K[top])

    def plane_pt(pix: np.ndarray) -> np.ndarray:
        # up points from the scene toward the camera, so the plane on the scene
        # side is up . X = -1 and the camera's foot on it is -up.
        ray = Kt_inv @ np.array([pix[0], pix[1], 1.0])
        return -ray / float(np.dot(up, ray))
    foot = -up

    def in_plane(v: np.ndarray) -> np.ndarray:
        v = v - up * float(np.dot(v, up))
        return v / (np.linalg.norm(v) + 1e-9)

    def plane_angle(v_from: np.ndarray, v_to: np.ndarray) -> float:
        """Rotation about +up taking v_from to v_to, degrees."""
        return math.degrees(math.atan2(float(np.dot(np.cross(v_from, v_to), up)), float(np.dot(v_from, v_to))))

    def tag_x_in_plane(corners: np.ndarray) -> np.ndarray:
        """Tag +x direction on the plane: back-project the corner-0 -> corner-1 chord."""
        return in_plane(plane_pt(corners[1]) - plane_pt(corners[0]))

    body_pt = plane_pt(center(tops[0])) if 0 in tops else None
    per_leg: dict[int, dict[str, list[int]]] = {}
    for tid, (leg, link) in link_of.items():
        per_leg.setdefault(leg, {"coxa": [], "femur": [], "tibia": []})[link].append(tid)

    new_tags: list[dict[str, Any]] = []
    notes: dict[int, dict[str, Any]] = {}
    lid_of: dict[tuple[int, str], int] = {}       # (leg, joint) -> lid tag id
    axis_of: dict[int, tuple[np.ndarray, np.ndarray]] = {}   # leg -> (point on axis, distal unit)
    kind_of: dict[int, str] = {}
    for leg, links in sorted(per_leg.items()):
        # Motion already says: coxa tags are the hip lid, tibia tags are knee yokes,
        # femur tags are the knee lid plus the two hip yokes. The knee lid is the
        # femur tag nearest the leg axis (body centre -> hip lid).
        hip_lid = None
        coxa_seen = [t for t in links["coxa"] if t in tops]
        if len(links["coxa"]) == 1:
            hip_lid = links["coxa"][0]
        elif coxa_seen and body_pt is not None:
            hip_lid = coxa_seen[0]
            notes.setdefault(hip_lid, {})["warning"] = f"several coxa tags {links['coxa']}; took the first seen"
        elif links["coxa"]:
            hip_lid = links["coxa"][0]
        for t in links["coxa"]:
            kind_of[t] = "servo_lid" if t == hip_lid else "unplaced"
        for t in links["tibia"]:
            kind_of[t] = "yoke_face"
        # axis through body centre and hip lid (both on the plane)
        axis = None
        if hip_lid in tops and body_pt is not None:
            a = plane_pt(center(tops[hip_lid]))
            axis = (a, in_plane(a - body_pt))
        femur_seen = [t for t in links["femur"] if t in tops]
        knee_lid = None

        def squareness(t: int) -> float:
            """A flat tag back-projects onto the lid plane as a square (chord ratio 1);
            a vertical face comes out stretched or squashed."""
            P = [plane_pt(tops[t][k]) for k in range(4)]
            a_, b_ = np.linalg.norm(P[1] - P[0]), np.linalg.norm(P[3] - P[0])
            return abs(math.log((a_ + 1e-9) / (b_ + 1e-9)))

        if femur_seen:
            knee_lid = min(femur_seen, key=squareness)
            sq = {t: round(squareness(t), 2) for t in femur_seen}
            notes.setdefault(knee_lid, {})["chord_log_ratio_by_tag"] = sq
            if sq[knee_lid] > 0.25:
                notes[knee_lid]["warning"] = "no femur tag projects as a square; knee lid uncertain"
            if axis is not None:
                q = plane_pt(center(tops[knee_lid])) - axis[0]
                edge_plane = float(np.linalg.norm(plane_pt(tops[knee_lid][1]) - plane_pt(tops[knee_lid][0])))
                notes[knee_lid]["off_axis_edges"] = round(abs(float(np.linalg.norm(np.cross(q, axis[1])))) / (edge_plane + 1e-9), 2)
        elif len(links["femur"]) == 1:
            knee_lid = links["femur"][0]
        elif links["femur"] and knee_lid is None:
            # no axis: fall back to the old layout's lid if it is among them
            old_lids = [t for t in links["femur"] if old_by_id.get(t, {}).get("kind") == "servo_lid"]
            knee_lid = old_lids[0] if old_lids else None
        for t in links["femur"]:
            kind_of[t] = "servo_lid" if t == knee_lid else "yoke_face"
        if hip_lid is not None:
            lid_of[(leg, "hip")] = hip_lid
        if knee_lid is not None:
            lid_of[(leg, "knee")] = knee_lid
        # distal direction: hip lid -> knee lid on the plane, else body -> a lid
        pts = [plane_pt(center(tops[t])) for t in (hip_lid, knee_lid) if t is not None and t in tops]
        if len(pts) == 2:
            axis_of[leg] = (pts[0], in_plane(pts[1] - pts[0]))
        elif pts and body_pt is not None:
            axis_of[leg] = (pts[0], in_plane(pts[0] - body_pt))
        elif axis is not None:
            axis_of[leg] = axis

    # Euler-z sign, calibrated on lids the old layout knows (majority wins, so a
    # few re-stuck lids do not flip it).
    votes = []
    for (leg, joint), tid in lid_of.items():
        if tid in tops and leg in axis_of and old_by_id.get(tid, {}).get("surface") == "horizontal":
            meas = plane_angle(tag_x_in_plane(tops[tid]), axis_of[leg][1])
            old = float(old_by_id[tid]["frame_from_tag"]["euler_xyz_deg"][2])
            log(f"lid {tid} L{leg} {joint}: measured {meas:+.1f} (sign +1) vs old {old:+.1f}")
            for sgn in (1.0, -1.0):
                votes.append((sgn, abs(((sgn * meas - old) + 180.0) % 360.0 - 180.0)))
    lid_sign = 1.0
    if votes:
        score = {sgn: float(np.median([d for ss, d in votes if ss == sgn])) for sgn in (1.0, -1.0)}
        lid_sign = min(score, key=score.get)
        log(f"lid euler-z sign {lid_sign:+.0f} (median disagreement with old layout: +1 {score[1.0]:.1f} deg, -1 {score[-1.0]:.1f} deg)")

    for tid, (leg, link) in sorted(link_of.items()):
        frame = f"L{leg}_{link}"
        entry: dict[str, Any] = {"id": tid, "leg": leg, "frame": frame}
        n = notes.setdefault(tid, {})
        kind = kind_of.get(tid, "yoke_face")
        if kind == "unplaced":
            n["unresolved"] = "second tag on the coxa; the layout has one hip lid per leg"
            new_tags.append(entry); continue
        if kind == "servo_lid":
            entry["kind"] = "servo_lid"; entry["joint"] = "hip" if link == "coxa" else "knee"; entry["surface"] = "horizontal"
            if tid in tops and leg in axis_of:
                raw = lid_sign * plane_angle(tag_x_in_plane(tops[tid]), axis_of[leg][1])
                snapped, resid = snap90(raw)
                entry["frame_from_tag"] = {"translation_m": None, "euler_xyz_deg": [0.0, 0.0, snapped]}
                n["measured_z_deg"] = round(raw, 1); n["snap_residual_deg"] = round(resid, 1)
            elif tid in old_by_id:
                entry["frame_from_tag"] = old_by_id[tid]["frame_from_tag"]; n["orientation_from_old_layout"] = True
            new_tags.append(entry); continue
        entry["kind"] = "yoke_face"; entry["joint"] = "hip" if link == "femur" else "knee"
        if link == "coxa":
            n["warning"] = "vertical tag on the coxa: the layout has no place for it"
        geo = None
        if tid in tops and leg in axis_of:
            a, d = axis_of[leg]
            y_link = np.cross(up, d)                 # z cross x = y (right-handed link frame)
            c = plane_pt(center(tops[tid]))
            off = c - a
            off -= d * float(np.dot(off, d))
            edge_plane = float(np.linalg.norm(plane_pt(tops[tid][1]) - plane_pt(tops[tid][0])))
            side_val = float(np.dot(off, y_link)) / (edge_plane + 1e-9)
            if abs(side_val) >= 0.15:
                side = "+y" if side_val > 0 else "-y"
                # tag +x is either along the leg (+-x) or vertical (+-z); on a square
                # mounting exactly one of the tag's two chords is horizontal. The chord
                # more parallel to the leg is the horizontal one. A vertical chord's
                # higher end back-projects farther from the camera's foot on the plane.
                P = [plane_pt(tops[tid][k]) for k in range(4)]
                ch_x, ch_y = P[1] - P[0], P[3] - P[0]
                ax = abs(float(np.dot(in_plane(ch_x), d))); ay = abs(float(np.dot(in_plane(ch_y), d)))
                if ax >= ay:
                    tx = "+x" if float(np.dot(ch_x, d)) > 0 else "-x"
                else:
                    tx = "+z" if np.linalg.norm(P[1] - foot) > np.linalg.norm(P[0] - foot) else "-z"
                comp = {"chord_x_along_leg": round(ax, 2), "chord_y_along_leg": round(ay, 2),
                        "x_end_heights": [round(float(np.linalg.norm(P[0] - foot)), 3), round(float(np.linalg.norm(P[1] - foot)), 3)]}
                tokens = {"x": tx, "z": side}
                yv = np.cross(AXIS_VECTORS[side], AXIS_VECTORS[tx]); j = int(np.argmax(np.abs(yv)))
                tokens["y"] = ("+" if yv[j] > 0 else "-") + "xyz"[j]
                geo = {"side": side, "side_margin": round(abs(side_val), 2), "tokens": tokens,
                       "quaternion_xyzw": quat_xyzw(tokens_to_rot(tokens)), "chords": comp, "camera": top}
            else:
                n["side_undecided"] = round(side_val, 2)
        if geo:
            entry["mount_side"] = geo["side"]
            entry["frame_from_tag"] = {"translation_m": None, "quaternion_xyzw": geo["quaternion_xyzw"], "tag_axes_in_frame": geo["tokens"]}
            n["yoke_geometry"] = geo
        elif tid in old_by_id and old_by_id[tid].get("kind") == "yoke_face":
            entry["mount_side"] = old_by_id[tid]["mount_side"]; entry["frame_from_tag"] = old_by_id[tid]["frame_from_tag"]
            n["orientation_from_old_layout"] = True
        else:
            n["unresolved"] = "top camera did not see this face with its leg's lids"
        entry["observed_in"] = sorted(c for c in cams.indices if tid in zero[c]["tags"])
        new_tags.append(entry)
    for e in new_tags:
        e.setdefault("observed_in", sorted(c for c in cams.indices if e["id"] in zero[c]["tags"]))

    # Chassis tag: body +x is leg 0's direction rotated -30 deg about up ((leg+0.5)*60 azimuths).
    if body_tag is not None:
        entry = {"id": 0, "kind": "chassis_tag", "frame": "body", "surface": "horizontal"}
        if 0 in tops and axis_of:
            # Each leg says where body +x is: its own direction minus its azimuth. In the
            # layout's convention the euler z is the angle from frame x to tag x, so the
            # measured angle (tag x -> leg) enters with the lids' sign.
            estimates = []
            tx0 = tag_x_in_plane(tops[0])
            for leg, (_a, d) in sorted(axis_of.items()):
                ang = lid_sign * plane_angle(tx0, d)             # frame_from_tag z if this leg were body +x
                # legs sit CCW about up at azimuth (leg+0.5)*60 from body +x
                estimates.append((leg, ((ang - lid_sign * (leg + 0.5) * 60.0 + 180.0) % 360.0) - 180.0))
            vals = np.radians([e[1] for e in estimates])
            raw = math.degrees(math.atan2(np.mean(np.sin(vals)), np.mean(np.cos(vals))))
            entry["frame_from_tag"] = {"translation_m": None, "euler_xyz_deg": [0.0, 0.0, round(raw, 1)]}
            notes[0] = {"measured_z_deg": round(raw, 1), "per_leg_estimates": [(l, round(v, 1)) for l, v in estimates],
                        "old_z_deg": old_by_id[0]["frame_from_tag"]["euler_xyz_deg"][2] if 0 in old_by_id else None}
        else:
            entry["frame_from_tag"] = old_by_id[0]["frame_from_tag"]; notes[0] = {"orientation_from_old_layout": True}
        entry["observed_in"] = sorted(c for c in cams.indices if 0 in zero[c]["tags"])
        new_tags.insert(0, entry)

    # ---- carry forward faces nobody saw, so the validator's pairing holds; mark them.
    have = {(t["leg"], t["joint"], t.get("mount_side")) for t in new_tags if t["kind"] == "yoke_face"}
    have_lids = {(t["leg"], t["joint"]) for t in new_tags if t["kind"] == "servo_lid"}
    carried: list[int] = []
    for tid, t in old_by_id.items():
        if any(e["id"] == tid for e in new_tags):
            continue
        key = (int(t["leg"]), t["joint"], t.get("mount_side")) if t["kind"] == "yoke_face" else None
        if t["kind"] == "yoke_face" and key not in have:
            e = dict(t); e["verified"] = False; new_tags.append(e); have.add(key); carried.append(tid)
        elif t["kind"] == "servo_lid" and (int(t["leg"]), t["joint"]) not in have_lids:
            e = dict(t); e["verified"] = False; new_tags.append(e); have_lids.add((int(t["leg"]), t["joint"])); carried.append(tid)

    # ---- diff against the old layout
    diff: dict[str, list] = {"unchanged": [], "changed": [], "new": [], "gone_or_unseen": [], "carried_unverified": carried,
                             "never_moved_unassigned": sorted(never_moved)}
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
    layout["captured"] = dt.date.today().isoformat()
    unresolved = [e["id"] for e in new_tags if "frame_from_tag" not in e or (e["kind"] == "yoke_face" and "mount_side" not in e)]
    new_tags = [e for e in new_tags if e["id"] not in unresolved]
    diff["unresolved"] = unresolved
    layout["robot_tags"] = new_tags
    layout["evidence"] = {"method": "hexapod-relayout: motion partition + IPPE geometry + Claude cross-check",
                          "cameras": cams.indices, "top_camera": top, "moved": not args.no_move,
                          "decoded_unique_ids": sorted(seen_ids), "output_dir": str(out)}
    layout["limitations"] = [
        "Robot-tag translations are null; the tracker never reads them.",
        "Yoke faces carried from the old layout with verified=false were not seen by any camera; rotate the robot and rerun to confirm them.",
        "Lid rotations are snapped to 90 deg; residuals are in the report.",
    ]
    tag_map = dict(old_map)
    parts = []
    for leg in LEGS:
        for joint in ("hip", "knee"):
            faces = {t.get("mount_side"): t["id"] for t in new_tags if t["kind"] == "yoke_face" and t["leg"] == leg and t["joint"] == joint}
            if set(faces) == {"+y", "-y"}:
                parts.append({"id": f"leg{leg}_{joint}_servo", "display_name": f"Leg {leg} {joint} servo assembly",
                              "tag_ids": [faces["+y"], faces["-y"]], "pose_reference": "centroid of the two opposing yoke-face tag centers",
                              "surface": "vertical", "yaw_period_degrees": 180.0})
    tag_map["parts"] = parts
    problems = validate_layout(layout, floor, tag_map)

    report = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "moved": not args.no_move,
              "diff": diff, "legs": leg_reports, "notes": {str(k): v for k, v in notes.items()},
              "validation_problems": problems, "focal_fit": focal_fit, "claude_cost_usd": round(claude_total, 3), "log": log_lines}
    (out / "layout.json").write_text(json.dumps(layout, indent=1))
    (out / "hexapod_tag_map.json").write_text(json.dumps(tag_map, indent=1))
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    md = [f"# Tag re-layout {report['generated']}", "",
          f"moved robot: {not args.no_move}; cameras {cams.indices}; top camera {top}", "",
          f"- unchanged: {diff['unchanged']}", f"- changed: {diff['changed']}", f"- new ids: {diff['new']}",
          f"- gone or unseen: {diff['gone_or_unseen']}", f"- carried unverified: {carried}",
          f"- never moved, unassigned (floor? body?): {diff['never_moved_unassigned']}", "",
          "## validation", *([f"- {p}" for p in problems] or ["- clean"]), "", "## per-leg motion"]
    for leg, r in leg_reports.items():
        md.append(f"- leg {leg}: coxa {r['coxa']} femur {r['femur']} tibia {r['tibia']}; conflicts {r['conflicts']}; "
                  f"claude: {json.dumps((r.get('claude') or {}).get('json'))[:400]}")
    md += ["", "## per-tag notes", *(f"- {k}: {json.dumps(v, default=str)[:300]}" for k, v in sorted(notes.items()))]
    (out / "report.md").write_text("\n".join(md) + "\n")
    log(f"diff: {json.dumps(diff)}")
    log(f"validation: {problems or 'clean'}")

    if args.write:
        if problems and not args.force:
            log("NOT writing: validation problems"); return 2
        if problems:
            log(f"writing despite validator notes (--force): {problems}")
        cfg = Path(args.config_dir)
        bak = cfg / f"backup-{dt.date.today().isoformat()}"
        bak.mkdir(exist_ok=True)
        for name in ("hexapod-1-apriltag-layout.json", "hexapod_tag_map.json"):
            if (cfg / name).exists() and not (bak / name).exists():
                shutil.copy2(cfg / name, bak / name)
        shutil.copy2(out / "layout.json", cfg / "hexapod-1-apriltag-layout.json")
        shutil.copy2(out / "hexapod_tag_map.json", cfg / "hexapod_tag_map.json")
        log(f"wrote configs; old copies in {bak}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--camera-url", default="http://127.0.0.1:8766")
    ap.add_argument("--robot-url", default="http://192.168.4.39:8080")
    ap.add_argument("--cameras", default="0,1,2")
    ap.add_argument("--top-camera", default="2")
    ap.add_argument("--config-dir", default=str(CONFIG_DIR))
    ap.add_argument("--no-move", action="store_true", help="analyse the current pose only")
    ap.add_argument("--assign-from", default=None, help="report.json of an earlier pass: reuse its motion-based link assignment")
    ap.add_argument("--force", action="store_true", help="with --write: install even if the validator lists missing faces")
    ap.add_argument("--no-claude", action="store_true")
    ap.add_argument("--write", action="store_true", help="install into configs/ if validation passes")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
