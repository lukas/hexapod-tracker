"""Fit a provisional focal length for one fixed camera from the floor grid.

``hexapod-fit-intrinsics`` pulls native-resolution luma frames from the
running camera server, detects the surveyed floor anchors, and picks the
focal length that minimises the pooled PnP reprojection error of their known
corners with the principal point fixed at the image centre, square pixels
and zero distortion. That is the constrained single-plane method behind every
provisional entry in the intrinsics file; it is not a substitute for a
multi-pose board calibration.

It refuses to write when the data cannot pin the focal length down: fewer
than three anchors, collinear anchors, or a near-fronto-parallel view where
reprojection error barely depends on focal length. Two methods that looked
plausible and failed on a near-top-down camera are deliberately absent here:
making chassis/coxa lid normals parallel (flat objective for small clustered
coplanar tags) and the closed-form focal length from a plane homography
(divides by perspective terms that vanish for a fronto-parallel plane).
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Sequence
import urllib.request

import cv2
import numpy as np

from .camera_server import detect_tag_corners, make_tag_detector
from .paths import CONFIG_DIR
from .planar_pose import PlanarPoseEstimator

DEFAULT_OUTPUT = CONFIG_DIR / "camera_intrinsics_lab_20260912.json"
MAX_BAND_FRACTION = 0.10  # refuse when +0.5 px of rms spans more than +/-10 % of f


@dataclass
class FocalFit:
    f_px: float
    rms_px: float
    band_px: tuple[float, float]  # focal lengths within +0.5 px rms of the minimum
    per_frame_std_px: float
    frames: int
    anchors: list[int]

    @property
    def band_fraction(self) -> float:
        low, high = self.band_px
        return max(self.f_px - low, high - self.f_px) / self.f_px


class FitError(ValueError):
    """The observations cannot determine a focal length."""


def anchors_are_collinear(centers: np.ndarray, tolerance_mm: float = 20.0) -> bool:
    """True when every anchor centre lies within ``tolerance_mm`` of one line."""
    points = np.asarray(centers, dtype=np.float64)
    if len(points) < 3:
        return True
    centred = points - points.mean(axis=0)
    _u, singular, _vt = np.linalg.svd(centred, full_matrices=False)
    return bool(singular[1] < tolerance_mm)


def pooled_rms(f: float, image_size: tuple[int, int], observations: Sequence[tuple[np.ndarray, np.ndarray]]) -> float:
    width, height = image_size
    K = np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])
    zeros = np.zeros(5)
    total = 0.0
    count = 0
    for world_xy, image in observations:
        obj = np.hstack([world_xy, np.zeros((len(world_xy), 1))]).astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj, image, K, zeros, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            continue
        ok, rvec, tvec = cv2.solvePnP(obj, image, K, zeros, rvec, tvec, True, cv2.SOLVEPNP_ITERATIVE)
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, zeros)
        total += float(np.sum((proj.reshape(-1, 2) - image) ** 2))
        count += len(image)
    if count == 0:
        raise FitError("PnP failed on every frame")
    return math.sqrt(total / count)


def fit_focal_length(
    observations: Sequence[tuple[np.ndarray, np.ndarray]],
    image_size: tuple[int, int],
    anchors: Sequence[int],
    *,
    search: tuple[float, float] = (0.4, 2.5),  # multiples of image width
    coarse_step_px: float = 20.0,
) -> FocalFit:
    """Choose the focal length minimising pooled reprojection error.

    ``observations`` holds one ``(world_xy_mm, image_px)`` corner pair set per
    frame. The result carries the width of the minimum so callers can refuse
    an ill-conditioned fit instead of writing a number that only looks exact.
    """
    if not observations:
        raise FitError("no frames with enough anchors")
    width = image_size[0]
    grid = np.arange(search[0] * width, search[1] * width, coarse_step_px)
    curve = np.array([pooled_rms(f, image_size, observations) for f in grid])
    best = int(np.argmin(curve))
    fine = np.linspace(grid[max(best - 2, 0)], grid[min(best + 2, len(grid) - 1)], 81)
    fine_curve = np.array([pooled_rms(f, image_size, observations) for f in fine])
    i = int(np.argmin(fine_curve))
    f_best, rms = float(fine[i]), float(fine_curve[i])
    within = grid[curve <= rms + 0.5]
    band = (float(within.min()), float(within.max()))
    per_frame = []
    for observation in observations:
        one = np.array([pooled_rms(f, image_size, [observation]) for f in grid])
        per_frame.append(grid[int(np.argmin(one))])
    return FocalFit(
        f_px=f_best,
        rms_px=rms,
        band_px=band,
        per_frame_std_px=float(np.std(per_frame)),
        frames=len(observations),
        anchors=sorted(int(a) for a in anchors),
    )


def check_conditioning(fit: FocalFit, anchor_centers: np.ndarray) -> None:
    if len(fit.anchors) < 3:
        raise FitError(f"only {len(fit.anchors)} floor anchors seen; need three that are not in a line")
    if anchors_are_collinear(anchor_centers):
        raise FitError(f"floor anchors {fit.anchors} are collinear; the fit cannot separate focal length from distance")
    if fit.band_fraction > MAX_BAND_FRACTION:
        low, high = fit.band_px
        raise FitError(
            f"focal length poorly constrained: +0.5 px of reprojection error spans {low:.0f}-{high:.0f} px "
            f"around {fit.f_px:.0f} (+/-{100 * fit.band_fraction:.0f} %); tilt the camera or show more anchors"
        )


def intrinsics_entry(fit: FocalFit, image_size: tuple[int, int], *, stable_id: str, device_name: str,
                     capture_size: tuple[int, int] | None) -> dict[str, Any]:
    width, height = image_size
    f = fit.f_px
    return {
        "device_name": device_name,
        "stable_id": stable_id,
        "image_size": {"width": width, "height": height},
        "camera_matrix": [[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]],
        "distortion_coefficients": [0.0] * 5,
        "quality": "provisional",
        "method": (
            f"Constrained single-plane fit to {fit.frames} native {width}x{height} luma frames of floor anchors "
            f"{fit.anchors}: focal length minimising pooled PnP reprojection error of the anchors' surveyed corners; "
            "principal point fixed at the image centre, square pixels, zero skew, zero distortion (hexapod-fit-intrinsics)."
        ),
        "floor_reprojection_rms_px": round(fit.rms_px, 3),
        "focal_sample_standard_deviation_px": round(fit.per_frame_std_px, 1),
        "focal_uncertainty_band_px": {
            "within_plus_0_5px_rms": [round(fit.band_px[0]), round(fit.band_px[1])],
            "note": "The floor map's anchor tolerance sets the error floor, not the lens.",
        },
        "horizontal_fov_deg": round(2 * math.degrees(math.atan(width / 2 / f)), 1),
        **({"capture_size": [capture_size[0], capture_size[1]]} if capture_size else {}),
        "captured": time.strftime("%Y-%m-%d"),
    }


def upsert_entry(document: dict[str, Any], key: str, entry: dict[str, Any]) -> dict[str, Any]:
    """Replace the entry with the same stable id, or add under ``key``."""
    cameras = dict(document.get("cameras") or {})
    for existing_key, spec in list(cameras.items()):
        if isinstance(spec, dict) and spec.get("stable_id") == entry["stable_id"]:
            cameras.pop(existing_key)
            key = existing_key
            break
    cameras[key] = entry
    return {**document, "cameras": cameras}


def _fetch(url: str, timeout: float = 15.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read()


def collect_observations(server: str, slot: int, frames: int, interval_s: float, estimator: PlanarPoseEstimator,
                         *, log=print) -> tuple[list[tuple[np.ndarray, np.ndarray]], tuple[int, int], set[int]]:
    detector = make_tag_detector()
    observations: list[tuple[np.ndarray, np.ndarray]] = []
    seen: set[int] = set()
    size: tuple[int, int] | None = None
    for i in range(frames):
        png = _fetch(f"{server}/native-luma/{slot}.png")
        gray = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FitError("server returned an undecodable frame")
        size = (gray.shape[1], gray.shape[0])
        corners = {int(k): np.asarray(v, dtype=np.float64).reshape(4, 2) for k, v in detect_tag_corners(gray, detector).items()}
        visible = [tag for tag in estimator.active_anchor_ids if tag in corners]
        if len(visible) >= 3:
            seen.update(visible)
            world = np.concatenate([estimator.anchor_corners(tag) for tag in visible])
            image = np.concatenate([corners[tag] for tag in visible])
            observations.append((world, image))
        log(f"frame {i + 1}/{frames}: anchors {visible}")
        time.sleep(interval_s)
    if size is None:
        raise FitError("no frames fetched")
    return observations, size, seen


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--server", default="http://127.0.0.1:8766")
    parser.add_argument("--slot", type=int, required=True, help="camera slot on the server (see /status.json)")
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--interval", type=float, default=0.5, help="seconds between frames")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="identity-keyed intrinsics file to update")
    parser.add_argument("--floor-map", type=Path, default=CONFIG_DIR / "floor_tag_map.json")
    parser.add_argument("--part-map", type=Path, default=CONFIG_DIR / "hexapod_tag_map.json")
    parser.add_argument("--robot-tag-layout", type=Path, default=CONFIG_DIR / "hexapod-1-apriltag-layout.json")
    parser.add_argument("--dry-run", action="store_true", help="fit and report without writing")
    args = parser.parse_args(argv)

    status = json.loads(_fetch(f"{args.server}/status.json"))
    camera = next((c for c in status["cameras"] if int(c["index"]) == args.slot), None)
    if camera is None:
        print(f"no slot {args.slot} on {args.server}", file=sys.stderr)
        return 2
    stable_id = str(camera.get("requested_stable_id") or "").strip()
    device_name = str(camera.get("device_name") or "").strip()
    if not stable_id:
        print("slot is not pinned to a stable id; start the server through tools/camera_service.sh", file=sys.stderr)
        return 2
    estimator = PlanarPoseEstimator(
        json.loads(args.floor_map.read_text()),
        json.loads(args.part_map.read_text()),
        json.loads(args.robot_tag_layout.read_text()),
        None,
    )
    try:
        observations, size, seen = collect_observations(args.server, args.slot, args.frames, args.interval, estimator)
        fit = fit_focal_length(observations, size, sorted(seen))
        centers = np.array([estimator.anchors[tag]["center"][:2] for tag in fit.anchors], dtype=np.float64)
        check_conditioning(fit, centers)
    except FitError as error:
        print(f"refusing to write: {error}", file=sys.stderr)
        return 1
    print(json.dumps({**asdict(fit), "device_name": device_name, "stable_id": stable_id,
                      "hfov_deg": round(2 * math.degrees(math.atan(size[0] / 2 / fit.f_px)), 1)}, indent=1))
    if args.dry_run:
        return 0
    capture = (int(camera["native_capture_width"]), int(camera["native_capture_height"])) if camera.get("native_capture_width") else None
    entry = intrinsics_entry(fit, size, stable_id=stable_id, device_name=device_name, capture_size=capture)
    document = json.loads(args.output.read_text()) if args.output.exists() else {"schema_version": 1, "quality": "provisional", "cameras": {}}
    key = device_name.lower().replace(" ", "_") or stable_id
    args.output.write_text(json.dumps(upsert_entry(document, key, entry), indent=1) + "\n")
    print(f"wrote {args.output}; restart the camera server to load it", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
