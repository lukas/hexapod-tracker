"""hexapod-zero-check: does the robot's zero pose look like zero from the top camera?

The runner's start-pose check reads the encoders, which sit on each servo's
output shaft. A horn whose screws have slipped passes that check while the
leg points somewhere else. The lids do not lie: at the zero pose every leg's
lid tags should sit along that leg's azimuth in the installed layout
(``leg_zero_azimuth_body_deg``). This re-derives the azimuths from one
observation with the calibration program's own geometry and reports, per
leg, how far the camera's answer is from the layout's.

    hexapod-zero-check [--top-camera 2] [--tol-deg 12] [--out DIR] [--json]

Exit 0 when every seen leg is within tolerance, 2 when a leg is off, 3 when
nothing could be measured (no frame, no plane). With ``--replay DIR`` the
check runs on a saved ``zero_tags.json`` instead of the cameras.

Written 2026-09-11; the operator asked that a robot "supposedly in zero pose"
that "looks wildly off" be double-checked before it moves.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .paths import CONFIG_DIR
from .tag_calibration import (
    Cameras, annotate, derive_layout, load_configs, load_tags, resolve_links, tags_only,
    votes_from_layout, wrap_deg,
)

DEFAULT_TOL_DEG = 12.0


def check(zero: dict[int, dict[int, np.ndarray]], sizes: dict[int, tuple[int, int]], layout: dict, floor: dict,
          top: int, *, tol_deg: float = DEFAULT_TOL_DEG, log: Optional[Callable[[str], None]] = None) -> dict[str, Any]:
    """Pure: per-leg azimuth residual of one zero-pose observation against the layout.

    ``zero`` maps camera -> tag id -> 4x2 corners (snapshot pixels). Returns
    ``ok`` (no seen leg is off), ``legs`` per leg with ``azimuth_deg``,
    ``expected_deg``, ``residual_deg``, ``lids_seen`` and ``verdict``
    (ok | off | unseen), plus ``off``/``unseen`` leg lists.
    """
    log = log or (lambda msg: None)
    floor_ids = {int(t["id"]) for t in floor["tags"]}
    link_of = resolve_links(votes_from_layout(layout), log, layout)
    expected = {int(k): float(v) for k, v in (layout.get("leg_zero_azimuth_body_deg") or {}).items()}
    out: dict[str, Any] = {"ok": False, "top_camera": top, "tol_deg": tol_deg, "legs": {}, "off": [], "unseen": [],
                           "error": None, "warning": None}
    if not expected:
        out["error"] = "layout has no leg_zero_azimuth_body_deg; run hexapod-calibrate-tags first"
        return out
    derived = derive_layout(zero, sizes, link_of, layout, floor_ids, top, device={0: "OV9281"}, log=log)
    out["chassis_tag_seen"] = bool(derived.get("chassis_tag_seen"))
    out["warning"] = derived.get("warning")
    if derived.get("error"):
        out["error"] = derived["error"]
        return out
    by_id = {int(t["id"]): t for t in layout.get("robot_tags", [])}
    seen_top = set(zero.get(top, {}))
    for leg in range(6):
        lids = sorted(tid for tid, (l, _link) in link_of.items()
                      if l == leg and by_id.get(tid, {}).get("surface") == "horizontal")
        az = derived["azimuth_deg"].get(leg)
        entry: dict[str, Any] = {"lids": lids, "lids_seen": sorted(t for t in lids if t in seen_top),
                                 "azimuth_deg": az, "expected_deg": expected.get(leg),
                                 "quality": derived["axis_quality"].get(leg)}
        if az is None or leg not in expected:
            entry["residual_deg"], entry["verdict"] = None, "unseen"
            out["unseen"].append(leg)
        else:
            res = round(wrap_deg(az - expected[leg]), 1)
            entry["residual_deg"] = res
            # One lid only: the axis runs through the chassis tag instead of
            # two lids and is about twice as noisy (leg 4 read 8.4 deg from a
            # single lid on the 2026-09-11 replay), so it gets more room.
            entry["tol_deg"] = tol_deg if entry["quality"] == "two_lids" else 1.5 * tol_deg
            entry["verdict"] = "off" if abs(res) > entry["tol_deg"] else "ok"
            if entry["verdict"] == "off":
                out["off"].append(leg)
        out["legs"][str(leg)] = entry
    out["ok"] = not out["off"] and len(out["unseen"]) < 6
    if len(out["unseen"]) == 6:
        out["error"] = "no leg's lids were seen by the top camera"
    out["summary"] = (
        "camera agrees with the layout" if out["ok"] and not out["unseen"] else
        f"legs {out['off']} point " + ", ".join(
            f"{out['legs'][str(l)]['residual_deg']:+.0f} deg" for l in out["off"]) + " from where the layout says zero is"
        if out["off"] else
        f"legs {out['unseen']} not seen; the rest agree" if out["ok"] else str(out["error"]))
    return out


def observe(camera_url: str, top: int) -> tuple[dict[int, dict[int, np.ndarray]], dict[int, tuple[int, int]], dict]:
    cams = Cameras(camera_url, [top])
    obs = cams.observe()
    return tags_only(obs), {c: o["size"] for c, o in obs.items()}, obs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--camera-url", default="http://127.0.0.1:8766")
    ap.add_argument("--top-camera", default="2")
    ap.add_argument("--config-dir", default=str(CONFIG_DIR))
    ap.add_argument("--tol-deg", type=float, default=DEFAULT_TOL_DEG)
    ap.add_argument("--replay", default=None, metavar="DIR", help="use DIR/zero_tags.json instead of the cameras")
    ap.add_argument("--out", default=None, help="write zero_check.json and the annotated top frame here")
    ap.add_argument("--json", action="store_true", help="print only the JSON result")
    a = ap.parse_args(argv)
    top = int(a.top_camera)
    layout, _tag_map, floor = load_configs(Path(a.config_dir))
    lines: list[str] = []
    log = lines.append
    image = None
    if a.replay:
        zero, sizes = load_tags(Path(a.replay).expanduser() / "zero_tags.json")
    else:
        try:
            zero, sizes, obs = observe(a.camera_url, top)
        except Exception as exc:  # noqa: BLE001 - no camera is a measured "cannot see"
            result = {"ok": False, "error": f"camera server: {type(exc).__name__}: {exc}", "top_camera": top}
            print(json.dumps(result, indent=None if a.json else 1))
            return 3
        image = (obs.get(top) or {}).get("image")
    result = check(zero, sizes, layout, floor, top, tol_deg=a.tol_deg, log=log)
    result["log"] = lines
    if a.out:
        out = Path(a.out).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        (out / "zero_check.json").write_text(json.dumps(result, indent=1))
        if image is not None:
            import cv2
            labels = {tid: f"{tid}" for tid in zero.get(top, {})}
            cv2.imwrite(str(out / f"zero_check_cam{top}.jpg"), annotate(image, zero.get(top, {}), labels=labels))
            result["frame"] = str(out / f"zero_check_cam{top}.jpg")
    if a.json:
        print(json.dumps(result))
    else:
        for l in lines:
            print(l)
        print(result.get("summary") or result.get("error"))
        for leg, e in sorted(result.get("legs", {}).items()):
            print(f"  leg {leg}: {e['verdict']:6} azimuth {e['azimuth_deg']} expected {e['expected_deg']} "
                  f"residual {e['residual_deg']} lids seen {e['lids_seen']}")
    return 0 if result["ok"] else (2 if result.get("off") else 3)


if __name__ == "__main__":
    sys.exit(main())
