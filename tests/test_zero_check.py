"""hexapod-zero-check: the saved 2026-09-11 zero pose agrees with the installed layout; a leg turned in the image is caught."""
import json
import math
from pathlib import Path

import numpy as np

from hexapod_tracker import zero_check as zc
from hexapod_tracker.paths import CONFIG_DIR
from hexapod_tracker.tag_calibration import center, load_configs, load_tags

FIXTURE = Path(__file__).parent / "fixtures" / "tag_calibration_20260911"


def _replay():
    layout, _m, floor = load_configs(CONFIG_DIR)
    zero, sizes = load_tags(FIXTURE / "zero_tags.json")
    return zero, sizes, layout, floor


def test_replay_of_the_installed_zero_pose_agrees_with_the_layout():
    zero, sizes, layout, floor = _replay()
    r = zc.check(zero, sizes, layout, floor, 2)
    assert r["ok"] and r["off"] == [] and r["error"] is None
    assert all(abs(e["residual_deg"]) < 10 for e in r["legs"].values() if e["residual_deg"] is not None)
    assert "agrees" in r["summary"]
    # one-lid legs get more room than two-lid legs
    tols = {e["quality"]: e["tol_deg"] for e in r["legs"].values() if e.get("tol_deg")}
    assert tols.get("chassis_fallback", 0) > tols.get("two_lids", 99) or "chassis_fallback" not in tols


def test_a_leg_swung_in_the_image_is_reported_off():
    zero, sizes, layout, floor = _replay()
    top = 2
    tags = dict(zero[top])
    pivot = center(tags[0])                     # chassis tag centre
    ang = math.radians(25.0)
    R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
    for tid in (3, 10):                          # leg 2's two lids
        tags[tid] = (R @ (tags[tid] - pivot).T).T + pivot
    zero2 = dict(zero)
    zero2[top] = tags
    r = zc.check(zero2, sizes, layout, floor, top)
    assert r["off"] == [2], r["summary"]
    assert not r["ok"]
    assert abs(abs(r["legs"]["2"]["residual_deg"]) - 25.0) < 8, r["legs"]["2"]
    assert "legs [2]" in r["summary"]


def test_missing_top_camera_is_an_error_not_a_pass():
    zero, sizes, layout, floor = _replay()
    r = zc.check({1: zero[1]}, {1: sizes[1]}, layout, floor, 2)
    assert not r["ok"] and r["error"]


def test_cli_replay_exits_zero_and_prints_json(tmp_path, capsys):
    rc = zc.main(["--replay", str(FIXTURE), "--json", "--out", str(tmp_path)])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and out["ok"]
    assert (tmp_path / "zero_check.json").exists()
