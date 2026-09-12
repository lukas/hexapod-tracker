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


def _swing(tags, ids, deg):
    pivot = center(tags[0])
    a = math.radians(deg)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    for tid in ids:
        tags[tid] = (R @ (tags[tid] - pivot).T).T + pivot
    return tags


def test_a_spare_tag_far_from_the_robot_is_ignored():
    zero, sizes, layout, floor = _replay()
    tags = dict(zero[2])
    del tags[4]                                                        # leg 5's real hip lid out of view
    from hexapod_tracker.tag_calibration import edge_px
    tags[4] = tags[1] + np.array([16.0 * edge_px(tags[0]), 0.0])         # a printed copy lying on the floor, far away
    r = zc.check({**zero, 2: tags}, sizes, layout, floor, 2)
    assert r["far_ids"] == [4]
    assert r["off"] == [], r["summary"]
    assert r["legs"]["5"]["lids_seen"] == [14]                          # knee lid only: still measurable
    assert r["legs"]["5"]["verdict"] == "ok"


def test_a_leg_seen_only_by_its_hip_lid_is_not_judged():
    zero, sizes, layout, floor = _replay()
    tags = dict(zero[2])
    del tags[14]                                                       # leg 5: only hip lid 4 remains
    r = zc.check({**zero, 2: tags}, sizes, layout, floor, 2)
    assert r["legs"]["5"]["verdict"] == "unseen" and r["legs"]["5"]["why"] == "hip lid only"
    assert r["ok"]


def test_one_bad_leg_does_not_drag_the_others_off():
    zero, sizes, layout, floor = _replay()
    tags = _swing(dict(zero[2]), (5, 9), 40.0)                          # leg 3 both lids turned 40 deg
    r = zc.check({**zero, 2: tags}, sizes, layout, floor, 2)
    assert r["off"] == [3], r["summary"]
    others = [abs(e["residual_deg"]) for l, e in r["legs"].items() if l != "3" and e["residual_deg"] is not None]
    assert max(others) < 12, r["legs"]      # leg 4 reads from one lid and sits near 10 anyway


def test_two_legs_that_disagree_are_ambiguous_not_a_hold():
    zero, sizes, layout, floor = _replay()
    keep = {0, 1, 7, 109, 113, 3, 10} | {tid for tid in zero[2] if tid >= 100}   # legs 0, 1 and 2 only
    tags = {tid: c for tid, c in zero[2].items() if tid in keep}
    del tags[109]; del tags[113]                                        # leg 1 gone: legs 0 and 2 remain
    tags = _swing(tags, (3, 10), 30.0)                                  # and leg 2 is turned
    r = zc.check({**zero, 2: tags}, sizes, layout, floor, 2)
    assert not r["ok"] and r["off"] == [] and r["error"] and "disagree" in r["error"]
