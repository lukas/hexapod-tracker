"""hexapod-calibrate-tags: geometry on a synthetic camera and a replay of real data."""
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from hexapod_tracker import tag_calibration as tc
from hexapod_tracker.layout_audit import validate_layout
from hexapod_tracker.paths import CONFIG_DIR

FIXTURE = Path(__file__).parent / "fixtures" / "tag_calibration_20260911"


# ------------------------------------------------------------ synthetic rig

def _camera(width=1280, height=720, f=1000.0, height_m=0.9, tilt_deg=8.0):
    """A camera looking down at the floor, slightly tilted so the plane is not degenerate."""
    K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])
    # world: z up, camera above the origin looking down (-z), rotated a little about x
    R_wc = Rotation.from_euler("xyz", [180.0 + tilt_deg, 0.0, 0.0], degrees=True).as_matrix()  # camera axes in world
    t_wc = np.array([0.0, 0.0, height_m])
    return K, R_wc, t_wc


def _project(points_w, K, R_wc, t_wc):
    Pc = (R_wc.T @ (np.asarray(points_w) - t_wc).T).T
    uv = (K @ Pc.T).T
    return uv[:, :2] / uv[:, 2:3]


def _tag_corners_world(center, R_wt, size=tc.TAG_M):
    """Corners 0..3 of a tag whose axes in world are the columns of R_wt (OpenCV order)."""
    h = size / 2
    local = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])
    return (R_wt @ local.T).T + np.asarray(center)


def _lid_rotation(leg_dir_deg, euler_z):
    """Flat tag on a link pointing at leg_dir_deg; frame_from_tag euler z = euler_z."""
    R_w_link = Rotation.from_euler("z", leg_dir_deg, degrees=True).as_matrix()
    R_link_tag = Rotation.from_euler("z", euler_z, degrees=True).as_matrix()
    return R_w_link @ R_link_tag


def _yoke_rotation(leg_dir_deg, tokens):
    R_w_link = Rotation.from_euler("z", leg_dir_deg, degrees=True).as_matrix()
    return R_w_link @ tc.tokens_to_rot(tokens)


def _synthetic_scene(body_x_deg=10.0, azimuth_offsets=None, chassis_z=-35.0):
    """Six straight legs at the zero pose plus a chassis tag and three floor tags."""
    azimuth_offsets = azimuth_offsets or {}
    K, R_wc, t_wc = _camera()
    tags_w = {}
    truth = {"lids": {}, "yokes": {}, "azimuth": {}}
    link_of = {}
    body_c = np.array([0.0, 0.0, 0.08])
    tid = 1
    for leg in range(6):
        az = body_x_deg + tc.nominal_azimuth_deg(leg) + azimuth_offsets.get(leg, 0.0)
        truth["azimuth"][leg] = tc.nominal_azimuth_deg(leg) + azimuth_offsets.get(leg, 0.0)
        d = np.array([math.cos(math.radians(az)), math.sin(math.radians(az)), 0.0])
        y = np.cross([0, 0, 1.0], d)
        hip_c = body_c + 0.11 * d
        knee_c = body_c + 0.21 * d
        hip_z = [90.0, -90.0, 180.0, 0.0, 90.0, -90.0][leg]
        knee_z = [90.0, 180.0, 0.0, -90.0, 90.0, 180.0][leg]
        tags_w[tid] = _tag_corners_world(hip_c, _lid_rotation(az, hip_z)); truth["lids"][tid] = hip_z
        link_of[tid] = (leg, "coxa"); tid += 1
        tags_w[tid] = _tag_corners_world(knee_c, _lid_rotation(az, knee_z)); truth["lids"][tid] = knee_z
        link_of[tid] = (leg, "femur"); tid += 1
        # hip yokes on the femur: one each side, 22 mm off axis, 15 mm below the lid plane
        for side, sign in (("+y", 1.0), ("-y", -1.0)):
            tokens = {"x": "+z" if side == "+y" else "-x", "z": side}
            yv = np.cross(tc.AXIS_VECTORS[side], tc.AXIS_VECTORS[tokens["x"]]); j = int(np.argmax(np.abs(yv)))
            tokens["y"] = ("+" if yv[j] > 0 else "-") + "xyz"[j]
            c = body_c + 0.15 * d + sign * 0.022 * y - np.array([0, 0, 0.015])
            tags_w[tid] = _tag_corners_world(c, _yoke_rotation(az, tokens)); truth["yokes"][tid] = (side, tokens)
            link_of[tid] = (leg, "femur"); tid += 1
    tags_w[0] = _tag_corners_world(body_c + np.array([0.02, -0.01, 0.0]),
                                   Rotation.from_euler("z", body_x_deg + chassis_z, degrees=True).as_matrix())
    truth["chassis_z"] = chassis_z
    floor = {100: (-0.3, -0.3), 101: (0.3, -0.3), 102: (-0.3, 0.3)}
    for f_id, (x, y_) in floor.items():
        tags_w[f_id] = _tag_corners_world([x, y_, 0.0], Rotation.from_euler("z", -90, degrees=True).as_matrix())
    corners = {t: _project(c, K, R_wc, t_wc) for t, c in tags_w.items()}
    old_layout = {"robot_tags": [
        {"id": 0, "kind": "chassis_tag", "frame": "body", "surface": "horizontal", "frame_from_tag": {"euler_xyz_deg": [0, 0, 0]}},
        *[{"id": t, "kind": "servo_lid", "surface": "horizontal", "leg": l, "frame": f"L{l}_{lk}",
           "joint": "hip" if lk == "coxa" else "knee", "frame_from_tag": {"euler_xyz_deg": [0, 0, 0]}}
          for t, (l, lk) in link_of.items() if t in truth["lids"]],
    ]}
    return {"zero": {2: corners}, "sizes": {2: (1280, 720)}, "link_of": link_of, "old_layout": old_layout,
            "floor_ids": set(floor), "truth": truth, "K": K}


def test_derive_layout_recovers_lids_yokes_azimuths_and_chassis_on_a_synthetic_rig():
    scene = _synthetic_scene(azimuth_offsets={1: 6.0, 4: -5.0})
    out = tc.derive_layout(scene["zero"], scene["sizes"], scene["link_of"], scene["old_layout"],
                           scene["floor_ids"], top=2)
    assert "error" not in out
    by_id = {t["id"]: t for t in out["tags"]}
    for tid, z in scene["truth"]["lids"].items():
        assert by_id[tid]["kind"] == "servo_lid"
        assert by_id[tid]["frame_from_tag"]["euler_xyz_deg"][2] == z, (tid, out["notes"][tid])
        assert abs(out["notes"][tid]["snap_residual_deg"]) < 4.0
    for tid, (side, tokens) in scene["truth"]["yokes"].items():
        assert by_id[tid]["kind"] == "yoke_face"
        assert by_id[tid]["mount_side"] == side, (tid, out["notes"][tid])
        assert by_id[tid]["frame_from_tag"]["tag_axes_in_frame"] == tokens, (tid, out["notes"][tid])
    for leg, az in scene["truth"]["azimuth"].items():
        assert abs(tc.wrap_deg(out["azimuth_deg"][leg] - az)) < 1.5, (leg, out["azimuth_deg"])
    assert abs(tc.wrap_deg(by_id[0]["frame_from_tag"]["euler_xyz_deg"][2] - scene["truth"]["chassis_z"])) < 2.0
    # the fitted focal is near the true one and the plane normal is consistent
    assert abs(out["focal_fit"][2]["f_px"] - scene["K"][0, 0]) / scene["K"][0, 0] < 0.15
    assert out["focal_fit"][2]["normal_spread_deg"] < 1.0


def test_derive_layout_reports_missing_top_camera_and_thin_planes():
    scene = _synthetic_scene()
    assert "error" in tc.derive_layout({}, {}, scene["link_of"], scene["old_layout"], scene["floor_ids"], top=2)
    only_one = {2: {k: v for k, v in scene["zero"][2].items() if k == 1}}
    assert "error" in tc.derive_layout(only_one, scene["sizes"], scene["link_of"], scene["old_layout"], set(), top=2)


def test_assemble_declares_gaps_carries_unseen_faces_and_validates():
    scene = _synthetic_scene()
    derived = tc.derive_layout(scene["zero"], scene["sizes"], scene["link_of"], scene["old_layout"],
                               scene["floor_ids"], top=2)
    old_layout = dict(scene["old_layout"])
    # the previous layout knew one knee face on leg 2 that no camera saw this time
    old_layout["robot_tags"] = old_layout["robot_tags"] + [
        {"id": 77, "kind": "yoke_face", "leg": 2, "joint": "knee", "frame": "L2_tibia", "mount_side": "+y",
         "frame_from_tag": {"quaternion_xyzw": [0.5, 0.5, 0.5, -0.5], "tag_axes_in_frame": {"x": "+z", "y": "+x", "z": "+y"}}}]
    floor = {"active_anchor_ids": [], "tags": []}      # the synthetic layout carries no floor block
    out = tc.assemble_layout(old_layout, {"parts": []}, floor, derived, yaw_sense="clockwise",
                             yaw_sense_source="test", moved=True, cameras=[2], out_dir="/tmp/x", today="2026-09-11")
    layout = out["layout"]
    assert out["problems"] == []
    carried = [t for t in layout["robot_tags"] if t["id"] == 77]
    assert carried and carried[0]["verified"] is False
    assert 77 in out["diff"]["carried_unverified"]
    gaps = {(g["leg"], g["joint"], g.get("mount_side")) for g in layout["unresolved_mounts"]}
    # every knee face except the carried one is a declared gap: 6 legs x 2 sides - 1
    assert len(gaps) == 11 and (2, "knee", "+y") not in gaps and (2, "knee", "-y") in gaps
    assert layout["joint_conventions"]["yaw_sign_in_body_frame"] == -1
    assert layout["leg_zero_azimuth_body_deg"]["0"] == derived["azimuth_deg"][0]
    assert all(t["verified"] for t in layout["robot_tags"] if t["id"] != 77)
    # yoke pairs land in the tag map only when both sides are known
    assert {p["id"] for p in out["tag_map"]["parts"]} == {f"leg{l}_hip_servo" for l in range(6)}


# ------------------------------------------------------------ real replay

def test_replay_of_the_2026_09_11_pass_reproduces_the_installed_layout():
    zero, sizes = tc.load_tags(FIXTURE / "zero_tags.json")
    report = json.loads((FIXTURE / "motion_report.json").read_text())
    expected = json.loads((FIXTURE / "expected_layout.json").read_text())
    # the layout that was installed when this pass ran; its horizontal tags seed the focal fit
    backup = json.loads((FIXTURE / "previous_layout.json").read_text())
    floor = json.loads((CONFIG_DIR / "floor_tag_map.json").read_text())
    floor_ids = {int(t["id"]) for t in floor["tags"]}
    link_of = tc.resolve_links(tc.votes_from_reports([report]), lambda m: None)
    derived = tc.derive_layout(tc.tags_only({c: {"tags": t} for c, t in zero.items()}), sizes, link_of, backup,
                               floor_ids, top=2, device={0: "OV9281"})
    assert "error" not in derived
    out = tc.assemble_layout(backup, {"parts": []}, floor, derived, yaw_sense="clockwise",
                             yaw_sense_source="test", moved=False, cameras=[1, 2], out_dir="x", today="2026-09-11")
    assert out["problems"] == []
    got = {t["id"]: t for t in out["layout"]["robot_tags"]}
    want = {t["id"]: t for t in expected["robot_tags"]}
    assert set(got) == set(want)
    for tid, t in want.items():
        assert got[tid]["frame"] == t["frame"], tid
        assert got[tid].get("mount_side") == t.get("mount_side"), tid
        assert tc.same_rotation(got[tid]["frame_from_tag"], t["frame_from_tag"]), tid
    assert out["layout"]["leg_zero_azimuth_body_deg"] == expected["leg_zero_azimuth_body_deg"]
    # what this pass established about the robot
    assert derived["azimuth_deg"] == {0: -18.5, 1: -95.7, 2: -158.9, 3: 141.1, 4: 94.4, 5: 37.6}
    assert got[0]["frame_from_tag"]["euler_xyz_deg"][2] == 10.6
    assert len(out["layout"]["unresolved_mounts"]) == 4


def test_installed_layout_keeps_what_the_replay_established():
    """Later passes may add tags and refine azimuths, but never contradict this one."""
    installed = json.loads((CONFIG_DIR / "hexapod-1-apriltag-layout.json").read_text())
    expected = json.loads((FIXTURE / "expected_layout.json").read_text())
    got = {t["id"]: t for t in installed["robot_tags"]}
    for t in expected["robot_tags"]:
        if not t.get("verified", True):
            continue                      # carried faces may be replaced by a seen tag later
        assert t["id"] in got, t["id"]
        assert got[t["id"]]["frame"] == t["frame"] and got[t["id"]].get("mount_side") == t.get("mount_side"), t["id"]
        assert tc.same_rotation(got[t["id"]]["frame_from_tag"], t["frame_from_tag"]), t["id"]
    for leg, az in expected["leg_zero_azimuth_body_deg"].items():
        assert abs(tc.wrap_deg(installed["leg_zero_azimuth_body_deg"][leg] - az)) < 8.0, leg
    assert installed["joint_conventions"]["yaw_sign_in_body_frame"] == -1


# ------------------------------------------------------------------ units

def test_nominal_azimuths_run_clockwise_from_above():
    assert [tc.nominal_azimuth_deg(i) for i in range(6)] == [-30.0, -90.0, -150.0, 150.0, 90.0, 30.0]


def test_snap90_and_wrap():
    assert tc.snap90(88.0) == (90.0, -2.0)
    assert tc.snap90(-179.0) == (180.0, 1.0)
    assert tc.snap90(-1.0)[0] == 0.0
    assert tc.wrap_deg(190.0) == -170.0


def _sq(cx, cy, size=30.0, turn=0.0):
    h = size / 2
    pts = np.array([[-h, -h], [h, -h], [h, h], [-h, h]])
    r = math.radians(turn)
    R = np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])
    return pts @ R.T + [cx, cy]


def test_partition_leg_from_three_moves_with_antisymmetric_yaw():
    def obs(**tags):
        return {2: {"tags": tags}}
    zero = obs(a=_sq(100, 100), b=_sq(200, 100), c=_sq(300, 100), f=_sq(500, 500))
    lifted = obs(a=_sq(100, 100), b=_sq(200, 60), c=_sq(300, 20), f=_sq(500, 500))          # b, c move with the hip
    plus = obs(a=_sq(100, 100, turn=15), b=_sq(200, 60, turn=15), c=_sq(300, 20, turn=15), f=_sq(500, 500))
    minus = obs(a=_sq(100, 100, turn=-15), b=_sq(200, 60, turn=-15), c=_sq(300, 20, turn=-15), f=_sq(500, 500))
    knee = obs(a=_sq(100, 100), b=_sq(200, 60), c=_sq(300, -40), f=_sq(500, 500))          # only c moves with the knee
    part = tc.partition_leg(zero, lifted, [plus, minus], knee)
    assert part["coxa"] == ["a"] and part["femur"] == ["b"] and part["tibia"] == ["c"]
    assert part["contradictions"] == [] and part["conflicts"] == []
    assert tc.yaw_sense_from_swing(lifted, plus, ["a"], 2) == "clockwise"
    assert tc.yaw_sense_from_swing(lifted, minus, ["a"], 2) == "counterclockwise"


def test_moved_ignores_tags_that_changed_apparent_size():
    a = {1: _sq(100, 100, size=30), 2: _sq(300, 300, size=30)}
    b = {1: _sq(100, 100, size=18), 2: _sq(340, 300, size=30)}   # 1 half hidden; 2 shifted
    mv, st = tc.moved(a, b)
    assert mv == {2} and st == set()


def test_tags_round_trip_through_the_saved_format(tmp_path):
    obs = {2: {"tags": {5: _sq(10, 10)}, "image": None, "size": (1280, 720)}}
    tc.save_tags(tmp_path / "zero_tags.json", obs)
    tags, sizes = tc.load_tags(tmp_path / "zero_tags.json")
    assert sizes == {2: (1280, 720)}
    assert np.allclose(tags[2][5], _sq(10, 10))


def test_main_replay_writes_report_without_touching_configs(tmp_path):
    cfg = tmp_path / "configs"; cfg.mkdir()
    for name in ("hexapod-1-apriltag-layout.json", "hexapod_tag_map.json", "floor_tag_map.json"):
        (cfg / name).write_bytes((CONFIG_DIR / name).read_bytes())
    replay = tmp_path / "replay"; replay.mkdir()
    (replay / "zero_tags.json").write_bytes((FIXTURE / "zero_tags.json").read_bytes())
    (replay / "report.json").write_bytes((FIXTURE / "motion_report.json").read_bytes())
    out = tmp_path / "out"
    rc = tc.main(["--replay", str(replay), "--out", str(out), "--config-dir", str(cfg), "--no-claude"])
    assert rc == 0
    report = json.loads((out / "report.json").read_text())
    assert report["validation_problems"] == []
    assert report["yaw_sense"] == "clockwise"
    assert (out / "report.md").read_text().startswith("# Tag calibration")
    assert (cfg / "hexapod-1-apriltag-layout.json").read_bytes() == (CONFIG_DIR / "hexapod-1-apriltag-layout.json").read_bytes()
    layout = json.loads((out / "layout.json").read_text())
    assert validate_layout(layout, json.loads((cfg / "floor_tag_map.json").read_text()),
                           json.loads((out / "hexapod_tag_map.json").read_text())) == []
