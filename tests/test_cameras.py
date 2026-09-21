"""hexapod-cameras: the registry, a per-run rig on a synthetic floor scene, and the session writer.

No camera hardware and no ffmpeg: frames come from a fake capture that renders the
surveyed floor tags (and the chassis tag) through a known homography, and the video
recorder is replaced by one that counts frames.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from hexapod_tracker import cameras
from hexapod_tracker.paths import CONFIG_DIR

FLOOR = json.loads((CONFIG_DIR / "floor_tag_map.json").read_text())
TAG_MM = float(FLOOR["tag_black_square_size"])
SID = "0x520000032e46678"


def _anchor_corners(tag: dict) -> np.ndarray:
    cx, cy = float(tag["center"][0]), float(tag["center"][1])
    yaw = np.radians(float(tag["yaw_degrees"]))
    h = TAG_MM / 2.0
    local = np.array([[-h, h], [h, h], [h, -h], [-h, -h]])           # corner0->1 is +x, 3->0 is +y
    rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return local @ rot.T + np.array([cx, cy])


class SyntheticScene:
    """Floor mm -> image px through a fixed homography; renders tags into a 1280x720 frame."""

    def __init__(self, width=1280, height=720, scale_px_per_mm=1.0, chassis=None):
        self.width, self.height = width, height
        # A camera looking down at a right-handed floor frame: floor +y points up the picture,
        # so the floor->image map flips y (real floor homographies have a negative determinant).
        # Slightly rotated, a touch of perspective, and the anchors' centre (305, 305 mm) at the middle.
        a = np.radians(4.0)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ np.diag([scale_px_per_mm, -scale_px_per_mm])
        centre_mm = np.array([304.8, 304.8])
        offset = np.array([width / 2.0, height / 2.0]) - R @ centre_mm
        S = np.array([[R[0, 0], R[0, 1], offset[0]], [R[1, 0], R[1, 1], offset[1]], [0.0, 0.0, 1.0]])
        S[2, 0] = 1.0e-5                                               # perspective
        self.H = S / S[2, 2]
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self.chassis = chassis                                         # (x_mm, y_mm, yaw_deg) or None
        self.hidden: set[int] = set()

    def project(self, pts_mm: np.ndarray) -> np.ndarray:
        return cv2.perspectiveTransform(pts_mm.reshape(-1, 1, 2).astype(np.float64), self.H).reshape(-1, 2)

    def render(self) -> np.ndarray:
        img = np.full((self.height, self.width, 3), 200, dtype=np.uint8)
        tags = [(int(t["id"]), _anchor_corners(t)) for t in FLOOR["tags"] if "yaw_degrees" in t and int(t["id"]) not in self.hidden]
        if self.chassis is not None:
            x, y, yaw = self.chassis
            tags.append((0, _anchor_corners({"center": [x, y], "yaw_degrees": yaw})))
        for tid, world in tags:
            marker = cv2.aruco.generateImageMarker(self.dictionary, tid, 160, borderBits=1)
            # marker image corners in the same order as the tag's world corners (0 top-left .. 3 bottom-left)
            src = np.array([[0, 0], [159, 0], [159, 159], [0, 159]], dtype=np.float32)
            dst = self.project(world).astype(np.float32)
            M = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(marker, M, (self.width, self.height), flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=255)
            mask = cv2.warpPerspective(np.full((160, 160), 255, np.uint8), M, (self.width, self.height))
            img[mask > 0] = np.repeat(warped[mask > 0][:, None], 3, axis=1)
        return img


class FakeCapture:
    def __init__(self, scene: SyntheticScene):
        self.scene = scene
        self.released = False
        self.last_error = None
        self.detection_gray = None                                     # no native luma: detect on the frame

    def isOpened(self):
        return not self.released

    def read(self):
        return True, self.scene.render()

    def release(self):
        self.released = True

    def capture_info(self):
        return {"backend": "fake", "capture_image_size_px": [self.scene.width, self.scene.height]}


class CountingRecorder:
    made: list["CountingRecorder"] = []

    def __init__(self, path, fps, size):
        self.path, self.fps, self.size, self.frames, self.closed = Path(path), fps, size, 0, False
        CountingRecorder.made.append(self)

    def write(self, frame):
        assert (frame.width, frame.height) == self.size
        self.frames += 1

    def close(self):
        self.closed = True
        self.path.write_bytes(b"fake mp4")


@pytest.fixture
def registry(tmp_path):
    doc = cameras.empty_registry()
    cameras.assign(doc, SID, device_name="4K U3 Camera", role="top")
    path = tmp_path / "cameras.json"
    cameras.save_registry(doc, path)
    return path


def _rig(doc, scene, **kw):
    return cameras.Rig(doc, ["top"], capture_factory=lambda slot, sid, entry: FakeCapture(scene), log=lambda m: None, **kw)


# ------------------------------------------------------------------ registry


def test_registry_round_trip_roles_and_resolution(tmp_path):
    path = tmp_path / "r.json"
    doc = cameras.load_registry(path)
    assert doc["cameras"] == {}
    cameras.assign(doc, "0xaaa", device_name="Cam A", role="top", capture_size=(1920, 1080), fps=30)
    cameras.assign(doc, "0xbbb", device_name="Cam B", role="side")
    cameras.save_registry(doc, path)
    again = cameras.load_registry(path)
    assert again["cameras"]["0xaaa"]["capture_size"] == [1920, 1080]
    assert cameras.entries_by_role(again) == {"top": "0xaaa", "side": "0xbbb"}
    assert cameras.resolve(again, "top") == "0xaaa" and cameras.resolve(again, "Cam B") == "0xbbb"
    assert cameras.resolve(again, "0xbbb") == "0xbbb"
    with pytest.raises(KeyError):
        cameras.resolve(again, "nope")


def test_a_role_names_one_camera_and_adopt_moves_calibration(tmp_path):
    doc = cameras.empty_registry()
    cameras.assign(doc, "0xaaa", device_name="Cam A", role="top")
    doc["cameras"]["0xaaa"]["floor"] = {"homography": np.eye(3).tolist(), "image_size": [1280, 720], "anchor_ids": [100]}
    cameras.assign(doc, "0xbbb", device_name="Cam B", role="top")
    assert doc["cameras"]["0xaaa"]["role"] is None and doc["cameras"]["0xbbb"]["role"] == "top"
    cameras.adopt(doc, "0xaaa", "0xccc")
    assert "0xaaa" not in doc["cameras"] and doc["cameras"]["0xccc"]["floor"]["anchor_ids"] == [100]
    assert doc["cameras"]["0xccc"]["adopted_from"] == "0xaaa"
    with pytest.raises(KeyError):
        cameras.adopt(doc, "0xccc", "0xbbb")


def test_import_intrinsics_keeps_identity_keys():
    doc = cameras.empty_registry()
    intr = json.loads((CONFIG_DIR / "camera_intrinsics_lab_20260912.json").read_text())
    doc, imported = cameras.import_intrinsics(doc, intr)
    assert imported == [SID]
    entry = doc["cameras"][SID]
    assert entry["device_name"] == "4K U3 Camera" and entry["intrinsics"]["camera_matrix"][0][0] > 1000
    assert "stable_id" not in entry["intrinsics"]


# ------------------------------------------------------------------ rig on a synthetic floor


def test_rig_opens_by_role_detects_and_releases(registry):
    doc = cameras.load_registry(registry)
    scene = SyntheticScene(chassis=(150.0, 120.0, 30.0))
    rig = _rig(doc, scene)
    with rig:
        obs = rig.observe()
        assert [o.role for o in obs] == ["top"] and obs[0].frame is not None
        assert 0 in obs[0].tags and set(FLOOR["active_anchor_ids"]) <= set(obs[0].tags)
        det = rig.detections_doc(obs)
        cam = det["cameras"][0]
        assert cam["index"] == 0 and cam["role"] == "top" and cam["width"] == 1280 and cam["height"] == 720
        assert np.asarray(cam["tags"]["0"]).shape == (4, 2) and det["roles"] == {"top": 0}
        poses = rig.poses_doc(obs)
        marker = poses["markers"]["0"]
        assert marker["status"] == "tracked"
        assert abs(marker["position_mm"]["x"] - 150.0) < 6 and abs(marker["position_mm"]["y"] - 120.0) < 6
        assert abs(marker["rotation_degrees"]["yaw"] - 30.0) < 2.0
    assert rig.cameras[0].capture is None                              # released
    assert not rig.opened


def test_rig_refuses_an_unassigned_role(registry):
    doc = cameras.load_registry(registry)
    with pytest.raises(KeyError):
        cameras.Rig(doc, ["side"], capture_factory=lambda *a: FakeCapture(SyntheticScene()))


def test_floor_fit_is_saved_with_quality_and_used_when_anchors_are_hidden(registry):
    doc = cameras.load_registry(registry)
    scene = SyntheticScene(chassis=(-100.0, 200.0, -45.0))
    with _rig(doc, scene) as rig:
        saved, report = rig.fit_floor("top", frames=5, interval_s=0.0, sleep=lambda s: None)
    assert saved["quality"] == "good" and saved["reprojection_rms_px"] < 1.5
    assert sorted(saved["anchor_ids"]) == sorted(FLOOR["active_anchor_ids"]) and saved["image_size"] == [1280, 720]
    assert report["robot_tags_seen"] == [0] and saved["floor_map_digest"]
    H = np.asarray(saved["homography"])
    assert np.allclose(H / H[2, 2], scene.H / scene.H[2, 2], atol=2e-3, rtol=2e-2)
    doc["cameras"][SID]["floor"] = saved
    # now every floor tag is covered: the saved fit must still place the chassis tag
    scene.hidden = set(FLOOR["active_anchor_ids"])
    with _rig(doc, scene) as rig:
        obs = rig.observe()
        assert not any(t in obs[0].tags for t in FLOOR["active_anchor_ids"])
        poses = rig.poses_doc(obs)
    marker = poses["markers"]["0"]
    assert marker["status"] == "tracked" and abs(marker["position_mm"]["x"] + 100.0) < 6
    assert poses["calibration"]["cameras"]["0"]["status"] == "held"


def test_floor_check_passes_on_the_fitted_scene_and_fails_after_the_camera_is_bumped(registry):
    doc = cameras.load_registry(registry)
    scene = SyntheticScene()
    with _rig(doc, scene) as rig:
        saved, _ = rig.fit_floor("top", frames=3, interval_s=0.0, sleep=lambda s: None)
    doc["cameras"][SID]["floor"] = saved
    with _rig(doc, scene) as rig:
        ok = rig.check_floor("top", frames=2, sleep=lambda s: None)
    assert ok["ok"] and ok["drift_rms_px"] < 1.5
    bumped = SyntheticScene()
    bumped.H = bumped.H @ np.array([[1, 0, 40.0], [0, 1, 0], [0, 0, 1]])   # 40 mm shift in the floor: ~18 px
    with _rig(doc, bumped) as rig:
        bad = rig.check_floor("top", frames=2, sleep=lambda s: None)
    assert not bad["ok"] and bad["drift_rms_px"] > 6 and "bumped" in bad["reason"]


def test_fit_refuses_with_too_few_anchors(registry):
    doc = cameras.load_registry(registry)
    scene = SyntheticScene()
    scene.hidden = set(FLOOR["active_anchor_ids"]) - {100, 103}
    with _rig(doc, scene) as rig, pytest.raises(RuntimeError, match="need 3"):
        rig.fit_floor("top", frames=2, interval_s=0.0, sleep=lambda s: None)


# ------------------------------------------------------------------ session


def test_session_writes_state_latest_jsonl_and_video_then_releases(registry, tmp_path, monkeypatch):
    doc = cameras.load_registry(registry)
    scene = SyntheticScene(chassis=(150.0, 150.0, 10.0))
    CountingRecorder.made.clear()
    clock = {"t": 1000.0}
    captures = []

    def factory(slot, sid, entry):
        cap = FakeCapture(scene)
        captures.append(cap)
        return cap

    out = tmp_path / "run"
    summary = cameras.run_session(doc, out, roles=["top"], hz=5.0, video=True, video_fps=10.0, seconds=1.0,
                                  capture_factory=factory, recorder_factory=CountingRecorder,
                                  clock=lambda: clock["t"], sleep=lambda s: clock.__setitem__("t", clock["t"] + max(s, 0.05)),
                                  log=lambda m: None)
    assert summary["stopped"] == "seconds" and summary["states"] >= 4
    state = json.loads((out / "state.json").read_text())
    assert state["performance"]["frame_sequence"] == state["seq"] == summary["states"]
    assert state["cameras"][0]["role"] == "top" and "0" in state["cameras"][0]["tags"]
    assert state["poses"]["markers"]["0"]["status"] == "tracked"
    assert (out / "latest_top.jpg").stat().st_size > 1000
    lines = [json.loads(l) for l in (out / "vision.jsonl").read_text().splitlines()]
    assert [l["seq"] for l in lines] == list(range(1, summary["states"] + 1))
    assert lines[-1]["markers"]["0"]["status"] == "tracked" and lines[-1]["tags"]["top"][0] == 0
    rec = CountingRecorder.made[0]
    # video is gated to video_fps (10) and capped by the state rate (5): one frame per state here
    assert rec.closed and 0.8 * min(5.0, 10.0) * 1.0 <= rec.frames <= summary["states"]
    assert summary["video"]["top"]["frames"] == rec.frames and summary["primary"] == "top"
    assert (out / "top.mp4").exists()
    session = json.loads((out / "session.json").read_text())
    assert session["stopped"] == "seconds" and session["cameras"][0]["role"] == "top"
    assert captures and captures[0].released


def test_session_stops_on_stop_file_and_on_closed_stdin(registry, tmp_path):
    doc = cameras.load_registry(registry)
    scene = SyntheticScene()
    clock = {"t": 0.0}

    def tick(s):
        clock["t"] += max(s, 0.05)
        if clock["t"] > 0.6:
            (tmp_path / "a" / "STOP").write_text("")

    summary = cameras.run_session(doc, tmp_path / "a", roles=["top"], hz=5.0, video=False, seconds=None,
                                  capture_factory=lambda *a: FakeCapture(scene), clock=lambda: clock["t"], sleep=tick,
                                  log=lambda m: None)
    assert summary["stopped"] == "STOP file" and summary["video"] == {}

    import io, os
    r, w = os.pipe()
    os.close(w)                                                        # parent gone: EOF on our end
    stdin = os.fdopen(r, "r")
    summary = cameras.run_session(doc, tmp_path / "b", roles=["top"], hz=5.0, video=False, seconds=None, stdin=stdin,
                                  capture_factory=lambda *a: FakeCapture(scene), clock=lambda: clock["t"],
                                  sleep=lambda s: clock.__setitem__("t", clock["t"] + 0.05), log=lambda m: None)
    assert summary["stopped"] == "stdin closed"


def test_cli_assign_show_and_import_work_offline(registry, capsys):
    rc = cameras.main(["--registry", str(registry), "assign", "top", "--role", "top", "--capture-size", "1920x1080",
                       "--offline", "--notes", "overhead"])
    assert rc == 0
    doc = cameras.load_registry(registry)
    assert doc["cameras"][SID]["capture_size"] == [1920, 1080] and doc["cameras"][SID]["notes"] == "overhead"
    rc = cameras.main(["--registry", str(registry), "import-intrinsics", str(CONFIG_DIR / "camera_intrinsics_lab_20260912.json")])
    assert rc == 0 and cameras.load_registry(registry)["cameras"][SID]["intrinsics"]["quality"] == "provisional"
    assert cameras.main(["--registry", str(registry), "show"]) == 0
    assert SID in capsys.readouterr().out


def test_a_saved_fit_beats_a_one_anchor_refit(registry):
    doc = cameras.load_registry(registry)
    scene = SyntheticScene(chassis=(-100.0, 200.0, -45.0))
    with _rig(doc, scene) as rig:
        saved, _ = rig.fit_floor("top", frames=3, interval_s=0.0, sleep=lambda s: None)
    doc["cameras"][SID]["floor"] = saved
    # only anchor 103 stays visible, and the camera has been bumped so a 1-tag refit would be wrong
    scene.hidden = set(FLOOR["active_anchor_ids"]) - {103}
    with _rig(doc, scene) as rig:
        obs = rig.observe()
        assert 103 in obs[0].tags
        snaps = rig.pose_snapshots(obs)
        assert 103 not in snaps[0]["tags"] and 0 in snaps[0]["tags"]      # anchor hidden from the estimator, chassis kept
        poses = rig.poses_doc(obs)
    assert poses["calibration"]["cameras"]["0"]["status"] == "held"
    m = poses["markers"]["0"]
    assert m["status"] == "tracked" and abs(m["position_mm"]["x"] + 100.0) < 6


# --------------------------------------------------------------------------- UVC controls (focus) at open

class _Run:
    def __init__(self, returncode=0, stdout=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, ""


def test_uvc_location_id_is_the_top_of_the_unique_id():
    assert cameras.uvc_location_id("0x110000032e40362") == "0x01100000"
    assert cameras.uvc_location_id("0x840000032e40362") == "0x08400000"
    assert cameras.uvc_location_id("96E41DC6-06DE-483D-9130-ABB500000001") is None     # an iPhone, not UVC


def test_apply_uvc_controls_reads_first_sets_only_what_differs_autofocus_first(tmp_path):
    tool = tmp_path / "uvc-util"
    tool.write_text("")
    calls, slept = [], []
    state = {"auto-focus": "true", "focus-abs": "512", "sharpness": "32"}

    def runner(argv, **_kw):
        calls.append(argv[1:])
        name = argv[-1]
        if argv[-2] == "-o":
            return _Run(0, state[name] + "\n")
        key, val = name.split("=", 1)
        state[key] = val
        return _Run(0)

    changed = cameras.apply_uvc_controls("0x110000032e40362", {"focus-abs": 496, "auto-focus": False, "sharpness": 32},
                                         runner=runner, util=tool, log=lambda m: None, sleep=slept.append)
    assert changed == {"auto-focus": "false", "focus-abs": "496"}
    sets = [c for c in calls if c[2] == "-s"]
    assert sets == [["-L", "0x01100000", "-s", "auto-focus=false"], ["-L", "0x01100000", "-s", "focus-abs=496"]]
    assert slept == [1.0]                                    # the lens moved: one settle wait
    # second pass: nothing differs, nothing set, no wait
    calls.clear(); slept.clear()
    assert cameras.apply_uvc_controls("0x110000032e40362", {"focus-abs": 496, "auto-focus": False},
                                      runner=runner, util=tool, log=lambda m: None, sleep=slept.append) == {}
    assert all(c[2] == "-o" for c in calls) and slept == []


def test_apply_uvc_controls_without_the_tool_logs_and_moves_on(tmp_path):
    logs = []
    out = cameras.apply_uvc_controls("0x110000032e40362", {"focus-abs": 496}, util=tmp_path / "missing",
                                     runner=lambda *a, **k: pytest.fail("must not run"), log=logs.append)
    assert out == {} and "not found" in logs[0]


def test_rig_applies_registry_uvc_controls_when_a_camera_opens(registry, monkeypatch):
    doc = cameras.load_registry(registry)
    doc["cameras"][SID]["uvc"] = {"auto-focus": False, "focus-abs": 496}
    applied = []
    monkeypatch.setattr(cameras, "apply_uvc_controls",
                        lambda sid, controls, **kw: applied.append((sid, controls)) or {"focus-abs": "496"})
    with cameras.Rig(doc, ["top"], capture_factory=lambda *a: FakeCapture(SyntheticScene()), log=lambda m: None) as rig:
        assert applied == [(SID, {"auto-focus": False, "focus-abs": 496})]
        assert rig.camera("top").uvc_changed == {"focus-abs": "496"}



# --------------------------------------------------------------------------- primary / auxiliary pacing

class PollableCapture(FakeCapture):
    """A capture with a non-blocking read: a new frame only every ``every`` reads, like a camera
    whose frames arrive slower than the loop polls it."""

    def __init__(self, scene, every=3):
        super().__init__(scene)
        self.every, self.polls, self.blocking_reads = every, 0, 0

    def read(self, wait=True):
        if wait:
            self.blocking_reads += 1
            return True, self.scene.render()
        self.polls += 1
        if self.polls % self.every:
            return False, None
        return True, self.scene.render()


def _two_camera_registry(tmp_path):
    doc = cameras.empty_registry()
    cameras.assign(doc, SID, device_name="4K U3 Camera", role="top")
    cameras.assign(doc, "0x413000032e46678", device_name="4K U3 Camera", role="side")
    path = tmp_path / "cameras.json"
    cameras.save_registry(doc, path)
    return doc


def test_session_paces_on_the_primary_and_detects_aux_cameras_at_aux_hz(tmp_path):
    doc = _two_camera_registry(tmp_path)
    scene = SyntheticScene(chassis=(150.0, 150.0, 10.0))
    caps = {}

    def factory(slot, sid, entry):
        caps[entry["role"]] = cap = (FakeCapture(scene) if entry["role"] == "top" else PollableCapture(scene, every=2))
        return cap

    clock = {"t": 1000.0}
    CountingRecorder.made.clear()
    summary = cameras.run_session(doc, tmp_path / "run", roles=["top", "side"], hz=20.0, aux_hz=2.0, primary="top",
                                  video=True, video_fps=10.0, seconds=2.0, background=False, capture_factory=factory,
                                  recorder_factory=CountingRecorder, clock=lambda: clock["t"],
                                  sleep=lambda s: clock.__setitem__("t", clock["t"] + max(s, 0.001)), log=lambda m: None)
    assert summary["primary"] == "top" and summary["stopped"] == "seconds"
    assert 35 <= summary["states"] <= 41                          # 20 Hz for 2 s
    assert summary["detections"]["top"] == summary["states"]     # the primary is detected every state
    assert 3 <= summary["detections"]["side"] <= 5                # the aux camera about twice a second
    assert caps["side"].blocking_reads == 0                        # never waited on the aux camera
    assert 10 <= caps["side"].polls <= summary["states"]           # polled only when a video frame / detection was due
    recs = {r.path.name: r for r in CountingRecorder.made}
    assert 15 <= recs["top.mp4"].frames <= 21                     # video gated to 10 fps
    assert recs["side.mp4"].frames <= recs["top.mp4"].frames
    lines = [json.loads(l) for l in (tmp_path / "run" / "vision.jsonl").read_text().splitlines()]
    assert all(l["tags"]["top"] for l in lines) and all(l["detect_age_s"]["top"] == 0.0 for l in lines)
    # between its detections the aux entry keeps the tags of the frame it detected on, and reports that frame's age
    ages = [l["detect_age_s"]["side"] for l in lines if l["detect_age_s"]["side"] is not None]
    assert ages and max(ages) >= 0.3 and any(l["tags"]["side"] for l in lines)
    state = json.loads((tmp_path / "run" / "state.json").read_text())
    assert state["primary"] == "top" and {c["role"] for c in state["cameras"]} == {"top", "side"}
    session = json.loads((tmp_path / "run" / "session.json").read_text())
    assert session["primary"] == "top" and session["aux_hz"] == 2.0


def test_pick_primary_prefers_the_request_then_top_then_the_first_role():
    assert cameras.pick_primary(["side", "top"]) == "top"
    assert cameras.pick_primary(["side", "top"], "side") == "side"
    assert cameras.pick_primary(["side", "side2"]) == "side"
    assert cameras.pick_primary(["side", "side2"], "ceiling") == "side"


def test_fast_detect_skips_the_upscale_pass_unless_an_id_went_missing_or_a_second_passed(monkeypatch):
    scene = SyntheticScene(chassis=(150.0, 150.0, 10.0))
    clock = {"t": 100.0}
    cam = cameras.Camera("top", 0, SID, cameras.new_entry("fake"), capture_factory=lambda *a: FakeCapture(scene), clock=lambda: clock["t"])
    cam.open()
    frame = cam.grab()
    calls = []
    real = cameras.detect_tag_corners_with_duplicates

    def spy(gray, det, *, enhance=True):
        calls.append(enhance)
        return real(gray, det, enhance=enhance)

    monkeypatch.setattr(cameras, "detect_tag_corners_with_duplicates", spy)
    assert 0 in cam.detect(frame, fast=True).tags and calls == [False, True]     # first time: the periodic full pass runs too
    calls.clear(); clock["t"] += 0.1
    assert 0 in cam.detect(frame, fast=True).tags and calls == [False]           # native found everything the full pass had: no 2x pass
    calls.clear(); clock["t"] += 0.1
    scene.hidden = {100}                                                          # an anchor vanishes from the picture
    obs = cam.detect(cam.grab(), fast=True)
    assert calls == [False, True] and 100 not in obs.tags                        # native lost an id -> full pass, which confirms it is gone
    calls.clear(); clock["t"] += 0.1
    assert cam.detect(cam.grab(), fast=True).tags and calls == [False]           # and the smaller set is now the reference
    calls.clear(); clock["t"] += 1.5
    cam.detect(cam.grab(), fast=True)
    assert calls == [False, True]                                                # a second later: the periodic full pass
    cam.release()


def test_worker_coalesces_jobs_per_key_and_drains_on_close():
    import threading
    gate = threading.Event()
    done = []
    w = cameras.Worker("t", log=lambda m: None)
    w.submit("a", gate.wait)                                     # occupies the thread
    w.submit("b", lambda: done.append("b1"))
    w.submit("b", lambda: done.append("b2"))                     # replaces b1: only the newest per key runs
    w.submit("c", lambda: 1 / 0)                                 # an exception is logged, not raised
    w.submit("d", lambda: done.append("d"))
    gate.set()
    w.close()
    assert done == ["b2", "d"] and w.ran == 4


def test_control_json_switches_the_primary_while_the_session_runs(tmp_path):
    doc = _two_camera_registry(tmp_path)
    scene = SyntheticScene(chassis=(150.0, 150.0, 10.0))
    caps = {}

    def factory(slot, sid, entry):
        caps[entry["role"]] = cap = PollableCapture(scene, every=1)   # both cameras always have a frame
        return cap

    out = tmp_path / "run"
    out.mkdir()
    clock = {"t": 1000.0}

    def tick(s):
        clock["t"] += max(s, 0.001)
        if clock["t"] >= 1001.0 and not (out / "control.json").exists():
            (out / "control.json").write_text(json.dumps({"primary": "side"}))

    summary = cameras.run_session(doc, out, roles=["top", "side"], hz=20.0, aux_hz=2.0, primary="top", video=False,
                                  seconds=2.0, background=False, capture_factory=factory, clock=lambda: clock["t"],
                                  sleep=tick, log=lambda m: None)
    assert summary["primary"] == "side" and len(summary["primary_switches"]) == 1
    sw = summary["primary_switches"][0]
    assert sw["from"] == "top" and sw["to"] == "side" and 1001.0 <= sw["t"] < 1001.2
    lines = [json.loads(l) for l in (out / "vision.jsonl").read_text().splitlines()]
    before = [l for l in lines if l["capture_unix"] < 1001.0]; after = [l for l in lines if l["capture_unix"] >= 1001.1]
    assert all(l["detect_age_s"]["top"] == 0.0 for l in before)      # top detected every state before the switch
    assert all(l["detect_age_s"]["side"] == 0.0 for l in after)      # side detected every state after it
    assert any(l["detect_age_s"]["top"] > 0.2 for l in after)        # top is now an aux camera (stale between its ticks)
    assert caps["side"].blocking_reads > 10 and caps["top"].blocking_reads > 10    # each was the paced camera for a while
    assert json.loads((out / "state.json").read_text())["primary"] == "side"
    session = json.loads((out / "session.json").read_text())
    assert session["primary"] == "side" and session["primary_switches"] == summary["primary_switches"]
