"""hexapod-cameras: the registry, a per-run rig on a synthetic floor scene, and the session writer.

No camera hardware and no ffmpeg: frames come from a fake capture that renders the
surveyed floor tags (and the chassis tag) through a known homography, and the video
recorder is replaced by one that counts frames.
"""
from __future__ import annotations

import json
import signal
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
    assert rec.closed and rec.frames >= summary["states"] and summary["video"]["top"]["frames"] == rec.frames
    assert (out / "top.mp4").exists()
    session = json.loads((out / "session.json").read_text())
    assert session["stopped"] == "seconds" and session["cameras"][0]["role"] == "top"
    assert session["status"] == "completed" and session["errors"] == []
    assert session["video"]["top"]["finalized"]
    assert captures and captures[0].released


@pytest.mark.parametrize("failure_stage", ["write", "capture", "state", "finalize", "release"])
def test_session_failure_finalizes_other_videos_releases_cameras_and_records_end(
        registry, tmp_path, monkeypatch, failure_stage):
    doc = cameras.load_registry(registry)
    cameras.assign(doc, "0xbbb", device_name="Side camera", role="side")
    scene = SyntheticScene()
    captures = []
    recorders = []
    clock = {"t": 1000.0}
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}

    class FailingCapture(FakeCapture):
        def __init__(self, slot):
            super().__init__(scene)
            self.slot, self.reads = slot, 0
            captures.append(self)

        def read(self):
            self.reads += 1
            if failure_stage == "capture" and self.slot == 0 and self.reads == 2:
                raise RuntimeError("capture failed")
            return super().read()

        def release(self):
            super().release()
            if failure_stage == "release" and self.slot == 0:
                raise RuntimeError("release failed")

    class FailingRecorder(CountingRecorder):
        def __init__(self, *args):
            super().__init__(*args)
            recorders.append(self)

        def write(self, frame):
            if failure_stage == "write" and self.path.stem == "top" and self.frames == 1:
                raise BrokenPipeError("encoder write failed")
            super().write(frame)

        def close(self):
            super().close()
            if failure_stage in ("write", "finalize") and self.path.stem == "top":
                raise RuntimeError("finalize failed")

    original_state = cameras.Rig.state

    def state(rig, observations):
        if failure_stage == "state" and rig.state_seq == 1:
            raise RuntimeError("state failed")
        return original_state(rig, observations)

    monkeypatch.setattr(cameras.Rig, "state", state)
    out = tmp_path / "failure"
    with pytest.raises((RuntimeError, BrokenPipeError), match=f"{failure_stage} failed"):
        cameras.run_session(doc, out, roles=["top", "side"], seconds=0.5,
                            capture_factory=lambda slot, *args: FailingCapture(slot),
                            recorder_factory=FailingRecorder, clock=lambda: clock["t"],
                            sleep=lambda s: clock.__setitem__("t", clock["t"] + max(s, 0.05)), log=lambda m: None)

    assert len(recorders) == len(captures) == 2
    assert all(rec.closed for rec in recorders) and all(cap.released for cap in captures)
    session = json.loads((out / "session.json").read_text())
    assert session["status"] == "failed" and session["ended_unix"] > session["started_unix"]
    assert any(f"{failure_stage} failed" in error for error in session["errors"])
    assert session["video"]["side"]["finalized"]
    assert session["video"]["top"]["finalized"] == (failure_stage not in ("write", "finalize"))
    assert session["stopped"] == ("seconds" if failure_stage in ("finalize", "release") else "error")
    assert all(signal.getsignal(sig) == handler for sig, handler in handlers.items())


def test_session_failed_camera_open_releases_partial_rig_and_records_failure(registry, tmp_path):
    doc = cameras.load_registry(registry)
    cameras.assign(doc, "0xbbb", device_name="Side camera", role="side")
    cap = FakeCapture(SyntheticScene())

    def factory(slot, *args):
        if slot == 1:
            raise RuntimeError("camera unavailable")
        return cap

    out = tmp_path / "failed-open"
    with pytest.raises(RuntimeError, match="camera unavailable"):
        cameras.run_session(doc, out, roles=["top", "side"], capture_factory=factory, log=lambda m: None)
    session = json.loads((out / "session.json").read_text())
    assert cap.released and session["status"] == "failed" and session["ended_unix"]
    assert session["video"] == {} and "camera unavailable" in session["errors"][0]


@pytest.mark.parametrize("fail_finish", [False, True])
def test_native_recording_does_not_use_analysis_frames_or_software_recorder(registry, tmp_path, fail_finish):
    doc = cameras.load_registry(registry)
    cameras.assign(doc, "0xbbb", device_name="Side camera", role="side", rotate_180=True)
    scene = SyntheticScene()
    captures = []
    clock = {"t": 1000.0}

    class NativeCapture(FakeCapture):
        def __init__(self):
            super().__init__(scene)
            self.movie = None
            self.started = False
            self.stopped = False
            captures.append(self)

        def prepare_recording(self, path, *, rotate_180):
            self.movie, self.rotation = path, rotate_180

        def start_recording(self):
            assert all(cap.movie is not None for cap in captures)
            self.started = True
            self.movie.write_bytes(b"native movie")

        def read(self):
            assert all(cap.started for cap in captures)
            return super().read()

        def stop_recording(self):
            self.stopped = True
            if fail_finish and self.movie.stem == "top":
                raise RuntimeError("native disk write failed")
            return {"path": str(self.movie), "fps": 30.0, "duration_s": 0.5, "finalized": True}

    out = tmp_path / "native"

    def run():
        return cameras.run_session(doc, out, roles=["top", "side"], seconds=0.5, video_fps=5.0,
                                   capture_factory=lambda *a: NativeCapture(),
                                   recorder_factory=lambda *a: pytest.fail("native video entered software encoder"),
                                   clock=lambda: clock["t"],
                                   sleep=lambda s: clock.__setitem__("t", clock["t"] + max(s, 0.05)),
                                   log=lambda m: None)

    if fail_finish:
        with pytest.raises(RuntimeError, match="native disk write failed"):
            run()
    else:
        run()
    session = json.loads((out / "session.json").read_text())
    assert all(cap.stopped and cap.released for cap in captures)
    assert [cap.rotation for cap in captures] == [False, True]
    assert session["status"] == ("failed" if fail_finish else "completed")
    assert session["video"]["side"]["fps"] == 30 and session["video"]["side"]["finalized"]
    assert "frames" not in session["video"]["side"]  # analysis reads do not count native video frames
    assert (out / "top.mov").exists() and not (out / "top.mp4").exists()
    assert not (out / "top_timestamps.csv").exists()


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
