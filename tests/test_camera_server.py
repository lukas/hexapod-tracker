import re
import subprocess

import cv2
import numpy as np

from hexapod_tracker.camera_server import (
    CameraWorker,
    CameraHTTPServer,
    INDEX_HTML,
    annotate_tags,
    decode_fourcc,
    make_tag_detector,
)
from hexapod_tracker.paths import CONFIG_DIR


def test_decode_fourcc():
    value = cv2.VideoWriter_fourcc(*"MJPG")
    assert decode_fourcc(value) == "MJPG"
    assert decode_fourcc(-1) is None


def test_detector_labels_generated_tag36h11():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = cv2.aruco.generateImageMarker(dictionary, 7, 240)
    canvas = np.full((400, 500), 255, dtype=np.uint8)
    canvas[80:320, 130:370] = marker
    frame = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    annotated, tag_ids = annotate_tags(frame, make_tag_detector(), camera_index=0)

    assert tag_ids == [7]
    assert annotated.shape == frame.shape


def test_camera_grid_pulls_annotated_frames_for_every_index():
    # The grid pulls one current frame at a time rather than subscribing to a
    # push stream, which is what keeps latency bounded over a slow link. It
    # shows the annotated preview, never the raw frame, and treats every index
    # alike.
    assert "img.src = `/preview/${index}.jpg?w=${PREVIEW_WIDTHS[step]}&t=${Date.now()}`;" in INDEX_HTML
    # Started after the card is in the document, and on every update so a
    # stopped poller is revived.
    assert "pollPreview(article.querySelector('img'), c.index);" in INDEX_HTML
    # No push-stream URLs at all; the routes still exist for local use.
    assert "/stream/" not in INDEX_HTML
    assert "raw-stream" not in INDEX_HTML
    assert "c.index >= 2" not in INDEX_HTML
    assert "transform:rotate(180deg)" not in INDEX_HTML


def test_camera_grid_requests_the_next_frame_only_after_the_last_one_decodes():
    # Chaining on load is what makes the browser the pacer: a slow link gets
    # fewer frames, each current, and no queue can form.
    assert "img.onload" in INDEX_HTML
    assert "setTimeout(tick" in INDEX_HTML
    assert "img.onerror" in INDEX_HTML
    # Loaded straight into the visible element: fetching into a second Image
    # and assigning its src depends on cache reuse a no-store response may
    # refuse, and a refusal re-fetches and blanks the picture every frame.
    assert "new Image()" not in INDEX_HTML


def test_camera_grid_drops_pollers_when_cards_go_away():
    # A poller outliving its card would keep requesting frames forever, and a
    # server restart invalidates every card at once.
    assert "server_run_id" in INDEX_HTML
    assert "polling.clear();" in INDEX_HTML
    assert "polling.delete(Number(article.dataset.index));" in INDEX_HTML



def test_native_snapshots_preserve_full_nv12_planes():
    worker = CameraWorker(0, 4, 4, 30.0, 10.0, 82, native_avfoundation=True)
    y = np.arange(24, dtype=np.uint8).reshape(4, 6)
    uv = np.arange(12, dtype=np.uint8).reshape(2, 3, 2)
    worker._native_planes = (y, uv)

    png, width, height = worker.native_luma_snapshot()
    decoded = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
    nv12, nv12_width, nv12_height = worker.native_nv12_snapshot()

    assert (width, height) == (6, 4)
    assert np.array_equal(decoded, y)
    assert (nv12_width, nv12_height) == (6, 4)
    assert nv12 == y.tobytes() + uv.tobytes()


def test_camera_grid_has_combined_pose_tab():
    assert 'data-tab="pose"' in INDEX_HTML
    assert "fetch('/api/pose-state'" in INDEX_HTML
    assert 'id="joint-rows"' in INDEX_HTML
    assert 'id="camera-joint-rows"' in INDEX_HTML
    assert 'id="imu-pose"' in INDEX_HTML
    assert "camera.camera_joint_pose" in INDEX_HTML
    assert "cameraDegrees(yaw)" in INDEX_HTML
    assert "cameraDegrees(hip)" in INDEX_HTML


def test_camera_grid_javascript_parses():
    script = re.search(r"<script>(.*)</script>", INDEX_HTML, re.DOTALL)
    assert script is not None
    result = subprocess.run(
        ["node", "--check", "-"],
        input=script.group(1),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_combined_pose_status_exposes_read_only_motors_and_calibrated_imu():
    class Feedback:
        def sample(self):
            return (
                {"L0_yaw": 12.5, "L0_hip": -4.0},
                {
                    "configured": True,
                    "ok": True,
                    "endpoint": "http://robot/api/feedback",
                    "sample_time_unix": 1.0,
                    "live_joint_count": 2,
                    "body_frame_calibrated": True,
                    "body_roll_deg": 1.25,
                    "body_pitch_deg": -2.5,
                    "rear_pose_pitch_reference_deg": -25.9,
                    "roll_deg": 3.0,
                    "pitch_deg": 4.0,
                    "gyro_dps": [0.1, 0.2, 0.3],
                },
            )

    server = object.__new__(CameraHTTPServer)
    server.workers = []
    server.pose_estimator = None
    server.feedback_client = Feedback()

    state = server.combined_pose_status()

    assert state["read_only"] is True
    assert state["motor_feedback"]["joint_frame"] == "robot_abs"
    assert state["motor_feedback"]["joint_contract"] == "robot_abs_tibia_v2"
    assert state["motor_feedback"]["joints"][0]["degrees"] == 12.5
    assert state["motor_feedback"]["joints"][2]["degrees"] is None
    assert state["imu"]["body_frame_calibrated"] is True
    assert state["imu"]["body_roll_deg"] == 1.25
    assert state["imu"]["body_pitch_deg"] == -2.5
    assert state["imu"]["rear_pose_pitch_reference_deg"] == -25.9
    assert "body_pitch_target_deg" not in state["imu"]
    assert "body pitch target" not in INDEX_HTML


def test_adaptive_thresholds_straddle_a_round_trip():
    # Through the relay a trivial request costs ~300 ms, so both thresholds
    # must sit above that or a viewer can never climb back to a larger frame.
    assert "const FRAME_MS_TOO_SLOW = 700;" in INDEX_HTML
    assert "const FRAME_MS_HAS_HEADROOM = 350;" in INDEX_HTML
    assert "median > FRAME_MS_TOO_SLOW" in INDEX_HTML
    assert "median < FRAME_MS_HAS_HEADROOM" in INDEX_HTML


def test_status_payload_carries_the_server_run_id():
    # Removing an unrelated constant once took SERVER_RUN_ID with it, and every
    # unit test still passed because none of them requests /status.json. The
    # page polls it every second, so the server 500ed on a live browser only.
    from hexapod_tracker.camera_server import SERVER_RUN_ID

    assert isinstance(SERVER_RUN_ID, str) and len(SERVER_RUN_ID) == 32
    assert '"server_run_id": SERVER_RUN_ID,' in \
        __import__("pathlib").Path(
            __import__("hexapod_tracker.camera_server", fromlist=["__file__"]).__file__
        ).read_text()


def test_detections_snapshot_serves_corners_in_snapshot_coordinates():
    worker = CameraWorker(0, 1280, 720, 30.0, 10.0, 82)
    worker.status.reported_width, worker.status.reported_height = 1280, 720
    worker.status.detect_width, worker.status.detect_height = 1920, 1080
    corners = np.array([[10.0, 10.0], [40.0, 10.0], [40.0, 40.0], [10.0, 40.0]])
    worker._publish(b"raw", b"jpeg", [7], {7: corners})
    worker._detect_seq = 3

    snap = worker.detections_snapshot()

    assert snap["index"] == 0
    assert (snap["width"], snap["height"]) == (1280, 720)
    assert (snap["detect_width"], snap["detect_height"]) == (1920, 1080)
    assert snap["detect_seq"] == 3
    assert snap["tags"] == {"7": corners.tolist()}
    assert snap["frame_age_s"] is not None and snap["captured_unix"] is not None


def test_duplicate_tag_ids_in_one_frame_are_dropped_and_reported():
    from hexapod_tracker.camera_server import detect_tag_corners_with_duplicates, make_tag_detector

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    image = np.full((300, 520, 3), 255, dtype=np.uint8)
    for tag_id, x in ((7, 40), (7, 200), (9, 360)):       # two copies of 7, one of 9
        marker = cv2.aruco.generateImageMarker(dictionary, tag_id, 120)
        image[90:210, x:x + 120] = marker[:, :, None]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    corners, duplicates = detect_tag_corners_with_duplicates(gray, make_tag_detector())

    assert duplicates == [7]
    assert sorted(corners) == [9]


def _fake_worker(index, stable_id=None, device_name=None, capture_size=None):
    from types import SimpleNamespace

    return SimpleNamespace(
        index=index,
        stable_id=stable_id,
        capture_size=capture_size,
        status=SimpleNamespace(device_name=device_name),
    )


def test_intrinsics_resolve_by_camera_identity_not_by_slot_number():
    from hexapod_tracker.camera_server import resolve_intrinsics_by_identity

    workers = [
        _fake_worker(0, "0x830000032e40362", "12MP AF Camera"),
        _fake_worker(1, "0x520000032e46678", "4K U3 Camera "),
        _fake_worker(3, "0x41100000c456366", "Arducam OV9281 USB Camera"),
    ]
    calibration = {
        "quality": "provisional",
        "cameras": {
            # Wrong numeric key on purpose: identity must win over the key.
            "7": {"stable_id": "0x520000032e46678", "camera_matrix": [[4280, 0, 1920], [0, 4280, 1080], [0, 0, 1]]},
            # Stable id from a port the camera no longer occupies; unique name rescues it.
            "ov": {"stable_id": "0x84000000c456366", "device_name": "Arducam OV9281 USB Camera", "camera_matrix": [[948, 0, 640], [0, 948, 360], [0, 0, 1]]},
            # Identity that matches nothing must be dropped, not fall back to key "0".
            "0": {"stable_id": "0x412000032e40362", "camera_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
            # Legacy entry with no identity keeps its numeric key.
            "5": {"camera_matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 1]]},
        },
    }
    messages = []
    resolved = resolve_intrinsics_by_identity(calibration, workers, log=messages.append)

    assert set(resolved["cameras"]) == {"1", "3", "5"}
    assert resolved["cameras"]["1"]["camera_matrix"][0][0] == 4280
    assert resolved["cameras"]["3"]["camera_matrix"][0][0] == 948
    assert resolved["cameras"]["5"]["camera_matrix"][0][0] == 2
    assert resolved["quality"] == "provisional"
    assert any("re-pin" in message for message in messages)
    assert any("dropped" in message for message in messages)
    # Every key is a slot number, which is what PlanarPoseEstimator requires.
    assert all(key.isdigit() for key in resolved["cameras"])


def test_intrinsics_ambiguous_device_name_is_dropped():
    from hexapod_tracker.camera_server import resolve_intrinsics_by_identity

    workers = [
        _fake_worker(0, "0x830000032e40362", "12MP AF Camera"),
        _fake_worker(1, "0x412000032e40362", "12MP AF Camera"),
    ]
    calibration = {"cameras": {"a": {"device_name": "12MP AF Camera", "camera_matrix": []}}}
    resolved = resolve_intrinsics_by_identity(calibration, workers, log=lambda _m: None)

    assert resolved["cameras"] == {}


def test_intrinsics_entry_supplies_default_capture_size_only_when_unset():
    from hexapod_tracker.camera_server import apply_intrinsics_capture_sizes

    pinned_by_flag = _fake_worker(0, "a", "cam a", capture_size=(1280, 720))
    unset = _fake_worker(1, "b", "cam b")
    calibration = {
        "cameras": {
            "0": {"capture_size": [3840, 2160]},
            "1": {"capture_size": [3840, 2160]},
        }
    }
    apply_intrinsics_capture_sizes(calibration, [pinned_by_flag, unset], log=lambda _m: None)

    assert pinned_by_flag.capture_size == (1280, 720)
    assert unset.capture_size == (3840, 2160)


def test_committed_lab_intrinsics_are_identity_keyed():
    import json

    document = json.loads((CONFIG_DIR / "camera_intrinsics_lab_20260912.json").read_text())
    for key, spec in document["cameras"].items():
        assert not key.isdigit(), "entries must not rely on slot numbers"
        assert spec.get("stable_id") and spec.get("device_name")
        assert len(spec["camera_matrix"]) == 3
        assert spec["image_size"]["width"] > 0


def test_rig_change_reason_names_missing_new_or_silent_cameras():
    from hexapod_tracker.camera_server import rig_change_reason

    assert rig_change_reason({"a", "b"}, {"a", "b"}) is None
    assert "no longer attached" in rig_change_reason({"a", "b"}, {"a"})
    assert "new camera" in rig_change_reason({"a"}, {"a", "c"})
    assert "slot 2 has delivered no frame for 90 s" in rig_change_reason({"a"}, {"a"}, [(2, 90.0)])
    # A camera that left outranks one that arrived; both are reported by a relaunch anyway.
    assert "no longer attached" in rig_change_reason({"a"}, {"c"})


class _FakeServer:
    def __init__(self, leases=False):
        self._leases = leases

    def has_leases(self):
        return self._leases


def _watch(monkeypatch, *, discovered, leases=False, armed=False, pinned=("a",)):
    from types import SimpleNamespace

    from hexapod_tracker.camera_server import TopologyWatch

    workers = [
        SimpleNamespace(index=i, stable_id=s, status=SimpleNamespace(state="streaming", frames=10),
                        frame_age_s=lambda: 0.1)
        for i, s in enumerate(pinned)
    ]
    exits = []
    watch = TopologyWatch(_FakeServer(leases), workers, exclude=[], calibration_path=None, robot_url="http://robot",
                          on_change=exits.append, interval_s=0.01, settle_s=0.03, grace_s=0.0,
                          unreachable_s=0.05, log=lambda _m: None)
    monkeypatch.setattr(watch, "discover", lambda: set(discovered))
    monkeypatch.setattr(watch, "robot_armed", lambda: armed)
    return watch, exits


def test_topology_watch_exits_75_after_the_change_settles(monkeypatch):
    import time

    watch, exits = _watch(monkeypatch, discovered={"a", "b"})
    watch.start()
    watch.join(timeout=2.0)
    assert exits == [75]


def test_topology_watch_holds_while_a_lease_or_an_armed_robot_says_so(monkeypatch):
    import time

    for kwargs in ({"leases": True}, {"armed": True}):
        watch, exits = _watch(monkeypatch, discovered={"a", "b"}, **kwargs)
        watch.start()
        time.sleep(0.3)
        watch.stop()
        watch.join(timeout=1.0)
        assert exits == [], kwargs


def test_topology_watch_waits_out_an_unreachable_robot_then_proceeds(monkeypatch):
    watch, exits = _watch(monkeypatch, discovered={"a", "b"}, armed=None)
    watch.start()
    watch.join(timeout=2.0)
    assert exits == [75]


def test_topology_watch_does_nothing_while_the_rig_matches(monkeypatch):
    import time

    watch, exits = _watch(monkeypatch, discovered={"a"})
    watch.start()
    time.sleep(0.2)
    watch.stop()
    watch.join(timeout=1.0)
    assert exits == []
