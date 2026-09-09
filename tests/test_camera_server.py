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
    load_capture_profile,
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
    assert "probe.src = `/preview/${index}.jpg?w=${PREVIEW_WIDTHS[step]}&t=${Date.now()}`;" in INDEX_HTML
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
    assert "probe.onload" in INDEX_HTML
    assert "setTimeout(tick" in INDEX_HTML
    assert "probe.onerror" in INDEX_HTML


def test_camera_grid_drops_pollers_when_cards_go_away():
    # A poller outliving its card would keep requesting frames forever, and a
    # server restart invalidates every card at once.
    assert "server_run_id" in INDEX_HTML
    assert "polling.clear();" in INDEX_HTML
    assert "polling.delete(Number(article.dataset.index));" in INDEX_HTML


def test_saved_lab_capture_profile_keeps_modes_and_calibration_together():
    profile = load_capture_profile(
        CONFIG_DIR / "camera_capture_profiles.json", "lab-tracking"
    )

    assert profile["indices"] == [0, 1, 2]
    assert profile["native_avfoundation"] == [0]
    assert profile["rotate_180"] == [1, 2]
    assert profile["camera_modes"][0] == (1280, 960, 30.0)
    assert profile["camera_modes"][1] == (1280, 800, 100.0)
    assert profile["camera_calibration"] == "camera_intrinsics.json"


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
