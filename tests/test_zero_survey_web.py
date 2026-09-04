"""Process-manager tests for resumable USB and browser-relayed Wi-Fi scans."""
from __future__ import annotations

import base64
import json
import signal
import threading

from hexapod_tracker.paths import CONFIG_DIR
from hexapod_tracker.zero_survey_web import ZeroPoseSurveyManager


CONFIG_PATH = CONFIG_DIR / "apriltag_pose_config_20260831.json"


class _RunningProcess:
    def __init__(self) -> None:
        self.stdout = None
        self._done = threading.Event()
        self._return_code: int | None = None

    def poll(self):
        return self._return_code

    def wait(self, timeout=None):
        if not self._done.wait(timeout):
            raise TimeoutError("fake process did not stop")
        return self._return_code

    def send_signal(self, requested_signal):
        assert requested_signal == signal.SIGINT
        self._return_code = 2
        self._done.set()

    def terminate(self):
        self._return_code = -15
        self._done.set()


def _stable_record(tag_id: int) -> dict:
    return {
        "tag_id": tag_id,
        "stable": True,
        "marker_size_m": 0.027,
        "world_from_tag": {
            "translation_m": [0.1, 0.2, 0.0],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "observations": 6,
        "used_observations": 5,
    }


def test_manager_recovers_latest_interrupted_run_and_resumes_it(tmp_path) -> None:
    run_dir = tmp_path / "zero_pose_20260904_120000"
    run_dir.mkdir()
    (run_dir / "floor-origin.json").write_text(json.dumps({
        "marker_size_m": 0.027,
        "floor_tags": {"104": {"world_from_tag": {}}},
    }))
    (run_dir / "progress.json").write_text(json.dumps({
        "status": "scanning",
        "phase": "survey",
        "anchor_ids": [104],
        "frame_sequence": 72,
        "records": [_stable_record(1)],
        "progress": {
            "stable_tag_ids": [1],
            "expected_ground_tag_ids": [100, 104],
            "robot_positions": [],
            "ground_tag_status": [],
        },
    }))
    commands: list[list[str]] = []
    processes: list[_RunningProcess] = []

    def factory(command, **_kwargs):
        commands.append(command)
        process = _RunningProcess()
        processes.append(process)
        return process

    manager = ZeroPoseSurveyManager(
        config_path=CONFIG_PATH,
        survey_dir=tmp_path,
        process_factory=factory,
    )
    try:
        before = manager.public_state()
        assert before["status"] == "connection_lost"
        assert before["can_resume"] is True

        after = manager.resume({"connection_mode": "usb"})

        assert after["active"] is True
        assert "--resume-progress" in commands[0]
        assert "--robot-layout" in commands[0]
        assert "--record3d-device" in commands[0]
        assert "--wifi-frame-dir" not in commands[0]
        assert after["run_dir"] == str(run_dir)
    finally:
        manager.shutdown()


def test_manager_invalidates_completed_legacy_13_tag_result(tmp_path) -> None:
    run_dir = tmp_path / "zero_pose_20260904_120000"
    run_dir.mkdir()
    (run_dir / "survey.json").write_text("{}")
    (run_dir / "progress.json").write_text(json.dumps({
        "status": "complete",
        "records": [_stable_record(1)],
        "progress": {"robot_positions": [{"position": "L0 hip"}]},
    }))

    manager = ZeroPoseSurveyManager(
        config_path=CONFIG_PATH,
        survey_dir=tmp_path,
        process_factory=lambda *_args, **_kwargs: _RunningProcess(),
    )
    state = manager.public_state()

    assert state["status"] == "idle"
    assert state["can_resume"] is False
    assert len(state["progress"]["robot_positions"]) == 37
    assert "old 13-tag" in state["message"]


def test_manager_accepts_browser_relayed_wifi_frame(tmp_path) -> None:
    commands: list[list[str]] = []

    def factory(command, **_kwargs):
        commands.append(command)
        return _RunningProcess()

    manager = ZeroPoseSurveyManager(
        config_path=CONFIG_PATH,
        survey_dir=tmp_path,
        process_factory=factory,
    )
    try:
        state = manager.start({
            "connection_mode": "wifi",
            "wifi_address": "myiphone.local",
        })
        assert state["connection_mode"] == "wifi"
        assert "--wifi-frame-dir" in commands[0]
        accepted = manager.ingest_wifi_frame({
            "sequence": 3,
            "rgbd_jpeg_base64": base64.b64encode(b"jpeg bytes").decode(),
            "camera_matrix": [[600, 0, 320], [0, 600, 240], [0, 0, 1]],
            "camera_pose_xyzw_xyz": [0, 0, 0, 1, 0, 0, 0],
            "max_depth_m": 3.0,
        })
        assert accepted == {"ok": True, "sequence": 3}
        latest = json.loads((tmp_path / state["run_dir"].split("/")[-1] / "wifi_frames" / "latest.json").read_text())
        assert latest["sequence"] == 3
    finally:
        manager.shutdown()


def test_manager_uses_all_seven_metric_floor_tags(tmp_path) -> None:
    manager = ZeroPoseSurveyManager(
        config_path=CONFIG_PATH,
        survey_dir=tmp_path,
        process_factory=lambda *_args, **_kwargs: _RunningProcess(),
    )

    board = manager._floor_board(manager._defaults)

    assert len(manager._robot_positions) == 37
    assert sum(
        item.get("kind") == "yoke_face" for item in manager._robot_positions
    ) == 24
    assert board["marker_size_m"] == 0.0272
    assert set(board["floor_tags"]) == {
        "100", "101", "102", "103", "104", "105", "112"
    }
    assert board["floor_tags"]["104"]["world_from_tag"]["translation_m"] == [
        0.0, 0.0, 0.0
    ]
    tag_101 = board["floor_tags"]["101"]["world_from_tag"]["translation_m"]
    assert abs((tag_101[0] ** 2 + tag_101[1] ** 2) ** 0.5 - 0.8621) < 0.001
