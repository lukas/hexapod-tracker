"""Offline tests for publishing reviewed tag calibration to Robot Lab."""
from __future__ import annotations

import json
import io
from types import SimpleNamespace
from urllib import error

import pytest

from hexapod_tracker.robot_lab import RobotLabHTTPError, RobotLabPublisher


class _Response:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        pass

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_publisher_accepts_hexipod_token_spelling(monkeypatch) -> None:
    monkeypatch.setenv("HEXIPOD_LAB_TOKEN", "from-keychain-export")
    monkeypatch.delenv("HEXAPOD_LAB_TOKEN", raising=False)

    publisher = RobotLabPublisher.from_env()

    assert publisher.token == "from-keychain-export"
    assert publisher.credential_source == "environment"


def test_publisher_reads_protected_token_file_without_logging_it(
    tmp_path, monkeypatch
) -> None:
    token_path = tmp_path / "robot-lab-token.txt"
    token_path.write_text("HEXIPOD_LAB_TOKEN=secret-token-from-file\n")
    monkeypatch.delenv("HEXIPOD_LAB_TOKEN", raising=False)
    monkeypatch.delenv("HEXAPOD_LAB_TOKEN", raising=False)
    monkeypatch.setenv("HEXIPOD_LAB_TOKEN_FILE", str(token_path))
    monkeypatch.setattr("hexapod_tracker.robot_lab.shutil.which", lambda _name: None)

    publisher = RobotLabPublisher.from_env()

    assert publisher.token == "secret-token-from-file"
    assert publisher.credential_source == "credential_file"


def test_publisher_can_read_onepassword_reference(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("HEXIPOD_LAB_TOKEN", raising=False)
    monkeypatch.delenv("HEXAPOD_LAB_TOKEN", raising=False)
    monkeypatch.setenv("HEXIPOD_LAB_TOKEN_FILE", str(tmp_path / "missing"))
    monkeypatch.setattr(
        "hexapod_tracker.robot_lab.shutil.which", lambda _name: "/usr/local/bin/op"
    )
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="secret-token-from-1password\n")

    monkeypatch.setattr("hexapod_tracker.robot_lab.subprocess.run", fake_run)

    publisher = RobotLabPublisher.from_env()

    assert publisher.token == "secret-token-from-1password"
    assert publisher.credential_source == "1password"
    assert calls == [[
        "/usr/local/bin/op", "read", "op://Private/Hexapod Lab API/credential"
    ]]


def test_publisher_uses_first_class_calibration_endpoint(
    tmp_path, monkeypatch
) -> None:
    result_path = tmp_path / "survey.json"
    config_path = tmp_path / "tracker.json"
    result_path.write_text(json.dumps({
        "schema_version": 1,
        "created_utc": "2026-09-05T00:00:00+00:00",
        "leg_zero_reference": {"declared_tag_id": 1},
        "survey": {
            "robot_positions": [{"position": "L0 hip", "tag_id": 1}],
            "ground_tag_status": [{"tag_id": 104}],
            "expected_ground_tag_ids": [104],
            "stable_tag_count": 2,
        },
    }))
    config_path.write_text(json.dumps({
        "schema_version": 1,
        "floor_tags": {"104": {}},
        "robot_pose": {},
    }))
    calls = []

    def fake_urlopen(outgoing, timeout):
        calls.append((outgoing, timeout))
        return _Response({
            "id": "abc123",
            "url": "https://robot-lab.example/api/calibrations/abc123",
        })

    monkeypatch.setattr("hexapod_tracker.robot_lab.request.urlopen", fake_urlopen)
    publisher = RobotLabPublisher("https://robot-lab.example", "secret")

    published = publisher.publish_zero_pose_calibration(
        result_path=result_path,
        config_path=config_path,
    )

    assert published["status"] == "published"
    assert published["transport"] == "calibration_config"
    assert len(calls) == 1
    outgoing, timeout = calls[0]
    assert outgoing.full_url.endswith("/api/calibrations")
    assert outgoing.get_method() == "POST"
    assert timeout == 20.0
    body = json.loads(outgoing.data)
    assert body["scope"] == "combined"
    assert body["report"]["kind"] == "iphone_lidar_zero_pose_survey"
    assert body["report"]["robot_id"] == "hexapod-1"
    assert body["report"]["observed_at"] == "2026-09-05T00:00:00+00:00"
    assert body["report"]["motor_commands_sent"] is False
    assert body["report"]["servo_zeros_changed"] is False
    assert body["pose_config"]["floor_tags"] == {"104": {}}
    assert body["report"]["survey"]["stable_tag_count"] == 2
    assert outgoing.get_header("Idempotency-key").startswith("zero-pose-")


def test_publisher_does_not_fall_back_to_legacy_result_api(
    tmp_path, monkeypatch
) -> None:
    result_path = tmp_path / "survey.json"
    config_path = tmp_path / "tracker.json"
    result_path.write_text(json.dumps({"survey": {"robot_positions": []}}))
    config_path.write_text(json.dumps({"floor_tags": {}}))
    calls = []

    def fake_urlopen(outgoing, timeout):
        calls.append((outgoing, timeout))
        raise error.HTTPError(
            outgoing.full_url, 404, "not found", {}, io.BytesIO(b"not found")
        )

    monkeypatch.setattr("hexapod_tracker.robot_lab.request.urlopen", fake_urlopen)
    publisher = RobotLabPublisher("https://robot-lab.example", "secret")

    with pytest.raises(RobotLabHTTPError):
        publisher.publish_zero_pose_calibration(
            result_path=result_path,
            config_path=config_path,
        )

    assert len(calls) == 1
    assert calls[0][0].full_url.endswith("/api/calibrations")
