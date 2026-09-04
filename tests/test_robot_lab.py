"""Offline tests for publishing reviewed tag calibration to Robot Lab."""
from __future__ import annotations

import json

from hexapod_tracker.robot_lab import RobotLabPublisher


class _Response:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        pass

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_publisher_uses_first_class_calibration_endpoint(
    tmp_path, monkeypatch
) -> None:
    result_path = tmp_path / "survey.json"
    config_path = tmp_path / "tracker.json"
    result_path.write_text(json.dumps({
        "schema_version": 1,
        "leg_zero_reference": {"declared_tag_id": 1},
        "survey": {
            "robot_positions": [{"position": "L0 hip", "tag_id": 1}],
            "ground_tag_status": [{"tag_id": 104}],
            "expected_ground_tag_ids": [104],
            "stable_tag_count": 2,
        },
    }))
    config_path.write_text(json.dumps({"floor_tags": {"104": {}}, "robot_pose": {}}))
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
        duration_seconds=4.2,
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
    assert body["robot_id"] == "hexapod-1"
    assert body["configuration"]["floor_tags"] == {"104": {}}
    assert body["survey"]["survey"]["stable_tag_count"] == 2
