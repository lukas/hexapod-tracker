"""Health rollup and the Robot Lab mode hint on the multi-camera server."""
from types import SimpleNamespace

import pytest

from hexapod_tracker import camera_server as cs


def _server(snapshots, anchors=(100, 101, 102)):
    """A CameraHTTPServer with its socket setup and workers stubbed out."""
    server = cs.CameraHTTPServer.__new__(cs.CameraHTTPServer)
    server.workers = [
        SimpleNamespace(snapshot=lambda item=item: (None, item)) for item in snapshots
    ]
    server.floor_anchor_ids = {int(value) for value in anchors}
    server._observation_mode = cs.OBSERVATION_MODE_TRACK
    import threading

    server._observation_mode_lock = threading.Lock()
    return server


def _snapshot(index, **overrides):
    item = {
        "index": index,
        "device_name": f"Camera {index}",
        "requested_stable_id": f"0x{index}",
        "state": "streaming",
        "measured_fps": 10.0,
        "frames": 100,
        "reconnects": 0,
        "last_frame_age_s": 0.05,
        "native_capture_width": 1280,
        "native_capture_height": 720,
        "tag_ids": [],
        "error": None,
    }
    item.update(overrides)
    return item


def test_default_mode_is_track_so_nothing_is_ever_turned_off():
    assert _server([]).observation_mode == cs.OBSERVATION_MODE_TRACK


@pytest.mark.parametrize("mode", ["survey", "TRACK", " Survey "])
def test_set_observation_mode_normalises_known_modes(mode):
    server = _server([])
    assert server.set_observation_mode(mode) == mode.strip().lower()


@pytest.mark.parametrize("mode", ["", "calibrate", "off", None])
def test_set_observation_mode_rejects_unknown_modes(mode):
    server = _server([])
    with pytest.raises(ValueError):
        server.set_observation_mode(mode)
    assert server.observation_mode == cs.OBSERVATION_MODE_TRACK


def test_healthy_camera_reports_no_reasons():
    health = _server([_snapshot(0, tag_ids=[1, 100])]).camera_health()
    camera = health["cameras"][0]
    assert camera["healthy"] and camera["reasons"] == []
    assert health["cameras_healthy"] == 1
    assert camera["floor_anchors_seen"] == [100]


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({"state": "stalled"}, "state is stalled"),
        ({"frames": 0}, "no frames delivered yet"),
        ({"last_frame_age_s": None}, "no frame timestamp"),
        ({"last_frame_age_s": 9.5}, "last frame 9.5s old"),
        ({"error": "capture stalled"}, "capture stalled"),
    ],
)
def test_unhealthy_camera_names_the_reason(overrides, expected):
    health = _server([_snapshot(0, **overrides)]).camera_health()
    camera = health["cameras"][0]
    assert not camera["healthy"]
    assert expected in camera["reasons"]
    assert health["cameras_healthy"] == 0


def test_redundant_camera_is_healthy_but_adds_nothing():
    # Slot 1 sees a strict subset of slot 0, so it is redundant, not broken.
    health = _server([
        _snapshot(0, tag_ids=[1, 2, 3]),
        _snapshot(1, tag_ids=[1, 2]),
    ]).camera_health()
    by_slot = {item["slot"]: item for item in health["cameras"]}
    assert by_slot[1]["healthy"] is True
    assert by_slot[1]["redundant"] is True
    assert by_slot[1]["unique_tags"] == []
    assert by_slot[0]["redundant"] is False
    assert by_slot[0]["unique_tags"] == [3]


def test_a_camera_seeing_nothing_is_not_called_redundant():
    health = _server([_snapshot(0, tag_ids=[]), _snapshot(1, tag_ids=[5])]).camera_health()
    by_slot = {item["slot"]: item for item in health["cameras"]}
    assert by_slot[0]["redundant"] is False


def test_rollup_reports_missing_floor_anchors():
    health = _server(
        [_snapshot(0, tag_ids=[100, 7]), _snapshot(1, tag_ids=[101])],
        anchors=(100, 101, 102),
    ).camera_health()
    assert health["union_tags_seen"] == 3
    assert health["floor_anchors_seen"] == [100, 101]
    assert health["floor_anchors_missing"] == [102]


def test_preview_downscale_only_shrinks_when_wider_than_the_limit():
    import numpy as np

    wide = np.zeros((720, 1280, 3), dtype=np.uint8)
    narrow = np.zeros((360, 640, 3), dtype=np.uint8)

    assert cs.downscale_preview(wide, 960).shape == (540, 960, 3)
    # Already under the limit: returned untouched rather than upscaled.
    assert cs.downscale_preview(narrow, 960) is narrow
    # 0 disables downscaling entirely.
    assert cs.downscale_preview(wide, 0) is wide


def test_frame_age_is_none_before_the_first_frame_and_measured_after():
    import threading
    import time

    worker = cs.CameraWorker.__new__(cs.CameraWorker)
    worker._condition = threading.Condition()
    worker._last_frame_at = None
    assert worker.frame_age_s() is None

    worker._last_frame_at = time.monotonic() - 0.5
    age = worker.frame_age_s()
    assert age is not None and 0.4 < age < 0.7


def test_label_reports_the_last_pass_when_no_boxes_are_drawn():
    import numpy as np

    frame = np.zeros((200, 800, 3), dtype=np.uint8)
    # No detections for this frame, but the camera did see tags recently.
    annotated, tag_ids = cs.annotate_tag_corners(
        frame, {}, 1, "12:00:00.000", label_tag_ids=[3, 9]
    )
    assert tag_ids == []
    assert annotated is frame


def test_label_without_a_last_pass_still_says_none():
    import numpy as np

    frame = np.zeros((200, 800, 3), dtype=np.uint8)
    _annotated, tag_ids = cs.annotate_tag_corners(frame, {}, 1, None)
    assert tag_ids == []


class _FakeCapture:
    def __init__(self, gray=None):
        self.detection_gray = gray


def test_detection_uses_the_full_resolution_luma_and_rescales_corners(monkeypatch):
    import numpy as np

    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    gray = np.zeros((1080, 1920), dtype=np.uint8)  # 3x the frame width
    seen = {}

    def fake_detect(image, _detector):
        seen['shape'] = image.shape
        return {7: np.array([[300.0, 600.0]], dtype=np.float32)}

    monkeypatch.setattr(cs, 'detect_tag_corners', fake_detect)
    corners, size = cs.detect_tags_at_best_resolution(_FakeCapture(gray), frame, None)

    assert seen['shape'] == (1080, 1920)   # detected on the big image
    assert size == (1920, 1080)
    # 3x down into frame coordinates, so annotation and pose stay consistent.
    assert corners[7].tolist() == [[100.0, 200.0]]


def test_detection_falls_back_when_no_larger_image_exists(monkeypatch):
    import numpy as np

    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    seen = {}

    def fake_detect(image, _detector):
        seen['shape'] = image.shape
        return {}

    monkeypatch.setattr(cs, 'detect_tag_corners', fake_detect)
    # No native plane at all, and a plane no bigger than the frame, both fall back.
    for capture in (_FakeCapture(None), _FakeCapture(np.zeros((360, 640), dtype=np.uint8))):
        _corners, size = cs.detect_tags_at_best_resolution(capture, frame, None)
        assert seen['shape'] == (360, 640)
        assert size == (640, 360)


def test_parse_capture_sizes_reads_per_slot_overrides():
    assert cs.parse_capture_sizes(['0:4000x3000', '2:1280X720']) == {
        0: (4000, 3000),
        2: (1280, 720),
    }


@pytest.mark.parametrize(
    'value', ['4000x3000', '0:4000', '0:axb', '0:0x100', '0:-4x3']
)
def test_parse_capture_sizes_rejects_malformed_values(value):
    with pytest.raises(SystemExit):
        cs.parse_capture_sizes([value])


def test_parse_capture_sizes_rejects_a_repeated_slot():
    with pytest.raises(SystemExit):
        cs.parse_capture_sizes(['1:640x480', '1:800x600'])
