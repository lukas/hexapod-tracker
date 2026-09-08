from types import SimpleNamespace

import pytest

from hexapod_tracker import avfoundation_capture as capture


@pytest.mark.parametrize('requested,expected', [(30, 10), (10, 10), (120, 120)])
def test_selects_supported_rate_instead_of_first_format(monkeypatch, requested, expected):
    subtype = 123
    formats = []
    for rate in (120, 10):
        interval = SimpleNamespace(minFrameRate=lambda rate=rate: rate,
                                   maxFrameRate=lambda rate=rate: rate)
        formats.append(SimpleNamespace(
            formatDescription=lambda rate=rate: (1280, 720, 123 if rate == 120 else 456),
            videoSupportedFrameRateRanges=lambda interval=interval: [interval],
            rate=rate,
        ))
    monkeypatch.setattr(capture, 'CM', SimpleNamespace(
        CMFormatDescriptionGetMediaSubType=lambda description: description[2],
        CMVideoFormatDescriptionGetDimensions=lambda description:
            SimpleNamespace(width=description[0], height=description[1]),
    ))
    monkeypatch.setattr(capture, 'Quartz', SimpleNamespace(
        kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange=subtype,
        kCVPixelFormatType_422YpCbCr8_yuvs=456,
    ))
    camera = capture.AVFoundationYuvCapture(0, fps=requested)
    selected = camera._select_format(SimpleNamespace(formats=lambda: formats))
    assert selected.rate == expected
    assert camera.fps == expected


def _range(min_rate, max_rate, duration):
    return SimpleNamespace(
        minFrameRate=lambda: min_rate,
        maxFrameRate=lambda: max_rate,
        maxFrameDuration=lambda: duration,
    )


def test_fixed_rate_range_reuses_the_devices_own_duration():
    # The 12MP AF module advertises 30 fps as 1000000/30000030 and rejects a
    # duration synthesised from the float rate.
    advertised = ('device', 1000000, 30000030)
    assert capture._frame_duration(_range(30.0, 30.0, advertised), 30.0) is advertised


def test_spanning_rate_range_computes_a_duration(monkeypatch):
    monkeypatch.setattr(capture, 'CM', SimpleNamespace(
        CMTimeMakeWithSeconds=lambda seconds, scale: ('computed', seconds, scale),
    ))
    computed = capture._frame_duration(_range(5.0, 60.0, ('unused',)), 30.0)
    assert computed[0] == 'computed'
    assert computed[1] == pytest.approx(1.0 / 30.0)


def test_parse_device_ids_maps_slots_to_stable_ids():
    from hexapod_tracker.camera_server import parse_device_ids

    assert parse_device_ids(['0:0x11000000c456366', '4:0x412000032e40362']) == {
        0: '0x11000000c456366',
        4: '0x412000032e40362',
    }


@pytest.mark.parametrize('value', ['0x1100', '1:', 'a:0x1100'])
def test_parse_device_ids_rejects_malformed_pairs(value):
    from hexapod_tracker.camera_server import parse_device_ids

    with pytest.raises(SystemExit):
        parse_device_ids([value])


def test_parse_device_ids_rejects_a_repeated_slot():
    from hexapod_tracker.camera_server import parse_device_ids

    with pytest.raises(SystemExit):
        parse_device_ids(['2:0x1100', '2:0x2100'])


def _fake_device(unique_id, name='Cam'):
    return SimpleNamespace(uniqueID=lambda: unique_id,
                           localizedName=lambda: name)


def test_device_stable_id_prefers_the_unique_id():
    assert capture.device_stable_id(_fake_device('0x84000000c456366')) == \
        '0x84000000c456366'


def test_device_stable_id_falls_back_to_the_name():
    assert capture.device_stable_id(_fake_device('', 'Studio Display')) == \
        'avfoundation:Studio Display'


def test_start_session_selects_the_pinned_device_not_the_slot(monkeypatch):
    # The pinned camera sits at index 2; a bare index would open the wrong one.
    wanted = _fake_device('0x21000000c456366', 'Arducam OV9281')
    devices = [_fake_device('0x11000000c456366'), _fake_device('0x84'), wanted]
    monkeypatch.setattr(capture.AVFoundationYuvCapture, '_devices',
                        staticmethod(lambda: devices))
    camera = capture.AVFoundationYuvCapture(0, stable_id='0x21000000c456366')

    class _Reached(Exception):
        pass

    picked = []

    def _capture_and_stop(device):
        picked.append(device)
        raise _Reached

    monkeypatch.setattr(camera, '_select_format', _capture_and_stop)
    with pytest.raises(_Reached):
        camera._start_session()
    assert picked == [wanted]
    assert camera.device_name == 'Arducam OV9281'
    assert camera.device_unique_id == '0x21000000c456366'


def test_start_session_reports_a_missing_pinned_device(monkeypatch):
    monkeypatch.setattr(capture.AVFoundationYuvCapture, '_devices',
                        staticmethod(lambda: [_fake_device('0x11')]))
    camera = capture.AVFoundationYuvCapture(0, stable_id='0xdeadbeef')
    with pytest.raises(RuntimeError, match='0xdeadbeef is not attached'):
        camera._start_session()
