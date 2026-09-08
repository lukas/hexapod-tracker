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
