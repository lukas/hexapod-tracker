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
