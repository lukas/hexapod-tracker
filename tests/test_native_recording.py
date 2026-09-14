"""Native movie lifecycle with fake frameworks; no camera discovery or capture."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hexapod_tracker import avfoundation_capture as capture


def allocated(instance):
    return SimpleNamespace(alloc=lambda: SimpleNamespace(init=lambda: instance))


@pytest.fixture
def recording(monkeypatch, tmp_path):
    camera = capture.AVFoundationYuvCapture(0, fps=29.97, processing_width=640)
    camera.capture_image_size_px = (3840, 2160)
    movie, analysis, session = Mock(), Mock(), Mock()
    state = SimpleNamespace(
        camera=camera, movie=movie, analysis=analysis, session=session,
        events=[], callbacks=[], now=0.0, deliver_start=True,
        deliver_finish=True, finish_error=None, path=tmp_path / "native.mov",
        frame_sizes=[(3840, 2160)],
    )
    device = SimpleNamespace(uniqueID=lambda: "synthetic-camera", localizedName=lambda: "Synthetic")
    monkeypatch.setattr(camera, "_devices", lambda: [device])
    monkeypatch.setattr(camera, "_select_format", lambda _: object())
    monkeypatch.setattr(camera, "_configure_device", lambda *args, **kwargs: True)

    constants = (
        "AVMediaTypeVideo", "AVCaptureSessionPresetInputPriority", "AVCaptureSessionPresetPhoto",
        "AVVideoCodecKey", "AVVideoCodecTypeH264", "AVVideoWidthKey", "AVVideoHeightKey",
        "AVVideoEncoderSpecificationKey", "AVVideoCompressionPropertiesKey",
        "AVVideoMaxKeyFrameIntervalDurationKey", "AVVideoAllowFrameReorderingKey",
    )
    monkeypatch.setattr(capture, "AV", SimpleNamespace(
        **{name: name for name in constants},
        AVCaptureSession=allocated(session),
        AVCaptureDeviceInput=SimpleNamespace(deviceInputWithDevice_error_=lambda *args: (object(), None)),
        AVCaptureVideoDataOutput=allocated(analysis),
        AVCaptureMovieFileOutput=allocated(movie),
        AVURLAsset=SimpleNamespace(assetWithURL_=lambda url: SimpleNamespace(duration=lambda: 4.2)),
    ))
    monkeypatch.setattr(capture, "Quartz", SimpleNamespace(
        kCVPixelBufferPixelFormatTypeKey="format",
        kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange="420v",
        kCVPixelBufferWidthKey="width", kCVPixelBufferHeightKey="height",
    ))
    monkeypatch.setattr(capture, "CM", SimpleNamespace(
        CMTimeMake=lambda value, scale: (value, scale), CMTimeGetSeconds=float,
    ))
    monkeypatch.setattr(capture, "NSURL", SimpleNamespace(fileURLWithPath_=lambda path: path))
    monkeypatch.setattr(capture, "_dispatch_queue", lambda label: object())
    monkeypatch.setattr(capture, "_FrameDelegate", allocated(SimpleNamespace()), raising=False)
    monkeypatch.setattr(capture, "_MovieDelegate", allocated(SimpleNamespace()), raising=False)

    session.canAddInput_.return_value = True
    session.canAddOutput_.return_value = True
    session.isRunning.return_value = True
    session.addOutput_.side_effect = lambda output: state.events.append(
        "attach-movie" if output is movie else "attach-analysis")
    session.startRunning.side_effect = lambda: state.events.append("session-start")
    # The live counter can include preroll trimmed from the finished movie.
    movie.recordedDuration.return_value = 4.9
    movie.connectionWithMediaType_.return_value.isVideoRotationAngleSupported_.return_value = True

    def did_start():
        state.events.append("did-start")
        camera._recording_started.set()

    def start_movie(*args):
        state.events.append("movie-start")
        if state.deliver_start:
            state.callbacks.append(did_start)

    def did_finish():
        # Finalization must happen before release shuts down its session loop.
        assert not camera._stop.is_set()
        state.events.append("did-finish")
        camera._recording_error = state.finish_error
        camera._recording_finished.set()

    def stop_movie():
        state.events.append("movie-stop")
        if state.deliver_finish:
            state.callbacks.append(did_finish)

    movie.startRecordingToOutputFileURL_recordingDelegate_.side_effect = start_movie
    movie.stopRecording.side_effect = stop_movie

    def run_loop(_date):
        state.now += 0.05
        if state.callbacks:
            state.callbacks.pop(0)()

    monkeypatch.setattr(capture, "NSRunLoop", SimpleNamespace(
        currentRunLoop=lambda: SimpleNamespace(runUntilDate_=run_loop),
    ))
    monkeypatch.setattr(capture, "NSDate", SimpleNamespace(dateWithTimeIntervalSinceNow_=lambda interval: interval))
    monkeypatch.setattr(capture, "time", SimpleNamespace(monotonic=lambda: state.now))

    def ensure_session():
        # Run the real session setup against fake frameworks, without a camera
        # thread or device. File callbacks still require the caller's run loop.
        camera._thread = Mock()
        camera._start_session()

    monkeypatch.setattr(camera, "_ensure_thread", ensure_session)

    def read_frame():
        state.now += 0.05
        width, height = state.frame_sizes[0]
        if len(state.frame_sizes) > 1:
            state.frame_sizes.pop(0)
        camera.detection_gray = SimpleNamespace(shape=(height, width))
        state.events.append("native-frame")
        return True, object()

    monkeypatch.setattr(camera, "read", Mock(side_effect=read_frame))
    return state


def test_native_movie_joins_analysis_session_before_start_and_keeps_full_size(recording):
    state = recording
    state.camera.prepare_recording(state.path, rotate_180=True)
    assert state.events.index("attach-movie") < state.events.index("session-start")
    assert state.events.index("attach-analysis") < state.events.index("session-start")
    assert state.events[-1] == "native-frame"
    state.movie.startRecordingToOutputFileURL_recordingDelegate_.assert_not_called()
    assert not state.camera._recording_started.is_set()
    state.camera.start_recording()
    assert state.events.index("session-start") < state.events.index("movie-start")
    assert state.events[-1] == "did-start"
    assert state.camera._recording_started.is_set()
    settings, connection = state.movie.setOutputSettings_forConnection_.call_args.args
    assert settings["AVVideoWidthKey"] == 3840
    assert settings["AVVideoHeightKey"] == 2160
    connection.setVideoRotationAngle_.assert_called_once_with(180)
    url, delegate = state.movie.startRecordingToOutputFileURL_recordingDelegate_.call_args.args
    assert url == str(state.path.resolve())
    assert delegate.capture_owner is state.camera


def test_stop_waits_for_finalization_even_if_is_recording_reports_false(recording):
    state = recording
    state.camera.prepare_recording(state.path)
    state.camera.start_recording()
    state.movie.isRecording.return_value = False
    result = state.camera.stop_recording()
    assert state.events[-2:] == ["movie-stop", "did-finish"]
    assert result == {
        "path": str(state.path.resolve()), "fps": 29.97,
        "size": [3840, 2160], "duration_s": 4.2, "finalized": True,
    }
    assert state.camera.stop_recording() == result
    state.movie.stopRecording.assert_called_once()


def test_failed_finish_never_returns_finalized_metadata(recording):
    state = recording
    state.camera.prepare_recording(state.path)
    state.camera.start_recording()
    state.finish_error = "movie writer ran out of disk space"
    for _ in range(2):
        with pytest.raises(RuntimeError, match="out of disk space"):
            state.camera.stop_recording()
    assert state.camera._recording_result is None
    state.movie.stopRecording.assert_called_once()


@pytest.mark.parametrize("phase", ["start", "finish"])
def test_missing_movie_callback_times_out_instead_of_claiming_success(recording, phase):
    state = recording
    state.camera.prepare_recording(state.path)
    if phase == "start":
        state.deliver_start = False
        operation = state.camera.start_recording
    else:
        state.camera.start_recording()
        state.deliver_finish = False
        operation = state.camera.stop_recording
    with pytest.raises(RuntimeError, match="callback timed out"):
        operation()
    assert state.camera._recording_result is None
    assert state.now >= 15


@pytest.mark.parametrize("finish_error", [None, "movie finalization failed"])
def test_release_waits_for_finish_and_releases_resources_even_after_error(recording, finish_error):
    state = recording
    state.camera.prepare_recording(state.path)
    state.camera.start_recording()
    state.finish_error = finish_error
    thread = state.camera._thread
    if finish_error:
        with pytest.raises(RuntimeError, match=finish_error):
            state.camera.release()
    else:
        state.camera.release()
    assert state.events[-2:] == ["movie-stop", "did-finish"]
    assert state.camera._released
    assert state.camera._stop.is_set()
    thread.join.assert_called_once()
    assert state.camera._movie is None
    assert state.camera._movie_delegate is None


def test_cannot_add_movie_after_camera_read_has_started(recording):
    state = recording
    state.camera._thread = Mock()
    with pytest.raises(RuntimeError, match="before the first camera read"):
        state.camera.prepare_recording(state.path)
    state.movie.startRecordingToOutputFileURL_recordingDelegate_.assert_not_called()


def test_prepare_waits_until_native_frame_matches_selected_movie_resolution(recording):
    state = recording
    state.frame_sizes = [(640, 480), (3840, 2160)]
    state.camera.prepare_recording(state.path)
    assert state.camera.read.call_count == 2
    state.movie.startRecordingToOutputFileURL_recordingDelegate_.assert_not_called()


def test_prepare_fails_if_frames_never_reach_selected_resolution(recording):
    state = recording
    state.frame_sizes = [(640, 480)]
    with pytest.raises(RuntimeError, match="did not settle at the selected resolution"):
        state.camera.prepare_recording(state.path)
    state.movie.startRecordingToOutputFileURL_recordingDelegate_.assert_not_called()


def test_release_of_prepared_camera_does_not_wait_for_nonexistent_movie(recording):
    state = recording
    state.camera.prepare_recording(state.path)
    thread = state.camera._thread
    before = state.now
    state.camera.release()
    assert state.now == before
    state.movie.stopRecording.assert_not_called()
    thread.join.assert_called_once()
    assert state.camera._released
    assert state.camera._movie is None


def test_start_without_preparation_is_rejected(recording):
    with pytest.raises(RuntimeError, match="prepare native recording before starting"):
        recording.camera.start_recording()
    recording.movie.startRecordingToOutputFileURL_recordingDelegate_.assert_not_called()
