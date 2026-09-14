"""Exercise recorder durability with real ffmpeg and entirely synthetic frames."""
from __future__ import annotations

import csv
import shutil
import signal
import struct
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

from hexapod_tracker import cameras


SIZE = (96, 64)


@pytest.fixture(scope="module")
def ffmpeg() -> str:
    executable = shutil.which(cameras.FFMPEG)
    if executable is None:
        pytest.skip("ffmpeg is not installed")
    encoders = subprocess.run(
        [executable, "-hide_banner", "-encoders"],
        capture_output=True, text=True, check=True, timeout=10,
    )
    if "libx264" not in encoders.stdout:
        pytest.skip("ffmpeg was built without the libx264 encoder")
    return executable


def frame(index: int, fps: float) -> cameras.Frame:
    # Distinct solid grey frames make both their order and decoded content testable
    # without depending on exact bytes from a lossy codec.
    gray = np.full((SIZE[1], SIZE[0]), 20 + index * 5, dtype=np.uint8)
    return cameras.Frame(
        bgr=np.repeat(gray[:, :, None], 3, axis=2),
        gray=gray,
        captured_unix=1_700_000_000 + index / fps,
        seq=100 + index,
    )


def timestamps(path: Path) -> list[dict[str, str]]:
    with path.with_name(path.stem + "_timestamps.csv").open(newline="") as stream:
        return list(csv.DictReader(stream))


def decode(path: Path, ffmpeg: str) -> np.ndarray:
    result = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(path), "-map", "0:v:0",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        capture_output=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert result.stdout, "the recording contained no decodable frames"
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, SIZE[1], SIZE[0], 3)


def completed_fragments(path: Path) -> int:
    """Count complete top-level media boxes, ignoring a still-being-written tail."""
    data = path.read_bytes() if path.exists() else b""
    offset = 0
    media_boxes = 0
    while offset + 8 <= len(data):
        size, kind = struct.unpack_from(">I4s", data, offset)
        if size == 1:
            if offset + 16 > len(data):
                break
            size = struct.unpack_from(">Q", data, offset + 8)[0]
        if size < 8 or offset + size > len(data):
            break
        media_boxes += kind == b"mdat"
        offset += size
    return media_boxes


def cleanup(recorder: cameras.VideoRecorder) -> None:
    # Also clean up if an assertion fails, so a failing regression never leaves
    # an encoder subprocess around after pytest exits.
    if recorder.proc.poll() is None:
        recorder.proc.kill()
        recorder.proc.wait(timeout=5)
    try:
        recorder.close()
    except (OSError, RuntimeError, ValueError):
        pass


def test_normal_close_preserves_every_frame_and_flushes_timestamps(tmp_path, ffmpeg):
    path = tmp_path / "normal.mp4"
    fps = 7.5
    recorder = cameras.VideoRecorder(path, fps, SIZE, ffmpeg=ffmpeg)
    try:
        for index in range(23):
            recorder.write(frame(index, fps))

        # Metadata must already be on disk while capture is still running.
        rows = timestamps(path)
        assert len(rows) == recorder.frames == 23
        assert [int(row["frame"]) for row in rows] == list(range(23))
        assert [int(row["seq"]) for row in rows] == list(range(100, 123))
        assert [float(row["captured_unix"]) for row in rows] == pytest.approx(
            [frame(index, fps).captured_unix for index in range(23)], rel=0, abs=1e-6,
        )

        recorder.close()
        recorder.close()
        assert recorder.proc.returncode == 0
        assert recorder.error is None
        assert recorder.proc.stdin.closed
        assert recorder._ts.closed

        decoded = decode(path, ffmpeg)
        assert len(decoded) == 23
        assert decoded.mean(axis=(1, 2, 3)) == pytest.approx(
            [20 + index * 5 for index in range(23)], abs=3,
        )
    finally:
        cleanup(recorder)


@pytest.mark.parametrize("detect_on_write", [False, True])
def test_encoder_kill_keeps_completed_video_and_reports_failure(tmp_path, ffmpeg, detect_on_write):
    path = tmp_path / "interrupted.mp4"
    recorder = cameras.VideoRecorder(path, 10, SIZE, ffmpeg=ffmpeg)
    try:
        for index in range(35):
            recorder.write(frame(index, 10))

        # Observe actual persisted fragments rather than sleeping for an assumed
        # encoder speed. There has been no EOF or graceful finalization yet.
        deadline = time.monotonic() + 5
        while completed_fragments(path) < 2 and time.monotonic() < deadline:
            assert recorder.proc.poll() is None, "encoder exited before capture stopped"
            time.sleep(0.02)
        assert completed_fragments(path) >= 2, "video was not persisted during recording"

        recorder.proc.kill()
        recorder.proc.wait(timeout=5)
        assert recorder.proc.returncode != 0

        # Decode before close: recovery must not depend on cleanup running.
        decoded = decode(path, ffmpeg)
        assert 20 <= len(decoded) <= recorder.frames
        assert decoded.mean(axis=(1, 2, 3)) == pytest.approx(
            [20 + index * 5 for index in range(len(decoded))], abs=3,
        )
        assert len(timestamps(path)) == recorder.frames == 35

        if detect_on_write:
            with pytest.raises(RuntimeError):
                recorder.write(frame(35, 10))
            assert recorder.error
            assert recorder.frames == 35

        with pytest.raises(RuntimeError) as first:
            recorder.close()
        assert recorder.error
        assert recorder.proc.stdin.closed
        assert recorder._ts.closed
        with pytest.raises(RuntimeError) as repeated:
            recorder.close()
        assert str(repeated.value) == str(first.value)
    finally:
        cleanup(recorder)


@pytest.mark.skipif(not hasattr(signal, "SIGSTOP"), reason="requires POSIX process signals")
def test_close_timeout_kills_and_reaps_stalled_encoder(tmp_path, ffmpeg, monkeypatch):
    recorder = cameras.VideoRecorder(tmp_path / "stalled.mp4", 10, SIZE, ffmpeg=ffmpeg)
    original_wait = recorder.proc.wait
    first_wait = True

    def shortened_wait(timeout=None):
        nonlocal first_wait
        if first_wait:
            first_wait = False
            timeout = 0.1
        return original_wait(timeout=timeout)

    try:
        recorder.write(frame(0, 10))
        recorder.proc.send_signal(signal.SIGSTOP)
        # The process really stalls; shorten only its initial graceful-exit wait
        # so testing the production timeout path does not take thirty seconds.
        monkeypatch.setattr(recorder.proc, "wait", shortened_wait)

        with pytest.raises(RuntimeError, match="did not finish") as first:
            recorder.close()
        assert recorder.proc.returncode == -signal.SIGKILL
        assert recorder.proc.stdin.closed
        assert recorder._ts.closed
        assert recorder.error
        with pytest.raises(RuntimeError) as repeated:
            recorder.close()
        assert str(repeated.value) == str(first.value)
    finally:
        cleanup(recorder)
