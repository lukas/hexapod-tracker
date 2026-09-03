"""Low-copy macOS camera capture with AVFoundation and native 420v planes.

Continuity Camera exposes bi-planar 4:2:0 video (``420v``/NV12), but OpenCV's
``VideoCapture`` converts it to BGR before Python sees it.  This same-process
AVFoundation adapter keeps the native luminance plane for AprilTag decoding
and creates a smaller BGR image only for the red foot-tip tracker and browser
preview.

This is camera I/O only.  It has no robot-control or network-control paths.
"""
from __future__ import annotations

import ctypes
import sys
import threading
import time
from typing import Any, Sequence

import cv2
import numpy as np


try:  # These frameworks are intentionally a macOS-only optional dependency.
    import AVFoundation as AV
    import CoreMedia as CM
    from Foundation import NSDate, NSObject, NSRunLoop
    import Quartz
    import objc
except ImportError:  # pragma: no cover - exercised by non-macOS installations
    AV = CM = Quartz = objc = None
    NSDate = NSObject = NSRunLoop = None


def _frameworks_available() -> bool:
    return sys.platform == "darwin" and AV is not None


def _dispatch_queue(label: bytes) -> Any:
    library = ctypes.CDLL(None)
    create = library.dispatch_queue_create
    create.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
    create.restype = ctypes.c_void_p
    pointer = create(label, None)
    if not pointer:
        raise RuntimeError("could not create the AVFoundation capture queue")
    return objc.objc_object(c_void_p=pointer)


if _frameworks_available():
    _CAPTURE_PROTOCOL = objc.protocolNamed(
        "AVCaptureVideoDataOutputSampleBufferDelegate"
    )

    class _FrameDelegate(NSObject, protocols=[_CAPTURE_PROTOCOL]):
        def captureOutput_didOutputSampleBuffer_fromConnection_(
            self, _output: Any, sample_buffer: Any, _connection: Any
        ) -> None:
            owner = getattr(self, "capture_owner", None)
            if owner is None:
                return
            try:
                owner._accept_sample_buffer(sample_buffer)
            except Exception as error:  # never unwind through AVFoundation
                owner._set_callback_error(str(error))


class AVFoundationYuvCapture:
    """Small ``cv2.VideoCapture``-compatible native 420v adapter."""

    provides_native_luma = True
    backend_name = "avfoundation-yuv"
    pixel_format = "NV12 / 420v"

    def __init__(
        self,
        index: int,
        *,
        preferred_sizes: Sequence[tuple[int, int]] = (
            (1920, 1440),
            (1920, 1080),
            (1280, 720),
        ),
        fps: float = 30.0,
        processing_width: int = 1280,
        frame_timeout_s: float = 6.0,
    ) -> None:
        self.index = int(index)
        self.preferred_sizes = tuple(
            (int(width), int(height)) for width, height in preferred_sizes
        )
        self.fps = float(fps)
        self.processing_width = int(processing_width)
        self.frame_timeout_s = float(frame_timeout_s)
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sequence = 0
        self._read_sequence = 0
        self._latest_planes: tuple[np.ndarray, np.ndarray] | None = None
        self._session: Any | None = None
        self._output: Any | None = None
        self._delegate: Any | None = None
        self._queue: Any | None = None
        self._released = False

        self.capture_image_size_px: tuple[int, int] | None = None
        self.image_size_px: tuple[int, int] | None = None
        self.detection_gray: np.ndarray | None = None
        self.tracking_gray: np.ndarray | None = None
        self.last_error: str | None = None

    def isOpened(self) -> bool:  # noqa: N802 - match OpenCV's API
        return (
            not self._released
            and _frameworks_available()
            and bool(self.preferred_sizes)
        )

    @staticmethod
    def _devices() -> list[Any]:
        discovery = AV.AVCaptureDeviceDiscoverySession \
            .discoverySessionWithDeviceTypes_mediaType_position_(
                [
                    AV.AVCaptureDeviceTypeBuiltInWideAngleCamera,
                    AV.AVCaptureDeviceTypeContinuityCamera,
                    AV.AVCaptureDeviceTypeExternal,
                ],
                AV.AVMediaTypeVideo,
                AV.AVCaptureDevicePositionUnspecified,
            )
        return list(discovery.devices())

    @classmethod
    def device_descriptors(cls) -> list[dict[str, Any]]:
        """Return the cameras AVFoundation can actually open, in index order.

        Camera indexes on macOS are ephemeral.  In particular, index 1 is not
        inherently an iPhone: Continuity Camera may disappear when the phone
        is out of range, unlocked, in use, or disabled.  The UI therefore
        needs the live AVFoundation names instead of guessing from an index.
        """
        descriptors: list[dict[str, Any]] = []
        for index, device in enumerate(cls._devices()):
            name = str(device.localizedName())
            device_type = str(
                device.deviceType() if hasattr(device, "deviceType") else ""
            )
            # macOS 26 currently reports an iPhone webcam using the generic
            # AVCaptureDeviceTypeExternal value, so keep the localized name
            # as the secondary Continuity signal.
            if "Continuity" in device_type or "iphone" in name.lower():
                kind = "continuity"
            elif "BuiltIn" in device_type:
                kind = "built_in"
            elif "External" in device_type:
                kind = "external"
            else:
                kind = "camera"
            connected = bool(
                device.isConnected() if hasattr(device, "isConnected") else True
            )
            suspended = bool(
                device.isSuspended() if hasattr(device, "isSuspended") else False
            )
            descriptors.append({
                "index": index,
                "name": name,
                "kind": kind,
                "available": connected and not suspended,
            })
        return descriptors

    def _select_format(self, device: Any) -> Any:
        formats: dict[tuple[int, int], Any] = {}
        for candidate in device.formats():
            description = candidate.formatDescription()
            subtype = CM.CMFormatDescriptionGetMediaSubType(description)
            if subtype != Quartz.kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange:
                continue
            dimensions = CM.CMVideoFormatDescriptionGetDimensions(description)
            formats.setdefault((dimensions.width, dimensions.height), candidate)
        for size in self.preferred_sizes:
            if size in formats:
                self.capture_image_size_px = size
                return formats[size]
        available = ", ".join(f"{w}x{h}" for w, h in sorted(formats))
        requested = ", ".join(f"{w}x{h}" for w, h in self.preferred_sizes)
        raise RuntimeError(
            f"camera {self.index} has no requested 420v mode "
            f"({requested}); available: {available or 'none'}"
        )

    def _configure_device(self, device: Any, capture_format: Any) -> None:
        locked, error = device.lockForConfiguration_(None)
        if not locked:
            raise RuntimeError(f"could not configure camera: {error}")
        try:
            device.setActiveFormat_(capture_format)
            rate = max(1, int(round(self.fps)))
            duration = CM.CMTimeMake(1, rate)
            ranges = list(capture_format.videoSupportedFrameRateRanges())
            if any(
                float(item.minFrameRate()) <= self.fps <= float(item.maxFrameRate())
                for item in ranges
            ):
                device.setActiveVideoMinFrameDuration_(duration)
                device.setActiveVideoMaxFrameDuration_(duration)
        finally:
            device.unlockForConfiguration()

    def _start_session(self) -> None:
        devices = self._devices()
        if self.index < 0 or self.index >= len(devices):
            raise RuntimeError(
                f"camera index {self.index} is unavailable ({len(devices)} found)"
            )
        device = devices[self.index]
        capture_format = self._select_format(device)

        # Configure the device before it belongs to a capture session.  A
        # Continuity Camera can be discovered and opened while still refusing
        # lockForConfiguration once an AVCaptureDeviceInput has claimed it
        # (AVError -11817, "Cannot Use ... Camera").  Built-in cameras are more
        # permissive, which hid this ordering bug when switching back to index
        # zero.
        self._configure_device(device, capture_format)

        session = AV.AVCaptureSession.alloc().init()
        # InputPriority preserves the exact active device format selected
        # above.  Photo is the older fallback that keeps Continuity Camera's
        # full 4:3 1920x1440 mode instead of silently cropping it to 16:9.
        input_priority = getattr(AV, "AVCaptureSessionPresetInputPriority", None)
        if input_priority and session.canSetSessionPreset_(input_priority):
            session.setSessionPreset_(input_priority)
        elif session.canSetSessionPreset_(AV.AVCaptureSessionPresetPhoto):
            session.setSessionPreset_(AV.AVCaptureSessionPresetPhoto)
        camera_input, error = AV.AVCaptureDeviceInput \
            .deviceInputWithDevice_error_(device, None)
        if camera_input is None:
            raise RuntimeError(f"could not open camera {self.index}: {error}")
        if not session.canAddInput_(camera_input):
            raise RuntimeError(f"camera {self.index} cannot be added to capture")
        session.addInput_(camera_input)

        output = AV.AVCaptureVideoDataOutput.alloc().init()
        output.setAlwaysDiscardsLateVideoFrames_(True)
        output.setVideoSettings_({
            Quartz.kCVPixelBufferPixelFormatTypeKey:
                Quartz.kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
            Quartz.kCVPixelBufferWidthKey: self.capture_image_size_px[0],
            Quartz.kCVPixelBufferHeightKey: self.capture_image_size_px[1],
        })
        if not session.canAddOutput_(output):
            raise RuntimeError("native 420v video output is unavailable")
        session.addOutput_(output)

        queue = _dispatch_queue(f"hexapod.camera.{self.index}".encode("ascii"))
        delegate = _FrameDelegate.alloc().init()
        delegate.capture_owner = self
        output.setSampleBufferDelegate_queue_(delegate, queue)

        self._session = session
        self._output = output
        self._delegate = delegate
        self._queue = queue
        session.startRunning()
        if not session.isRunning():
            raise RuntimeError(f"camera {self.index} did not start")

    def _session_loop(self) -> None:
        try:
            self._start_session()
            # Continuity Camera needs a run loop even though sample delivery is
            # on a serial dispatch queue.  Keep it isolated from the web/UI
            # threads and wake frequently for prompt camera shutdown.
            while not self._stop.is_set():
                NSRunLoop.currentRunLoop().runUntilDate_(
                    NSDate.dateWithTimeIntervalSinceNow_(0.10)
                )
        except Exception as error:
            self.last_error = str(error)
            with self._condition:
                self._condition.notify_all()
        finally:
            if self._output is not None:
                self._output.setSampleBufferDelegate_queue_(None, None)
            if self._session is not None and self._session.isRunning():
                self._session.stopRunning()
            self._output = None
            self._delegate = None
            self._queue = None
            self._session = None

    def _ensure_thread(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._session_loop,
            name=f"avfoundation-camera-{self.index}",
            daemon=True,
        )
        self._thread.start()

    def _set_callback_error(self, message: str) -> None:
        self.last_error = f"native frame callback failed: {message}"
        with self._condition:
            self._condition.notify_all()

    def _accept_sample_buffer(self, sample_buffer: Any) -> None:
        pixel_buffer = CM.CMSampleBufferGetImageBuffer(sample_buffer)
        Quartz.CVPixelBufferLockBaseAddress(
            pixel_buffer, Quartz.kCVPixelBufferLock_ReadOnly
        )
        try:
            if Quartz.CVPixelBufferGetPlaneCount(pixel_buffer) != 2:
                raise RuntimeError("Continuity Camera frame is not bi-planar")

            width = int(Quartz.CVPixelBufferGetWidth(pixel_buffer))
            height = int(Quartz.CVPixelBufferGetHeight(pixel_buffer))
            y_stride = int(
                Quartz.CVPixelBufferGetBytesPerRowOfPlane(pixel_buffer, 0)
            )
            y_address = Quartz.CVPixelBufferGetBaseAddressOfPlane(
                pixel_buffer, 0
            )
            y = np.frombuffer(
                y_address.as_buffer(y_stride * height), dtype=np.uint8
            ).reshape(height, y_stride)[:, :width].copy()

            uv_width = int(
                Quartz.CVPixelBufferGetWidthOfPlane(pixel_buffer, 1)
            )
            uv_height = int(
                Quartz.CVPixelBufferGetHeightOfPlane(pixel_buffer, 1)
            )
            uv_stride = int(
                Quartz.CVPixelBufferGetBytesPerRowOfPlane(pixel_buffer, 1)
            )
            uv_address = Quartz.CVPixelBufferGetBaseAddressOfPlane(
                pixel_buffer, 1
            )
            uv = np.frombuffer(
                uv_address.as_buffer(uv_stride * uv_height), dtype=np.uint8
            ).reshape(uv_height, uv_stride)[:, :uv_width * 2].copy()
            uv = uv.reshape(uv_height, uv_width, 2)
        finally:
            Quartz.CVPixelBufferUnlockBaseAddress(
                pixel_buffer, Quartz.kCVPixelBufferLock_ReadOnly
            )

        with self._condition:
            self._latest_planes = (y, uv)
            self.capture_image_size_px = (width, height)
            self._sequence += 1
            self._condition.notify_all()

    def _frame_from_planes(
        self, y: np.ndarray, uv: np.ndarray
    ) -> np.ndarray:
        height, width = y.shape
        target_width = width
        target_height = height
        if self.processing_width and self.processing_width < width:
            target_width = self.processing_width - self.processing_width % 2
            target_height = int(round(height * target_width / width))
            target_height -= target_height % 2
            y_for_color = cv2.resize(
                y, (target_width, target_height), interpolation=cv2.INTER_AREA
            )
            uv_for_color = cv2.resize(
                uv,
                (target_width // 2, target_height // 2),
                interpolation=cv2.INTER_AREA,
            )
        else:
            y_for_color = y
            uv_for_color = uv

        self.detection_gray = y
        self.tracking_gray = y_for_color
        self.image_size_px = (target_width, target_height)
        return cv2.cvtColorTwoPlane(
            y_for_color, uv_for_color, cv2.COLOR_YUV2BGR_NV12
        )

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self.isOpened():
            self.last_error = "native AVFoundation capture is unavailable"
            return False, None
        self._ensure_thread()
        deadline = time.monotonic() + self.frame_timeout_s
        with self._condition:
            while (
                self._sequence <= self._read_sequence
                and not self._released
                and self.last_error is None
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    self.last_error = (
                        f"camera {self.index} produced no native 420v frame"
                    )
                    return False, None
                self._condition.wait(remaining)
            if self._latest_planes is None or self._sequence <= self._read_sequence:
                return False, None
            self._read_sequence = self._sequence
            y, uv = self._latest_planes
        self.last_error = None
        return True, self._frame_from_planes(y, uv)

    def capture_info(self) -> dict[str, object]:
        return {
            "backend": self.backend_name,
            "pixel_format": self.pixel_format,
            "native_luma": True,
            "capture_fps": self.fps,
            "capture_image_size_px": (
                None
                if self.capture_image_size_px is None
                else list(self.capture_image_size_px)
            ),
            "detection_image_size_px": (
                None
                if self.detection_gray is None
                else [self.detection_gray.shape[1], self.detection_gray.shape[0]]
            ),
        }

    def release(self) -> None:
        self._released = True
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self.detection_gray = None
        self.tracking_gray = None
