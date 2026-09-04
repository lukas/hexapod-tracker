#!/usr/bin/env python3
"""Calibrate a fixed iPhone RGB-D view against mapped AprilTags.

The live path consumes Record3D's USB stream.  The offline path consumes NPZ
frames with ``rgb``, ``depth``, ``camera_matrix`` and optional ``confidence``
arrays.  Both paths feed the same tested RGB-D optimizer.
"""
from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time
from typing import Any

import cv2
import numpy as np

from .apriltag_vision import detect_tag_corners
from .housing_pose import RigidTransform
from .rgbd_calibration import (
    RGBDCalibrationError,
    RGBDCalibrationOptions,
    RGBDWorldReference,
    average_fixed_camera_calibration,
    refine_world_reference_with_depth,
)


@dataclass(frozen=True)
class RGBDFrame:
    rgb_bgr: np.ndarray
    depth_m: np.ndarray
    confidence: np.ndarray | None
    camera_matrix: np.ndarray
    source_label: str
    arkit_world_from_opengl_camera: RigidTransform | None = None


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _calibration_target(
    tracker_config: dict[str, Any],
    board_manifest: dict[str, Any] | None,
) -> tuple[dict[int, RigidTransform], float]:
    source = tracker_config if board_manifest is None else board_manifest
    raw_tags = source.get("floor_tags", {})
    if not raw_tags:
        raise ValueError("calibration target does not define floor_tags")
    floor_tags = {
        int(raw_id): RigidTransform.from_dict(
            spec.get("world_from_tag", spec)
        )
        for raw_id, spec in raw_tags.items()
    }
    marker_size_m = float(source.get(
        "floor_marker_size_m",
        source.get("marker_size_m", tracker_config.get("marker_size_m", 0.0)),
    ))
    if not np.isfinite(marker_size_m) or marker_size_m <= 0.0:
        raise ValueError("calibration marker size must be positive")
    return floor_tags, marker_size_m


def _npz_frames(directory: Path) -> Iterator[RGBDFrame]:
    paths = sorted(directory.glob("*.npz"))
    if not paths:
        raise ValueError(f"no .npz frames found in {directory}")
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            missing = [
                name for name in ("rgb", "depth", "camera_matrix")
                if name not in data
            ]
            if missing:
                raise ValueError(f"{path} is missing arrays: {', '.join(missing)}")
            rgb = np.asarray(data["rgb"])
            if rgb.ndim == 2:
                rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
            elif rgb.ndim == 3 and rgb.shape[2] == 3:
                # Offline fixtures are specified as BGR to match OpenCV.
                rgb_bgr = rgb.copy()
            else:
                raise ValueError(f"{path}: rgb must be HxW or HxWx3")
            confidence = (
                None if "confidence" not in data
                else np.asarray(data["confidence"]).copy()
            )
            arkit_pose = None
            if "camera_pose_xyzw_xyz" in data:
                pose = np.asarray(
                    data["camera_pose_xyzw_xyz"], dtype=float
                ).reshape(7)
                arkit_pose = RigidTransform.from_dict({
                    "quaternion_xyzw": pose[:4],
                    "translation_m": pose[4:],
                })
            yield RGBDFrame(
                rgb_bgr=rgb_bgr,
                depth_m=np.asarray(data["depth"], dtype=np.float32).copy(),
                confidence=confidence,
                camera_matrix=np.asarray(
                    data["camera_matrix"], dtype=float
                ).reshape(3, 3).copy(),
                source_label=path.name,
                arkit_world_from_opengl_camera=arkit_pose,
            )


class Record3DReader:
    """Copy coherent frames from Record3D's callback-driven USB API."""

    def __init__(self, device_index: int) -> None:
        try:
            from record3d import Record3DStream
        except ImportError as error:
            raise RuntimeError(
                "Record3D support is not installed; run "
                "`uv run --with cmake uv sync --extra dev --extra rgbd`"
            ) from error
        devices = Record3DStream.get_connected_devices()
        if not devices:
            raise RuntimeError(
                "no Record3D USB device found; connect the iPhone, open "
                "Record3D, and enable USB Streaming mode"
            )
        if device_index < 0 or device_index >= len(devices):
            raise RuntimeError(
                f"device index {device_index} is unavailable; found "
                f"{len(devices)} device(s)"
            )
        self._event = threading.Event()
        self._stopped = threading.Event()
        self._closed = False
        self._session = Record3DStream()
        self._session.on_new_frame = self._event.set
        self._session.on_stream_stopped = self._stopped.set
        if not self._session.connect(devices[device_index]):
            raise RuntimeError(f"could not connect to Record3D device {device_index}")
        self.device = devices[device_index]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        disconnect = getattr(self._session, "disconnect", None)
        if disconnect is not None:
            disconnect()
            # Record3D 1.4.1 can leave its native callback thread unwinding
            # briefly after Disconnect returns.  Let on_stream_stopped arrive
            # and the thread exit before Python destroys its callback mutexes.
            self._stopped.wait(1.0)
            time.sleep(0.1)

    def next_frame(self, timeout_s: float = 5.0) -> RGBDFrame:
        if not self._event.wait(timeout_s):
            if self._stopped.is_set():
                raise RuntimeError("Record3D stream stopped")
            raise TimeoutError("timed out waiting for an RGB-D frame")
        # Match Record3D's reference client: copy immediately in the consumer
        # thread, then clear the notification.  Do not process in the callback.
        rgb = self._session.get_rgb_frame()
        depth = self._session.get_depth_frame()
        confidence = self._session.get_confidence_frame()
        coefficients = self._session.get_intrinsic_mat()
        camera_pose = self._session.get_camera_pose()
        device_type = int(self._session.get_device_type())
        self._event.clear()
        if device_type != 1:
            raise RuntimeError(
                "Record3D is streaming the front TrueDepth camera; select "
                "the rear LiDAR camera for this calibration"
            )
        matrix = np.asarray([
            [coefficients.fx, 0.0, coefficients.tx],
            [0.0, coefficients.fy, coefficients.ty],
            [0.0, 0.0, 1.0],
        ], dtype=float)
        arkit_pose = RigidTransform.from_dict({
            "quaternion_xyzw": [
                camera_pose.qx,
                camera_pose.qy,
                camera_pose.qz,
                camera_pose.qw,
            ],
            "translation_m": [camera_pose.tx, camera_pose.ty, camera_pose.tz],
        })
        return RGBDFrame(
            rgb_bgr=cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR),
            depth_m=np.asarray(depth, dtype=np.float32),
            confidence=(
                None if np.asarray(confidence).size == 0
                else np.asarray(confidence)
            ),
            camera_matrix=matrix,
            source_label="Record3D USB",
            arkit_world_from_opengl_camera=arkit_pose,
        )


def _annotate(
    frame: RGBDFrame,
    detections: Sequence[Any],
    target_ids: set[int],
    observation: RGBDWorldReference | None,
    error: str | None,
) -> np.ndarray:
    output = frame.rgb_bgr.copy()
    for item in detections:
        points = np.rint(item.corners_px).astype(np.int32)
        target = item.tag_id in target_ids
        color = (30, 220, 30) if target else (0, 180, 255)
        cv2.polylines(output, [points], True, color, 3, cv2.LINE_AA)
        center = tuple(np.rint(item.center_px).astype(int))
        cv2.putText(
            output,
            str(item.tag_id),
            (center[0] + 8, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
    if observation is not None:
        message = (
            f"ACCEPT tags={list(observation.floor_tag_ids)} "
            f"rgb={observation.reprojection_rms_px:.2f}px "
            f"depth={observation.depth_plane_rms_m * 1000.0:.1f}mm"
        )
        color = (30, 220, 30)
    else:
        message = f"WAIT {error or 'no observation'}"
        color = (0, 180, 255)
    cv2.rectangle(output, (0, 0), (output.shape[1], 48), (10, 10, 10), -1)
    cv2.putText(
        output,
        message[:110],
        (14, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        color,
        2,
        cv2.LINE_AA,
    )
    return output


def _write_results(
    output_path: Path,
    calibration: Any,
    *,
    tracker_config_path: Path,
    board_path: Path | None,
    updated_config_path: Path | None,
    tracker_config: dict[str, Any],
    board_manifest: dict[str, Any] | None,
    frame_summaries: Sequence[dict[str, Any]],
) -> None:
    payload = calibration.to_dict()
    payload.update({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_tracker_config": str(tracker_config_path),
        "source_board_manifest": None if board_path is None else str(board_path),
        "frame_observations": list(frame_summaries),
        "motor_commands_sent": False,
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if updated_config_path is None:
        return
    updated = json.loads(json.dumps(tracker_config))
    updated["camera"] = payload["camera"]
    updated["fixed_camera_world_reference"] = payload["world_from_camera"]
    updated["rgbd_calibration"] = {
        "created_utc": payload["created_utc"],
        "source": str(output_path),
        **payload["quality"],
    }
    if board_manifest is not None:
        updated["floor_tags"] = board_manifest["floor_tags"]
        updated["floor_marker_size_m"] = float(board_manifest["marker_size_m"])
    updated_config_path.parent.mkdir(parents=True, exist_ok=True)
    updated_config_path.write_text(
        json.dumps(updated, indent=2) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Refine the hexapod tracker's mapped-AprilTag camera pose with "
            "registered iPhone LiDAR depth. The phone and board must remain fixed."
        )
    )
    parser.add_argument("tracker_config", type=Path)
    parser.add_argument("--board", type=Path, help="optional generated board manifest")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--npz-dir", type=Path, help="offline RGB-D fixture directory")
    source.add_argument("--record3d-device", type=int, default=0)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--min-tags", type=int, default=2)
    parser.add_argument("--min-confidence", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--updated-config",
        type=Path,
        help="write a tracker config using measured intrinsics and this target",
    )
    parser.add_argument("--preview-output", type=Path)
    parser.add_argument("--preview", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.frames < 8:
        raise SystemExit("--frames must be at least 8")
    if args.min_tags < 1:
        raise SystemExit("--min-tags must be at least 1")
    tracker_config = _load_json(args.tracker_config)
    board_manifest = None if args.board is None else _load_json(args.board)
    floor_tags, floor_marker_size_m = _calibration_target(
        tracker_config, board_manifest
    )
    options = RGBDCalibrationOptions(min_confidence=args.min_confidence)
    target_ids = set(floor_tags)
    observations: list[RGBDWorldReference] = []
    frame_summaries: list[dict[str, Any]] = []
    attempts = 0
    last_preview: np.ndarray | None = None
    previous_world_from_camera: RigidTransform | None = None
    reader: Record3DReader | None = None
    if args.npz_dir is not None:
        frames: Iterator[RGBDFrame] = _npz_frames(args.npz_dir)
    else:
        reader = Record3DReader(args.record3d_device)

        def live_frames() -> Iterator[RGBDFrame]:
            assert reader is not None
            while True:
                yield reader.next_frame()

        frames = live_frames()

    try:
        for frame in frames:
            attempts += 1
            detections = detect_tag_corners(frame.rgb_bgr)
            visible = [item for item in detections if item.tag_id in target_ids]
            observation = None
            error = None
            if len(visible) < args.min_tags:
                error = f"visible target tags {len(visible)}/{args.min_tags}"
            else:
                try:
                    height, width = frame.rgb_bgr.shape[:2]
                    observation = refine_world_reference_with_depth(
                        visible,
                        floor_tags,
                        frame.camera_matrix,
                        np.zeros(5),
                        frame.depth_m,
                        image_size_px=(width, height),
                        confidence=frame.confidence,
                        marker_size_m=floor_marker_size_m,
                        previous_world_from_camera=previous_world_from_camera,
                        options=options,
                    )
                    observations.append(observation)
                    previous_world_from_camera = observation.world_from_camera
                    frame_summaries.append({
                        "source": frame.source_label,
                        **observation.to_dict(),
                    })
                except (RGBDCalibrationError, cv2.error, ValueError) as exc:
                    error = str(exc)
            last_preview = _annotate(
                frame, detections, target_ids, observation, error
            )
            if args.preview:
                cv2.imshow("hexapod RGB-D calibration", last_preview)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            state = "accepted" if observation is not None else f"rejected: {error}"
            print(f"frame {attempts}: {state} ({len(observations)}/{args.frames})")
            if len(observations) >= args.frames:
                break
            if args.npz_dir is None and attempts >= args.frames * 10:
                break
    finally:
        if reader is not None:
            reader.close()
        if args.preview:
            cv2.destroyAllWindows()

    calibration = average_fixed_camera_calibration(
        observations,
        input_frames=attempts,
        min_frames=min(8, args.frames),
    )
    _write_results(
        args.output,
        calibration,
        tracker_config_path=args.tracker_config,
        board_path=args.board,
        updated_config_path=args.updated_config,
        tracker_config=tracker_config,
        board_manifest=board_manifest,
        frame_summaries=frame_summaries,
    )
    if args.preview_output is not None and last_preview is not None:
        args.preview_output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.preview_output), last_preview):
            raise RuntimeError(f"could not write {args.preview_output}")
    print(f"wrote calibration: {args.output}")
    if args.updated_config is not None:
        print(f"wrote tracker config: {args.updated_config}")
    print(
        f"quality: {calibration.accepted_frames} frames, "
        f"{calibration.translation_spread_mm:.2f} mm translation spread, "
        f"{calibration.rotation_spread_deg:.3f} deg rotation spread"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
