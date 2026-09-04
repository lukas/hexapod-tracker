"""Local, read-only process manager for the iPhone zero-pose survey UI."""
from __future__ import annotations

import base64
from collections import deque
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Mapping

from .housing_pose import RigidTransform
from .paths import CONFIG_DIR, REPO_ROOT
from .robot_lab import RobotLabPublisher
from .tag_survey import (
    apply_survey_to_config,
    learn_zero_pose_mounts,
    merge_robot_layout_into_config,
)


DEFAULT_FLOOR_IDS = (100, 101, 102, 103, 104, 105, 112)
DEFAULT_ORIGIN_TAG_ID = 104
DEFAULT_MARKER_SIZE_MM = 27.2
DEFAULT_SURVEY_DIR = REPO_ROOT / "artifacts" / "zero_pose_surveys"
DEFAULT_ROBOT_LAYOUT = CONFIG_DIR / "hexapod-1-apriltag-layout.json"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class ZeroPoseSurveyManager:
    """Launch and observe the proven CLI without adding robot authority."""

    def __init__(
        self,
        *,
        config_path: Path,
        robot_layout_path: Path = DEFAULT_ROBOT_LAYOUT,
        survey_dir: Path = DEFAULT_SURVEY_DIR,
        process_factory: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
        robot_lab: RobotLabPublisher | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.robot_layout_path = Path(robot_layout_path)
        self.survey_dir = Path(survey_dir)
        self.process_factory = process_factory
        self.robot_lab = robot_lab or RobotLabPublisher.from_env()
        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None
        self._run_dir: Path | None = None
        self._progress_path: Path | None = None
        self._result_path: Path | None = None
        self._camera_path: Path | None = None
        self._wifi_dir: Path | None = None
        self._settings: dict[str, Any] = {}
        self._connection_mode = "usb"
        self._wifi_address = ""
        self._status = "idle"
        self._error: str | None = None
        self._started_unix: float | None = None
        self._completed_unix: float | None = None
        self._legacy_completed_run = False
        self._log_tail: deque[str] = deque(maxlen=12)
        self._robot_lab_state: dict[str, Any] = {
            "status": "ready" if self.robot_lab.configured else "not_configured",
            "url": None,
            "error": (
                None if self.robot_lab.configured
                else self.robot_lab.credential_error
            ),
            "credential_source": self.robot_lab.credential_source,
        }
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.robot_layout = json.loads(
            self.robot_layout_path.read_text(encoding="utf-8")
        )
        self._merged_config = merge_robot_layout_into_config(
            config, self.robot_layout
        )
        robot_tags = self._merged_config.get("robot_pose", {}).get("tags", {})
        self._robot_positions = [
            {
                "position": str(spec.get("position", spec.get("label", f"tag {tag_id}"))),
                "tag_id": int(tag_id),
                "state": "not_seen",
                "identity_reference": str(spec.get("frame")) == "L0_coxa",
                "replacement": False,
                "kind": spec.get("kind"),
                "surface": spec.get("surface"),
                "leg": spec.get("leg"),
                "joint": spec.get("joint"),
                "mount_side": spec.get("mount_side"),
            }
            for tag_id, spec in robot_tags.items()
        ]
        self._robot_positions.sort(
            key=lambda item: (item["position"] != "chassis", item["position"])
        )
        l0 = next(
            (item["tag_id"] for item in self._robot_positions if item["identity_reference"]),
            None,
        )
        self._defaults = {
            "record3d_device": 0,
            "connection_mode": "usb",
            "wifi_address": "",
            "origin_tag_id": DEFAULT_ORIGIN_TAG_ID,
            "floor_tag_ids": list(DEFAULT_FLOOR_IDS),
            "marker_size_mm": DEFAULT_MARKER_SIZE_MM,
            "body_anchor_tag_id": 0,
            "leg_zero_anchor_tag_id": l0,
        }
        self._restore_latest_run()

    def _restore_latest_run(self) -> None:
        """Recover the newest interrupted run after a web-server restart."""
        candidates = sorted(self.survey_dir.glob("zero_pose_*"), reverse=True)
        for run_dir in candidates:
            progress_path = run_dir / "progress.json"
            if not progress_path.is_file():
                continue
            self._run_dir = run_dir
            self._progress_path = progress_path
            self._result_path = run_dir / "survey.json"
            self._camera_path = run_dir / "camera.jpg"
            self._wifi_dir = run_dir / "wifi_frames"
            settings_path = run_dir / "session.json"
            if settings_path.is_file():
                try:
                    value = json.loads(settings_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    value = {}
                if isinstance(value, dict):
                    self._settings = value
            progress = self._read_progress()
            progress_details = progress.get("progress") or {}
            self._connection_mode = str(
                self._settings.get(
                    "connection_mode", progress.get("connection_mode", "usb")
                )
            )
            self._wifi_address = str(self._settings.get("wifi_address", ""))
            if self._result_path.is_file():
                if int(progress.get("calibration_model_version", 1)) < 2:
                    self._legacy_completed_run = True
                    self._status = "idle"
                else:
                    self._status = str(progress.get("status", "incomplete"))
            elif progress_details.get("stable_tag_ids") or progress.get("records"):
                self._status = "connection_lost"
                self._error = None
            return

    def shutdown(self) -> None:
        with self._lock:
            process = self._process
        if process is None or process.poll() is not None:
            return
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            process.terminate()

    def _empty_progress(self) -> dict[str, Any]:
        return {
            "complete": False,
            "robot_positions": [dict(item) for item in self._robot_positions],
            "ground_tag_status": [
                {"tag_id": tag_id, "state": "not_seen", "observations": 0}
                for tag_id in self._defaults["floor_tag_ids"]
            ],
            "unseen_robot_positions": [
                item["position"] for item in self._robot_positions
            ],
            "robot_positions_needing_another_view": [],
            "unseen_ground_tag_ids": list(self._defaults["floor_tag_ids"]),
            "ground_tags_needing_another_view": [],
            "stable_tag_ids": [],
            "discovered_unexpected_tag_ids": [],
        }

    def _read_progress(self) -> dict[str, Any]:
        path = self._progress_path
        if path is None or not path.is_file():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}

    def public_state(self) -> dict[str, Any]:
        with self._lock:
            progress_state = self._read_progress()
            process = self._process
            active = process is not None and process.poll() is None
            status = (
                str(progress_state.get("status", self._status))
                if active else self._status
            )
            progress = (
                self._empty_progress()
                if self._legacy_completed_run
                else progress_state.get("progress") or self._empty_progress()
            )
            can_resume = bool(
                not active
                and status in {"connection_lost", "incomplete", "failed"}
                and self._progress_path is not None
                and self._progress_path.is_file()
                and progress_state.get("records")
                and not self._legacy_completed_run
            )
            stable_count = len(progress.get("stable_tag_ids", []))
            message = progress_state.get(
                "message",
                "Ready to start a camera-only iPhone LiDAR survey.",
            )
            instruction = progress_state.get(
                "instruction",
                "Put the stationary robot in zero pose near floor tags 100–105 and 112.",
            )
            if status == "connection_lost":
                message = f"Connection lost. {stable_count} stable tags are saved."
                instruction = "Reconnect the phone, then continue this calibration."
            elif self._legacy_completed_run:
                message = (
                    "The previous result used the old 13-tag, single-floor-anchor model."
                )
                instruction = (
                    "Start a new calibration to measure 13 top/chassis tags, "
                    "24 vertical angle tags, and the seven-tag floor grid."
                )
            return {
                "available": True,
                "active": active,
                "status": status,
                "phase": (
                    "setup" if self._legacy_completed_run
                    else "connect" if status == "connection_lost"
                    else progress_state.get("phase", "connect" if active else "setup")
                ),
                "message": message,
                "instruction": instruction,
                "guidance": (
                    None if self._legacy_completed_run
                    else progress_state.get("guidance")
                ),
                "quality": (
                    None if self._legacy_completed_run
                    else progress_state.get("quality")
                ),
                "error": self._error,
                "can_resume": can_resume,
                "connection_mode": self._connection_mode,
                "wifi_address": self._wifi_address,
                "started_unix": self._started_unix,
                "completed_unix": self._completed_unix,
                "run_dir": None if self._run_dir is None else str(self._run_dir),
                "result_available": self._result_path is not None and self._result_path.is_file(),
                "reviewed_config_path": (
                    None if self._run_dir is None else str(self._run_dir / "tracker-config.reviewed.json")
                ),
                "reviewed_config_available": (
                    self._run_dir is not None
                    and (self._run_dir / "tracker-config.reviewed.json").is_file()
                ),
                "camera_frame_available": self._camera_path is not None and self._camera_path.is_file(),
                "camera_frame_version": (
                    None if self._camera_path is None or not self._camera_path.is_file()
                    else self._camera_path.stat().st_mtime_ns
                ),
                "anchor_ids": progress_state.get("anchor_ids", [self._defaults["origin_tag_id"]]),
                "alignment_count": int(progress_state.get("alignment_count", 0)),
                "anchor_frames": int(progress_state.get("anchor_frames", 8)),
                "detected_tag_ids": progress_state.get("detected_tag_ids", []),
                "elapsed_s": float(progress_state.get("elapsed_s", 0.0)),
                "frame_sequence": int(progress_state.get("frame_sequence", 0)),
                "progress": progress,
                "records": (
                    [] if self._legacy_completed_run
                    else progress_state.get("records", [])
                ),
                "camera_path_m": (
                    [] if self._legacy_completed_run
                    else progress_state.get("camera_path_m", [])
                ),
                "camera_position_m": (
                    None if self._legacy_completed_run
                    else progress_state.get("camera_position_m")
                ),
                "mount_learning": progress_state.get("mount_learning"),
                "defaults": dict(self._defaults),
                "log_tail": list(self._log_tail),
                "robot_lab": dict(self._robot_lab_state),
                "motor_commands_sent": False,
            }

    @staticmethod
    def _validated_ids(raw_ids: Any) -> list[int]:
        if not isinstance(raw_ids, list) or not raw_ids:
            raise ValueError("floor_tag_ids must be a non-empty list")
        ids = sorted({int(value) for value in raw_ids})
        if any(value < 0 for value in ids):
            raise ValueError("tag IDs cannot be negative")
        return ids

    def _validated_settings(
        self,
        payload: Mapping[str, Any],
        *,
        fallback: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        base = {**self._defaults, **(fallback or {})}
        connection_mode = str(payload.get(
            "connection_mode", base.get("connection_mode", "usb")
        )).lower()
        if connection_mode not in {"usb", "wifi"}:
            raise ValueError("connection_mode must be usb or wifi")
        settings = {
            "record3d_device": int(payload.get(
                "record3d_device", base["record3d_device"]
            )),
            "connection_mode": connection_mode,
            "wifi_address": str(payload.get(
                "wifi_address", base.get("wifi_address", "")
            )).strip(),
            "origin_tag_id": int(payload.get(
                "origin_tag_id", base["origin_tag_id"]
            )),
            "floor_tag_ids": self._validated_ids(payload.get(
                "floor_tag_ids", base["floor_tag_ids"]
            )),
            "marker_size_mm": float(payload.get(
                "marker_size_mm", base["marker_size_mm"]
            )),
            "body_anchor_tag_id": int(payload.get(
                "body_anchor_tag_id", base["body_anchor_tag_id"]
            )),
            "leg_zero_anchor_tag_id": int(payload.get(
                "leg_zero_anchor_tag_id", base["leg_zero_anchor_tag_id"]
            )),
        }
        if not 0 <= settings["record3d_device"] <= 12:
            raise ValueError("Record3D device must be between 0 and 12")
        if any(settings[key] < 0 for key in (
            "origin_tag_id", "body_anchor_tag_id", "leg_zero_anchor_tag_id"
        )):
            raise ValueError("tag IDs cannot be negative")
        if settings["marker_size_mm"] <= 0.0:
            raise ValueError("marker size must be positive")
        if connection_mode == "wifi" and not settings["wifi_address"]:
            raise ValueError("enter the iPhone address shown by Record3D")
        return settings

    def _command(self, settings: Mapping[str, Any], *, resume: bool) -> list[str]:
        assert self._run_dir is not None
        assert self._progress_path is not None
        assert self._result_path is not None
        assert self._camera_path is not None
        command = [
            sys.executable,
            "-m", "hexapod_tracker.zero_pose_survey",
            str(self.config_path),
            "--board", str(self._run_dir / "floor-origin.json"),
            "--robot-layout", str(self.robot_layout_path),
            "--output", str(self._result_path),
            "--preview-output", str(self._run_dir / "dashboard.jpg"),
            "--camera-preview-output", str(self._camera_path),
            "--progress-output", str(self._progress_path),
            "--expected-floor-ids", ",".join(
                str(value) for value in settings["floor_tag_ids"]
            ),
            "--survey-marker-size-mm", str(settings["marker_size_mm"]),
            "--body-anchor-tag-id", str(settings["body_anchor_tag_id"]),
            "--leg-zero-anchor-tag-id", str(settings["leg_zero_anchor_tag_id"]),
            "--no-preview",
        ]
        if settings["connection_mode"] == "wifi":
            assert self._wifi_dir is not None
            self._wifi_dir.mkdir(parents=True, exist_ok=True)
            command.extend(["--wifi-frame-dir", str(self._wifi_dir)])
        else:
            command.extend([
                "--record3d-device", str(settings["record3d_device"])
            ])
        if resume:
            command.extend(["--resume-progress", str(self._progress_path)])
        return command

    def _floor_board(self, settings: Mapping[str, Any]) -> dict[str, Any]:
        marker_size_m = float(settings["marker_size_mm"]) / 1000.0
        requested = {int(value) for value in settings["floor_tag_ids"]}
        origin_id = int(settings["origin_tag_id"])
        layout_tags = {
            int(item["id"]): RigidTransform.from_dict(item["world_from_tag"])
            for item in self.robot_layout.get("floor", {}).get("tags", [])
            if isinstance(item, Mapping)
        }
        if origin_id not in layout_tags:
            return {
                "schema_version": 1,
                "tag_family": "tag36h11",
                "marker_size_m": marker_size_m,
                "world_frame": f"tag {origin_id} center",
                "floor_tags": {
                    str(origin_id): {
                        "label": "survey floor origin",
                        "world_from_tag": RigidTransform.identity().to_dict(),
                    }
                },
            }
        origin_from_layout_world = layout_tags[origin_id].inverse()
        mapped = {
            str(tag_id): {
                "label": (
                    "survey floor origin" if tag_id == origin_id
                    else f"mapped floor tag {tag_id}"
                ),
                "marker_size_m": marker_size_m,
                "world_from_tag": origin_from_layout_world.compose(
                    layout_tags[tag_id]
                ).to_dict(),
            }
            for tag_id in sorted(requested | {origin_id})
            if tag_id in layout_tags
        }
        return {
            "schema_version": 1,
            "tag_family": "tag36h11",
            "marker_size_m": marker_size_m,
            "world_frame": (
                f"tag {origin_id} center; metric seven-tag floor grid"
            ),
            "floor_tags": mapped,
            "reference_grid_spacing_m": self.robot_layout.get(
                "floor", {}
            ).get("grid_spacing_m"),
        }

    def _launch(self, settings: dict[str, Any], *, resume: bool) -> dict[str, Any]:
        assert self._run_dir is not None
        command = self._command(settings, resume=resume)
        self._settings = dict(settings)
        self._connection_mode = str(settings["connection_mode"])
        self._wifi_address = str(settings["wifi_address"])
        _atomic_json(self._run_dir / "session.json", self._settings)
        self._status = "connecting"
        self._error = None
        self._started_unix = round(time.time(), 3)
        self._completed_unix = None
        self._log_tail.clear()
        self._robot_lab_state = {
            "status": "ready" if self.robot_lab.configured else "not_configured",
            "url": None,
            "error": (
                None if self.robot_lab.configured
                else self.robot_lab.credential_error
            ),
            "credential_source": self.robot_lab.credential_source,
        }
        try:
            self._process = self.process_factory(
                command,
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception:
            self._status = "failed"
            self._process = None
            raise
        threading.Thread(
            target=self._watch_process,
            args=(self._process,),
            name="zero-pose-survey",
            daemon=True,
        ).start()
        return self.public_state()

    def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError("a zero-pose survey is already running")
            settings = self._validated_settings(payload)

            stamp = time.strftime("%Y%m%d_%H%M%S")
            run_dir = self.survey_dir / f"zero_pose_{stamp}"
            run_dir.mkdir(parents=True, exist_ok=False)
            anchor_path = run_dir / "floor-origin.json"
            _atomic_json(anchor_path, self._floor_board(settings))
            result_path = run_dir / "survey.json"
            progress_path = run_dir / "progress.json"
            camera_path = run_dir / "camera.jpg"
            self._run_dir = run_dir
            self._progress_path = progress_path
            self._result_path = result_path
            self._camera_path = camera_path
            self._wifi_dir = run_dir / "wifi_frames"
            self._legacy_completed_run = False
            return self._launch(settings, resume=False)

    def resume(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError("a zero-pose survey is already running")
            if self._run_dir is None or self._progress_path is None:
                raise RuntimeError("there is no saved calibration to continue")
            progress = self._read_progress()
            if not progress.get("records"):
                raise RuntimeError("the previous calibration has no saved tag observations")
            progress_details = progress.get("progress") or {}
            fallback = {
                **self._defaults,
                **self._settings,
                "connection_mode": payload.get(
                    "connection_mode", self._connection_mode
                ),
                "wifi_address": payload.get(
                    "wifi_address", self._wifi_address
                ),
                "origin_tag_id": (progress.get("anchor_ids") or [
                    self._defaults["origin_tag_id"]
                ])[0],
                "floor_tag_ids": progress_details.get(
                    "expected_ground_tag_ids", self._defaults["floor_tag_ids"]
                ),
            }
            settings = self._validated_settings(payload, fallback=fallback)
            self._result_path = self._run_dir / "survey.json"
            self._camera_path = self._run_dir / "camera.jpg"
            self._wifi_dir = self._run_dir / "wifi_frames"
            return self._launch(settings, resume=True)

    def ingest_wifi_frame(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            process = self._process
            if (
                self._connection_mode != "wifi"
                or process is None
                or process.poll() is not None
                or self._wifi_dir is None
            ):
                raise RuntimeError("a Wi-Fi zero-pose calibration is not running")
            encoded = str(payload.get("rgbd_jpeg_base64", ""))
            raw_encoded = encoded.split(",", 1)[-1]
            try:
                image_bytes = base64.b64decode(raw_encoded, validate=True)
            except (ValueError, TypeError) as error:
                raise ValueError("invalid Wi-Fi RGB-D frame") from error
            if not image_bytes or len(image_bytes) > 6_000_000:
                raise ValueError("Wi-Fi RGB-D frame must be between 1 byte and 6 MB")
            camera_matrix = payload.get("camera_matrix")
            camera_pose = payload.get("camera_pose_xyzw_xyz")
            if not isinstance(camera_matrix, list) or len(camera_matrix) != 3:
                raise ValueError("Wi-Fi frame is missing its 3x3 camera matrix")
            if not isinstance(camera_pose, list) or len(camera_pose) != 7:
                raise ValueError("Wi-Fi frame is missing its 7-value ARKit pose")
            frame = {
                "sequence": int(payload.get("sequence", 0)),
                "captured_unix": float(payload.get("captured_unix", time.time())),
                "rgbd_jpeg_base64": raw_encoded,
                "camera_matrix": camera_matrix,
                "camera_pose_xyzw_xyz": camera_pose,
                "max_depth_m": float(payload.get("max_depth_m", 3.0)),
            }
            _atomic_json(self._wifi_dir / "latest.json", frame)
            return {"ok": True, "sequence": frame["sequence"]}

    def _watch_process(self, process: subprocess.Popen[str]) -> None:
        if process.stdout is not None:
            for raw_line in process.stdout:
                line = raw_line.strip()
                if line:
                    with self._lock:
                        self._log_tail.append(line)
        return_code = process.wait()
        with self._lock:
            progress = self._read_progress()
            final_status = str(progress.get("status", ""))
            if final_status in {"complete", "incomplete", "connection_lost"}:
                self._status = final_status
                self._error = None
            elif return_code == 0 and self._result_path is not None and self._result_path.is_file():
                self._status = "complete"
            else:
                self._status = "failed"
                self._error = self._log_tail[-1] if self._log_tail else (
                    f"survey process exited with code {return_code}"
                )
            self._completed_unix = round(time.time(), 3)

    def stop(self) -> dict[str, Any]:
        with self._lock:
            process = self._process
            if process is None or process.poll() is not None:
                return self.public_state()
            self._status = "stopping"
            process.send_signal(signal.SIGINT)
            return self.public_state()

    def latest_camera_jpeg(self) -> bytes | None:
        with self._lock:
            path = self._camera_path
            if path is None or not path.is_file():
                return None
            try:
                return path.read_bytes()
            except OSError:
                return None

    def result(self) -> dict[str, Any] | None:
        with self._lock:
            path = self._result_path
            if path is None or not path.is_file():
                return None
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return None
            return value if isinstance(value, dict) else None

    def save_reviewed_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("confirm_body_anchor_unchanged") is not True:
            raise ValueError(
                "confirm that chassis tag #0 stayed in its original mount and orientation"
            )
        with self._lock:
            result = self.result()
            if result is None:
                raise RuntimeError("no completed survey result is available")
            survey = result.get("survey") or {}
            if not survey.get("complete"):
                raise RuntimeError("finish all robot positions and floor tags before saving")
            config = merge_robot_layout_into_config(
                json.loads(self.config_path.read_text(encoding="utf-8")),
                self.robot_layout,
            )
            learned_mounts, geometry = learn_zero_pose_mounts(
                config,
                survey,
                body_anchor_tag_id=int(self._defaults["body_anchor_tag_id"]),
            )
            if not geometry.get("ok"):
                raise RuntimeError(str(geometry.get("error", "mount learning failed")))
            updated = apply_survey_to_config(config, survey, learned_mounts)
            assert self._run_dir is not None
            output_path = self._run_dir / "tracker-config.reviewed.json"
            _atomic_json(output_path, updated)
            robot_lab = self._publish_locked(output_path)
            return {
                "ok": True,
                "config_path": str(output_path),
                "source_config_changed": False,
                "motor_commands_sent": False,
                "mount_count": len(learned_mounts),
                "robot_lab": robot_lab,
            }

    def _publish_locked(
        self,
        config_path: Path,
    ) -> dict[str, Any]:
        assert self._result_path is not None
        if not self.robot_lab.configured:
            self._robot_lab_state = {
                "status": "not_configured",
                "url": None,
                "error": self.robot_lab.credential_error or (
                    "Robot Lab token is not available to the vision server"
                ),
                "credential_source": self.robot_lab.credential_source,
            }
            return dict(self._robot_lab_state)
        self._robot_lab_state = {
            "status": "publishing",
            "url": None,
            "error": None,
            "credential_source": self.robot_lab.credential_source,
        }
        try:
            published = self.robot_lab.publish_zero_pose_calibration(
                result_path=self._result_path,
                config_path=config_path,
            )
        except (OSError, RuntimeError, ValueError) as error:
            self._robot_lab_state = {
                "status": "failed",
                "url": None,
                "error": str(error),
                "credential_source": self.robot_lab.credential_source,
            }
        else:
            self._robot_lab_state = {
                **published,
                "error": None,
                "credential_source": self.robot_lab.credential_source,
            }
        return dict(self._robot_lab_state)

    def publish_to_robot_lab(self) -> dict[str, Any]:
        with self._lock:
            result = self.result()
            if result is None or self._run_dir is None:
                raise RuntimeError("no completed survey result is available")
            config_path = self._run_dir / "tracker-config.reviewed.json"
            if not config_path.is_file():
                raise RuntimeError("review and save the calibrated config first")
            return self._publish_locked(config_path)
