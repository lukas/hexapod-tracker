"""Local, read-only process manager for the iPhone zero-pose survey UI."""
from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Mapping

from .paths import REPO_ROOT
from .robot_lab import RobotLabPublisher
from .tag_survey import apply_survey_to_config, learn_zero_pose_mounts


DEFAULT_FLOOR_IDS = (100, 101, 102, 103, 104, 105, 112)
DEFAULT_ORIGIN_TAG_ID = 104
DEFAULT_MARKER_SIZE_MM = 27.0
DEFAULT_SURVEY_DIR = REPO_ROOT / "artifacts" / "zero_pose_surveys"


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
        survey_dir: Path = DEFAULT_SURVEY_DIR,
        process_factory: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
        robot_lab: RobotLabPublisher | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.survey_dir = Path(survey_dir)
        self.process_factory = process_factory
        self.robot_lab = robot_lab or RobotLabPublisher.from_env()
        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None
        self._run_dir: Path | None = None
        self._progress_path: Path | None = None
        self._result_path: Path | None = None
        self._camera_path: Path | None = None
        self._status = "idle"
        self._error: str | None = None
        self._started_unix: float | None = None
        self._completed_unix: float | None = None
        self._log_tail: deque[str] = deque(maxlen=12)
        self._robot_lab_state: dict[str, Any] = {
            "status": "ready" if self.robot_lab.configured else "not_configured",
            "url": None,
            "error": None,
        }
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        robot_tags = config.get("robot_pose", {}).get("tags", {})
        self._robot_positions = [
            {
                "position": (
                    "chassis" if str(spec.get("frame")) == "body"
                    else str(spec.get("frame", f"tag {tag_id}")).replace("_coxa", " hip").replace("_femur", " knee")
                ),
                "tag_id": int(tag_id),
                "state": "not_seen",
                "identity_reference": str(spec.get("frame")) == "L0_coxa",
                "replacement": False,
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
            "origin_tag_id": DEFAULT_ORIGIN_TAG_ID,
            "floor_tag_ids": list(DEFAULT_FLOOR_IDS),
            "marker_size_mm": DEFAULT_MARKER_SIZE_MM,
            "body_anchor_tag_id": 0,
            "leg_zero_anchor_tag_id": l0,
        }

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
            status = str(progress_state.get("status", self._status))
            progress = progress_state.get("progress") or self._empty_progress()
            process = self._process
            active = process is not None and process.poll() is None
            return {
                "available": True,
                "active": active,
                "status": status,
                "phase": progress_state.get("phase", "connect" if active else "setup"),
                "message": progress_state.get(
                    "message",
                    "Ready to start a camera-only iPhone LiDAR survey.",
                ),
                "instruction": progress_state.get(
                    "instruction",
                    "Put the stationary robot in zero pose near floor tags 100–105 and 112.",
                ),
                "error": self._error,
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
                "records": progress_state.get("records", []),
                "camera_path_m": progress_state.get("camera_path_m", []),
                "camera_position_m": progress_state.get("camera_position_m"),
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

    def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError("a zero-pose survey is already running")
            device = int(payload.get("record3d_device", self._defaults["record3d_device"]))
            origin_id = int(payload.get("origin_tag_id", self._defaults["origin_tag_id"]))
            marker_size_mm = float(payload.get("marker_size_mm", self._defaults["marker_size_mm"]))
            floor_ids = self._validated_ids(
                payload.get("floor_tag_ids", self._defaults["floor_tag_ids"])
            )
            l0_id = int(payload.get(
                "leg_zero_anchor_tag_id",
                self._defaults["leg_zero_anchor_tag_id"],
            ))
            body_id = int(payload.get("body_anchor_tag_id", self._defaults["body_anchor_tag_id"]))
            if device < 0 or device > 12:
                raise ValueError("Record3D device must be between 0 and 12")
            if origin_id < 0 or l0_id < 0 or body_id < 0:
                raise ValueError("tag IDs cannot be negative")
            if marker_size_mm <= 0.0:
                raise ValueError("marker size must be positive")

            stamp = time.strftime("%Y%m%d_%H%M%S")
            run_dir = self.survey_dir / f"zero_pose_{stamp}"
            run_dir.mkdir(parents=True, exist_ok=False)
            anchor_path = run_dir / "floor-origin.json"
            _atomic_json(anchor_path, {
                "schema_version": 1,
                "tag_family": "tag36h11",
                "marker_size_m": marker_size_mm / 1000.0,
                "world_frame": (
                    f"tag {origin_id} center; +x tag right; +y tag top; +z up from floor"
                ),
                "floor_tags": {
                    str(origin_id): {
                        "label": "survey floor origin",
                        "world_from_tag": {
                            "translation_m": [0.0, 0.0, 0.0],
                            "euler_xyz_deg": [0.0, 0.0, 0.0],
                        },
                    }
                },
            })
            result_path = run_dir / "survey.json"
            progress_path = run_dir / "progress.json"
            camera_path = run_dir / "camera.jpg"
            preview_path = run_dir / "dashboard.jpg"
            command = [
                sys.executable,
                "-m", "hexapod_tracker.zero_pose_survey",
                str(self.config_path),
                "--board", str(anchor_path),
                "--record3d-device", str(device),
                "--output", str(result_path),
                "--preview-output", str(preview_path),
                "--camera-preview-output", str(camera_path),
                "--progress-output", str(progress_path),
                "--expected-floor-ids", ",".join(str(value) for value in floor_ids),
                "--survey-marker-size-mm", str(marker_size_mm),
                "--body-anchor-tag-id", str(body_id),
                "--leg-zero-anchor-tag-id", str(l0_id),
                "--no-preview",
            ]
            self._run_dir = run_dir
            self._progress_path = progress_path
            self._result_path = result_path
            self._camera_path = camera_path
            self._status = "connecting"
            self._error = None
            self._started_unix = round(time.time(), 3)
            self._completed_unix = None
            self._log_tail.clear()
            self._robot_lab_state = {
                "status": "ready" if self.robot_lab.configured else "not_configured",
                "url": None,
                "error": None,
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
            if final_status in {"complete", "incomplete"}:
                self._status = final_status
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
            config = json.loads(self.config_path.read_text(encoding="utf-8"))
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
            robot_lab = self._publish_locked(result, output_path)
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
        result: Mapping[str, Any],
        config_path: Path,
    ) -> dict[str, Any]:
        assert self._result_path is not None
        if not self.robot_lab.configured:
            self._robot_lab_state = {
                "status": "not_configured",
                "url": None,
                "error": "HEXAPOD_LAB_TOKEN is not available to the vision server",
            }
            return dict(self._robot_lab_state)
        self._robot_lab_state = {"status": "publishing", "url": None, "error": None}
        try:
            published = self.robot_lab.publish_zero_pose_calibration(
                result_path=self._result_path,
                config_path=config_path,
                duration_seconds=float(result.get("survey", {}).get("frames", 1)) / 10.0,
            )
        except (OSError, RuntimeError, ValueError) as error:
            self._robot_lab_state = {
                "status": "failed",
                "url": None,
                "error": str(error),
            }
        else:
            self._robot_lab_state = {**published, "error": None}
        return dict(self._robot_lab_state)

    def publish_to_robot_lab(self) -> dict[str, Any]:
        with self._lock:
            result = self.result()
            if result is None or self._run_dir is None:
                raise RuntimeError("no completed survey result is available")
            config_path = self._run_dir / "tracker-config.reviewed.json"
            if not config_path.is_file():
                raise RuntimeError("review and save the calibrated config first")
            return self._publish_locked(result, config_path)
