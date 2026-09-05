"""Publish completed camera calibrations to the authenticated Robot Lab."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping
from urllib import error, request


DEFAULT_ROBOT_LAB_URL = "https://robot-lab.cwd1f0-new-cluster.coreweave.app"


class RobotLabHTTPError(RuntimeError):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"Robot Lab returned HTTP {status}: {detail[:300]}")
        self.status = status


class RobotLabPublisher:
    """Small client for Robot Lab's versioned calibration API."""

    def __init__(
        self,
        base_url: str,
        token: str | None,
        timeout_s: float = 20.0,
        *,
        credential_source: str | None = None,
        credential_error: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_s = timeout_s
        self.credential_source = credential_source
        self.credential_error = credential_error

    @staticmethod
    def _token_from_text(value: str) -> str | None:
        candidates: list[str] = []
        for raw_line in value.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            for prefix in (
                "HEXIPOD_LAB_TOKEN=", "HEXAPOD_LAB_TOKEN=", "token:", "Token:"
            ):
                if line.startswith(prefix):
                    line = line[len(prefix):].strip().strip('"\'')
                    break
            if line.lower().startswith("bearer "):
                line = line[7:].strip()
            if len(line) >= 16 and not any(character.isspace() for character in line):
                candidates.append(line)
        return candidates[0] if len(candidates) == 1 else None

    @classmethod
    def _token_from_file(cls, path: Path) -> str | None:
        if not path.is_file():
            return None
        if path.suffix.lower() == ".rtf" and Path("/usr/bin/textutil").is_file():
            completed = subprocess.run(
                ["/usr/bin/textutil", "-convert", "txt", "-stdout", str(path)],
                check=False,
                capture_output=True,
                text=True,
                timeout=3.0,
            )
            if completed.returncode != 0:
                return None
            return cls._token_from_text(completed.stdout)
        return cls._token_from_text(path.read_text(encoding="utf-8"))

    @classmethod
    def from_env(cls) -> "RobotLabPublisher":
        token = os.getenv("HEXIPOD_LAB_TOKEN") or os.getenv("HEXAPOD_LAB_TOKEN")
        source = "environment" if token else None
        diagnostic: str | None = None
        raw_token_path = os.getenv("HEXIPOD_LAB_TOKEN_FILE")
        token_paths = (
            [Path(raw_token_path).expanduser()]
            if raw_token_path else [
                Path.home() / "Documents" / "hexapod.rtf",
                Path.home() / "Library" / "Mobile Documents"
                / "com~apple~TextEdit" / "Documents" / "hexapod.rtf",
            ]
        )
        for token_path in token_paths:
            if token or not token_path.is_file():
                continue
            try:
                token = cls._token_from_file(token_path)
            except (OSError, subprocess.SubprocessError, UnicodeError):
                token = None
            if token:
                source = "credential_file"
            else:
                diagnostic = f"could not read one token from {token_path}"
        if not token:
            op_path = shutil.which("op")
            op_ref = os.getenv(
                "HEXIPOD_LAB_TOKEN_OP_REF",
                "op://Private/Hexapod Lab API/credential",
            )
            if op_path:
                try:
                    completed = subprocess.run(
                        [op_path, "read", op_ref],
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=4.0,
                    )
                except (OSError, subprocess.SubprocessError):
                    completed = None
                if completed is not None and completed.returncode == 0:
                    token = completed.stdout.strip() or None
                if token:
                    source = "1password"
                elif diagnostic is None:
                    diagnostic = "1Password CLI could not read the configured item"
            elif diagnostic is None:
                diagnostic = (
                    "1Password CLI is not installed and no credential file "
                    "was found in Documents or TextEdit Documents"
                )
        return cls(
            os.getenv("HEXAPOD_LAB_URL", DEFAULT_ROBOT_LAB_URL),
            token,
            credential_source=source,
            credential_error=diagnostic,
        )

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.token)

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        payload: Mapping[str, Any] | None = None,
        body: bytes | None = None,
        content_type: str = "application/json",
    ) -> dict[str, Any]:
        if not self.token:
            raise RuntimeError("Robot Lab token is not available to the vision server")
        data = body
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
        target = path if path.startswith("http") else f"{self.base_url}{path}"
        outgoing = request.Request(
            target,
            method=method,
            data=data,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": content_type,
            },
        )
        try:
            with request.urlopen(outgoing, timeout=self.timeout_s) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as caught:
            detail = caught.read().decode("utf-8", errors="replace")
            raise RobotLabHTTPError(caught.code, detail) from caught
        except error.URLError as caught:
            raise RuntimeError(f"Robot Lab could not be reached: {caught.reason}") from caught
        if not isinstance(decoded, dict):
            raise RuntimeError("Robot Lab returned an unexpected response")
        return decoded

    def publish_zero_pose_calibration(
        self,
        *,
        result_path: Path,
        config_path: Path,
    ) -> dict[str, Any]:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        configuration = json.loads(config_path.read_text(encoding="utf-8"))
        survey = result.get("survey") or {}
        positions = survey.get("robot_positions") or []
        replacements = [
            {
                "position": item.get("position"),
                "tag_id": item.get("tag_id"),
            }
            for item in positions if item.get("replacement")
        ]
        summary = (
            "# iPhone LiDAR zero-pose calibration\n\n"
            f"Recorded {len(positions)} robot mount positions and "
            f"{len(survey.get('ground_tag_status') or [])} floor tags. "
            "The robot remained stationary and no motor commands were sent.\n\n"
            f"Replacement assignments: {replacements or 'none'}.\n"
        )
        saved = self._request_json("POST", "/api/calibrations", payload={
            "robot_id": "hexapod-1",
            "scope": "combined",
            "source": "iphone_lidar_zero_pose_survey",
            "configuration": configuration,
            "survey": result,
            "summary": summary,
        })
        return {
            "status": "published",
            "calibration_id": saved.get("id"),
            "url": saved.get("url") or f"{self.base_url}/calibrations/{saved.get('id')}",
            "artifacts": ["configuration", "survey"],
            "transport": "calibration_config",
        }

    def publish_lab_camera_calibration(
        self,
        calibration_path: Path,
    ) -> dict[str, Any]:
        """Publish one quality-gated fixed-camera calibration revision."""
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        quality = calibration.get("quality") or {}
        if quality.get("passed") is not True:
            raise ValueError("only a passed lab camera calibration may be published")
        saved = self._request_json(
            "POST",
            "/api/camera-calibrations",
            payload={
                "lab_id": calibration["lab_id"],
                "camera_id": calibration["camera_id"],
                "camera_name": calibration["camera_name"],
                "camera_kind": calibration.get("camera_kind", "camera"),
                "source": calibration.get(
                    "source", "hexapod_tracker_fixed_camera_workflow"
                ),
                "capture_mode": calibration["capture_mode"],
                "intrinsics": calibration["intrinsics"],
                "extrinsics": calibration["extrinsics"],
                "floor_map_sha256": calibration["extrinsics"].get(
                    "floor_map_sha256"
                ),
                "quality": quality,
                "evidence": calibration.get("evidence", {}),
            },
        )
        calibration_id = saved.get("id")
        return {
            "status": "published",
            "camera_calibration_id": calibration_id,
            "url": saved.get("url") or (
                f"{self.base_url}/camera-calibrations/{calibration_id}"
            ),
            "transport": "camera_calibration_revision",
        }
