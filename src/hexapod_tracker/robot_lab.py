"""Publish completed camera calibrations to the authenticated Robot Lab."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping
from urllib import error, request


DEFAULT_ROBOT_LAB_URL = "https://robot-lab.cwd1f0-new-cluster.coreweave.app"


class RobotLabHTTPError(RuntimeError):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"Robot Lab returned HTTP {status}: {detail[:300]}")
        self.status = status


class RobotLabPublisher:
    """Small stdlib client for Robot Lab's completed-result/artifact API."""

    def __init__(self, base_url: str, token: str | None, timeout_s: float = 20.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_s = timeout_s

    @classmethod
    def from_env(cls) -> "RobotLabPublisher":
        return cls(
            os.getenv("HEXAPOD_LAB_URL", DEFAULT_ROBOT_LAB_URL),
            os.getenv("HEXAPOD_LAB_TOKEN"),
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
            raise RuntimeError("HEXAPOD_LAB_TOKEN is not available to the vision server")
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
        duration_seconds: float,
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
        try:
            saved = self._request_json("POST", "/api/calibrations", payload={
                "robot_id": "hexapod-1",
                "scope": "combined",
                "source": "iphone_lidar_zero_pose_survey",
                "configuration": configuration,
                "survey": result,
                "summary": summary,
            })
        except RobotLabHTTPError as caught:
            if caught.status not in {404, 405}:
                raise
        else:
            return {
                "status": "published",
                "calibration_id": saved.get("id"),
                "url": saved.get("url") or f"{self.base_url}/calibrations/{saved.get('id')}",
                "artifacts": ["configuration", "survey"],
                "transport": "calibration_config",
            }

        # Compatibility with Robot Lab versions deployed before the dedicated
        # calibration-config endpoint: register a durable result and attach
        # the same two JSON documents as immutable artifacts.
        registered = self._request_json("POST", "/api/results", payload={
            "name": "iPhone LiDAR zero-pose calibration",
            "description": (
                "Surveyed AprilTag identities, 6-DoF mounts, orientations, and floor distances"
            ),
            "duration_seconds": max(0.001, float(duration_seconds)),
            "parameters": {
                "kind": "apriltag_zero_pose_calibration",
                "schema_version": result.get("schema_version"),
                "robot_position_count": len(positions),
                "floor_tag_ids": survey.get("expected_ground_tag_ids", []),
                "stable_tag_count": survey.get("stable_tag_count"),
                "leg_zero_reference": result.get("leg_zero_reference"),
                "motor_commands_sent": False,
            },
            "status": "succeeded",
            "summary_markdown": summary,
        })
        experiment_id = str(registered["id"])
        uploaded: list[str] = []
        for filename, path in (
            ("zero-pose-survey.json", result_path),
            ("apriltag-tracker-config.json", config_path),
        ):
            self._request_json(
                "PUT",
                f"/api/experiments/{experiment_id}/artifacts/{filename}",
                body=path.read_bytes(),
            )
            uploaded.append(filename)
        return {
            "status": "published",
            "experiment_id": experiment_id,
            "url": f"{self.base_url}/experiments/{experiment_id}",
            "artifacts": uploaded,
            "transport": "result_artifacts_compatibility",
        }
