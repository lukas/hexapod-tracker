#!/usr/bin/env python3
"""CLI for offline housing-marker pose estimation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .housing_pose import HousingPoseEstimator


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate STS3215 body pose and joint angles from AprilTag "
            "detections or precomputed rigid-frame transforms."
        )
    )
    parser.add_argument("config", type=Path, help="camera and tag-mount JSON")
    parser.add_argument("observations", type=Path,
                        help="detections/frame-transforms JSON")
    parser.add_argument("--output", type=Path,
                        help="write result JSON here instead of stdout")
    args = parser.parse_args(argv)

    config = _read_json(args.config)
    payload = _read_json(args.observations)
    estimator = HousingPoseEstimator.from_dict(config)
    encoder = payload.get("encoder_joint_deg")
    if "detections" in payload:
        result = estimator.estimate_detections(
            payload["detections"], encoder_joint_deg=encoder
        )
    elif "frame_transforms" in payload:
        result = estimator.estimate_frame_transforms(
            payload["frame_transforms"], encoder_joint_deg=encoder
        )
    else:
        raise ValueError(
            "observations JSON needs detections or frame_transforms"
        )

    rendered = json.dumps(result, indent=2, sort_keys=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0 if result.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
