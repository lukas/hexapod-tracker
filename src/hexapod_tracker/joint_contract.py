"""Minimal shared coordinate contract for tracker artifacts."""

from typing import Any, Mapping


FRAME_ROBOT_ABS = "robot_abs"
JOINT_CONTRACT = "robot_abs_tibia_v2"


def require_robot_abs_joint_frame(
    meta: Mapping[str, Any] | None,
    *,
    source: str = "artifact",
) -> str:
    """Reject telemetry that uses a different joint-coordinate contract."""
    raw = None if meta is None else meta.get("joint_frame")
    contract = None if meta is None else meta.get("joint_contract")
    if raw != FRAME_ROBOT_ABS or contract != JOINT_CONTRACT:
        raise ValueError(
            f"{source}: expected joint_frame={FRAME_ROBOT_ABS!r} and "
            f"joint_contract={JOINT_CONTRACT!r}, got {raw!r}/{contract!r}"
        )
    return FRAME_ROBOT_ABS
