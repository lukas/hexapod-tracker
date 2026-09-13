"""Which attached cameras form the rig, and which slot each one gets.

One function decides this for everyone: ``tools/camera_service.sh`` calls it
to build the server's arguments at every launch, and the running server calls
it (in a fresh process, because AVFoundation's device list goes stale inside
a long-running one) to notice that the rig has changed and hand itself back
to launchd for a relaunch.

Rules, in order:

* Only external cameras count. The Studio Display and a Continuity iPhone
  are not part of the rig.
* Anything listed in the exclude set is skipped.
* Two cameras on one USB host controller cannot both stream (see
  ``docs/LLM_HANDOFF.md``, "USB bus bandwidth, not camera count, sets the
  ceiling"), so only one per controller is pinned. The winner is the camera
  with an intrinsics entry, then the lowest stable id, so the choice does not
  flip with enumeration order. The loser is reported so the operator can move
  it or pick the other one with ``CAMERA_SERVICE_EXCLUDE``.
* Slots are numbered 0..n-1 in the order the survivors enumerate. Slot
  numbers are not identities; pin and calibrate by stable id.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import sys
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class RigCamera:
    slot: int
    stable_id: str
    name: str


@dataclass
class Rig:
    cameras: list[RigCamera] = field(default_factory=list)
    excluded: list[tuple[str, str, str]] = field(default_factory=list)  # (stable_id, name, reason)

    @property
    def stable_ids(self) -> set[str]:
        return {camera.stable_id for camera in self.cameras}


def usb_controller(stable_id: str) -> str | None:
    """The USB host controller a camera hangs off, from its AVFoundation id.

    A UVC camera's ``uniqueID`` is ``0x`` + 32-bit USB location + vendor id +
    product id; the top byte of the location names the controller. Anything
    that is not such an id (a Continuity Camera UUID) has no controller.
    """
    text = str(stable_id).strip().lower()
    if not text.startswith("0x"):
        return None
    try:
        value = int(text, 16)
    except ValueError:
        return None
    location = value >> 32
    if location == 0:
        return None
    return f"0x{location >> 24:02x}"


def choose_rig(
    devices: Iterable[dict[str, Any]],
    *,
    exclude: Iterable[str] = (),
    prefer: Iterable[str] = (),
) -> Rig:
    """Pick the cameras to run from AVFoundation device descriptors.

    ``prefer`` is the set of stable ids that have an intrinsics entry; a
    calibrated camera wins a controller conflict over an uncalibrated one.
    """
    excluded_ids = {str(value).strip() for value in exclude if str(value).strip()}
    preferred = {str(value).strip() for value in prefer}
    rig = Rig()
    candidates: list[dict[str, Any]] = []
    for item in devices:
        stable_id = str(item.get("stable_id", "")).strip()
        name = str(item.get("name", "")).strip()
        if not item.get("available", True) or item.get("kind") != "external":
            continue
        if "studio display" in name.lower():
            continue
        if stable_id in excluded_ids:
            rig.excluded.append((stable_id, name, "listed in CAMERA_SERVICE_EXCLUDE"))
            continue
        candidates.append({"stable_id": stable_id, "name": name})

    winners: dict[str | None, dict[str, Any]] = {}
    for item in candidates:
        controller = usb_controller(item["stable_id"])
        if controller is None:
            winners[f"id:{item['stable_id']}"] = item  # no controller known: never conflicts
            continue
        current = winners.get(controller)
        if current is None or _beats(item, current, preferred):
            if current is not None:
                rig.excluded.append(_conflict(current, item, controller))
            winners[controller] = item
        else:
            rig.excluded.append(_conflict(item, current, controller))

    chosen = {item["stable_id"] for item in winners.values()}
    slot = 0
    for item in candidates:  # keep enumeration order for the survivors
        if item["stable_id"] in chosen:
            rig.cameras.append(RigCamera(slot, item["stable_id"], item["name"]))
            slot += 1
    return rig


def _beats(item: dict[str, Any], current: dict[str, Any], preferred: set[str]) -> bool:
    a = (item["stable_id"] in preferred, item["stable_id"])
    b = (current["stable_id"] in preferred, current["stable_id"])
    if a[0] != b[0]:
        return a[0]
    return a[1] < b[1]


def _conflict(loser: dict[str, Any], winner: dict[str, Any], controller: str) -> tuple[str, str, str]:
    return (
        loser["stable_id"],
        loser["name"],
        f"shares USB controller {controller} with {winner['name']} "
        f"({winner['stable_id']}); move it to another port or exclude the other",
    )


def intrinsics_ids(calibration_path: str | None) -> set[str]:
    """Stable ids that have an entry in an identity-keyed intrinsics file."""
    if not calibration_path:
        return set()
    try:
        with open(calibration_path, encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, ValueError):
        return set()
    return {
        str(spec.get("stable_id")).strip()
        for spec in (document.get("cameras") or {}).values()
        if isinstance(spec, dict) and spec.get("stable_id")
    }


def discover_rig(*, exclude: Iterable[str] = (), calibration_path: str | None = None) -> Rig:
    """Enumerate the attached cameras and apply :func:`choose_rig`.

    Call this in a short-lived process: a long-running process's AVFoundation
    device list has been seen to keep an unplugged camera and to miss a
    re-enumerated one.
    """
    from .avfoundation_capture import AVFoundationYuvCapture

    devices = AVFoundationYuvCapture.device_descriptors()
    return choose_rig(devices, exclude=exclude, prefer=intrinsics_ids(calibration_path))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m hexapod_tracker.rig",
        description="print the cameras that make up the rig, one per slot",
    )
    parser.add_argument("--exclude", default="", help="comma-separated stable ids to skip")
    parser.add_argument("--calibration", default=None, help="identity-keyed intrinsics file; calibrated cameras win controller conflicts")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)
    exclude = [value for value in args.exclude.split(",") if value.strip()]
    try:
        rig = discover_rig(exclude=exclude, calibration_path=args.calibration)
    except Exception as error:  # never emit a half-built camera list
        print(f"camera discovery failed: {error}", file=sys.stderr)
        return 1
    for stable_id, name, reason in rig.excluded:
        print(f"excluded {name} ({stable_id}): {reason}", file=sys.stderr)
    if args.json:
        print(json.dumps({
            "cameras": [camera.__dict__ for camera in rig.cameras],
            "excluded": [{"stable_id": s, "name": n, "reason": r} for s, n, r in rig.excluded],
        }))
    else:
        for camera in rig.cameras:
            print(f"{camera.slot}\t{camera.stable_id}\t{camera.name}")
    if not rig.cameras:
        print("no rig cameras found", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
