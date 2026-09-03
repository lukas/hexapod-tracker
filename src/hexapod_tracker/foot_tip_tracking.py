"""Track the STS3215's red boot tips in overhead phone video.

The detector deliberately uses the robot's geometry in the image rather than
assuming that every red object is a foot.  Each candidate is associated with
the outward ray from the chassis through that leg's femur tag.  Short missed
detections use optical flow and then a bounded constant-velocity prediction;
every result records which source was used.

This module performs no robot I/O.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


LEG_COUNT = 6


@dataclass(frozen=True)
class FootTipObservation:
    leg: int
    point_px: np.ndarray
    component_center_px: np.ndarray
    source: str
    occlusion_age_frames: int
    confidence: float
    velocity_px_per_frame: np.ndarray

    def to_dict(self) -> dict:
        return {
            "leg": self.leg,
            "point_px": [round(float(v), 3) for v in self.point_px],
            "component_center_px": [
                round(float(v), 3) for v in self.component_center_px
            ],
            "source": self.source,
            "occlusion_age_frames": self.occlusion_age_frames,
            "confidence": round(float(self.confidence), 3),
            "velocity_px_per_frame": [
                round(float(v), 3) for v in self.velocity_px_per_frame
            ],
        }


@dataclass(frozen=True)
class _RedCandidate:
    center_px: np.ndarray
    pixels_xy: np.ndarray
    area_px: float


@dataclass
class _FootState:
    point_px: np.ndarray
    center_px: np.ndarray
    velocity_px_per_frame: np.ndarray
    age: int
    gray: np.ndarray


def _red_candidates(image: np.ndarray, tag_scale_px: float) -> list[_RedCandidate]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    low = cv2.inRange(hsv, (0, 90, 45), (13, 255, 255))
    high = cv2.inRange(hsv, (168, 90, 45), (179, 255, 255))
    mask = cv2.bitwise_or(low, high)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8)
    )
    minimum_area = 0.12 * tag_scale_px * tag_scale_px
    maximum_area = 1.00 * tag_scale_px * tag_scale_px
    minimum_dimension = 0.25 * tag_scale_px
    maximum_dimension = 1.70 * tag_scale_px
    candidates: list[_RedCandidate] = []
    contours, _hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    for contour in contours:
        area = float(cv2.contourArea(contour))
        x, y, width, height = cv2.boundingRect(contour)
        if not minimum_area <= area <= maximum_area:
            continue
        if min(width, height) < minimum_dimension:
            continue
        if max(width, height) > maximum_dimension:
            continue
        pixels_xy = contour.reshape(-1, 2).astype(float)
        moments = cv2.moments(contour)
        if abs(float(moments["m00"])) < 1e-9:
            continue
        center = np.asarray([
            moments["m10"] / moments["m00"],
            moments["m01"] / moments["m00"],
        ])
        candidates.append(_RedCandidate(
            center_px=center,
            pixels_xy=pixels_xy,
            area_px=area,
        ))
    return candidates


def _flow_point(
    previous_gray: np.ndarray,
    gray: np.ndarray,
    point: np.ndarray,
) -> tuple[np.ndarray, float] | None:
    old = np.asarray(point, dtype=np.float32).reshape(1, 1, 2)
    new, status, error = cv2.calcOpticalFlowPyrLK(
        previous_gray, gray, old, None,
        winSize=(31, 31), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 0.01),
    )
    if new is None or status is None or int(status.reshape(-1)[0]) != 1:
        return None
    back, back_status, _ = cv2.calcOpticalFlowPyrLK(
        gray, previous_gray, new, None,
        winSize=(31, 31), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 0.01),
    )
    if back is None or back_status is None or int(back_status.reshape(-1)[0]) != 1:
        return None
    forward = new.reshape(2).astype(float)
    fb_error = float(np.linalg.norm(back.reshape(2) - old.reshape(2)))
    lk_error = 0.0 if error is None else float(error.reshape(-1)[0])
    if fb_error > 2.5 or not np.all(np.isfinite(forward)):
        return None
    return forward, max(fb_error, lk_error / 20.0)


class FootTipTracker:
    """Associate red boot-tip blobs with L0..L5 and bridge short occlusions."""

    def __init__(self, *, max_occlusion_frames: int = 8) -> None:
        if max_occlusion_frames < 0:
            raise ValueError("max_occlusion_frames cannot be negative")
        self.max_occlusion_frames = max_occlusion_frames
        self._states: dict[int, _FootState] = {}

    def reset(self) -> None:
        self._states.clear()

    def update(
        self,
        image: np.ndarray,
        *,
        body_center_px: np.ndarray | None,
        femur_anchor_px: Mapping[int, np.ndarray],
        tag_scale_px: float,
        gray_image: np.ndarray | None = None,
    ) -> list[FootTipObservation]:
        if image.ndim != 3:
            raise ValueError("foot tracking requires a BGR image")
        if gray_image is None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = np.asarray(gray_image)
            if gray.ndim != 2 or gray.shape != image.shape[:2]:
                raise ValueError(
                    "gray_image must match the BGR foot-tracking image"
                )
        scale = max(8.0, float(tag_scale_px))
        measured: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}

        if body_center_px is not None and femur_anchor_px:
            candidates = _red_candidates(image, scale)
            legs: list[int] = []
            costs: list[list[float]] = []
            candidate_tips: dict[tuple[int, int], np.ndarray] = {}
            body = np.asarray(body_center_px, dtype=float)
            for leg, raw_anchor in sorted(femur_anchor_px.items()):
                anchor = np.asarray(raw_anchor, dtype=float)
                direction = anchor - body
                norm = float(np.linalg.norm(direction))
                if norm < scale:
                    continue
                direction /= norm
                row: list[float] = []
                for index, candidate in enumerate(candidates):
                    offsets = candidate.pixels_xy - anchor
                    forward_values = offsets @ direction
                    tip = candidate.pixels_xy[int(np.argmax(forward_values))]
                    delta = tip - anchor
                    forward = float(delta @ direction)
                    lateral = abs(float(delta[0] * direction[1] - delta[1] * direction[0]))
                    if forward < 1.5 * scale or forward > 9.0 * scale:
                        row.append(1e9)
                        continue
                    if lateral > 2.0 * scale:
                        row.append(1e9)
                        continue
                    expected_forward = 5.5 * scale
                    expected_area = 0.55 * scale * scale
                    area_penalty = 0.45 * scale * abs(math.log(
                        max(1.0, candidate.area_px) / expected_area
                    ))
                    cost = (
                        lateral
                        + 0.10 * abs(forward - expected_forward)
                        + area_penalty
                    )
                    previous = self._states.get(leg)
                    if previous is not None:
                        predicted = previous.point_px + previous.velocity_px_per_frame
                        cost += 0.08 * float(np.linalg.norm(tip - predicted))
                    row.append(cost)
                    candidate_tips[(leg, index)] = tip
                legs.append(leg)
                costs.append(row)

            if legs and candidates:
                row_indices, column_indices = linear_sum_assignment(
                    np.asarray(costs, dtype=float)
                )
                for row, column in zip(row_indices, column_indices):
                    cost = costs[row][column]
                    if cost >= 1e8:
                        continue
                    leg = legs[row]
                    tip = candidate_tips[(leg, column)]
                    confidence = max(0.25, min(1.0, 1.0 - cost / (2.5 * scale)))
                    measured[leg] = (tip, candidates[column].center_px, confidence)

        observations: list[FootTipObservation] = []
        next_states: dict[int, _FootState] = {}
        for leg in range(LEG_COUNT):
            previous = self._states.get(leg)
            if leg in measured:
                point, center, confidence = measured[leg]
                velocity = np.zeros(2) if previous is None else point - previous.point_px
                # Damping reduces single-frame color-mask jitter.
                if previous is not None:
                    velocity = 0.65 * previous.velocity_px_per_frame + 0.35 * velocity
                source = "color"
                age = 0
            elif previous is not None and previous.age < self.max_occlusion_frames:
                flow = _flow_point(previous.gray, gray, previous.point_px)
                if flow is not None:
                    point, flow_error = flow
                    delta = point - previous.point_px
                    velocity = 0.65 * previous.velocity_px_per_frame + 0.35 * delta
                    center = previous.center_px + delta
                    source = "optical_flow"
                    age = previous.age + 1
                    confidence = max(0.15, 0.72 ** age / (1.0 + flow_error))
                else:
                    age = previous.age + 1
                    point = previous.point_px + previous.velocity_px_per_frame
                    center = previous.center_px + previous.velocity_px_per_frame
                    velocity = 0.8 * previous.velocity_px_per_frame
                    source = "prediction"
                    confidence = max(0.08, 0.45 ** age)
            else:
                continue

            height, width = gray.shape
            if not (-scale <= point[0] <= width + scale and
                    -scale <= point[1] <= height + scale):
                continue
            observation = FootTipObservation(
                leg=leg,
                point_px=np.asarray(point, dtype=float),
                component_center_px=np.asarray(center, dtype=float),
                source=source,
                occlusion_age_frames=age,
                confidence=float(confidence),
                velocity_px_per_frame=np.asarray(velocity, dtype=float),
            )
            observations.append(observation)
            next_states[leg] = _FootState(
                observation.point_px,
                observation.component_center_px,
                observation.velocity_px_per_frame,
                observation.occlusion_age_frames,
                gray.copy(),
            )

        self._states = next_states
        return observations


def robust_tag_scale_px(corners_by_id: Mapping[int, np.ndarray]) -> float | None:
    """Median tag edge length, robust to perspective and outliers."""
    lengths: list[float] = []
    for corners in corners_by_id.values():
        points = np.asarray(corners, dtype=float).reshape(4, 2)
        lengths.extend(
            float(np.linalg.norm(points[(index + 1) % 4] - points[index]))
            for index in range(4)
        )
    if not lengths:
        return None
    value = float(np.median(lengths))
    return value if math.isfinite(value) and value > 0.0 else None
