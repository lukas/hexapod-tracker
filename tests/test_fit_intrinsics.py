import json

import cv2
import numpy as np
import pytest

from hexapod_tracker.fit_intrinsics import (
    FitError,
    FocalFit,
    anchors_are_collinear,
    check_conditioning,
    fit_focal_length,
    intrinsics_entry,
    upsert_entry,
)
from hexapod_tracker.paths import CONFIG_DIR
from hexapod_tracker.planar_pose import PlanarPoseEstimator


def _estimator():
    return PlanarPoseEstimator(
        json.loads((CONFIG_DIR / "floor_tag_map.json").read_text()),
        json.loads((CONFIG_DIR / "hexapod_tag_map.json").read_text()),
        json.loads((CONFIG_DIR / "hexapod-1-apriltag-layout.json").read_text()),
        None,
    )


def _synthetic_observations(f, anchors, tilt_deg, height_mm=1250.0, noise_px=0.3, frames=6, size=(3840, 2160)):
    """Project the surveyed anchors through a known camera at a known tilt."""
    estimator = _estimator()
    world = np.concatenate([estimator.anchor_corners(tag) for tag in anchors])
    obj = np.hstack([world, np.zeros((len(world), 1))])
    K = np.array([[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1.0]])
    tilt = np.radians(tilt_deg)
    # Camera above the grid centre looking down, tilted about x; world z up -> camera looks along -z.
    look_down = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]])
    tilt_x = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]])
    R = tilt_x @ look_down
    centre = obj.mean(axis=0)
    cam_pos = centre + np.array([0.0, 0.0, height_mm])
    t = -R @ cam_pos
    rvec, _ = cv2.Rodrigues(R)
    rng = np.random.default_rng(7)
    observations = []
    for _ in range(frames):
        proj, _ = cv2.projectPoints(obj, rvec, t, K, np.zeros(5))
        observations.append((world, proj.reshape(-1, 2) + rng.normal(0, noise_px, (len(world), 2))))
    return observations, size


def test_fit_recovers_the_focal_length_from_a_tilted_view_of_three_anchors():
    observations, size = _synthetic_observations(4280.0, [100, 103, 112], tilt_deg=25)
    fit = fit_focal_length(observations, size, [100, 103, 112], coarse_step_px=40.0)
    assert abs(fit.f_px - 4280.0) / 4280.0 < 0.03
    assert fit.rms_px < 1.0
    centres = np.array([_estimator().anchors[t]["center"][:2] for t in (100, 103, 112)])
    check_conditioning(fit, centres)  # must not raise


def test_a_fronto_parallel_view_is_refused_as_ill_conditioned():
    observations, size = _synthetic_observations(4280.0, [100, 103, 112], tilt_deg=0.0, noise_px=1.5)
    fit = fit_focal_length(observations, size, [100, 103, 112], coarse_step_px=40.0)
    centres = np.array([_estimator().anchors[t]["center"][:2] for t in (100, 103, 112)])
    with pytest.raises(FitError, match="poorly constrained"):
        check_conditioning(fit, centres)


def test_collinear_anchors_are_refused():
    assert anchors_are_collinear(np.array([[609.6, 0.0], [609.6, 304.8], [609.6, 609.6]]))
    assert not anchors_are_collinear(np.array([[609.6, 0.0], [609.6, 304.8], [0.0, 304.8]]))
    fit = FocalFit(4000.0, 1.0, (3900.0, 4100.0), 5.0, 10, [100, 101, 103])
    with pytest.raises(FitError, match="collinear"):
        check_conditioning(fit, np.array([[609.6, 304.8], [609.6, 609.6], [609.6, 0.0]]))


def test_entry_is_identity_keyed_and_upsert_replaces_by_stable_id():
    fit = FocalFit(4280.0, 4.75, (4080.0, 4500.0), 11.0, 40, [100, 103, 112])
    entry = intrinsics_entry(fit, (3840, 2160), stable_id="0x520000032e46678", device_name="4K U3 Camera",
                             capture_size=(3840, 2160))
    assert entry["camera_matrix"][0][0] == 4280.0 and entry["camera_matrix"][0][2] == 1920.0
    assert entry["capture_size"] == [3840, 2160]
    document = {"cameras": {"elp_4k_u3": {"stable_id": "0x520000032e46678", "camera_matrix": []}}}
    updated = upsert_entry(document, "4k_u3_camera", entry)
    assert list(updated["cameras"]) == ["elp_4k_u3"]  # existing key kept, content replaced
    assert updated["cameras"]["elp_4k_u3"]["camera_matrix"][0][0] == 4280.0
    added = upsert_entry(updated, "ov9281", {**entry, "stable_id": "0x41100000c456366"})
    assert set(added["cameras"]) == {"elp_4k_u3", "ov9281"}


def test_pooled_rms_accepts_three_dimensional_world_points():
    from hexapod_tracker.fit_intrinsics import pooled_rms
    rng = np.random.default_rng(3)
    K = np.array([[1000.0, 0, 640.0], [0, 1000.0, 360.0], [0, 0, 1]])
    world = np.array([[x, y, z] for x, y in [(0, 0), (300, 0), (0, 300), (300, 300), (150, 150)]
                      for z in (0.0,)] + [[150.0, 450.0, 3.0]] * 1, dtype=np.float64)
    world = np.vstack([world, [[-300.0, 150.0, 3.0], [600.0, 150.0, 3.0], [150.0, -300.0, 3.0]]])
    rvec = np.array([np.pi, 0.0, 0.0]); tvec = np.array([-150.0, 150.0, 1500.0])
    image, _ = cv2.projectPoints(world, rvec, tvec, K, np.zeros(5))
    image = image.reshape(-1, 2) + rng.normal(0, 0.05, (len(world), 2))
    assert pooled_rms(1000.0, (1280, 720), [(world, image)]) < 0.2
    # the same corners flattened to z = 0 still solve (planar path), just less exactly
    assert pooled_rms(1000.0, (1280, 720), [(world[:, :2], image)]) < 5.0
