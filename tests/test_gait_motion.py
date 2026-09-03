from __future__ import annotations

import cv2
import numpy as np

from hexapod_tracker.gait_motion import _best_floor_homography


def test_motion_homography_drops_inconsistent_duplicate_floor_tag():
    world = {
        12: np.asarray([[0, 0], [.04, 0], [.04, .04], [0, .04]], dtype=float),
        13: np.asarray([[.6, .7], [.64, .7], [.64, .74], [.6, .74]], dtype=float),
        15: np.asarray([[.6, 0], [.64, 0], [.64, .04], [.6, .04]], dtype=float),
    }
    floor_to_image = np.asarray([
        [900.0, 120.0, 300.0],
        [50.0, 820.0, 240.0],
        [0.2, 0.1, 1.0],
    ])
    image = {
        tag_id: cv2.perspectiveTransform(
            corners.astype(np.float32).reshape(1, -1, 2), floor_to_image
        )[0]
        for tag_id, corners in world.items()
    }
    image[13] = image[13] + np.asarray([500.0, -250.0])
    detections = {
        tag_id: {"corners_px": corners.tolist()}
        for tag_id, corners in image.items()
    }
    fit = _best_floor_homography(detections, world)
    assert fit is not None
    _homography, selected, rms_m = fit
    assert selected == [12, 15]
    assert rms_m < 1e-5
