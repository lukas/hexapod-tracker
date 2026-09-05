from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from hexapod_tracker.apriltag_vision import TagCorners
from hexapod_tracker.housing_pose import RigidTransform
from hexapod_tracker.rgbd_calibrate import RGBDFrame, _npz_frames
from hexapod_tracker.zero_pose_survey import _archive_frame


def test_archived_rgbd_frame_replays_through_calibration_reader(tmp_path) -> None:
    pose = RigidTransform(
        np.asarray([0.1, -0.2, 0.7]),
        Rotation.from_euler("xyz", [1.0, 2.0, 3.0], degrees=True),
    )
    frame = RGBDFrame(
        rgb_bgr=np.full((48, 64, 3), [20, 80, 140], dtype=np.uint8),
        depth_m=np.full((12, 16), 0.75, dtype=np.float32),
        confidence=np.full((12, 16), 2, dtype=np.uint8),
        camera_matrix=np.asarray(
            [[50.0, 0.0, 32.0], [0.0, 50.0, 24.0], [0.0, 0.0, 1.0]]
        ),
        source_label="test",
        arkit_world_from_opengl_camera=pose,
    )
    detection = TagCorners(
        7,
        np.asarray([[10, 10], [20, 10], [20, 20], [10, 20]], dtype=np.float32),
    )

    saved = _archive_frame(tmp_path, frame, [detection])
    replayed = list(_npz_frames(tmp_path))

    assert saved.is_file()
    assert len(replayed) == 1
    assert replayed[0].rgb_bgr.shape == frame.rgb_bgr.shape
    assert np.allclose(replayed[0].depth_m, frame.depth_m)
    assert np.array_equal(replayed[0].confidence, frame.confidence)
    assert np.allclose(
        replayed[0].arkit_world_from_opengl_camera.translation_m,
        pose.translation_m,
    )
