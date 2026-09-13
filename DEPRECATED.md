# Deprecated modules (2026-09-11)

`hexapod-calibrate-tags` (`src/hexapod_tracker/tag_calibration.py`) replaced
the handheld iPhone survey and the browser calibration studio. The modules
below were deleted on 2026-09-12 together with `web/vision_ui` and the
capture-profile configs, once the robot repository stopped importing
`web_server` through `linux_control/vision_server.py`. Recover them from git
history before that date if ever needed.

| module | replaced by |
| --- | --- |
| `tag_survey.py` | `tag_calibration.py`: link assignment by motion under the fixed cameras |
| `zero_pose_survey.py` (`hexapod-zero-survey`) | `hexapod-calibrate-tags` |
| `zero_pose_refinement.py` | lid-plane geometry in `tag_calibration.derive_layout` |
| `zero_survey_web.py` | `report.md` / `report.json` written by every calibration run |
| `lab_camera_calibration.py` | the calibration program fits the focal length it needs; `camera_server.py` and `planar_pose.py` own the floor frame |
| `web_server.py` | `camera_server.py` (frames, leases, `/api/poses`, `/api/detections.json`) |
| `vision_web.py` (`hexapod-vision-web`) | `camera_server.py` plus the Robot Lab v2 web UI |

`relayout.py` is the previous name of `tag_calibration.py` and stays as an
alias (`hexapod-relayout` runs the same program).

Also removed 2026-09-12: `configs/camera_intrinsics.json` and
`configs/camera_capture_profiles.json` with the `--capture-profile` option
(slot-keyed, described the September 3 rig). Intrinsics are now identity-keyed
in `configs/camera_intrinsics_lab_20260912.json`; `hexapod-fit-intrinsics`
adds entries. `robot_lab.py` (the Robot Lab publisher used only by the
studio) went with them; `rgbd_calibrate.archive_frame` kept the frame
archiver the survey used.

Kept, not deprecated: `apriltag_vision.py`, `housing_pose.py`, `track.py`,
the RGB-D tools, `layout_audit.py`, `planar_pose.py`, `camera_server.py`,
`rig.py`, `fit_intrinsics.py`.
