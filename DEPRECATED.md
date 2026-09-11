# Deprecated modules (2026-09-11)

`hexapod-calibrate-tags` (`src/hexapod_tracker/tag_calibration.py`) replaced
the handheld iPhone survey and the browser calibration studio. The modules
below still import, emit a `DeprecationWarning`, and have no console scripts.
They will be deleted once nothing outside this repository imports them
(`linux_control/vision_server.py` still star-imports `web_server`).

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

Kept, not deprecated: `apriltag_vision.py`, `housing_pose.py`, `track.py`,
the RGB-D tools, `layout_audit.py`, `planar_pose.py`, `camera_server.py`.
