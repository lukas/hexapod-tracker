# hexapod-tracker

Camera-only AprilTag tracking and visualization tools extracted from
[`lukas/hexapod`](https://github.com/lukas/hexapod). The package detects
tag36h11 markers, estimates calibrated body and joint poses, tracks colored
foot tips, analyzes recorded gait motion, and serves the local camera UI.

The standalone package never commands motors. Its optional web-server survey
hooks are disabled unless the consuming robot project supplies an explicit
motion adapter.

## Quick start

Install the Python environment and launch the two-camera viewer:

```sh
uv sync --extra dev
uv run hexapod-camera-server \
  --indices 0 1 --host 0.0.0.0 --port 8766
```

Open `http://localhost:8766/` locally, or replace `localhost` with the
computer's LAN address. The viewer exposes annotated and raw MJPEG feeds,
snapshots, tag/calibration status, and the current planar pose estimate.

Useful endpoints include:

- `/` — camera grid and tracking status
- `/status.json` — capture details and detected tags
- `/api/poses` — floor-referenced part poses
- `/stream/0.mjpg` and `/raw-stream/0.mjpg` — annotated and raw video
- `/snapshot/0.jpg` — current annotated frame
- `/calibration-status.json` — calibration capture state

## Calibrated tracking

Run the full 6-D tracker on a still, recording, or camera:

```sh
uv run hexapod-track configs/apriltag_pose_config_20260831.json \
  --input recording.mp4 \
  --pose-output poses.jsonl \
  --annotated-output annotated.mp4
```

The main calibration and tag-map files live in `configs/`. See
[`docs/HOUSING_POSE.md`](docs/HOUSING_POSE.md) for coordinate conventions,
mount calibration, multi-camera behavior, and output formats.

## Web UI and tests

The React source and its checked-in production build are in `web/vision_ui`.

```sh
make check
make web-build
```

`hexapod_tracker.web_server.VisionRuntime` and
`wrap_handler_with_vision(...)` let another Python HTTP server mount the UI at
`/vision` and the JSON/MJPEG API at `/api/vision/*`. Pass a `survey_factory`
only in a robot repository that owns its own guarded motion policy.

## Relationship to the robot repository

The main hexapod repository includes this project as the
`hexapod_walker/prototype_sts3215/hexapod-tracker` Git submodule. Compatibility
entry points at the historical paths import this package, while robot-specific
gait-survey orchestration remains in the main repository.
