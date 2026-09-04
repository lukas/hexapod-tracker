# hexapod-tracker

Camera-only AprilTag tracking and visualization tools extracted from
[`lukas/hexapod`](https://github.com/lukas/hexapod). The package detects
tag36h11 markers, estimates calibrated body and joint poses, tracks colored
foot tips, analyzes recorded gait motion, serves the local camera UI, and can
use an iPhone LiDAR stream to make fixed-camera calibration repeatable.

The standalone package never commands motors. Its optional web-server survey
hooks are disabled unless the consuming robot project supplies an explicit
motion adapter.

For architecture, physical-test context, configuration caveats, and the
current next steps, read [`docs/LLM_HANDOFF.md`](docs/LLM_HANDOFF.md). It is the
fastest orientation document for both human and LLM maintainers.

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

The main calibration and tag-map files live in `configs/`.
`hexapod-1-apriltag-layout.json` is the photographed physical inventory for
Hexapod 1: 37 unique robot-tag mounts, seven floor anchors, and each tag's
frame-relative orientation. See
[`docs/HOUSING_POSE.md`](docs/HOUSING_POSE.md) for coordinate conventions,
mount calibration, multi-camera behavior, and output formats.

Audit a new photo set against that inventory with:

```sh
uv run hexapod-audit-layout \
  --require-all-layout-ids \
  --require-all-orientations \
  --output-dir artifacts/apriltag-audit/annotated \
  --report artifacts/apriltag-audit/report.json \
  /path/to/photos/*.jpeg
```

The report preserves repeated detections of the same ID, cross-checks the
floor and planar-viewer configs, and independently checks all robot/floor
orientations when the photo set has enough geometry. Annotated images draw tag
`+X` in red and `+Y` in green.

## iPhone LiDAR-assisted calibration

Generate the included-size calibration target (or use the checked-in copy):

```sh
uv run hexapod-calibration-board \
  --svg configs/rgbd_calibration_board.svg \
  --manifest configs/rgbd_calibration_board.json
```

Print the SVG at **100% / actual size**, mount it flat on a rigid matte board,
and verify that a black tag square is 70 mm. Lock the iPhone in its final
tracking position, open Record3D 1.10 or newer in USB-streaming mode, then run:

```sh
uv run --with cmake uv sync --extra dev --extra rgbd
uv run hexapod-rgbd-calibrate \
  configs/apriltag_pose_config_20260831.json \
  --board configs/rgbd_calibration_board.json \
  --robot-layout configs/hexapod-1-apriltag-layout.json \
  --frames 30 \
  --output artifacts/rgbd-calibration.json \
  --updated-config artifacts/apriltag_pose_config_rgbd.json \
  --preview
```

The command detects the mapped tag corners, robustly fits the LiDAR floor
plane, jointly refines the camera pose, rejects moved/bad frames, and writes
measured stream intrinsics plus a fixed `world_from_camera` transform. It does
not connect to or move the robot. After calibration, use the same Record3D RGB
stream so its lens/crop and per-frame ARKit intrinsics stay matched:

```sh
uv run hexapod-track artifacts/apriltag_pose_config_rgbd.json \
  --record3d-device 0 --preview
```

The board may leave the image after calibration as long as neither the phone
nor the board-defined world frame moves. See
[`docs/RGBD_CALIBRATION.md`](docs/RGBD_CALIBRATION.md) for setup, quality gates,
offline fixtures, coordinate conventions, and limitations.

## Guided zero-pose tag survey

The same iPhone stream can survey a stationary robot from a slow handheld walk.
Put the robot in zero pose beside the calibration board, identify one chassis
tag whose existing mount has not moved, and keep the configured L0 hip tag as
the leg-number reference. Then run:

```sh
uv run hexapod-zero-survey \
  configs/apriltag_pose_config_20260831.json \
  --board configs/rgbd_calibration_board.json \
  --body-anchor-tag-id 0 \
  --output artifacts/zero-pose-tag-survey.json \
  --updated-config artifacts/apriltag_pose_config_surveyed.json
```

The production layout expands the checklist to 37 robot mounts: the chassis and
12 servo-lid tags plus four vertical angle tags on each of six legs. The preview
first asks for a stable mapped-floor lock, then becomes a scan dashboard
with the live camera, an isometric 3-D tag map, the phone path, tracking health,
and a physical-position checklist (`L0 hip`, `L0 knee`, and so on). It clearly
separates a position that has never been seen from a tag that was decoded but
needs another clean view. It records each tag's metric 6-D pose, orientation
axes, observation spread, automatically discovered tag IDs, and all pairwise
floor-tag distances. Stable floor poses and relearned robot-tag mounts are
written to the optional new config; the trusted body anchor is deliberately
left unchanged. Use `--expected-floor-ids 12,13,15` when the input floor map is
not the exact list that should gate completion.

Robot completion is position-based rather than old-ID-based. When a configured
ID is absent, the survey fits the existing calibration-photo layout to the
recognized tags and may assign a nearby stable new ID to that empty mount. The
L0 hip is protected because it defines leg numbering; if that particular tag
was replaced, declare the new identity with `--leg-zero-anchor-tag-id NEW_ID`.

This captures every configured robot position and expected floor tag, but it cannot
know that an unlisted, never-visible physical tag exists. The seven known floor
tags are solved jointly whenever two or more are visible, and Save stays disabled
until the floor-grid, LiDAR-plane, per-tag spread, and full coverage checks pass.
One zero-pose capture
can relearn tag mounts against the current kinematic model and measure static
inter-tag baselines. It cannot uniquely separate link lengths, joint-axis
locations, and tag offsets; exact geometry fitting needs several stationary,
encoder-known poses using the existing tibia-fixed side tags. See
[`docs/RGBD_CALIBRATION.md`](docs/RGBD_CALIBRATION.md#handheld-zero-pose-tag-survey).

## Web UI and tests

The React source and its checked-in production build are in `web/vision_ui`.
Launch the camera-only calibration studio directly from this repository:

```sh
uv run --extra rgbd hexapod-vision-web
# open http://127.0.0.1:8898/vision
```

The default **Tag survey** page walks through Record3D connection, mapped-floor
lock, 13 top/chassis + 24 vertical robot mounts, a live 3-D schematic, reviewed config
creation, and Robot Lab publication. USB is the precision path. Record3D 1.11+
can also relay its WebRTC Wi-Fi RGB-D stream, synchronized intrinsics, and ARKit
pose through the page; the app's paid Wi-Fi extension is required and its lossy
depth is expected to be noisier. The live quality coach reports corner error,
position/angle spread, phone speed, and corrective guidance. Stable observations
are checkpointed so an unplugged or dropped connection can be continued after
re-locking any mapped floor tag instead of starting over.

The workflow never commands the robot. Robot Lab publication uses only the
versioned `/api/calibrations` endpoint. The server first reads
`HEXIPOD_LAB_TOKEN` (or `HEXAPOD_LAB_TOKEN`), then a protected path from
`HEXIPOD_LAB_TOKEN_FILE` (by default it checks both `~/Documents/hexapod.rtf`
and TextEdit's sandboxed Documents folder), then the
`HEXIPOD_LAB_TOKEN_OP_REF` 1Password reference (default
`op://Private/Hexapod Lab API/credential`). The `op` CLI must be installed and
signed in for 1Password lookup. `HEXAPOD_LAB_URL` selects another server.
Without a token the survey stays local and the page shows a specific retry
diagnostic.

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
