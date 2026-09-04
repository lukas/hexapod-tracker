# LLM handoff: what this repository is doing

Read this before changing the tracker. This file records the architecture,
physical assumptions, evidence, and repository boundary that are easy to miss
by reading one module in isolation.

## The short version

`hexapod-tracker` is the camera and AprilTag subsystem extracted from
[`lukas/hexapod`](https://github.com/lukas/hexapod). It has four related uses:

1. A simple multi-camera webpage for seeing USB feeds and tag detections.
2. A calibrated single-camera pipeline for 6-D body/link pose, joint
   diagnostics, red foot-tip tracking, and optional read-only encoder
   comparison.
3. Offline gait analysis and annotated telemetry-video generation.
4. Optional iPhone LiDAR-assisted calibration of a fixed RGB camera against a
   generated, dimensioned AprilTag board.
5. A guided handheld zero-pose survey that discovers tags, maps floor markers,
   and relearns non-anchor robot-tag mounts in an ARKit-tracked world frame.

The package is observation software. It must not acquire authority to move the
robot. The standalone web runtime deliberately installs
`UnavailableSurveyManager`; only the main robot repository may inject a
guarded motion adapter.

Start an investigation with:

```sh
uv sync --extra dev
make check
git status --short --branch
```

`make check` currently runs 58 synthetic/off-robot Python tests and the React
type-check. It does not prove that a particular camera index, intrinsic
calibration, tag placement, or physical measurement is valid.

## Repository boundary

The canonical integration is the Git submodule at:

```text
hexapod/hexapod_walker/prototype_sts3215/hexapod-tracker
```

The main repository retains only thin compatibility entry points at its old
paths. Important integration files there are:

- `linux_control/vision_server.py`: injects the robot-specific
  `GaitSurveyManager` into this repo's `VisionRuntime`.
- `linux_control/gait_survey.py`: owns guarded hardware runs and their
  lifecycle. It is intentionally not part of this repository.
- `vision/apriltag_stream.py`, `linux_control/track_apriltags.py`, and the
  other old modules: compatibility imports/CLIs that forward here.
- `linux_control/vision_ui` and the old JSON config paths: compatibility
  symlinks into the submodule.

A fresh main-repo checkout needs:

```sh
git submodule update --init
```

When changing this repository, push its commit first. Then update and commit
the submodule pointer in `lukas/hexapod`. Never point the main repository at a
tracker commit that has not been pushed.

## There are two live camera paths

They solve different problems and should not be accidentally merged.

### 1. Simple multi-camera viewer

Entry point: `hexapod_tracker.camera_server` / `hexapod-camera-server`.

```sh
uv run hexapod-camera-server \
  --indices 0 1 --host 0.0.0.0 --port 8766
```

The code default is port `8765`; `8766` is commonly used to avoid colliding
with an already-running local viewer. Each `CameraWorker` independently opens
an OpenCV AVFoundation index, requests MJPG input, drops excess capture frames,
detects tag36h11 markers, and publishes raw plus annotated JPEGs. It reconnects
after three consecutive capture failures.

The embedded HTML at `/` shows every requested camera. Relevant routes are:

- `/status.json`: capture backend/mode, frame counters, errors, and tag IDs.
- `/stream/<index>.mjpg`: annotated stream.
- `/raw-stream/<index>.mjpg`: raw stream.
- `/snapshot/<index>.jpg`: raw current frame.
- `/api/poses`: planar floor-referenced fusion from visible cameras.
- `/calibration-status.json`: state produced by an optional external
  calibration capture directory.

`PlanarPoseEstimator` fits a separate floor homography for every view and then
fuses ground-projected part estimates. This is useful for relative flex and
walking displacement. It is not calibrated stereo and does not triangulate
3-D points between cameras.

### 2. Calibrated tracker and React UI

Core entry point: `hexapod_tracker.track` / `hexapod-track`.

```sh
uv run hexapod-track configs/apriltag_pose_config_20260831.json \
  --input recording.mp4 \
  --pose-output poses.jsonl \
  --annotated-output annotated.mp4 \
  --summary-output summary.json
```

This path runs PnP tag pose, floor/world reference estimation, temporal
optical-flow bridging, rigid housing pose, logical joint diagnostics, and red
foot-tip tracking. `--robot-url` is optional and may only read
`GET /api/feedback`; it must never send a motor command. Output joint metadata
uses:

```text
joint_frame = robot_abs
joint_contract = robot_abs_tibia_v2
```

The React source and checked-in build live in `web/vision_ui`. The Python
runtime is `hexapod_tracker.web_server.VisionRuntime`, mounted into another
HTTP server by `wrap_handler_with_vision(...)` at `/vision` and
`/api/vision/*`. It owns one active camera at a time, unlike the simple camera
grid. The UI contains gait-survey controls because the main robot repo supplies
that adapter. In this standalone repo those routes are unavailable and the
runtime reports `read_only: true`.

`hexapod-vision-web` is the standalone local entry point at `:8898/vision`.
Its default light Tag survey workflow wraps `hexapod-zero-survey`, publishes
atomic live progress and a clean labelled camera JPEG, renders the tag geometry
in SVG, and only creates the reviewed config after the operator confirms the
unchanged chassis anchor. The final action publishes the survey and config to
Robot Lab using its first-class calibration endpoint, with the older completed-
result artifact API as a compatibility fallback.

Do not add robot-control HTTP calls here to make the standalone UI's survey
buttons work. That would break the intentional safety and ownership boundary.

### iPhone RGB-D calibration path

`hexapod-calibration-board` creates the printable board/map and
`hexapod-rgbd-calibrate` consumes registered Record3D RGB, depth, confidence,
and per-frame intrinsics. RGB tag corners initialize the existing mapped-floor
PnP solve. A robust LiDAR plane then constrains distance, roll, and pitch in a
joint least-squares refinement. Multiple stationary observations are averaged
after translation/rotation outlier rejection.

The output tracker config has separate `marker_size_m` (robot tags) and
`floor_marker_size_m` (calibration tags), plus a
`fixed_camera_world_reference`. `AprilTagPoseTracker` uses that fixed transform
when mapped floor tags leave the image. `hexapod-track --record3d-device` keeps
using Record3D and refreshes RGB intrinsics on every frame, avoiding a silent
switch to a different Continuity Camera crop. See `docs/RGBD_CALIBRATION.md`.

`hexapod-zero-survey` is the moving-phone companion. A repeated board sighting
aligns Record3D's OpenGL/ARKit camera trajectory to the board world frame. A
slow walk then aggregates every decoded tag's 6-D transform. Completion is one
stable tag per named robot mount position plus every expected floor ID, not the
continued presence of every old robot ID. A missing configured robot ID may be
replaced by a nearby stable newly discovered ID after fitting the original
calibration-photo layout to recognized tags. The configured L0 hip tag is the
protected leg-number reference unless the operator explicitly supplies its new
ID. The live dashboard distinguishes `not seen` from `seen, needs another view`
and shows an isometric tag/orientation map and phone path. The updated config
replaces surveyed floor poses and learns robot `frame_from_tag` values from a
known stationary pose. One explicitly trusted body tag remains unchanged as the
unavoidable body-frame gauge. A one-pose result is mount calibration plus
static baselines, not independently identified link lengths or joint axes.

## Data flow

```text
camera/image/video
  -> tag36h11 corners
  -> per-tag PnP pose + floor world reference
  -> rigid body/coxa/femur estimates
  -> robot_abs yaw/hip diagnostics
  -> red foot-tip projection + unsigned knee evidence
  -> JSON/JSONL, annotated media, and web state

optional registered iPhone RGB + depth + confidence
  -> mapped board-tag corners + robust depth plane
  -> fixed world_from_camera and measured RGB intrinsics
  -> same AprilTagPoseTracker world frame after the board leaves view

optional handheld Record3D ARKit trajectory + initial/revisited board
  -> board-aligned moving camera poses
  -> robust 6-D pose for every decoded robot/ground tag
  -> floor-tag distances + non-anchor zero-pose mount updates

optional GET /api/feedback
  -> encoder comparison only
  -> no commands, no zero writes
```

The knees are not directly signed from the lid-tag layout. Red boot tips can
provide projected/unsigned knee evidence, and encoders can fill the logical
joint vector, but output fields such as `visual_source`, `confidence`,
`prediction_only_joints`, and `calibration_disagreements` must remain honest
about how each value was obtained.

## Configuration truth and caveats

There are three deliberately different tag maps:

- `configs/apriltag_pose_config_20260831.json` is the full calibrated-tracker
  layout. It maps tag 0 to the chassis, tags 1/7 to L0, 4/14 to L1, 6/11 to
  L2, 5/9 to L3, 3/10 to L4, and 2/8 to L5. The paired values are hip/coxa and
  knee/femur tags.
- `configs/hexapod-1-apriltag-layout.json` is the 2026-09-03 photographed
  physical inventory for Hexapod 1. It records 37 unique robot-tag mounts,
  all per-tag orthogonal orientations, and seven one-foot-grid floor anchors.
  Robot tag translations remain unmeasured.
- `configs/hexapod_tag_map.json` is the side-tag grouping used by the simple
  planar flex viewer. It now derives all 12 hip/knee pairs from the Hexapod 1
  inventory; each pair is ordered `[+y face, -y face]`.

The photographed black squares are modeled as 27.2 mm, excluding the white
quiet zone.

Important limitations in the current checked-in configs:

- The full tracker camera intrinsics are approximate iPhone 17 Pro intrinsics,
  not USB-camera calibration. Do not use them for USB-camera metric 3-D.
- Several `frame_from_tag` translations are zero/photo-inferred placeholders.
  Measure tag-to-joint-axis transforms before calling a tag center a mechanical
  joint center.
- The planar floor map is provisional. Active anchors are tags 100, 101, 102,
  103, 104, 105, and 112. Their one-foot center spacing comes from the
  operator's placement description, while yaw comes from two photographed
  planar rectifications; survey the centers before millimeter-level claims.
- The current Hexapod 1 and floor inventory has no duplicate IDs. Other loose
  prints in the garage are outside this map and must not be introduced without
  checking for collisions.
- Camera indexes are not identities. iPhone Continuity Camera and reconnecting
  USB devices can reorder indexes. Confirm device names and live images after
  every rescan/restart.

## What the latest physical tests established

The raw experiment data remains in the main hexapod workspace's ignored
artifact tree; it is not versioned in either repository. The durable summary
is `robots/experiments/hexapod-1-joint-flex/status.yaml` in `lukas/hexapod`.
As of 2026-09-03, the latest repeated two-USB-camera flex run is at:

```text
hexapod_walker/prototype_sts3215/
  artifacts/joint_flex/hexapod-1/repeat_usb_20260903T082639
```

The two Arducam USB cameras saw at least one AprilTag in all recorded frames:
8,324/8,324 for camera 0 and 14,942/14,942 for camera 1. This established that
the pair is good enough for tag visibility and relative motion tracking in the
tested L4/L5 views. It did not establish calibrated stereo accuracy.

The corresponding experiment status reports that L5 had substantially more
hip/knee deadband and planted hysteresis than L4, with no meaningful creep at
the tested load. That diagnosis combines encoder and experiment telemetry; it
is not an inference made by this package alone. Absolute stiffness and exact
localization within the fastener/yoke/link stack remain unknown because there
was no calibrated force measurement and no component-local marker stack.

Do not turn the successful tag-frame counts into stronger claims about pose
accuracy. Detection coverage, camera calibration, common-world extrinsics,
force calibration, and component localization are separate questions.

## Module map

- `camera_server.py`: simple multi-camera MJPEG site and planar pose API.
- `planar_pose.py`: per-camera homographies and cross-camera planar fusion.
- `apriltag_vision.py`: calibrated tag detection, PnP/world pose, temporal
  tracking, fixed-camera reference fallback, and combined frame diagnostics.
- `calibration_board.py`: exact-size printable tag36h11 grid and matching map.
- `rgbd_calibration.py`: registered depth sampling, robust plane fit, joint
  RGB-D refinement, and fixed-camera consensus.
- `rgbd_calibrate.py`: Record3D/offline capture and calibrated-config writer.
- `tag_survey.py`: ARKit/OpenCV frame alignment, robust per-tag pose consensus,
  floor-distance reporting, and zero-pose mount/config updates.
- `zero_pose_survey.py`: guided live Record3D/offline walk-around CLI.
- `housing_pose.py`: rigid transforms, kinematic frame fusion, and joint-angle
  reconstruction.
- `foot_tip_tracking.py`: red boot-tip segmentation, assignment, and short
  optical-flow/prediction bridges.
- `track.py`: still/video/live CLI, read-only feedback client, summaries, and
  annotated output.
- `web_server.py`: one-camera runtime, calibration reports, React/API mounting,
  and the optional survey-adapter boundary.
- `avfoundation_capture.py`: macOS native 420v/luma capture adapter; it degrades
  to unavailable on non-macOS systems.
- `gait_motion.py`: offline floor-homography displacement analysis.
- `telemetry_video.py`: synchronized overlays and ffmpeg output.
- `joint_contract.py`: the minimal artifact coordinate contract copied out of
  the robot repo to avoid a reverse dependency.

## Dependencies and validation

- Use `uv`; do not use bare `pip`.
- AprilTag support requires `opencv-contrib-python`, not `opencv-python`,
  because the detector uses `cv2.aruco`.
- Native Mac capture needs the PyObjC AVFoundation framework.
- iPhone depth is optional. On macOS, Record3D has no wheel; use
  `uv run --with cmake uv sync --extra dev --extra rgbd` to build Record3D
  1.4.1+ for the Record3D iOS 1.10+ USB stream. The import stays lazy so
  normal camera tools do not require it.
- `telemetry_video.py` shells out to `ffmpeg`, which is not a Python package.
- UI changes require both `make check` and `make web-build`; commit the changed
  `web/vision_ui/dist` assets.
- Unit tests use generated tags, synthetic geometry, and fake captures. Add an
  explicit hardware smoke check when changing capture backends or camera-mode
  negotiation.

The package currently assumes an editable/source checkout when locating
`configs/` and `web/vision_ui/` through `paths.py`. A future wheel-distribution
effort must package those resources and switch to `importlib.resources`; do
not assume the present wheel has self-contained defaults.

## Good next steps

In roughly descending value:

1. Run physical fixed and handheld iPhone RGB-D smoke tests; record observed
   depth, board-alignment, ARKit-drift, and per-tag spread thresholds.
2. Capture multiple stationary, encoder-known poses with tibia-fixed tags, then
   fit joint axes/link lengths separately from tag-mount transforms.
3. Calibrate each Arducam's intrinsics at every capture mode actually used.
4. Use one unmoved RGB-D board pose to establish a measured common frame if
   true multi-view 3-D or stereo claims are needed.
5. Replace provisional floor coordinates/yaws with a physical handheld survey
   and eliminate duplicate tag IDs.
6. Measure tag-to-joint-axis mount transforms or add component-local markers
   before trying to localize flex within an assembly.
7. Add recorded-camera regression clips with expected tag/pose summaries. Keep
   large media out of Git and document how to retrieve it.
8. Make package resources wheel-safe if this project will be installed outside
   a source checkout.

Before reporting a result, state separately: tag coverage, calibration quality,
pose/relative-motion result, encoder evidence, force evidence, and remaining
unobservable quantities.
