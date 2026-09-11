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

Install the Python environment and launch the multi-camera viewer:

```sh
uv sync --extra dev
uv run hexapod-camera-server \
  --indices 0 1 --host 0.0.0.0 --port 8766 \
  --rotate-180 0 1 \
  --robot-url http://hexapod.local:8080
```

Open `http://localhost:8766/` locally, or replace `localhost` with the
computer's LAN address. The viewer exposes annotated and raw MJPEG feeds,
snapshots, tag/calibration status, and the current planar pose estimate.
The optional `--robot-url` enables read-only motor-angle and calibrated IMU
telemetry in the Pose tab. The tracker only calls `GET /api/feedback` and has
no robot command path.

On the September 3 macOS setup, use the saved profile so the Continuity Camera
native path, USB modes, rotations, and matching intrinsic calibration cannot
drift apart:

```sh
OPENCV_AVFOUNDATION_SKIP_AUTH=1 uv run hexapod-camera-server \
  --capture-profile lab-tracking \
  --host 127.0.0.1 --port 8766 \
  --robot-url http://192.168.4.39:8080
```

If Continuity Camera is disconnected, use the USB-only profile instead. This
is intentionally a separate profile: leaving native camera index 0 enabled can
silently select an OV9281 after the iPhone disappears and duplicate one of the
OpenCV USB feeds.

```sh
OPENCV_AVFOUNDATION_SKIP_AUTH=1 uv run hexapod-camera-server \
  --capture-profile lab-usb-only \
  --host 127.0.0.1 --port 8766 \
  --robot-url http://192.168.4.39:8080
```

Here native AVFoundation index 0 is `lukas's iPhone Camera`; OpenCV indices 1
and 2 are the OV9281 USB cameras. These numbers describe this boot only and
must be checked from `/status.json` and the live images after reconnecting.
The source of truth for this setup and the observed device mode inventory is
`configs/camera_capture_profiles.json`.

The live IMU card shows measurements only. The combined JSON API renames the
robot's legacy `body_pitch_target_deg` field to
`rear_pose_pitch_reference_deg`: it is fixed metadata captured during the
known rear-lean body-frame calibration pose, not a current measurement or a
live controller target, and it is intentionally not displayed in the UI.

### Choose the correct camera web server

There are two different camera web interfaces in this repository:

- **Show all configured cameras:** use the standalone
  `hexapod-camera-server` above. Its page is served at `/` and displays one
  card per index passed with `--indices`. A request such as “show the available
  cameras” or “show both USB cameras” refers to this server, not the React
  vision UI.
- **Inspect one selected camera in the robot application:** use
  `hexapod_tracker.web_server.VisionRuntime`, mounted at `/vision`. This is the
  single-camera calibrated-tracking interface and is not the all-camera page.

The normal `hexapod-camera-server` command starts capture workers immediately.
If the operator asks to list camera cards without turning cameras on, serve the
standalone page with dormant `CameraWorker` instances; do not substitute the
single-camera `/vision` UI. In that state `/status.json` must report zero
frames for every camera until capture is explicitly started by a later action.

On macOS, do not assume AVFoundation discovery-list indices match the indices
used by OpenCV's `VideoCapture`. Treat the live image and reported capture mode
as the identity check, and visually confirm every selected feed after a server
restart. Camera indices are ephemeral and can change when a display, iPhone, or
USB camera reconnects.

Use `--rotate-180 INDEX [INDEX ...]` for physically inverted cameras. Rotation
is applied before AprilTag detection and JPEG encoding so the annotated feed,
raw snapshots, and pose coordinates all share the same upright image frame;
do not rotate only the browser image with CSS.

Useful endpoints include:

- `/` — camera grid and tracking status
- `/status.json` — capture details and detected tags
- `/api/poses` — floor-referenced part poses
- `/api/pose-state` — camera poses plus read-only motor and calibrated IMU data
- `/stream/0.mjpg` and `/raw-stream/0.mjpg` — annotated and raw video
- `/snapshot/0.jpg` — current raw frame
- `/native-luma/0.png` — lossless, full-resolution iPhone luminance plane
- `/native-frame/0.nv12` — exact full-resolution iPhone NV12 video frame
- `/calibration-status.json` — calibration capture state

The recommended `lab-tracking` profile asks Continuity Camera for its full
1920x1440, 30 fps, 8-bit video-range NV12 source. It keeps the browser/color
processing path at 1280x960 for responsiveness, but the two native routes
retain the unscaled 1920x1440 data. The `.nv12` response concatenates the Y
plane and interleaved UV plane and includes `X-Frame-Width`, `X-Frame-Height`,
and `X-Pixel-Format` headers. This is raw decoded video—not Bayer sensor RAW
or ProRAW, which Continuity Camera does not expose to this capture path.

The USB profile uses the full 1280x800 sensor field at its reported 100 fps;
the server publishes 10 fps to the browser to control CPU and bandwidth. Do
not replace it with 1280x720 unless the vertical crop is intentional.

The floor homography produces physical `x`, `y`, and yaw for markers on the
floor plane. It also rectifies directions parallel to that plane, so the Pose
tab can calculate each visible leg's robot-relative yaw from the horizontal
chassis tag plus that leg's horizontal coxa servo-lid tag. The documented tag
mount rotations come from `configs/hexapod-1-apriltag-layout.json` (override
with `--robot-tag-layout`). This yaw assumes both tag faces remain parallel to
the floor. Its uncertainty comes from paired tag headings in the same camera,
corner precision, and simultaneous cross-camera disagreement. It does not add
the two absolute floor-heading bounds because their shared component cancels
in the robot-relative angle. A provisional 5° 95% floor reflects the observed
stationary repeatability; simultaneous camera disagreement can raise it.

Elevated vertical yoke tags are still returned only as `projection_only`
diagnostics in the planar `parts` output: their camera-ray/floor intersections
are not physical part positions. Joint orientation is handled separately.
With `configs/camera_intrinsics.json`, a camera that sees chassis tag
`0`, floor anchors, and any documented `L*_femur` tag uses square-tag PnP and
the layout's mount rotation to calculate robot-relative hip pitch. The solver
enumerates both planar pose branches and rejects the one inconsistent with the
hexapod's yaw-then-pitch kinematics. A visible documented `L*_tibia` tag uses
the same body-relative 3-D orientation path to calculate the contract's
absolute tibia/knee angle. The value is unavailable only when no accepted
tibia tag shares a calibrated camera view with chassis tag `0`.

The checked-in iPhone and OV9281 calibrations are explicitly provisional:
they are constrained fits to the single floor plane, with fixed principal
points, square pixels, and zero distortion. Replace them with multi-pose
ChArUco calibrations before treating 3-D results as metrology-grade.

Intrinsic calibration describes the camera/lens and can be reused after the
camera is moved if the physical lens, zoom, resolution, crop, orientation,
and stabilization mode stay unchanged. Extrinsic calibration describes the
camera's position in the room and changes whenever the camera moves. This
server re-establishes extrinsics from the surveyed floor tags on every current
frame, so moving a camera does not normally require a new manual extrinsic
calibration; keep at least enough floor anchors visible. Lens switching,
digital zoom, a different Continuity Camera mode, or a changed crop/resolution
does require a new intrinsic profile.

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

## Tag calibration (`hexapod-calibrate-tags`)

After tags fall off or the robot is reassembled, re-derive the tag layout by
moving the robot under the fixed cameras. No handheld scan, no pre-run
checklist: the program checks that the robot is resting flat and the cameras
are still, lifts each leg in turn (hip, an antisymmetric yaw swing, knee) to
learn which tag rides on which link, then does all the geometry in the
top camera's lid plane and writes a report that says in plain words what
changed, which mounts have no tag, and which conventions it measured.

```sh
uv run hexapod-calibrate-tags                 # measure; outputs under Hexapod Lab/v2/tag-calibration-<stamp>/
uv run hexapod-calibrate-tags --write         # ...and install into configs/ if the validator is clean
uv run hexapod-calibrate-tags --replay DIR    # redo the geometry from an earlier run's zero_tags.json
uv run hexapod-calibrate-tags --replay DIR --assign-from other/report.json   # merge passes
```

What it writes into `configs/hexapod-1-apriltag-layout.json` besides the tags:

- `leg_zero_azimuth_body_deg`: where each leg points at the commanded zero
  pose, measured. The legs are numbered clockwise seen from above; the body
  frame is z up, x forward between legs 0 and 5.
- `joint_conventions.yaw_sign_in_body_frame`: a positive yaw command turns a
  leg clockwise from above, so the tracker multiplies by -1 to report yaw in
  the robot's sense. (The gait code's frame is right-handed with z down, x
  forward, y right: the tracker's frame rotated 180 degrees about x, not a
  reflection.) `planar_pose.py` reads both fields and falls back to the old
  `(leg + 0.5) * 60` assumption when they are absent.
- `unresolved_mounts`: mounts no camera saw a tag on, declared so the
  validator can tell a known gap from a mistake. Faces carried unseen from the
  previous layout have `verified: false`.

Claude (optional, `--no-claude` to skip) only reads annotated crops with the
ids drawn on, as a placement cross-check and a census of blank faces. The
program's geometry is deterministic and covered by `tests/test_tag_calibration.py`,
including a replay of the 2026-09-11 pass. Turn the robot and run again to
see the faces the first pass could not.

The camera server publishes its own detector's corners at
`/api/detections.json` (pixels in `/snapshot/{i}.jpg`, with `detect_seq` so a
poller can tell a fresh detection from a repeat); the calibration program uses
that when the running server has it and detects on snapshots otherwise.


## Relationship to the robot repository

The main hexapod repository includes this project as the
`hexapod_walker/prototype_sts3215/hexapod-tracker` Git submodule. Compatibility
entry points at the historical paths import this package, while robot-specific
gait-survey orchestration remains in the main repository.

## Deprecated calibration tools

The handheld iPhone survey and the browser calibration studio are superseded
by `hexapod-calibrate-tags` above; their modules still import (with a
`DeprecationWarning`) but have no console scripts. See
[`DEPRECATED.md`](DEPRECATED.md) for what replaced each one.

### Guided zero-pose tag survey (deprecated)

The same iPhone stream can survey a stationary robot from a slow handheld walk.
Put the robot in zero pose beside the calibration board, identify one chassis
tag whose existing mount has not moved, and keep the configured L0 hip tag as
the leg-number reference. Then run:

```sh
uv run python -m hexapod_tracker.zero_pose_survey \
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

### Web UI (deprecated)

The React source and its checked-in production build are in `web/vision_ui`.
Launch the camera-only calibration studio directly from this repository:

```sh
uv run --extra rgbd python -m hexapod_tracker.vision_web
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
