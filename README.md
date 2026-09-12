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

Install the Python environment and run the multi-camera server as the lab
does, discovering the attached cameras and pinning each slot by identity:

```sh
uv sync --extra dev
tools/camera_service.sh cameras       # what would be pinned, and why anything is excluded
CAMERA_SERVICE_HOST=:: CAMERA_SERVICE_ROBOT_URL=http://hexapod.local:8080 \
  tools/camera_service.sh start       # launchd job; relaunches itself when the rig changes
tools/camera_service.sh status
```

`hexapod-camera-server` can also be run by hand with explicit `--indices`
and `--device-id SLOT:uniqueID` pins; see `--help`.

Open `http://localhost:8766/` locally, or replace `localhost` with the
computer's LAN address. The viewer exposes annotated and raw MJPEG feeds,
snapshots, tag/calibration status, and the current planar pose estimate.
The optional `--robot-url` enables read-only motor-angle and calibrated IMU
telemetry in the Pose tab. The tracker only calls `GET /api/feedback` and has
no robot command path.

The live IMU card shows measurements only. The combined JSON API renames the
robot's legacy `body_pitch_target_deg` field to
`rear_pose_pitch_reference_deg`: it is fixed metadata captured during the
known rear-lean body-frame calibration pose, not a current measurement or a
live controller target, and it is intentionally not displayed in the UI.

### One server owns the cameras

`hexapod-camera-server` is the only camera web server. Its page at `/` shows
one card per slot; frames, `/status.json`, `/api/cameras/health`, camera
leases and the fused `/api/poses` document are all served from it, and every
other tool (Robot Lab, the sysid runner, the calibration programs) reads
frames over HTTP rather than opening a device. Two processes opening one
camera fight over its active format and, on a shared USB controller, starve
each other.

Slots are numbered by `tools/camera_service.sh` from enumeration order at
every launch, so a slot number is not an identity. Pin, calibrate and exclude
cameras by their AVFoundation stable id, which `/status.json` reports per
slot as `requested_stable_id`.

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
With an intrinsics entry (`configs/camera_intrinsics_lab_20260912.json`), a camera that sees chassis tag
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

## Zero check (`hexapod-zero-check`)

Before a protocol that starts from the zero pose, Robot Lab asks the top
camera whether the pose looks like zero. The encoders sit on the servo output
shafts, so a horn whose screws have slipped passes the runner's start-pose
check while the leg points somewhere else; the lid tags cannot be fooled.
`hexapod-zero-check` re-derives each leg's azimuth from one observation with
the calibration program's geometry and compares it with
`leg_zero_azimuth_body_deg` in the installed layout. A two-lid leg gets 12 deg
of tolerance, a one-lid leg 18.

    uv run hexapod-zero-check                      # cameras, top camera 2
    uv run hexapod-zero-check --replay DIR --json  # a saved zero_tags.json

Exit 0 when every seen leg agrees, 2 when a leg is off, 3 when nothing could
be measured. The lab (`hexapod-lab2 zero-check`) combines it with the encoders:
only "encoders at zero, camera says a leg is off" holds the loop; "not at zero"
and "camera blind" are noted and the run goes ahead.

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
`hexapod_walker/prototype_sts3215/hexapod-tracker` Git submodule and runs the
camera server from it. The handheld iPhone survey and the browser calibration
studio that used to live here were removed on 2026-09-12 after
`hexapod-calibrate-tags` replaced them; see [`DEPRECATED.md`](DEPRECATED.md).

## Intrinsics (`hexapod-fit-intrinsics`)

Per-camera intrinsics live in `configs/camera_intrinsics_lab_20260912.json`,
keyed by camera identity (`stable_id`, then a unique `device_name`), never by
slot. An entry may pin the slot's `capture_size` to the mode it was fitted
for. Fit or refresh one camera from the running server:

```sh
uv run hexapod-fit-intrinsics --slot 1 --frames 40          # add --dry-run to only report
tools/camera_service.sh restart                             # load it
```

It needs at least three floor anchors that are not in a line and a view with
enough tilt that reprojection error actually depends on focal length; it
refuses to write otherwise and says what to change. The result is a
constrained single-plane fit (centred principal point, square pixels, zero
distortion) and is labelled provisional. Replace it with a multi-pose board
calibration before making metrology-grade 3-D claims.
