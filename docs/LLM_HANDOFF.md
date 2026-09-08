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

`make check` currently runs the synthetic/off-robot Python tests and the React
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

This is the intended server whenever an operator asks to **show all cameras**,
**show the available cameras**, or **show both USB cameras**. It serves the
camera grid at `/`. Do not answer those requests by launching the React
`/vision` interface described below; that interface owns only one active camera
at a time and is for a different workflow.

```sh
uv run hexapod-camera-server \
  --indices 0 1 --host 0.0.0.0 --port 8766 \
  --rotate-180 0 1 \
  --robot-url http://hexapod.local:8080
```

The September 3 three-camera setup is saved as one named profile. Use it so
capture modes, rotations, and intrinsic calibration remain paired:

```sh
OPENCV_AVFOUNDATION_SKIP_AUTH=1 uv run hexapod-camera-server \
  --capture-profile lab-tracking \
  --host 127.0.0.1 --port 8766 \
  --robot-url http://192.168.4.39:8080
```

For that boot, native index 0 is `lukas's iPhone Camera`, while OpenCV indices
1 and 2 are the OV9281 devices. AVFoundation-native and OpenCV enumeration are
different namespaces; never assume an index identifies the same device in
both.

`configs/camera_capture_profiles.json` is the source of truth for a saved
setup, but both checked-in profiles describe the September 3 layout and no
longer match the hardware. As of 2026-09-08 there are four Arducam OV9281
modules attached and no Continuity Camera slot in use, so `lab-tracking`
(iPhone at index 0 plus two OV9281s) and `lab-usb-only` (two OV9281s at
indexes 1 and 2) both mis-describe the rig. Do not pass `--capture-profile`
until a profile is rewritten for the current cabling. The same staleness
applies to `configs/camera_intrinsics.json`, whose index-keyed entries assume
index 0 is the iPhone; supplying an empty `cameras` map is the safe way to run
the planar-only pose API without silently attaching iPhone intrinsics to an
Arducam.

The `lab-tracking` profile captures the iPhone's full 1920x1440 420v/NV12 source at
30 fps, processes a 1280x960 color preview, uses both OV9281 cameras at their
full 1280x800/100 fps mode, publishes 10 fps, rotates USB indices 1 and 2, and
loads `camera_intrinsics.json`. Full-resolution unscaled iPhone data remains
available on `/native-frame/0.nv12` (Y followed by interleaved UV) and
`/native-luma/0.png` (lossless luminance). Continuity Camera supplies decoded
8-bit video-range NV12, not Bayer sensor RAW or ProRAW.

The code default is port `8765`; `8766` is commonly used to avoid colliding
with an already-running local viewer. Each `CameraWorker` independently opens
an OpenCV AVFoundation index, requests MJPG input, drops excess capture frames,
detects tag36h11 markers, and publishes raw plus annotated JPEGs. It reconnects
after three consecutive capture failures.

Starting the standard CLI also starts every `CameraWorker`, so it turns on the
requested cameras. If the operator explicitly wants an inventory-only page
with cameras off, run this standalone `CameraHTTPServer` with dormant workers
(construct the workers but do not call `worker.start()`). Verify
`/status.json` reports `frames: 0` for each camera. This is still the
multi-camera server; the dormant requirement is not a reason to use `/vision`.

Do not use `AVFoundationYuvCapture.device_descriptors()` to infer the numeric
indices accepted by OpenCV `VideoCapture`; the two APIs can enumerate the same
devices in different orders. Once capture is enabled, verify identity from the
live images and capture modes. On the September 3 setup, the two OV9281 feeds
reported 1280x800 at 100 fps, while the Studio Display feed reported 1280x720
at 30 fps, but these numeric indices remain ephemeral.

#### USB bus bandwidth, not camera count, sets the ceiling

Measured 2026-09-08 on the lab Mac Studio (`Mac15,14`) with four Arducam
OV9281 modules attached.

The OV9281 modules advertise no compressed format. AVFoundation reports only
uncompressed `420v` (320x240, 640x480, 800x600, 1280x720, and 1280x800 at
100–120 fps) plus `yuvs` 1280x800 at 10 fps, and MJPG requests are refused
(`mjpg_request_accepted: false`). Every OV9281 stream therefore costs full raw
bytes on its bus. The 12MP AF module is different — see below.

The 2026-09-08 rig went through three cablings, and the difference is
instructive. With all four modules on a single hub they shared one USB 2.0
domain:

```text
AppleT8122USBXHCI@04000000
  USB2.0 Hub@04100000                    480 Mbps, shared by all four
    Arducam @04110000 .. @04140000       Device Speed = 2
```

Two of them streamed 1280x800 cleanly while a third opened and received no
frames. Moving two cameras onto their own controllers did not simply raise
that count: the two direct cameras became flawless in every combination, while
the pair still behind the hub got *worse* as unrelated streams started
elsewhere. Both hub cameras failed once four ran at once, one delivering torn
frames and the other an all-green invalid frame, and through `camera_server`
(which adds tag detection and JPEG encoding per frame) even a single hub camera
tore alongside two direct ones.

Giving every camera its own host controller resolved it completely. The final
working rig is four cameras on four controllers, all clean at once:

```text
@01000000  Arducam OV9281  @01100000
@02000000  Arducam OV9281  @02100000
@08000000  Arducam OV9281  @08400000   (ASMedia controller)
@04000000  12MP AF Camera  @04120000   (alone behind USB2.0 Hub@04100000)
```

So the rule is **one camera per USB host controller**, not a byte budget per
bus. A hub adds no bandwidth, and a USB 3 hub does not help these USB 2.0
devices, which share that hub's single USB 2.0 upstream. The host has six
USB XHCI controllers (`@00000000`–`@05000000`) plus the ASMedia one, so there
is room to keep them separate. Confirm a split worked by checking that the
`ioreg` `locationID` prefixes differ.

Do not trade resolution for camera count. Dropping to 320x240 does let more
cameras stream at once, but the operator has ruled that out: full-resolution
feeds are the requirement, because low-resolution AprilTag pose is not useful
for this work.

Prefer `--native-avfoundation` for these cameras. The native adapter delivered
pristine 1280x800 frames with 14–17 tags detected, while the OpenCV
`AVFOUNDATION` backend produced torn frames in every test here and silently
ignores `--camera-mode` width and height. The OpenCV runs were confounded by
the same-device contention described next, so that backend is not proven at
fault; the native path is simply the one verified clean. Note that
`camera_server.py` hardcodes the native `preferred_sizes` to
`((1920, 1440), (1920, 1080), (1280, 720))`, so `--camera-mode` cannot select a
native capture size.

Torn, blocky frames usually mean two processes opened the same camera, not a
saturated bus. When this server and the main repository's `:8898` vision
runtime each held one device, that feed came back sheared into displaced blocks
and its tag count collapsed to 0–4, while the same camera alone was pristine.
Rule out a second owner before blaming bandwidth; the stop-route caveat under
the React UI section below explains the usual second owner.

#### Frame durations must come from the device, not from the rate

`_configure_device` pins capture rate by setting the active min/max frame
duration. A duration synthesised from the requested rate is not always
accepted: the 12MP AF module advertises its fixed 30 fps as the exact rational
`1000000/30000030`, and rejects `CMTimeMakeWithSeconds(1/30)` with

```text
NSInvalidArgumentException -[AVCaptureDevice setActiveVideoMinFrameDuration:]
Not supported - Supported ranges: (... 30.00 - 30.00 (1000000 / 30000030 ...))
tried to set maxFrameRate to 30.000031
```

The camera then opens and delivers nothing. `_frame_duration` therefore reuses
an advertised range's own `maxFrameDuration()` whenever that range pins a
single rate, and only computes a duration for a range that genuinely spans
rates. The OV9281 modules never hit this because their durations happen to
match the synthesised value; do not assume a new camera will.

#### The 12MP AF module

`12MP AF Camera` (uniqueID `0x412000032e40362`) is colour, unlike the mono
OV9281s, and advertises `420v` from 320x240 up to 4000x3000 at 15/10/5 fps --
macOS decodes its MJPEG stream, which is how a 12MP sensor fits on a USB 2.0
link. Because `camera_server` hardcodes the native `preferred_sizes`, it runs
at 1920x1080; 2592x1944 and 4000x3000 are available but need that list
changed. It also autofocuses, which is worth remembering before trusting it
for metric work: a refocus changes intrinsics, so a saved profile is only
valid while focus is fixed.

One OV9281 is physically mounted rotated about 90 degrees. `--rotate-180` is
the only rotation the server offers, so a 90-degree mount cannot be corrected
in software -- rotate it in hardware, or add 90/270 support.

Physically inverted cameras must be listed under `--rotate-180`. This rotates
frames before tag detection, annotation, raw/annotated JPEG encoding, and pose
snapshots. Never implement orientation as an `img` CSS transform: that makes
labels upside down and leaves displayed coordinates inconsistent with pose
coordinates.

The embedded HTML at `/` shows every requested camera. Relevant routes are:

- `/status.json`: capture backend/mode, frame counters, errors, and tag IDs.
- `/stream/<index>.mjpg`: annotated stream.
- `/raw-stream/<index>.mjpg`: raw stream.
- `/snapshot/<index>.jpg`: raw current frame.
- `/native-luma/<index>.png`: lossless unscaled native luminance, when the
  worker uses native AVFoundation.
- `/native-frame/<index>.nv12`: exact unscaled NV12 video frame, with size and
  pixel-format response headers, when the worker uses native AVFoundation.
- `/api/poses`: planar floor-referenced fusion from visible cameras.
- `/api/pose-state`: the planar camera result together with read-only encoder
  angles and calibrated IMU fields from the robot's `GET /api/feedback` route.
- `/calibration-status.json`: state produced by an optional external
  calibration capture directory.

The page's Pose tab presents camera pose, all 18 encoder angles, and IMU tilt.
Use body-frame roll/pitch only when `body_frame_calibrated` is true; retain the
sensor-frame values for diagnosis. Camera and encoder sources remain separate
until the USB cameras have validated metric 3-D calibration—do not label this
display as fused 6-D pose.

The upstream robot feedback has a legacy field named
`body_pitch_target_deg`. Despite that name, it is not a live measurement or a
currently commanded controller setpoint. It is the fixed pitch magnitude
recorded in the known rear-lean pose used to establish the IMU body-frame
axis. `FeedbackClient` accepts that legacy wire key but exposes it to tracker
clients as `rear_pose_pitch_reference_deg`. Keep this calibration metadata out
of the live IMU UI; showing it beside current roll and pitch makes a level
robot look as though it has a large pitch error.

Raw planar tag yaw is an absolute floor-world direction, while encoder values
use robot-relative `robot_abs`; never compare those raw numbers directly. The
standalone estimator does make one explicit conversion: a floor homography
rectifies headings parallel to the floor, and the documented
`frame_from_tag` rotations convert the horizontal chassis tag and each visible
horizontal coxa servo-lid tag into a robot-relative `L*_yaw`. Those values are
reported under `camera_joint_pose` with `joint_frame = robot_abs`. This assumes
the tag faces remain parallel to the floor and does not replace lens
calibration. Yaw uncertainty must come from paired same-camera relative
headings, tag-corner precision, and simultaneous cross-camera disagreement.
Do not add the chassis and coxa tags' absolute floor-heading bounds: those
bounds are strongly correlated and their shared homography component cancels
in the relative joint angle. Retain the provisional 5° 95% repeatability floor
until a larger stationary and moving validation dataset supports replacing it;
simultaneous cross-camera disagreement may raise the reported value.

For vertical yoke tags, the planar `parts` estimator reports
`projection_only`, sets part yaw and uncertainty to null, and retains the old
ray/floor intersection under `projection_diagnostic` for debugging only. The
joint-orientation path is separate: configured camera intrinsics plus chassis
tag `0` and any documented femur tag in the same view recover hip pitch from
relative tag rotations. Tag translation is not needed for this angle. The two
IPPE square-pose branches are scored against the robot's `Rz(yaw) * Ry(hip)`
kinematic model; high-residual branches are rejected. Documented tibia tags
use the same calculation to recover `robot_abs_tibia_v2`'s absolute
tibia/knee angle. The knee-servo lid itself is on the femur and cannot observe
its own output; one of that leg's rigid `L*_tibia` yoke tags must be visible.

`configs/camera_intrinsics.json` contains the September 3 provisional profiles
for the iPhone and both USB cameras. Each was fitted to 40 frames against the
floor grid while fixing the principal point, enforcing square pixels/zero
skew, and holding distortion at zero. Native camera 0 (iPhone Continuity
Camera, 1280x960) fit `f=1042.096 px` at `2.095 px` RMS; OpenCV camera 1 fit
`f=947.666 px` at `0.841 px` RMS; and OpenCV camera 2 fit `f=893.875 px` at
`1.333 px` RMS. This is sufficient for provisional hip and absolute tibia/knee
estimates but is not a substitute for a multi-pose ChArUco calibration.

Separate intrinsic from extrinsic calibration. A saved intrinsic profile
remains useful after the camera moves only while physical lens, zoom,
resolution, crop, orientation, and stabilization mode remain identical.
Camera extrinsics become stale as soon as the camera moves, but this viewer
fits each current view to visible surveyed floor tags, so it re-establishes
the world transform automatically rather than storing the old camera pose.
After a reconnect, camera indices must still be revalidated before attaching
an index-keyed intrinsic profile.

`PlanarPoseEstimator` fits a separate floor homography for every view and then
fuses ground-projected part estimates and horizontal tag headings in the
shared floor frame. Its optional intrinsic-calibrated path compares body and
femur orientation within each camera, then fuses the resulting robot-relative
angles; camera extrinsics cancel for same-view relative rotation. It is not
calibrated stereo and does not triangulate 3-D points between cameras.

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
Robot Lab using only its first-class versioned calibration endpoint.

Do not add robot-control HTTP calls here to make the standalone UI's survey
buttons work. That would break the intentional safety and ownership boundary.

`POST /api/vision/camera/stop` reports the camera as off without releasing the
capture device. After a stop, `/api/vision/state` shows `enabled: false`,
`status: "off"`, and `error: null`, yet the serving process can still hold the
AVFoundation device. Another process that tries to open it then fails with
`AVFoundationErrorDomain Code=-11817 "Cannot Use <device>"`, naming the holder
in `AVErrorPIDKey`. Read that PID instead of assuming a hardware fault or a
bandwidth limit; freeing the device requires restarting the holding process,
because a second stop call will not do it. Restarting the main repository's
`:8898` hub is not a free action — it is the robot-control surface, and
`make web-8898-restart` relaunches from the working tree, which changes the
served code. This route leaking its device is a bug worth fixing.

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

`hexapod-zero-survey` is the moving-phone companion. The production web flow
merges `configs/hexapod-1-apriltag-layout.json`, yielding 37 named robot mounts:
13 horizontal chassis/lid tags and 24 vertical yoke tags (four on each leg).
The six normally visible mapped floor tags (100–105) jointly align Record3D's
OpenGL/ARKit trajectory; tag 112 is normally covered by the zero-pose robot and
is therefore not required. It can be included explicitly when it is visible.
Their direct RGB-D solution takes precedence whenever visible. A slow walk then
aggregates every decoded tag's 6-D transform. Completion is one stable tag per
named robot mount position plus every expected visible floor ID, not the
continued presence of every old robot ID. A missing configured robot ID may be
replaced by a nearby stable newly discovered ID after fitting the original
calibration-photo layout to recognized tags. The configured L0 hip tag is the
protected leg-number reference unless the operator explicitly supplies its new
ID. The live dashboard distinguishes `not seen` from `seen, needs another view`
and shows an isometric tag/orientation map and phone path. The updated config
replaces surveyed floor poses and learns robot `frame_from_tag` values from a
known stationary pose. Chassis tag 0 fixes the body-origin translation while
the explicitly unchanged L0 hip tag aligns the survey to the BuildViz body
axes. After capture, an image-first bundle adjustment jointly refines up to 32
coverage- and viewpoint-selected full-resolution keyframes plus every robot and
floor tag. Confidence-filtered LiDAR samples inside sufficiently large tags add
point-to-tag-plane range factors. A floor map explicitly marked `surveyed` uses
its configured position/yaw uncertainties as metric priors; a `provisional`
map keeps floor tags as loose coplanar landmarks. One floor tag defines the
final origin. BuildViz is
used to initialize the leg/face branches and diagnose disagreement, not as a
fixed answer. Long capture gaps split the ARKit relative-motion prior because
each reconnect may reset the phone's coordinate frame, and a per-segment rigid
robot-motion variable prevents a physical nudge from warping the tag map. A
robust pass identifies the largest mutually consistent set of robot/floor
observations, and an ordinary least-squares finish reports calibration RMS only
on that set while recording every rejected observation and its residual.
Representative source images receive detected-versus-predicted corner overlays
in `reprojection-audit-v5`. A one-pose result still cannot independently
identify new link lengths or joint axes.
Coverage is necessary but not sufficient: the quality gate requires every
expected floor tag to be co-visible with another mapped tag, at least six
multi-tag reference frames, a bounded LiDAR floor-plane residual, at least two
accepted image views per floor tag, and bounded final image reprojection error.
Previous-grid position, height, orientation, and reprojection disagreements are
diagnostic because the purpose of the survey is to remeasure moved tags.
Estimates continue refining after first becoming stable. Known yoke
faces use their +Y/-Y surface normal to select the planar PnP branch.
The web workflow can receive the same synchronized RGB-D/pose data over
Record3D 1.11+ WebRTC Wi-Fi through a browser relay, though USB remains the
precision path. Atomic stable-landmark checkpoints survive stream loss; a
continued run re-locks the mapped floor grid because every new ARKit session has
a new arbitrary world frame. Live guidance selects one target and reports fit
error, tag spread, camera speed, and a concrete corrective movement.

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
- The planar floor map uses an operator-confirmed one-foot (304.8 mm) grid.
  Active visible anchors are tags 100, 101, 102, 103, 104, and 105; tag 112 is
  optional because it is normally under the robot. Yaw comes from two
  photographed planar rectifications. The configured 5 mm center uncertainty
  allows for placement and recollection error; perform a fresh physical survey
  before making finer absolute-position claims.
- The current Hexapod 1 and floor inventory has no duplicate IDs. Other loose
  prints in the garage are outside this map and must not be introduced without
  checking for collisions.
- Camera indexes are not identities. iPhone Continuity Camera and reconnecting
  USB devices reorder indexes. Repeated probes agree while the attached set is
  unchanged, but the order moved every time a device joined or left — a
  Continuity Camera appearing was enough to renumber the USB cameras — and
  `camera_server`'s in-process order has differed from a separate probe's at
  the same moment. Address a specific device by its AVFoundation `uniqueID`
  when identity matters, and otherwise confirm device names and live images
  after every rescan/restart. Device *names* now distinguish the mono
  `Arducam OV9281 USB Camera` from the colour `12MP AF Camera`, which makes
  `/status.json` enough to spot a mis-selected slot.

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
3. Release the capture device in `POST /api/vision/camera/stop` so stopping a
   camera actually frees it for other processes.
4. Calibrate each Arducam's intrinsics at every capture mode actually used.
5. Rewrite `camera_capture_profiles.json` and `camera_intrinsics.json` for the
   current four-Arducam cabling once the cameras are distributed across USB
   controllers, and key intrinsics by device `uniqueID` rather than by index.
6. Use one unmoved RGB-D board pose to establish a measured common frame if
   true multi-view 3-D or stereo claims are needed.
7. Re-survey floor-tag centers if accuracy finer than the current 5 mm prior is
   needed, and eliminate duplicate tag IDs.
8. Measure tag-to-joint-axis mount transforms or add component-local markers
   before trying to localize flex within an assembly.
9. Add recorded-camera regression clips with expected tag/pose summaries. Keep
   large media out of Git and document how to retrieve it.
10. Make package resources wheel-safe if this project will be installed outside
    a source checkout.

Before reporting a result, state separately: tag coverage, calibration quality,
pose/relative-motion result, encoder evidence, force evidence, and remaining
unobservable quantities.
