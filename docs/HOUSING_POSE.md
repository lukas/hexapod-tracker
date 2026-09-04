# Housing-marker pose estimation

`apriltag_vision.py` detects tags in pixels and estimates their calibrated 6-D
camera/world poses. `housing_pose.py` then converts those rigid poses into the
chassis pose, observable joint angles, and (when all links are observed) foot
positions. Both are offline/read-only: they do not connect to or move the
robot, and they never write servo zeros.

## What each marker observes

| Physical marker location | Kinematic frame | Observable quantity |
| --- | --- | --- |
| Chassis-center/body-fixed marker | `body` | Chassis 6-DoF pose |
| Hip-servo housing | `L*_coxa` | Yaw angle |
| Knee-servo housing | `L*_femur` | Yaw + hip angles |
| Tibia | `L*_tibia` | Yaw + absolute tibia/knee angle |

The knee servo is bolted to the femur. Its housing does **not** rotate when its
own output shaft moves, so motor-housing markers alone provide 12 signed joint
angles, not 18. The tracker now detects the existing red boot tips instead of
requiring fragile tags on the feet. Foot-tip foreshortening remains available
as a low-confidence diagnostic, but it is excluded from safety and calibration
because an oblique phone view can make a straight tibia appear bent. Read-only
encoders provide the knee values. A rigid tibia/yoke tag remains the visual way
to measure a knee angle.

## Live phone zero/checkup view

If markers may have been reattached, run the guided handheld zero-pose survey
before using their old mount transforms. It records one tag in every named
robot mount position and every expected ground tag in the calibration-board
world frame, including full orientation and floor-tag distances, then writes a
separate reviewed config:

```sh
uv run hexapod-zero-survey \
  configs/apriltag_pose_config_20260831.json \
  --board configs/rgbd_calibration_board.json \
  --body-anchor-tag-id 0 \
  --output artifacts/zero-pose-tag-survey.json \
  --updated-config artifacts/apriltag_pose_config_surveyed.json
```

The anchor must be a chassis tag whose old mount is still trustworthy. The
survey relearns all other stable configured mounts from the known pose and
nominal kinematics; one static pose cannot independently identify exact link
lengths, joint-axis positions, and tag offsets. See
[`RGBD_CALIBRATION.md`](RGBD_CALIBRATION.md#handheld-zero-pose-tag-survey).
The configured L0 hip tag separately anchors leg numbering. If it is replaced,
pass its new ID explicitly with `--leg-zero-anchor-tag-id`; other missing robot
IDs can be matched automatically to empty physical positions.

On macOS, select the iPhone as a Continuity Camera and find its OpenCV camera
index (often 0 or 1). This opens a live overlay; press Q or Escape to stop:

```sh
uv run hexapod-track \
  configs/apriltag_pose_config_20260831.json \
  --camera 0 --preview \
  --pose-output phone_checkup.jsonl
```

While the preview has keyboard focus, press **C** to switch between camera
indexes 0 and 1 without ending the diagnostic recording. The active index and
controls appear at the bottom of the preview. Use `--camera-cycle 0,1,2` if
more cameras should participate in the cycle.

Live/video input is downscaled to 1280 pixels wide before detection by default;
this keeps the preview responsive while retaining the full decoded set in the
reference image. Use `--processing-width 960` for more speed or
`--processing-width 0` for full-resolution metrology. Raw video output, when
requested, keeps the capture resolution.

Add read-only servo/IMU feedback for a signed 18-joint pose and visual-vs-
encoder zero diagnosis:

```sh
uv run hexapod-track \
  configs/apriltag_pose_config_20260831.json \
  --camera 0 --preview \
  --robot-url http://hexapod.local:8080 \
  --pose-output phone_checkup.jsonl \
  --summary-output phone_checkup_summary.json
```

`--robot-url` only performs `GET /api/feedback`; the tracker has no POST path
and sends no motor command. The overlay and `full_pose.zero_check` tell the
operator which joints appear away from zero. They never invoke `set_zero` or
move a joint. That boundary is intentional until the camera, mounts, and
printed size have been physically calibrated and repeated stationary trials
show that visual/encoder errors are trustworthy.

### Browser vision lab

The React/TypeScript page on the shared `:8898` web hub replaces the dense OpenCV text overlay with
a clean video canvas, short marker IDs, a six-leg joint matrix, a calibration
readiness gate, a camera picker, and an explicitly acknowledged **Gait survey**
panel:

```sh
make vision-build
make vision                       # http://localhost:8898/vision
```

The shared local Python service leaves the camera off until **Start camera**
is pressed, then owns the selected camera and drops stale frames; React only
renders the newest image and state. A visual calibration starts
only after all 13 robot markers and at least two floor markers are directly
decoded, 18/18 encoder feedback is available, and twelve consecutive frames
are encoder-stationary. Visual jitter is measured rather than mistaken for
physical robot motion.

The Gait survey is the page's one motion-capable feature. It requires a live
robot connection, direct chassis tag 0 plus two floor tags, a SAFE Vision/IMU
preflight, selected gaits, and an operator motion acknowledgement. During a
run, the recorder temporarily owns the camera, logs raw video/timestamps,
motor/IMU telemetry, commands, and recovery/centering events, and adaptively
returns the chassis toward the operator-approved starting image anchor. This
is intentionally not the optical center: an oblique camera can project its
image center to an unsafe or unreachable floor point. Duplicate floor-tag IDs
are resolved by the lowest global corner-reprojection error instead of decode
order. Pre-trip warnings may use a bounded
pause → collision-aware safe-zero → stand → retry sequence. A tip, brownout,
confirmed hot motor, missing servo, hard-current event, or failed recovery
always stops and limps; it is never automatically retried. Heat requires three
consecutive over-threshold readings from the same joint. After a thermal stop,
raw video and telemetry continue until three complete readings are below the
warm threshold (or a five-minute timeout); the robot remains limp until the
operator explicitly authorizes a collision-aware safe-zero recovery. The
operator must remain present. A hard tilt stop likewise requires three valid
consecutive samples. An approximately 180-degree Euler jump is rejected when
it is discontinuous with the last trusted attitude and the gyro reports only
small motion; the raw sample and a `tilt_glitch_ignored` event are still saved.

After hardware motion ends, the Mac processes the raw video into AprilTag
JSONL plus annotated MP4, replays the selected protocol in MuJoCo, renders a
sim MP4, and writes `apriltag_motion.json`, a comparison report, and a manifest
in the consuming robot project's run directory. In `lukas/hexapod` these live
under `rl_move/hardware_traces/`. The raw recordings remain available even if
post-processing or a hardware trial fails.

Offline AprilTag processing samples the 30 Hz recording at 10 Hz by default.
The JSONL retains original video frame indexes and timestamps, and the
annotated video is written at the corresponding reduced frame rate so its
duration remains correct. Run `hexapod-track --frame-step 1` when a
full-frame diagnostic render is needed.

### Follow-up gait reliability protocol

Keep gait 0 and one candidate in the **same** survey. The paired ratio cancels
most camera-scale drift and is more trustworthy than comparing two videos
captured on different days. Start at 30 mm/s with 6.4–8 seconds in each
direction and collect at least three complete surveys. Advance a candidate to
35 or 40 mm/s only after it is 3/3 complete without recovery or a safety trip.
Do not promote a gait from raw path length: sideways drift and turning can look
fast. Use `commanded_axis_speed_mm_s` from `apriltag_motion.json`.

When this tracker is used from `lukas/hexapod`, aggregate any number of
completed run directories with that repository's reliability command:

```sh
uv run python -m rl_move.scripts.summarize_scripted_gait_reliability \
  --runs rl_move/hardware_traces \
  --output rl_move/hardware_traces/gait_reliability.json
```

The report retains every attempted phase, counts missing phases, reports
median/MAD/stdev/CV of measured speed, and computes within-run speed ratios
against gait 0. Keep the raw video, timestamp sidecar, telemetry, events,
AprilTag pose JSONL, motion report, MuJoCo replay, and manifest together; those
files are the reproducible experiment record.

If forward/backward performance is strongly asymmetric, repeat the paired
survey after physically rotating the robot 180 degrees while leaving the
camera and mapped floor tags fixed. A speed advantage that stays in the same
garage direction points to floor grade or surface friction; an advantage that
stays in robot-local forward/backward points to the gait/controller. Capture a
new starting anchor after the rotation. Do not compare unpaired sessions or
reuse the old image anchor.

The 2026-09-01 rotation test found gait 0's large forward/backward split in
both robot orientations, while gait 9 remained nearly symmetric. That result
points primarily to gait/controller asymmetry rather than garage grade. A
small floor contribution is still possible and should remain in MuJoCo domain
randomization.

For ordinary infrastructure or centering failures after motion, the suite now
stops, runs collision-aware safe-zero, verifies all 18 joints within 6°, and
limps. Confirmed hard trips are different: they limp immediately and never
command a recovery move. One or two impossible temperature bytes are recorded
as glitches; the same joint must cross the threshold on three consecutive
samples to be treated as real heat. A thermal stop keeps the camera and motor
telemetry recorder running through three consecutive cool readings. Safe-zero
afterward is an explicit supervised operator action, not an automatic retry.

The report robustly aggregates `visual_minus_encoder_deg` for the twelve
signed yaw/hip observations. Knees are reported as visually unobservable
without rigid tibia/yoke markers. Reports are saved beneath
`artifacts/apriltag_pose/calibrations/`. **Apply visual calibration** accumulates
the 12 stable residuals into `robot_pose.visual_joint_bias_deg`; it changes only
the vision model and never servo zeros or motor state. Until a checkerboard
calibration replaces the EXIF-derived camera intrinsics, the UI labels every
report **provisional**.

The preview also displays `POSE SAFETY: SAFE`, `UNSAFE`, or `UNVERIFIED`.
That verdict uses the live IMU as the primary tilt measurement, plus broad
encoder joint envelopes and observation coverage. Biased monocular tilt and
unsigned foot-tip knee estimates are calibration warnings, not physical
`UNSAFE` evidence. `UNVERIFIED` is not safe. Because one overhead camera cannot prove
that the chassis is physically supported, `safe_for_alignment_motion` remains
false unless the operator starts the program with `--robot-supported`; even
that flag only records the assertion and does not enable or send motion.

Each feature records `source`, `confidence`, and `occlusion_age_frames`.
Decoded tags and color-detected boots are measurements. Optical flow bridges a
brief decoder/segmentation miss; constant-velocity prediction is bounded to
eight frames by the supplied config. After that the feature becomes
unobservable instead of being silently extrapolated.

For walking video, every JSONL frame contains `full_pose.walking_check` and the
terminal prints a cross-frame `diagnostic_summary`: per-leg visibility,
floor-projection speed, maximum body tilt, persistent zero-pose errors, and
persistent visual/encoder disagreements. Use repeated asymmetry and persistent
disagreement to diagnose a leg; one monocular floor-projection speed is only a
possible slip signal because the image alone cannot prove foot contact.

## As-photographed map (2026-08-31)

The physical handwritten `0` is beside tag 1, so tag 1 is authoritative L0.
The other legs follow the repository sequence around the chassis. “Orientation”
below means the decoded tag **+Y/top direction**, in degrees clockwise from the
top edge of `apriltags.jpeg`; it is not a robot joint angle.

| Location in photo | Tag | Kinematic frame | Tag +Y orientation |
| --- | ---: | --- | ---: |
| Chassis center | 0 | `body` | +68.3° |
| L0 hip lid (handwritten 0), upper-left | 1 | `L0_coxa` | +108.3° |
| L0 knee lid, outer upper-left | 7 | `L0_femur` | +109.7° |
| L1 hip lid, lower-left | 4 | `L1_coxa` | −138.1° |
| L1 knee lid, outer lower-left | 14 | `L1_femur` | +39.2° |
| L2 hip lid, bottom | 6 | `L2_coxa` | +76.5° |
| L2 knee lid, outer bottom | 11 | `L2_femur` | +76.3° |
| L3 hip lid, lower-right | 5 | `L3_coxa` | −71.7° |
| L3 knee lid, outer lower-right | 9 | `L3_femur` | −161.2° |
| L4 hip lid, upper-right | 3 | `L4_coxa` | −131.1° |
| L4 knee lid, outer upper-right | 10 | `L4_femur` | +137.0° |
| L5 hip lid, top | 2 | `L5_coxa` | +166.7° |
| L5 knee lid, outer top | 8 | `L5_femur` | −11.8° |
| Floor, left/origin | 12 | world reference | −175.1° |
| Floor, upper-right | 15 | world reference | −1.1° |
| Floor, lower-right | 13 | world reference | +176.6° |

The machine-readable version is `apriltag_pose_config_20260831.json`. Its floor
map chooses tag 12 as `(0, 0, 0)` and records tags 13/15 relative to it. Do not
move those three floor tags after establishing the map.

## Photo, video, and live-camera use

Run from the `hexapod-tracker` repository root. A still image produces one
JSON record:

```sh
uv run hexapod-track \
  configs/apriltag_pose_config_20260831.json \
  --input /path/to/photo.jpg \
  --pose-output pose.json \
  --annotated-output annotated.jpg
```

An existing video produces JSONL (one JSON object per frame) and an annotated
MP4:

```sh
uv run hexapod-track \
  configs/apriltag_pose_config_20260831.json \
  --input /path/to/video.mov \
  --pose-output poses.jsonl \
  --annotated-output annotated.mp4
```

Capture one raw + annotated photo from camera 0:

```sh
uv run hexapod-track \
  configs/apriltag_pose_config_20260831.json \
  --camera 0 \
  --raw-output capture.jpg \
  --annotated-output capture_annotated.jpg \
  --pose-output capture_pose.json
```

Or record 10 seconds:

```sh
uv run hexapod-track \
  configs/apriltag_pose_config_20260831.json \
  --camera 0 --duration 10 \
  --raw-output capture.mp4 \
  --annotated-output capture_annotated.mp4 \
  --pose-output capture_poses.jsonl
```

The supplied configuration's iPhone intrinsics are an EXIF-derived first
estimate. The configured `allow_quarter_turn` and `allow_center_crop` adapt the
matrix to portrait/landscape and the common 16:9 Continuity Camera crop while
keeping the optical center explicit. Replace `camera_matrix` and
`distortion_coefficients` with a checkerboard calibration of the actual video
mode for accurate metric height and tilt. The three mapped floor tags then
solve the camera extrinsics in every frame. If none is visible, output is
camera-relative unless the config contains a measured
`fixed_camera_world_reference`; the iPhone RGB-D workflow can provide that
fixed reference for a rigidly mounted phone.

The physical black squares (excluding the white quiet zone) were measured as
27 mm on 2026-08-31. That value is recorded in `marker_size_m`, and the
photo-derived floor-tag translations were rescaled with it. The phone lens
calibration is still approximate, so metric positions and signed video-only
knee fits remain provisional until a checkerboard calibration is recorded.

## Input contract

Every transform is named `A_from_B`: it maps B-frame coordinates into A.
Quaternions are `[x, y, z, w]`. The generic example configuration is
`housing_pose_config.example.json`; its identity tag mounts are placeholders,
not measured values.

Detector output looks like:

```json
{
  "detections": [
    {
      "camera": "overhead",
      "tag_id": 0,
      "camera_from_tag": {
        "translation_m": [0.12, -0.04, 0.88],
        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]
      },
      "decision_margin": 80.0
    }
  ],
  "encoder_joint_deg": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}
```

Run it from the `hexapod-tracker` repository root:

```sh
uv run hexapod-housing-pose \
  configs/housing_pose_config.example.json detections.json \
  --output pose.json
```

For a tracker that already produces rigid-link poses, replace `detections`
with `frame_transforms`, keyed by `body`, `L0_coxa`, `L0_femur`, and so on.

## Mount calibration and identifiability

`frame_from_tag` is the fixed pose of the printed tag in its rigid robot frame.
Use the CAD mount transform or a mechanically indexed tag plate. Translation
errors mainly affect the reported body/foot positions; orientation errors bias
joint angles directly.

A tag's unknown rotation about a joint axis and that joint's unknown encoder
zero are mathematically indistinguishable. Video alone cannot solve both. Each
tag therefore needs either an indexed mounting orientation or one mechanically
known reference pose. After that one reference, many stationary video frames
can estimate `visual_minus_encoder_deg` automatically and consistently.

Treat those values as suggestions. Average repeated stationary captures,
reject poor/mount-residual frames, and inspect the result before changing the
robot calibration.
