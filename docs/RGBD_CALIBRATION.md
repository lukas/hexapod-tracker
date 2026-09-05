# iPhone RGB-D calibration

This workflow uses AprilTag image corners for precise lateral position and
yaw, and registered iPhone LiDAR depth for distance and floor-plane tilt. The
result is a fixed-camera world transform that the existing pose tracker can
use even when the calibration board is no longer visible.

It is deliberately camera-only and read-only. No module in this path sends a
robot command.

## What is included

- `hexapod-calibration-board` generates an exact-size tag36h11 SVG and JSON map.
- `hexapod-rgbd-calibrate` captures a Record3D USB stream or replays NPZ frames.
- `rgbd_calibration.py` rejects weak depth, robustly fits a plane, combines RGB
  reprojection and depth-plane residuals, and averages a stationary sequence.
- `hexapod-track --record3d-device 0` tracks from the same RGB stream while
  applying the intrinsics paired with each frame.
- `AprilTagPoseTracker` falls back to `fixed_camera_world_reference` when mapped
  floor tags are not visible. Visible mapped tags still refresh the reference.

The optional Record3D backend is version 1.4.1 or newer and is intended for
Record3D iOS 1.10 or newer, where confidence-map streaming is available. Its
upstream project is <https://github.com/marek-simonik/record3d>. Record3D has
no prebuilt macOS wheel, so the setup command temporarily supplies CMake while
building it. The Xcode Command Line Tools are also required; CMake itself is
not added to the tracker's runtime dependencies.

## Physical setup

1. Print `configs/rgbd_calibration_board.svg` at actual size. Disable every
   fit-to-page or scaling option.
2. Measure a black square; it must be 70.0 mm. If the physical measurement is
   different, regenerate the SVG and manifest with the measured size rather
   than editing only the JSON.
3. Bond the page to a flat, rigid, matte surface. Paper curl directly becomes
   a plane error.
4. Put the board on the floor near the hexapod. Its center is world origin,
   printed right is +x, printed up is +y, and out of the printed face is +z.
5. Rigidly mount the iPhone in the final tracking position. Do not hand-hold it.
6. Connect it over USB, trust the computer, open Record3D, select the rear
   LiDAR camera, and enable USB streaming.
7. Keep the board and phone still while 30 accepted frames are collected.

The default board is 188 x 206 mm and fits A4 or US Letter. IDs 40–43 avoid
the robot and surveyed floor IDs in the current tracker configuration.

## Run calibration

```sh
uv run --with cmake uv sync --extra dev --extra rgbd
uv run hexapod-rgbd-calibrate \
  configs/apriltag_pose_config_20260831.json \
  --board configs/rgbd_calibration_board.json \
  --frames 30 \
  --min-tags 2 \
  --min-confidence 1 \
  --output artifacts/rgbd-calibration.json \
  --updated-config artifacts/apriltag_pose_config_rgbd.json \
  --preview-output artifacts/rgbd-calibration-preview.jpg \
  --preview
```

Press Q or Escape only if you want to stop early. At least eight accepted
frames are required. The generated tracker config preserves the 27 mm robot
tag size in `marker_size_m`, records the board's independent size in
`floor_marker_size_m`, replaces the old provisional floor map with the exact
board map, and stores both the RGB intrinsics and `fixed_camera_world_reference`.

Then track through the matching stream:

```sh
uv run hexapod-track artifacts/apriltag_pose_config_rgbd.json \
  --record3d-device 0 \
  --pose-output artifacts/poses.jsonl \
  --annotated-output artifacts/annotated.mp4 \
  --preview
```

The one-frame/no-preview behavior matches ordinary camera mode. Add
`--duration 30`, `--max-frames`, or `--preview` for a continuing capture.

Do not calibrate through Record3D and then silently switch to a Continuity
Camera or USB capture mode. A different lens, crop, stabilization mode, or
resolution can invalidate the intrinsics. The Record3D tracking adapter uses
the ARKit matrix attached to each RGB frame to avoid that mismatch.

## Handheld zero-pose tag survey

`hexapod-zero-survey` uses the same registered RGB, depth, confidence, and
intrinsics plus Record3D's ARKit camera trajectory. Unlike fixed-camera
calibration, the phone is meant to move during this workflow.

1. Put the robot in its configured zero pose and do not move it during the
   capture. The survey never sends a motor command, so set and support the pose
   before starting.
2. Put the rigid calibration board flat near the robot. Its printed axes remain
   the world frame for the entire survey.
3. Leave the configured L0 hip tag in its known orientation. It fixes leg
   numbering and the BuildViz body-axis direction. Chassis tag 0 fixes the body
   origin, but its historical yaw is deliberately not used as the axis gauge.
4. Start with the board filling a useful part of the view. Hold still until the
   preview advances from board lock to the walk-around step.
5. Walk slowly around the stationary robot. The dashboard shows the live image,
   an isometric 3-D tag/orientation map, the phone path, tracking health, and one
   checklist row per physical robot position. `FIND` means the position has not
   been seen; `VIEW` means its tag was decoded but needs another clean angle;
   `OK` means its 6-D pose was recorded. Vary the viewing angle rather than
   collecting all samples from one oblique direction.
6. Keep a mapped floor tag in frame whenever practical. A direct RGB-D solution
   from the mapped metric grid overrides accumulated ARKit drift. Each floor
   tag must also appear together with another mapped floor tag so the grid
   geometry is genuinely checked rather than self-validating one tag at a time.

The web studio supports both Record3D USB and Wi-Fi. USB keeps the uncompressed
depth/confidence stream and remains the recommended metric-calibration path.
Wi-Fi uses Record3D 1.11+'s WebRTC stream and synchronized per-frame pose and
intrinsics metadata; it requires Record3D's Wi-Fi Streaming extension and its
video-encoded depth can be noisier. The page highlights exactly one next target
in the checklist and 3-D map, displays reprojection/translation/rotation error
and phone speed, and suggests when to slow down, move closer, step sideways, or
return to the mapped floor grid. If either transport drops, stable world-frame
tag records remain in the atomic progress checkpoint. Continue the same run and
re-lock any visible mapped floor tag to align the new ARKit session.

```sh
uv run hexapod-zero-survey \
  configs/apriltag_pose_config_20260831.json \
  --board configs/rgbd_calibration_board.json \
  --robot-layout configs/hexapod-1-apriltag-layout.json \
  --body-anchor-tag-id 0 \
  --expected-floor-ids 12,13,15 \
  --output artifacts/zero-pose-tag-survey.json \
  --updated-config artifacts/apriltag_pose_config_surveyed.json \
  --preview-output artifacts/zero-pose-tag-survey.jpg
```

The production completion checklist has a finite definition: 13 horizontal
lid/chassis tags, 24 vertical angle tags (four per leg), and the six normally
visible mapped floor tags (100–105) must have stable estimates. Tag 112 is under
the zero-pose robot and is optional unless explicitly requested. It additionally
requires multi-tag floor
views to fit below 1.25 px RMS, a LiDAR plane below 12 mm RMS, mapped floor
positions below 10 mm RMS, heights below 6 mm RMS, and orientations below 3
degrees RMS. Coverage alone cannot enable Save.
The configured ID fills its position directly. If that ID is absent, a stable
unconfigured robot tag can fill the empty position when it is close to the
calibration-photo layout fitted from the recognized tags. The output records
both the configured and actual ID and marks the assignment as a replacement.
The L0 hip tag is a protected identity anchor because it labels the otherwise
symmetric legs; its configured ID is required unless a new one is explicitly
given with `--leg-zero-anchor-tag-id`. Unexpected decoded tags are recorded,
and an unconfigured tag close to the world floor with an upward-facing normal
is classified as ground. A never-seen, unlisted physical marker cannot be
inferred; add its ID to the input config or `--expected-floor-ids` when it must
gate completion. Unknown tags use the global robot marker size unless
`--survey-marker-size-mm` is supplied. Known floor, robot, and board tags may
each specify their own `marker_size_m`.
Two internally consistent but separated poses for one ID are flagged as a
possible duplicate print or ARKit tracking jump and cannot satisfy completion.
The dashboard still reports that ID as seen and asks for another view; it does
not misleadingly call the tag unseen.

The JSON output retains, for each tag:

- `world_from_tag` translation and quaternion;
- XYZ Euler angles and the decoded tag x/y/normal axes in world coordinates;
- decoded +y heading in the floor plane, height, and normal error from up;
- observations, reprojection error, and translation/rotation spread;
- role (`robot`, `ground`, `calibration_anchor`, or `unassigned`) and stability.

It also records every pairwise ground/anchor center distance, planar distance,
and XYZ delta. The optional updated config replaces surveyed expected floor
poses with the new measurements, carries per-ID marker sizes, and updates stable
robot tag mounts. The final offline pass jointly fits every full-resolution tag
corner from a coverage- and viewpoint-selected set of at most 32 keyframes,
every selected camera pose, every robot tag, and every observed floor tag. All
source photos remain archived. Registered, confidence-filtered LiDAR samples
from the interior of sufficiently large tags add point-to-tag-plane range
constraints; RGB corners remain the lateral and orientation measurement. Floor
coordinates are treated as metric priors only when the floor map has
`"reference_status": "surveyed"`; their `position_uncertainty_m` and
`yaw_uncertainty_deg` values set the prior weights. A `provisional` grid remains
a set of loose coplanar landmarks even when it carries estimated uncertainty.
The selected floor tag defines the coordinate origin after the fit.

Long stream gaps form separate trajectory segments. The solver may estimate a
small rigid movement of the whole robot between segments, preventing a nudge
during a reconnect from being hidden by rejecting floor observations or
warping individual tag locations. Excessive movement fails the quality gate.
The report separately exposes accepted-image RMS, all-observation RMS, LiDAR
range error, floor rejection fraction, and recovered reconnect movement. It
also writes representative `reprojection-audit-v5` images with detected corners
in green, predicted corners in magenta, and a line showing every corner error.
BuildViz supplies only initial leg/face branches and a post-fit discrepancy
report. The unchanged L0 hip tag labels the body orientation; if it was
replaced, specify the new reference explicitly before saving.

To promote a floor map from provisional to measured, update each
`world_from_tag.translation_m` in the robot layout, set
`floor.reference_status` to `surveyed`, and set realistic uncertainty values.
Coordinates are tag centers relative to the chosen origin tag; X/Y center
measurements are sufficient, while measured tag yaw is optional. A single tag
may instead carry its own `reference_status: surveyed` for a partially measured
map.

If the robot is stationary at known nonzero angles, pass a JSON object mapping
all or some joint names to degrees with `--joint-angles-json`. Omitted joint
angles are treated as zero.

### What “learn geometry” means here

A single static pose measures tag-center locations, orientations, and
inter-tag baselines. Given the existing kinematic dimensions and a trusted body
anchor, it can solve a new full `frame_from_tag` for a tag that was reattached.
It cannot uniquely distinguish an error in a link length from an error in that
tag's mount translation, nor locate an unmarked knee axis. The generated
`mount_learning` report labels these as unidentifiable instead of reporting
them as exact geometry.

A future exact-geometry fit needs several stationary captures with independently
known encoder angles spanning each joint, the existing tibia-fixed side tags,
and at least one trusted chassis datum. Those observations make joint axes,
link lengths, and mount offsets separable. The current output preserves the
per-tag poses and static baselines needed to inspect such a dataset, but does
not claim that multi-pose optimizer exists yet.

## Acceptance checks

A frame is rejected unless it has the requested visible tags and enough depth
samples with the requested confidence. Defaults then require:

- depth between 0.20 and 4.0 m;
- at least 40 usable plane samples;
- at least 55% inliers within 18 mm of one plane;
- fitted-plane RMS no worse than 18 mm;
- RGB and depth plane normals within 25 degrees.

The session rejects observations more than 25 mm or 2 degrees from the robust
center. A good rigid indoor setup should be comfortably better than those
failure gates. Treat the reported translation/rotation spread and median RGB
and depth residuals as evidence; successful completion alone is not a
millimeter-accuracy claim.

Recalibrate whenever the phone mount or board-defined world frame moves. Also
recalibrate after changing the lens/capture mode, or when the tracker reports a
different image aspect ratio.

## Offline replay format

For reproducible debugging, pass `--npz-dir DIRECTORY` instead of a live
device. Files are read in sorted order and must contain:

- `rgb`: an OpenCV BGR `H x W x 3` array, or a grayscale `H x W` array;
- `depth`: a registered `Hd x Wd` float array in metres;
- `camera_matrix`: the 3 x 3 RGB intrinsic matrix at `W x H`;
- `confidence`: optional `Hd x Wd` values where 0/1/2 are low/medium/high.
- `camera_pose_xyzw_xyz`: required only by `hexapod-zero-survey`; Record3D's
  OpenGL/ARKit camera pose as quaternion x/y/z/w followed by translation x/y/z.

RGB and depth must be registered and have the same aspect ratio. The code
scales RGB intrinsics to the depth grid before unprojection and refuses an
aspect-ratio mismatch instead of inventing an extrinsic registration.

## Multiple cameras

For two or more fixed cameras, leave the board in the identical world pose and
run the workflow separately for every camera. That gives every view a
`world_from_camera` in the same board frame. It is the prerequisite for true
multi-view fusion, but this command does not synchronize frames or perform
stereo triangulation; the existing simple multi-camera viewer remains planar.

## Limits

- iPhone scene depth is coarse compared with AprilTag corners, so depth is a
  plane constraint—not a replacement for the visual detector.
- Glossy, black, transparent, distant, or edge-of-frame surfaces can have weak
  LiDAR returns. The confidence and RANSAC gates are intentional.
- Fixed-camera calibration estimates only the optical camera relative to the
  board. The separate zero-pose survey can update tag mounts against an existing
  kinematic model, subject to the body-anchor and geometry-identifiability limits
  above.
- The fixed reference is valid only for a truly fixed phone. If the phone can
  be bumped, keep at least two board tags visible so the ordinary RGB solve can
  refresh the world reference.
