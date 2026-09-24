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
the robot and provisional floor IDs in the current tracker configuration.

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
- This estimates the optical camera relative to the board. It does not measure
  tag-to-joint-axis mounts on the robot.
- The fixed reference is valid only for a truly fixed phone. If the phone can
  be bumped, keep at least two board tags visible so the ordinary RGB solve can
  refresh the world reference.
