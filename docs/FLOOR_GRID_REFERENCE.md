# Robot Lab AprilTag floor grid — IMG_3837

Current floor-layout evidence recorded 2026-09-11. The canonical machine-readable reference is [`configs/floor_grid_reference.json`](../configs/floor_grid_reference.json). Vision serves it at `GET /api/vision/floor-grid-reference` (advertised by `/api/vision/health`) when running this revision.

## Evidence and frame

The user supplied actual OpenCV ArucoDetector DICT_APRILTAG_36h11 decoding of IMG_3837.jpeg (1152 × 1536). The image itself was not supplied to this checkout; no independent re-decoding is claimed. Thirteen floor tags and five moving robot tags were decoded; this is not a complete inventory. Blank cells have no decoded visible tag.

Spacing is user-stated one foot, assumed center-to-center (0.3048 m). Rows run far-to-near; columns left-to-right. Origin is tag 101; +X toward 103; +Y away from the camera; +Z upward. This is right-handed. X = column × 0.3048 m, Y = −row × 0.3048 m, Z = 0. Positions are inferred nominal grid sites, not surveyed ground truth.

| Row | Left 0 | Middle 1 | Right 2 |
|---|---|---|---|
| 0 | 101 | 100 | 103 |
| 1 | — | — | 102 |
| 2 | 105 | 112 | 104 |
| 3 | — | 111 | 110 |
| 4 | 106 | 118 | 115 |
| 5 | 116 | — | — |

| ID | Row | Column | X m | Y m | Z m | u px | v px |
|---|---|---|---|---|---|---|---|
| 100 | 0 | 1 | 0.3048 | 0 | 0 | 555 | 42 |
| 101 | 0 | 0 | 0 | 0 | 0 | 297 | 51 |
| 102 | 1 | 2 | 0.6096 | -0.3048 | 0 | 840 | 208 |
| 103 | 0 | 2 | 0.6096 | 0 | 0 | 811 | 32 |
| 104 | 2 | 2 | 0.6096 | -0.6096 | 0 | 876 | 422 |
| 105 | 2 | 0 | 0 | -0.6096 | 0 | 261 | 449 |
| 106 | 4 | 0 | 0 | -1.2192 | 0 | 197 | 1018 |
| 110 | 3 | 2 | 0.6096 | -0.9144 | 0 | 920 | 682 |
| 111 | 3 | 1 | 0.3048 | -0.9144 | 0 | 580 | 690 |
| 112 | 2 | 1 | 0.3048 | -0.6096 | 0 | 568 | 435 |
| 115 | 4 | 2 | 0.6096 | -1.2192 | 0 | 970 | 1008 |
| 116 | 5 | 0 | 0 | -1.524 | 0 | 142 | 1443 |
| 118 | 4 | 1 | 0.3048 | -1.2192 | 0 | 583 | 1013 |

Image centers are rounded averages of detected corners (u right, v down from top left), for locating tags only. The JSON preserves every supplied canonical corner, nominal and estimated yaw, and robot-tag pixel position.

## Calibration limits and corner correspondence

Use only the listed floor IDs as fixed world references. Robot IDs **3, 10, 59, 61, 113 move** and must never enter the floor map. Their observed pixels are respectively (41,190), (125,179), (96,106), (142,221), (56,83).

Black-border tag side length, camera intrinsics and robot mounting transforms are **unknown (`null`)**. Grid spacing is not tag size. A floor homography is valid on the floor plane; applying it to elevated robot tags produces height-dependent error. Full robot 3D geometry needs camera calibration and additional constraints such as measured tag sizes, mounting transforms or multiple views.

All 13 tags have nominal canonical top toward +Y and canonical right toward +X, with zero roll/pitch/yaw. OpenCV canonical corners are [top-left, top-right, bottom-right, bottom-left], identity-aware rather than image-position sorted. Explicitly map other detectors' ordering and axes. Local +x is canonical left-to-right, +y bottom-to-top, +z out of the printed face. Positive yaw rotates local +x toward world +Y about +Z.

For measured side s, canonical local corners are `[-s/2,+s/2,0]`, `[+s/2,+s/2,0]`, `[+s/2,-s/2,0]`, `[-s/2,-s/2,0]`. Transform by `p_world = center_world + Rz(yaw) * p_local`. Do not generate metric corners until s is measured.

Estimated yaw values were fitted through an image-to-floor homography of nominal centers. Integer pixels, low resolution, distortion, placement and inferred grid limit accuracy; decimals do not imply precision. Use zero yaw for an idealized map, estimates only as calibration initial values. Tag 105 has the largest apparent deviation (−3.9°).

## Consumer migration

This is the latest supplied floor reference. Older `floor_tag_map.json`, `floor_tag_map_20260903.json` and the floor portion of `hexapod-1-apriltag-layout.json` describe the previous tag-104-origin frame and assumed tag size. They remain legacy runtime inputs, not evidence of the current physical layout. This reference is deliberately distinct from their executable calibration schemas: inserting null size into their corner/PnP solvers would fail, while substituting an old size would assert an unmeasured quantity.

Consumers can read `floor_tag_centers_m` for nominal floor-center matching now. Before metric corner-based pose calibration, measure size, validate the grid, explicitly migrate frame conventions, and invalidate/recalibrate old camera extrinsics. Do not silently merge old robot mount inventories with the five detections or infer missing floor IDs.
