"""Generate a dimensioned AprilTag grid for effortless RGB-D calibration."""
from __future__ import annotations

import argparse
import base64
import json
import math
from pathlib import Path
from typing import Sequence

import cv2


def _positive_millimetres(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise argparse.ArgumentTypeError("value must be a positive number")
    return number


def make_board_files(
    svg_path: Path,
    manifest_path: Path,
    *,
    rows: int = 2,
    columns: int = 2,
    first_id: int = 40,
    tag_size_mm: float = 70.0,
    gap_mm: float = 18.0,
    margin_mm: float = 15.0,
) -> dict[str, object]:
    """Write an exact-size printable SVG and the matching floor-tag manifest."""
    if rows <= 0 or columns <= 0:
        raise ValueError("rows and columns must be positive")
    if first_id < 0:
        raise ValueError("first_id cannot be negative")
    if min(tag_size_mm, gap_mm, margin_mm) <= 0.0:
        raise ValueError("tag size, gap, and margin must be positive")
    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_APRILTAG_36h11
    )
    marker_count = rows * columns
    if first_id + marker_count > len(dictionary.bytesList):
        raise ValueError("requested marker IDs exceed tag36h11 dictionary")

    board_width_mm = columns * tag_size_mm + (columns - 1) * gap_mm
    board_height_mm = rows * tag_size_mm + (rows - 1) * gap_mm
    page_width_mm = board_width_mm + 2.0 * margin_mm
    page_height_mm = board_height_mm + 2.0 * margin_mm + 18.0
    svg_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'width="{page_width_mm:.3f}mm" height="{page_height_mm:.3f}mm" '
            f'viewBox="0 0 {page_width_mm:.3f} {page_height_mm:.3f}">'
        ),
        f'<rect width="{page_width_mm:.3f}" height="{page_height_mm:.3f}" fill="white"/>',
    ]
    floor_tags: dict[str, object] = {}
    center_x = board_width_mm / 2.0
    center_y = board_height_mm / 2.0
    marker_pixels = 900
    for row in range(rows):
        for column in range(columns):
            tag_id = first_id + row * columns + column
            marker = cv2.aruco.generateImageMarker(
                dictionary, tag_id, marker_pixels, borderBits=1
            )
            ok, encoded = cv2.imencode(".png", marker)
            if not ok:
                raise RuntimeError(f"could not encode marker {tag_id}")
            payload = base64.b64encode(encoded.tobytes()).decode("ascii")
            x_mm = margin_mm + column * (tag_size_mm + gap_mm)
            y_mm = margin_mm + row * (tag_size_mm + gap_mm)
            svg_lines.append(
                f'<image x="{x_mm:.3f}" y="{y_mm:.3f}" '
                f'width="{tag_size_mm:.3f}" height="{tag_size_mm:.3f}" '
                f'image-rendering="pixelated" '
                f'xlink:href="data:image/png;base64,{payload}"/>'
            )
            tag_center_x_m = (
                column * (tag_size_mm + gap_mm) + tag_size_mm / 2.0 - center_x
            ) / 1000.0
            # Tag +y points toward its decoded top, hence rows below center are -y.
            tag_center_y_m = (
                center_y - row * (tag_size_mm + gap_mm) - tag_size_mm / 2.0
            ) / 1000.0
            floor_tags[str(tag_id)] = {
                "label": f"RGB-D calibration grid r{row} c{column}",
                "world_from_tag": {
                    "translation_m": [
                        round(tag_center_x_m, 9),
                        round(tag_center_y_m, 9),
                        0.0,
                    ],
                    "euler_xyz_deg": [0.0, 0.0, 0.0],
                },
            }
    outline_x = margin_mm
    outline_y = margin_mm
    svg_lines.extend([
        (
            f'<rect x="{outline_x:.3f}" y="{outline_y:.3f}" '
            f'width="{board_width_mm:.3f}" height="{board_height_mm:.3f}" '
            f'fill="none" stroke="#888" stroke-width="0.25"/>'
        ),
        (
            f'<text x="{margin_mm:.3f}" y="{page_height_mm - 8.0:.3f}" '
            'font-family="system-ui,sans-serif" font-size="4">'
            f'tag36h11 IDs {first_id}–{first_id + marker_count - 1} · '
            f'black square {tag_size_mm:.1f} mm · PRINT AT 100%</text>'
        ),
        '</svg>',
    ])
    manifest: dict[str, object] = {
        "schema_version": 1,
        "tag_family": "tag36h11",
        "marker_size_m": tag_size_mm / 1000.0,
        "board_size_m": [board_width_mm / 1000.0, board_height_mm / 1000.0],
        "world_frame": (
            "origin at board center; +x printed right; +y printed up; +z out of face"
        ),
        "print_instructions": (
            "Print the SVG at 100% / actual size on a matte rigid surface and "
            "verify one black square with calipers. Do not use fit-to-page."
        ),
        "floor_tags": floor_tags,
    }
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text("\n".join(svg_lines) + "\n", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a printable tag36h11 RGB-D calibration grid."
    )
    parser.add_argument("--svg", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--columns", type=int, default=2)
    parser.add_argument("--first-id", type=int, default=40)
    parser.add_argument("--tag-size-mm", type=_positive_millimetres, default=70.0)
    parser.add_argument("--gap-mm", type=_positive_millimetres, default=18.0)
    parser.add_argument("--margin-mm", type=_positive_millimetres, default=15.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = make_board_files(
        args.svg,
        args.manifest,
        rows=args.rows,
        columns=args.columns,
        first_id=args.first_id,
        tag_size_mm=args.tag_size_mm,
        gap_mm=args.gap_mm,
        margin_mm=args.margin_mm,
    )
    print(f"wrote {args.svg}")
    print(f"wrote {args.manifest}")
    print(manifest["print_instructions"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
