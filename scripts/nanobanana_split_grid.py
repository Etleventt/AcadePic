#!/usr/bin/env python3
"""
Split a composite reference image into an evenly spaced grid.

Typical uses:
1. 4x8 montage with headers/row labels around a dark inner panel
2. 4x1 GT stack on a black background
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image


def parse_names(raw: str | None, expected: int, prefix: str) -> list[str]:
    if not raw:
        return [f"{prefix}{idx + 1}" for idx in range(expected)]
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if len(values) != expected:
        raise ValueError(f"Expected {expected} names, got {len(values)}: {values}")
    return values


def slugify(value: str) -> str:
    value = value.strip()
    value = re.sub(r"[^\w.-]+", "_", value, flags=re.UNICODE)
    return value.strip("_") or "cell"


def detect_dark_bbox(image: Image.Image, threshold: int, pad: int) -> tuple[int, int, int, int]:
    gray = np.asarray(image.convert("L"))
    row_mean = gray.mean(axis=1)
    col_mean = gray.mean(axis=0)

    dark_rows = np.where(row_mean < threshold)[0]
    dark_cols = np.where(col_mean < threshold)[0]

    if dark_rows.size == 0 or dark_cols.size == 0:
        return (0, 0, image.width, image.height)

    left = max(int(dark_cols[0]) - pad, 0)
    right = min(int(dark_cols[-1]) + pad + 1, image.width)
    top = max(int(dark_rows[0]) - pad, 0)
    bottom = min(int(dark_rows[-1]) + pad + 1, image.height)
    return (left, top, right, bottom)


def split_bbox(
    bbox: tuple[int, int, int, int],
    rows: int,
    cols: int,
) -> list[tuple[int, int, int, int]]:
    left, top, right, bottom = bbox
    width = right - left
    height = bottom - top
    boxes: list[tuple[int, int, int, int]] = []

    for row in range(rows):
        y0 = top + round(row * height / rows)
        y1 = top + round((row + 1) * height / rows)
        for col in range(cols):
            x0 = left + round(col * width / cols)
            x1 = left + round((col + 1) * width / cols)
            boxes.append((x0, y0, x1, y1))
    return boxes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to the composite image")
    parser.add_argument("--output-dir", required=True, help="Directory for cropped cells")
    parser.add_argument("--rows", required=True, type=int)
    parser.add_argument("--cols", required=True, type=int)
    parser.add_argument("--row-names", help="Comma-separated row names")
    parser.add_argument("--col-names", help="Comma-separated column names")
    parser.add_argument("--dark-threshold", type=int, default=35, help="Mean grayscale threshold for dark panel detection")
    parser.add_argument("--pad", type=int, default=2, help="Padding added around the detected dark panel")
    parser.add_argument("--skip-bbox-detection", action="store_true", help="Split the full image without auto-cropping")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(input_path).convert("RGB")
    row_names = parse_names(args.row_names, args.rows, "row")
    col_names = parse_names(args.col_names, args.cols, "col")

    if args.skip_bbox_detection:
        bbox = (0, 0, image.width, image.height)
    else:
        bbox = detect_dark_bbox(image, threshold=args.dark_threshold, pad=args.pad)

    boxes = split_bbox(bbox, rows=args.rows, cols=args.cols)
    manifest: dict[str, object] = {
        "input": str(input_path.resolve()),
        "bbox": {"left": bbox[0], "top": bbox[1], "right": bbox[2], "bottom": bbox[3]},
        "rows": row_names,
        "cols": col_names,
        "cells": [],
    }

    for idx, box in enumerate(boxes):
        row_idx = idx // args.cols
        col_idx = idx % args.cols
        row_name = row_names[row_idx]
        col_name = col_names[col_idx]
        if args.cols == 1:
            filename = f"{slugify(row_name)}.png"
        else:
            filename = f"{slugify(row_name)}__{slugify(col_name)}.png"

        out_path = output_dir / filename
        image.crop(box).save(out_path)
        manifest["cells"].append(
            {
                "row": row_name,
                "col": col_name,
                "path": str(out_path.resolve()),
                "bbox": {"left": box[0], "top": box[1], "right": box[2], "bottom": box[3]},
            }
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(boxes)} cells to {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
