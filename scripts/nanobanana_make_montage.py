#!/usr/bin/env python3
"""
Build a simple montage from split reference cells or generated results.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw


def slugify(value: str) -> str:
    return re.sub(r"[^\w.-]+", "_", value.strip(), flags=re.UNICODE).strip("_") or "item"


def load_image(path: Path, size: tuple[int, int] | None) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if size and image.size != size:
        image = image.resize(size, Image.Resampling.BICUBIC)
    return image


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    draw.text(xy, text, fill=(0, 0, 0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", required=True, help="Comma-separated row names")
    parser.add_argument("--cols", required=True, help="Comma-separated column names")
    parser.add_argument("--mode", choices=["reference", "run"], required=True)
    parser.add_argument("--reference-dir", help="Directory with split reference cells")
    parser.add_argument("--run-dir", help="Run directory created by nanobanana_batch.py")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cell-size", default="192x192")
    parser.add_argument("--padding", type=int, default=24)
    parser.add_argument("--header", type=int, default=48)
    parser.add_argument("--label", type=int, default=64)
    args = parser.parse_args()

    rows = [item.strip() for item in args.rows.split(",") if item.strip()]
    cols = [item.strip() for item in args.cols.split(",") if item.strip()]
    cell_w, cell_h = [int(x) for x in args.cell_size.lower().split("x", 1)]

    canvas_w = args.label + len(cols) * (cell_w + args.padding) + args.padding
    canvas_h = args.header + len(rows) * (cell_h + args.padding) + args.padding
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(235, 235, 235))
    draw = ImageDraw.Draw(canvas)

    for col_idx, col in enumerate(cols):
        x = args.label + args.padding + col_idx * (cell_w + args.padding)
        draw_text(draw, (x + 8, 12), col)

    for row_idx, row in enumerate(rows):
        y = args.header + args.padding + row_idx * (cell_h + args.padding)
        draw_text(draw, (10, y + cell_h // 2 - 8), row)

    for row_idx, row in enumerate(rows):
        for col_idx, col in enumerate(cols):
            if args.mode == "reference":
                if not args.reference_dir:
                    raise ValueError("--reference-dir is required for reference mode")
                path = Path(args.reference_dir) / f"{slugify(row)}__{slugify(col)}.png"
            else:
                if not args.run_dir:
                    raise ValueError("--run-dir is required for run mode")
                status_path = Path(args.run_dir) / slugify(row) / slugify(col) / "status.json"
                if not status_path.exists():
                    continue
                status = json.loads(status_path.read_text(encoding="utf-8"))
                best_path = status.get("best_image_path", "")
                if not best_path:
                    continue
                path = Path(best_path)

            if not path.exists():
                continue

            image = load_image(path, (cell_w, cell_h))
            x = args.label + args.padding + col_idx * (cell_w + args.padding)
            y = args.header + args.padding + row_idx * (cell_h + args.padding)
            canvas.paste(image, (x, y))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    print(output)


if __name__ == "__main__":
    main()
