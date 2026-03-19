#!/usr/bin/env python3
"""
Remove yellow annotation arrows from split reference cells.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def inpaint_mask(arr: np.ndarray, mask: np.ndarray, max_iters: int = 200) -> np.ndarray:
    filled = arr.astype(np.float32).copy()
    remaining = mask.copy()

    for _ in range(max_iters):
        if not remaining.any():
            break

        updated = 0
        coords = np.argwhere(remaining)
        for y, x in coords:
            values = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    ny = y + dy
                    nx = x + dx
                    if 0 <= ny < remaining.shape[0] and 0 <= nx < remaining.shape[1] and not remaining[ny, nx]:
                        values.append(filled[ny, nx])
            if values:
                filled[y, x] = np.mean(values, axis=0)
                remaining[y, x] = False
                updated += 1
        if updated == 0:
            # Fallback: unresolved pixels become black.
            filled[remaining] = 0
            remaining[:] = False

    return np.clip(filled, 0, 255).astype(np.uint8)


def clean_arrow(image: Image.Image) -> Image.Image:
    arr = np.array(image.convert("RGB"))
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    yellow_mask = (r > 180) & (g > 170) & (b < 120)

    if yellow_mask.any():
        # Expand by a few pixels so bright arrow edges disappear as well.
        mask = yellow_mask.copy()
        for _ in range(3):
            expanded = mask.copy()
            expanded[:-1, :] |= mask[1:, :]
            expanded[1:, :] |= mask[:-1, :]
            expanded[:, :-1] |= mask[:, 1:]
            expanded[:, 1:] |= mask[:, :-1]
            expanded[:-1, :-1] |= mask[1:, 1:]
            expanded[1:, 1:] |= mask[:-1, :-1]
            expanded[:-1, 1:] |= mask[1:, :-1]
            expanded[1:, :-1] |= mask[:-1, 1:]
            mask = expanded

        arr = inpaint_mask(arr, mask)

    return Image.fromarray(arr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in sorted(input_dir.glob("*.png")):
        image = Image.open(path)
        cleaned = clean_arrow(image)
        cleaned.save(output_dir / path.name)

    print(output_dir)


if __name__ == "__main__":
    main()
