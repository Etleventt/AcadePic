"""
Utilities for exporting generated result images to disk.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .image_utils import save_base64_image_as_png


_CRITIC_KEY_RE = re.compile(r"^target_(?P<task>\w+)_critic_desc(?P<round>\d+)_base64_jpg$")


def get_final_image_key(result: dict[str, Any], task_name: str, exp_mode: str = "") -> str | None:
    """
    Resolve the best available final image key for a single result item.
    """
    critic_candidates: list[tuple[int, str]] = []
    for key, value in result.items():
        if not value:
            continue
        match = _CRITIC_KEY_RE.match(key)
        if match and match.group("task") == task_name:
            critic_candidates.append((int(match.group("round")), key))

    if critic_candidates:
        critic_candidates.sort(reverse=True)
        return critic_candidates[0][1]

    polished_key = f"polished_{task_name}_base64_jpg"
    if result.get(polished_key):
        return polished_key

    if "full" in exp_mode:
        stylist_key = f"target_{task_name}_stylist_desc0_base64_jpg"
        if result.get(stylist_key):
            return stylist_key

    planner_key = f"target_{task_name}_desc0_base64_jpg"
    if result.get(planner_key):
        return planner_key

    vanilla_key = f"vanilla_{task_name}_base64_jpg"
    if result.get(vanilla_key):
        return vanilla_key

    return None


def export_batch_result_images(
    results: list[dict[str, Any]],
    output_dir: str | Path,
    task_name: str = "diagram",
    exp_mode: str = "",
    filename_prefix: str = "candidate",
) -> list[Path]:
    """
    Export the final image for each result item as a PNG file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for idx, result in enumerate(results):
        final_image_key = get_final_image_key(result, task_name=task_name, exp_mode=exp_mode)
        if not final_image_key:
            continue

        saved_path = save_base64_image_as_png(
            result[final_image_key],
            output_dir / f"{filename_prefix}_{idx}.png",
        )
        if saved_path is not None:
            saved_paths.append(saved_path)

    return saved_paths
