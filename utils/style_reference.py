from __future__ import annotations

from typing import Any


STYLE_REFERENCE_MODE_OFF = "off"
STYLE_REFERENCE_MODE_STYLIST_ONLY = "stylist_only"
STYLE_REFERENCE_MODE_PLANNER_AND_STYLIST = "planner_and_stylist"

VALID_STYLE_REFERENCE_MODES = {
    STYLE_REFERENCE_MODE_OFF,
    STYLE_REFERENCE_MODE_STYLIST_ONLY,
    STYLE_REFERENCE_MODE_PLANNER_AND_STYLIST,
}

STYLE_REFERENCE_RAW_FIELDS = {
    "style_reference_image_base64",
    "style_reference_image_media_type",
    "style_reference_image_filename",
}


def normalize_style_reference_mode(mode: Any) -> str:
    if not isinstance(mode, str):
        return STYLE_REFERENCE_MODE_OFF
    normalized = mode.strip().lower()
    if normalized in VALID_STYLE_REFERENCE_MODES:
        return normalized
    return STYLE_REFERENCE_MODE_OFF


def normalize_style_reference_image(image_b64: Any) -> str:
    if not isinstance(image_b64, str):
        return ""
    normalized = image_b64.strip()
    if not normalized:
        return ""
    if normalized.startswith("data:") and "," in normalized:
        _, normalized = normalized.split(",", 1)
    return normalized.strip()


def resolve_style_reference_targets(mode: Any, has_image: bool) -> tuple[bool, bool]:
    if not has_image:
        return False, False

    normalized_mode = normalize_style_reference_mode(mode)
    if normalized_mode == STYLE_REFERENCE_MODE_PLANNER_AND_STYLIST:
        return True, True
    if normalized_mode == STYLE_REFERENCE_MODE_STYLIST_ONLY:
        return False, True
    return False, False


def build_style_reference_prompt_summary(consumer: str = "stylist") -> str:
    consumer_name = (consumer or "stylist").strip().lower()
    if consumer_name == "planner":
        return (
            "Style Reference Image (planner): use the attached image only as aesthetic guidance for color, "
            "rendering feel, typography, icon treatment, and polish. Do not copy its semantic content."
        )
    return (
        "Style Reference Image (stylist): use the attached image only as aesthetic guidance for color, "
        "rendering feel, typography, icon treatment, and polish. Preserve the figure semantics."
    )


def build_style_reference_contents(
    image_b64: str,
    media_type: str = "image/png",
    consumer: str = "stylist",
) -> list[dict[str, Any]]:
    normalized_image = normalize_style_reference_image(image_b64)
    if not normalized_image:
        return []

    return [
        {"type": "text", "text": build_style_reference_prompt_summary(consumer)},
        {
            "type": "image",
            "source": {
                "type": "base64",
                "data": normalized_image,
                "media_type": (media_type or "image/png").strip() or "image/png",
            },
        },
    ]


def strip_style_reference_fields(payload_dict: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(payload_dict or {})
    for field in STYLE_REFERENCE_RAW_FIELDS:
        cleaned.pop(field, None)
    return cleaned
