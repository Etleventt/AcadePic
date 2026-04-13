import unittest

from utils.style_reference import (
    build_style_reference_contents,
    build_style_reference_prompt_summary,
    normalize_style_reference_image,
    normalize_style_reference_mode,
    resolve_style_reference_targets,
    strip_style_reference_fields,
)


class StyleReferenceHelpersTest(unittest.TestCase):
    def test_normalize_style_reference_mode_accepts_known_values(self):
        self.assertEqual(normalize_style_reference_mode("off"), "off")
        self.assertEqual(normalize_style_reference_mode(" stylist_only "), "stylist_only")
        self.assertEqual(
            normalize_style_reference_mode("planner_and_stylist"),
            "planner_and_stylist",
        )

    def test_normalize_style_reference_mode_rejects_unknown_values(self):
        self.assertEqual(normalize_style_reference_mode("planner-only"), "off")
        self.assertEqual(normalize_style_reference_mode(None), "off")

    def test_normalize_style_reference_image_strips_data_url_prefix(self):
        self.assertEqual(
            normalize_style_reference_image("data:image/png;base64,YWJjMTIz"),
            "YWJjMTIz",
        )

    def test_resolve_style_reference_targets_returns_expected_stage_matrix(self):
        self.assertEqual(resolve_style_reference_targets("off", has_image=True), (False, False))
        self.assertEqual(
            resolve_style_reference_targets("stylist_only", has_image=True),
            (False, True),
        )
        self.assertEqual(
            resolve_style_reference_targets("planner_and_stylist", has_image=True),
            (True, True),
        )
        self.assertEqual(
            resolve_style_reference_targets("planner_and_stylist", has_image=False),
            (False, False),
        )

    def test_build_style_reference_contents_is_multimodal(self):
        contents = build_style_reference_contents(
            "YWJjMTIz",
            media_type="image/png",
            consumer="stylist",
        )
        self.assertEqual(len(contents), 2)
        self.assertEqual(contents[0]["type"], "text")
        self.assertEqual(contents[1]["type"], "image")
        self.assertEqual(contents[1]["source"]["data"], "YWJjMTIz")

    def test_build_style_reference_prompt_summary_mentions_consumer(self):
        self.assertIn("planner", build_style_reference_prompt_summary("planner").lower())
        self.assertIn("stylist", build_style_reference_prompt_summary("stylist").lower())

    def test_strip_style_reference_fields_removes_raw_image_fields(self):
        payload = {
            "style_reference_mode": "stylist_only",
            "style_reference_image_base64": "secret",
            "style_reference_image_media_type": "image/png",
            "style_reference_image_filename": "ref.png",
        }
        cleaned = strip_style_reference_fields(payload)
        self.assertEqual(cleaned["style_reference_mode"], "stylist_only")
        self.assertNotIn("style_reference_image_base64", cleaned)
        self.assertNotIn("style_reference_image_media_type", cleaned)
        self.assertNotIn("style_reference_image_filename", cleaned)


if __name__ == "__main__":
    unittest.main()
