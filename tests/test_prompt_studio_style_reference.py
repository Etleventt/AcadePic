import unittest

from prompt_studio_app import INDEX_HTML, planner_user_prompt, stylist_user_prompt


class PromptStudioStyleReferenceTest(unittest.TestCase):
    def test_planner_prompt_mentions_reference_only_when_planner_uses_it(self):
        planner_prompt = planner_user_prompt(
            "method body",
            "diagram caption",
            "diagram",
            style_reference_mode="planner_and_stylist",
            has_style_reference_image=True,
        )
        planner_prompt_without_reference = planner_user_prompt(
            "method body",
            "diagram caption",
            "diagram",
            style_reference_mode="stylist_only",
            has_style_reference_image=True,
        )

        self.assertIn("Style Reference Image", planner_prompt)
        self.assertNotIn("Style Reference Image", planner_prompt_without_reference)

    def test_stylist_prompt_mentions_reference_when_enabled(self):
        stylist_prompt = stylist_user_prompt(
            "planner description",
            "method body",
            "diagram caption",
            "diagram",
            style_reference_mode="stylist_only",
            has_style_reference_image=True,
        )
        self.assertIn("Style Reference Image", stylist_prompt)

    def test_only_generate_prompts_sends_payload_base_true(self):
        self.assertEqual(INDEX_HTML.count("payloadBase(true)"), 1)
        self.assertIn("JSON.stringify(payloadBase(true))", INDEX_HTML)


if __name__ == "__main__":
    unittest.main()
