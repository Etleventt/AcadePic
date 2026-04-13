import asyncio
import unittest
from unittest.mock import patch

from agents.planner_agent import PlannerAgent
from agents.stylist_agent import StylistAgent
from utils.config import ExpConfig


class PlannerStyleReferenceRoutingTest(unittest.TestCase):
    def test_planner_skips_reference_in_stylist_only_mode(self):
        captured = {}
        agent = PlannerAgent(
            exp_config=ExpConfig(
                dataset_name="PromptStudio",
                task_name="diagram",
                text_provider="openai_compatible",
                model_name="gpt-5.4",
            )
        )

        async def fake_call(*, contents, **kwargs):
            captured["contents"] = contents
            return ["planner output"]

        with patch(
            "agents.planner_agent.generation_utils.call_evolink_text_with_retry_async",
            side_effect=fake_call,
        ):
            asyncio.run(
                agent.process(
                    {
                        "content": "method body",
                        "visual_intent": "diagram caption",
                        "retrieved_examples": [],
                        "top10_references": [],
                        "style_reference_mode": "stylist_only",
                        "style_reference_image_base64": "YWJjMTIz",
                        "style_reference_image_media_type": "image/png",
                    }
                )
            )

        self.assertFalse(any(item.get("type") == "image" for item in captured["contents"]))

    def test_planner_uses_reference_in_planner_and_stylist_mode(self):
        captured = {}
        agent = PlannerAgent(
            exp_config=ExpConfig(
                dataset_name="PromptStudio",
                task_name="diagram",
                text_provider="openai_compatible",
                model_name="gpt-5.4",
            )
        )

        async def fake_call(*, contents, **kwargs):
            captured["contents"] = contents
            return ["planner output"]

        with patch(
            "agents.planner_agent.generation_utils.call_evolink_text_with_retry_async",
            side_effect=fake_call,
        ):
            asyncio.run(
                agent.process(
                    {
                        "content": "method body",
                        "visual_intent": "diagram caption",
                        "retrieved_examples": [],
                        "top10_references": [],
                        "style_reference_mode": "planner_and_stylist",
                        "style_reference_image_base64": "YWJjMTIz",
                        "style_reference_image_media_type": "image/png",
                    }
                )
            )

        self.assertTrue(any(item.get("type") == "image" for item in captured["contents"]))


class StylistStyleReferenceRoutingTest(unittest.TestCase):
    def test_stylist_uses_reference_in_stylist_only_mode(self):
        captured = {}
        agent = StylistAgent(
            exp_config=ExpConfig(
                dataset_name="PromptStudio",
                task_name="diagram",
                text_provider="openai_compatible",
                model_name="gpt-5.4",
            )
        )

        async def fake_call(*, contents, **kwargs):
            captured["contents"] = contents
            return ["stylist output"]

        with patch(
            "agents.stylist_agent.generation_utils.call_evolink_text_with_retry_async",
            side_effect=fake_call,
        ):
            asyncio.run(
                agent.process(
                    {
                        "content": "method body",
                        "visual_intent": "diagram caption",
                        "target_diagram_desc0": "planner output",
                        "style_reference_mode": "stylist_only",
                        "style_reference_image_base64": "YWJjMTIz",
                        "style_reference_image_media_type": "image/png",
                    }
                )
            )

        self.assertTrue(any(item.get("type") == "image" for item in captured["contents"]))


if __name__ == "__main__":
    unittest.main()
