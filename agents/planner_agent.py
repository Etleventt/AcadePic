# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Planner Agent - 根据方法章节生成图表的详细描述。
"""

import json
from typing import Any, Callable, Dict
import base64

from utils import generation_utils
from utils.style_reference import (
    build_style_reference_contents,
    normalize_style_reference_image,
    normalize_style_reference_mode,
    resolve_style_reference_targets,
)
from .base_agent import BaseAgent


class PlannerAgent(BaseAgent):
    """Planner Agent to generate images based on user queries"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = self.exp_config.model_name

        # Task-specific configurations
        if "plot" in self.exp_config.task_name:
            self.system_prompt = PLOT_PLANNER_AGENT_SYSTEM_PROMPT
            self.task_config = {
                "task_name": "plot",
                "content_label": "Plot Raw Data",
                "visual_intent_label": "Visual Intent of the Desired Plot",
            }
        else:
            self.system_prompt = DIAGRAM_PLANNER_AGENT_SYSTEM_PROMPT
            self.task_config = {
                "task_name": "diagram",
                "content_label": "Methodology Section",
                "visual_intent_label": "Diagram Caption",
            }

    async def process(
        self,
        data: Dict[str, Any],
        progress_callback: Callable[[Any], Any] | None = None,
    ) -> Dict[str, Any]:
        cfg = self.task_config
        print(f"[DEBUG] [PlannerAgent] 开始处理, task={cfg['task_name']}, provider={self.exp_config.text_provider}, model={self.model_name}")

        raw_content = data["content"]
        content = json.dumps(raw_content) if isinstance(raw_content, (dict, list)) else raw_content
        description = data["visual_intent"]

        content_list = []

        examples = data.get("retrieved_examples", [])
        if not examples:
            retrieved_ids = data.get("top10_references", [])
            ref_file = self.exp_config.work_dir / f"data/PaperBananaBench/{cfg['task_name']}/ref.json"
            if ref_file.exists():
                with open(ref_file, "r", encoding="utf-8") as f:
                    candidate_pool = json.load(f)
                id_to_item = {item["id"]: item for item in candidate_pool}
                examples = [id_to_item[ref_id] for ref_id in retrieved_ids if ref_id in id_to_item]

        user_prompt = ""
        for idx, item in enumerate(examples):
            user_prompt += f"Example {idx+1}:\n"

            item_content = item["content"]
            if isinstance(item_content, (dict, list)):
                item_content = json.dumps(item_content)

            user_prompt += f"{cfg['content_label']}: {item_content}\n"
            user_prompt += f"{cfg['visual_intent_label']}: {item['visual_intent']}\nReference {cfg['task_name'].capitalize()}: "
            content_list.append({"type": "text", "text": user_prompt})

            image_path = self.exp_config.work_dir / f"data/PaperBananaBench/{cfg['task_name']}" / item["path_to_gt_image"]
            with open(image_path, "rb") as f:
                ref_image_base64 = base64.b64encode(f.read()).decode("utf-8")
            content_list.append({"type": "image", "image_base64": ref_image_base64})
            user_prompt = ""

        style_reference_mode = normalize_style_reference_mode(data.get("style_reference_mode"))
        style_reference_image_base64 = normalize_style_reference_image(
            data.get("style_reference_image_base64")
        )
        style_reference_image_media_type = data.get(
            "style_reference_image_media_type", "image/png"
        )
        use_in_planner, _ = resolve_style_reference_targets(
            style_reference_mode,
            has_image=bool(style_reference_image_base64),
        )
        if use_in_planner:
            content_list.extend(
                build_style_reference_contents(
                    style_reference_image_base64,
                    media_type=style_reference_image_media_type,
                    consumer="planner",
                )
            )

        user_prompt += f"Now, based on the following {cfg['content_label'].lower()} and {cfg['visual_intent_label'].lower()}, provide a detailed description for the figure to be generated.\n"
        user_prompt += f"{cfg['content_label']}: {content}\n{cfg['visual_intent_label']}: {description}\n"
        user_prompt += "Detailed description of the target figure to be generated"
        if cfg["task_name"] == "diagram":
            user_prompt += " (do not include figure titles)"
        user_prompt += ":"

        content_list.append({"type": "text", "text": user_prompt})
        print(f"[DEBUG] [PlannerAgent] content_list 长度={len(content_list)}, 示例数={len(examples)}")

        # 根据 provider 路由 API 调用
        max_output_tokens_override = data.get("text_max_output_tokens")
        if isinstance(max_output_tokens_override, int) and max_output_tokens_override > 0:
            max_output_tokens = max_output_tokens_override
        else:
            max_output_tokens = generation_utils.resolve_text_max_output_tokens(
                model_name=self.model_name,
                provider=self.exp_config.text_provider,
                runtime_clients=self.exp_config.text_runtime_clients,
                fallback=50000,
            )
        if self.exp_config.text_provider == "openai_compatible":
            response_list = await generation_utils.call_evolink_text_with_retry_async(
                model_name=self.model_name,
                contents=content_list,
                config={
                    "system_prompt": self.system_prompt,
                    "temperature": self.exp_config.temperature,
                    "max_output_tokens": max_output_tokens,
                },
                max_attempts=5,
                retry_delay=5,
                runtime_clients=self.exp_config.text_runtime_clients,
                progress_callback=progress_callback,
            )
        else:
            from google.genai import types
            response_list = await generation_utils.call_gemini_with_retry_async(
                model_name=self.model_name,
                contents=content_list,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_prompt,
                    temperature=self.exp_config.temperature,
                    candidate_count=1,
                    max_output_tokens=max_output_tokens,
                ),
                max_attempts=5,
                retry_delay=5,
                runtime_clients=self.exp_config.text_runtime_clients,
            )

        for idx, response in enumerate(response_list):
            data[f"target_{cfg['task_name']}_desc{idx}"] = response.strip()

        print(f"[DEBUG] [PlannerAgent] 完成, 生成 {len(response_list)} 个描述, desc0 长度={len(response_list[0]) if response_list else 0}")
        return data




DIAGRAM_PLANNER_AGENT_SYSTEM_PROMPT = """
I am working on a task: given the 'Methodology' section of a paper, and the caption of the desired figure, automatically generate a corresponding illustrative diagram. I will input the text of the 'Methodology' section, the figure caption, and your output should be a detailed description of an illustrative figure that effectively represents the methods described in the text.

To help you understand the task better, and grasp the principles for generating such figures, I will also provide you with several examples. You should learn from these examples to provide your figure description.

** IMPORTANT: **
Your description should be as detailed as possible. Semantically, clearly describe each element and their connections. Formally, include various details such as background style (typically pure white or very light pastel), colors, line thickness, icon styles, etc. Remember: vague or unclear specifications will only make the generated figure worse, not better.
"""

PLOT_PLANNER_AGENT_SYSTEM_PROMPT = """
I am working on a task: given the raw data (typically in tabular or json format) and a visual intent of the desired plot, automatically generate a corresponding statistical plot that are both accurate and aesthetically pleasing. I will input the raw data and the plot visual intent, and your output should be a detailed description of an illustrative plot that effectively represents the data.  Note that your description should include all the raw data points to be plotted.

To help you understand the task better, and grasp the principles for generating such plots, I will also provide you with several examples. You should learn from these examples to provide your plot description.

** IMPORTANT: **
Your description should be as detailed as possible. For content, explain the precise mapping of variables to visual channels (x, y, hue) and explicitly enumerate every raw data point's coordinate to be drawn to ensure accuracy. For presentation, specify the exact aesthetic parameters, including specific HEX color codes, font sizes for all labels, line widths, marker dimensions, legend placement, and grid styles. You should learn from the examples' content presentation and aesthetic design (e.g., color schemes).
"""
