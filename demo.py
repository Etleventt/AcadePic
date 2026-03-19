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
PaperVizAgent 并行 Streamlit 演示
接受用户文本输入，复制 10 份，并行处理以生成多个图表候选方案供比较。
"""

import streamlit as st
import asyncio
import base64
import json
import queue
import threading
import uuid
from io import BytesIO
from PIL import Image
from pathlib import Path
import sys
import os
from datetime import datetime
from urllib import error as urllib_error
from urllib import request as urllib_request

# 将项目根目录添加到路径
sys.path.insert(0, str(Path(__file__).parent))

print("调试：正在导入代理模块...")
try:
    from agents.planner_agent import PlannerAgent
    print("调试：已导入 PlannerAgent")
    from agents.visualizer_agent import VisualizerAgent
    from agents.stylist_agent import StylistAgent
    from agents.critic_agent import CriticAgent
    from agents.retriever_agent import RetrieverAgent
    from agents.vanilla_agent import VanillaAgent
    from agents.polish_agent import PolishAgent
    print("调试：已导入所有代理模块")
    from utils import config
    from utils import generation_utils
    from utils.paperviz_processor import PaperVizProcessor
    from utils.result_export import export_batch_result_images, get_final_image_key
    from utils.image_utils import save_base64_image_as_png
    print("调试：已导入工具模块")

    import yaml
    config_path = Path(__file__).parent / "configs" / "model_config.yaml"
    model_config_data = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            model_config_data = yaml.safe_load(f) or {}

    def get_config_val(section, key, env_var, default=""):
        val = os.getenv(env_var)
        if not val and section in model_config_data:
            val = model_config_data[section].get(key)
        return val or default

except ImportError as e:
    print(f"调试：导入错误：{e}")
    import traceback
    traceback.print_exc()
    raise e
except Exception as e:
    print(f"调试：导入过程中发生异常：{e}")
    import traceback
    traceback.print_exc()
    raise e

st.set_page_config(
    layout="wide",
    page_title="PaperVizAgent 并行演示",
    page_icon="🍌"
)

@st.cache_resource
def get_runner_registry():
    """后台任务注册表，需要跨 Streamlit rerun 持久化。"""
    return {}

def clean_text(text):
    """清理文本，移除无效的 UTF-8 代理字符。"""
    if not text:
        return text
    if isinstance(text, str):
        # 移除导致 UnicodeEncodeError 的代理字符
        return text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
    return text

def base64_to_image(b64_str):
    """将 base64 字符串转换为 PIL 图像。"""
    if not b64_str:
        return None
    try:
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]
        image_data = base64.b64decode(b64_str)
        return Image.open(BytesIO(image_data))
    except Exception:
        return None


def list_history_json_files():
    """列出已保存的历史结果文件，按时间倒序。"""
    results_dir = Path(__file__).parent / "results" / "demo"
    if not results_dir.exists():
        return []
    return sorted(results_dir.glob("demo_*.json"), key=lambda path: path.stat().st_mtime, reverse=True)


def format_history_label(json_path: Path) -> str:
    """格式化历史记录标签，便于在下拉列表中查看。"""
    modified_time = datetime.fromtimestamp(json_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    size_mb = json_path.stat().st_size / (1024 * 1024)
    return f"{json_path.name} | {modified_time} | {size_mb:.1f} MB"


def infer_exp_mode_from_results(results):
    """根据结果字段推断流水线模式，兼容旧版历史 JSON。"""
    if not results:
        return "demo_planner_critic"

    for item in results:
        if any(key.startswith("target_diagram_stylist_desc0") for key in item.keys()):
            return "demo_full"
    return "demo_planner_critic"


def load_history_results(json_path: Path):
    """从历史 JSON 文件加载结果列表。"""
    with open(json_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    if not isinstance(loaded, list):
        raise ValueError("历史 JSON 格式不正确，预期为结果列表。")

    return loaded


def fetch_openai_compatible_models(base_url: str, api_key: str):
    """从 OpenAI-compatible 站点读取 /v1/models 列表。"""
    if not base_url:
        raise ValueError("缺少 Base URL。")

    url = f"{base_url.rstrip('/')}/v1/models"
    req = urllib_request.Request(url)
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")

    with urllib_request.urlopen(req, timeout=20) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        payload = json.loads(body)

    models = payload.get("data", [])
    if not isinstance(models, list):
        raise ValueError("模型列表格式不正确。")
    return models


def filter_model_ids(models, usage: str):
    """按用途筛选模型。usage: text | image"""
    filtered = []
    for item in models:
        model_id = item.get("id", "")
        description = str(item.get("description", "")).lower()
        lowered_id = model_id.lower()
        is_image = (
            "image generation" in description
            or "image" in lowered_id
            or "imagen" in lowered_id
        ) and "video" not in description and not lowered_id.startswith("veo")

        if usage == "image" and is_image:
            filtered.append(model_id)
        elif usage == "text" and not is_image:
            filtered.append(model_id)

    return filtered


def save_current_settings_to_config(
    *,
    text_provider: str,
    text_api_key: str,
    text_base_url: str,
    model_name: str,
    image_provider: str,
    image_api_key: str,
    image_base_url: str,
    image_model_name: str,
):
    """将当前 UI 配置写回 model_config.yaml，作为新页面默认值。"""
    config_path = Path(__file__).parent / "configs" / "model_config.yaml"
    config_data = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

    defaults = config_data.setdefault("defaults", {})
    defaults["model_name"] = model_name
    defaults["image_model_name"] = image_model_name

    ui_defaults = config_data.setdefault("ui_defaults", {})
    ui_defaults["text_provider"] = text_provider
    ui_defaults["image_provider"] = image_provider

    openai_cfg = config_data.setdefault("openai_compatible", {})
    google_cfg = config_data.setdefault("google_compatible", {})
    api_keys_cfg = config_data.setdefault("api_keys", {})

    if text_provider == "openai_compatible":
        openai_cfg["text_api_key"] = text_api_key
        openai_cfg["text_base_url"] = text_base_url
    else:
        google_cfg["text_api_key"] = text_api_key
        google_cfg["base_url"] = text_base_url
        api_keys_cfg["google_api_key"] = text_api_key

    if image_provider == "openai_compatible":
        openai_cfg["image_api_key"] = image_api_key
        openai_cfg["image_base_url"] = image_base_url
    else:
        google_cfg["image_api_key"] = image_api_key
        google_cfg["base_url"] = image_base_url
        api_keys_cfg["google_api_key"] = image_api_key

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_data, f, allow_unicode=True, sort_keys=False)

def create_sample_inputs(method_content, caption, diagram_type="Pipeline", aspect_ratio="16:9", num_copies=10, max_critic_rounds=3):
    """创建多份输入数据副本用于并行处理。"""
    base_input = {
        "filename": "demo_input",
        "caption": caption,
        "content": method_content,
        "visual_intent": caption,
        "additional_info": {
            "rounded_ratio": aspect_ratio
        },
        "max_critic_rounds": max_critic_rounds  # 添加评审轮次控制
    }

    # 创建 num_copies 份相同的输入，每份带有唯一标识符
    inputs = []
    for i in range(num_copies):
        input_copy = base_input.copy()
        input_copy["filename"] = f"demo_input_candidate_{i}"
        input_copy["candidate_id"] = i
        inputs.append(input_copy)

    return inputs

async def process_parallel_candidates(
    data_list,
    exp_mode="dev_planner_critic",
    retrieval_setting="auto",
    model_name="",
    image_model_name="",
    text_provider="openai_compatible",
    text_api_key="",
    text_base_url="",
    image_provider="openai_compatible",
    image_api_key="",
    image_base_url="",
    progress_callback=None,
    stop_event=None,
    prompt_only: bool = False,
):
    """使用 PaperVizProcessor 并行处理多个候选方案。"""
    print(f"\n{'='*60}")
    print(f"[DEBUG] process_parallel_candidates 开始")
    print(f"[DEBUG]   text_provider={text_provider}, image_provider={image_provider}, model={model_name}, image_model={image_model_name}")
    print(f"[DEBUG]   exp_mode={exp_mode}, retrieval={retrieval_setting}, candidates={len(data_list)}")
    print(f"[DEBUG]   text_api_key={'已设置 (' + text_api_key[:8] + '...)' if text_api_key else '未设置'}")
    print(f"[DEBUG]   image_api_key={'已设置 (' + image_api_key[:8] + '...)' if image_api_key else '未设置'}")
    print(f"{'='*60}")

    text_runtime_clients = generation_utils.create_runtime_clients(
        provider=text_provider,
        api_key=text_api_key,
        base_url=text_base_url,
    )
    image_runtime_clients = generation_utils.create_runtime_clients(
        provider=image_provider,
        api_key=image_api_key,
        base_url=image_base_url,
    )
    if not text_api_key:
        print(f"[DEBUG] ⚠️ 未提供文本 API Key，将回退到配置文件/环境变量")
    if not image_api_key:
        print(f"[DEBUG] ⚠️ 未提供图像 API Key，将回退到配置文件/环境变量")

    # 创建实验配置
    exp_config = config.ExpConfig(
        dataset_name="Demo",
        split_name="demo",
        exp_mode=exp_mode,
        retrieval_setting=retrieval_setting,
        model_name=model_name,
        image_model_name=image_model_name,
        provider=text_provider,
        text_provider=text_provider,
        image_provider=image_provider,
        work_dir=Path(__file__).parent,
        text_runtime_clients=text_runtime_clients,
        image_runtime_clients=image_runtime_clients,
    )
    print(f"[DEBUG] ExpConfig 已创建: text_provider={exp_config.text_provider}, image_provider={exp_config.image_provider}, model={exp_config.model_name}, image_model={exp_config.image_model_name}")

    # 初始化处理器及所有代理
    processor = PaperVizProcessor(
        exp_config=exp_config,
        vanilla_agent=VanillaAgent(exp_config=exp_config),
        planner_agent=PlannerAgent(exp_config=exp_config),
        visualizer_agent=VisualizerAgent(exp_config=exp_config),
        stylist_agent=StylistAgent(exp_config=exp_config),
        critic_agent=CriticAgent(exp_config=exp_config),
        retriever_agent=RetrieverAgent(exp_config=exp_config),
        polish_agent=PolishAgent(exp_config=exp_config),
    )

    # 并行处理所有候选方案（并发量由处理器控制）
    results = []
    concurrent_num = 3  # 默认并发量
    if text_provider == "openai_compatible":
        # OpenAI-compatible 文本代理在长 prompt 下更容易被并发压垮，默认串行更稳
        concurrent_num = 1

    try:
        async for result_data in processor.process_queries_batch(
            data_list,
            max_concurrent=concurrent_num,
            do_eval=False,
            progress_callback=progress_callback,
            stop_event=stop_event,
            prompt_only=prompt_only,
        ):
            results.append(result_data)
    finally:
        await generation_utils.close_runtime_clients(text_runtime_clients)
        await generation_utils.close_runtime_clients(image_runtime_clients)

    return results


def init_generation_state():
    """初始化生成任务状态。"""
    st.session_state.setdefault("generation_run_id", None)
    st.session_state.setdefault("generation_status", "idle")
    st.session_state.setdefault("generation_total", 0)
    st.session_state.setdefault("generation_completed", 0)
    st.session_state.setdefault("generation_candidates", {})
    st.session_state.setdefault("partial_results", {})
    st.session_state.setdefault("generation_error", "")
    st.session_state.setdefault("generation_timestamp", "")
    st.session_state.setdefault("generation_prompt_only", False)
    st.session_state.setdefault("current_run_json_file", "")
    st.session_state.setdefault("current_run_images_dir", "")


def reset_generation_state(clear_results: bool = False):
    """重置生成任务状态。"""
    for key, value in {
        "generation_run_id": None,
        "generation_status": "idle",
        "generation_total": 0,
        "generation_completed": 0,
        "generation_candidates": {},
        "partial_results": {},
        "generation_error": "",
        "generation_timestamp": "",
        "generation_prompt_only": False,
        "current_run_json_file": "",
        "current_run_images_dir": "",
    }.items():
        st.session_state[key] = value
    if clear_results:
        for key in ["results", "json_file", "images_dir", "exp_mode", "timestamp"]:
            st.session_state.pop(key, None)


def build_candidate_state(total_candidates: int):
    """创建候选方案状态表。"""
    return {
        idx: {
            "status": "queued",
            "stage": "queued",
            "message": "等待处理",
        }
        for idx in range(total_candidates)
    }


def persist_generation_results(results, exp_mode: str):
    """保存结果 JSON 和图片。"""
    results_dir = Path(__file__).parent / "results" / "demo"
    results_dir.mkdir(parents=True, exist_ok=True)

    json_filename = results_dir / f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_filename, "w", encoding="utf-8", errors="surrogateescape") as f:
        json_string = json.dumps(results, ensure_ascii=False, indent=4)
        json_string = json_string.encode("utf-8", "ignore").decode("utf-8")
        f.write(json_string)

    images_dir = json_filename.with_suffix("")
    saved_image_paths = export_batch_result_images(
        results,
        output_dir=images_dir,
        task_name="diagram",
        exp_mode=exp_mode,
    )
    return json_filename, images_dir, saved_image_paths


def init_current_run_output_paths():
    """为当前运行创建固定的实时输出路径。"""
    results_dir = Path(__file__).parent / "results" / "demo"
    results_dir.mkdir(parents=True, exist_ok=True)
    json_filename = results_dir / f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    images_dir = json_filename.with_suffix("")
    images_dir.mkdir(parents=True, exist_ok=True)
    st.session_state["current_run_json_file"] = str(json_filename)
    st.session_state["current_run_images_dir"] = str(images_dir)


def persist_partial_generation_state():
    """将当前已完成的候选实时写入 JSON 和图片文件。"""
    json_file = st.session_state.get("current_run_json_file", "")
    images_dir = st.session_state.get("current_run_images_dir", "")
    if not json_file or not images_dir:
        return

    partial_results = st.session_state.get("partial_results", {})
    ordered_results = [partial_results[idx] for idx in sorted(partial_results.keys())]
    json_path = Path(json_file)
    with open(json_path, "w", encoding="utf-8", errors="surrogateescape") as f:
        json_string = json.dumps(ordered_results, ensure_ascii=False, indent=4)
        json_string = json_string.encode("utf-8", "ignore").decode("utf-8")
        f.write(json_string)

    images_path = Path(images_dir)
    exp_mode = st.session_state.get("active_exp_mode", "demo_planner_critic")
    for candidate_id, result in partial_results.items():
        final_image_key = get_final_image_key(result, task_name="diagram", exp_mode=exp_mode)
        if not final_image_key:
            continue
        target_png = images_path / f"candidate_{candidate_id}.png"
        if target_png.exists():
            continue
        save_base64_image_as_png(result[final_image_key], target_png)


def launch_generation_worker(worker_params: dict):
    """在后台线程中启动生成任务。"""
    runner_registry = get_runner_registry()
    run_id = str(uuid.uuid4())
    event_queue: queue.Queue = queue.Queue()
    stop_event = threading.Event()

    def progress_callback(event):
        event_queue.put(event)

    def worker():
        try:
            event_queue.put({"type": "run_started"})
            results = asyncio.run(
                process_parallel_candidates(
                    **worker_params,
                    progress_callback=progress_callback,
                    stop_event=stop_event,
                )
            )
            event_queue.put({"type": "run_finished", "results": results})
        except Exception as e:
            event_queue.put({"type": "run_failed", "error": str(e)})
        finally:
            event_queue.put({"type": "worker_done"})

    thread = threading.Thread(target=worker, daemon=True)
    runner_registry[run_id] = {
        "thread": thread,
        "queue": event_queue,
        "stop_event": stop_event,
    }
    thread.start()
    return run_id


def consume_generation_events():
    """消费后台任务事件并更新页面状态。"""
    runner_registry = get_runner_registry()
    run_id = st.session_state.get("generation_run_id")
    if not run_id or run_id not in runner_registry:
        return

    runner = runner_registry[run_id]
    while True:
        try:
            event = runner["queue"].get_nowait()
        except queue.Empty:
            break

        event_type = event.get("type")
        if event_type == "run_started":
            st.session_state["generation_status"] = "running"
        elif event_type == "candidate_progress":
            candidate_id = event["candidate_id"]
            candidate_state = st.session_state["generation_candidates"].setdefault(candidate_id, {})
            candidate_state.update(
                {
                    "status": event.get("status", candidate_state.get("status", "queued")),
                    "stage": event.get("stage", candidate_state.get("stage", "queued")),
                    "message": event.get("message", ""),
                }
            )
        elif event_type == "candidate_prompt":
            candidate_id = event["candidate_id"]
            candidate_state = st.session_state["generation_candidates"].setdefault(candidate_id, {})
            prompts = candidate_state.setdefault("prompts", [])
            prompts.append(
                {
                    "stage": event.get("stage", ""),
                    "prompt_key": event.get("prompt_key", ""),
                    "text": event.get("text", ""),
                }
            )
        elif event_type == "candidate_result":
            candidate_id = event["candidate_id"]
            st.session_state["partial_results"][candidate_id] = event["result"]
            st.session_state["generation_completed"] = len(st.session_state["partial_results"])
            candidate_state = st.session_state["generation_candidates"].setdefault(candidate_id, {})
            candidate_state.update({"status": "completed", "stage": "completed", "message": "候选方案已完成"})
            persist_partial_generation_state()
        elif event_type == "run_finished":
            results = event["results"]
            st.session_state["results"] = results
            st.session_state["exp_mode"] = st.session_state.get("active_exp_mode", "demo_planner_critic")
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state["timestamp"] = timestamp_str
            st.session_state["generation_timestamp"] = timestamp_str
            st.session_state["generation_completed"] = len(results)
            json_file = st.session_state.get("current_run_json_file", "")
            images_dir = st.session_state.get("current_run_images_dir", "")
            if json_file and images_dir:
                st.session_state["json_file"] = json_file
                st.session_state["images_dir"] = images_dir
                st.session_state["generation_saved_images"] = len(list(Path(images_dir).glob("candidate_*.png")))
            else:
                json_filename, images_dir_path, saved_image_paths = persist_generation_results(
                    results,
                    st.session_state["exp_mode"],
                )
                st.session_state["json_file"] = str(json_filename)
                st.session_state["images_dir"] = str(images_dir_path)
                st.session_state["generation_saved_images"] = len(saved_image_paths)
            st.session_state["generation_status"] = "completed"
        elif event_type == "run_failed":
            st.session_state["generation_status"] = "failed"
            st.session_state["generation_error"] = event.get("error", "未知错误")
        elif event_type == "worker_done":
            if st.session_state.get("generation_status") == "stopping":
                st.session_state["generation_status"] = "stopped"
            runner_registry.pop(run_id, None)
            st.session_state["generation_run_id"] = None


def stop_generation_worker():
    """请求停止后台任务。"""
    runner_registry = get_runner_registry()
    run_id = st.session_state.get("generation_run_id")
    if not run_id or run_id not in runner_registry:
        return
    runner_registry[run_id]["stop_event"].set()
    st.session_state["generation_status"] = "stopping"


@st.fragment(run_every=1)
def render_generation_progress(current_mode):
    """实时渲染后台任务进度。"""
    consume_generation_events()
    status = st.session_state.get("generation_status", "idle")
    total = st.session_state.get("generation_total", 0)
    completed = st.session_state.get("generation_completed", 0)

    if status not in {"running", "stopping", "completed", "failed", "stopped"}:
        return

    st.divider()
    st.markdown("## ⏱️ 生成进度")

    progress_ratio = (completed / total) if total else 0.0
    st.progress(progress_ratio, text=f"{completed}/{total} 个候选方案已完成")

    status_text = {
        "running": "正在生成中",
        "stopping": "正在停止，当前步骤结束后会退出",
        "completed": "生成完成",
        "failed": "生成失败",
        "stopped": "已停止",
    }[status]
    st.caption(status_text)

    if status in {"running", "stopping"}:
        if st.button("⏹️ 停止生成", type="secondary", width="stretch"):
            stop_generation_worker()

    if status == "failed" and st.session_state.get("generation_error"):
        st.error(st.session_state["generation_error"])

    if status == "completed":
        saved_images = st.session_state.get("generation_saved_images", 0)
        st.success(f"已完成 {completed} 个候选方案，并自动保存 {saved_images} 张图片。")
    elif status == "stopped":
        st.warning(f"任务已停止，已完成 {completed} / {total} 个候选方案。")

    candidate_states = st.session_state.get("generation_candidates", {})
    partial_results = st.session_state.get("partial_results", {})
    if total:
        cols = st.columns(3)
        for idx in range(total):
            with cols[idx % 3]:
                state = candidate_states.get(idx, {"status": "queued", "stage": "queued", "message": "等待处理"})
                with st.expander(f"候选 {idx} | {state['status']}", expanded=state["status"] in {"running", "error"}):
                    st.caption(f"阶段：{state.get('stage', 'queued')}")
                    st.write(state.get("message", ""))
                    prompts = state.get("prompts", [])
                    if prompts:
                        for prompt_idx, prompt in enumerate(prompts):
                            with st.expander(f"提示词 {prompt_idx + 1} | {prompt.get('stage', '')}", expanded=(prompt_idx == len(prompts) - 1)):
                                st.code(clean_text(prompt.get("text", "")), language="markdown")
                    if idx in partial_results:
                        display_candidate_result(partial_results[idx], idx, current_mode)

async def refine_image_with_nanoviz(image_bytes, edit_prompt, aspect_ratio="21:9", image_size="2K", api_key="", provider="openai_compatible", base_url=""):
    """
    使用图像编辑 API 精修图像，支持 OpenAI-compatible 和 Google-compatible 两种 Provider。

    参数：
        image_bytes: 图像字节数据
        edit_prompt: 描述所需修改的文本
        aspect_ratio: 输出宽高比 (21:9, 16:9, 3:2)
        image_size: 输出分辨率 (2K 或 4K)
        api_key: API 密钥
        provider: "openai_compatible" 或 "google_compatible"

    返回：
        元组 (编辑后的图像字节数据, 成功消息)
    """
    try:
        runtime_clients = generation_utils.create_runtime_clients(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
        )

        if provider == "google_compatible":
            # ====== Google-compatible 路径：多模态 API，直接传图片字节 ======
            if runtime_clients.gemini_client is None:
                return None, "❌ Google-compatible Client 未初始化，请在侧边栏填入图像 API Key。"

            from google.genai import types

            contents = [
                types.Part.from_text(text=edit_prompt),
                types.Part.from_bytes(mime_type="image/jpeg", data=image_bytes),
            ]
            config = types.GenerateContentConfig(
                temperature=1.0,
                max_output_tokens=8192,
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(
                    aspect_ratio=aspect_ratio,
                    image_size=image_size,
                ),
            )

            image_model = st.session_state.get("tab1_image_model_name", "gemini-2.0-flash-preview-image-generation")
            response = await asyncio.to_thread(
                runtime_clients.gemini_client.models.generate_content,
                model=image_model,
                contents=contents,
                config=config,
            )

            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "inline_data") and part.inline_data:
                        edited_image_data = part.inline_data.data
                        if isinstance(edited_image_data, bytes):
                            return edited_image_data, "✅ 图像精修成功！"
                        elif isinstance(edited_image_data, str):
                            return base64.b64decode(edited_image_data), "✅ 图像精修成功！"

            return None, "❌ Google-compatible 接口未返回图像数据"

        else:
            # ====== OpenAI-compatible 路径：上传图片获取 URL → image_urls ======
            if runtime_clients.evolink_provider is None:
                return None, "❌ OpenAI-compatible Provider 未初始化，请在侧边栏填入图像 API Key。"

            image_model = st.session_state.get("tab1_image_model_name", "nano-banana-2-lite")

            # 步骤 1：上传原始图片到文件服务
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            ref_image_url = await generation_utils.upload_image_to_evolink(
                image_b64,
                media_type="image/jpeg",
                runtime_clients=runtime_clients,
            )
            print(f"[精修] 参考图已上传: {ref_image_url[:80]}...")

            # 步骤 2：图像生成 API（传入参考图 URL）
            result = await runtime_clients.evolink_provider.generate_image(
                model_name=image_model,
                prompt=edit_prompt,
                aspect_ratio=aspect_ratio,
                quality=image_size,
                image_urls=[ref_image_url],
                max_attempts=3,
                retry_delay=10,
            )

            if result and result[0] and result[0] != "Error":
                edited_image_data = base64.b64decode(result[0])
                return edited_image_data, "✅ 图像精修成功！"

            return None, "❌ 图像精修失败，未返回有效图像数据"

    except Exception as e:
        return None, f"❌ 错误：{str(e)}"
    finally:
        await generation_utils.close_runtime_clients(locals().get("runtime_clients"))


def get_evolution_stages(result, exp_mode):
    """从结果中提取所有演化阶段（图像和描述）。"""
    task_name = "diagram"
    stages = []

    # 阶段 1：规划器输出
    planner_img_key = f"target_{task_name}_desc0_base64_jpg"
    planner_desc_key = f"target_{task_name}_desc0"
    if planner_img_key in result and result[planner_img_key]:
        stages.append({
            "name": "📋 规划器",
            "image_key": planner_img_key,
            "desc_key": planner_desc_key,
            "description": "基于方法内容生成的初始图表规划"
        })

    # 阶段 2：风格化器输出（仅限 demo_full 模式）
    if exp_mode == "demo_full":
        stylist_img_key = f"target_{task_name}_stylist_desc0_base64_jpg"
        stylist_desc_key = f"target_{task_name}_stylist_desc0"
        if stylist_img_key in result and result[stylist_img_key]:
            stages.append({
                "name": "✨ 风格化器",
                "image_key": stylist_img_key,
                "desc_key": stylist_desc_key,
                "description": "经过风格优化的描述"
            })

    # 阶段 3+：评审迭代
    for round_idx in range(4):  # 检查最多 4 轮
        critic_img_key = f"target_{task_name}_critic_desc{round_idx}_base64_jpg"
        critic_desc_key = f"target_{task_name}_critic_desc{round_idx}"
        critic_sugg_key = f"target_{task_name}_critic_suggestions{round_idx}"

        if critic_img_key in result and result[critic_img_key]:
            stages.append({
                "name": f"🔍 评审第 {round_idx} 轮",
                "image_key": critic_img_key,
                "desc_key": critic_desc_key,
                "suggestions_key": critic_sugg_key,
                "description": f"根据评审反馈进行优化（第 {round_idx} 次迭代）"
            })

    return stages

def display_candidate_result(result, candidate_id, exp_mode):
    """展示单个候选方案的结果。"""
    task_name = "diagram"
    if result.get("_prompt_only"):
        st.info(f"候选方案 {candidate_id}（仅提示词）")
        prompt_keys = [
            ("规划器", f"target_{task_name}_desc0"),
            ("风格化器", f"target_{task_name}_stylist_desc0"),
        ]
        shown = False
        for label, key in prompt_keys:
            if key in result and result[key]:
                shown = True
                with st.expander(f"📝 {label}提示词", expanded=(label == "规划器")):
                    st.code(clean_text(result[key]), language="markdown")
        if not shown:
            st.info("暂无提示词内容")
        return

    # 根据 exp_mode 决定展示哪张图像
    # 对于演示模式，始终尝试查找最后一轮评审结果
    final_image_key = None
    final_desc_key = None

    # 尝试查找最后一轮评审
    for round_idx in range(3, -1, -1):  # 检查第 3、2、1、0 轮
        image_key = f"target_{task_name}_critic_desc{round_idx}_base64_jpg"
        if image_key in result and result[image_key]:
            final_image_key = image_key
            final_desc_key = f"target_{task_name}_critic_desc{round_idx}"
            break

    # 如果没有完成评审轮次则使用备选方案
    if not final_image_key:
        if exp_mode == "demo_full":
            # demo_full 在可视化之前使用风格化器
            final_image_key = f"target_{task_name}_stylist_desc0_base64_jpg"
            final_desc_key = f"target_{task_name}_stylist_desc0"
        else:
            # demo_planner_critic 使用规划器输出
            final_image_key = f"target_{task_name}_desc0_base64_jpg"
            final_desc_key = f"target_{task_name}_desc0"

    # 展示最终图像
    if final_image_key and final_image_key in result:
        img = base64_to_image(result[final_image_key])
        if img:
            st.image(img, width="stretch", caption=f"候选方案 {candidate_id}（最终版）")

            # 添加下载按钮
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            st.download_button(
                label="⬇️ 下载",
                data=buffered.getvalue(),
                file_name=f"candidate_{candidate_id}.png",
                mime="image/png",
                key=f"download_candidate_{candidate_id}",
                width="stretch"
            )
        else:
            st.error(f"候选方案 {candidate_id} 的图像解码失败")
    else:
        st.warning(f"候选方案 {candidate_id} 未生成图像")

    # 在折叠面板中展示演化时间线
    stages = get_evolution_stages(result, exp_mode)
    if len(stages) > 1:
        with st.expander(f"🔄 查看演化时间线（{len(stages)} 个阶段）", expanded=False):
            st.caption("查看图表在不同流水线阶段的演化过程")

            for idx, stage in enumerate(stages):
                st.markdown(f"### {stage['name']}")
                st.caption(stage['description'])

                # 展示该阶段的图像
                stage_img = base64_to_image(result.get(stage['image_key']))
                if stage_img:
                    st.image(stage_img, width="stretch")

                # 展示描述
                if stage['desc_key'] in result:
                    with st.expander(f"📝 描述", expanded=False):
                        cleaned_desc = clean_text(result[stage['desc_key']])
                        st.write(cleaned_desc)

                # 展示评审建议（如有）
                if 'suggestions_key' in stage and stage['suggestions_key'] in result:
                    suggestions = result[stage['suggestions_key']]
                    with st.expander(f"💡 评审建议", expanded=False):
                        cleaned_sugg = clean_text(suggestions)
                        if cleaned_sugg.strip() == "No changes needed.":
                            st.success("✅ 无需修改——迭代已停止。")
                        else:
                            st.write(cleaned_sugg)

                # 在阶段之间添加分隔线（最后一个除外）
                if idx < len(stages) - 1:
                    st.divider()
    else:
        # 如果只有一个阶段，使用更简洁的折叠面板展示描述
        with st.expander(f"📝 查看描述", expanded=False):
            if final_desc_key and final_desc_key in result:
                # 清理文本，移除无效的 UTF-8 字符
                cleaned_desc = clean_text(result[final_desc_key])
                st.write(cleaned_desc)
            else:
                st.info("暂无描述")

def main():
    init_generation_state()
    st.title("🍌 PaperVizAgent 演示")
    st.markdown("AI 驱动的科学图表生成与精修")

    # 创建选项卡
    tab1, tab2 = st.tabs(["📊 生成候选方案", "✨ 精修图像"])

    # ==================== 选项卡 1：生成候选方案 ====================
    with tab1:
        st.markdown("### 从您的方法章节和图注生成多个图表候选方案")

        # 侧边栏配置（选项卡 1）
        with st.sidebar:
            st.title("⚙️ 生成设置")

            exp_mode = st.selectbox(
                "流水线模式",
                ["demo_planner_critic", "demo_full"],
                index=0,
                key="tab1_exp_mode",
                help="选择使用哪种代理流水线"
            )

            mode_info = {
                "demo_planner_critic": "规划器 → 可视化器 → 评审器 → 可视化器",
                "demo_full": "检索器 → 规划器 → 风格化器 → 可视化器 → 评审器 → 可视化器。（风格化器能让图表更具美感，但可能过度简化。建议两种模式都尝试并选择最佳结果）"
            }
            st.info(f"**流水线：** {mode_info[exp_mode]}")

            retrieval_setting = st.selectbox(
                "检索设置",
                ["auto", "auto-full", "random", "none"],
                index=0,
                key="tab1_retrieval_setting",
                help="如何检索参考图表",
                format_func=lambda x: {
                    "auto": "auto — LLM 智能选参考，仅 caption（~3万 tokens/候选）",
                    "auto-full": "auto-full — LLM 智能选参考，含完整论文（⚠️ ~80万 tokens/候选）",
                    "random": "random — 随机选 10 个参考（免费）",
                    "none": "none — 不检索参考（免费）",
                }[x],
            )

            _retrieval_cost_info = {
                "auto": "💡 轻量 auto：仅发送图注（caption）给 LLM 做匹配，每个候选约 **3 万 tokens**，性价比最高。",
                "auto-full": "⚠️ **注意**：完整 auto 将 200 篇参考论文的全文发给 LLM，每个候选消耗约 **80 万 tokens**。仅在需要高精度检索时使用。",
                "random": "✅ 随机从 298 篇参考中选 10 个，不调用 API，零费用。",
                "none": "✅ 跳过检索，不使用参考图表，零费用。",
            }
            st.info(_retrieval_cost_info[retrieval_setting])

            num_candidates = st.number_input(
                "候选方案数量",
                min_value=1,
                max_value=20,
                value=5,
                key="tab1_num_candidates",
                help="要并行生成多少个候选方案"
            )

            aspect_ratio = st.selectbox(
                "宽高比",
                ["21:9", "16:9", "3:2"],
                key="tab1_aspect_ratio",
                help="生成图表的宽高比"
            )

            max_critic_rounds = st.number_input(
                "最大评审轮次",
                min_value=1,
                max_value=5,
                value=3,
                key="tab1_max_critic_rounds",
                help="评审优化迭代的最大轮次"
            )

            # Provider 选择
            provider_options = ["openai_compatible", "google_compatible"]
            default_text_provider = get_config_val("ui_defaults", "text_provider", "UI_DEFAULT_TEXT_PROVIDER", "openai_compatible")
            default_image_provider = get_config_val("ui_defaults", "image_provider", "UI_DEFAULT_IMAGE_PROVIDER", "openai_compatible")
            if "tab1_text_provider" not in st.session_state:
                st.session_state["tab1_text_provider"] = default_text_provider if default_text_provider in provider_options else provider_options[0]
            if "tab1_image_provider" not in st.session_state:
                st.session_state["tab1_image_provider"] = default_image_provider if default_image_provider in provider_options else provider_options[0]

            text_provider = st.selectbox(
                    "文本 Provider",
                    provider_options,
                    key="tab1_text_provider",
                    help="用于检索、规划、风格化、评审的文本模型提供商"
                )

            image_provider = st.selectbox(
                    "图像 Provider",
                    provider_options,
                    key="tab1_image_provider",
                    help="用于出图和精修的图像模型提供商"
            )

            # 文本 Provider 对应的默认配置
            _text_provider_defaults = {
                "openai_compatible": {
                    "api_key_label": "文本 API Key",
                    "api_key_help": "OpenAI 兼容接口的 API 密钥",
                    "api_key_default": get_config_val("openai_compatible", "text_api_key", "OPENAI_COMPATIBLE_TEXT_API_KEY", get_config_val("openai_compatible", "api_key", "OPENAI_COMPATIBLE_API_KEY", get_config_val("evolink", "api_key", "EVOLINK_API_KEY", ""))),
                    "base_url_default": get_config_val("openai_compatible", "text_base_url", "OPENAI_COMPATIBLE_TEXT_BASE_URL", get_config_val("openai_compatible", "base_url", "OPENAI_COMPATIBLE_BASE_URL", "https://cliproxy.bingot.codes")),
                    "model_name": get_config_val("defaults", "model_name", "MODEL_NAME", "gpt-5.4"),
                },
                "google_compatible": {
                    "api_key_label": "文本 Google-Compatible API Key",
                    "api_key_help": "Google 兼容接口的 API 密钥",
                    "api_key_default": get_config_val("google_compatible", "text_api_key", "GOOGLE_COMPATIBLE_TEXT_API_KEY", get_config_val("api_keys", "google_api_key", "GOOGLE_API_KEY", "")),
                    "base_url_default": get_config_val("google_compatible", "base_url", "GOOGLE_COMPATIBLE_BASE_URL", get_config_val("gemini", "base_url", "GEMINI_BASE_URL", "")),
                    "model_name": get_config_val("defaults", "model_name", "MODEL_NAME", "gemini-2.5-flash-preview-05-20"),
                },
            }
            _tpd = _text_provider_defaults[text_provider]

            # 图像 Provider 对应的默认配置
            _provider_defaults = {
                "openai_compatible": {
                    "api_key_label": "图像 API Key",
                    "api_key_help": "OpenAI 兼容接口的 API 密钥",
                    "api_key_default": get_config_val("openai_compatible", "image_api_key", "OPENAI_COMPATIBLE_IMAGE_API_KEY", get_config_val("openai_compatible", "api_key", "OPENAI_COMPATIBLE_API_KEY", get_config_val("evolink", "api_key", "EVOLINK_API_KEY", ""))),
                    "base_url_default": get_config_val("openai_compatible", "image_base_url", "OPENAI_COMPATIBLE_IMAGE_BASE_URL", get_config_val("openai_compatible", "base_url", "OPENAI_COMPATIBLE_BASE_URL", "http://155.94.132.145:38000")),
                    "image_model_name": get_config_val("defaults", "image_model_name", "IMAGE_MODEL_NAME", "nano-banana-2-lite"),
                },
                "google_compatible": {
                    "api_key_label": "图像 Google-Compatible API Key",
                    "api_key_help": "Google 兼容接口的 API 密钥",
                    "api_key_default": get_config_val("google_compatible", "image_api_key", "GOOGLE_COMPATIBLE_IMAGE_API_KEY", get_config_val("api_keys", "google_api_key", "GOOGLE_API_KEY", "")),
                    "base_url_default": get_config_val("google_compatible", "base_url", "GOOGLE_COMPATIBLE_BASE_URL", get_config_val("gemini", "base_url", "GEMINI_BASE_URL", "")),
                    "image_model_name": get_config_val("defaults", "image_model_name", "IMAGE_MODEL_NAME", "gemini-2.0-flash-preview-image-generation"),
                },
            }
            _ipd = _provider_defaults[image_provider]

            # 首次加载时设置默认值
            if "tab1_text_api_key" not in st.session_state:
                st.session_state["tab1_text_api_key"] = _tpd["api_key_default"]
            if "tab1_model_name" not in st.session_state:
                st.session_state["tab1_model_name"] = _tpd["model_name"]
            if "tab1_text_base_url" not in st.session_state:
                st.session_state["tab1_text_base_url"] = _tpd["base_url_default"]
            if "tab1_image_api_key" not in st.session_state:
                st.session_state["tab1_image_api_key"] = _ipd["api_key_default"]
            if "tab1_image_model_name" not in st.session_state:
                st.session_state["tab1_image_model_name"] = _ipd["image_model_name"]
            if "tab1_image_base_url" not in st.session_state:
                st.session_state["tab1_image_base_url"] = _ipd["base_url_default"]

            # 检测文本 provider 切换
            if "prev_text_provider" not in st.session_state:
                st.session_state["prev_text_provider"] = text_provider
            if st.session_state["prev_text_provider"] != text_provider:
                st.session_state["prev_text_provider"] = text_provider
                st.session_state["tab1_model_name"] = _tpd["model_name"]
                st.session_state["tab1_text_api_key"] = _tpd["api_key_default"]
                st.session_state["tab1_text_base_url"] = _tpd["base_url_default"]
                st.rerun()

            # 检测图像 provider 切换
            if "prev_image_provider" not in st.session_state:
                st.session_state["prev_image_provider"] = image_provider
            if st.session_state["prev_image_provider"] != image_provider:
                st.session_state["prev_image_provider"] = image_provider
                st.session_state["tab1_image_model_name"] = _ipd["image_model_name"]
                st.session_state["tab1_image_api_key"] = _ipd["api_key_default"]
                st.session_state["tab1_image_base_url"] = _ipd["base_url_default"]
                st.rerun()

            # 文本 API Key
            text_api_key = st.text_input(
                _tpd["api_key_label"],
                type="password",
                key="tab1_text_api_key",
                help=_tpd["api_key_help"]
            )

            text_base_url = st.text_input(
                "文本 API 代理站点",
                key="tab1_text_base_url",
                help="用于文本模型请求的 Base URL / 代理地址。",
            )

            text_models_cache_key = f"text_models::{text_provider}::{text_base_url}::{text_api_key}"
            if st.button("读取文本模型列表", width="stretch"):
                if text_provider != "openai_compatible":
                    st.info("当前仅对 openai_compatible 站点读取 /v1/models。")
                else:
                    try:
                        st.session_state[text_models_cache_key] = filter_model_ids(
                            fetch_openai_compatible_models(text_base_url, text_api_key),
                            usage="text",
                        )
                        st.success(f"已读取 {len(st.session_state[text_models_cache_key])} 个文本模型")
                    except urllib_error.HTTPError as e:
                        st.error(f"读取文本模型失败：HTTP {e.code}")
                    except Exception as e:
                        st.error(f"读取文本模型失败：{e}")

            text_model_options = st.session_state.get(text_models_cache_key, [])
            if text_model_options:
                current_text_model = st.session_state.get("tab1_model_name", _tpd["model_name"])
                if current_text_model not in text_model_options:
                    text_model_options = [current_text_model] + text_model_options
                model_name = st.selectbox(
                    "文本模型",
                    options=text_model_options,
                    index=text_model_options.index(current_text_model),
                    key="tab1_model_name_select",
                    help="用于推理/规划/评审的模型名称",
                )
                st.session_state["tab1_model_name"] = model_name
            else:
                model_name = st.text_input(
                    "文本模型",
                    key="tab1_model_name",
                    help="用于推理/规划/评审的模型名称",
                )

            st.caption("图像链路单独配置，不会复用文本 provider。")

            image_api_key = st.text_input(
                _ipd["api_key_label"],
                type="password",
                key="tab1_image_api_key",
                help=_ipd["api_key_help"]
            )

            image_base_url = st.text_input(
                "图像 API 代理站点",
                key="tab1_image_base_url",
                help="用于图像模型请求的 Base URL / 代理地址。",
            )

            image_models_cache_key = f"image_models::{image_provider}::{image_base_url}::{image_api_key}"
            if st.button("读取图像模型列表", width="stretch"):
                if image_provider != "openai_compatible":
                    st.info("当前仅对 openai_compatible 站点读取 /v1/models。")
                else:
                    try:
                        st.session_state[image_models_cache_key] = filter_model_ids(
                            fetch_openai_compatible_models(image_base_url, image_api_key),
                            usage="image",
                        )
                        st.success(f"已读取 {len(st.session_state[image_models_cache_key])} 个图像模型")
                    except urllib_error.HTTPError as e:
                        st.error(f"读取图像模型失败：HTTP {e.code}")
                    except Exception as e:
                        st.error(f"读取图像模型失败：{e}")

            image_model_options = st.session_state.get(image_models_cache_key, [])
            if image_model_options:
                current_image_model = st.session_state.get("tab1_image_model_name", _ipd["image_model_name"])
                if current_image_model not in image_model_options:
                    image_model_options = [current_image_model] + image_model_options
                image_model_name = st.selectbox(
                    "图像模型",
                    options=image_model_options,
                    index=image_model_options.index(current_image_model),
                    key="tab1_image_model_name_select",
                    help="用于图像生成的模型名称",
                )
                st.session_state["tab1_image_model_name"] = image_model_name
            else:
                image_model_name = st.text_input(
                    "图像模型",
                    key="tab1_image_model_name",
                    help="用于图像生成的模型名称",
                )

            if st.button("💾 设为默认配置", width="stretch"):
                try:
                    save_current_settings_to_config(
                        text_provider=text_provider,
                        text_api_key=text_api_key,
                        text_base_url=text_base_url,
                        model_name=model_name,
                        image_provider=image_provider,
                        image_api_key=image_api_key,
                        image_base_url=image_base_url,
                        image_model_name=image_model_name,
                    )
                    st.success("当前 Provider、API、Key 和模型已保存为默认配置。")
                except Exception as e:
                    st.error(f"保存默认配置失败：{e}")

            st.divider()
            st.markdown("### 🕘 历史记录")

            history_files = list_history_json_files()
            history_options = [""] + [str(path) for path in history_files]
            history_labels = {"": "选择历史记录..."}
            history_labels.update({str(path): format_history_label(path) for path in history_files})

            selected_history = st.selectbox(
                "历史 JSON",
                options=history_options,
                key="history_json_selector",
                format_func=lambda value: history_labels.get(value, value),
                help="从 results/demo 中选择一份历史结果",
            )

            col_history1, col_history2 = st.columns(2)
            with col_history1:
                if st.button("加载历史", width="stretch"):
                    if not selected_history:
                        st.warning("请先选择一份历史记录。")
                    else:
                        history_path = Path(selected_history)
                        try:
                            loaded_results = load_history_results(history_path)
                            st.session_state["results"] = loaded_results
                            st.session_state["json_file"] = str(history_path)
                            st.session_state["images_dir"] = str(history_path.with_suffix(""))
                            st.session_state["exp_mode"] = infer_exp_mode_from_results(loaded_results)
                            st.session_state["timestamp"] = datetime.fromtimestamp(history_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                            st.success(f"已加载历史记录：{history_path.name}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"加载历史记录失败：{e}")
            with col_history2:
                if st.button("清空结果", width="stretch"):
                    for key in ["results", "json_file", "images_dir", "exp_mode", "timestamp"]:
                        st.session_state.pop(key, None)
                    st.rerun()

        st.divider()

        # 输入区域
        st.markdown("## 📝 输入")

        # 示例内容
        example_method = r"""## Methodology: The PaperVizAgent Framework

        In this section, we present the architecture of PaperVizAgent, a reference-driven agentic framework for automated academic illustration. As illustrated in Figure \ref{fig:methodology_diagram}, PaperVizAgent orchestrates a collaborative team of five specialized agents—Retriever, Planner, Stylist, Visualizer, and Critic—to transform raw scientific content into publication-quality diagrams and plots. (See Appendix \ref{app_sec:agent_prompts} for prompts)

### Retriever Agent

Given the source context $S$ and the communicative intent $C$, the Retriever Agent identifies $N$ most relevant examples $\mathcal{E} = \{E_n\}_{n=1}^{N} \subset \mathcal{R}$ from the fixed reference set $\mathcal{R}$ to guide the downstream agents. As defined in Section \ref{sec:task_formulation}, each example $E_i \in \mathcal{R}$ is a triplet $(S_i, C_i, I_i)$.
To leverage the reasoning capabilities of VLMs, we adopt a generative retrieval approach where the VLM performs selection over candidate metadata:
$$
\mathcal{E} = \text{VLM}_{\text{Ret}} \left( S, C, \{ (S_i, C_i) \}_{E_i \in \mathcal{R}} \right)
$$
Specifically, the VLM is instructed to rank candidates by matching both research domain (e.g., Agent & Reasoning) and diagram type (e.g., pipeline, architecture), with visual structure being prioritized over topic similarity. By explicitly reasoned selection of reference illustrations $I_i$ whose corresponding contexts $(S_i, C_i)$ best match the current requirements, the Retriever provides a concrete foundation for both structural logic and visual style.

### Planner Agent

The Planner Agent serves as the cognitive core of the system. It takes the source context $S$, communicative intent $C$, and retrieved examples $\mathcal{E}$ as inputs. By performing in-context learning from the demonstrations in $\mathcal{E}$, the Planner translates the unstructured or structured data in $S$ into a comprehensive and detailed textual description $P$ of the target illustration:
$$
P = \text{VLM}_{\text{plan}}(S, C, \{ (S_i, C_i, I_i) \}_{E_i \in \mathcal{E}})
$$

### Stylist Agent

To ensure the output adheres to the aesthetic standards of modern academic manuscripts, the Stylist Agent acts as a design consultant.
A primary challenge lies in defining a comprehensive "academic style," as manual definitions are often incomplete.
To address this, the Stylist traverses the entire reference collection $\mathcal{R}$ to automatically synthesize an *Aesthetic Guideline* $\mathcal{G}$ covering key dimensions such as color palette, shapes and containers, lines and arrows, layout and composition, and typography and icons (see Appendix \ref{app_sec:auto_summarized_style_guide} for the summarized guideline and implementation details). Armed with this guideline, the Stylist refines each initial description $P$ into a stylistically optimized version $P^*$:
$$
P^* = \text{VLM}_{\text{style}}(P, \mathcal{G})
$$
This ensures that the final illustration is not only accurate but also visually professional.

### Visualizer Agent

After receiving the stylistically optimized description $P^*$, the Visualizer Agent collaborates with the Critic Agent to render academic illustrations and iteratively refine their quality. The Visualizer Agent leverages an image generation model to transform textual descriptions into visual output. In each iteration $t$, given a description $P_t$, the Visualizer generates:
$$
I_t = \text{Image-Gen}(P_t)
$$
where the initial description $P_0$ is set to $P^*$.

### Critic Agent

The Critic Agent forms a closed-loop refinement mechanism with the Visualizer by closely examining the generated image $I_t$ and providing refined description $P_{t+1}$ to the Visualizer. Upon receiving the generated image $I_t$ at iteration $t$, the Critic inspects it against the original source context $(S, C)$ to identify factual misalignments, visual glitches, or areas for improvement. It then provides targeted feedback and produces a refined description $P_{t+1}$ that addresses the identified issues:
$$
P_{t+1} = \text{VLM}_{\text{critic}}(I_t, S, C, P_t)
$$
This revised description is then fed back to the Visualizer for regeneration. The Visualizer-Critic loop iterates for $T=3$ rounds, with the final output being $I = I_T$. This iterative refinement process ensures that the final illustration meets the high standards required for academic dissemination.

### Extension to Statistical Plots

The framework extends to statistical plots by adjusting the Visualizer and Critic agents. For numerical precision, the Visualizer converts the description $P_t$ into executable Python Matplotlib code: $I_t = \text{VLM}_{\text{code}}(P_t)$. The Critic evaluates the rendered plot and generates a refined description $P_{t+1}$ addressing inaccuracies or imperfections: $P_{t+1} = \text{VLM}_{\text{critic}}(I_t, S, C, P_t)$. The same $T=3$ round iterative refinement process applies. While we prioritize this code-based approach for accuracy, we also explore direct image generation in Section \ref{sec:discussion}. See Appendix \ref{app_sec:plot_agent_prompt} for adjusted prompts."""

        example_caption = "Figure 1: Overview of our PaperVizAgent framework. Given the source context and communicative intent, we first apply a Linear Planning Phase to retrieve relevant reference examples and synthesize a stylistically optimized description. We then use an Iterative Refinement Loop (consisting of Visualizer and Critic agents) to transform the description into visual output and conduct multi-round refinements to produce the final academic illustration."

        col_input1, col_input2 = st.columns([3, 2])

        with col_input1:
            # 方法内容示例选择器
            method_example = st.selectbox(
                "加载示例（方法章节）",
                ["无", "PaperVizAgent 框架"],
                key="method_example_selector"
            )

            # 根据示例选择或会话状态设置值
            if method_example == "PaperVizAgent 框架":
                method_value = example_method
            else:
                method_value = st.session_state.get("method_content", "")

            method_content = st.text_area(
                "方法章节内容（建议使用 Markdown 格式）",
                value=method_value,
                height=250,
                placeholder="在此粘贴方法章节内容...",
                help="论文中描述方法的章节内容。建议使用 Markdown 格式。"
            )

        with col_input2:
            # 图注示例选择器
            caption_example = st.selectbox(
                "加载示例（图注）",
                ["无", "PaperVizAgent 框架"],
                key="caption_example_selector"
            )

            # 根据示例选择或会话状态设置值
            if caption_example == "PaperVizAgent 框架":
                caption_value = example_caption
            else:
                caption_value = st.session_state.get("caption", "")

            caption = st.text_area(
                "图注（建议使用 Markdown 格式）",
                value=caption_value,
                height=250,
                placeholder="输入图注...",
                help="要生成的图表的标题或描述。建议使用 Markdown 格式。"
            )

        # 处理按钮
        col_run1, col_run2 = st.columns(2)
        start_generate = col_run1.button("🚀 生成候选方案", type="primary", width="stretch")
        start_prompt_only = col_run2.button("📝 只生成提示词", width="stretch")

        if start_generate or start_prompt_only:
            if not method_content or not caption:
                st.error("请同时提供方法内容和图注！")
            elif st.session_state.get("generation_status") in {"running", "stopping"}:
                st.warning("当前已有生成任务在运行，请先等待完成或停止。")
            else:
                # 保存到会话状态
                st.session_state["method_content"] = method_content
                st.session_state["caption"] = caption
                reset_generation_state(clear_results=True)
                init_current_run_output_paths()
                st.session_state["active_exp_mode"] = exp_mode
                st.session_state["generation_total"] = num_candidates
                st.session_state["generation_candidates"] = build_candidate_state(num_candidates)
                st.session_state["generation_prompt_only"] = bool(start_prompt_only)

                input_data_list = create_sample_inputs(
                    method_content=method_content,
                    caption=caption,
                    aspect_ratio=aspect_ratio,
                    num_copies=num_candidates,
                    max_critic_rounds=max_critic_rounds
                )

                run_id = launch_generation_worker(
                    {
                        "data_list": input_data_list,
                        "exp_mode": exp_mode,
                        "retrieval_setting": retrieval_setting,
                        "model_name": model_name,
                        "image_model_name": image_model_name,
                        "text_provider": text_provider,
                        "text_api_key": text_api_key,
                        "text_base_url": text_base_url,
                        "image_provider": image_provider,
                        "image_api_key": image_api_key,
                        "image_base_url": image_base_url,
                        "prompt_only": bool(start_prompt_only),
                    }
                )
                st.session_state["generation_run_id"] = run_id
                st.session_state["generation_status"] = "running"
                st.rerun()

        render_generation_progress(exp_mode)

        # 展示结果
        if "results" in st.session_state and st.session_state["results"]:
            results = st.session_state["results"]
            current_mode = st.session_state.get("exp_mode", exp_mode)
            timestamp = st.session_state.get("timestamp", "N/A")
            prompt_only_mode = st.session_state.get("generation_prompt_only", False)

            st.divider()
            st.markdown("## 🎨 已生成的候选方案" if not prompt_only_mode else "## 📝 已生成的提示词方案")
            st.caption(f"生成时间：{timestamp} | 流水线：{mode_info.get(current_mode, current_mode)}")

            # 如果有 JSON 文件则显示下载按钮
            if "json_file" in st.session_state:
                json_file_path = Path(st.session_state["json_file"])
                if json_file_path.exists():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.info(f"📄 结果已保存至：`{json_file_path.relative_to(Path.cwd())}`")
                    with col2:
                        with open(json_file_path, "r", encoding="utf-8") as f:
                            json_data = f.read()
                        st.download_button(
                            label="⬇️ 下载 JSON",
                            data=json_data,
                            file_name=json_file_path.name,
                            mime="application/json",
                            width="stretch"
                        )

            if "images_dir" in st.session_state:
                images_dir_path = Path(st.session_state["images_dir"])
                if images_dir_path.exists():
                    st.info(f"🖼️ 图片已自动保存至：`{images_dir_path.relative_to(Path.cwd())}`")

            # 以网格形式展示结果（3 列）
            num_cols = 3
            num_results = len(results)

            for row_start in range(0, num_results, num_cols):
                cols = st.columns(num_cols)
                for col_idx in range(num_cols):
                    result_idx = row_start + col_idx
                    if result_idx < num_results:
                        with cols[col_idx]:
                            display_candidate_result(results[result_idx], result_idx, current_mode)

            if not prompt_only_mode:
                # 添加 ZIP 下载按钮
                st.divider()
                st.markdown("### 💾 批量下载")

                try:
                    import zipfile

                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                        task_name = "diagram"

                        for candidate_id, result in enumerate(results):

                            # 查找最终图像键（逻辑与展示一致）
                            final_image_key = None

                            # 尝试查找最后一轮评审
                            for round_idx in range(3, -1, -1):
                                image_key = f"target_{task_name}_critic_desc{round_idx}_base64_jpg"
                                if image_key in result and result[image_key]:
                                    final_image_key = image_key
                                    break

                            # 如果没有完成评审轮次则使用备选方案
                            if not final_image_key:
                                if current_mode == "demo_full":
                                    final_image_key = f"target_{task_name}_stylist_desc0_base64_jpg"
                                else:
                                    final_image_key = f"target_{task_name}_desc0_base64_jpg"

                            if final_image_key and final_image_key in result:
                                img = base64_to_image(result[final_image_key])
                                if img:
                                    img_buffer = BytesIO()
                                    img.save(img_buffer, format="PNG")
                                    zip_file.writestr(
                                        f"candidate_{candidate_id}.png",
                                        img_buffer.getvalue()
                                    )

                    zip_buffer.seek(0)
                    st.download_button(
                        label="⬇️ 下载 ZIP 压缩包",
                        data=zip_buffer.getvalue(),
                        file_name=f"papervizagent_candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip",
                        width="stretch"
                    )
                    st.success("ZIP 压缩包已准备好，可以下载！")
                except Exception as e:
                    st.error(f"创建 ZIP 压缩包失败：{e}")

    # ==================== 选项卡 2：精修图像 ====================
    with tab2:
        st.markdown("### 精修并放大您的图表至高分辨率（2K/4K）")
        st.caption("上传候选方案中的图像或任意图表，描述修改需求，生成高分辨率版本")

        # 精修设置侧边栏
        with st.sidebar:
            st.title("✨ 精修设置")

            refine_resolution = st.selectbox(
                "目标分辨率",
                ["2K", "4K"],
                index=0,
                key="refine_resolution",
                help="更高的分辨率需要更长时间但能产生更好的质量"
            )

            refine_aspect_ratio = st.selectbox(
                "宽高比",
                ["21:9", "16:9", "3:2"],
                index=0,
                key="refine_aspect_ratio",
                help="精修图像的宽高比"
            )

        st.divider()

        # 上传区域
        st.markdown("## 📤 上传图像")
        uploaded_file = st.file_uploader(
            "选择一个图像文件",
            type=["png", "jpg", "jpeg"],
            help="上传您想要精修的图表"
        )

        if uploaded_file is not None:
            # 展示上传的图像
            uploaded_image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 原始图像")
                st.image(uploaded_image, width="stretch")

            with col2:
                st.markdown("### 编辑指令")
                edit_prompt = st.text_area(
                    "描述您想要的修改",
                    height=200,
                    placeholder="例如：'将配色方案改为学术论文风格' 或 '将文字放大加粗' 或 '保持内容不变但输出更高分辨率'",
                    help="描述您想要的修改，或使用'保持内容不变'仅进行放大",
                    key="edit_prompt"
                )

                if st.button("✨ 精修图像", type="primary", width="stretch"):
                    if not edit_prompt:
                        st.error("请提供编辑指令！")
                    else:
                        with st.spinner(f"正在将图像精修至 {refine_resolution} 分辨率... 这可能需要一分钟。"):
                            try:
                                # 将 PIL 图像转换为字节
                                img_byte_arr = BytesIO()
                                uploaded_image.save(img_byte_arr, format='JPEG')
                                image_bytes = img_byte_arr.getvalue()

                                # 调用精修 API
                                refined_bytes, message = asyncio.run(
                                    refine_image_with_nanoviz(
                                        image_bytes=image_bytes,
                                        edit_prompt=edit_prompt,
                                        aspect_ratio=refine_aspect_ratio,
                                        image_size=refine_resolution,
                                        api_key=image_api_key,
                                        provider=image_provider,
                                        base_url=image_base_url,
                                    )
                                )

                                if refined_bytes:
                                    st.session_state["refined_image"] = refined_bytes
                                    st.session_state["refine_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    st.success(message)
                                    st.rerun()
                                else:
                                    st.error(message)
                            except Exception as e:
                                st.error(f"精修过程中出错：{e}")
                                import traceback
                                st.code(traceback.format_exc())

            # 展示精修结果（如有）
            if "refined_image" in st.session_state:
                st.divider()
                st.markdown("## 🎨 精修结果")
                st.caption(f"生成时间：{st.session_state.get('refine_timestamp', 'N/A')} | 分辨率：{refine_resolution}")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### 精修前")
                    st.image(uploaded_image, width="stretch")

                with col2:
                    st.markdown(f"### 精修后（{refine_resolution}）")
                    refined_image = Image.open(BytesIO(st.session_state["refined_image"]))
                    st.image(refined_image, width="stretch")

                    # 下载按钮
                    st.download_button(
                        label=f"⬇️ 下载 {refine_resolution} 图像",
                        data=st.session_state["refined_image"],
                        file_name=f"refined_{refine_resolution}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png",
                        width="stretch"
                    )

if __name__ == "__main__":
    main()
