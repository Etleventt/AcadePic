from __future__ import annotations

import asyncio
import base64
import json
import queue
import threading
import uuid
import shutil
from datetime import datetime
from pathlib import Path
import re
from typing import Any
import copy

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import json_repair

from agents.critic_agent import CriticAgent
from agents.planner_agent import PlannerAgent
from agents.stylist_agent import StylistAgent
from agents.visualizer_agent import VisualizerAgent, _execute_plot_code_worker
from utils import generation_utils
from utils.config import ExpConfig
from utils.image_utils import base64_to_image, save_base64_image_as_png


APP_ROOT = Path(__file__).parent
HISTORY_ROOT = APP_ROOT / "prompt_studio_history"
HISTORY_ROOT.mkdir(parents=True, exist_ok=True)
ARCHIVE_ROOT = APP_ROOT / "prompt_studio_archive"
ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
LEGACY_DEMO_ROOT = APP_ROOT / "results" / "demo"
ARCHIVE_MANIFEST_PATH = APP_ROOT / "prompt_studio_history" / ".archive_manifest.json"

TASK_META = {
    "diagram": {
        "content_label": "Methodology Section",
        "content_heading": "方法内容",
        "content_placeholder": "方法内容 / 某一章某一节",
        "intent_label": "Diagram Caption",
        "intent_heading": "图题 / 图注",
        "intent_placeholder": "图题 / 图注",
        "tree_action_label": "用作方法内容",
        "style_guide": "neurips2025_diagram_style_guide.md",
        "critic_target": "Target Diagram for Critique:",
        "critic_context_labels": ("Methodology Section", "Figure Caption"),
        "desc_output_suffix": " (do not include figure titles):",
        "generated_code_label": "生成代码",
    },
    "plot": {
        "content_label": "Plot Raw Data",
        "content_heading": "原始数据",
        "content_placeholder": "原始数据 / 表格 / JSON / CSV",
        "intent_label": "Visual Intent of the Desired Plot",
        "intent_heading": "作图意图",
        "intent_placeholder": "例如：比较不同方法的 PSNR/SSIM 柱状图",
        "tree_action_label": "用作原始数据",
        "style_guide": "neurips2025_plot_style_guide.md",
        "critic_target": "Target Plot for Critique:",
        "critic_context_labels": ("Raw Data", "Visual Intent"),
        "desc_output_suffix": ":",
        "generated_code_label": "Matplotlib 代码",
    },
}

JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()
APP_SHUTTING_DOWN = False
RECORD_LOCKS: dict[str, threading.Lock] = {}
RECORD_LOCKS_GUARD = threading.Lock()
ARCHIVE_LOCK = threading.Lock()


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_task_meta(task_type: str) -> dict[str, Any]:
    return TASK_META.get(task_type, TASK_META["diagram"])


def target_desc_key(task_type: str) -> str:
    return f"target_{task_type}_desc0"


def target_stylist_key(task_type: str) -> str:
    return f"target_{task_type}_stylist_desc0"


def target_desc_image_key(task_type: str) -> str:
    return f"{target_desc_key(task_type)}_base64_jpg"


def target_stylist_image_key(task_type: str) -> str:
    return f"{target_stylist_key(task_type)}_base64_jpg"


def critic_suggestions_key(task_type: str, round_idx: int) -> str:
    return f"target_{task_type}_critic_suggestions{round_idx}"


def critic_desc_key(task_type: str, round_idx: int) -> str:
    return f"target_{task_type}_critic_desc{round_idx}"


def critic_desc_image_key(task_type: str, round_idx: int) -> str:
    return f"{critic_desc_key(task_type, round_idx)}_base64_jpg"


def generated_code_key(task_type: str, base_key: str) -> str:
    return f"{base_key}_code" if task_type == "plot" else ""


def select_candidate_image_key(result: dict[str, Any], task_type: str) -> str | None:
    for round_idx in range(3, -1, -1):
        key = critic_desc_image_key(task_type, round_idx)
        if result.get(key):
            return key
    for key in (target_stylist_image_key(task_type), target_desc_image_key(task_type)):
        if result.get(key):
            return key
    return None


def select_candidate_code(result: dict[str, Any], task_type: str) -> str:
    if task_type != "plot":
        return ""
    for round_idx in range(3, -1, -1):
        key = generated_code_key(task_type, critic_desc_key(task_type, round_idx))
        if result.get(key):
            return result.get(key, "")
    for key in (
        generated_code_key(task_type, target_stylist_key(task_type)),
        generated_code_key(task_type, target_desc_key(task_type)),
    ):
        if result.get(key):
            return result.get(key, "")
    return ""


def infer_task_type_from_result_item(item: dict[str, Any]) -> str:
    for key in item.keys():
        if key.startswith("target_plot_"):
            return "plot"
    return "diagram"


def parse_latex_sections(tex_content: str) -> list[dict[str, Any]]:
    pattern = re.compile(r"\\(section|subsection|subsubsection)\{([^}]*)\}")
    matches = list(pattern.finditer(tex_content))
    sections: list[dict[str, Any]] = []
    for idx, match in enumerate(matches):
        level = match.group(1)
        title = match.group(2).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(tex_content)
        body = tex_content[start:end].strip()
        sections.append(
            {
                "id": f"{level}-{idx}",
                "level": level,
                "title": title,
                "local_content": body,
                "content": body,
                "subtree_content": body,
            }
        )
    return sections


def build_section_tree(flat_sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build a nested section tree from flat LaTeX section entries."""
    level_rank = {"section": 1, "subsection": 2, "subsubsection": 3}
    root: list[dict[str, Any]] = []
    stack: list[dict[str, Any]] = []
    for item in flat_sections:
        node = {**item, "children": []}
        current_rank = level_rank.get(item["level"], 99)
        while stack and level_rank.get(stack[-1]["level"], 99) >= current_rank:
            stack.pop()
        if stack:
            stack[-1]["children"].append(node)
        else:
            root.append(node)
        stack.append(node)

    def attach_subtree_content(node: dict[str, Any]) -> str:
        parts = [node.get("local_content", "").strip()]
        for child in node.get("children", []):
            child_text = attach_subtree_content(child).strip()
            if child_text:
                parts.append(child_text)
        subtree = "\n\n".join(part for part in parts if part).strip()
        node["subtree_content"] = subtree
        node["content"] = subtree
        return subtree

    for item in root:
        attach_subtree_content(item)
    return root


def record_dir(record_id: str) -> Path:
    path = HISTORY_ROOT / record_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def record_json_path(record_id: str) -> Path:
    return record_dir(record_id) / "record.json"


def get_record_lock(record_id: str) -> threading.Lock:
    with RECORD_LOCKS_GUARD:
        lock = RECORD_LOCKS.get(record_id)
        if lock is None:
            lock = threading.Lock()
            RECORD_LOCKS[record_id] = lock
        return lock


def load_archive_manifest() -> set[str]:
    if not ARCHIVE_MANIFEST_PATH.exists():
        return set()
    try:
        payload = json.loads(ARCHIVE_MANIFEST_PATH.read_text(encoding="utf-8"))
        ids = payload.get("ids", [])
        return set(ids) if isinstance(ids, list) else set()
    except Exception:
        return set()


def save_archive_manifest(ids: set[str]) -> None:
    ARCHIVE_MANIFEST_PATH.write_text(
        json.dumps({"ids": sorted(ids)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def archive_history_record(record_id: str) -> None:
    with ARCHIVE_LOCK:
        archived_ids = load_archive_manifest()
        archived_ids.add(record_id)
        save_archive_manifest(archived_ids)

        src_dir = HISTORY_ROOT / record_id
        dst_dir = ARCHIVE_ROOT / record_id
        if src_dir.exists() and not dst_dir.exists():
            shutil.move(str(src_dir), str(dst_dir))


def is_archived(record_id: str) -> bool:
    return record_id in load_archive_manifest()


def load_record(record_id: str) -> dict[str, Any]:
    path = record_json_path(record_id)
    if not path.exists():
        raise FileNotFoundError(record_id)
    lock = get_record_lock(record_id)
    with lock:
        text = path.read_text(encoding="utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        repaired = json_repair.loads(text)
        if isinstance(repaired, list):
            dict_items = [item for item in repaired if isinstance(item, dict)]
            if not dict_items:
                raise
            repaired = max(
                dict_items,
                key=lambda item: (
                    len(item.get("candidates", []) or []),
                    len(item.get("critic_runs", []) or []),
                    len(item.keys()),
                ),
            )
        if not isinstance(repaired, dict):
            raise
        save_record(repaired)
        return repaired


def save_record(record: dict[str, Any]) -> None:
    path = record_json_path(record["id"])
    lock = get_record_lock(record["id"])
    payload = json.dumps(record, ensure_ascii=False, indent=2)
    tmp_path = path.with_suffix(".json.tmp")
    with lock:
        tmp_path.write_text(payload, encoding="utf-8")
        tmp_path.replace(path)


def update_record(record_id: str, patch: dict[str, Any]) -> dict[str, Any]:
    record = load_record(record_id)
    record.update(patch)
    record["updated_at"] = now_str()
    save_record(record)
    return record


def update_record_candidate(record_id: str, candidate_id: int, patch: dict[str, Any]) -> dict[str, Any]:
    record = load_record(record_id)
    candidates = record.setdefault("candidates", [])
    for item in candidates:
        if item.get("candidate_id") == candidate_id:
            item.update(patch)
            break
    else:
        candidates.append({"candidate_id": candidate_id, **patch})
    record["updated_at"] = now_str()
    save_record(record)
    return record


def list_records() -> list[dict[str, Any]]:
    records = []
    seen_ids = set()
    archived_ids = load_archive_manifest()
    for path in sorted(HISTORY_ROOT.glob("*/record.json"), reverse=True):
        try:
            data = load_record(path.parent.name)
            record_id = data["id"]
            if record_id in archived_ids:
                continue
            seen_ids.add(record_id)
            records.append(
                {
                    "id": record_id,
                    "title": data.get("title", ""),
                    "task_type": data.get("task_type", "diagram"),
                    "created_at": data.get("created_at", ""),
                    "updated_at": data.get("updated_at", ""),
                    "candidate_count": len(data.get("candidates", [])),
                    "source": data.get("source", "prompt_studio"),
                    "legacy_name": data.get("config", {}).get("legacy_demo_json", ""),
                    "_sort_ts": path.stat().st_mtime,
                }
            )
        except Exception:
            continue
    if LEGACY_DEMO_ROOT.exists():
        for path in sorted(LEGACY_DEMO_ROOT.glob("demo_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            legacy_id = legacy_record_id_from_path(path)
            if legacy_id in archived_ids:
                continue
            if legacy_id in seen_ids:
                continue
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                first = loaded[0] if isinstance(loaded, list) and loaded else {}
                candidate_count = len(loaded) if isinstance(loaded, list) else 0
            except Exception:
                first = {}
                candidate_count = 0
            timestamp = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            records.append(
                {
                    "id": legacy_id,
                    "title": path.stem,
                    "task_type": infer_task_type_from_result_item(first) if first else "diagram",
                    "created_at": timestamp,
                    "updated_at": timestamp,
                    "candidate_count": candidate_count,
                    "source": "legacy_demo",
                    "legacy_name": path.stem,
                    "_sort_ts": path.stat().st_mtime,
                }
            )
    records.sort(key=lambda item: item.get("_sort_ts", 0), reverse=True)
    for item in records:
        item.pop("_sort_ts", None)
    return records


def normalize_record_for_client(record: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(record)
    record_id = normalized.get("id", "")
    for candidate in normalized.get("candidates", []) or []:
        if not candidate.get("image_url") and candidate.get("image_path") and record_id:
            candidate["image_url"] = f"/prompt-studio-history/{record_id}/{candidate['image_path']}"
    return normalized


def legacy_record_id_from_path(path: Path) -> str:
    return f"legacy_demo__{path.stem}"


def legacy_demo_path_from_record_id(record_id: str) -> Path | None:
    prefix = "legacy_demo__"
    if not record_id.startswith(prefix):
        return None
    stem = record_id[len(prefix):]
    path = LEGACY_DEMO_ROOT / f"{stem}.json"
    return path if path.exists() else None


def choose_legacy_image_base64(item: dict[str, Any]) -> str:
    task_type = infer_task_type_from_result_item(item)
    eval_field = item.get("eval_image_field")
    if isinstance(eval_field, str) and item.get(eval_field):
        return item.get(eval_field, "")
    keys = [
        critic_desc_image_key(task_type, round_idx)
        for round_idx in range(2, -1, -1)
    ] + [
        target_stylist_image_key(task_type),
        target_desc_image_key(task_type),
    ]
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and len(value) > 100:
            return value
    return ""


def materialize_legacy_demo_record(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, list):
        raise ValueError("legacy demo json is not a list")

    imported_id = legacy_record_id_from_path(path)
    imported_dir = record_dir(imported_id)
    candidates: list[dict[str, Any]] = []
    critic_runs: list[dict[str, Any]] = []
    first = loaded[0] if loaded else {}

    task_type = infer_task_type_from_result_item(first) if first else "diagram"
    meta = get_task_meta(task_type)

    for idx, item in enumerate(loaded):
        candidate_id = item.get("candidate_id", idx)
        candidate_image = ""
        candidate_filename = f"candidate_{candidate_id}.png"
        legacy_image_path = path.with_suffix("") / candidate_filename
        imported_image_path = imported_dir / candidate_filename
        if legacy_image_path.exists():
            if not imported_image_path.exists():
                imported_image_path.write_bytes(legacy_image_path.read_bytes())
            candidate_image = f"/prompt-studio-history/{imported_id}/{candidate_filename}"
        else:
            image_b64 = choose_legacy_image_base64(item)
            if image_b64:
                saved = save_base64_image_as_png(image_b64, imported_image_path)
                if saved:
                    candidate_image = f"/prompt-studio-history/{imported_id}/{saved.name}"

        critic_data = None
        latest_suggestion = ""
        latest_revision = ""
        for round_idx in range(3, -1, -1):
            suggestion_key = critic_suggestions_key(task_type, round_idx)
            revision_key = critic_desc_key(task_type, round_idx)
            if item.get(suggestion_key) or item.get(revision_key):
                latest_suggestion = item.get(suggestion_key, "")
                latest_revision = item.get(revision_key, "")
                break
        candidate_code = select_candidate_code(item, task_type)
        if latest_suggestion or latest_revision:
            critic_data = {
                "candidate_id": candidate_id,
                "candidate_label": f"候选 {candidate_id}",
                "raw_output": json.dumps(
                    {
                        "critic_suggestions": latest_suggestion,
                        "revised_description": latest_revision,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                "critic_suggestions": latest_suggestion,
                "revised_description": latest_revision,
            }
            critic_runs.append(critic_data)

        candidates.append(
            {
                "candidate_id": candidate_id,
                "status": "completed",
                "message": "旧 demo 历史导入",
                "image_path": imported_image_path.name if candidate_image else "",
                "image_url": candidate_image,
                "critic": critic_data,
                "generated_code": candidate_code,
            }
        )

    title = first.get("filename") or path.stem
    record = {
        "id": imported_id,
        "source": "legacy_demo",
        "task_type": task_type,
        "created_at": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "updated_at": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "title": title,
        "caption": first.get("caption", "") or first.get("visual_intent", ""),
        "method_text": first.get("content", ""),
        "planner_user_prompt": planner_user_prompt(first.get("content", ""), first.get("caption", "") or first.get("visual_intent", ""), task_type),
        "planner_output": first.get(target_desc_key(task_type), ""),
        "stylist_user_prompt": "",
        "stylist_output": first.get(target_stylist_key(task_type), "") or first.get(target_desc_key(task_type), ""),
        "candidates": candidates,
        "critic_runs": critic_runs,
        "config": {"legacy_demo_json": str(path)},
    }
    save_record(record)
    return record


def init_job(*, job_type: str, total: int = 0, record_id: str | None = None) -> str:
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {
        "id": job_id,
        "type": job_type,
        "record_id": record_id,
        "status": "running",
        "created_at": now_str(),
        "completed": 0,
        "total": total,
        "candidates": [],
        "stop_event": threading.Event(),
        "events": queue.Queue(),
    }
    return job_id


def snapshot_job(job_id: str) -> dict[str, Any]:
    job = JOBS[job_id]
    snap = {
        "id": job["id"],
        "type": job["type"],
        "record_id": job.get("record_id"),
        "status": job["status"],
        "created_at": job["created_at"],
        "completed": job.get("completed", 0),
        "total": job.get("total", 0),
        "candidates": job.get("candidates", []),
        "error": job.get("error", ""),
    }
    if "critic" in job:
        snap["critic"] = copy.deepcopy(job["critic"])
    return snap


def emit_job_event(job_id: str, event: dict[str, Any]) -> None:
    with JOBS_LOCK:
        if job_id not in JOBS:
            return
        JOBS[job_id]["events"].put(event)


def load_default_config() -> dict[str, Any]:
    import yaml

    config_path = APP_ROOT / "configs" / "model_config.yaml"
    config_data = {}
    if config_path.exists():
        config_data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    openai_cfg = config_data.get("openai_compatible", {})
    google_cfg = config_data.get("google_compatible", {})
    defaults_cfg = config_data.get("defaults", {})
    ui_cfg = config_data.get("ui_defaults", {})
    api_keys_cfg = config_data.get("api_keys", {})

    return {
        "task_type": ui_cfg.get("task_type", "diagram"),
        "text_provider": ui_cfg.get("text_provider", "openai_compatible"),
        "image_provider": ui_cfg.get("image_provider", "openai_compatible"),
        "text_api_key": openai_cfg.get("text_api_key") or openai_cfg.get("api_key") or api_keys_cfg.get("google_api_key", ""),
        "text_base_url": openai_cfg.get("text_base_url", ""),
        "text_model": defaults_cfg.get("model_name", "gpt-5.4"),
        "image_api_key": openai_cfg.get("image_api_key") or openai_cfg.get("api_key") or api_keys_cfg.get("google_api_key", ""),
        "image_base_url": openai_cfg.get("image_base_url", ""),
        "image_model": defaults_cfg.get("image_model_name", "gemini-3.0-pro-image-landscape"),
        "google_text_api_key": google_cfg.get("text_api_key") or api_keys_cfg.get("google_api_key", ""),
        "google_image_api_key": google_cfg.get("image_api_key") or api_keys_cfg.get("google_api_key", ""),
        "google_base_url": google_cfg.get("base_url", ""),
        "paper_file_path": ui_cfg.get("paper_file_path", ""),
    }


def load_paper_file(path_str: str) -> str:
    path = Path(path_str).expanduser()
    if not path.exists():
        raise FileNotFoundError(path_str)
    return path.read_text(encoding="utf-8", errors="ignore")


def build_exp_config(
    *,
    task_type: str,
    text_provider: str,
    text_api_key: str,
    text_base_url: str,
    text_model: str,
    image_provider: str,
    image_api_key: str,
    image_base_url: str,
    image_model: str,
) -> tuple[ExpConfig, Any, Any]:
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
    exp_config = ExpConfig(
        dataset_name="PromptStudio",
        task_name=task_type,
        split_name="manual",
        exp_mode="prompt_studio",
        retrieval_setting="none",
        model_name=text_model,
        image_model_name=image_model,
        provider=text_provider,
        text_provider=text_provider,
        image_provider=image_provider,
        work_dir=APP_ROOT,
        text_runtime_clients=text_runtime_clients,
        image_runtime_clients=image_runtime_clients,
    )
    return exp_config, text_runtime_clients, image_runtime_clients


def planner_user_prompt(method_text: str, caption: str, task_type: str) -> str:
    meta = get_task_meta(task_type)
    return (
        f"Now, based on the following {meta['content_label'].lower()} and {meta['intent_label'].lower()}, provide a detailed description for the "
        "figure to be generated.\n"
        f"{meta['content_label']}: {method_text}\n"
        f"{meta['intent_label']}: {caption}\n"
        f"Detailed description of the target figure to be generated{meta['desc_output_suffix']}"
    )


def stylist_user_prompt(planner_description: str, method_text: str, caption: str, task_type: str) -> str:
    meta = get_task_meta(task_type)
    style_guide = (APP_ROOT / "style_guides" / meta["style_guide"]).read_text(encoding="utf-8")
    return (
        f"Detailed Description: {planner_description}\n"
        f"Style Guidelines: {style_guide}\n"
        f"{meta['content_label']}: {method_text}\n"
        f"{meta['intent_label']}: {caption}\n"
        "Your Output:"
    )


def export_prompt_markdown(record: dict[str, Any]) -> str:
    task_type = record.get("task_type", "diagram")
    meta = get_task_meta(task_type)
    lines = [
        f"# {record.get('title') or 'Prompt Studio Export'}",
        "",
        f"- Created: {record.get('created_at', '')}",
        f"- Updated: {record.get('updated_at', '')}",
        f"- Task Type: {task_type}",
        f"- {meta['intent_label']}: {record.get('caption', '')}",
        "",
        f"## {meta['content_label']}",
        "",
        record.get("method_text", ""),
        "",
        "## Planner System Prompt",
        "",
        "```text",
        PlannerAgent(exp_config=ExpConfig(dataset_name='PromptStudio', task_name=task_type)).system_prompt,
        "```",
        "",
        "## Planner User Prompt",
        "",
        "```text",
        record.get("planner_user_prompt", ""),
        "```",
        "",
        "## Planner Output",
        "",
        "```text",
        record.get("planner_output", ""),
        "```",
        "",
        "## Stylist System Prompt",
        "",
        "```text",
        StylistAgent(exp_config=ExpConfig(dataset_name='PromptStudio', task_name=task_type)).system_prompt,
        "```",
        "",
        "## Stylist User Prompt",
        "",
        "```text",
        record.get("stylist_user_prompt", ""),
        "```",
        "",
        "## Stylist Output",
        "",
        "```text",
        record.get("stylist_output", ""),
        "```",
    ]
    if record.get("critic_runs"):
        lines.extend(["", "## Critic Runs", ""])
        for idx, run in enumerate(record["critic_runs"], start=1):
            lines.extend(
                [
                    f"### Critic Run {idx}",
                    "",
                    "```text",
                    run.get("critic_suggestions", ""),
                    "```",
                    "",
                    "```text",
                    run.get("revised_description", ""),
                    "```",
                    "",
                ]
            )
    return "\n".join(lines)


def encode_preview(path: Path) -> str:
    data = path.read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("utf-8")


async def run_prompt_pipeline(payload: "PromptRequest", progress_callback=None) -> dict[str, Any]:
    exp_config, text_runtime_clients, image_runtime_clients = build_exp_config(
        task_type=payload.task_type,
        text_provider=payload.text_provider,
        text_api_key=payload.text_api_key,
        text_base_url=payload.text_base_url,
        text_model=payload.text_model,
        image_provider=payload.image_provider,
        image_api_key=payload.image_api_key,
        image_base_url=payload.image_base_url,
        image_model=payload.image_model,
    )
    try:
        planner = PlannerAgent(exp_config=exp_config)
        stylist = StylistAgent(exp_config=exp_config)
        max_output_tokens = generation_utils.resolve_text_max_output_tokens(
            model_name=payload.text_model,
            provider=payload.text_provider,
            runtime_clients=text_runtime_clients,
            fallback=12000,
        )
        planner_prompt = planner_user_prompt(payload.method_text, payload.caption, payload.task_type)

        async def emit_partial(field: str, event: Any):
            if not progress_callback:
                return
            if isinstance(event, dict):
                current_text = event.get("text")
            else:
                current_text = str(event)
            if current_text is not None:
                progress_callback({"type": "prompt_output", "field": field, "value": current_text})

        if progress_callback:
            progress_callback({"type": "prompt_stage", "stage": "planner", "status": "running"})

        if payload.text_provider == "openai_compatible":
            response_list = await generation_utils.call_evolink_text_with_retry_async(
                model_name=payload.text_model,
                contents=[{"type": "text", "text": planner_prompt}],
                config={
                    "system_prompt": planner.system_prompt,
                    "temperature": exp_config.temperature,
                    "max_output_tokens": max_output_tokens,
                },
                max_attempts=3,
                retry_delay=3,
                runtime_clients=text_runtime_clients,
                progress_callback=lambda event: emit_partial("planner_output", event),
            )
        else:
            data = {
                "content": payload.method_text,
                "visual_intent": payload.caption,
                "top10_references": [],
                "retrieved_examples": [],
            }
            data = await planner.process(data)
            response_list = [data[target_desc_key(payload.task_type)]]

        planner_output = response_list[0].strip()
        if not planner_output or planner_output == "Error":
            raise RuntimeError("Planner 文本生成失败")
        planner_system = planner.system_prompt
        if progress_callback:
            progress_callback({"type": "prompt_output", "field": "planner_output", "value": planner_output})
            progress_callback({"type": "prompt_stage", "stage": "planner", "status": "completed"})

        stylist_prompt = stylist_user_prompt(planner_output, payload.method_text, payload.caption, payload.task_type)
        if progress_callback:
            progress_callback({"type": "prompt_stage", "stage": "stylist", "status": "running"})

        if payload.text_provider == "openai_compatible":
            response_list = await generation_utils.call_evolink_text_with_retry_async(
                model_name=payload.text_model,
                contents=[{"type": "text", "text": stylist_prompt}],
                config={
                    "system_prompt": stylist.system_prompt,
                    "temperature": exp_config.temperature,
                    "max_output_tokens": max_output_tokens,
                },
                max_attempts=3,
                retry_delay=3,
                runtime_clients=text_runtime_clients,
                progress_callback=lambda event: emit_partial("stylist_output", event),
            )
        else:
            data = {
                "content": payload.method_text,
                "visual_intent": payload.caption,
                target_desc_key(payload.task_type): planner_output,
            }
            data = await stylist.process(data)
            response_list = [data[target_stylist_key(payload.task_type)]]

        stylist_output = response_list[0].strip()
        if not stylist_output or stylist_output == "Error":
            raise RuntimeError("Stylist 文本生成失败")
        stylist_system = stylist.system_prompt
        if progress_callback:
            progress_callback({"type": "prompt_output", "field": "stylist_output", "value": stylist_output})
            progress_callback({"type": "prompt_stage", "stage": "stylist", "status": "completed"})

        return {
            "planner_system_prompt": planner_system,
            "planner_user_prompt": planner_prompt,
            "planner_output": planner_output,
            "stylist_system_prompt": stylist_system,
            "stylist_user_prompt": stylist_prompt,
            "stylist_output": stylist_output,
            "task_type": payload.task_type,
        }
    finally:
        await generation_utils.close_runtime_clients(text_runtime_clients)
        await generation_utils.close_runtime_clients(image_runtime_clients)


async def generate_single_candidate(
    *,
    task_type: str,
    final_prompt: str,
    caption: str,
    method_text: str,
    aspect_ratio: str,
    text_provider: str,
    text_api_key: str,
    text_base_url: str,
    text_model: str,
    image_provider: str,
    image_api_key: str,
    image_base_url: str,
    image_model: str,
    candidate_id: int,
    progress_callback=None,
) -> dict[str, Any]:
    exp_config, text_runtime_clients, image_runtime_clients = build_exp_config(
        task_type=task_type,
        text_provider=text_provider,
        text_api_key=text_api_key,
        text_base_url=text_base_url,
        text_model=text_model,
        image_provider=image_provider,
        image_api_key=image_api_key,
        image_base_url=image_base_url,
        image_model=image_model,
    )
    try:
        visualizer = VisualizerAgent(exp_config=exp_config)
        data = {
            "candidate_id": candidate_id,
            "content": method_text,
            "visual_intent": caption,
            "additional_info": {"rounded_ratio": aspect_ratio},
            target_desc_key(task_type): final_prompt,
        }
        data = await visualizer.process(data, progress_callback=progress_callback)
        return data
    finally:
        await generation_utils.close_runtime_clients(text_runtime_clients)
        await generation_utils.close_runtime_clients(image_runtime_clients)


async def run_manual_critic(
    *,
    task_type: str,
    image_path: Path,
    current_prompt: str,
    caption: str,
    method_text: str,
    text_provider: str,
    text_api_key: str,
    text_base_url: str,
    text_model: str,
    image_provider: str,
    image_api_key: str,
    image_base_url: str,
    image_model: str,
) -> dict[str, Any]:
    exp_config, text_runtime_clients, image_runtime_clients = build_exp_config(
        task_type=task_type,
        text_provider=text_provider,
        text_api_key=text_api_key,
        text_base_url=text_base_url,
        text_model=text_model,
        image_provider=image_provider,
        image_api_key=image_api_key,
        image_base_url=image_base_url,
        image_model=image_model,
    )
    try:
        critic = CriticAgent(exp_config=exp_config)
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        data = {
            "content": method_text,
            "visual_intent": caption,
            target_desc_key(task_type): current_prompt,
            target_desc_image_key(task_type): image_b64,
            "current_critic_round": 0,
        }
        data = await critic.process(data, source="planner")
        return {
            "critic_suggestions": data.get(critic_suggestions_key(task_type, 0), ""),
            "revised_description": data.get(critic_desc_key(task_type, 0), ""),
        }
    finally:
        await generation_utils.close_runtime_clients(text_runtime_clients)
        await generation_utils.close_runtime_clients(image_runtime_clients)


async def run_manual_critic_stream(
    *,
    task_type: str,
    image_path: Path,
    current_prompt: str,
    caption: str,
    method_text: str,
    text_provider: str,
    text_api_key: str,
    text_base_url: str,
    text_model: str,
    image_provider: str,
    image_api_key: str,
    image_base_url: str,
    image_model: str,
    progress_callback=None,
) -> dict[str, Any]:
    exp_config, text_runtime_clients, image_runtime_clients = build_exp_config(
        task_type=task_type,
        text_provider=text_provider,
        text_api_key=text_api_key,
        text_base_url=text_base_url,
        text_model=text_model,
        image_provider=image_provider,
        image_api_key=image_api_key,
        image_base_url=image_base_url,
        image_model=image_model,
    )
    try:
        critic = CriticAgent(exp_config=exp_config)
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        meta = get_task_meta(task_type)
        content_list = [
            {"type": "text", "text": meta["critic_target"]},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "data": image_b64,
                    "media_type": "image/jpeg",
                },
            },
            {
                "type": "text",
                "text": (
                    f"Detailed Description: {current_prompt}\n"
                    f"{meta['critic_context_labels'][0]}: {method_text}\n"
                    f"{meta['critic_context_labels'][1]}: {caption}\n"
                    "Your Output:"
                ),
            },
        ]

        max_output_tokens = generation_utils.resolve_text_max_output_tokens(
            model_name=text_model,
            provider=text_provider,
            runtime_clients=text_runtime_clients,
            fallback=12000,
        )

        raw_output = ""
        if progress_callback:
            progress_callback({"type": "critic_stage", "status": "running"})

        if text_provider == "openai_compatible":
            response_list = await generation_utils.call_evolink_text_with_retry_async(
                model_name=text_model,
                contents=content_list,
                config={
                    "system_prompt": critic.system_prompt,
                    "temperature": exp_config.temperature,
                    "max_output_tokens": max_output_tokens,
                },
                max_attempts=3,
                retry_delay=3,
                runtime_clients=text_runtime_clients,
                progress_callback=progress_callback,
            )
            raw_output = response_list[0]
        else:
            data = {
                "content": method_text,
                "visual_intent": caption,
                target_desc_key(task_type): current_prompt,
                target_desc_image_key(task_type): image_b64,
                "current_critic_round": 0,
            }
            data = await critic.process(data, source="planner")
            result = {
                "critic_suggestions": data.get(critic_suggestions_key(task_type, 0), ""),
                "revised_description": data.get(critic_desc_key(task_type, 0), ""),
            }
            raw_output = json.dumps(result, ensure_ascii=False, indent=2)

        cleaned_response = raw_output.replace("```json", "").replace("```", "").strip()
        try:
            eval_result = json_repair.loads(cleaned_response)
            if not isinstance(eval_result, dict):
                eval_result = {}
        except Exception:
            eval_result = {}

        result = {
            "raw_output": raw_output,
            "critic_suggestions": eval_result.get("critic_suggestions", "No changes needed."),
            "revised_description": eval_result.get("revised_description", "No changes needed."),
        }
        if progress_callback:
            progress_callback({"type": "critic_stage", "status": "completed"})
        return result
    finally:
        await generation_utils.close_runtime_clients(text_runtime_clients)
        await generation_utils.close_runtime_clients(image_runtime_clients)


class ParseRequest(BaseModel):
    paper_text: str


class PaperPathRequest(BaseModel):
    path: str


class PromptRequest(BaseModel):
    title: str = ""
    task_type: str = "diagram"
    caption: str
    method_text: str
    text_provider: str = "openai_compatible"
    text_api_key: str = ""
    text_base_url: str = ""
    text_model: str = "gpt-5.4"
    image_provider: str = "openai_compatible"
    image_api_key: str = ""
    image_base_url: str = ""
    image_model: str = "gemini-3.0-pro-image-landscape"


class BatchRequest(PromptRequest):
    final_prompt: str
    aspect_ratio: str = "16:9"
    count: int = Field(default=4, ge=1, le=20)
    concurrency: int = Field(default=2, ge=1, le=10)
    record_id: str


class CriticRequest(PromptRequest):
    record_id: str
    candidate_id: int
    current_prompt: str


class PlotCodeRequest(BaseModel):
    code: str


app = FastAPI(title="Prompt Studio")
app.mount("/prompt-studio-history", StaticFiles(directory=str(HISTORY_ROOT)), name="prompt_studio_history")


@app.get("/", include_in_schema=False)
async def root_home():
    return RedirectResponse(url="/studio", status_code=307)


@app.on_event("shutdown")
async def on_shutdown():
    global APP_SHUTTING_DOWN
    APP_SHUTTING_DOWN = True
    with JOBS_LOCK:
        for job in JOBS.values():
            job["status"] = "stopped"
            job["stop_event"].set()
            try:
                job["events"].put_nowait({"type": "done"})
            except Exception:
                pass


INDEX_HTML = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Prompt Studio</title>
  <style>
    body{font-family:ui-sans-serif,system-ui,-apple-system,sans-serif;background:#f7f6f2;color:#1b1b18;margin:0}
    .wrap{max-width:1700px;margin:0 auto;padding:24px}
    .grid{display:grid;grid-template-columns:380px minmax(0,1fr) 320px;gap:20px;align-items:start}
    .panel{background:#fff;border:1px solid #ded9cc;border-radius:18px;padding:18px;box-shadow:0 8px 28px rgba(0,0,0,.04)}
    .history-panel{position:sticky;top:24px;max-height:calc(100vh - 48px);overflow:auto}
    textarea,input,select{width:100%;box-sizing:border-box;border:1px solid #cfc8ba;border-radius:12px;padding:10px;margin-top:8px;margin-bottom:12px;font:inherit}
    textarea{min-height:120px;resize:vertical}
    button{border:0;border-radius:999px;padding:10px 16px;background:#1b1b18;color:#fff;cursor:pointer}
    button.secondary{background:#e9e3d8;color:#1b1b18}
    .row{display:flex;gap:10px;flex-wrap:wrap}
    .candidate{border:1px solid #ddd4c2;border-radius:14px;padding:12px;margin-bottom:12px}
    .muted{color:#6b665d;font-size:14px}
    .sections button{margin:4px 6px 4px 0}
    .img{width:100%;border-radius:12px;border:1px solid #ddd}
    .history-item{border-top:1px solid #eee;padding:10px 0}
    pre{white-space:pre-wrap;word-break:break-word;background:#f7f6f2;border-radius:12px;padding:12px}
    .toggle{display:flex;align-items:center;gap:8px;margin:8px 0 14px}
    .toggle input{width:auto;margin:0}
    .md-preview{display:none;border:1px solid #e4dccd;border-radius:12px;background:#faf8f3;padding:12px;margin:-4px 0 12px}
    .md-preview.active{display:block}
    .md-preview h1,.md-preview h2,.md-preview h3,.md-preview h4,.md-preview h5,.md-preview h6{margin:0 0 10px}
    .md-preview p{margin:0 0 10px;line-height:1.6}
    .md-preview ul,.md-preview ol{margin:0 0 10px 22px}
    .md-preview blockquote{margin:0 0 10px;padding-left:12px;border-left:3px solid #cfc8ba;color:#575249}
    .md-preview code{background:#efe9dd;border-radius:6px;padding:1px 5px}
    .md-preview pre{margin:0 0 10px;background:#f1ede5;border:1px solid #e0d7c6}
    .md-preview img{max-width:100%;border-radius:10px;border:1px solid #ddd}
    @media (max-width: 1200px){
      .grid{grid-template-columns:1fr}
      .history-panel{position:static;max-height:none}
    }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Prompt Studio</h1>
    <p class="muted">章节切分、提示词生成、单张试画、批量候选、手动 Critic、历史记录。</p>
    <div class="grid">
      <div class="panel">
        <h3>论文切分</h3>
        <label>默认论文文件路径</label>
        <input id="paperFilePath" placeholder="/absolute/path/to/content.tex" />
        <textarea id="paperText" placeholder="粘贴完整论文（LaTeX/TeX）"></textarea>
        <div class="row">
          <button class="secondary" onclick="loadPaperFile()">从路径读取</button>
          <button onclick="parsePaper()">解析章节</button>
        </div>
        <div id="sections" class="sections"></div>
        <hr>
        <h3>配置</h3>
        <label>文本 Provider</label>
        <select id="textProvider"><option>openai_compatible</option><option>google_compatible</option></select>
        <label>文本 API Key</label>
        <input id="textApiKey" type="password" />
        <label>文本 Base URL</label>
        <input id="textBaseUrl" />
        <label>文本模型</label>
        <input id="textModel" />
        <div class="row">
          <button class="secondary" onclick="loadTextModels()">读取文本模型</button>
        </div>
        <div id="textModels" class="muted"></div>
        <label>图像 Provider</label>
        <select id="imageProvider"><option>openai_compatible</option><option>google_compatible</option></select>
        <label>图像 API Key</label>
        <input id="imageApiKey" type="password" />
        <label>图像 Base URL</label>
        <input id="imageBaseUrl" />
        <label>图像模型</label>
        <input id="imageModel" />
        <div class="row">
          <button class="secondary" onclick="loadImageModels()">读取图像模型</button>
        </div>
        <div id="imageModels" class="muted"></div>
        <label>宽高比</label>
        <select id="aspectRatio"><option>21:9</option><option selected>16:9</option><option>3:2</option></select>
      </div>
      <div class="panel">
        <h3>当前任务</h3>
        <label>任务类型</label>
        <select id="taskType" onchange="refreshTaskMode()">
          <option value="diagram">Diagram</option>
          <option value="plot">Plot</option>
        </select>
        <input id="title" placeholder="本次任务标题（例如：方法总架构）" />
        <label id="contentLabel">方法内容</label>
        <textarea id="methodText" placeholder="方法内容 / 某一章某一节"></textarea>
        <label id="intentLabel">图题 / 图注</label>
        <textarea id="caption" placeholder="图题 / 图注"></textarea>
        <div class="row">
          <button id="generatePromptsBtn" onclick="generatePrompts()">生成 Planner + Stylist 提示词</button>
          <button class="secondary" onclick="saveDefaults()">保存为默认配置</button>
        </div>
        <hr>
        <h3>提示词工作台</h3>
        <div class="toggle">
          <input id="markdownPreviewToggle" type="checkbox" onchange="refreshMarkdownPreviews()">
          <label for="markdownPreviewToggle">Markdown 渲染预览</label>
        </div>
        <label>Planner 输出</label>
        <textarea id="plannerOutput" oninput="refreshMarkdownPreviews()"></textarea>
        <div id="plannerPreview" class="md-preview"></div>
        <label>Stylist 输出（最终绘图提示词）</label>
        <textarea id="stylistOutput" oninput="refreshMarkdownPreviews()"></textarea>
        <div id="stylistPreview" class="md-preview"></div>
        <div class="row">
          <button id="generateDraftBtn" onclick="generateDraft()">先试画一张</button>
          <div>
            <label for="batchCount">候选数量</label>
            <input id="batchCount" type="number" value="4" min="1" max="20" style="width:90px">
          </div>
          <div>
            <label for="batchConcurrency">并发数量</label>
            <input id="batchConcurrency" type="number" value="2" min="1" max="10" style="width:90px">
          </div>
          <button id="generateBatchBtn" onclick="generateBatch()">批量生成</button>
          <button id="stopJobBtn" class="secondary" onclick="stopJob()">停止任务</button>
          <button class="secondary" onclick="downloadMarkdown()">导出 Markdown Prompt Pack</button>
        </div>
        <div id="jobStatus" class="muted"></div>
        <div id="plotCodeStudio" style="display:none">
          <hr>
          <h3>Plot 代码工作台</h3>
          <div id="plotCodeStatus" class="muted">未载入代码</div>
          <textarea id="plotCodeEditor" placeholder="这里会显示生成的 Matplotlib 代码"></textarea>
          <div class="row">
            <button class="secondary" onclick="runPlotCode()">运行代码</button>
            <button class="secondary" onclick="downloadPlotCode()">下载 .py</button>
            <button class="secondary" onclick="downloadPlotImage()">下载 PNG</button>
          </div>
          <div id="plotCodePreviewWrap" style="display:none">
            <img id="plotCodePreview" class="img" src="" />
          </div>
        </div>
        <hr>
        <h3>Critic 工作台</h3>
        <div id="criticStatus" class="muted">未运行</div>
        <label>Critic 目标候选</label>
        <input id="criticTarget" readonly placeholder="尚未选择候选" />
        <label>Critic 原始输出（流式）</label>
        <textarea id="criticRawOutput" placeholder="运行 Critic 后，这里会实时显示原始 JSON 输出" oninput="refreshMarkdownPreviews()"></textarea>
        <div id="criticRawPreview" class="md-preview"></div>
        <label>Critic 批评建议</label>
        <textarea id="criticSuggestions" placeholder="Critic suggestions" oninput="refreshMarkdownPreviews()"></textarea>
        <div id="criticSuggestionsPreview" class="md-preview"></div>
        <label>Critic 修正版提示词</label>
        <textarea id="criticRevisedPrompt" placeholder="Critic revised prompt" oninput="refreshMarkdownPreviews()"></textarea>
        <div id="criticRevisedPreview" class="md-preview"></div>
        <div class="row">
          <button class="secondary" onclick="applyCriticRevision()">采用修正版提示词</button>
          <button class="secondary" onclick="generateCriticDraft()">用修正版试画一张</button>
        </div>
        <div id="candidates"></div>
      </div>
      <div class="panel history-panel">
        <h3>历史记录</h3>
        <div id="history"></div>
      </div>
    </div>
  </div>
<script>
let currentRecordId = null;
let currentJobId = null;
let sectionContentMap = {};
let currentEventSource = null;
let currentCriticCandidateId = null;
let candidateCodeMap = {};

function getTaskMeta() {
  const taskType = document.getElementById('taskType')?.value || 'diagram';
  if (taskType === 'plot') {
    return {
      taskType,
      contentLabel: '原始数据',
      contentPlaceholder: '原始数据 / 表格 / JSON / CSV',
      intentLabel: '作图意图',
      intentPlaceholder: '例如：比较不同方法的 PSNR/SSIM 柱状图',
      sectionActionLabel: '用作原始数据',
      codeLabel: 'Matplotlib 代码',
    };
  }
  return {
    taskType: 'diagram',
    contentLabel: '方法内容',
    contentPlaceholder: '方法内容 / 某一章某一节',
    intentLabel: '图题 / 图注',
    intentPlaceholder: '图题 / 图注',
    sectionActionLabel: '用作方法内容',
    codeLabel: '生成代码',
  };
}

function payloadBase() {
  return {
    title: document.getElementById('title').value,
    task_type: document.getElementById('taskType').value,
    caption: document.getElementById('caption').value,
    method_text: document.getElementById('methodText').value,
    text_provider: document.getElementById('textProvider').value,
    text_api_key: document.getElementById('textApiKey').value,
    text_base_url: document.getElementById('textBaseUrl').value,
    text_model: document.getElementById('textModel').value,
    image_provider: document.getElementById('imageProvider').value,
    image_api_key: document.getElementById('imageApiKey').value,
    image_base_url: document.getElementById('imageBaseUrl').value,
    image_model: document.getElementById('imageModel').value,
  };
}

function setBusy(buttonId, busy, busyText, idleText) {
  const btn = document.getElementById(buttonId);
  if (!btn) return;
  btn.disabled = !!busy;
  btn.textContent = busy ? busyText : idleText;
}

function refreshTaskMode() {
  const meta = getTaskMeta();
  const contentLabel = document.getElementById('contentLabel');
  const intentLabel = document.getElementById('intentLabel');
  const methodText = document.getElementById('methodText');
  const caption = document.getElementById('caption');
  if (contentLabel) contentLabel.textContent = meta.contentLabel;
  if (intentLabel) intentLabel.textContent = meta.intentLabel;
  if (methodText) methodText.placeholder = meta.contentPlaceholder;
  if (caption) caption.placeholder = meta.intentPlaceholder;
  const plotStudio = document.getElementById('plotCodeStudio');
  if (plotStudio) plotStudio.style.display = meta.taskType === 'plot' ? 'block' : 'none';
  if (Object.keys(sectionContentMap).length) {
    const box = document.getElementById('sections');
    if (box && box.dataset.treeJson) {
      renderSectionTree(JSON.parse(box.dataset.treeJson));
    }
  }
}

function loadPlotCodeFromCandidate(candidateId) {
  const code = candidateCodeMap[candidateId] || '';
  document.getElementById('plotCodeEditor').value = code;
  document.getElementById('plotCodeStatus').innerText = code ? `已载入候选 ${candidateId} 的代码` : '候选没有可用代码';
}

function escapeHtml(text) {
  return (text || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function renderInlineMarkdown(text) {
  let html = escapeHtml(text);
  html = html.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, '<img alt="$1" src="$2">');
  html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
  html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/__([^_]+)__/g, '<strong>$1</strong>');
  html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');
  html = html.replace(/_([^_]+)_/g, '<em>$1</em>');
  return html;
}

function renderMarkdownToHtml(markdown) {
  const md = (markdown || '').replace(/\\r\\n?/g, '\\n');
  if (!md.trim()) return '';
  const lines = md.split('\\n');
  const html = [];
  let paragraph = [];
  let listItems = [];
  let listTag = '';
  let inCode = false;
  let codeLines = [];

  function flushParagraph() {
    if (paragraph.length) {
      html.push(`<p>${renderInlineMarkdown(paragraph.join(' '))}</p>`);
      paragraph = [];
    }
  }

  function flushList() {
    if (listItems.length) {
      html.push(`<${listTag}>${listItems.join('')}</${listTag}>`);
      listItems = [];
      listTag = '';
    }
  }

  function flushCode() {
    if (codeLines.length) {
      html.push(`<pre><code>${escapeHtml(codeLines.join('\\n'))}</code></pre>`);
      codeLines = [];
    }
  }

  for (const line of lines) {
    const trimmed = line.trim();

    if (trimmed.startsWith('```')) {
      flushParagraph();
      flushList();
      if (inCode) {
        flushCode();
        inCode = false;
      } else {
        inCode = true;
      }
      continue;
    }

    if (inCode) {
      codeLines.push(line);
      continue;
    }

    if (!trimmed) {
      flushParagraph();
      flushList();
      continue;
    }

    const heading = trimmed.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      flushParagraph();
      flushList();
      const level = heading[1].length;
      html.push(`<h${level}>${renderInlineMarkdown(heading[2])}</h${level}>`);
      continue;
    }

    if (/^---+$/.test(trimmed) || /^\*\*\*+$/.test(trimmed)) {
      flushParagraph();
      flushList();
      html.push('<hr>');
      continue;
    }

    const blockquote = trimmed.match(/^>\s?(.*)$/);
    if (blockquote) {
      flushParagraph();
      flushList();
      html.push(`<blockquote>${renderInlineMarkdown(blockquote[1])}</blockquote>`);
      continue;
    }

    const ordered = trimmed.match(/^\d+\.\s+(.*)$/);
    if (ordered) {
      flushParagraph();
      if (listTag && listTag !== 'ol') flushList();
      listTag = 'ol';
      listItems.push(`<li>${renderInlineMarkdown(ordered[1])}</li>`);
      continue;
    }

    const unordered = trimmed.match(/^[-*+]\s+(.*)$/);
    if (unordered) {
      flushParagraph();
      if (listTag && listTag !== 'ul') flushList();
      listTag = 'ul';
      listItems.push(`<li>${renderInlineMarkdown(unordered[1])}</li>`);
      continue;
    }

    paragraph.push(trimmed);
  }

  flushParagraph();
  flushList();
  flushCode();
  return html.join('');
}

function refreshMarkdownPreviews() {
  const enabled = document.getElementById('markdownPreviewToggle').checked;
  const pairs = [
    ['plannerOutput', 'plannerPreview'],
    ['stylistOutput', 'stylistPreview'],
    ['criticRawOutput', 'criticRawPreview'],
    ['criticSuggestions', 'criticSuggestionsPreview'],
    ['criticRevisedPrompt', 'criticRevisedPreview'],
  ];
  for (const [sourceId, previewId] of pairs) {
    const source = document.getElementById(sourceId);
    const preview = document.getElementById(previewId);
    if (!source || !preview) continue;
    preview.innerHTML = enabled ? renderMarkdownToHtml(source.value) : '';
    preview.classList.toggle('active', enabled && !!source.value.trim());
  }
}

async function loadDefaults() {
  const data = await api('/studio/api/defaults');
  document.getElementById('taskType').value = data.task_type || 'diagram';
  document.getElementById('paperFilePath').value = data.paper_file_path || '';
  document.getElementById('textProvider').value = data.text_provider;
  document.getElementById('imageProvider').value = data.image_provider;
  if (data.text_provider === 'openai_compatible') {
    document.getElementById('textApiKey').value = data.text_api_key || '';
    document.getElementById('textBaseUrl').value = data.text_base_url || '';
  } else {
    document.getElementById('textApiKey').value = data.google_text_api_key || '';
    document.getElementById('textBaseUrl').value = data.google_base_url || '';
  }
  if (data.image_provider === 'openai_compatible') {
    document.getElementById('imageApiKey').value = data.image_api_key || '';
    document.getElementById('imageBaseUrl').value = data.image_base_url || '';
  } else {
    document.getElementById('imageApiKey').value = data.google_image_api_key || '';
    document.getElementById('imageBaseUrl').value = data.google_base_url || '';
  }
  document.getElementById('textModel').value = data.text_model || '';
  document.getElementById('imageModel').value = data.image_model || '';
  refreshTaskMode();
  refreshMarkdownPreviews();
  if (data.paper_file_path) {
    try { await loadPaperFile(false); } catch (e) {}
  }
}

async function api(path, options={}) {
  const res = await fetch(path, {headers:{'Content-Type':'application/json'}, ...options});
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

async function parsePaper() {
  const data = await api('/studio/api/parse', {method:'POST', body: JSON.stringify({paper_text: document.getElementById('paperText').value})});
  sectionContentMap = {};
  function collect(nodes) {
    for (const node of nodes || []) {
      sectionContentMap[node.id] = node.subtree_content || node.content || node.local_content || '';
      collect(node.children || []);
    }
  }
  collect(data.tree || []);
  renderSectionTree(data.tree || []);
}

function renderSectionTree(nodes) {
  const box = document.getElementById('sections');
  box.dataset.treeJson = JSON.stringify(nodes || []);
  const meta = getTaskMeta();
  function renderNode(node) {
    const children = (node.children || []).map(renderNode).join('');
    return `
      <details class="history-item" ${node.level === 'section' ? 'open' : ''}>
        <summary><strong>${node.level}</strong> ${node.title}</summary>
        <div class="muted">长度：${(node.content || '').length} 字</div>
        <div class="row">
          <button class="secondary" onclick="useSectionById('${node.id}')">${meta.sectionActionLabel}</button>
          <button class="secondary" onclick="copySectionById('${node.id}')">复制正文</button>
        </div>
        ${children ? `<div style="padding-left:16px">${children}</div>` : ''}
      </details>
    `;
  }
  box.innerHTML = nodes.map(renderNode).join('');
}

async function loadPaperFile(showAlert=true) {
  const path = document.getElementById('paperFilePath').value;
  if (!path) { if (showAlert) alert('请先填写论文路径'); return; }
  const data = await api('/studio/api/load-paper', {method:'POST', body: JSON.stringify({path})});
  document.getElementById('paperText').value = data.paper_text || '';
  if (showAlert) alert('论文内容已读取');
  await parsePaper();
}

function useSectionById(sectionId) {
  document.getElementById('methodText').value = sectionContentMap[sectionId] || '';
  document.getElementById('methodText').scrollIntoView({behavior:'smooth', block:'center'});
}

function copySectionById(sectionId) {
  navigator.clipboard.writeText(sectionContentMap[sectionId] || '');
}

async function generatePrompts() {
  setBusy('generatePromptsBtn', true, '正在生成提示词...', '生成 Planner + Stylist 提示词');
  try {
    const data = await api('/studio/api/prompts', {method:'POST', body: JSON.stringify(payloadBase())});
    currentJobId = data.job_id;
    currentRecordId = data.record_id;
    document.getElementById('jobStatus').innerText = 'planner running...';
    subscribeToJob(data.job_id);
  } catch (e) {
    alert(`生成提示词失败: ${e}`);
    setBusy('generatePromptsBtn', false, '', '生成 Planner + Stylist 提示词');
  }
}

async function generateDraft() {
  document.getElementById('batchCount').value = 1;
  document.getElementById('batchConcurrency').value = 1;
  await generateBatch();
}

function applyCriticRevision() {
  const revised = document.getElementById('criticRevisedPrompt').value.trim();
  if (!revised) {
    alert('还没有可用的 Critic 修正版提示词');
    return;
  }
  document.getElementById('stylistOutput').value = revised;
  refreshMarkdownPreviews();
  document.getElementById('stylistOutput').scrollIntoView({behavior:'smooth', block:'center'});
}

async function generateCriticDraft() {
  const revised = document.getElementById('criticRevisedPrompt').value.trim();
  if (!revised) {
    alert('还没有可用的 Critic 修正版提示词');
    return;
  }
  document.getElementById('batchCount').value = 1;
  document.getElementById('batchConcurrency').value = 1;
  await generateBatch(revised);
}

async function generateBatch(promptOverride = null) {
  if (!currentRecordId) { alert('请先生成提示词'); return; }
  const finalPrompt = promptOverride || document.getElementById('stylistOutput').value || document.getElementById('plannerOutput').value;
  const payload = {
    ...payloadBase(),
    record_id: currentRecordId,
    final_prompt: finalPrompt,
    aspect_ratio: document.getElementById('aspectRatio').value,
    count: Number(document.getElementById('batchCount').value),
    concurrency: Number(document.getElementById('batchConcurrency').value),
  };
  setBusy('generateBatchBtn', true, '正在启动批量任务...', '批量生成');
  setBusy('generateDraftBtn', true, '请等待...', '先试画一张');
  try {
    const data = await api('/studio/api/jobs/image-batch', {method:'POST', body: JSON.stringify(payload)});
    currentJobId = data.job_id;
    subscribeToJob(data.job_id);
  } catch (e) {
    alert(`启动批量生成失败: ${e}`);
    setBusy('generateBatchBtn', false, '', '批量生成');
    setBusy('generateDraftBtn', false, '', '先试画一张');
  }
}

async function stopJob() {
  if (!currentJobId) return;
  await api(`/studio/api/jobs/${currentJobId}/stop`, {method:'POST', body:'{}'});
  setBusy('stopJobBtn', true, '正在停止...', '停止任务');
}

function subscribeToJob(jobId) {
  if (currentEventSource) currentEventSource.close();
  currentEventSource = new EventSource(`/studio/api/jobs/${jobId}/events`);
  currentEventSource.onmessage = async (evt) => {
    const data = JSON.parse(evt.data);
    if (data.type === 'job_update') {
      const job = data.job;
      if (job.type === 'critic') {
        document.getElementById('criticStatus').innerText = `critic | ${job.status}`;
      } else {
        document.getElementById('jobStatus').innerText = `${job.status} | ${job.completed}/${job.total}`;
        renderCandidates(job.candidates || []);
      }
      if (job.record_id) currentRecordId = job.record_id;
      if (job.status === 'completed' || job.status === 'failed' || job.status === 'stopped') {
        setBusy('generateBatchBtn', false, '', '批量生成');
        setBusy('generateDraftBtn', false, '', '先试画一张');
        setBusy('generatePromptsBtn', false, '', '生成 Planner + Stylist 提示词');
        setBusy('stopJobBtn', false, '', '停止任务');
        if (currentCriticCandidateId !== null) {
          setBusy(`criticBtn-${currentCriticCandidateId}`, false, '', '运行 Critic');
        }
      } else {
        setBusy('stopJobBtn', false, '', '停止任务');
      }
    } else if (data.type === 'candidate_update') {
      const loaded = await api(`/studio/api/jobs/${jobId}`);
      if (loaded.type !== 'critic') {
        document.getElementById('jobStatus').innerText = `${loaded.status} | ${loaded.completed}/${loaded.total}`;
        renderCandidates(loaded.candidates || []);
      }
    } else if (data.type === 'prompt_stage') {
      document.getElementById('jobStatus').innerText = `${data.stage} | ${data.status}`;
    } else if (data.type === 'prompt_output') {
      if (data.field === 'planner_output') document.getElementById('plannerOutput').value = data.value || '';
      if (data.field === 'stylist_output') document.getElementById('stylistOutput').value = data.value || '';
      refreshMarkdownPreviews();
    } else if (data.type === 'critic_stage') {
      document.getElementById('criticStatus').innerText = `critic | ${data.status}`;
    } else if (data.type === 'critic_output') {
      document.getElementById('criticRawOutput').value = data.value || '';
      refreshMarkdownPreviews();
    } else if (data.type === 'critic_result') {
      document.getElementById('criticStatus').innerText = 'critic | completed';
      document.getElementById('criticRawOutput').value = data.critic.raw_output || '';
      document.getElementById('criticSuggestions').value = data.critic.critic_suggestions || '';
      document.getElementById('criticRevisedPrompt').value = data.critic.revised_description || '';
      refreshMarkdownPreviews();
      await renderHistory();
      const loaded = await api(`/studio/api/history/${currentRecordId}`);
      renderCandidates(loaded.candidates || []);
    } else if (data.type === 'prompt_result') {
      currentRecordId = data.record.id;
      document.getElementById('plannerOutput').value = data.record.planner_output || '';
      document.getElementById('stylistOutput').value = data.record.stylist_output || '';
      refreshMarkdownPreviews();
      await renderHistory();
    } else if (data.type === 'done') {
      currentEventSource.close();
      currentEventSource = null;
      currentJobId = null;
      setBusy('generateBatchBtn', false, '', '批量生成');
      setBusy('generateDraftBtn', false, '', '先试画一张');
      setBusy('generatePromptsBtn', false, '', '生成 Planner + Stylist 提示词');
      setBusy('stopJobBtn', false, '', '停止任务');
      if (currentCriticCandidateId !== null) {
        setBusy(`criticBtn-${currentCriticCandidateId}`, false, '', '运行 Critic');
      }
      await renderHistory();
    }
  };
}

window.addEventListener('beforeunload', () => {
  if (currentEventSource) {
    currentEventSource.close();
    currentEventSource = null;
  }
});

function renderCandidates(candidates) {
  const box = document.getElementById('candidates');
  candidateCodeMap = {};
  for (const c of candidates || []) {
    if (c.generated_code) candidateCodeMap[c.candidate_id] = c.generated_code;
  }
  const meta = getTaskMeta();
  box.innerHTML = candidates.map(c => `
    <div class="candidate">
      <div><strong>候选 ${c.candidate_id}</strong> | ${c.status}</div>
      <div class="muted">${c.message || ''}</div>
      ${((c.image_url || (c.image_path && currentRecordId ? `/prompt-studio-history/${currentRecordId}/${c.image_path}` : ''))) ? `<img class="img" src="${c.image_url || `/prompt-studio-history/${currentRecordId}/${c.image_path}`}">` : ''}
      ${c.generated_code ? `<details><summary>${meta.codeLabel}</summary><pre>${escapeHtml(c.generated_code)}</pre></details>` : ''}
      <div class="row">
        ${c.generated_code ? `<button class="secondary" onclick="loadPlotCodeFromCandidate(${c.candidate_id})">载入代码</button>` : ''}
        ${((c.image_url || (c.image_path && currentRecordId ? `/prompt-studio-history/${currentRecordId}/${c.image_path}` : ''))) ? `<button id="criticBtn-${c.candidate_id}" class="secondary" onclick="runCritic(${c.candidate_id})">运行 Critic</button>` : ''}
      </div>
      ${c.critic ? `<pre>${c.critic.critic_suggestions}\\n\\n---\\n\\n${c.critic.revised_description}</pre>` : ''}
    </div>
  `).join('');
}

async function runPlotCode() {
  const code = document.getElementById('plotCodeEditor').value.trim();
  if (!code) {
    alert('请先载入或填写 Matplotlib 代码');
    return;
  }
  document.getElementById('plotCodeStatus').innerText = '正在执行代码...';
  try {
    const data = await api('/studio/api/plot/execute', {method:'POST', body: JSON.stringify({code})});
    if (data.image_data_url) {
      document.getElementById('plotCodePreview').src = data.image_data_url;
      document.getElementById('plotCodePreviewWrap').style.display = 'block';
      document.getElementById('plotCodeStatus').innerText = '代码执行成功';
    } else {
      document.getElementById('plotCodeStatus').innerText = '代码执行失败';
    }
  } catch (e) {
    document.getElementById('plotCodeStatus').innerText = '代码执行失败';
    alert(`执行 Matplotlib 代码失败: ${e}`);
  }
}

function downloadPlotCode() {
  const code = document.getElementById('plotCodeEditor').value.trim();
  if (!code) {
    alert('没有可下载的代码');
    return;
  }
  const blob = new Blob([code], {type: 'text/x-python'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'plot_candidate.py';
  a.click();
  URL.revokeObjectURL(url);
}

function downloadPlotImage() {
  const src = document.getElementById('plotCodePreview').src;
  if (!src) {
    alert('请先运行代码并生成预览图');
    return;
  }
  const a = document.createElement('a');
  a.href = src;
  a.download = 'plot_candidate.png';
  a.click();
}

async function runCritic(candidateId) {
  if (!currentRecordId) return;
  currentCriticCandidateId = candidateId;
  document.getElementById('criticTarget').value = `候选 ${candidateId}`;
  document.getElementById('criticStatus').innerText = 'critic | running';
  document.getElementById('criticRawOutput').value = '';
  document.getElementById('criticSuggestions').value = '';
  document.getElementById('criticRevisedPrompt').value = '';
  setBusy(`criticBtn-${candidateId}`, true, 'Critic 运行中...', '运行 Critic');
  const payload = {
    ...payloadBase(),
    record_id: currentRecordId,
    candidate_id: candidateId,
    current_prompt: document.getElementById('stylistOutput').value || document.getElementById('plannerOutput').value,
  };
  try {
    const data = await api('/studio/api/critic', {method:'POST', body: JSON.stringify(payload)});
    currentJobId = data.job_id;
    subscribeToJob(data.job_id);
  } catch (e) {
    document.getElementById('criticStatus').innerText = 'critic | failed';
    setBusy(`criticBtn-${candidateId}`, false, '', '运行 Critic');
    alert(`运行 Critic 失败: ${e}`);
  }
}

async function renderHistory() {
  const data = await api('/studio/api/history');
  const box = document.getElementById('history');
  box.innerHTML = data.records.map(r => `
    <div class="history-item">
      <div><strong>${r.title || r.id}</strong></div>
      ${r.source === 'legacy_demo' ? `<div class="muted">旧文件：${r.legacy_name || r.id}</div>` : ''}
      <div class="muted">${r.updated_at}</div>
      <div class="muted">任务：${r.task_type || 'diagram'}</div>
      <div class="muted">来源：${r.source === 'legacy_demo' ? '旧 demo' : 'Prompt Studio'}</div>
      <div class="muted">候选图：${r.candidate_count || 0}</div>
      <div class="row">
        <button class="secondary" onclick="loadHistory('${r.id}')">加载</button>
        <button class="secondary" onclick="archiveHistory('${r.id}')">归档</button>
      </div>
    </div>
  `).join('');
}

async function archiveHistory(recordId) {
  const ok = window.confirm('确认归档这条历史记录？归档后默认列表将不再显示。');
  if (!ok) return;
  await api(`/studio/api/history/${recordId}/archive`, {method:'POST', body:'{}'});
  if (currentRecordId === recordId) {
    currentRecordId = null;
  }
  await renderHistory();
}

async function loadHistory(recordId) {
  const data = await api(`/studio/api/history/${recordId}`);
  currentRecordId = recordId;
  document.getElementById('taskType').value = data.task_type || 'diagram';
  refreshTaskMode();
  document.getElementById('title').value = data.title || '';
  document.getElementById('caption').value = data.caption || '';
  document.getElementById('methodText').value = data.method_text || '';
  document.getElementById('plannerOutput').value = data.planner_output || '';
  document.getElementById('stylistOutput').value = data.stylist_output || '';
  const latestCritic = (data.critic_runs || []).at(-1);
  document.getElementById('criticTarget').value = latestCritic?.candidate_label || '';
  document.getElementById('criticRawOutput').value = latestCritic?.raw_output || '';
  document.getElementById('criticSuggestions').value = latestCritic?.critic_suggestions || '';
  document.getElementById('criticRevisedPrompt').value = latestCritic?.revised_description || '';
  document.getElementById('criticStatus').innerText = latestCritic ? 'critic | completed' : '未运行';
  refreshMarkdownPreviews();
  renderCandidates(data.candidates || []);
}

async function saveDefaults() {
  await api('/studio/api/defaults', {method:'POST', body: JSON.stringify({...payloadBase(), paper_file_path: document.getElementById('paperFilePath').value})});
  alert('已保存为默认配置');
}

async function loadModels(kind) {
  const isText = kind === 'text';
  const provider = document.getElementById(isText ? 'textProvider' : 'imageProvider').value;
  const apiKey = document.getElementById(isText ? 'textApiKey' : 'imageApiKey').value;
  const baseUrl = document.getElementById(isText ? 'textBaseUrl' : 'imageBaseUrl').value;
  const data = await api('/studio/api/models', {method:'POST', body: JSON.stringify({provider, api_key: apiKey, base_url: baseUrl, usage: kind})});
  const box = document.getElementById(isText ? 'textModels' : 'imageModels');
  const target = document.getElementById(isText ? 'textModel' : 'imageModel');
  box.innerHTML = data.models.map(m => `<button class="secondary" onclick="document.getElementById('${isText ? 'textModel' : 'imageModel'}').value='${m}'">${m}</button>`).join('');
  if (data.models.length && !target.value) target.value = data.models[0];
}

async function loadTextModels() { await loadModels('text'); }
async function loadImageModels() { await loadModels('image'); }

function downloadMarkdown() {
  if (!currentRecordId) { alert('请先生成提示词'); return; }
  window.open(`/studio/api/history/${currentRecordId}/markdown`, '_blank');
}

window.onload = async () => {
  await loadDefaults();
  await renderHistory();
};
</script>
</body>
</html>
"""


class ModelListRequest(BaseModel):
    provider: str
    api_key: str = ""
    base_url: str = ""
    usage: str = "text"


@app.get("/studio", response_class=HTMLResponse)
async def studio_home():
    return HTMLResponse(INDEX_HTML)


@app.get("/studio/api/defaults")
async def get_defaults():
    return load_default_config()


@app.post("/studio/api/load-paper")
async def load_paper(req: PaperPathRequest):
    try:
        return {"paper_text": load_paper_file(req.path)}
    except FileNotFoundError:
        raise HTTPException(404, "paper file not found")


@app.post("/studio/api/parse")
async def parse_paper(req: ParseRequest):
    sections = parse_latex_sections(req.paper_text)
    return {"sections": sections, "tree": build_section_tree(sections)}


@app.post("/studio/api/plot/execute")
async def execute_plot_code(req: PlotCodeRequest):
    image_b64 = await asyncio.to_thread(_execute_plot_code_worker, req.code)
    if not image_b64:
        raise HTTPException(400, "plot code execution failed")
    return {
        "image_base64": image_b64,
        "image_data_url": f"data:image/jpeg;base64,{image_b64}",
    }


@app.post("/studio/api/prompts")
async def generate_prompts(req: PromptRequest):
    record_id = uuid.uuid4().hex
    job_id = init_job(job_type="prompt_generation", total=2, record_id=record_id)

    def worker():
        async def runner():
            record = {
                "id": record_id,
                "created_at": now_str(),
                "updated_at": now_str(),
                "title": req.title,
                "task_type": req.task_type,
                "caption": req.caption,
                "method_text": req.method_text,
                "planner_user_prompt": planner_user_prompt(req.method_text, req.caption, req.task_type),
                "planner_output": "",
                "stylist_user_prompt": "",
                "stylist_output": "",
                "candidates": [],
                "critic_runs": [],
                "config": req.model_dump(),
            }
            save_record(record)

            def progress_callback(event):
                emit_job_event(job_id, event)
                if event.get("type") == "prompt_stage" and event.get("status") == "completed":
                    with JOBS_LOCK:
                        JOBS[job_id]["completed"] += 1
                    emit_job_event(job_id, {"type": "job_update", "job": snapshot_job(job_id)})
                if event.get("type") == "prompt_output":
                    if event.get("field") == "planner_output":
                        update_record(record_id, {"planner_output": event.get("value", "")})
                    elif event.get("field") == "stylist_output":
                        current = load_record(record_id)
                        update_record(
                            record_id,
                            {
                                "stylist_output": event.get("value", ""),
                                "stylist_user_prompt": stylist_user_prompt(
                                    current.get("planner_output", ""),
                                    req.method_text,
                                    req.caption,
                                    req.task_type,
                                ),
                            },
                        )

            prompt_data = await run_prompt_pipeline(req, progress_callback=progress_callback)
            record = update_record(
                record_id,
                {
                    "planner_user_prompt": prompt_data["planner_user_prompt"],
                    "planner_output": prompt_data["planner_output"],
                    "stylist_user_prompt": prompt_data["stylist_user_prompt"],
                    "stylist_output": prompt_data["stylist_output"],
                },
            )
            update_job(job_id, {"status": "completed", "record_id": record_id, "completed": 2})
            emit_job_event(job_id, {"type": "prompt_result", "record": record})

        try:
            asyncio.run(runner())
        except Exception as e:
            update_job(job_id, {"status": "failed", "error": str(e)})

    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": job_id, "record_id": record_id}


def update_job(job_id: str, patch: dict[str, Any]):
    with JOBS_LOCK:
        JOBS[job_id].update(patch)
    emit_job_event(job_id, {"type": "job_update", "job": snapshot_job(job_id)})


def patch_candidate(job_id: str, candidate_id: int, patch: dict[str, Any]):
    with JOBS_LOCK:
        for item in JOBS[job_id]["candidates"]:
            if item["candidate_id"] == candidate_id:
                item.update(patch)
                updated = dict(item)
                break
        else:
            return
    emit_job_event(job_id, {"type": "candidate_update", "candidate": updated, "job_id": job_id})


def persist_job_candidates(record_id: str, candidates: list[dict[str, Any]]):
    record = load_record(record_id)
    record["candidates"] = candidates
    record["updated_at"] = now_str()
    save_record(record)


@app.post("/studio/api/jobs/image-batch")
async def start_image_batch(req: BatchRequest):
    try:
        record = load_record(req.record_id)
    except FileNotFoundError:
        raise HTTPException(404, "record not found")

    job_id = init_job(job_type="image_batch", total=req.count, record_id=req.record_id)
    stop_event = JOBS[job_id]["stop_event"]
    existing_candidates = copy.deepcopy(record.get("candidates", []))
    existing_ids = [c.get("candidate_id", -1) for c in existing_candidates if isinstance(c.get("candidate_id"), int)]
    start_candidate_id = (max(existing_ids) + 1) if existing_ids else 0
    batch_candidate_ids = list(range(start_candidate_id, start_candidate_id + req.count))
    new_candidates = [
        {"candidate_id": candidate_id, "status": "queued", "message": "等待生成"}
        for candidate_id in batch_candidate_ids
    ]
    with JOBS_LOCK:
        JOBS[job_id]["candidates"] = existing_candidates + new_candidates
        JOBS[job_id]["batch_candidate_ids"] = batch_candidate_ids
    emit_job_event(job_id, {"type": "job_update", "job": snapshot_job(job_id)})
    persist_job_candidates(req.record_id, JOBS[job_id]["candidates"])

    def worker():
        async def runner():
            sem = asyncio.Semaphore(req.concurrency)
            async def one(candidate_id: int):
                if stop_event.is_set():
                    patch_candidate(job_id, candidate_id, {"status": "cancelled", "message": "已取消"})
                    persist_job_candidates(req.record_id, JOBS[job_id]["candidates"])
                    return
                patch_candidate(job_id, candidate_id, {"status": "running", "message": "正在生成"})
                persist_job_candidates(req.record_id, JOBS[job_id]["candidates"])

                async def progress_callback(message: str):
                    patch_candidate(job_id, candidate_id, {"status": "running", "message": message})
                    persist_job_candidates(req.record_id, JOBS[job_id]["candidates"])

                result = await generate_single_candidate(
                    task_type=req.task_type,
                    final_prompt=req.final_prompt,
                    caption=req.caption,
                    method_text=req.method_text,
                    aspect_ratio=req.aspect_ratio,
                    text_provider=req.text_provider,
                    text_api_key=req.text_api_key,
                    text_base_url=req.text_base_url,
                    text_model=req.text_model,
                    image_provider=req.image_provider,
                    image_api_key=req.image_api_key,
                    image_base_url=req.image_base_url,
                    image_model=req.image_model,
                    candidate_id=candidate_id,
                    progress_callback=progress_callback,
                )
                final_key = select_candidate_image_key(result, req.task_type)
                generated_code = select_candidate_code(result, req.task_type)
                image_url = ""
                image_path = ""
                if final_key:
                    image_path_obj = record_dir(req.record_id) / f"{req.task_type}_candidate_{candidate_id}.png"
                    saved = save_base64_image_as_png(result[final_key], image_path_obj)
                    if saved:
                        image_path = saved.name
                        image_url = f"/prompt-studio-history/{req.record_id}/{saved.name}"
                patch_candidate(
                    job_id,
                    candidate_id,
                    {
                        "status": "completed",
                        "message": "已完成",
                        "image_url": image_url,
                        "image_path": image_path,
                        "generated_code": generated_code,
                    },
                )
                persist_job_candidates(req.record_id, JOBS[job_id]["candidates"])
                update_job(
                    job_id,
                    {
                        "completed": sum(
                            1
                            for c in JOBS[job_id]["candidates"]
                            if c.get("candidate_id") in batch_candidate_ids and c.get("status") == "completed"
                        )
                    },
                )

            async def wrapped(candidate_id: int):
                async with sem:
                    await one(candidate_id)

            await asyncio.gather(*(wrapped(candidate_id) for candidate_id in batch_candidate_ids))

        try:
            asyncio.run(runner())
            update_job(job_id, {"status": "stopped" if stop_event.is_set() else "completed"})
        except Exception as e:
            update_job(job_id, {"status": "failed", "error": str(e)})

    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": job_id}


@app.get("/studio/api/jobs/{job_id}")
async def get_job(job_id: str):
    with JOBS_LOCK:
        if job_id not in JOBS:
            raise HTTPException(404, "job not found")
        job = JOBS[job_id].copy()
    job.pop("stop_event", None)
    job.pop("events", None)
    return job


@app.get("/studio/api/jobs/{job_id}/events")
async def job_events(job_id: str):
    with JOBS_LOCK:
        if job_id not in JOBS:
            raise HTTPException(404, "job not found")
        event_queue = JOBS[job_id]["events"]

    async def event_stream():
        yield f"data: {json.dumps({'type': 'job_update', 'job': snapshot_job(job_id)}, ensure_ascii=False)}\n\n"
        while True:
            if APP_SHUTTING_DOWN:
                yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
                break
            try:
                event = event_queue.get_nowait()
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except queue.Empty:
                with JOBS_LOCK:
                    if job_id not in JOBS:
                        break
                    status = JOBS[job_id]["status"]
                if status in {"completed", "failed", "stopped"}:
                    yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
                    break
                await asyncio.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/studio/api/jobs/{job_id}/stop")
async def stop_job(job_id: str):
    with JOBS_LOCK:
        if job_id not in JOBS:
            raise HTTPException(404, "job not found")
        JOBS[job_id]["stop_event"].set()
        JOBS[job_id]["status"] = "stopping"
    return {"ok": True}


@app.post("/studio/api/critic")
async def run_critic(req: CriticRequest):
    record = load_record(req.record_id)
    candidate = next((c for c in record.get("candidates", []) if c.get("candidate_id") == req.candidate_id), None)
    if not candidate or not candidate.get("image_path"):
        raise HTTPException(400, "candidate image not found")
    image_path = record_dir(req.record_id) / candidate["image_path"]

    job_id = init_job(job_type="critic", total=1, record_id=req.record_id)
    with JOBS_LOCK:
        JOBS[job_id]["critic"] = {
            "candidate_id": req.candidate_id,
            "status": "running",
            "raw_output": "",
            "critic_suggestions": "",
            "revised_description": "",
        }
    emit_job_event(job_id, {"type": "job_update", "job": snapshot_job(job_id)})

    def worker():
        async def runner():
            def progress_callback(event):
                if isinstance(event, dict):
                    current_text = event.get("text", "")
                else:
                    current_text = str(event)
                with JOBS_LOCK:
                    if job_id in JOBS:
                        JOBS[job_id]["critic"]["raw_output"] = current_text
                emit_job_event(job_id, {"type": "critic_output", "value": current_text})
                update_record_candidate(req.record_id, req.candidate_id, {"critic": {"raw_output": current_text}})

            emit_job_event(job_id, {"type": "critic_stage", "status": "running"})
            critic = await run_manual_critic_stream(
                task_type=req.task_type,
                image_path=image_path,
                current_prompt=req.current_prompt,
                caption=req.caption,
                method_text=req.method_text,
                text_provider=req.text_provider,
                text_api_key=req.text_api_key,
                text_base_url=req.text_base_url,
                text_model=req.text_model,
                image_provider=req.image_provider,
                image_api_key=req.image_api_key,
                image_base_url=req.image_base_url,
                image_model=req.image_model,
                progress_callback=progress_callback,
            )
            critic["candidate_id"] = req.candidate_id
            critic["candidate_label"] = f"候选 {req.candidate_id}"
            update_record_candidate(req.record_id, req.candidate_id, {"critic": critic})
            record_now = load_record(req.record_id)
            critic_runs = record_now.setdefault("critic_runs", [])
            critic_runs.append(critic)
            record_now["updated_at"] = now_str()
            save_record(record_now)
            with JOBS_LOCK:
                if job_id in JOBS:
                    JOBS[job_id]["critic"] = {
                        **critic,
                        "status": "completed",
                    }
            update_job(job_id, {"status": "completed", "completed": 1})
            emit_job_event(job_id, {"type": "critic_result", "critic": critic})

        try:
            asyncio.run(runner())
        except Exception as e:
            with JOBS_LOCK:
                if job_id in JOBS:
                    JOBS[job_id]["critic"]["status"] = "failed"
            update_job(job_id, {"status": "failed", "error": str(e)})

    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": job_id, "record_id": req.record_id}


@app.get("/studio/api/history")
async def history_index():
    return {"records": list_records()}


@app.get("/studio/api/history/{record_id}")
async def history_detail(record_id: str):
    if is_archived(record_id):
        raise HTTPException(404, "record archived")
    legacy_path = legacy_demo_path_from_record_id(record_id)
    if legacy_path is not None:
        try:
            return normalize_record_for_client(materialize_legacy_demo_record(legacy_path))
        except Exception as e:
            raise HTTPException(400, f"legacy record load failed: {e}")
    try:
        return normalize_record_for_client(load_record(record_id))
    except FileNotFoundError:
        raise HTTPException(404, "record not found")


@app.post("/studio/api/history/{record_id}/archive")
async def archive_history(record_id: str):
    archive_history_record(record_id)
    return {"ok": True}


@app.get("/studio/api/history/{record_id}/markdown", response_class=HTMLResponse)
async def history_markdown(record_id: str):
    try:
        record = load_record(record_id)
    except FileNotFoundError:
        raise HTTPException(404, "record not found")
    return HTMLResponse(f"<pre>{export_prompt_markdown(record)}</pre>")


@app.post("/studio/api/defaults")
async def save_defaults(req: dict[str, Any]):
    import yaml
    config_path = APP_ROOT / "configs" / "model_config.yaml"
    config_data = {}
    if config_path.exists():
        config_data = yaml.safe_load(config_path.read_text(encoding='utf-8')) or {}
    config_data.setdefault("defaults", {})["model_name"] = req.get("text_model", "")
    config_data.setdefault("defaults", {})["image_model_name"] = req.get("image_model", "")
    ui_defaults = config_data.setdefault("ui_defaults", {})
    ui_defaults["task_type"] = req.get("task_type", "diagram")
    ui_defaults["text_provider"] = req.get("text_provider", "openai_compatible")
    ui_defaults["image_provider"] = req.get("image_provider", "openai_compatible")
    ui_defaults["paper_file_path"] = req.get("paper_file_path", "")
    openai_cfg = config_data.setdefault("openai_compatible", {})
    google_cfg = config_data.setdefault("google_compatible", {})
    api_keys_cfg = config_data.setdefault("api_keys", {})

    if req.get("text_provider") == "openai_compatible":
        openai_cfg["text_api_key"] = req.get("text_api_key", "")
        openai_cfg["text_base_url"] = req.get("text_base_url", "")
    else:
        google_cfg["text_api_key"] = req.get("text_api_key", "")
        google_cfg["base_url"] = req.get("text_base_url", "")
        api_keys_cfg["google_api_key"] = req.get("text_api_key", "")

    if req.get("image_provider") == "openai_compatible":
        openai_cfg["image_api_key"] = req.get("image_api_key", "")
        openai_cfg["image_base_url"] = req.get("image_base_url", "")
    else:
        google_cfg["image_api_key"] = req.get("image_api_key", "")
        google_cfg["base_url"] = req.get("image_base_url", "")
        api_keys_cfg["google_api_key"] = req.get("image_api_key", "")

    config_path.write_text(yaml.safe_dump(config_data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return {"ok": True}


@app.post("/studio/api/models")
async def list_models(req: ModelListRequest):
    models: list[str] = []
    if req.provider == "openai_compatible":
        try:
            raw = generation_utils.fetch_openai_compatible_models(req.base_url, req.api_key)
        except Exception as e:
            raise HTTPException(400, f"load openai-compatible models failed: {e}")
        for item in raw:
            model_id = item.get("id", "")
            description = str(item.get("description", "")).lower()
            lowered = model_id.lower()
            is_image = (
                "image generation" in description
                or "image" in lowered
                or "imagen" in lowered
            ) and "video" not in description and not lowered.startswith("veo")
            if req.usage == "image" and is_image:
                models.append(model_id)
            elif req.usage == "text" and not is_image:
                models.append(model_id)
    elif req.provider == "google_compatible":
        try:
            from google import genai
            client = genai.Client(
                api_key=req.api_key,
                http_options={"base_url": req.base_url} if req.base_url else None,
            )
            if hasattr(client.models, "list"):
                for item in client.models.list():
                    model_id = getattr(item, "name", "") or getattr(item, "id", "")
                    if model_id:
                        models.append(model_id)
        except Exception as e:
            raise HTTPException(400, f"load google-compatible models failed: {e}")
    return {"models": models}
