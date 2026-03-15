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
API 调用工具函数，支持 Evolink、Gemini、Claude、OpenAI 等多种 Provider。
"""

import json
import asyncio
import base64
from io import BytesIO
from functools import partial
from ast import literal_eval
from typing import List, Dict, Any

from PIL import Image

import os
import yaml
from pathlib import Path
from dataclasses import dataclass
from urllib import error as urllib_error
from urllib import request as urllib_request

# ==================== 配置加载 ====================

config_path = Path(__file__).parent.parent / "configs" / "model_config.yaml"
model_config = {}
if config_path.exists():
    with open(config_path, "r") as f:
        model_config = yaml.safe_load(f) or {}

def get_config_val(section, key, env_var, default=""):
    val = os.getenv(env_var)
    if not val and section in model_config:
        val = model_config[section].get(key)
    return val or default

# ==================== Evolink Provider 初始化 ====================


@dataclass
class RuntimeClients:
    """Per-run API clients to avoid cross-session interference."""

    evolink_provider: Any = None
    gemini_client: Any = None
    anthropic_client: Any = None
    openai_client: Any = None
    model_catalog: list[dict[str, Any]] | None = None

evolink_provider = None

openai_compatible_api_key = get_config_val(
    "openai_compatible", "api_key", "OPENAI_COMPATIBLE_API_KEY",
    get_config_val("evolink", "api_key", "EVOLINK_API_KEY", ""),
)
openai_compatible_text_base_url = get_config_val(
    "openai_compatible", "text_base_url", "OPENAI_COMPATIBLE_TEXT_BASE_URL",
    get_config_val("openai_compatible", "base_url", "OPENAI_COMPATIBLE_BASE_URL", "https://cliproxy.bingot.codes"),
)
openai_compatible_image_base_url = get_config_val(
    "openai_compatible", "image_base_url", "OPENAI_COMPATIBLE_IMAGE_BASE_URL",
    get_config_val("openai_compatible", "base_url", "OPENAI_COMPATIBLE_BASE_URL", "http://155.94.132.145:38000"),
)
openai_compatible_base_url = openai_compatible_image_base_url
openai_compatible_file_base_url = get_config_val(
    "openai_compatible", "file_base_url", "OPENAI_COMPATIBLE_FILE_BASE_URL",
    "",
)
google_compatible_base_url = get_config_val(
    "google_compatible", "base_url", "GOOGLE_COMPATIBLE_BASE_URL",
    get_config_val("gemini", "base_url", "GEMINI_BASE_URL", ""),
)

if openai_compatible_api_key:
    from providers.openai_compatible import OpenAICompatibleProvider
    evolink_provider = OpenAICompatibleProvider(
        api_key=openai_compatible_api_key,
        base_url=openai_compatible_image_base_url,
        file_base_url=openai_compatible_file_base_url,
    )
    print(f"已初始化 OpenAI-compatible Provider (base_url={openai_compatible_image_base_url})")
else:
    print("警告：未配置 OpenAI-compatible API Key，OpenAI-compatible Provider 不可用。")


def init_evolink_provider(api_key: str, base_url: str = ""):
    """兼容旧接口：初始化或更新 OpenAI-compatible Provider。"""
    global evolink_provider
    if not api_key:
        return
    url = base_url or openai_compatible_base_url
    from providers.openai_compatible import OpenAICompatibleProvider
    evolink_provider = OpenAICompatibleProvider(
        api_key=api_key,
        base_url=url,
        file_base_url=openai_compatible_file_base_url,
    )
    print(f"已通过界面初始化 OpenAI-compatible Provider (base_url={url})")


def init_gemini_client(api_key: str):
    """用指定的 API Key 初始化或更新 Gemini Client（供界面动态传入）。"""
    global gemini_client
    if not api_key:
        return
    try:
        from google import genai
        gemini_client = genai.Client(
            api_key=api_key,
            http_options={"base_url": google_compatible_base_url} if google_compatible_base_url else None,
        )
        print("已通过界面初始化 Google-compatible Client")
    except ImportError:
        print("警告：未安装 google-genai，Gemini Client 不可用。请运行 pip install google-genai")


def fetch_openai_compatible_models(base_url: str, api_key: str) -> list[dict[str, Any]]:
    """同步读取 OpenAI-compatible 站点的 /v1/models。"""
    url = f"{base_url.rstrip('/')}/v1/models"
    req = urllib_request.Request(url)
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")

    with urllib_request.urlopen(req, timeout=20) as resp:
        payload = json.loads(resp.read().decode("utf-8", errors="replace"))

    data = payload.get("data", [])
    return data if isinstance(data, list) else []


def get_model_metadata(runtime_clients: RuntimeClients | None, model_name: str) -> dict[str, Any] | None:
    """从运行时缓存的模型目录里取单个模型元数据。"""
    if runtime_clients is None or not runtime_clients.model_catalog:
        return None
    for item in runtime_clients.model_catalog:
        if item.get("id") == model_name:
            return item
    return None


def resolve_text_max_output_tokens(
    model_name: str,
    provider: str,
    runtime_clients: RuntimeClients | None,
    fallback: int = 50000,
) -> int:
    """
    根据模型元数据尽量自动推断文本输出 token 上限。
    如果 /v1/models 不提供能力字段，则回退到保守默认值。
    """
    metadata = get_model_metadata(runtime_clients, model_name)
    candidate_fields = [
        "max_output_tokens",
        "max_completion_tokens",
        "output_token_limit",
        "completion_token_limit",
    ]

    if metadata:
        for field in candidate_fields:
            value = metadata.get(field)
            if isinstance(value, int) and value > 0:
                return min(value, fallback)

    known_high_context_models = (
        "gpt-5",
        "gpt-5.1",
        "gpt-5.2",
        "gpt-5.3",
        "gpt-5.4",
    )
    if provider == "openai_compatible" and model_name.startswith(known_high_context_models):
        # 这些模型即便 /v1/models 不暴露能力字段，也不应被压到 8192。
        # 这里保留调用方给出的 fallback，让高上下文模型继续按较高预算运行。
        return fallback

    if provider == "openai_compatible":
        # 对未知 OpenAI-compatible 模型，如果 /v1/models 不暴露能力字段，保守回退。
        return min(fallback, 8192)
    return fallback


def create_runtime_clients(
    provider: str = "",
    api_key: str = "",
    base_url: str = "",
    google_api_key: str = "",
    openai_api_key_override: str = "",
    anthropic_api_key_override: str = "",
) -> RuntimeClients:
    """
    Create a per-run client bundle without mutating module globals.
    """
    clients = RuntimeClients()

    if provider in ("openai_compatible", "evolink"):
        key = api_key or openai_compatible_api_key
        if key:
            from providers.openai_compatible import OpenAICompatibleProvider
            clients.evolink_provider = OpenAICompatibleProvider(
                api_key=key,
                base_url=base_url or openai_compatible_base_url,
                file_base_url=openai_compatible_file_base_url,
            )
            try:
                clients.model_catalog = fetch_openai_compatible_models(
                    base_url=base_url or openai_compatible_base_url,
                    api_key=key,
                )
            except Exception as e:
                print(f"警告：读取 OpenAI-compatible /v1/models 失败，将回退到默认 token 上限：{e}")
    elif provider in ("google_compatible", "gemini"):
        key = api_key or google_api_key or get_config_val("api_keys", "google_api_key", "GOOGLE_API_KEY", "")
        if key:
            try:
                from google import genai
                resolved_base_url = base_url or google_compatible_base_url
                clients.gemini_client = genai.Client(
                    api_key=key,
                    http_options={"base_url": resolved_base_url} if resolved_base_url else None,
                )
            except ImportError:
                print("警告：未安装 google-genai，Gemini Client 不可用。请运行 pip install google-genai")

    openai_key = openai_api_key_override or get_config_val("api_keys", "openai_api_key", "OPENAI_API_KEY", "")
    if openai_key:
        try:
            from openai import AsyncOpenAI
            clients.openai_client = AsyncOpenAI(api_key=openai_key)
        except ImportError:
            print("警告：未安装 openai，OpenAI Client 不可用。")

    anthropic_key = anthropic_api_key_override or get_config_val("api_keys", "anthropic_api_key", "ANTHROPIC_API_KEY", "")
    if anthropic_key:
        try:
            from anthropic import AsyncAnthropic
            clients.anthropic_client = AsyncAnthropic(api_key=anthropic_key)
        except ImportError:
            print("警告：未安装 anthropic，Anthropic Client 不可用。")

    return clients


async def close_runtime_clients(runtime_clients: RuntimeClients | None):
    """Close any closable clients owned by a per-run bundle."""
    if runtime_clients is None:
        return

    evolink = runtime_clients.evolink_provider
    if evolink and hasattr(evolink, "close"):
        await evolink.close()


# ==================== 原始 Provider 初始化（保留兼容性） ====================

gemini_client = None
anthropic_client = None
openai_client = None

api_key = get_config_val("api_keys", "google_api_key", "GOOGLE_API_KEY", "")
if api_key:
    try:
        from google import genai
        from google.genai import types
        gemini_client = genai.Client(
            api_key=api_key,
            http_options={"base_url": google_compatible_base_url} if google_compatible_base_url else None,
        )
        print("已初始化 Google-compatible Client")
    except ImportError:
        print("警告：未安装 google-genai，Gemini Client 不可用。")

anthropic_api_key = get_config_val("api_keys", "anthropic_api_key", "ANTHROPIC_API_KEY", "")
if anthropic_api_key:
    try:
        from anthropic import AsyncAnthropic
        anthropic_client = AsyncAnthropic(api_key=anthropic_api_key)
        print("已初始化 Anthropic Client")
    except ImportError:
        print("警告：未安装 anthropic，Anthropic Client 不可用。")

openai_api_key = get_config_val("api_keys", "openai_api_key", "OPENAI_API_KEY", "")
if openai_api_key:
    try:
        from openai import AsyncOpenAI
        openai_client = AsyncOpenAI(api_key=openai_api_key)
        print("已初始化 OpenAI Client")
    except ImportError:
        print("警告：未安装 openai，OpenAI Client 不可用。")


# ==================== Evolink 调用函数 ====================

async def call_evolink_text_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=5, error_context="", runtime_clients: RuntimeClients | None = None, progress_callback=None
):
    """
    通过 Evolink Provider 进行文本生成。

    Args:
        model_name: 模型名称（如 "gemini-2.5-flash"）
        contents: 通用内容列表
        config: 配置字典或对象，需包含 system_instruction, temperature, max_output_tokens
        max_attempts: 最大重试次数
        retry_delay: 重试间隔
        error_context: 错误上下文
    """
    provider = runtime_clients.evolink_provider if runtime_clients else evolink_provider
    print(f"[DEBUG] call_evolink_text: model={model_name}, provider={'已初始化' if provider else '未初始化'}")
    if provider is None:
        raise RuntimeError("Evolink Provider 未初始化，请检查 EVOLINK_API_KEY 配置。")

    # 从 config 中提取参数（兼容 types.GenerateContentConfig 和 dict）
    if hasattr(config, 'system_instruction'):
        system_prompt = config.system_instruction or ""
        temperature = config.temperature
        max_output_tokens = config.max_output_tokens
        print(f"[DEBUG] call_evolink_text: 从 GenerateContentConfig 提取参数")
    elif isinstance(config, dict):
        system_prompt = config.get("system_prompt", "")
        temperature = config.get("temperature", 1.0)
        max_output_tokens = config.get("max_output_tokens", 50000)
        print(f"[DEBUG] call_evolink_text: 从 dict 提取参数")
    else:
        system_prompt = ""
        temperature = 1.0
        max_output_tokens = 50000
        print(f"[DEBUG] call_evolink_text: 使用默认参数, config type={type(config)}")

    return await provider.generate_text(
        model_name=model_name,
        contents=contents,
        system_prompt=system_prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        max_attempts=max_attempts,
        retry_delay=retry_delay,
        error_context=error_context,
        progress_callback=progress_callback,
    )


async def upload_image_to_evolink(image_b64: str, media_type: str = "image/jpeg", runtime_clients: RuntimeClients | None = None) -> str:
    """
    将 base64 图片上传到 Evolink 文件服务，返回可访问的 URL。

    用于 image-to-image 场景（如 Polish Agent），需要先把本地 base64 图片
    上传为 URL，才能传给图像生成 API 的 image_urls 参数。
    """
    provider = runtime_clients.evolink_provider if runtime_clients else evolink_provider
    if provider is None:
        raise RuntimeError("Evolink Provider 未初始化，请检查 EVOLINK_API_KEY 配置。")
    url = await provider.upload_image_base64(image_b64, media_type)
    if not url:
        raise RuntimeError("图片上传到 Evolink 文件服务失败")
    return url


async def call_evolink_image_with_retry_async(
    model_name, prompt, config, max_attempts=5, retry_delay=30, error_context="", runtime_clients: RuntimeClients | None = None, progress_callback=None
):
    """
    通过 Evolink Provider 进行图像生成。

    Args:
        model_name: 图像模型名称（如 "nano-banana-2-lite"，通过 /v1/images/generations）
        prompt: 图像描述提示词
        config: 配置字典，需包含 aspect_ratio, quality 等
        max_attempts: 最大重试次数
        retry_delay: 重试间隔
        error_context: 错误上下文
    """
    provider = runtime_clients.evolink_provider if runtime_clients else evolink_provider
    print(f"[DEBUG] call_evolink_image: model={model_name}, config={config}, provider={'已初始化' if provider else '未初始化'}")
    if provider is None:
        raise RuntimeError("Evolink Provider 未初始化，请检查 EVOLINK_API_KEY 配置。")

    aspect_ratio = config.get("aspect_ratio", "16:9")
    quality = config.get("quality", "2K")
    image_urls = config.get("image_urls", None)

    return await provider.generate_image(
        model_name=model_name,
        prompt=prompt,
        aspect_ratio=aspect_ratio,
        quality=quality,
        image_urls=image_urls,
        max_attempts=max_attempts,
        retry_delay=retry_delay,
        error_context=error_context,
        progress_callback=progress_callback,
    )


# ==================== 原始 Gemini 调用函数（保留兼容性） ====================

def _convert_to_gemini_parts(contents):
    """将通用内容列表转换为 Gemini 的 Part 对象列表"""
    from google.genai import types
    gemini_parts = []
    for item in contents:
        if item.get("type") == "text":
            gemini_parts.append(types.Part.from_text(text=item["text"]))
        elif item.get("type") == "image":
            source = item.get("source", {})
            if source.get("type") == "base64":
                gemini_parts.append(
                    types.Part.from_bytes(
                        data=base64.b64decode(source["data"]),
                        mime_type=source["media_type"],
                    )
                )
    return gemini_parts


def _is_gemini_daily_quota_exhausted(error: Exception) -> bool:
    """
    Detect non-recoverable Gemini daily quota exhaustion errors.
    """
    error_text = str(error)
    lowered = error_text.lower()
    return (
        "resource_exhausted" in lowered
        and (
            "generate_requests_per_model_per_day" in lowered
            or "generaterequestsperdayperprojectpermodel" in lowered
            or "please retry in" in lowered
        )
    )


async def call_gemini_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=5, error_context="", runtime_clients: RuntimeClients | None = None
):
    """原始 Gemini API 异步调用（保留兼容性）"""
    from google.genai import types

    result_list = []
    target_candidate_count = config.candidate_count
    if config.candidate_count > 8:
        config.candidate_count = 8

    current_contents = contents
    for attempt in range(max_attempts):
        try:
            client = runtime_clients.gemini_client if runtime_clients else gemini_client
            if client is None:
                raise RuntimeError("Gemini Client 未初始化，请检查 GOOGLE_API_KEY 配置。")
            gemini_contents = _convert_to_gemini_parts(current_contents)
            response = await client.aio.models.generate_content(
                model=model_name, contents=gemini_contents, config=config
            )

            if "nanoviz" in model_name or "image" in model_name:
                raw_response_list = []
                if not response.candidates or not response.candidates[0].content.parts:
                    print(f"[Warning]: Failed to generate image, retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    continue
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        raw_response_list.append(
                            base64.b64encode(part.inline_data.data).decode("utf-8")
                        )
                        break
            else:
                raw_response_list = [
                    part.text
                    for candidate in response.candidates
                    for part in candidate.content.parts
                ]
            result_list.extend([r for r in raw_response_list if r.strip() != ""])
            if len(result_list) >= target_candidate_count:
                result_list = result_list[:target_candidate_count]
                break

        except Exception as e:
            context_msg = f" for {error_context}" if error_context else ""
            if _is_gemini_daily_quota_exhausted(e):
                print(
                    f"Gemini daily quota exhausted for model {model_name}{context_msg}: {e}. "
                    "Skipping further retries for this request."
                )
                result_list = ["QuotaExceeded"] * target_candidate_count
                break
            current_delay = min(retry_delay * (2 ** attempt), 30)
            print(
                f"Attempt {attempt + 1} for model {model_name} failed{context_msg}: {e}. Retrying in {current_delay} seconds..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(current_delay)
            else:
                print(f"Error: All {max_attempts} attempts failed{context_msg}")
                result_list = ["Error"] * target_candidate_count

    if len(result_list) < target_candidate_count:
        result_list.extend(["Error"] * (target_candidate_count - len(result_list)))
    return result_list


# ==================== 原始 Claude/OpenAI 调用函数（保留兼容性） ====================

def _convert_to_claude_format(contents):
    return contents

def _convert_to_openai_format(contents):
    openai_contents = []
    for item in contents:
        if item.get("type") == "text":
            openai_contents.append({"type": "text", "text": item["text"]})
        elif item.get("type") == "image":
            source = item.get("source", {})
            if source.get("type") == "base64":
                media_type = source.get("media_type", "image/jpeg")
                data = source.get("data", "")
                data_url = f"data:{media_type};base64,{data}"
                openai_contents.append({
                    "type": "image_url",
                    "image_url": {"url": data_url}
                })
    return openai_contents


async def call_claude_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=30, error_context="", runtime_clients: RuntimeClients | None = None
):
    """原始 Claude API 异步调用（保留兼容性）"""
    system_prompt = config["system_prompt"]
    temperature = config["temperature"]
    candidate_num = config["candidate_num"]
    max_output_tokens = config["max_output_tokens"]
    response_text_list = []

    current_contents = contents
    is_input_valid = False
    for attempt in range(max_attempts):
        try:
            client = runtime_clients.anthropic_client if runtime_clients else anthropic_client
            if client is None:
                raise RuntimeError("Anthropic Client 未初始化，请检查 ANTHROPIC_API_KEY 配置。")
            claude_contents = _convert_to_claude_format(current_contents)
            first_response = await client.messages.create(
                model=model_name,
                max_tokens=max_output_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": claude_contents}],
                system=system_prompt,
            )
            response_text_list.append(first_response.content[0].text)
            is_input_valid = True
            break
        except Exception as e:
            error_str = str(e).lower()
            context_msg = f" for {error_context}" if error_context else ""
            print(f"Validation attempt {attempt + 1} failed{context_msg}: {error_str}. Retrying in {retry_delay} seconds...")
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)

    if not is_input_valid:
        print(f"Error: All {max_attempts} attempts failed to validate the input. Returning errors.")
        return ["Error"] * candidate_num

    remaining_candidates = candidate_num - 1
    if remaining_candidates > 0:
        valid_claude_contents = _convert_to_claude_format(current_contents)
        client = runtime_clients.anthropic_client if runtime_clients else anthropic_client
        tasks = [
            client.messages.create(
                model=model_name,
                max_tokens=max_output_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": valid_claude_contents}],
                system=system_prompt,
            )
            for _ in range(remaining_candidates)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                response_text_list.append("Error")
            else:
                response_text_list.append(res.content[0].text)

    return response_text_list


async def call_openai_with_retry_async(
    model_name, contents, config, max_attempts=5, retry_delay=30, error_context="", runtime_clients: RuntimeClients | None = None
):
    """原始 OpenAI API 异步调用（保留兼容性）"""
    system_prompt = config["system_prompt"]
    temperature = config["temperature"]
    candidate_num = config["candidate_num"]
    max_completion_tokens = config["max_completion_tokens"]
    response_text_list = []

    current_contents = contents
    is_input_valid = False
    for attempt in range(max_attempts):
        try:
            client = runtime_clients.openai_client if runtime_clients else openai_client
            if client is None:
                raise RuntimeError("OpenAI Client 未初始化，请检查 OPENAI_API_KEY 配置。")
            openai_contents = _convert_to_openai_format(current_contents)
            first_response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": openai_contents}
                ],
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )
            response_text_list.append(first_response.choices[0].message.content)
            is_input_valid = True
            break
        except Exception as e:
            error_str = str(e).lower()
            context_msg = f" for {error_context}" if error_context else ""
            print(f"Validation attempt {attempt + 1} failed{context_msg}: {error_str}. Retrying in {retry_delay} seconds...")
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)

    if not is_input_valid:
        print(f"Error: All {max_attempts} attempts failed to validate the input. Returning errors.")
        return ["Error"] * candidate_num

    remaining_candidates = candidate_num - 1
    if remaining_candidates > 0:
        valid_openai_contents = _convert_to_openai_format(current_contents)
        client = runtime_clients.openai_client if runtime_clients else openai_client
        tasks = [
            client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": valid_openai_contents}
                ],
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )
            for _ in range(remaining_candidates)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                response_text_list.append("Error")
            else:
                response_text_list.append(res.choices[0].message.content)

    return response_text_list


async def call_openai_image_generation_with_retry_async(
    model_name, prompt, config, max_attempts=5, retry_delay=30, error_context="", runtime_clients: RuntimeClients | None = None
):
    """原始 OpenAI 图像生成 API 异步调用（保留兼容性）"""
    size = config.get("size", "1536x1024")
    quality = config.get("quality", "high")
    background = config.get("background", "opaque")
    output_format = config.get("output_format", "png")

    gen_params = {
        "model": model_name,
        "prompt": prompt,
        "n": 1,
        "size": size,
        "quality": quality,
        "background": background,
        "output_format": output_format,
    }

    for attempt in range(max_attempts):
        try:
            client = runtime_clients.openai_client if runtime_clients else openai_client
            if client is None:
                raise RuntimeError("OpenAI Client 未初始化，请检查 OPENAI_API_KEY 配置。")
            response = await client.images.generate(**gen_params)
            if response.data and response.data[0].b64_json:
                return [response.data[0].b64_json]
            else:
                print(f"[Warning]: Failed to generate image via OpenAI, no data returned.")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(retry_delay)
                continue
        except Exception as e:
            context_msg = f" for {error_context}" if error_context else ""
            print(f"Attempt {attempt + 1} for OpenAI image generation model {model_name} failed{context_msg}: {e}. Retrying in {retry_delay} seconds...")
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_delay)
            else:
                print(f"Error: All {max_attempts} attempts failed{context_msg}")
                return ["Error"]

    return ["Error"]
