"""
OpenAI-compatible API Provider
支持文本生成（OpenAI 兼容接口）和图像生成（异步任务接口）
"""

import asyncio
import base64
import json
import re
from typing import List, Dict, Any, Optional, Callable

import aiohttp

from .base import BaseProvider


class ClientError(Exception):
    """4xx 客户端错误，不应重试（如 400 Bad Request、401 Unauthorized）"""
    pass


class OpenAICompatibleProvider(BaseProvider):
    """
    OpenAI-compatible API Provider

    文本模型: 通过 /v1/chat/completions (OpenAI 兼容)
    图像模型: 通过 /v1/images/generations (异步任务) + /v1/tasks/{id} (轮询)
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.evolink.ai",
        file_base_url: str = "",
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.file_base_url = file_base_url.rstrip("/") if file_base_url else ""
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """获取共享的 aiohttp session，避免每次请求都创建新 session"""
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=30)
            self._session = aiohttp.ClientSession(connector=connector)
        return self._session

    async def close(self):
        """关闭共享 session"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    # ==================== 内容格式转换 ====================

    def _convert_contents_to_messages(
        self,
        contents: List[Dict[str, Any]],
        system_prompt: str = "",
    ) -> List[Dict[str, Any]]:
        """
        将通用内容列表转换为 OpenAI 兼容的 messages 格式

        通用格式（项目中使用的）:
        [
            {"type": "text", "text": "..."},
            {"type": "image", "source": {"type": "base64", "data": "...", "media_type": "image/jpeg"}},
            {"type": "image", "image_base64": "..."},  # planner agent 使用的简化格式
        ]

        转换为 OpenAI 格式:
        [
            {"role": "system", "content": "..."},
            {"role": "user", "content": [
                {"type": "text", "text": "..."},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
            ]},
        ]
        """
        messages = []

        # system prompt
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # 构建 user message 的 content 部分
        user_parts = []
        has_image = False

        for item in contents:
            item_type = item.get("type", "")

            if item_type == "text":
                user_parts.append({"type": "text", "text": item["text"]})

            elif item_type == "image":
                has_image = True
                # 两种图片格式：source 嵌套格式 和 image_base64 直接格式
                source = item.get("source", {})
                if source.get("type") == "base64":
                    media_type = source.get("media_type", "image/jpeg")
                    data = source.get("data", "")
                    data_url = f"data:{media_type};base64,{data}"
                    user_parts.append({
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    })
                elif "image_base64" in item:
                    data_url = f"data:image/jpeg;base64,{item['image_base64']}"
                    user_parts.append({
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    })

        # 如果没有图片，可以简化为纯文本
        if not has_image and len(user_parts) == 1:
            messages.append({"role": "user", "content": user_parts[0]["text"]})
        else:
            messages.append({"role": "user", "content": user_parts})

        return messages

    # ==================== 请求构建 ====================

    def _build_text_payload(
        self,
        model_name: str,
        contents: List[Dict[str, Any]],
        system_prompt: str,
        temperature: float,
        max_output_tokens: int,
    ) -> Dict[str, Any]:
        """构建文本生成请求体"""
        messages = self._convert_contents_to_messages(contents, system_prompt)
        return {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_output_tokens,
        }

    def _build_image_payload(
        self,
        model_name: str,
        prompt: str,
        aspect_ratio: str,
        quality: str,
        image_urls: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """构建图像生成请求体"""
        payload = {
            "model": model_name,
            "prompt": prompt,
            "size": aspect_ratio,
            "quality": quality,
        }
        if image_urls:
            payload["image_urls"] = image_urls
        return payload

    # ==================== HTTP 请求封装 ====================

    async def _post_json(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """发送 POST 请求并返回 JSON 响应"""
        print(f"[DEBUG] [Evolink] POST {url}")
        print(f"[DEBUG] [Evolink]   model={payload.get('model', 'N/A')}, payload keys={list(payload.keys())}")
        session = await self._get_session()
        async with session.post(
            url,
            json=payload,
            headers=self._get_headers(),
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            status = resp.status
            content_type = resp.headers.get("Content-Type", "").lower()
            raw_text = await resp.text()
            if "json" in content_type:
                try:
                    body = json.loads(raw_text)
                except Exception:
                    body = raw_text
            else:
                body = raw_text

            if isinstance(body, dict):
                body_summary = f"keys={list(body.keys())}"
            else:
                preview = str(body)[:160].replace("\n", " ")
                body_summary = f"content_type={content_type or 'unknown'}, body_preview={preview}"
            print(f"[DEBUG] [Evolink]   响应 status={status}, {body_summary}")
            if status >= 400:
                error_msg = body.get("error", body) if isinstance(body, dict) else body
                print(f"[DEBUG] [Evolink]   ❌ 错误详情: {error_msg}")
                # 4xx 客户端错误不重试，直接抛出特定异常
                if 400 <= status < 500 and status != 429:
                    raise ClientError(f"HTTP {status}: {error_msg}")
            resp.raise_for_status()
            if not isinstance(body, dict):
                raise RuntimeError(f"HTTP {status}: expected JSON response but got {content_type or 'unknown'}")
            return body

    async def _get_json(self, url: str) -> Dict[str, Any]:
        """发送 GET 请求并返回 JSON 响应"""
        session = await self._get_session()
        async with session.get(
            url,
            headers=self._get_headers(),
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            status = resp.status
            content_type = resp.headers.get("Content-Type", "").lower()
            raw_text = await resp.text()
            if "json" in content_type:
                try:
                    body = json.loads(raw_text)
                except Exception:
                    body = raw_text
            else:
                body = raw_text
            if status >= 400:
                print(f"[DEBUG] [Evolink] GET {url} ❌ status={status}, body={body}")
            resp.raise_for_status()
            if not isinstance(body, dict):
                raise RuntimeError(f"HTTP {status}: expected JSON response but got {content_type or 'unknown'}")
            return body

    async def _download_image_as_base64(self, url: str) -> Optional[str]:
        """从 URL 下载图片并转换为 base64"""
        try:
            session = await self._get_session()
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                resp.raise_for_status()
                image_data = await resp.read()
                return base64.b64encode(image_data).decode("utf-8")
        except Exception as e:
            print(f"下载图片失败 ({url}): {e}")
            return None

    async def _emit_progress(self, progress_callback: Optional[Callable[[str], Any]], message: str):
        if progress_callback is None:
            return
        result = progress_callback(message)
        if asyncio.iscoroutine(result):
            await result

    async def _iter_sse_events(self, resp):
        """Yield full SSE data payloads instead of raw transport chunks."""
        data_lines = []
        while True:
            line_bytes = await resp.content.readline()
            if not line_bytes:
                if data_lines:
                    yield "\n".join(data_lines)
                break

            line = line_bytes.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line:
                if data_lines:
                    yield "\n".join(data_lines)
                    data_lines = []
                continue

            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())

    def _extract_text_delta(self, delta: Any) -> str:
        if isinstance(delta, str):
            return delta
        if isinstance(delta, list):
            parts = []
            for item in delta:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "".join(parts)
        return ""

    def _extract_image_reference(self, value: Any) -> Optional[str]:
        """Extract a usable image reference (URL or base64/data URL) from nested values."""
        if isinstance(value, str):
            candidate = value.strip()
            if not candidate:
                return None
            if candidate.startswith("http://") or candidate.startswith("https://"):
                return candidate
            if candidate.startswith("data:image/"):
                return candidate
            if re.fullmatch(r"[A-Za-z0-9+/=]{256,}", candidate):
                return candidate
            md_match = re.search(r"!\[[^\]]*\]\((https?://[^)]+)\)", candidate)
            if md_match:
                return md_match.group(1)
            return None

        if isinstance(value, dict):
            for key in ("image", "image_url", "url", "b64_json", "data"):
                if key in value:
                    extracted = self._extract_image_reference(value[key])
                    if extracted:
                        return extracted
            for nested in value.values():
                extracted = self._extract_image_reference(nested)
                if extracted:
                    return extracted
            return None

        if isinstance(value, list):
            for item in value:
                extracted = self._extract_image_reference(item)
                if extracted:
                    return extracted

        return None

    async def _materialize_image_reference(self, ref: str) -> Optional[str]:
        """Turn a URL/data URL/raw base64 into a plain base64 image payload."""
        if not ref:
            return None
        if ref.startswith("http://") or ref.startswith("https://"):
            return await self._download_image_as_base64(ref)
        if ref.startswith("data:image/") and "," in ref:
            return ref.split(",", 1)[1]
        return ref

    async def _generate_image_via_chat_completions_stream(
        self,
        model_name: str,
        prompt: str,
        image_urls: Optional[List[str]] = None,
        max_attempts: int = 3,
        retry_delay: float = 30,
        error_context: str = "",
        progress_callback: Optional[Callable[[str], Any]] = None,
    ) -> List[str]:
        """Fallback image generation path for providers exposing image models via streamed chat completions."""
        url = f"{self.base_url}/v1/chat/completions"
        content = [{"type": "text", "text": prompt}]
        for image_url in image_urls or []:
            content.append({"type": "image_url", "image_url": {"url": image_url}})

        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": [{"role": "user", "content": content}],
            "stream": True,
            "temperature": 1,
            "max_tokens": 1024,
        }
        if image_urls:
            payload["image"] = image_urls[0]

        for attempt in range(max_attempts):
            try:
                print(f"[OpenAI-compatible 图像] 回退到 /v1/chat/completions 流式生成")
                session = await self._get_session()
                async with session.post(
                    url,
                    json=payload,
                    headers=self._get_headers(),
                    timeout=aiohttp.ClientTimeout(total=600),
                ) as resp:
                    resp.raise_for_status()
                    if "text/event-stream" not in resp.headers.get("Content-Type", "").lower():
                        text = await resp.text()
                        raise RuntimeError(f"Unexpected non-stream response: {text[:200]}")

                    async for data_str in self._iter_sse_events(resp):
                        if data_str == "[DONE]":
                            break
                        try:
                            payload_obj = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        if isinstance(payload_obj, dict) and "error" in payload_obj:
                            raise RuntimeError(payload_obj["error"].get("message", payload_obj["error"]))

                        delta = None
                        if isinstance(payload_obj, dict):
                            choices = payload_obj.get("choices", [])
                            if choices and isinstance(choices, list):
                                delta = choices[0].get("delta", {})
                        if isinstance(delta, dict):
                            progress_text = delta.get("reasoning_content")
                            if progress_text:
                                await self._emit_progress(progress_callback, progress_text.strip())

                        ref = self._extract_image_reference(payload_obj)
                        if ref:
                            print(f"[OpenAI-compatible 图像] 捕获到图片引用: {ref[:120]}...")
                            b64_image = await self._materialize_image_reference(ref)
                            if b64_image:
                                print(f"[OpenAI-compatible 图像] 成功获取图片数据，长度={len(b64_image)}")
                                return [b64_image]

                print("[OpenAI-compatible 图像] 流式返回中未找到图片数据")

            except ClientError as e:
                context_msg = f" ({error_context})" if error_context else ""
                print(f"[OpenAI-compatible 图像] ❌ 客户端错误{context_msg}: {e}。不再重试。")
                return ["Error"]
            except Exception as e:
                context_msg = f" ({error_context})" if error_context else ""
                current_delay = min(retry_delay * (2 ** attempt), 60)
                print(f"[OpenAI-compatible 图像] 第 {attempt + 1} 次流式尝试失败{context_msg}: {e}。{current_delay}s 后重试...")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(current_delay)
                else:
                    print(f"[OpenAI-compatible 图像] 全部 {max_attempts} 次流式尝试失败{context_msg}")

        return ["Error"]

    async def _generate_text_stream(
        self,
        model_name: str,
        contents: List[Dict[str, Any]],
        system_prompt: str = "",
        temperature: float = 1.0,
        max_output_tokens: int = 12000,
        max_attempts: int = 3,
        retry_delay: float = 3,
        error_context: str = "",
        progress_callback: Optional[Callable[[Any], Any]] = None,
    ) -> List[str]:
        """Generate text via streamed chat completions and emit partial deltas."""
        url = f"{self.base_url}/v1/chat/completions"
        payload = self._build_text_payload(
            model_name=model_name,
            contents=contents,
            system_prompt=system_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        payload["stream"] = True

        content_types = [item.get("type", "?") for item in contents]
        sys_len = len(system_prompt) if system_prompt else 0
        print(f"[DEBUG] [Evolink 文本流] 请求: model={model_name}, temp={temperature}, max_tokens={max_output_tokens}")
        print(f"[DEBUG] [Evolink 文本流]   内容: {content_types}, system_prompt 长度={sys_len}")

        for attempt in range(max_attempts):
            try:
                session = await self._get_session()
                async with session.post(
                    url,
                    json=payload,
                    headers=self._get_headers(),
                    timeout=aiohttp.ClientTimeout(total=600),
                ) as resp:
                    resp.raise_for_status()
                    content_type = resp.headers.get("Content-Type", "").lower()

                    if "text/event-stream" not in content_type:
                        text = await resp.text()
                        try:
                            body = json.loads(text)
                            choices = body.get("choices", [])
                            if choices:
                                final_text = choices[0].get("message", {}).get("content", "")
                                if final_text:
                                    await self._emit_progress(progress_callback, {"text": final_text, "delta": final_text})
                                    return [final_text]
                        except Exception:
                            pass
                        raise RuntimeError(f"Unexpected non-stream response: {text[:200]}")

                    full_text_parts: list[str] = []
                    reasoning_parts: list[str] = []

                    async for data_str in self._iter_sse_events(resp):
                        if data_str == "[DONE]":
                            break
                        try:
                            payload_obj = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        if isinstance(payload_obj, dict) and "error" in payload_obj:
                            raise RuntimeError(payload_obj["error"].get("message", payload_obj["error"]))

                        choices = payload_obj.get("choices", []) if isinstance(payload_obj, dict) else []
                        if not choices or not isinstance(choices, list):
                            continue
                        delta = choices[0].get("delta", {})
                        if not isinstance(delta, dict):
                            continue

                        reasoning_delta = delta.get("reasoning_content")
                        if isinstance(reasoning_delta, str) and reasoning_delta:
                            reasoning_parts.append(reasoning_delta)

                        text_delta = self._extract_text_delta(delta.get("content"))
                        if text_delta:
                            full_text_parts.append(text_delta)
                            await self._emit_progress(
                                progress_callback,
                                {
                                    "delta": text_delta,
                                    "text": "".join(full_text_parts),
                                    "reasoning": "".join(reasoning_parts),
                                },
                            )

                    final_text = "".join(full_text_parts).strip()
                    if final_text:
                        print(f"[DEBUG] [Evolink 文本流] ✓ 成功, 响应长度={len(final_text)}")
                        return [final_text]

                    print(f"[Evolink 文本流] 响应为空，{retry_delay}s 后重试...")
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(retry_delay)

            except ClientError as e:
                context_msg = f" ({error_context})" if error_context else ""
                print(f"[Evolink 文本流] ❌ 客户端错误{context_msg}: {e}。不再重试。")
                return ["Error"]
            except Exception as e:
                context_msg = f" ({error_context})" if error_context else ""
                current_delay = min(retry_delay * (2 ** attempt), 30)
                print(
                    f"[Evolink 文本流] 第 {attempt + 1} 次尝试失败{context_msg}: {e}。"
                    f"{current_delay}s 后重试..."
                )
                if attempt < max_attempts - 1:
                    await asyncio.sleep(current_delay)
                else:
                    print(f"[Evolink 文本流] 全部 {max_attempts} 次尝试失败{context_msg}")

        return ["Error"]

    # ==================== 文件上传 ====================

    async def upload_image_base64(self, image_b64: str, media_type: str = "image/jpeg") -> Optional[str]:
        """
        将 base64 图片上传到 Evolink 文件服务，返回可访问的 URL。

        用于 image-to-image 场景：先上传参考图，再把 URL 传给图像生成 API 的 image_urls 参数。

        Args:
            image_b64: 纯 base64 编码的图片数据（不带 data: 前缀）
            media_type: MIME 类型，默认 image/jpeg

        Returns:
            上传成功返回 file_url，失败返回 None
        """
        if not self.file_base_url:
            print("[OpenAI-compatible 上传] ❌ 未配置 file_base_url，无法上传参考图。")
            return None

        upload_url = f"{self.file_base_url}/api/v1/files/upload/base64"
        data_url = f"data:{media_type};base64,{image_b64}"

        try:
            session = await self._get_session()
            async with session.post(
                upload_url,
                json={"base64_data": data_url},
                headers=self._get_headers(),
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                status = resp.status
                body = await resp.json()
                if status == 200 and body.get("success"):
                    file_url = body.get("data", {}).get("file_url", "")
                    print(f"[Evolink 上传] ✓ 图片已上传: {file_url[:80]}...")
                    return file_url
                else:
                    print(f"[Evolink 上传] ❌ 上传失败: status={status}, body={body}")
                    return None
        except Exception as e:
            print(f"[Evolink 上传] ❌ 上传异常: {e}")
            return None

    # ==================== 文本生成 ====================

    async def generate_text(
        self,
        model_name: str,
        contents: List[Dict[str, Any]],
        system_prompt: str = "",
        temperature: float = 1.0,
        max_output_tokens: int = 50000,
        max_attempts: int = 3,
        retry_delay: float = 5,
        error_context: str = "",
        progress_callback: Optional[Callable[[Any], Any]] = None,
    ) -> List[str]:
        """
        通过 /v1/chat/completions 生成文本

        兼容 OpenAI Chat Completions API 格式
        """
        if progress_callback is not None:
            return await self._generate_text_stream(
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

        url = f"{self.base_url}/v1/chat/completions"
        payload = self._build_text_payload(
            model_name=model_name,
            contents=contents,
            system_prompt=system_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

        # 计算内容摘要
        content_types = [item.get("type", "?") for item in contents]
        sys_len = len(system_prompt) if system_prompt else 0
        print(f"[DEBUG] [Evolink 文本] 请求: model={model_name}, temp={temperature}, max_tokens={max_output_tokens}")
        print(f"[DEBUG] [Evolink 文本]   内容: {content_types}, system_prompt 长度={sys_len}")

        for attempt in range(max_attempts):
            try:
                response = await self._post_json(url, payload)

                # 提取文本响应
                choices = response.get("choices", [])
                if choices:
                    text = choices[0].get("message", {}).get("content", "")
                    if text.strip():
                        usage = response.get("usage", {})
                        print(f"[DEBUG] [Evolink 文本] ✓ 成功, 响应长度={len(text)}, usage={usage}")
                        return [text]

                print(f"[Evolink 文本] 响应为空，{retry_delay}s 后重试...")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(retry_delay)

            except ClientError as e:
                # 4xx 客户端错误，立即失败不重试（模型名错误、参数错误等）
                context_msg = f" ({error_context})" if error_context else ""
                print(f"[Evolink 文本] ❌ 客户端错误{context_msg}: {e}。不再重试。")
                return ["Error"]

            except Exception as e:
                context_msg = f" ({error_context})" if error_context else ""
                current_delay = min(retry_delay * (2 ** attempt), 30)
                print(
                    f"[Evolink 文本] 第 {attempt + 1} 次尝试失败{context_msg}: {e}。"
                    f"{current_delay}s 后重试..."
                )
                if attempt < max_attempts - 1:
                    await asyncio.sleep(current_delay)
                else:
                    print(f"[Evolink 文本] 全部 {max_attempts} 次尝试失败{context_msg}")

        return ["Error"]

    # ==================== 图像生成 ====================

    async def generate_image(
        self,
        model_name: str,
        prompt: str,
        aspect_ratio: str = "16:9",
        quality: str = "2K",
        image_urls: Optional[List[str]] = None,
        max_attempts: int = 3,
        retry_delay: float = 30,
        poll_interval: float = 3,
        max_polls: int = 60,
        error_context: str = "",
        progress_callback: Optional[Callable[[str], Any]] = None,
    ) -> List[str]:
        """
        生成图像。

        对 OpenAI-compatible 图像站点，优先走 /v1/chat/completions 的流式生成。
        旧的 /v1/images/generations 任务式接口仅保留为兼容代码，不再默认尝试。
        """
        return await self._generate_image_via_chat_completions_stream(
            model_name=model_name,
            prompt=prompt,
            image_urls=image_urls,
            max_attempts=max_attempts,
            retry_delay=retry_delay,
            error_context=error_context,
            progress_callback=progress_callback,
        )

    async def _generate_image_via_legacy_task_api(
        self,
        model_name: str,
        prompt: str,
        aspect_ratio: str = "16:9",
        quality: str = "2K",
        image_urls: Optional[List[str]] = None,
        max_attempts: int = 3,
        retry_delay: float = 30,
        poll_interval: float = 3,
        max_polls: int = 60,
        error_context: str = "",
    ) -> List[str]:
        """
        通过 /v1/images/generations 异步生成图像（保留兼容代码，不作为默认路径）。
        """
        create_url = f"{self.base_url}/v1/images/generations"
        print(f"[DEBUG] [Evolink 图像] 请求: model={model_name}, ratio={aspect_ratio}, quality={quality}")
        if image_urls:
            print(f"[DEBUG] [Evolink 图像]   附带 {len(image_urls)} 张参考图片")
        print(f"[DEBUG] [Evolink 图像]   prompt 长度={len(prompt)}, 前100字: {prompt[:100]}...")

        for attempt in range(max_attempts):
            try:
                # 步骤 1：创建任务
                payload = self._build_image_payload(
                    model_name=model_name,
                    prompt=prompt,
                    aspect_ratio=aspect_ratio,
                    quality=quality,
                    image_urls=image_urls,
                )
                create_response = await self._post_json(create_url, payload)
                task_id = create_response.get("id")

                if not task_id:
                    print(f"[Evolink 图像] 创建任务失败，未返回任务 ID")
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(retry_delay)
                    continue

                print(f"[Evolink 图像] 任务已创建: {task_id}")

                # 步骤 2：轮询任务状态
                poll_url = f"{self.base_url}/v1/tasks/{task_id}"
                for poll_count in range(max_polls):
                    if poll_interval > 0:
                        await asyncio.sleep(poll_interval)

                    poll_response = await self._get_json(poll_url)
                    status = poll_response.get("status", "")
                    progress = poll_response.get("progress", 0)

                    if status == "completed":
                        # 步骤 3：下载图片
                        results = poll_response.get("results", [])
                        if results:
                            image_url = results[0]
                            print(f"[Evolink 图像] 任务完成，下载图片: {image_url[:80]}...")
                            b64_image = await self._download_image_as_base64(image_url)
                            if b64_image:
                                return [b64_image]
                            else:
                                print(f"[Evolink 图像] 图片下载失败")
                                break
                        else:
                            print(f"[Evolink 图像] 任务完成但无图片结果")
                            break

                    elif status in ("failed", "cancelled"):
                        print(f"[Evolink 图像] 任务失败: {status}")
                        break

                    else:
                        # 仍在处理中
                        if poll_count % 5 == 0:
                            print(f"[Evolink 图像] 轮询 #{poll_count + 1}: {status}, 进度: {progress}%")

                # 如果轮询结束仍未完成
                context_msg = f" ({error_context})" if error_context else ""
                print(f"[Evolink 图像] 第 {attempt + 1} 次尝试未成功{context_msg}")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(retry_delay)

            except ClientError as e:
                # 4xx 客户端错误，立即失败不重试
                context_msg = f" ({error_context})" if error_context else ""
                print(f"[Evolink 图像] ❌ 客户端错误{context_msg}: {e}。不再重试。")
                return ["Error"]

            except Exception as e:
                context_msg = f" ({error_context})" if error_context else ""
                current_delay = min(retry_delay * (2 ** attempt), 60)
                print(
                    f"[Evolink 图像] 第 {attempt + 1} 次尝试失败{context_msg}: {e}。"
                    f"{current_delay}s 后重试..."
                )
                if attempt < max_attempts - 1:
                    await asyncio.sleep(current_delay)
                else:
                    print(f"[Evolink 图像] 全部 {max_attempts} 次尝试失败{context_msg}")

        return ["Error"]


EvolinkProvider = OpenAICompatibleProvider
