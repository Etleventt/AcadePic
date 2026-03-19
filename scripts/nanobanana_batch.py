#!/usr/bin/env python3
"""
Semi-automated Nano Banana workflow for MRI degradation transfer.

Inputs:
- a directory of reference cells, e.g. FLAIR__MM-GAN.png
- a directory of GT slices, e.g. FLAIR.png
- the prompt table JSON in configs/nanobanana_prompts.json

Outputs:
- per-job folders with round/candidate images
- review JSON for each candidate
- revised prompt drafts for later rounds
"""

from __future__ import annotations

import argparse
import asyncio
import json
import mimetypes
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from json_repair import repair_json


REVIEW_PROMPT = """你正在做严格的医学图像编辑质检。

你会看到三张图：
1. 目标退化参考图：告诉你这一格想模仿的退化方式
2. 原始 GT 解剖图：告诉你必须保留的解剖结构、切片形状和病灶大体位置
3. 候选生成结果：需要被打分的图

当前模态：{modality}
当前方法标签只供你区分任务，不要基于方法名作常识推断：{method}

原始提示词：
{prompt}

请严格判断候选结果是否同时满足以下几点：
- 解剖保持：脑轮廓、脑室拓扑、切片形状、病灶大体位置仍然来自 GT
- 退化匹配：候选图与参考退化图的整体质量、纹理、边缘、噪声、对比度、病灶失真方式足够相似
- 全局一致：退化是整图一致的，不是只有病灶一小块变差而其他区域异常清楚
- 医学合理：允许中等退化，但不能出现完全不合理的新大结构、病灶跑位、整体模态完全错乱

返回严格 JSON，不要加 markdown：
{{
  "pass": true,
  "scores": {{
    "anatomy_preservation": 0-10,
    "degradation_match": 0-10,
    "global_consistency": 0-10,
    "medical_plausibility": 0-10
  }},
  "summary": "一句话总结",
  "major_issues": ["..."],
  "minor_issues": ["..."],
  "prompt_adjustments": [
    "给下一轮提示词的可执行修改建议1",
    "给下一轮提示词的可执行修改建议2"
  ]
}}
"""

REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "pass": {"type": "boolean"},
        "scores": {
            "type": "object",
            "properties": {
                "anatomy_preservation": {"type": "number"},
                "degradation_match": {"type": "number"},
                "global_consistency": {"type": "number"},
                "medical_plausibility": {"type": "number"},
            },
            "required": [
                "anatomy_preservation",
                "degradation_match",
                "global_consistency",
                "medical_plausibility",
            ],
        },
        "summary": {"type": "string"},
        "major_issues": {"type": "array", "items": {"type": "string"}},
        "minor_issues": {"type": "array", "items": {"type": "string"}},
        "prompt_adjustments": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "pass",
        "scores",
        "summary",
        "major_issues",
        "minor_issues",
        "prompt_adjustments",
    ],
}


REWRITE_PROMPT = """你要把一段中文图像编辑提示词改写成下一轮可直接发送给图像编辑模型的完整提示词。

约束：
- 输出必须是完整中文提示词，不要解释，不要列表，不要 markdown
- 必须明确：第一张图是目标退化参考图，第二张图是待编辑 GT 图
- 必须继续强调：保留 GT 的切片形状、脑轮廓、脑室拓扑、病灶大体位置
- 必须把 reviewer 指出的失败点改进去
- 不要发散，不要新增和本任务无关的要求

当前模态：{modality}
当前方法标签：{method}

上一轮完整提示词：
{prompt}

上一轮 reviewer JSON：
{review_json}
"""


def slugify(value: str) -> str:
    return re.sub(r"[^\w.-]+", "_", value.strip(), flags=re.UNICODE).strip("_") or "item"


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", stripped)
        stripped = re.sub(r"\n?```$", "", stripped)
    return stripped.strip()


def parse_json_maybe(text: str) -> dict[str, Any]:
    repaired = repair_json(strip_code_fences(text), ensure_ascii=False)
    data = json.loads(repaired)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict review payload, got: {type(data)}")
    return data


def average_score(review: dict[str, Any]) -> float:
    scores = review.get("scores", {})
    values = [
        float(scores.get("anatomy_preservation", 0)),
        float(scores.get("degradation_match", 0)),
        float(scores.get("global_consistency", 0)),
        float(scores.get("medical_plausibility", 0)),
    ]
    return sum(values) / len(values)


def normalize_review(review: dict[str, Any]) -> dict[str, Any]:
    review.setdefault("pass", False)
    review.setdefault("summary", "")
    review.setdefault("major_issues", [])
    review.setdefault("minor_issues", [])
    review.setdefault("prompt_adjustments", [])
    scores = review.setdefault("scores", {})
    scores.setdefault("anatomy_preservation", 0)
    scores.setdefault("degradation_match", 0)
    scores.setdefault("global_consistency", 0)
    scores.setdefault("medical_plausibility", 0)
    review["average_score"] = average_score(review)
    return review


def guess_mime_type(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "image/png"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_parts(response: Any) -> tuple[bytes | None, str]:
    image_bytes: bytes | None = None
    text_chunks: list[str] = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        if content is None:
            continue
        for part in getattr(content, "parts", []) or []:
            if getattr(part, "text", None):
                text_chunks.append(part.text)
            inline = getattr(part, "inline_data", None)
            if inline is not None and getattr(inline, "data", None) and image_bytes is None:
                image_bytes = inline.data
    return image_bytes, "\n".join(chunk.strip() for chunk in text_chunks if chunk and chunk.strip()).strip()


@dataclass
class JobSpec:
    modality: str
    method: str
    prompt: str
    reference_path: Path
    gt_path: Path
    output_dir: Path

    @property
    def job_id(self) -> str:
        return f"{self.modality}__{self.method}"


class NanoBananaWorkflow:
    def __init__(
        self,
        *,
        api_key: str,
        image_model: str,
        review_model: str,
        pass_threshold: float,
        candidates_per_round: int,
        max_rounds: int,
        parallel_jobs: int,
        image_temperature: float,
        review_temperature: float,
    ) -> None:
        self.client = genai.Client(api_key=api_key)
        self.image_model = image_model
        self.review_model = review_model
        self.pass_threshold = pass_threshold
        self.candidates_per_round = candidates_per_round
        self.max_rounds = max_rounds
        self.parallel_jobs = asyncio.Semaphore(parallel_jobs)
        self.image_temperature = image_temperature
        self.review_temperature = review_temperature

    async def run(self, jobs: list[JobSpec]) -> None:
        await asyncio.gather(*(self._run_job(job) for job in jobs))

    async def _run_job(self, job: JobSpec) -> None:
        async with self.parallel_jobs:
            job.output_dir.mkdir(parents=True, exist_ok=True)
            status_path = job.output_dir / "status.json"
            current_prompt = job.prompt
            best_review: dict[str, Any] | None = None
            best_score = -1.0
            best_image_path = ""

            for round_idx in range(1, self.max_rounds + 1):
                round_dir = job.output_dir / f"round_{round_idx:02d}"
                round_dir.mkdir(parents=True, exist_ok=True)
                (round_dir / "prompt.txt").write_text(current_prompt, encoding="utf-8")

                status = {
                    "job_id": job.job_id,
                    "modality": job.modality,
                    "method": job.method,
                    "state": "running",
                    "current_round": round_idx,
                    "best_score_so_far": best_score,
                    "best_image_path": best_image_path,
                }
                write_json(status_path, status)

                candidate_tasks = [
                    self._generate_and_review_candidate(job, current_prompt, round_dir, candidate_idx)
                    for candidate_idx in range(1, self.candidates_per_round + 1)
                ]
                results = await asyncio.gather(*candidate_tasks, return_exceptions=True)
                errors: list[str] = []

                round_best_score = -1.0
                round_best_result: dict[str, Any] | None = None
                for result in results:
                    if isinstance(result, Exception):
                        errors.append(str(result))
                        continue
                    score = average_score(result["review"])
                    if score > round_best_score:
                        round_best_score = score
                        round_best_result = result

                if errors:
                    write_json(round_dir / "errors.json", {"errors": errors})

                if round_best_result is None:
                    status.update({"state": "failed", "error": "all candidates failed"})
                    write_json(status_path, status)
                    return

                best_review = round_best_result["review"]
                best_score = round_best_score
                best_image_path = round_best_result["image_path"]

                round_summary = {
                    "round": round_idx,
                    "best_candidate": round_best_result["candidate_id"],
                    "best_score": best_score,
                    "best_image_path": best_image_path,
                    "review": best_review,
                }
                write_json(round_dir / "round_summary.json", round_summary)

                passed = bool(best_review.get("pass")) and best_score >= self.pass_threshold
                status.update(
                    {
                        "state": "passed" if passed else "needs_revision",
                        "current_round": round_idx,
                        "best_score_so_far": best_score,
                        "best_image_path": best_image_path,
                        "review": best_review,
                    }
                )
                write_json(status_path, status)

                if passed:
                    break

                if round_idx < self.max_rounds:
                    current_prompt = await self._rewrite_prompt(job, current_prompt, best_review)
                    next_prompt_path = job.output_dir / f"round_{round_idx + 1:02d}_prompt_draft.txt"
                    next_prompt_path.write_text(current_prompt, encoding="utf-8")

            passed = bool(best_review and best_review.get("pass")) and best_score >= self.pass_threshold
            final_status = {
                "job_id": job.job_id,
                "modality": job.modality,
                "method": job.method,
                "state": "passed" if passed else "needs_manual_review",
                "best_score": best_score,
                "best_image_path": best_image_path,
                "review": best_review,
            }
            write_json(status_path, final_status)

    async def _generate_and_review_candidate(
        self,
        job: JobSpec,
        prompt: str,
        round_dir: Path,
        candidate_idx: int,
    ) -> dict[str, Any]:
        candidate_id = f"candidate_{candidate_idx:02d}"
        candidate_dir = round_dir / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        try:
            generated_image, response_text = await self._generate_image(job, prompt)
            if generated_image is None:
                raise RuntimeError(f"{job.job_id} {candidate_id}: no image returned")

            image_path = candidate_dir / "generated.png"
            image_path.write_bytes(generated_image)
            if response_text:
                (candidate_dir / "response.txt").write_text(response_text, encoding="utf-8")
            (candidate_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

            review = await self._review_candidate(job, prompt, image_path)
            write_json(candidate_dir / "review.json", review)
            return {
                "candidate_id": candidate_id,
                "image_path": str(image_path.resolve()),
                "review": review,
            }
        except Exception as exc:
            (candidate_dir / "error.txt").write_text(str(exc), encoding="utf-8")
            raise

    async def _generate_image(self, job: JobSpec, prompt: str) -> tuple[bytes | None, str]:
        reference_bytes = job.reference_path.read_bytes()
        gt_bytes = job.gt_path.read_bytes()

        response = await self.client.aio.models.generate_content(
            model=self.image_model,
            contents=[
                prompt,
                types.Part.from_bytes(data=reference_bytes, mime_type=guess_mime_type(job.reference_path)),
                types.Part.from_bytes(data=gt_bytes, mime_type=guess_mime_type(job.gt_path)),
            ],
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                temperature=self.image_temperature,
            ),
        )
        return extract_parts(response)

    async def _review_candidate(self, job: JobSpec, prompt: str, image_path: Path) -> dict[str, Any]:
        response = await self.client.aio.models.generate_content(
            model=self.review_model,
            contents=[
                REVIEW_PROMPT.format(modality=job.modality, method=job.method, prompt=prompt),
                types.Part.from_bytes(data=job.reference_path.read_bytes(), mime_type=guess_mime_type(job.reference_path)),
                types.Part.from_bytes(data=job.gt_path.read_bytes(), mime_type=guess_mime_type(job.gt_path)),
                types.Part.from_bytes(data=image_path.read_bytes(), mime_type=guess_mime_type(image_path)),
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=REVIEW_SCHEMA,
                temperature=self.review_temperature,
                max_output_tokens=2048,
            ),
        )
        _, text = extract_parts(response)
        if not text:
            text = getattr(response, "text", "") or ""
        return normalize_review(parse_json_maybe(text))

    async def _rewrite_prompt(self, job: JobSpec, prompt: str, review: dict[str, Any]) -> str:
        response = await self.client.aio.models.generate_content(
            model=self.review_model,
            contents=[
                REWRITE_PROMPT.format(
                    modality=job.modality,
                    method=job.method,
                    prompt=prompt,
                    review_json=json.dumps(review, ensure_ascii=False, indent=2),
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=2048,
            ),
        )
        _, text = extract_parts(response)
        if not text:
            text = getattr(response, "text", "") or ""
        return strip_code_fences(text)


def load_prompt_table(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_filter(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    return {item.strip() for item in raw.split(",") if item.strip()}


def build_jobs(
    *,
    reference_dir: Path,
    gt_dir: Path,
    prompt_table: dict[str, Any],
    output_dir: Path,
    modality_filter: set[str] | None,
    method_filter: set[str] | None,
) -> list[JobSpec]:
    jobs: list[JobSpec] = []
    prompts = prompt_table["prompts"]
    methods = prompt_table["methods"]
    modalities = prompt_table["modalities"]

    for modality in modalities:
        if modality_filter and modality not in modality_filter:
            continue
        gt_path = gt_dir / f"{slugify(modality)}.png"
        if not gt_path.exists():
            raise FileNotFoundError(f"Missing GT image for {modality}: {gt_path}")

        for method in methods:
            if method_filter and method not in method_filter:
                continue
            reference_path = reference_dir / f"{slugify(modality)}__{slugify(method)}.png"
            if not reference_path.exists():
                raise FileNotFoundError(f"Missing reference cell for {modality}/{method}: {reference_path}")

            jobs.append(
                JobSpec(
                    modality=modality,
                    method=method,
                    prompt=prompts[modality][method],
                    reference_path=reference_path,
                    gt_path=gt_path,
                    output_dir=output_dir / slugify(modality) / slugify(method),
                )
            )
    return jobs


async def async_main(args: argparse.Namespace) -> None:
    api_key = args.api_key or os.environ.get(args.api_key_env, "")
    if not api_key:
        raise RuntimeError(f"Missing API key. Pass --api-key or set {args.api_key_env}.")

    prompt_table = load_prompt_table(Path(args.prompt_table))
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(
        reference_dir=Path(args.reference_dir),
        gt_dir=Path(args.gt_dir),
        prompt_table=prompt_table,
        output_dir=output_dir,
        modality_filter=parse_filter(args.modalities),
        method_filter=parse_filter(args.methods),
    )

    write_json(
        output_dir / "run_manifest.json",
        {
            "created_at": datetime.now().isoformat(),
            "image_model": args.image_model,
            "review_model": args.review_model,
            "jobs": [job.job_id for job in jobs],
        },
    )

    workflow = NanoBananaWorkflow(
        api_key=api_key,
        image_model=args.image_model,
        review_model=args.review_model,
        pass_threshold=args.pass_threshold,
        candidates_per_round=args.candidates_per_round,
        max_rounds=args.max_rounds,
        parallel_jobs=args.parallel_jobs,
        image_temperature=args.image_temperature,
        review_temperature=args.review_temperature,
    )
    await workflow.run(jobs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", required=True, help="Directory containing split reference cells")
    parser.add_argument("--gt-dir", required=True, help="Directory containing split GT slices")
    parser.add_argument("--prompt-table", default="configs/nanobanana_prompts.json")
    parser.add_argument("--output-dir", default="results/nanobanana_runs")
    parser.add_argument("--run-name", help="Optional run folder name")
    parser.add_argument("--api-key", help="Optional API key; otherwise read from --api-key-env")
    parser.add_argument("--api-key-env", default="GOOGLE_API_KEY")
    parser.add_argument("--image-model", default="gemini-2.5-flash-image")
    parser.add_argument("--review-model", default="gemini-2.5-flash")
    parser.add_argument("--modalities", help="Comma-separated subset, e.g. FLAIR,T1-w")
    parser.add_argument("--methods", help="Comma-separated subset, e.g. MM-GAN,HiNet")
    parser.add_argument("--parallel-jobs", type=int, default=2)
    parser.add_argument("--candidates-per-round", type=int, default=2)
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument("--pass-threshold", type=float, default=7.5)
    parser.add_argument("--image-temperature", type=float, default=0.4)
    parser.add_argument("--review-temperature", type=float, default=0.1)
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
