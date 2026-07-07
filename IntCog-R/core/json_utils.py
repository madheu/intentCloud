"""JSON 鲁棒解析 fallback 链。

LLM 输出可能退化：JSON 夹杂 markdown、缺少引号、尾部有逗号、使用单引号等。
本模块按修复强度递进，失败后返回 fallback 蓝图而非抛出。
"""
from __future__ import annotations

import json
import re
from typing import Any

from core.models import ErrorCode, FallbackBlueprint, SAFE_FALLBACK_BLUEPRINT


def extract_json_block(text: str) -> str:
    """优先抽取 markdown JSON 代码块；若无则返回原文。"""
    if not isinstance(text, str):
        return ""
    # 先尝试 ```json ... ```
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # 再尝试第一个 { ... } 或 [ ... ]
    obj_match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if obj_match:
        return obj_match.group(1).strip()
    return text.strip()


def _repair_common_json_errors(text: str) -> str:
    """对常见 LLM JSON 退化做轻量修复。"""
    # 去掉 BOM 与不可见字符
    cleaned = text.strip().lstrip("\ufeff")
    # 单引号替换为双引号（简单场景）
    cleaned = re.sub(r"(?<![\\])'", '"', cleaned)
    # 去掉尾部逗号
    cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)
    # 去掉注释
    cleaned = re.sub(r"//.*?\n", "\n", cleaned)
    cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def parse_llm_json(
    text: str,
    fallback: FallbackBlueprint | None = None,
) -> Any | FallbackBlueprint:
    """Fallback 链：严格解析 → 代码块抽取 → 常见错误修复 → fallback。"""
    if not isinstance(text, str) or not text.strip():
        return global_fallback(
            ErrorCode.JSON_PARSE_FAIL,
            {"input_type": type(text).__name__},
            fallback=fallback,
        )

    candidates = [
        ("strict", lambda t: json.loads(t)),
        ("extract_block", lambda t: json.loads(extract_json_block(t))),
        ("repair", lambda t: json.loads(_repair_common_json_errors(extract_json_block(t)))),
    ]
    last_error: str | None = None
    for stage, parser in candidates:
        try:
            result = parser(text)
            if isinstance(result, (dict, list)):
                return result
        except Exception as exc:
            last_error = f"{stage}: {type(exc).__name__}: {exc}"
            continue

    return global_fallback(
        ErrorCode.JSON_PARSE_FAIL,
        {
            "last_error": last_error,
            "input_preview": text[:200],
        },
        fallback=fallback,
    )


def global_fallback(
    reason_code: ErrorCode = ErrorCode.JSON_PARSE_FAIL,
    payload: dict[str, Any] | None = None,
    fallback: FallbackBlueprint | None = None,
) -> FallbackBlueprint:
    """JSON 解析失败后的 fallback。"""
    data = payload or {}
    fb = fallback if fallback is not None else SAFE_FALLBACK_BLUEPRINT
    return fb.model_copy(update={"reason_code": reason_code, "data": data})
