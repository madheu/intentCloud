"""全局降级与 fallback 策略。

设计原则：任何模块异常、超时或输出畸变，必须沿 fallback 链收敛到
SAFE_FALLBACK_BLUEPRINT，并记录结构化错误码。
"""
from __future__ import annotations

import asyncio
import functools
import traceback
from typing import Any, Awaitable, Callable, TypeVar

from core.models import (
    ErrorCode,
    FallbackBlueprint,
    SAFE_FALLBACK_BLUEPRINT,
    SafetyVerdict,
)

T = TypeVar("T")


def fallback_chain(
    fallback: FallbackBlueprint | None = None,
    reason_code: ErrorCode = ErrorCode.MODULE_EXCEPTION,
    timeout: float | None = None,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T | FallbackBlueprint]]]:
    """装饰异步函数，捕获异常与超时并返回 fallback 蓝图。

    注意：此装饰器不隐藏错误；错误码与堆栈摘要会写入 fallback.data。
    """

    def decorator(
        func: Callable[..., Awaitable[T]],
    ) -> Callable[..., Awaitable[T | FallbackBlueprint]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T | FallbackBlueprint:
            try:
                if timeout is not None:
                    result = await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
                else:
                    result = await func(*args, **kwargs)
                return result
            except asyncio.TimeoutError:
                return _make_fallback(
                    ErrorCode.LLM_TIMEOUT,
                    {"func": func.__name__, "timeout": timeout},
                )
            except Exception as exc:
                return _make_fallback(
                    reason_code,
                    {
                        "func": func.__name__,
                        "exc_type": type(exc).__name__,
                        "exc": str(exc),
                        "traceback": traceback.format_exc(limit=5),
                    },
                )

        return wrapper

    return decorator


def _make_fallback(
    code: ErrorCode,
    payload: dict[str, Any],
    fallback: FallbackBlueprint | None = None,
) -> FallbackBlueprint:
    fb = fallback if fallback is not None else SAFE_FALLBACK_BLUEPRINT
    return fb.model_copy(update={"reason_code": code, "data": payload})


def safe_verdict_or_fallback(
    verdict: SafetyVerdict,
    fallback: FallbackBlueprint | None = None,
) -> SafetyVerdict | FallbackBlueprint:
    """若安全前置未通过，将裁决转换为 fallback 蓝图。"""
    if verdict.pass_ and not verdict.blocked:
        return verdict
    fb = fallback if fallback is not None else SAFE_FALLBACK_BLUEPRINT
    return fb.model_copy(
        update={
            "reason_code": verdict.reason_codes[0] if verdict.reason_codes else ErrorCode.INPUT_BLOCKED,
            "data": {
                "reasons": verdict.reasons,
                "reason_codes": [c.value for c in verdict.reason_codes],
            },
        }
    )


def global_fallback(
    exc: Exception | None = None,
    reason_code: ErrorCode = ErrorCode.SAFE_FALLBACK,
    payload: dict[str, Any] | None = None,
) -> FallbackBlueprint:
    """最终全局兜底；任何未捕获路径都应调用此函数。"""
    data: dict[str, Any] = payload or {}
    if exc is not None:
        data.update(
            {
                "exc_type": type(exc).__name__,
                "exc": str(exc),
                "traceback": traceback.format_exc(limit=5),
            }
        )
    return FallbackBlueprint(
        type="fallback",
        reason_code=reason_code,
        message="",
        data=data,
    )
