import asyncio

import pytest

from core.fallback import (
    fallback_chain,
    global_fallback,
    safe_verdict_or_fallback,
)
from core.models import ErrorCode, FallbackBlueprint, SafetyVerdict


def test_safe_verdict_passes_through():
    v = SafetyVerdict(pass_=True, blocked=False)
    result = safe_verdict_or_fallback(v)
    assert isinstance(result, SafetyVerdict)
    assert result.pass_ is True


def test_safe_verdict_converts_to_fallback():
    v = SafetyVerdict(
        pass_=False,
        blocked=True,
        reasons=["bad"],
        reason_codes=[ErrorCode.INPUT_BLOCKED],
    )
    result = safe_verdict_or_fallback(v)
    assert isinstance(result, FallbackBlueprint)
    assert result.reason_code == ErrorCode.INPUT_BLOCKED


def test_global_fallback_includes_exception():
    try:
        raise ValueError("boom")
    except Exception as exc:
        fb = global_fallback(exc, reason_code=ErrorCode.MODULE_EXCEPTION)
    assert fb.reason_code == ErrorCode.MODULE_EXCEPTION
    assert fb.data["exc_type"] == "ValueError"
    assert "boom" in fb.data["exc"]


@pytest.mark.asyncio
async def test_fallback_chain_catches_exception():
    @fallback_chain(reason_code=ErrorCode.MODULE_EXCEPTION)
    async def fail():
        raise RuntimeError("oops")

    result = await fail()
    assert isinstance(result, FallbackBlueprint)
    assert result.reason_code == ErrorCode.MODULE_EXCEPTION
    assert result.data["exc_type"] == "RuntimeError"


@pytest.mark.asyncio
async def test_fallback_chain_catches_timeout():
    @fallback_chain(timeout=0.01, reason_code=ErrorCode.LLM_TIMEOUT)
    async def slow():
        await asyncio.sleep(10)

    result = await slow()
    assert isinstance(result, FallbackBlueprint)
    assert result.reason_code == ErrorCode.LLM_TIMEOUT


@pytest.mark.asyncio
async def test_fallback_chain_allows_success():
    @fallback_chain()
    async def ok():
        return 42

    result = await ok()
    assert result == 42
