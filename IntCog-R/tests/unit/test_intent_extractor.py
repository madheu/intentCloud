import pytest

from core.intent_extractor import IntentExtractor
from core.models import ErrorCode, FallbackBlueprint
from tests.conftest import MockLLMClient, make_extraction_response


@pytest.mark.asyncio
async def test_extract_normal_input():
    client = MockLLMClient(
        response=make_extraction_response(
            core_task="编写快速排序代码并解释时间复杂度",
            deep_goal="学习或验证算法知识",
            concepts=["Python", "快速排序", "时间复杂度"],
            trust_score=0.9,
        )
    )
    extractor = IntentExtractor(client)
    blueprint = await extractor.extract("帮我写一段 Python 快速排序代码，并解释时间复杂度。")
    assert blueprint.core_task == "编写快速排序代码并解释时间复杂度"
    assert blueprint.deep_goal == "学习或验证算法知识"
    assert blueprint.trust_score == 0.9
    assert "Python" in blueprint.concepts


@pytest.mark.asyncio
async def test_extract_deep_goal_unknown():
    """深层动机无法推断时，deep_goal 为信息不足。"""
    client = MockLLMClient(
        response=make_extraction_response(
            core_task="询问某物制作方法",
            deep_goal="信息不足",
            concepts=["某物"],
            trust_score=0.5,
        )
    )
    extractor = IntentExtractor(client)
    blueprint = await extractor.extract("那个东西怎么做？")
    assert blueprint.deep_goal == "信息不足"
    assert blueprint.trust_score == 0.5


@pytest.mark.asyncio
async def test_extract_malformed_json_fallback():
    client = MockLLMClient(response="not valid json")
    extractor = IntentExtractor(client)
    blueprint = await extractor.extract("hello")
    assert blueprint.trust_score < 0.5
    assert any("JSON_PARSE_FAIL" in c for c in blueprint.constraints)


@pytest.mark.asyncio
async def test_extract_llm_timeout_fallback():
    from tests.conftest import FailingLLMClient

    extractor = IntentExtractor(FailingLLMClient())
    result = await extractor.extract("hello")
    assert isinstance(result, FallbackBlueprint)
    assert result.reason_code == ErrorCode.LLM_TIMEOUT


@pytest.mark.asyncio
async def test_extract_with_contradiction():
    """矛盾输入降低 trust_score。"""
    client = MockLLMClient(
        response=make_extraction_response(
            core_task="获取详细说明",
            deep_goal="信息不足",
            constraints=["只用一句话", "矛盾：详细说明与一句话冲突"],
            trust_score=0.35,
        )
    )
    extractor = IntentExtractor(client)
    blueprint = await extractor.extract("请详细说明，但只用一句话。")
    assert blueprint.trust_score <= 0.4
    assert any("矛盾" in c for c in blueprint.constraints)