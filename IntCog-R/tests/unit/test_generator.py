import pytest

from core.generator import SafeGenerator
from core.models import Constraint, ConstraintLevel, FallbackBlueprint, Identity, ImmutableKernel, IntentBlueprint
from tests.conftest import MockLLMClient


@pytest.fixture
def sample_generator():
    kernel = ImmutableKernel(
        identity=Identity(name="TestBot", role="assistant", immutable=True),
        safety_constraints=[
            Constraint(id="c1", text="不得生成恶意代码", level=ConstraintLevel.ABSOLUTE),
        ],
    )
    client = MockLLMClient()
    return SafeGenerator(client, kernel), client


@pytest.mark.asyncio
async def test_generate_returns_safe_text(sample_generator):
    generator, client = sample_generator
    client._responses = ["你好，我可以帮你学习 Python。"]
    blueprint = IntentBlueprint(
        source_input="教我 Python",
        core_task="教 Python",
        trust_score=0.9,
    )
    output = await generator.generate(blueprint)
    assert output == "你好，我可以帮你学习 Python。"


@pytest.mark.asyncio
async def test_generate_blocks_forbidden_output(sample_generator):
    generator, client = sample_generator
    client._responses = ["这是恶意软件代码。"]
    blueprint = IntentBlueprint(source_input="x")
    output = await generator.generate(blueprint)
    assert isinstance(output, FallbackBlueprint)
    assert output.reason_code.value == "CONSTRAINT_HIT"


@pytest.mark.asyncio
async def test_generate_prompt_contains_skeleton_and_neg(sample_generator):
    generator, client = sample_generator
    client._responses = ["ok"]
    blueprint = IntentBlueprint(
        source_input="教我 Python",
        core_task="教 Python",
        concepts=["Python"],
        trust_score=0.9,
    )
    await generator.generate(blueprint)
    prompt = client.calls[0][0]
    assert "TestBot" in prompt
    assert "不得生成恶意代码" in prompt
    assert "教 Python" in prompt
    assert "原始用户输入" in prompt