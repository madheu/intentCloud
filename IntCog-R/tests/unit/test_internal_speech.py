import pytest

from core.internal_speech import InternalSpeechPlanner
from core.memory import MemoryEntry
from core.models import ConsciousFrame, FrameModality, FrameType, IntentBlueprint


@pytest.fixture
def planner():
    return InternalSpeechPlanner(use_llm=False)


@pytest.fixture
def perception_frame():
    return ConsciousFrame(
        modality=FrameModality.PERCEPTION,
        frame_type=FrameType.USER_INPUT,
        data={"text": "hello"},
        confidence=1.0,
    )


@pytest.mark.asyncio
async def test_plan_generates_steps(planner, perception_frame):
    blueprint = IntentBlueprint(source_input="hello", goals=["greet"])
    frames = await planner.plan(perception_frame, blueprint, [])
    assert len(frames) >= 2
    assert all(f.modality == FrameModality.INTERNAL_SPEECH for f in frames)
    assert all(f.recursion_depth == perception_frame.recursion_depth + 1 for f in frames)


@pytest.mark.asyncio
async def test_plan_includes_memory_step(planner, perception_frame):
    blueprint = IntentBlueprint(source_input="hello")
    memories = [MemoryEntry(text="previous greeting")]
    frames = await planner.plan(perception_frame, blueprint, memories)
    assert any(f.frame_type == FrameType.MEMORY_RETRIEVAL for f in frames)
