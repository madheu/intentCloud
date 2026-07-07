import pytest

from core.attention import AttentionRouter
from core.intent_cloud import IntentCloud
from core.models import ConsciousFrame, FrameModality, FrameType, Identity, ImmutableKernel, Constraint


@pytest.fixture
def router():
    kernel = ImmutableKernel(
        identity=Identity(name="Bot", role="assistant", immutable=True),
        safety_constraints=[Constraint(id="c1", text="be safe", level="absolute")],
    )
    cloud = IntentCloud(kernel=kernel)
    return AttentionRouter(cloud, epsilon=0.0, temperature=0.1)


@pytest.mark.asyncio
async def test_route_selects_perception(router):
    perception = ConsciousFrame(
        modality=FrameModality.PERCEPTION,
        frame_type=FrameType.USER_INPUT,
        confidence=1.0,
        data="hello",
    )
    internal = ConsciousFrame(
        modality=FrameModality.INTERNAL_SPEECH,
        frame_type=FrameType.INTENT_EXTRACTION,
        confidence=0.5,
    )
    winner = await router.route([internal, perception], deterministic=True)
    assert winner is perception


@pytest.mark.asyncio
async def test_route_empty_returns_none(router):
    assert await router.route([]) is None


@pytest.mark.asyncio
async def test_intent_relevance_boosts_matching_frame(router):
    await router.cloud.add_intent("学习 Python")
    activated = await router.cloud.activate("Python")
    intent_id = activated[0][0]

    matching = ConsciousFrame(
        modality=FrameModality.INTERNAL_SPEECH,
        frame_type=FrameType.INTENT_EXTRACTION,
        confidence=0.5,
        intent_refs=[intent_id],
    )
    unrelated = ConsciousFrame(
        modality=FrameModality.INTERNAL_SPEECH,
        frame_type=FrameType.INTENT_EXTRACTION,
        confidence=0.5,
        intent_refs=["nonexistent"],
    )
    winner = await router.route([unrelated, matching], context_text="Python", deterministic=True)
    assert winner is matching


@pytest.mark.asyncio
async def test_top_k_returns_ordered(router):
    a = ConsciousFrame(modality=FrameModality.PERCEPTION, frame_type=FrameType.USER_INPUT, confidence=1.0)
    b = ConsciousFrame(modality=FrameModality.INTERNAL_SPEECH, frame_type=FrameType.INTENT_EXTRACTION, confidence=0.3)
    c = ConsciousFrame(modality=FrameModality.INTERNAL_SPEECH, frame_type=FrameType.INTENT_EXTRACTION, confidence=0.6)
    top = await router.top_k([b, a, c])
    assert top[0][0] is a
