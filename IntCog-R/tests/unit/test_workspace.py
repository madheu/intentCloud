import pytest

from core.models import ConsciousFrame, FrameModality, FrameType
from core.workspace import GlobalWorkspace


@pytest.fixture
def workspace():
    return GlobalWorkspace()


@pytest.mark.asyncio
async def test_publish_first_frame(workspace):
    frame = ConsciousFrame(modality=FrameModality.PERCEPTION, frame_type=FrameType.USER_INPUT, data="hello")
    result = await workspace.publish(frame)
    assert result is frame
    assert workspace.current() is frame


@pytest.mark.asyncio
async def test_perception_overwrites_lower_confidence(workspace):
    low = ConsciousFrame(
        modality=FrameModality.INTERNAL_SPEECH,
        frame_type=FrameType.INTENT_EXTRACTION,
        confidence=0.5,
    )
    await workspace.publish(low)
    high = ConsciousFrame(
        modality=FrameModality.PERCEPTION,
        frame_type=FrameType.USER_INPUT,
        confidence=0.9,
        data="user",
    )
    result = await workspace.publish(high)
    assert result is high
    assert workspace.current() is high


@pytest.mark.asyncio
async def test_lower_confidence_dropped(workspace):
    high = ConsciousFrame(
        modality=FrameModality.INTERNAL_SPEECH,
        frame_type=FrameType.INTENT_EXTRACTION,
        confidence=0.8,
    )
    await workspace.publish(high)
    low = ConsciousFrame(
        modality=FrameModality.INTERNAL_SPEECH,
        frame_type=FrameType.INTENT_EXTRACTION,
        confidence=0.3,
    )
    result = await workspace.publish(low)
    assert result is high


@pytest.mark.asyncio
async def test_silence_blocks_overwrite_until_expired(workspace):
    silence = ConsciousFrame.silence(
        reason="test",
        duration_frames=2,
    )
    await workspace.publish(silence)

    other = ConsciousFrame(
        modality=FrameModality.INTERNAL_SPEECH,
        frame_type=FrameType.INTENT_EXTRACTION,
        confidence=0.9,
    )
    result = await workspace.publish(other)
    assert result.modality == FrameModality.SILENCE

    await workspace.next_tick()
    await workspace.next_tick()
    result2 = await workspace.publish(other)
    assert result2 is other


@pytest.mark.asyncio
async def test_broadcast_calls_listeners(workspace):
    received = []

    async def listener(frame):
        received.append(frame)

    workspace.subscribe(listener)
    frame = ConsciousFrame(modality=FrameModality.PERCEPTION, frame_type=FrameType.USER_INPUT)
    await workspace.publish(frame)
    await workspace.broadcast()
    assert len(received) == 1


@pytest.mark.asyncio
async def test_tick_increments_tick_id(workspace):
    await workspace.next_tick()
    await workspace.next_tick()
    assert workspace.tick_id == 2
