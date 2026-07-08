import pytest

from core.metacognition import MetacognitionMonitor
from core.models import ConsciousFrame, ErrorCode, FrameModality, FrameType, IntentBlueprint


@pytest.fixture
def monitor():
    return MetacognitionMonitor(max_recursion_depth=2, min_confidence_threshold=0.2)


def test_recursion_depth_exceeded(monitor):
    frame = ConsciousFrame(
        modality=FrameModality.INTERNAL_SPEECH,
        frame_type=FrameType.INTENT_EXTRACTION,
        recursion_depth=3,
    )
    blueprint = IntentBlueprint(source_input="x", trust_score=0.9)
    result = monitor.check(frame, blueprint)
    assert result.modality == FrameModality.SILENCE
    assert result.data.reason_code == ErrorCode.METACOG_DEPTH_EXCEEDED


def test_low_confidence_triggers_silence(monitor):
    frame = ConsciousFrame(
        modality=FrameModality.PERCEPTION,
        frame_type=FrameType.USER_INPUT,
        recursion_depth=0,
    )
    blueprint = IntentBlueprint(source_input="x", trust_score=0.1)
    result = monitor.check(frame, blueprint)
    assert result.modality == FrameModality.SILENCE
    assert result.data.reason_code == ErrorCode.METACOG_DEPTH_EXCEEDED


def test_normal_returns_metacog_frame(monitor):
    frame = ConsciousFrame(
        modality=FrameModality.PERCEPTION,
        frame_type=FrameType.USER_INPUT,
        recursion_depth=0,
    )
    blueprint = IntentBlueprint(
        source_input="x",
        trust_score=0.9,
        goals=["编写代码"],
        concepts=["Python"],
    )
    result = monitor.check(frame, blueprint)
    assert result.modality == FrameModality.METACOG
    assert result.frame_type == FrameType.CONSTRAINT_CHECK


def test_no_content_triggers_silence(monitor):
    """三段式元认知：无可用内容时返回静默。"""
    frame = ConsciousFrame(
        modality=FrameModality.PERCEPTION,
        frame_type=FrameType.USER_INPUT,
        recursion_depth=0,
    )
    blueprint = IntentBlueprint(
        source_input="x",
        trust_score=0.9,
        goals=[],
        constraints=[],
        concepts=[],
    )
    result = monitor.check(frame, blueprint)
    assert result.modality == FrameModality.SILENCE


def test_contradiction_triggers_retry(monitor):
    """三段式元认知：检测到矛盾关系时返回重试。"""
    frame = ConsciousFrame(
        modality=FrameModality.PERCEPTION,
        frame_type=FrameType.USER_INPUT,
        recursion_depth=0,
    )
    blueprint = IntentBlueprint(
        source_input="x",
        trust_score=0.9,
        goals=["获取详细说明"],
        constraints=["限制只用一句话"],
        relations=[{"type": "contradiction", "from": "详细说明", "to": "只用一句话"}],
    )
    result = monitor.check(frame, blueprint)
    assert result.modality == FrameModality.METACOG
    assert result.data.get("action") == "retry_extract"
