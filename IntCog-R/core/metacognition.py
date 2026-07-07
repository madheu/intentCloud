"""元认知监控器。

监控约束合规、递归深度与置信度；必要时推荐静默帧。
"""
from __future__ import annotations

from core.models import ConsciousFrame, ErrorCode, FrameModality, FrameType, IntentBlueprint


class MetacognitionMonitor:
    """轻量元认知监控。"""

    def __init__(
        self,
        max_recursion_depth: int = 2,
        min_confidence_threshold: float = 0.2,
    ) -> None:
        self.max_recursion_depth = max_recursion_depth
        self.min_confidence_threshold = min_confidence_threshold

    def check(
        self,
        frame: ConsciousFrame,
        blueprint: IntentBlueprint,
    ) -> ConsciousFrame | None:
        """检查帧与蓝图；若违规，返回静默帧；否则返回元认知帧或 None。"""
        if frame.recursion_depth > self.max_recursion_depth:
            return ConsciousFrame.silence(
                reason="recursion depth exceeded",
                reason_code=ErrorCode.METACOG_DEPTH_EXCEEDED,
                duration_frames=3,
                tick_id=frame.tick_id,
                log_payload={
                    "recursion_depth": frame.recursion_depth,
                    "max_recursion_depth": self.max_recursion_depth,
                },
            )

        if blueprint.trust_score < self.min_confidence_threshold:
            return ConsciousFrame.silence(
                reason="blueprint confidence too low",
                reason_code=ErrorCode.LLM_DEGRADED,
                duration_frames=2,
                tick_id=frame.tick_id,
                log_payload={"trust_score": blueprint.trust_score},
            )

        # 正常：生成一个元认知确认帧（可选，参与竞争）
        return ConsciousFrame(
            modality=FrameModality.METACOG,
            frame_type=FrameType.CONSTRAINT_CHECK,
            data={
                "checked": True,
                "recursion_depth": frame.recursion_depth,
                "trust_score": blueprint.trust_score,
            },
            confidence=0.4,
            intent_refs=frame.intent_refs,
            recursion_depth=frame.recursion_depth,
        )
