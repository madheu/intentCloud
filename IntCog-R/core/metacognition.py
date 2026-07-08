"""元认知监控器：三段式元认知（监控 → 评估 → 规划）。

参考 MetaRAG 三段式元认知：
  1. 监控(Monitoring)   — 判断是否需要启动自省
  2. 评估(Evaluating)   — 定位可能出错的环节
  3. 规划(Planning)     — 根据评估结果，决定下一步动作
"""
from __future__ import annotations

from enum import StrEnum

from core.models import ConsciousFrame, ErrorCode, FrameModality, FrameType, IntentBlueprint


class MetaAction(StrEnum):
    """元认知决策动作。"""
    RETRY_EXTRACT = "retry_extract"
    RETRIEVE_MEMORY = "retrieve_memory"
    GENERATE = "generate"
    SILENCE = "silence"


class MetaDiagnosis:
    """元认知诊断结果。"""

    def __init__(self) -> None:
        self.action: MetaAction = MetaAction.GENERATE
        self.issues: list[str] = []
        self.confidence: float = 0.5


class MetacognitionMonitor:
    """三段式元认知监控：监控 → 评估 → 规划。"""

    def __init__(
        self,
        max_recursion_depth: int = 2,
        min_confidence_threshold: float = 0.2,
        min_goals_threshold: int = 0,
    ) -> None:
        self.max_recursion_depth = max_recursion_depth
        self.min_confidence_threshold = min_confidence_threshold
        self.min_goals_threshold = min_goals_threshold

    def check(
        self,
        frame: ConsciousFrame,
        blueprint: IntentBlueprint,
    ) -> ConsciousFrame | None:
        """三段式元认知：监控 → 评估 → 规划。"""
        diagnosis = self._monitor_and_evaluate(frame, blueprint)
        return self._plan(frame, blueprint, diagnosis)

    def _monitor_and_evaluate(
        self,
        frame: ConsciousFrame,
        blueprint: IntentBlueprint,
    ) -> MetaDiagnosis:
        """阶段 1+2：监控信号并评估问题来源。"""
        diagnosis = MetaDiagnosis()

        # 监控：递归深度
        if frame.recursion_depth > self.max_recursion_depth:
            diagnosis.issues.append("recursion depth exceeded")
            diagnosis.action = MetaAction.SILENCE
            diagnosis.confidence = 0.1
            return diagnosis

        # 监控：蓝图置信度
        if blueprint.trust_score < self.min_confidence_threshold:
            diagnosis.issues.append("blueprint confidence too low")
            diagnosis.action = MetaAction.SILENCE
            diagnosis.confidence = 0.1
            return diagnosis

        # 评估：是否有目标/约束/概念可供生成
        has_content = (
            len(blueprint.goals) > self.min_goals_threshold
            or len(blueprint.concepts) > 0
            or len(blueprint.constraints) > 0
        )
        if not has_content:
            diagnosis.issues.append("no actionable content in blueprint")
            diagnosis.action = MetaAction.SILENCE
            diagnosis.confidence = 0.15
            return diagnosis

        # 评估：关系自洽性（若存在 relations 字段）
        if hasattr(blueprint, "relations") and blueprint.relations:
            consistent = all(
                rel.get("type", "") != "contradiction" for rel in blueprint.relations
            )
            if not consistent:
                diagnosis.issues.append("contradictory relations detected")
                diagnosis.action = MetaAction.RETRY_EXTRACT
                diagnosis.confidence = 0.25
                return diagnosis

        diagnosis.action = MetaAction.GENERATE
        diagnosis.confidence = blueprint.trust_score
        return diagnosis

    def _plan(
        self,
        frame: ConsciousFrame,
        blueprint: IntentBlueprint,
        diagnosis: MetaDiagnosis,
    ) -> ConsciousFrame | None:
        """阶段 3：根据诊断结果规划下一步动作。"""
        if diagnosis.action == MetaAction.SILENCE:
            return ConsciousFrame.silence(
                reason="; ".join(diagnosis.issues),
                reason_code=ErrorCode.METACOG_DEPTH_EXCEEDED,
                duration_frames=3,
                tick_id=frame.tick_id,
                log_payload={
                    "issues": diagnosis.issues,
                    "recursion_depth": frame.recursion_depth,
                    "trust_score": blueprint.trust_score,
                },
            )

        if diagnosis.action == MetaAction.RETRY_EXTRACT:
            return ConsciousFrame(
                modality=FrameModality.METACOG,
                frame_type=FrameType.CONSTRAINT_CHECK,
                data={
                    "action": "retry_extract",
                    "issues": diagnosis.issues,
                    "confidence": diagnosis.confidence,
                },
                confidence=0.3,
                intent_refs=frame.intent_refs,
                recursion_depth=frame.recursion_depth,
            )

        # 正常：生成元认知确认帧
        return ConsciousFrame(
            modality=FrameModality.METACOG,
            frame_type=FrameType.CONSTRAINT_CHECK,
            data={
                "action": "generate",
                "checked": True,
                "recursion_depth": frame.recursion_depth,
                "trust_score": blueprint.trust_score,
                "issues": diagnosis.issues,
            },
            confidence=0.5,
            intent_refs=frame.intent_refs,
            recursion_depth=frame.recursion_depth,
        )