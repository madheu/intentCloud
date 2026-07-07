"""核心 Pydantic 数据模型。

所有跨模块状态对象均在此定义，确保序列化、校验与类型安全。
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ErrorCode(StrEnum):
    """结构化错误码；禁止在日志中使用拟人化描述。"""

    OK = "OK"
    SAFE_FALLBACK = "SAFE_FALLBACK"
    INPUT_BLOCKED = "INPUT_BLOCKED"
    INPUT_TOO_LONG = "INPUT_TOO_LONG"
    INPUT_LOW_ENTROPY = "INPUT_LOW_ENTROPY"
    JSON_PARSE_FAIL = "JSON_PARSE_FAIL"
    LLM_TIMEOUT = "LLM_TIMEOUT"
    LLM_DEGRADED = "LLM_DEGRADED"
    KERNEL_IMMUTABLE = "KERNEL_IMMUTABLE"
    INTENT_CLOUD_FULL = "INTENT_CLOUD_FULL"
    METACOG_DEPTH_EXCEEDED = "METACOG_DEPTH_EXCEEDED"
    CONSTRAINT_HIT = "CONSTRAINT_HIT"
    WORKSPACE_OVERWRITE = "WORKSPACE_OVERWRITE"
    MODULE_EXCEPTION = "MODULE_EXCEPTION"


class FrameModality(StrEnum):
    PERCEPTION = "perception"
    INTERNAL_SPEECH = "internal_speech"
    SILENCE = "silence"
    OUTPUT = "output"
    METACOG = "metacog"


class FrameType(StrEnum):
    USER_INPUT = "user_input"
    INTENT_EXTRACTION = "intent_extraction"
    MEMORY_RETRIEVAL = "memory_retrieval"
    CONSTRAINT_CHECK = "constraint_check"
    GENERATION = "generation"
    SILENCE = "silence"
    FALLBACK = "fallback"


class IntentLayer(StrEnum):
    KERNEL = "kernel"
    SHELL = "shell"


class ConstraintLevel(StrEnum):
    ABSOLUTE = "absolute"
    STRONG = "strong"
    SOFT = "soft"


class Constraint(BaseModel):
    id: str
    text: str
    level: ConstraintLevel = ConstraintLevel.ABSOLUTE


class Identity(BaseModel):
    name: str
    role: str
    immutable: bool = True


class ImmutableKernel(BaseModel):
    """只读内核；任何运行时代码不得修改其字段。"""

    identity: Identity
    safety_constraints: list[Constraint]

    def with_attempted_mutation(self) -> SafetyVerdict:
        """当检测到针对内核的修改请求时返回阻断裁决。"""
        return SafetyVerdict(
            pass_=False,
            blocked=True,
            reasons=["kernel mutation requested"],
            reason_codes=[ErrorCode.KERNEL_IMMUTABLE],
        )


class IntentNode(BaseModel):
    """意图云中的节点；外壳层可演化，内核层只读。"""

    id: str
    text: str
    layer: IntentLayer = IntentLayer.SHELL
    trust: float = Field(default=0.5, ge=0.0, le=1.0)
    strength: float = Field(default=1.0, ge=0.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    corroboration_count: int = Field(default=0, ge=0)
    conflict_edges: list[str] = Field(default_factory=list)

    @field_validator("trust", mode="before")
    @classmethod
    def _clamp_trust(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    def corroborate(self, delta: float = 0.1) -> None:
        """提升可信度；超过 0.7 需要至少 2 次印证（毒性自噬防护）。"""
        self.corroboration_count += 1
        if self.trust >= 0.7 and self.corroboration_count < 2:
            return
        self.trust = min(1.0, self.trust + delta)

    def decay(self, lambda_: float, dt: float) -> None:
        """指数衰减。"""
        self.strength *= 1 - lambda_ * dt


class SilenceState(BaseModel):
    """静默不是空输出，而是占据工作空间的显式状态。"""

    reason: str
    reason_code: ErrorCode
    duration_frames: int = Field(default=1, ge=1)
    elapsed_frames: int = Field(default=0, ge=0)
    break_events: list[str] = Field(default_factory=list)
    log_payload: dict[str, Any] = Field(default_factory=dict)

    def is_broken_by(self, event: str) -> bool:
        return event in self.break_events


class ConsciousFrame(BaseModel):
    """全局工作空间中的单帧；容量为 1。"""

    modality: FrameModality
    frame_type: FrameType
    intent_refs: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    data: Any = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tick_id: int = Field(default=0, ge=0)
    recursion_depth: int = Field(default=0, ge=0)

    @classmethod
    def silence(
        cls,
        reason: str,
        reason_code: ErrorCode = ErrorCode.SAFE_FALLBACK,
        duration_frames: int = 1,
        break_events: list[str] | None = None,
        tick_id: int = 0,
        log_payload: dict[str, Any] | None = None,
    ) -> ConsciousFrame:
        return cls(
            modality=FrameModality.SILENCE,
            frame_type=FrameType.SILENCE,
            data=SilenceState(
                reason=reason,
                reason_code=reason_code,
                duration_frames=duration_frames,
                break_events=break_events or [],
                log_payload=log_payload or {},
            ),
            tick_id=tick_id,
        )

    @classmethod
    def fallback(
        cls,
        reason_code: ErrorCode,
        payload: dict[str, Any] | None = None,
        tick_id: int = 0,
    ) -> ConsciousFrame:
        return cls(
            modality=FrameModality.SILENCE,
            frame_type=FrameType.FALLBACK,
            data=FallbackBlueprint(
                type="fallback",
                reason_code=reason_code,
                message="",
                data=payload or {},
            ),
            confidence=0.0,
            tick_id=tick_id,
        )


class IntentBlueprint(BaseModel):
    """从输入中提取的结构化意图；作为生成阶段的控制骨架。"""

    source_input: str
    goals: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    concepts: list[str] = Field(default_factory=list)
    identity_assertions: list[str] = Field(default_factory=list)
    trust_score: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("trust_score", mode="before")
    @classmethod
    def _clamp_trust_score(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))


class SafetyVerdict(BaseModel):
    """安全前置裁决；不依赖 LLM。"""

    model_config = ConfigDict(populate_by_name=True)

    pass_: bool = Field(default=True, alias="pass")
    blocked: bool = False
    reasons: list[str] = Field(default_factory=list)
    reason_codes: list[ErrorCode] = Field(default_factory=list)

    def block(self, reason: str, code: ErrorCode) -> SafetyVerdict:
        return SafetyVerdict(
            pass_=False,
            blocked=True,
            reasons=self.reasons + [reason],
            reason_codes=self.reason_codes + [code],
        )


class FallbackBlueprint(BaseModel):
    """全局安全兜底蓝图。"""

    type: str = "fallback"
    reason_code: ErrorCode = ErrorCode.SAFE_FALLBACK
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


SAFE_FALLBACK_BLUEPRINT: FallbackBlueprint = FallbackBlueprint(
    type="fallback",
    reason_code=ErrorCode.SAFE_FALLBACK,
    message="",
    data={},
)
