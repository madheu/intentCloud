"""内部言语规划器。

为当前工作空间帧生成多步思维骨架；每一步作为候选帧参与注意竞争。
提供 LLM 路径与确定性回退，便于在无模型环境下运行主循环。
"""
from __future__ import annotations

from core.intent_extractor import LLMClient
from core.memory import MemoryEntry
from core.models import ConsciousFrame, FrameModality, FrameType, IntentBlueprint


class InternalSpeechPlanner:
    """内部言语规划器：慢通道，异步生成思维候选帧。"""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        max_steps: int = 3,
        use_llm: bool = False,
    ) -> None:
        self.llm = llm_client
        self.max_steps = max_steps
        self.use_llm = use_llm

    async def plan(
        self,
        current_frame: ConsciousFrame,
        blueprint: IntentBlueprint,
        memories: list[MemoryEntry],
    ) -> list[ConsciousFrame]:
        """生成思维候选帧列表。"""
        if self.use_llm and self.llm is not None:
            return await self._plan_with_llm(current_frame, blueprint, memories)
        return self._plan_deterministic(current_frame, blueprint, memories)

    def _plan_deterministic(
        self,
        current_frame: ConsciousFrame,
        blueprint: IntentBlueprint,
        memories: list[MemoryEntry],
    ) -> list[ConsciousFrame]:
        """确定性思维链：分析 → 记忆检索 → 规划回复。"""
        candidates: list[ConsciousFrame] = []

        # 步骤 1：理解当前输入
        candidates.append(
            ConsciousFrame(
                modality=FrameModality.INTERNAL_SPEECH,
                frame_type=FrameType.INTENT_EXTRACTION,
                data={
                    "thought": f"正在理解输入：{blueprint.source_input}",
                    "core_task": blueprint.core_task,
                },
                confidence=0.6,
                intent_refs=current_frame.intent_refs,
                recursion_depth=current_frame.recursion_depth + 1,
            )
        )

        # 步骤 2：检索相关记忆
        if memories:
            memory_texts = [m.text for m in memories[:3]]
            candidates.append(
                ConsciousFrame(
                    modality=FrameModality.INTERNAL_SPEECH,
                    frame_type=FrameType.MEMORY_RETRIEVAL,
                    data={
                        "thought": "检索到相关情节记忆",
                        "memories": memory_texts,
                    },
                    confidence=0.5,
                    intent_refs=current_frame.intent_refs,
                    recursion_depth=current_frame.recursion_depth + 1,
                )
            )

        # 步骤 3：规划回复
        candidates.append(
            ConsciousFrame(
                modality=FrameModality.INTERNAL_SPEECH,
                frame_type=FrameType.GENERATION,
                data={
                    "thought": "正在规划安全回复",
                    "constraints": blueprint.constraints,
                },
                confidence=0.55,
                intent_refs=current_frame.intent_refs,
                recursion_depth=current_frame.recursion_depth + 1,
            )
        )

        return candidates[: self.max_steps]

    async def _plan_with_llm(
        self,
        current_frame: ConsciousFrame,
        blueprint: IntentBlueprint,
        memories: list[MemoryEntry],
    ) -> list[ConsciousFrame]:
        """LLM 驱动的思维规划（占位实现）。"""
        # 未来可接入 Qwythos 7B 生成多步思维 JSON
        return self._plan_deterministic(current_frame, blueprint, memories)
