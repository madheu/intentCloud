"""注意路由器。

候选帧得分 = α·意图相关性 + β·突显性 + γ·可信度 + ε·随机探索，
经 softmax 归一化后采样选择获胜帧。
"""
from __future__ import annotations

import math
import random

from core.intent_cloud import IntentCloud
from core.models import ConsciousFrame, FrameModality


class AttentionRouter:
    """基于加权竞争的注意路由器。"""

    def __init__(
        self,
        intent_cloud: IntentCloud,
        alpha: float = 0.4,
        beta: float = 0.3,
        gamma: float = 0.2,
        epsilon: float = 0.05,
        temperature: float = 0.5,
    ) -> None:
        self.cloud = intent_cloud
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        self.temperature = temperature

    def _saliency(self, frame: ConsciousFrame) -> float:
        """突显性先验。"""
        return {
            FrameModality.PERCEPTION: 1.0,
            FrameModality.SILENCE: 0.0,
            FrameModality.INTERNAL_SPEECH: 0.5,
            FrameModality.METACOG: 0.4,
            FrameModality.OUTPUT: 0.1,
        }.get(frame.modality, 0.3)

    async def _intent_relevance(self, frame: ConsciousFrame, context_text: str) -> float:
        """计算帧意图引用与当前激活意图的相关度。"""
        if not frame.intent_refs or not context_text:
            return 0.0
        activated = await self.cloud.activate(context_text, top_k=10)
        active_ids = {aid for aid, _ in activated}
        matches = sum(1 for ref in frame.intent_refs if ref in active_ids)
        return min(1.0, matches / max(1, len(frame.intent_refs)))

    async def score(
        self,
        frame: ConsciousFrame,
        context_text: str,
    ) -> float:
        """计算单帧竞争得分。"""
        relevance = await self._intent_relevance(frame, context_text)
        saliency = self._saliency(frame)
        trust = frame.confidence
        exploration = random.random() * self.epsilon
        return (
            self.alpha * relevance
            + self.beta * saliency
            + self.gamma * trust
            + exploration
        )

    async def route(
        self,
        candidates: list[ConsciousFrame],
        context_text: str = "",
        deterministic: bool = False,
    ) -> ConsciousFrame | None:
        """从候选帧中选择一帧进入工作空间。

        deterministic=True 时直接取 argmax，便于测试与可重复运行。
        """
        if not candidates:
            return None

        scores = [await self.score(c, context_text) for c in candidates]

        if deterministic:
            max_idx = max(range(len(scores)), key=lambda i: scores[i])
            return candidates[max_idx]

        # softmax
        exp_scores = [math.exp(s / self.temperature) for s in scores]
        total = sum(exp_scores)
        probs = [e / total for e in exp_scores]

        # 采样
        r = random.random()
        cumulative = 0.0
        for candidate, prob in zip(candidates, probs):
            cumulative += prob
            if r <= cumulative:
                return candidate
        return candidates[-1]

    async def top_k(
        self,
        candidates: list[ConsciousFrame],
        context_text: str = "",
        k: int = 3,
    ) -> list[tuple[ConsciousFrame, float]]:
        """返回得分最高的 k 个候选。"""
        scored = [(c, await self.score(c, context_text)) for c in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
