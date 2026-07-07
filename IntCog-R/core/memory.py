"""情节记忆接口。

提供 Chroma 抽象与内存回退实现。检索时结合意图云当前目标调整相似度权重，
并支持时间 / 情感衰减。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

from core.intent_cloud import EmbeddingProvider, SimpleEmbeddingProvider, _cosine_similarity


@dataclass
class MemoryEntry:
    text: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    emotional_valence: float = 0.0  # -1 到 1
    metadata: dict[str, Any] = field(default_factory=dict)
    vector: list[float] | None = None


class MemoryProvider(Protocol):
    """记忆提供器协议。"""

    async def store(self, entry: MemoryEntry) -> None: ...

    async def retrieve(
        self,
        query: str,
        intent_goals: list[str],
        top_k: int = 5,
    ) -> list[MemoryEntry]: ...


class InMemoryMemory:
    """内存中的情节记忆，用于测试与无 Chroma 场景。"""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider | None = None,
        time_decay_lambda: float = 0.01,
        emotional_boost: float = 0.1,
    ) -> None:
        self.embedder = embedding_provider if embedding_provider is not None else SimpleEmbeddingProvider()
        self.entries: list[MemoryEntry] = []
        self.time_decay_lambda = time_decay_lambda
        self.emotional_boost = emotional_boost

    async def store(self, entry: MemoryEntry) -> None:
        if entry.vector is None:
            entry.vector = await self.embedder.embed(entry.text)
        self.entries.append(entry)

    async def retrieve(
        self,
        query: str,
        intent_goals: list[str],
        top_k: int = 5,
    ) -> list[MemoryEntry]:
        query_vec = await self.embedder.embed(query)
        goal_vec = await self.embedder.embed(" ".join(intent_goals))

        now = datetime.now(timezone.utc)
        scored: list[tuple[MemoryEntry, float]] = []
        for entry in self.entries:
            if entry.vector is None:
                entry.vector = await self.embedder.embed(entry.text)
            sim_query = _cosine_similarity(query_vec, entry.vector)
            sim_goal = _cosine_similarity(goal_vec, entry.vector)

            dt_seconds = (now - entry.timestamp).total_seconds()
            time_decay = math.exp(-self.time_decay_lambda * dt_seconds)
            emotional_weight = 1.0 + abs(entry.emotional_valence) * self.emotional_boost

            # 综合分：查询相似度 + 目标相关度，经时间与情感调制
            score = (sim_query + sim_goal) * time_decay * emotional_weight
            scored.append((entry, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [entry for entry, _ in scored[:top_k]]


class ChromaMemory:
    """Chroma 记忆占位实现；阶段 2/3 可替换为真实 Chroma 客户端。"""

    def __init__(self, collection_name: str = "episodes") -> None:
        self.collection_name = collection_name
        self._fallback = InMemoryMemory()

    async def store(self, entry: MemoryEntry) -> None:
        await self._fallback.store(entry)

    async def retrieve(
        self,
        query: str,
        intent_goals: list[str],
        top_k: int = 5,
    ) -> list[MemoryEntry]:
        return await self._fallback.retrieve(query, intent_goals, top_k)
