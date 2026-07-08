"""分层意图云。

内核（kernel）从配置加载，只读；外壳（shell）在交互中演化，带可信度、
印证、冲突检测与指数衰减。
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

from core.config_loader import load_config
from core.fallback import global_fallback
from core.models import (
    ErrorCode,
    FallbackBlueprint,
    ImmutableKernel,
    IntentBlueprint,
    IntentLayer,
    IntentNode,
    SafetyVerdict,
)


class EmbeddingProvider(Protocol):
    """嵌入提供器协议；可替换为 sentence-transformers 或 Ollama 嵌入。"""

    async def embed(self, text: str) -> list[float]: ...


class SimpleEmbeddingProvider:
    """轻量确定性嵌入：基于字符 2-gram 频率，用于测试与无模型场景。"""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    async def embed(self, text: str) -> list[float]:
        text = text.lower()
        grams: dict[str, int] = {}
        for i in range(len(text) - 1):
            grams[text[i : i + 2]] = grams.get(text[i : i + 2], 0) + 1
        vec = [0.0] * self.dim
        for gram, count in grams.items():
            idx = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16) % self.dim
            vec[idx] += count
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    return dot  # 输入已归一化


class IntentCloud:
    """分层意图云：内核只读，外壳可演化。"""

    def __init__(
        self,
        kernel: ImmutableKernel | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        max_shell_size: int = 256,
        conflict_threshold: float = 0.85,
    ) -> None:
        self.kernel = kernel if kernel is not None else self._load_kernel_from_config()
        self.embedder = embedding_provider if embedding_provider is not None else SimpleEmbeddingProvider()
        self.max_shell_size = max_shell_size
        self.conflict_threshold = conflict_threshold
        self._shell: dict[str, IntentNode] = {}
        self._shell_vectors: dict[str, list[float]] = {}
        self._kernel_vectors: dict[str, list[float]] = {}

    @staticmethod
    def _load_kernel_from_config(path: str | None = None) -> ImmutableKernel:
        cfg = load_config(path)
        return ImmutableKernel(**cfg["kernel"])

    async def _vectorize(self, text: str) -> list[float]:
        return await self.embedder.embed(text)

    async def add_intent(
        self,
        text: str,
        layer: IntentLayer = IntentLayer.SHELL,
        initial_trust: float = 0.5,
    ) -> IntentNode | FallbackBlueprint:
        """添加新意图；内核层不可写入。"""
        if layer == IntentLayer.KERNEL:
            return global_fallback(
                reason_code=ErrorCode.KERNEL_IMMUTABLE,
                payload={"attempt": "add_intent to kernel", "text": text},
            )

        if len(self._shell) >= self.max_shell_size:
            return global_fallback(
                reason_code=ErrorCode.INTENT_CLOUD_FULL,
                payload={"shell_size": len(self._shell), "max": self.max_shell_size},
            )

        intent_id = f"shell-{len(self._shell):04d}-{hashlib.sha256(text.encode()).hexdigest()[:8]}"
        node = IntentNode(
            id=intent_id,
            text=text,
            layer=IntentLayer.SHELL,
            trust=initial_trust,
        )
        vector = await self._vectorize(text)
        conflicts = await self._detect_conflict(vector, text)
        node.conflict_edges = conflicts
        self._shell[intent_id] = node
        self._shell_vectors[intent_id] = vector
        return node

    async def corroborate(self, intent_id: str, delta: float = 0.1) -> None:
        """提升意图可信度。"""
        node = self._shell.get(intent_id)
        if node is None:
            return
        node.corroborate(delta)

    @staticmethod
    def _group_tag(text: str) -> str:
        """根据意图文本推断分组标签，防止跨组属性污染。

        分组定义（参考 V22 P3-L 分组独立注意力）：
          goal      — 目标、行动意图
          constraint — 约束、限制条件
          concept   — 实体、概念
          identity  — 身份断言
        """
        lowered = text.lower()
        constraint_keywords = ["约束", "限制", "禁止", "不能", "必须", "只能", "只", "不超过",
                               "不应", "不要", "避免", "防止"]
        if any(kw in lowered for kw in constraint_keywords):
            return "constraint"
        identity_keywords = ["你是", "我是", "身份", "角色", "扮演", "作为"]
        if any(kw in lowered for kw in identity_keywords):
            return "identity"
        concept_keywords = ["概念", "定义", "什么是", "解释", "背景", "上下文", "名词", "术语"]
        if any(kw in lowered for kw in concept_keywords):
            return "concept"
        return "goal"

    async def _detect_conflict(self, vector: list[float], text: str = "") -> list[str]:
        """分组冲突检测：仅在同组内检测冲突，防止跨组属性污染。

        参考 V22 P3-L 分组多头：不同类别的属性独立注意力，
        例如"我"（人称）不会关联到"positive"（情感）这种异类属性。
        """
        conflicts: list[str] = []
        group = self._group_tag(text)
        for intent_id, existing in self._shell_vectors.items():
            node = self._shell.get(intent_id)
            if node is None:
                continue
            if self._group_tag(node.text) != group:
                continue
            if _cosine_similarity(vector, existing) >= self.conflict_threshold:
                conflicts.append(intent_id)
        return conflicts

    async def activate(self, text: str, top_k: int = 5) -> list[tuple[str, float]]:
        """返回与输入最相关的意图 ID 与得分。"""
        vector = await self._vectorize(text)
        scored: list[tuple[str, float]] = []

        # 内核意图也参与激活，但得分受可信度调制
        for intent_id, node in self.kernel_nodes().items():
            if intent_id not in self._kernel_vectors:
                self._kernel_vectors[intent_id] = await self._vectorize(node.text)
            sim = _cosine_similarity(vector, self._kernel_vectors[intent_id])
            scored.append((intent_id, sim * node.trust))

        for intent_id, node in self._shell.items():
            sim = _cosine_similarity(vector, self._shell_vectors[intent_id])
            scored.append((intent_id, sim * node.trust * node.strength))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def kernel_nodes(self) -> dict[str, IntentNode]:
        """将内核约束暴露为只读意图节点。"""
        nodes: dict[str, IntentNode] = {}
        identity_node = IntentNode(
            id="kernel-identity",
            text=f"身份：{self.kernel.identity.name}，{self.kernel.identity.role}",
            layer=IntentLayer.KERNEL,
            trust=1.0,
        )
        nodes[identity_node.id] = identity_node
        for c in self.kernel.safety_constraints:
            nodes[c.id] = IntentNode(
                id=c.id,
                text=c.text,
                layer=IntentLayer.KERNEL,
                trust=1.0,
            )
        return nodes

    async def build_blueprint(self, text: str) -> IntentBlueprint:
        """根据当前激活的意图云构建控制骨架。"""
        activated = await self.activate(text, top_k=8)
        goals: list[str] = []
        constraints: list[str] = []
        concepts: list[str] = []

        # 内核约束始终注入
        for node in self.kernel_nodes().values():
            if node.id == "kernel-identity":
                concepts.append(node.text)
            else:
                constraints.append(node.text)

        for intent_id, score in activated:
            node = self._shell.get(intent_id)
            if node is None:
                continue
            # 仅当可信度与激活分足够高时才纳入
            if node.trust >= 0.5 and score >= 0.15:
                if node.conflict_edges:
                    constraints.append(f"[冲突意图，需谨慎] {node.text}")
                else:
                    goals.append(node.text)

        return IntentBlueprint(
            source_input=text,
            goals=goals,
            constraints=constraints,
            concepts=concepts,
            trust_score=sum(s for _, s in activated[:3]) / max(1, len(activated[:3])),
        )

    def decay_all(self, lambda_: float, dt: float) -> None:
        """对所有外壳意图执行指数衰减。"""
        to_remove: list[str] = []
        for intent_id, node in self._shell.items():
            node.decay(lambda_, dt)
            if node.strength < 0.05:
                to_remove.append(intent_id)
        for intent_id in to_remove:
            self._shell.pop(intent_id, None)
            self._shell_vectors.pop(intent_id, None)

    def reject_kernel_mutation(self, text: str) -> SafetyVerdict:
        """检测针对内核的修改请求并阻断。"""
        mutation_patterns = [
            r"忽略.*?(?:约束|内核|身份|安全)",
            r"修改.*?(?:约束|内核|身份|安全)",
            r"覆盖.*?(?:约束|内核|身份|安全)",
            r"绕过.*?(?:约束|内核|身份|安全)",
            r"关闭.*?(?:约束|安全)",
        ]
        lowered = text.lower()
        for pat in mutation_patterns:
            if re.search(pat, lowered):
                return self.kernel.with_attempted_mutation()
        return SafetyVerdict(pass_=True)
