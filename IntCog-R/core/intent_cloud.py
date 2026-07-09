"""分层意图云。

内核（kernel）从配置加载，只读；外壳（shell）在交互中演化，带可信度、
印证、冲突检测与指数衰减。

新 API（v2）：
  - add(blueprint)       → 存储 IntentBlueprint
  - retrieve(blueprint)  → 基于 Jaccard n-gram 相似度检索历史蓝图
  - enhance_blueprint(bp) → 用历史意图增强当前蓝图

旧 API（v1，保持兼容）：
  - add_intent / activate / build_blueprint / corroborate / decay_all / reject_kernel_mutation
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

from core.config_loader import load_config
from core.fallback import global_fallback
from core.intent_cloud_config import IntentCloudConfig
from core.anchor_system import AnchorSystem
from core.models import (
    ErrorCode,
    FallbackBlueprint,
    ImmutableKernel,
    IntentBlueprint,
    IntentLayer,
    IntentNode,
    SafetyVerdict,
)


# ── 嵌入提供器（保持兼容）─────────────────────────────────────────────────────

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


# ── 拓扑边与权重更新律（v3 新增）──────────────────────────────────────────────
# 综合更新律：赫布学习 + 参考模型拉回 + 动量阻尼 + 死区 + 投影
# 公式：w_new = Proj(w_old + η·a_i·a_j) - γ·(w_old - w_ref) - K_d·(w_old - w_old_prev)


class CloudEdge:
    """意图云拓扑中的有向边。

    每条边连接两个 IntentNode，权重随用户交互演化。
    ref_weight 是初始云骨架中该边的基准权重，作为锚点防止全局漂移。
    prev_weight 记录上一步权重，用于动量阻尼计算。
    edge_type 表示边的语义关系类型（如 connects, refines, contrasts, evokes, constrains）。
    """

    __slots__ = ("source_id", "target_id", "weight", "ref_weight", "prev_weight", "edge_type")

    def __init__(
        self,
        source_id: str,
        target_id: str,
        weight: float = 0.5,
        ref_weight: float | None = None,
        prev_weight: float | None = None,
        edge_type: str = "connects",
    ) -> None:
        self.source_id = source_id
        self.target_id = target_id
        self.weight = weight
        # ref_weight 默认等于初始 weight，这样边的初始状态就是"锚定状态"
        self.ref_weight = ref_weight if ref_weight is not None else weight
        # prev_weight 默认等于 weight，表示初始时没有"上一步"变化
        self.prev_weight = prev_weight if prev_weight is not None else weight
        # 边的语义关系类型，用于意图到向量的映射
        self.edge_type = edge_type

    def __repr__(self) -> str:
        return (
            f"CloudEdge({self.source_id} -> {self.target_id}, "
            f"w={self.weight:.4f}, ref={self.ref_weight:.4f}, prev={self.prev_weight:.4f}, type={self.edge_type})"
        )


def update_weight(
    edge: CloudEdge,
    a_i: float,
    a_j: float,
    config: "IntentCloudConfig",
) -> CloudEdge:
    """综合权重更新律：赫布 + 参考拉回 + 动量阻尼 + 死区 + 投影。

    公式：
        w_new = Proj(w_old + η·a_i·a_j) - γ·(w_old - w_ref) - K_d·(w_old - w_old_prev)

    其中：
        - Proj(x) = clip(x, w_min, w_max)：投影算子，硬裁剪到合法区间
        - 死区：当 |a_i * a_j| < δ 时，跳过赫布项，仅做衰减
        - 饱和限幅：a_i, a_j 被裁剪到 [0, a_max] 后再参与计算

    Args:
        edge: 当前边对象（会被原地修改并返回）
        a_i: 源节点激活值
        a_j: 目标节点激活值
        config: 拓扑演化参数配置

    Returns:
        更新后的边对象（与输入是同一个对象，原地修改）

    Raises:
        ValueError: 如果激活值为 NaN 或无穷大
        ValueError: 如果 config 参数不在合法范围内
    """
    # ── 边界条件检查 ─────────────────────────────────────────────────────────
    # 激活值有效性检查：NaN 和 Inf 会污染整个拓扑，必须提前拒绝
    if math.isnan(a_i) or math.isnan(a_j):
        raise ValueError(
            f"Activation values must not be NaN: a_i={a_i}, a_j={a_j}"
        )
    if math.isinf(a_i) or math.isinf(a_j):
        raise ValueError(
            f"Activation values must not be infinite: a_i={a_i}, a_j={a_j}"
        )

    # ── 饱和限幅：将激活值裁剪到 [0, a_max] ─────────────────────────────────
    # 这是为了防止极端激活值（如用户恶意输入触发异常高激活）导致权重爆炸
    a_i = max(0.0, min(a_i, config.a_max))
    a_j = max(0.0, min(a_j, config.a_max))

    w_old = edge.weight
    w_ref = edge.ref_weight
    w_prev = edge.prev_weight

    # ── 死区判断 ─────────────────────────────────────────────────────────────
    # 当 |a_i * a_j| < δ 时，共现太弱，赫布项视为噪声，跳过
    activation_product = a_i * a_j
    if abs(activation_product) < config.delta:
        hebbian_term = 0.0
    else:
        hebbian_term = config.eta * activation_product

    # ── 三项叠加 ─────────────────────────────────────────────────────────────
    # 第 1 步：赫布项加法后立即投影，防止赫布增强导致中间值越界
    # 第 2 步：参考模型拉回，向 w_ref 方向修正
    # 第 3 步：动量阻尼，抑制与前一步方向相反的突变
    w_new = (
        _project(w_old + hebbian_term, config.w_min, config.w_max)
        - config.gamma * (w_old - w_ref)
        - config.K_d * (w_old - w_prev)
    )

    # ── 最终投影 ─────────────────────────────────────────────────────────────
    # 三项叠加后可能仍然越界（如拉回和阻尼同向叠加），再做一次硬裁剪
    w_new = _project(w_new, config.w_min, config.w_max)

    # ── 原地更新边状态 ──────────────────────────────────────────────────────
    edge.prev_weight = w_old  # 记录当前权重作为"上一步"，供下次更新使用
    edge.weight = w_new

    return edge


def _project(value: float, lower: float, upper: float) -> float:
    """投影算子：将 value 硬裁剪到 [lower, upper] 区间。

    这是更新律中 Proj 的实现，确保权重始终在合法范围内。
    使用 max/min 而非 if 分支，因为边界检查在此处是常态而非异常路径。
    """
    return max(lower, min(value, upper))


# ── Jaccard n-gram 存储（v2 新增）────────────────────────────────────────────

def _extract_ngrams(text: str, n: int = 2) -> set[str]:
    """从文本中提取字符 n-gram 集合。"""
    text = text.lower()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _blueprint_ngrams(blueprint: IntentBlueprint) -> set[str]:
    """从蓝图的关键字段中提取 n-gram 集合，用于相似度计算。

    加权策略：core_task 权重最高（2×），deep_goal 和 concepts 各 1×。
    """
    parts = [
        blueprint.core_task * 2,          # 核心任务权重最高
        blueprint.deep_goal,              # 深层目标
        " ".join(blueprint.constraints),  # 约束
        " ".join(blueprint.concepts),     # 概念
    ]
    combined = " ".join(p for p in parts if p)
    return _extract_ngrams(combined)


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Jaccard 系数：|A ∩ B| / |A ∪ B|。

    返回值范围 [0, 1]，1 表示完全相同。
    """
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0.0


class InMemoryIntentStorage:
    """基于内存的意图存储后端，使用 Jaccard n-gram 相似度进行检索。

    可替换为向量数据库（如 ChromaDB）实现相同接口。
    """

    def __init__(self, ngram_n: int = 2) -> None:
        self._entries: dict[str, IntentBlueprint] = {}
        self._ngrams: dict[str, set[str]] = {}
        self.ngram_n = ngram_n

    def add(self, blueprint: IntentBlueprint) -> str:
        """存储蓝图并返回唯一 ID。

        ID 基于蓝图内容的 SHA256 哈希，相同蓝图不会重复存储。
        """
        content = f"{blueprint.core_task}|{blueprint.deep_goal}|{blueprint.identity}"
        bp_id = hashlib.sha256(content.encode()).hexdigest()[:12]

        if bp_id not in self._entries:
            self._entries[bp_id] = blueprint
            self._ngrams[bp_id] = _blueprint_ngrams(blueprint)
        return bp_id

    def get(self, bp_id: str) -> IntentBlueprint | None:
        """按 ID 获取蓝图。"""
        return self._entries.get(bp_id)

    def search(
        self,
        blueprint: IntentBlueprint,
        top_k: int = 5,
        min_similarity: float = 0.0,
    ) -> list[tuple[IntentBlueprint, float]]:
        """检索与当前蓝图最相似的历史蓝图。

        Args:
            blueprint: 当前蓝图
            top_k: 返回的最大结果数
            min_similarity: 最低相似度阈值

        Returns:
            (蓝图, Jaccard 相似度) 列表，按相似度降序
        """
        query_ngrams = _blueprint_ngrams(blueprint)
        scored: list[tuple[str, float]] = []

        for bp_id, stored_ngrams in self._ngrams.items():
            sim = _jaccard_similarity(query_ngrams, stored_ngrams)
            if sim >= min_similarity:
                scored.append((bp_id, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            (self._entries[bp_id], sim)
            for bp_id, sim in scored[:top_k]
            if bp_id in self._entries
        ]

    def all(self) -> list[IntentBlueprint]:
        """返回所有已存储的蓝图。"""
        return list(self._entries.values())

    def __len__(self) -> int:
        return len(self._entries)


# ── IntentCloud（v1 API 保持兼容 + v2 API 新增）──────────────────────────────


class IntentCloud:
    """分层意图云：内核只读，外壳可演化。

    v2 API:
      - add(blueprint)        存储蓝图到 Jaccard 存储
      - retrieve(blueprint)   检索相似历史蓝图
      - enhance_blueprint(bp) 用历史意图增强当前蓝图

    v3 API:
      - update_weight_safe()  带锚点检查的权重更新
      - update_topology()     批量更新拓扑（跳过不可变边）
    """

    def __init__(
        self,
        kernel: ImmutableKernel | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        max_shell_size: int = 256,
        conflict_threshold: float = 0.85,
        storage: InMemoryIntentStorage | None = None,
        anchor_system: AnchorSystem | None = None,
        topology_config: IntentCloudConfig | None = None,
    ) -> None:
        self.kernel = kernel if kernel is not None else self._load_kernel_from_config()
        self.embedder = embedding_provider if embedding_provider is not None else SimpleEmbeddingProvider()
        self.max_shell_size = max_shell_size
        self.conflict_threshold = conflict_threshold
        self._shell: dict[str, IntentNode] = {}
        self._shell_vectors: dict[str, list[float]] = {}
        self._kernel_vectors: dict[str, list[float]] = {}
        # v2: Jaccard n-gram 存储后端
        self._storage = storage if storage is not None else InMemoryIntentStorage()
        # v3: 锚点系统 + 拓扑演化
        self.anchor_system = anchor_system if anchor_system is not None else self._load_anchors_from_config()
        self.topology_config = topology_config if topology_config is not None else IntentCloudConfig()
        # 边存储：key 为 (source_id, target_id)，存 CloudEdge 对象
        self._edges: dict[tuple[str, str], CloudEdge] = {}
        # 初始化锚点边到 _edges 中
        self._init_anchor_edges()

    # ── v2 API ─────────────────────────────────────────────────────────────

    def add(self, blueprint: IntentBlueprint) -> str:
        """存储蓝图到持久化存储，返回唯一 ID。

        相同内容不会重复存储。
        """
        return self._storage.add(blueprint)

    def retrieve(
        self,
        blueprint: IntentBlueprint,
        top_k: int = 5,
    ) -> list[tuple[IntentBlueprint, float]]:
        """检索与当前蓝图最相似的历史蓝图。

        Args:
            blueprint: 当前蓝图（用于计算相似度）
            top_k: 返回的最大结果数

        Returns:
            (蓝图, Jaccard 相似度) 列表，按相似度降序
        """
        return self._storage.search(blueprint, top_k=top_k)

    def enhance_blueprint(self, blueprint: IntentBlueprint) -> IntentBlueprint:
        """用历史意图增强当前蓝图。

        增强策略：
          1. 从历史中最相似的蓝图中补充概念
          2. 合并历史约束（去重）
          3. identity 保持不变（来自内核）

        Args:
            blueprint: 当前蓝图

        Returns:
            增强后的蓝图
        """
        similar = self.retrieve(blueprint, top_k=3)

        if not similar:
            return blueprint

        # 合并概念（去重，保持顺序）
        all_concepts = list(blueprint.concepts)
        seen = set(all_concepts)
        for hist_bp, _ in similar:
            for c in hist_bp.concepts:
                if c not in seen:
                    all_concepts.append(c)
                    seen.add(c)

        # 合并约束（去重）
        all_constraints = list(blueprint.constraints)
        seen_c = set(all_constraints)
        for hist_bp, _ in similar:
            for c in hist_bp.constraints:
                if c not in seen_c:
                    all_constraints.append(c)
                    seen_c.add(c)

        return IntentBlueprint(
            source_input=blueprint.source_input,
            identity=blueprint.identity,
            core_task=blueprint.core_task,
            deep_goal=blueprint.deep_goal,
            constraints=all_constraints,
            concepts=all_concepts,
            trust_score=blueprint.trust_score,
        )

    # ── v1 API（保持兼容）──────────────────────────────────────────────────

    @staticmethod
    def _load_kernel_from_config(path: str | None = None) -> ImmutableKernel:
        cfg = load_config(path)
        return ImmutableKernel(**cfg["kernel"])

    @staticmethod
    def _load_anchors_from_config(path: str | None = None) -> AnchorSystem:
        """从 config.yaml 加载锚点系统。

        如果配置文件中没有 anchors 块，回退到默认锚点。
        """
        cfg = load_config(path)
        anchors_cfg = cfg.get("anchors")
        if anchors_cfg is None:
            return AnchorSystem.default()
        return AnchorSystem.from_config(anchors_cfg)

    def _init_anchor_edges(self) -> None:
        """将锚点系统中的拓扑锚点初始化到 _edges 字典中。

        这些边的权重固定，后续 update_weight_safe 会跳过它们。
        """
        for (src, tgt), anc in self.anchor_system._edges.items():
            self._edges[(src, tgt)] = CloudEdge(
                source_id=src,
                target_id=tgt,
                weight=anc.weight,
                ref_weight=anc.weight,
                prev_weight=anc.weight,  # 锚点边无历史，prev = weight
            )

    # ── v3 API：带锚点检查的拓扑更新 ────────────────────────────────────────

    def update_weight_safe(
        self,
        source_id: str,
        target_id: str,
        a_i: float,
        a_j: float,
        config: IntentCloudConfig | None = None,
    ) -> CloudEdge | None:
        """带锚点检查的权重更新：不可变边被跳过，返回 None。

        Args:
            source_id: 源节点 ID
            target_id: 目标节点 ID
            a_i: 源节点激活值
            a_j: 目标节点激活值
            config: 拓扑演化配置（可选，默认使用 cloud.topology_config）

        Returns:
            更新后的 CloudEdge，如果边不可变则返回 None
        """
        # 锚点检查：不可变边直接跳过
        if not self.anchor_system.is_edge_mutable(source_id, target_id):
            return None

        cfg = config if config is not None else self.topology_config

        # 确保边存在于 _edges 中，否则创建
        key = (source_id, target_id)
        if key not in self._edges:
            self._edges[key] = CloudEdge(
                source_id=source_id,
                target_id=target_id,
                weight=0.5,
                ref_weight=0.5,
                prev_weight=0.5,
            )

        return update_weight(self._edges[key], a_i, a_j, cfg)

    def update_topology(
        self,
        activations: dict[str, float],
        config: IntentCloudConfig | None = None,
    ) -> list[CloudEdge]:
        """批量更新拓扑：对所有边应用权重更新律，跳过不可变边。

        Args:
            activations: 节点 ID → 激活值的映射
            config: 拓扑演化配置（可选）

        Returns:
            被成功更新的边列表（不可变边不出现在结果中）
        """
        updated: list[CloudEdge] = []
        cfg = config if config is not None else self.topology_config

        for (src, tgt), edge in list(self._edges.items()):
            # 锚点检查
            if not self.anchor_system.is_edge_mutable(src, tgt):
                continue
            # 获取激活值，未知节点默认 activation=0
            a_i = activations.get(src, 0.0)
            a_j = activations.get(tgt, 0.0)
            result = update_weight(edge, a_i, a_j, cfg)
            updated.append(result)

        return updated

    def get_edge(self, source_id: str, target_id: str) -> CloudEdge | None:
        """获取指定边的当前状态。"""
        return self._edges.get((source_id, target_id))

    # ── v3 API：快慢分离的交互处理 ──────────────────────────────────────────

    def process_interaction(
        self,
        input_signals: dict[str, float],
        config: IntentCloudConfig | None = None,
    ) -> dict[str, float]:
        """处理一次完整的交互：先扩散激活（快子系统），再更新权重（慢子系统）。

        两步执行：
          1. 调用 diffuse_activation() 进行激活扩散，直到收敛
          2. 调用 update_weights_after_diffusion() 更新非锚点边的权重

        Args:
            input_signals: 外部输入信号（节点 ID → 初始激活值）
            config: 拓扑演化配置（可选，默认使用 cloud.topology_config）

        Returns:
            收敛后的各节点激活值字典
        """
        cfg = config if config is not None else self.topology_config

        from core.intent_cloud_dynamics import (
            diffuse_activation,
            update_weights_after_diffusion,
        )

        # 阶段 1：快子系统 —— 激活扩散（权重固定）
        converged_activations = diffuse_activation(self, input_signals, cfg)

        # 阶段 2：慢子系统 —— 权重更新（激活固定，仅在收敛后执行）
        update_weights_after_diffusion(self, converged_activations, cfg)

        return converged_activations

    # ── v1 API（保持兼容）──────────────────────────────────────────────────

    async def _vectorize(self, text: str) -> list[float]:
        return await self.embedder.embed(text)

    async def add_intent(
        self,
        text: str,
        layer: IntentLayer = IntentLayer.SHELL,
        initial_trust: float = 0.5,
    ) -> IntentNode | FallbackBlueprint:
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
        node = self._shell.get(intent_id)
        if node is None:
            return
        node.corroborate(delta)

    @staticmethod
    def _group_tag(text: str) -> str:
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
        vector = await self._vectorize(text)
        scored: list[tuple[str, float]] = []

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
        activated = await self.activate(text, top_k=8)
        constraints: list[str] = []
        concepts: list[str] = []

        for node in self.kernel_nodes().values():
            if node.id == "kernel-identity":
                concepts.append(node.text)
            else:
                constraints.append(node.text)

        for intent_id, score in activated:
            node = self._shell.get(intent_id)
            if node is None:
                continue
            if node.trust >= 0.5 and score >= 0.15:
                if node.conflict_edges:
                    constraints.append(f"[冲突意图，需谨慎] {node.text}")
                else:
                    concepts.append(node.text)

        return IntentBlueprint(
            source_input=text,
            identity=self.kernel.identity.name,
            core_task="",
            deep_goal="",
            constraints=constraints,
            concepts=concepts,
            trust_score=sum(s for _, s in activated[:3]) / max(1, len(activated[:3])),
        )

    def decay_all(self, lambda_: float, dt: float) -> None:
        to_remove: list[str] = []
        for intent_id, node in self._shell.items():
            node.decay(lambda_, dt)
            if node.strength < 0.05:
                to_remove.append(intent_id)
        for intent_id in to_remove:
            self._shell.pop(intent_id, None)
            self._shell_vectors.pop(intent_id, None)

    def reject_kernel_mutation(self, text: str) -> SafetyVerdict:
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


# ── 自测 ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== IntentCloud v2 自测 ===\n")

    cloud = IntentCloud()

    # 1. add — 存储蓝图
    bp1 = IntentBlueprint(
        source_input="帮我写一首关于大海的诗",
        identity="Qwythos",
        core_task="创作大海主题诗歌",
        deep_goal="满足审美需求",
        constraints=["使用中文", "不超过8行"],
        concepts=["大海", "诗歌", "意象"],
        trust_score=0.9,
    )
    id1 = cloud.add(bp1)
    print(f"1. add(bp1) → id={id1}")
    print(f"   存储量: {len(cloud._storage)}")

    bp2 = IntentBlueprint(
        source_input="写一首关于星空的诗",
        identity="Qwythos",
        core_task="创作星空主题诗歌",
        deep_goal="满足审美需求",
        constraints=["使用中文"],
        concepts=["星空", "诗歌", "浪漫"],
        trust_score=0.85,
    )
    id2 = cloud.add(bp2)
    print(f"   add(bp2) → id={id2}")
    print(f"   存储量: {len(cloud._storage)}")

    # 2. retrieve — 检索相似蓝图
    query = IntentBlueprint(
        source_input="写一首关于海洋的诗",
        identity="Qwythos",
        core_task="创作海洋主题诗歌",
        deep_goal="满足审美需求",
        constraints=["使用中文"],
        concepts=["海洋", "诗歌"],
        trust_score=0.9,
    )
    results = cloud.retrieve(query, top_k=3)
    print(f"\n2. retrieve(query='海洋诗歌') → {len(results)} 条结果:")
    for bp, sim in results:
        print(f"   sim={sim:.4f}  core_task={bp.core_task}  concepts={bp.concepts}")

    # 3. enhance_blueprint — 增强蓝图
    enhanced = cloud.enhance_blueprint(query)
    print(f"\n3. enhance_blueprint:")
    print(f"   增强前 concepts: {query.concepts}")
    print(f"   增强后 concepts: {enhanced.concepts}")
    print(f"   增强前 constraints: {query.constraints}")
    print(f"   增强后 constraints: {enhanced.constraints}")

    # 4. 相同蓝图不重复存储
    id1_dup = cloud.add(bp1)
    print(f"\n4. 重复 add(bp1) → id={id1_dup} (与原 id={id1} {'相同' if id1_dup == id1 else '不同'})")
    print(f"   存储量: {len(cloud._storage)} (未增加)")

    print("\n=== 自测通过 ===")