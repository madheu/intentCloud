"""意图云锚点系统。

三类锚点共同构成不可变语义骨架，防止意图云在长期演化中漂移：

1. 概念锚点：固定激活值的核心节点，永远不被演化修改
2. 拓扑锚点：概念锚点之间的固定权重边，在权重更新时被跳过
3. 结构锚点：全局约束（激活限幅、权重投影、侧抑制），已在 IntentCloudConfig 中实现

配置从 config.yaml 的 anchors 块加载，格式：
  anchors:
    nodes:
      - id: "self"
        activation: 1.0
        mutable: false
    edges:
      - from: "negation"
        to: "self"
        type: "contrasts"
        weight: 0.8
        mutable: false
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── 锚点数据模型 ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AnchorNodeConfig:
    """概念锚点配置：一个不可变的意图云节点。

    Attributes:
        id: 节点唯一标识
        activation: 固定激活值，在演化中保持不变
        label: 人类可读标签，用于日志和调试
        mutable: 是否允许演化修改（锚点永远为 False）
    """

    id: str
    activation: float = 0.5
    label: str = ""
    mutable: bool = False

    def __post_init__(self) -> None:
        if self.activation < 0:
            raise ValueError(
                f"Anchor node '{self.id}' activation must be >= 0, got {self.activation}"
            )


@dataclass(frozen=True)
class AnchorEdgeConfig:
    """拓扑锚点配置：一条不可变的意图云边。

    Attributes:
        source: 源节点 ID
        target: 目标节点 ID
        edge_type: 关系类型（如 "contrasts", "connects", "supports"）
        weight: 固定权重，在权重更新时被跳过
        mutable: 是否允许演化修改（锚点永远为 False）
    """

    source: str
    target: str
    edge_type: str = "connects"
    weight: float = 0.5
    mutable: bool = False

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError(
                f"Anchor edge '{self.source} -> {self.target}' weight must be >= 0, got {self.weight}"
            )
        if self.source == self.target:
            raise ValueError(
                f"Anchor edge cannot be self-loop: '{self.source} -> {self.target}'"
            )


# ── 锚点系统 ──────────────────────────────────────────────────────────────────


class AnchorSystem:
    """锚点系统：管理概念锚点和拓扑锚点，提供不可变性检查。

    使用方式：
        anchors = AnchorSystem.from_config(config_dict)
        if anchors.is_node_mutable(node_id):
            # 可以演化
        if anchors.is_edge_mutable(source, target):
            # 可以更新权重
    """

    def __init__(
        self,
        nodes: dict[str, AnchorNodeConfig] | None = None,
        edges: dict[tuple[str, str], AnchorEdgeConfig] | None = None,
    ) -> None:
        """初始化锚点系统。

        Args:
            nodes: 概念锚点字典，key 为节点 ID
            edges: 拓扑锚点字典，key 为 (source, target) 元组
        """
        self._nodes: dict[str, AnchorNodeConfig] = nodes or {}
        self._edges: dict[tuple[str, str], AnchorEdgeConfig] = edges or {}

    # ── 工厂方法 ──────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> AnchorSystem:
        """从配置字典加载锚点系统。

        配置格式（config.yaml 中 anchors 块的值）：
            {
                "nodes": [{ "id": "...", "activation": ..., "mutable": false }, ...],
                "edges": [{ "from": "...", "to": "...", "type": "...", "weight": ..., "mutable": false }, ...]
            }

        Args:
            config: 配置字典（anchors 块的值）

        Returns:
            加载完成的 AnchorSystem 实例
        """
        nodes: dict[str, AnchorNodeConfig] = {}
        edges: dict[tuple[str, str], AnchorEdgeConfig] = {}

        for node_cfg in config.get("nodes", []):
            anc = AnchorNodeConfig(
                id=node_cfg["id"],
                activation=float(node_cfg.get("activation", 0.5)),
                label=node_cfg.get("label", node_cfg["id"]),
                mutable=bool(node_cfg.get("mutable", False)),
            )
            nodes[anc.id] = anc

        for edge_cfg in config.get("edges", []):
            anc = AnchorEdgeConfig(
                source=edge_cfg["from"],
                target=edge_cfg["to"],
                edge_type=edge_cfg.get("type", "connects"),
                weight=float(edge_cfg.get("weight", 0.5)),
                mutable=bool(edge_cfg.get("mutable", False)),
            )
            edges[(anc.source, anc.target)] = anc

        return cls(nodes=nodes, edges=edges)

    @classmethod
    def default(cls) -> AnchorSystem:
        """返回预置的默认锚点系统，用于无配置文件场景。

        默认锚点定义了最基础的语义骨架：
          - 概念锚点：self, user, negation, affirmation, task, goal, constraint, concept
          - 拓扑锚点：negation↔affirmation (contrasts), self→user (connects),
                      user→task (connects), task→goal (connects)
        """
        return cls(
            nodes={
                "self": AnchorNodeConfig(id="self", activation=1.0, label="系统自身"),
                "user": AnchorNodeConfig(id="user", activation=0.8, label="用户"),
                "negation": AnchorNodeConfig(id="negation", activation=0.5, label="否定"),
                "affirmation": AnchorNodeConfig(id="affirmation", activation=0.5, label="肯定"),
                "task": AnchorNodeConfig(id="task", activation=0.7, label="核心任务"),
                "goal": AnchorNodeConfig(id="goal", activation=0.7, label="深层目标"),
                "constraint": AnchorNodeConfig(id="constraint", activation=0.6, label="约束"),
                "concept": AnchorNodeConfig(id="concept", activation=0.6, label="概念"),
            },
            edges={
                ("negation", "affirmation"): AnchorEdgeConfig(
                    source="negation", target="affirmation", edge_type="contrasts", weight=0.1
                ),
                ("affirmation", "negation"): AnchorEdgeConfig(
                    source="affirmation", target="negation", edge_type="contrasts", weight=0.1
                ),
                ("self", "user"): AnchorEdgeConfig(
                    source="self", target="user", edge_type="connects", weight=1.0
                ),
                ("user", "task"): AnchorEdgeConfig(
                    source="user", target="task", edge_type="connects", weight=0.8
                ),
                ("task", "goal"): AnchorEdgeConfig(
                    source="task", target="goal", edge_type="connects", weight=0.8
                ),
            },
        )

    # ── 查询 API ───────────────────────────────────────────────────────────

    def is_node_mutable(self, node_id: str) -> bool:
        """检查节点是否可演化。

        锚点节点返回 False（不可变），非锚点节点返回 True（可演化）。
        """
        anc = self._nodes.get(node_id)
        if anc is None:
            # 不在锚点中 → 可演化
            return True
        return anc.mutable

    def is_edge_mutable(self, source: str, target: str) -> bool:
        """检查边是否可演化。

        锚点边返回 False（不可变），非锚点边返回 True（可演化）。
        """
        anc = self._edges.get((source, target))
        if anc is None:
            # 不在锚点中 → 可演化
            return True
        return anc.mutable

    def get_node_activation(self, node_id: str) -> float | None:
        """获取锚点节点的固定激活值。

        如果节点不是锚点，返回 None。
        """
        anc = self._nodes.get(node_id)
        return anc.activation if anc else None

    def get_edge_weight(self, source: str, target: str) -> float | None:
        """获取锚点边的固定权重。

        如果边不是锚点，返回 None。
        """
        anc = self._edges.get((source, target))
        return anc.weight if anc else None

    # ── 集合查询 ──────────────────────────────────────────────────────────

    @property
    def node_ids(self) -> set[str]:
        """所有锚点节点的 ID 集合。"""
        return set(self._nodes.keys())

    @property
    def edge_keys(self) -> set[tuple[str, str]]:
        """所有锚点边的 (source, target) 集合。"""
        return set(self._edges.keys())

    def __len__(self) -> int:
        """锚点数量（节点 + 边）。"""
        return len(self._nodes) + len(self._edges)

    def __repr__(self) -> str:
        return (
            f"AnchorSystem(nodes={len(self._nodes)}, edges={len(self._edges)})"
        )