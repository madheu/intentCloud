"""意图云锚点系统单元测试。

覆盖：
  - AnchorNodeConfig / AnchorEdgeConfig 数据模型
  - AnchorSystem 查询 API（is_node_mutable, is_edge_mutable, get_*）
  - AnchorSystem 工厂方法（from_config, default）
  - IntentCloud 集成（update_weight_safe 跳过不可变边, update_topology）
  - 配置加载（config.yaml anchors 块）
"""

from __future__ import annotations

import pytest

from core.anchor_system import AnchorNodeConfig, AnchorEdgeConfig, AnchorSystem
from core.intent_cloud import IntentCloud, CloudEdge
from core.intent_cloud_config import IntentCloudConfig
from core.models import ImmutableKernel, Identity, Constraint


# ── 共享 fixture ──────────────────────────────────────────────────────────────

@pytest.fixture
def default_anchors() -> AnchorSystem:
    return AnchorSystem.default()


@pytest.fixture
def sample_cloud() -> IntentCloud:
    """创建一个带锚点系统的 IntentCloud 实例。"""
    kernel = ImmutableKernel(
        identity=Identity(name="TestBot", role="test assistant", immutable=True),
        safety_constraints=[Constraint(id="c1", text="不得生成恶意代码", level="absolute")],
    )
    return IntentCloud(kernel=kernel, max_shell_size=10)


# ── AnchorNodeConfig 测试 ─────────────────────────────────────────────────────

def test_anchor_node_config_defaults():
    """默认值：activation=0.5, label="", mutable=False。"""
    node = AnchorNodeConfig(id="test")
    assert node.id == "test"
    assert node.activation == 0.5
    assert node.label == ""
    assert node.mutable is False


def test_anchor_node_config_custom():
    """自定义激活值和标签。"""
    node = AnchorNodeConfig(id="self", activation=1.0, label="系统自身")
    assert node.activation == 1.0
    assert node.label == "系统自身"


def test_anchor_node_config_rejects_negative_activation():
    """负激活值应抛出 ValueError。"""
    with pytest.raises(ValueError, match="must be >= 0"):
        AnchorNodeConfig(id="bad", activation=-0.1)


def test_anchor_node_config_is_frozen():
    """frozen dataclass 不可修改。"""
    node = AnchorNodeConfig(id="x")
    with pytest.raises(Exception):
        node.activation = 0.9  # type: ignore[misc]


# ── AnchorEdgeConfig 测试 ─────────────────────────────────────────────────────

def test_anchor_edge_config_defaults():
    """默认值：edge_type="connects", weight=0.5, mutable=False。"""
    edge = AnchorEdgeConfig(source="A", target="B")
    assert edge.edge_type == "connects"
    assert edge.weight == 0.5
    assert edge.mutable is False


def test_anchor_edge_config_custom():
    """自定义类型和权重。"""
    edge = AnchorEdgeConfig(source="negation", target="affirmation", edge_type="contrasts", weight=0.1)
    assert edge.edge_type == "contrasts"
    assert edge.weight == 0.1


def test_anchor_edge_config_rejects_negative_weight():
    """负权重应抛出 ValueError。"""
    with pytest.raises(ValueError, match="must be >= 0"):
        AnchorEdgeConfig(source="A", target="B", weight=-0.1)


def test_anchor_edge_config_rejects_self_loop():
    """自环应抛出 ValueError。"""
    with pytest.raises(ValueError, match="self-loop"):
        AnchorEdgeConfig(source="A", target="A")


def test_anchor_edge_config_is_frozen():
    """frozen dataclass 不可修改。"""
    edge = AnchorEdgeConfig(source="A", target="B")
    with pytest.raises(Exception):
        edge.weight = 0.9  # type: ignore[misc]


# ── AnchorSystem 查询 API ─────────────────────────────────────────────────────

def test_is_node_mutable_anchor_return_false(default_anchors):
    """锚点节点返回 False。"""
    assert default_anchors.is_node_mutable("self") is False
    assert default_anchors.is_node_mutable("negation") is False
    assert default_anchors.is_node_mutable("task") is False


def test_is_node_mutable_unknown_return_true(default_anchors):
    """非锚点节点返回 True。"""
    assert default_anchors.is_node_mutable("random_node") is True
    assert default_anchors.is_node_mutable("shell-0001-abc") is True


def test_is_edge_mutable_anchor_return_false(default_anchors):
    """锚点边返回 False。"""
    assert default_anchors.is_edge_mutable("negation", "affirmation") is False
    assert default_anchors.is_edge_mutable("self", "user") is False


def test_is_edge_mutable_unknown_return_true(default_anchors):
    """非锚点边返回 True。"""
    assert default_anchors.is_edge_mutable("A", "B") is True
    assert default_anchors.is_edge_mutable("random_src", "random_tgt") is True


def test_get_node_activation(default_anchors):
    """获取锚点节点的固定激活值。"""
    assert default_anchors.get_node_activation("self") == 1.0
    assert default_anchors.get_node_activation("user") == 0.8
    assert default_anchors.get_node_activation("negation") == 0.5


def test_get_node_activation_unknown_return_none(default_anchors):
    """非锚点节点返回 None。"""
    assert default_anchors.get_node_activation("unknown") is None


def test_get_edge_weight(default_anchors):
    """获取锚点边的固定权重。"""
    assert default_anchors.get_edge_weight("negation", "affirmation") == 0.1
    assert default_anchors.get_edge_weight("self", "user") == 1.0


def test_get_edge_weight_unknown_return_none(default_anchors):
    """非锚点边返回 None。"""
    assert default_anchors.get_edge_weight("A", "B") is None


def test_node_ids(default_anchors):
    """node_ids 返回所有锚点节点 ID。"""
    ids = default_anchors.node_ids
    assert "self" in ids
    assert "user" in ids
    assert "negation" in ids
    assert len(ids) == 8  # 8 个默认锚点节点


def test_edge_keys(default_anchors):
    """edge_keys 返回所有锚点边的 (source, target) 集合。"""
    keys = default_anchors.edge_keys
    assert ("negation", "affirmation") in keys
    assert ("self", "user") in keys
    assert len(keys) == 5  # 5 条默认锚点边


def test_len(default_anchors):
    """__len__ 返回节点 + 边的总数。"""
    assert len(default_anchors) == 8 + 5  # 8 nodes + 5 edges


def test_repr(default_anchors):
    """__repr__ 包含节点和边计数。"""
    r = repr(default_anchors)
    assert "nodes=8" in r
    assert "edges=5" in r


# ── AnchorSystem 工厂方法 ─────────────────────────────────────────────────────

def test_default_has_expected_nodes():
    """默认锚点系统包含所有 8 个核心概念节点。"""
    anchors = AnchorSystem.default()
    expected = {"self", "user", "negation", "affirmation", "task", "goal", "constraint", "concept"}
    assert anchors.node_ids == expected


def test_default_has_expected_edges():
    """默认锚点系统包含 5 条拓扑锚点边。"""
    anchors = AnchorSystem.default()
    expected = {
        ("negation", "affirmation"),
        ("affirmation", "negation"),
        ("self", "user"),
        ("user", "task"),
        ("task", "goal"),
    }
    assert anchors.edge_keys == expected


def test_from_config_minimal():
    """从最小配置字典加载。"""
    config = {
        "nodes": [{"id": "A", "activation": 0.8}],
        "edges": [],
    }
    anchors = AnchorSystem.from_config(config)
    assert anchors.is_node_mutable("A") is False
    assert anchors.get_node_activation("A") == 0.8


def test_from_config_full():
    """从完整配置字典加载。"""
    config = {
        "nodes": [
            {"id": "self", "activation": 1.0, "label": "自身", "mutable": False},
            {"id": "user", "activation": 0.9, "label": "用户", "mutable": False},
        ],
        "edges": [
            {"from": "self", "to": "user", "type": "connects", "weight": 0.9, "mutable": False},
        ],
    }
    anchors = AnchorSystem.from_config(config)
    assert anchors.get_node_activation("self") == 1.0
    assert anchors.get_node_activation("user") == 0.9
    assert anchors.get_edge_weight("self", "user") == 0.9
    assert len(anchors) == 3


def test_from_config_empty():
    """空配置返回空锚点系统。"""
    anchors = AnchorSystem.from_config({})
    assert len(anchors) == 0


# ── IntentCloud 集成：update_weight_safe ──────────────────────────────────────

def test_update_weight_safe_skips_anchor_edge(sample_cloud):
    """锚点边的权重不应被更新，返回 None。"""
    result = sample_cloud.update_weight_safe("negation", "affirmation", 1.0, 1.0)
    assert result is None
    # 锚点边权重保持不变
    edge = sample_cloud.get_edge("negation", "affirmation")
    assert edge is not None
    assert edge.weight == 0.1  # 锚点固定权重


def test_update_weight_safe_updates_non_anchor_edge(sample_cloud):
    """非锚点边正常更新。"""
    result = sample_cloud.update_weight_safe("A", "B", 1.0, 1.0)
    assert result is not None
    assert result.weight > 0.5  # 赫布增强


def test_update_weight_safe_creates_new_edge(sample_cloud):
    """不存在的边自动创建（非锚点）。"""
    result = sample_cloud.update_weight_safe("new_src", "new_tgt", 0.5, 0.5)
    assert result is not None
    assert result.source_id == "new_src"
    assert result.target_id == "new_tgt"


def test_update_weight_safe_anchor_edge_unchanged_after_multiple_calls(sample_cloud):
    """多次调用 update_weight_safe 后，锚点边权重始终不变。"""
    for _ in range(10):
        result = sample_cloud.update_weight_safe("self", "user", 1.0, 1.0)
        assert result is None

    edge = sample_cloud.get_edge("self", "user")
    assert edge.weight == 1.0  # 锚点固定权重，始终不变


# ── IntentCloud 集成：update_topology ─────────────────────────────────────────

def test_update_topology_skips_anchor_edges(sample_cloud):
    """update_topology 跳过所有锚点边，只更新可变边。"""
    # 添加一条可变边
    sample_cloud.update_weight_safe("A", "B", 0.5, 0.5)

    activations = {
        "self": 1.0, "user": 1.0, "negation": 1.0, "affirmation": 1.0,
        "A": 1.0, "B": 1.0,
    }

    updated = sample_cloud.update_topology(activations)

    # 只有 1 条非锚点边被更新
    assert len(updated) == 1
    assert updated[0].source_id == "A"
    assert updated[0].target_id == "B"


def test_update_topology_no_activations_defaults_to_zero(sample_cloud):
    """未知节点 activation 默认 0，死区内只衰减。

    第一步先做一次赫布增强（权重从 0.5 升到 0.525），
    然后 update_topology 传入空激活字典，验证权重向 w_ref 衰减。
    """
    sample_cloud.update_weight_safe("A", "B", 0.5, 0.5)
    w_after_hebb = sample_cloud.get_edge("A", "B").weight

    # 不提供任何激活值
    updated = sample_cloud.update_topology({})

    assert len(updated) == 1
    # 零激活 → 死区，仅衰减（权重从 hebb 增强后的值向 w_ref 回归）
    assert updated[0].weight < w_after_hebb


def test_update_topology_preserves_anchor_edge_weights(sample_cloud):
    """update_topology 后锚点边权重保持不变。"""
    # 记录原始锚点边权重
    orig_neg_aff = sample_cloud.get_edge("negation", "affirmation").weight
    orig_self_user = sample_cloud.get_edge("self", "user").weight

    activations = {"negation": 1.0, "affirmation": 1.0, "self": 1.0, "user": 1.0}
    sample_cloud.update_topology(activations)

    assert sample_cloud.get_edge("negation", "affirmation").weight == orig_neg_aff
    assert sample_cloud.get_edge("self", "user").weight == orig_self_user


# ── IntentCloud 集成：配置加载 ────────────────────────────────────────────────

def test_cloud_loads_anchors_from_config():
    """IntentCloud 默认从 config.yaml 加载锚点。"""
    cloud = IntentCloud()
    # 确认锚点已加载
    assert cloud.anchor_system is not None
    assert "self" in cloud.anchor_system.node_ids
    # 锚点边已初始化
    assert cloud.get_edge("self", "user") is not None


def test_cloud_accepts_custom_anchor_system():
    """IntentCloud 接受自定义锚点系统。"""
    custom = AnchorSystem(
        nodes={"custom_node": AnchorNodeConfig(id="custom_node", activation=0.9)},
        edges={},
    )
    cloud = IntentCloud(anchor_system=custom)
    assert cloud.anchor_system.is_node_mutable("custom_node") is False
    assert cloud.anchor_system.get_node_activation("custom_node") == 0.9


# ── 结构锚点：激活限幅和权重投影（已在 IntentCloudConfig 中） ────────────────

def test_structural_anchor_activation_clamp():
    """激活饱和限幅：a_max 约束生效。"""
    cfg = IntentCloudConfig(a_max=1.0)
    edge = CloudEdge("A", "B", weight=0.5, ref_weight=0.5, prev_weight=0.5)
    from core.intent_cloud import update_weight
    result = update_weight(edge, 2.0, 3.0, cfg)
    # 激活被截断到 1.0，赫布 = 0.1 * 1.0 * 1.0 = 0.1
    assert result.weight == pytest.approx(0.6, abs=1e-9)


def test_structural_anchor_weight_projection():
    """权重投影：w_max 约束生效。"""
    cfg = IntentCloudConfig(w_max=0.8)
    edge = CloudEdge("A", "B", weight=0.75, ref_weight=0.5, prev_weight=0.75)
    from core.intent_cloud import update_weight
    result = update_weight(edge, 1.0, 1.0, cfg)
    # Proj(0.75 + 0.1) = Proj(0.85) = 0.8，然后拉回和阻尼
    # 拉回 = -0.01*(0.75-0.5) = -0.0025, 阻尼 = -0.3*(0.75-0.75) = 0
    # w_new = 0.8 - 0.0025 = 0.7975
    assert result.weight <= 0.8
    assert result.weight == pytest.approx(0.7975, abs=1e-9)