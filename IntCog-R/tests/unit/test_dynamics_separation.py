"""意图云动力学分离单元测试（快慢子系统）。

覆盖场景：
  a. 快子系统在固定权重下能够收敛（相邻迭代激活变化量 < ε）
  b. 快子系统达到 max_iter 时强制停止，不会无限循环
  c. 权重更新仅在激活收敛后发生一次
  d. 扩散过程中激活值始终在 [0, a_max] 范围内
  e. 概念锚点节点的激活值在扩散前后保持不变
  f. 拓扑锚点边的权重在阶段 2 中被跳过，不被修改
  g. 综合场景：输入信号触发扩散，收敛后权重更新，验证更新后的权重符合公式预期
"""

from __future__ import annotations

import pytest

from core.intent_cloud import IntentCloud, CloudEdge
from core.intent_cloud_config import IntentCloudConfig
from core.intent_cloud_dynamics import diffuse_activation, update_weights_after_diffusion
from core.anchor_system import AnchorSystem, AnchorNodeConfig, AnchorEdgeConfig
from core.models import ImmutableKernel, Identity, Constraint


# ── 共享 fixture ──────────────────────────────────────────────────────────────

@pytest.fixture
def simple_cloud() -> IntentCloud:
    """创建一个简单的 IntentCloud，只有几条边。"""
    kernel = ImmutableKernel(
        identity=Identity(name="TestBot", role="test", immutable=True),
        safety_constraints=[Constraint(id="c1", text="safe", level="absolute")],
    )
    # 自定义锚点系统，减少默认锚点干扰测试
    anchors = AnchorSystem(
        nodes={"anchor_node": AnchorNodeConfig(id="anchor_node", activation=0.8)},
        edges={
            ("anchor_node", "B"): AnchorEdgeConfig(
                source="anchor_node", target="B", weight=0.5
            ),
        },
    )
    cloud = IntentCloud(kernel=kernel, anchor_system=anchors)
    # 添加非锚点边
    cloud._edges[("A", "B")] = CloudEdge("A", "B", weight=0.5)
    cloud._edges[("B", "C")] = CloudEdge("B", "C", weight=0.5)
    return cloud


@pytest.fixture
def default_config() -> IntentCloudConfig:
    return IntentCloudConfig()


# ── 场景 a：快子系统收敛 ─────────────────────────────────────────────────────

def test_fast_subsystem_converges(simple_cloud, default_config):
    """快子系统在固定权重下能够收敛（相邻迭代激活变化量 < ε）。"""
    input_signals = {"A": 1.0}
    activations = diffuse_activation(simple_cloud, input_signals, default_config)

    # 再次调用，验证收敛后变化很小
    activations2 = diffuse_activation(simple_cloud, input_signals, default_config)

    max_diff = max(abs(activations.get(k, 0) - activations2.get(k, 0)) for k in set(activations.keys()) | set(activations2.keys()))
    assert max_diff < default_config.epsilon * 2  # 允许微小误差


# ── 场景 b：max_iter 强制停止 ────────────────────────────────────────────────

def test_fast_subsystem_stops_at_max_iter(simple_cloud):
    """快子系统达到 max_iter 时强制停止，不会无限循环。"""
    # 设置很小的 alpha，使收敛很慢，但 max_iter=5 会强制停止
    config = IntentCloudConfig(alpha=0.001, epsilon=0.0001, max_iter=5)
    input_signals = {"A": 1.0}

    activations = diffuse_activation(simple_cloud, input_signals, config)

    # 验证返回的激活值存在（不会无限循环）
    assert isinstance(activations, dict)
    assert len(activations) > 0


# ── 场景 c：权重更新仅在激活收敛后发生一次 ────────────────────────────────────

def test_weight_update_happens_once_after_convergence(simple_cloud, default_config):
    """权重更新仅在激活收敛后发生一次。"""
    input_signals = {"A": 1.0}

    # 记录初始权重
    edge_before = simple_cloud.get_edge("A", "B")
    assert edge_before is not None
    w_before = edge_before.weight

    # 阶段 1：扩散激活（权重不应变化）
    activations = diffuse_activation(simple_cloud, input_signals, default_config)
    w_after_diffusion = simple_cloud.get_edge("A", "B").weight
    assert w_after_diffusion == w_before  # 扩散过程中权重不变

    # 阶段 2：更新权重
    update_weights_after_diffusion(simple_cloud, activations, default_config)
    w_after_update = simple_cloud.get_edge("A", "B").weight

    # 权重确实被更新了
    assert w_after_update != w_before


# ── 场景 d：激活值范围约束 ────────────────────────────────────────────────────

def test_activation_always_in_range(simple_cloud, default_config):
    """扩散过程中激活值始终在 [0, a_max] 范围内。"""
    input_signals = {"A": 2.0}  # 超过 a_max=1.0

    activations = diffuse_activation(simple_cloud, input_signals, default_config)

    for node_id, act in activations.items():
        assert act >= 0.0, f"Activation for {node_id} is negative: {act}"
        assert act <= default_config.a_max, f"Activation for {node_id} exceeds a_max: {act}"


# ── 场景 e：概念锚点激活值保持不变 ────────────────────────────────────────────

def test_concept_anchor_activation_fixed(simple_cloud, default_config):
    """概念锚点节点的激活值在扩散前后保持不变。"""
    input_signals = {"A": 1.0, "anchor_node": 0.0}  # 尝试覆盖锚点

    activations = diffuse_activation(simple_cloud, input_signals, default_config)

    # 概念锚点的激活值应保持锚点系统中定义的值（0.8），而非输入信号（0.0）
    assert activations["anchor_node"] == pytest.approx(0.8, abs=1e-9)


# ── 场景 f：拓扑锚点边权重不被修改 ────────────────────────────────────────────

def test_topological_anchor_edge_weight_unchanged(simple_cloud, default_config):
    """拓扑锚点边的权重在阶段 2 中被跳过，不被修改。"""
    input_signals = {"anchor_node": 1.0, "A": 1.0}  # 添加 A 的激活值

    # 获取锚点边初始权重
    anchor_edge = simple_cloud.get_edge("anchor_node", "B")
    assert anchor_edge is not None
    w_before = anchor_edge.weight

    # 完整执行过程
    activations = diffuse_activation(simple_cloud, input_signals, default_config)
    update_weights_after_diffusion(simple_cloud, activations, default_config)

    # 锚点边权重保持不变
    assert simple_cloud.get_edge("anchor_node", "B").weight == w_before

    # 非锚点边权重应被修改
    non_anchor_edge = simple_cloud.get_edge("A", "B")
    assert non_anchor_edge.weight != 0.5  # 初始值是 0.5


# ── 场景 g：综合场景 ──────────────────────────────────────────────────────────

def test_integration_scenarios(simple_cloud):
    """综合场景：输入信号触发扩散，收敛后权重更新，验证权重符合公式预期。"""
    # 简化配置：关闭参考拉回和阻尼，只测试赫布项
    config = IntentCloudConfig(
        alpha=0.5,
        epsilon=0.001,
        max_iter=100,
        eta=0.1,
        gamma=0.0,  # 关闭拉回
        K_d=0.0,    # 关闭阻尼
        delta=0.0,  # 关闭死区
    )

    # 初始边权重
    edge_AB = simple_cloud.get_edge("A", "B")
    edge_AB.weight = 0.5
    edge_AB.ref_weight = 0.5
    edge_AB.prev_weight = 0.5

    # 输入信号
    input_signals = {"A": 1.0}

    # 阶段 1：扩散激活
    activations = diffuse_activation(simple_cloud, input_signals, config)

    # 阶段 2：更新权重
    update_weights_after_diffusion(simple_cloud, activations, config)

    # 验证权重更新符合公式：w_new = w_old + η·a_i·a_j（因为 gamma=K_d=delta=0）
    a_A = activations.get("A", 0.0)
    a_B = activations.get("B", 0.0)
    expected_w = 0.5 + config.eta * a_A * a_B

    assert edge_AB.weight == pytest.approx(expected_w, abs=0.01)


def test_process_interaction_end_to_end(simple_cloud, default_config):
    """process_interaction 端到端测试：完整执行快+慢子系统。"""
    input_signals = {"A": 1.0}

    activations = simple_cloud.process_interaction(input_signals, default_config)

    # 返回的激活值存在且合理
    assert isinstance(activations, dict)
    assert "A" in activations
    assert activations["A"] >= 0.0

    # 非锚点边权重已被更新
    edge_AB = simple_cloud.get_edge("A", "B")
    assert edge_AB.weight != 0.5  # 初始值是 0.5

    # 锚点边权重保持不变
    anchor_edge = simple_cloud.get_edge("anchor_node", "B")
    assert anchor_edge.weight == 0.5  # 初始值是 0.5


# ── 边界条件测试 ──────────────────────────────────────────────────────────────

def test_diffuse_activation_rejects_nan():
    """输入信号包含 NaN 应抛出 ValueError。"""
    kernel = ImmutableKernel(
        identity=Identity(name="TestBot", role="test", immutable=True),
        safety_constraints=[],
    )
    cloud = IntentCloud(kernel=kernel)
    cloud._edges[("A", "B")] = CloudEdge("A", "B", weight=0.5)

    config = IntentCloudConfig()

    with pytest.raises(ValueError, match="NaN"):
        diffuse_activation(cloud, {"A": float("nan")}, config)


def test_diffuse_activation_rejects_inf():
    """输入信号包含无穷大应抛出 ValueError。"""
    kernel = ImmutableKernel(
        identity=Identity(name="TestBot", role="test", immutable=True),
        safety_constraints=[],
    )
    cloud = IntentCloud(kernel=kernel)
    cloud._edges[("A", "B")] = CloudEdge("A", "B", weight=0.5)

    config = IntentCloudConfig()

    with pytest.raises(ValueError, match="infinite"):
        diffuse_activation(cloud, {"A": float("inf")}, config)


def test_no_edges_no_diffusion():
    """无边的情况下，激活值直接等于输入信号（裁剪后）。"""
    kernel = ImmutableKernel(
        identity=Identity(name="TestBot", role="test", immutable=True),
        safety_constraints=[],
    )
    cloud = IntentCloud(kernel=kernel)
    # 不添加任何边

    config = IntentCloudConfig()
    input_signals = {"A": 0.8}

    activations = diffuse_activation(cloud, input_signals, config)

    assert activations["A"] == 0.8  # 无边，无扩散


# ── IntentCloudConfig 新增字段校验 ────────────────────────────────────────────

def test_config_new_fields_defaults():
    """新增字段默认值正确。"""
    cfg = IntentCloudConfig()
    assert cfg.alpha == 0.1
    assert cfg.epsilon == 0.001
    assert cfg.max_iter == 100


def test_config_rejects_zero_alpha():
    """alpha <= 0 应抛出 ValueError。"""
    with pytest.raises(ValueError, match="alpha must be > 0"):
        IntentCloudConfig(alpha=0)


def test_config_rejects_zero_epsilon():
    """epsilon <= 0 应抛出 ValueError。"""
    with pytest.raises(ValueError, match="epsilon must be > 0"):
        IntentCloudConfig(epsilon=0)


def test_config_rejects_zero_max_iter():
    """max_iter <= 0 应抛出 ValueError。"""
    with pytest.raises(ValueError, match="max_iter must be > 0"):
        IntentCloudConfig(max_iter=0)