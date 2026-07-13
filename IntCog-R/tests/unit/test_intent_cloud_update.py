"""意图云权重更新律单元测试。

覆盖场景：
  a. 正常赫布增强
  b. 自然衰减
  c. 参考模型拉回
  d. 阻尼消振
  e. 投影截断
  f. 死区生效
"""

from __future__ import annotations

import math

import pytest

from core.intent_cloud import CloudEdge, update_weight, _project
from core.intent_cloud_config import IntentCloudConfig


# ── 共享 fixture ──────────────────────────────────────────────────────────────

@pytest.fixture
def default_config() -> IntentCloudConfig:
    """默认配置：η=0.1, γ=0.01, K_d=0.02, δ=0.01, w_min=0.0, w_max=1.0, a_max=1.0。"""
    return IntentCloudConfig()


@pytest.fixture
def edge_at_ref() -> CloudEdge:
    """一条处于锚定状态的边：weight == ref_weight == prev_weight == 0.5。"""
    return CloudEdge("A", "B", weight=0.5, ref_weight=0.5, prev_weight=0.5)


# ── 场景 a：正常赫布增强 ─────────────────────────────────────────────────────

def test_hebbian_enhancement(default_config, edge_at_ref):
    """两节点高激活 (a_i=1.0, a_j=0.8)，权重应上升但不超 w_max。

    预期行为：
      - 赫布项 = 0.1 * 1.0 * 0.8 = 0.08
      - 参考拉回 = 0（因为 w_old == w_ref）
      - 动量阻尼 = 0（因为 w_old == w_prev）
      - 综合：w_new ≈ 0.5 + 0.08 = 0.58
    """
    edge = update_weight(edge_at_ref, 1.0, 0.8, default_config)
    assert edge.weight > 0.5  # 权重上升
    assert edge.weight <= default_config.w_max  # 不超过上界
    assert edge.weight == pytest.approx(0.58, abs=1e-9)


def test_hebbian_enhancement_respects_w_max(default_config, edge_at_ref):
    """权重接近 w_max 时，赫布增强后投影截断，再被拉回项修正。

    设置 w_old=0.95，a_i=a_j=1.0，赫布项=0.1。
      - Proj(0.95 + 0.1) = Proj(1.05) = 1.0（投影截断到 w_max）
      - 拉回 = -0.01*(0.95-0.5) = -0.0045
      - 阻尼 = -0.3*(0.95-0.95) = 0
      - w_new = 1.0 - 0.0045 = 0.9955
    """
    edge = CloudEdge("A", "B", weight=0.95, ref_weight=0.5, prev_weight=0.95)
    edge = update_weight(edge, 1.0, 1.0, default_config)
    assert edge.weight <= default_config.w_max  # 不超过上界
    assert edge.weight == pytest.approx(0.9955, abs=1e-9)


# ── 场景 b：自然衰减 ─────────────────────────────────────────────────────────

def test_natural_decay(default_config):
    """长期不激活（a_i=a_j=0），权重逐步衰减到 w_ref 附近。

    使用 K_d=0 的配置以消除阻尼对收敛的干扰，专注测试参考拉回衰减。

    零激活时：
      - 赫布项 = 0（死区生效，因为 0 < δ）
      - 参考拉回 = -γ*(w_old - w_ref) = -0.01*(0.7 - 0.5) = -0.002
      - 动量阻尼 = 0（K_d=0）
      - w_new = 0.7 - 0.002 = 0.698

    持续迭代 500 步后，权重应向 w_ref=0.5 收敛。
    """
    cfg = IntentCloudConfig(K_d=0.0)  # 无阻尼，纯参考拉回衰减
    edge = CloudEdge("A", "B", weight=0.7, ref_weight=0.5, prev_weight=0.7)

    # 单步衰减
    edge = update_weight(edge, 0.0, 0.0, cfg)
    assert edge.weight < 0.7  # 权重下降
    assert edge.weight == pytest.approx(0.698, abs=1e-9)

    # 多步衰减，持续迭代 500 步
    for _ in range(500):
        edge = update_weight(edge, 0.0, 0.0, cfg)

    # 500 步后应接近 w_ref
    assert edge.weight == pytest.approx(0.5, abs=0.01)


# ── 场景 c：参考模型拉回 ─────────────────────────────────────────────────────

def test_reference_pullback(default_config):
    """权重偏离 w_ref 时，参考拉回项产生向 w_ref 方向的作用力。

    设 w_old=0.8, w_ref=0.5, a_i=a_j=0（死区内）：
      - 参考拉回 = -0.01*(0.8 - 0.5) = -0.003
      - 动量阻尼 = 0（w_old == w_prev）
      - w_new = 0.8 - 0.003 = 0.797
    """
    edge = CloudEdge("A", "B", weight=0.8, ref_weight=0.5, prev_weight=0.8)
    edge = update_weight(edge, 0.0, 0.0, default_config)
    assert edge.weight < 0.8  # 被拉回
    assert edge.weight == pytest.approx(0.797, abs=1e-9)


def test_reference_pullback_below_ref(default_config):
    """权重低于 w_ref 时，拉回项产生向上的力。

    设 w_old=0.3, w_ref=0.5, a_i=a_j=0：
      - 参考拉回 = -0.01*(0.3 - 0.5) = +0.002
      - w_new = 0.3 + 0.002 = 0.302
    """
    edge = CloudEdge("A", "B", weight=0.3, ref_weight=0.5, prev_weight=0.3)
    edge = update_weight(edge, 0.0, 0.0, default_config)
    assert edge.weight > 0.3  # 向上拉回
    assert edge.weight == pytest.approx(0.302, abs=1e-9)


# ── 场景 d：阻尼消振 ─────────────────────────────────────────────────────────

def test_damping_oscillation_suppression(default_config):
    """连续两次反向更新，第二次幅度明显小于第一次。

    第一步：权重从 0.5 上升到 0.58（赫布增强）
    第二步：反向更新（a_i=a_j=0），由于动量阻尼，下降幅度应小于无阻尼情况。

    量化验证：
      第一步：w_old=0.5, w_prev=0.5, a_i=a_j=1.0
        → w_new = 0.5 + 0.1 = 0.6（无拉回无阻尼）
      第二步：w_old=0.6, w_prev=0.5, a_i=a_j=0（死区）
        → 无阻尼：w_new = 0.6 - 0.01*(0.6-0.5) = 0.599
        → 有阻尼：w_new = 0.6 - 0.01*(0.6-0.5) - 0.3*(0.6-0.5) = 0.6 - 0.001 - 0.03 = 0.569

    有阻尼的下降幅度 (0.6 - 0.569 = 0.031) 远大于无阻尼 (0.6 - 0.599 = 0.001)。
    """
    edge = CloudEdge("A", "B", weight=0.5, ref_weight=0.5, prev_weight=0.5)

    # 第一步：赫布增强
    config_no_damping = IntentCloudConfig(K_d=0.0)  # 无阻尼对照
    edge_no_damp = CloudEdge("A", "B", weight=0.5, ref_weight=0.5, prev_weight=0.5)
    edge_no_damp = update_weight(edge_no_damp, 1.0, 1.0, config_no_damping)

    edge = update_weight(edge, 1.0, 1.0, default_config)

    # 第二步：反向（零激活，衰减）
    w_before_no_damp = edge_no_damp.weight
    edge_no_damp = update_weight(edge_no_damp, 0.0, 0.0, config_no_damping)
    drop_no_damp = w_before_no_damp - edge_no_damp.weight

    w_before = edge.weight
    edge = update_weight(edge, 0.0, 0.0, default_config)
    drop_with_damp = w_before - edge.weight

    # 有阻尼的下降幅度 > 无阻尼的下降幅度
    assert drop_with_damp > drop_no_damp


def test_damping_reduces_second_step(default_config):
    """连续两次同向更新，第二次净变化应小于第一次（惯性）。

    第一步：w_old=0.5, w_prev=0.5, a_i=a_j=1.0
      → 赫布 = 0.1, 拉回 = 0, 阻尼 = 0
      → w_new = 0.6, delta1 = 0.1

    第二步：w_old=0.6, w_prev=0.5, a_i=a_j=1.0
      → 赫布 = 0.1, 拉回 = -0.01*(0.6-0.5) = -0.001, 阻尼 = -0.3*(0.6-0.5) = -0.03
      → w_new = 0.6 + 0.1 - 0.001 - 0.03 = 0.669, delta2 = 0.069

    delta2 (0.069) < delta1 (0.1)，阻尼生效。
    """
    edge = CloudEdge("A", "B", weight=0.5, ref_weight=0.5, prev_weight=0.5)

    # 第一步
    edge = update_weight(edge, 1.0, 1.0, default_config)
    delta1 = edge.weight - 0.5

    # 第二步
    w_before = edge.weight
    edge = update_weight(edge, 1.0, 1.0, default_config)
    delta2 = edge.weight - w_before

    assert delta2 < delta1  # 第二次变化幅度小于第一次


# ── 场景 e：投影截断 ─────────────────────────────────────────────────────────

def test_projection_clamp(default_config):
    """权重触及 w_max 后被投影截断，不会溢出。

    设置 w_old=0.99, a_i=a_j=1.0, w_ref=0.5, w_prev=0.99：
      - 赫布项 = 0.1
      - 中间值 Proj(0.99 + 0.1) = Proj(1.09) = 1.0
      - 拉回 = -0.01*(0.99-0.5) = -0.0049
      - 阻尼 = -0.3*(0.99-0.99) = 0
      - w_new = 1.0 - 0.0049 = 0.9951
    """
    edge = CloudEdge("A", "B", weight=0.99, ref_weight=0.5, prev_weight=0.99)
    edge = update_weight(edge, 1.0, 1.0, default_config)
    assert edge.weight <= default_config.w_max


def test_projection_clamp_w_min(default_config):
    """权重触及 w_min 后被投影截断，不会下溢。

    设置 w_old=0.001, a_i=a_j=0, w_ref=0.5, w_prev=0.001：
      - 赫布 = 0（死区，0 < δ）
      - 拉回 = -0.01*(0.001-0.5) = +0.00499
      - 阻尼 = -0.3*(0.001-0.001) = 0
      - w_new = 0.001 + 0.00499 = 0.00599（仍在范围内）
    """
    edge = CloudEdge("A", "B", weight=0.001, ref_weight=0.5, prev_weight=0.001)
    edge = update_weight(edge, 0.0, 0.0, default_config)
    assert edge.weight >= default_config.w_min


def test_projection_prev_weight_updated(default_config, edge_at_ref):
    """更新后 prev_weight 应被设为旧的 weight。"""
    old_weight = edge_at_ref.weight
    result = update_weight(edge_at_ref, 0.5, 0.5, default_config)
    assert result.prev_weight == old_weight


# ── 场景 f：死区生效 ─────────────────────────────────────────────────────────

def test_dead_zone_blocks_hebbian(default_config):
    """激活乘积低于 δ 时，赫布项不产生效果。

    δ=0.01, a_i=0.09, a_j=0.1 → 乘积 = 0.009 < 0.01。
    对比两个配置：一个有死区（δ=0.01），一个无死区（δ=0.0）。
    无死区时赫布生效，有死区时赫布被跳过。
    """
    config_no_dead = IntentCloudConfig(delta=0.0)

    edge_with_dead = CloudEdge("A", "B", weight=0.5, ref_weight=0.5, prev_weight=0.5)
    edge_no_dead = CloudEdge("A", "B", weight=0.5, ref_weight=0.5, prev_weight=0.5)

    edge_with_dead = update_weight(edge_with_dead, 0.09, 0.1, default_config)
    edge_no_dead = update_weight(edge_no_dead, 0.09, 0.1, config_no_dead)

    # 有死区时权重更低（因为没有赫布增强）
    assert edge_with_dead.weight < edge_no_dead.weight


def test_dead_zone_still_applies_decay(default_config):
    """死区内虽然跳过赫布项，但参考拉回和阻尼仍然生效。

    w_old=0.6, w_ref=0.5, w_prev=0.5, a_i=a_j=0（死区内）：
      - 赫布 = 0
      - 拉回 = -0.01*(0.6-0.5) = -0.001
      - 阻尼 = -0.02*(0.6-0.5) = -0.002
      - w_new = 0.6 - 0.001 - 0.002 = 0.597
    """
    edge = CloudEdge("A", "B", weight=0.6, ref_weight=0.5, prev_weight=0.5)
    edge = update_weight(edge, 0.0, 0.0, default_config)
    assert edge.weight < 0.6  # 衰减生效
    assert edge.weight == pytest.approx(0.597, abs=1e-9)


# ── 边界条件 ──────────────────────────────────────────────────────────────────

def test_activation_saturation_clipping(default_config):
    """激活值超过 a_max 时被截断。

    a_max=1.0, 输入 a_i=2.0, a_j=1.5：
      - 截断后 a_i=1.0, a_j=1.0
      - 赫布项 = 0.1 * 1.0 * 1.0 = 0.1
    """
    edge = CloudEdge("A", "B", weight=0.5, ref_weight=0.5, prev_weight=0.5)
    edge = update_weight(edge, 2.0, 1.5, default_config)
    # 如果没截断，乘积 = 3.0，赫布 = 0.3，w_new = 0.8
    # 截断后乘积 = 1.0，赫布 = 0.1，w_new = 0.6
    assert edge.weight == pytest.approx(0.6, abs=1e-9)


def test_negative_activation_clipped(default_config):
    """负激活值被截断到 0，等价于死区场景。

    a_i=-0.5 截断为 0，a_j=0.5 不变。乘积 = 0 < δ，赫布 = 0。
    当 w_old=w_prev=w_ref=0.5 时，拉回和阻尼均为 0，权重不变。
    """
    edge = CloudEdge("A", "B", weight=0.5, ref_weight=0.5, prev_weight=0.5)
    edge = update_weight(edge, -0.5, 0.5, default_config)
    # a_i 截断为 0，乘积 = 0 < δ，赫布 = 0；w_old == w_ref == w_prev，拉回和阻尼均为 0
    assert edge.weight == pytest.approx(0.5, abs=1e-9)


def test_nan_activation_raises(default_config, edge_at_ref):
    """NaN 激活值应抛出 ValueError。"""
    with pytest.raises(ValueError, match="NaN"):
        update_weight(edge_at_ref, float("nan"), 0.5, default_config)
    with pytest.raises(ValueError, match="NaN"):
        update_weight(edge_at_ref, 0.5, float("nan"), default_config)


def test_inf_activation_raises(default_config, edge_at_ref):
    """无穷大激活值应抛出 ValueError。"""
    with pytest.raises(ValueError, match="infinite"):
        update_weight(edge_at_ref, float("inf"), 0.5, default_config)
    with pytest.raises(ValueError, match="infinite"):
        update_weight(edge_at_ref, 0.5, float("-inf"), default_config)


def test_update_returns_same_object(default_config, edge_at_ref):
    """update_weight 返回的是同一个对象引用（原地修改）。"""
    result = update_weight(edge_at_ref, 0.5, 0.5, default_config)
    assert result is edge_at_ref


# ── _project 辅助函数 ─────────────────────────────────────────────────────────

def test_project_within_range():
    """值在范围内时原样返回。"""
    assert _project(0.5, 0.0, 1.0) == 0.5


def test_project_above_range():
    """值超出上界时裁剪到上界。"""
    assert _project(1.5, 0.0, 1.0) == 1.0


def test_project_below_range():
    """值低于下界时裁剪到下界。"""
    assert _project(-0.5, 0.0, 1.0) == 0.0


def test_project_at_boundaries():
    """值恰好等于边界时原样返回。"""
    assert _project(0.0, 0.0, 1.0) == 0.0
    assert _project(1.0, 0.0, 1.0) == 1.0


# ── IntentCloudConfig 校验 ────────────────────────────────────────────────────

def test_config_default_values():
    """默认配置各参数值正确。"""
    cfg = IntentCloudConfig()
    assert cfg.eta == 0.1
    assert cfg.gamma == 0.01
    assert cfg.K_d == 0.02
    assert cfg.delta == 0.01
    assert cfg.w_min == 0.0
    assert cfg.w_max == 1.0
    assert cfg.a_max == 1.0


def test_config_rejects_invalid_eta():
    """eta 为负应抛出 ValueError。"""
    with pytest.raises(ValueError):
        IntentCloudConfig(eta=-0.1)


def test_config_rejects_w_min_gt_w_max():
    """w_min > w_max 应抛出 ValueError。"""
    with pytest.raises(ValueError):
        IntentCloudConfig(w_min=1.0, w_max=0.0)


def test_config_is_frozen():
    """配置是 frozen dataclass，不可修改。"""
    cfg = IntentCloudConfig()
    with pytest.raises(Exception):
        cfg.eta = 0.2  # type: ignore[misc]