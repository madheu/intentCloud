"""内化数学化单元测试。

覆盖场景：
  a. 内化后 effective_gamma = 0.08
  b. 内化后 effective_K_d = config.K_d * 10
  c. 内化后 forget_factor = 0.02
  d. 内化边衰减远慢于普通边
  e. check_internalization 标记交互中的所有边
  f. 系统纠正时效果减弱
  g. 未内化边使用 config 默认值
  h. 多次内化 internalization_count 递增
"""

from __future__ import annotations

import pytest

from core.intent_cloud import CloudEdge, update_weight, IntentCloud
from core.intent_cloud_config import IntentCloudConfig
from core.models import ImmutableKernel, Identity, Constraint


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def default_config() -> IntentCloudConfig:
    return IntentCloudConfig()


@pytest.fixture
def normal_edge() -> CloudEdge:
    """普通边：weight=0.5, ref=0.5, prev=0.5，未内化。"""
    return CloudEdge("A", "B", weight=0.5, ref_weight=0.5, prev_weight=0.5)


@pytest.fixture
def internalized_edge() -> CloudEdge:
    """内化边：weight=0.5, ref=0.5, prev=0.5，已内化。"""
    return CloudEdge(
        "A", "B",
        weight=0.5, ref_weight=0.5, prev_weight=0.5,
        internalized=True, internalization_count=1,
        forget_factor=0.02,
        effective_gamma=0.08,
        effective_K_d=0.2,  # K_d=0.02 * 10 = 0.2
    )


@pytest.fixture
def cloud_with_edges() -> IntentCloud:
    """带两条边的 IntentCloud。"""
    kernel = ImmutableKernel(
        identity=Identity(name="TestBot", role="test", immutable=True),
        safety_constraints=[Constraint(id="c1", text="safe", level="absolute")],
    )
    cloud = IntentCloud(kernel=kernel)

    # 添加两条普通边
    cloud._edges[("A", "B")] = CloudEdge("A", "B", weight=0.5, ref_weight=0.5, prev_weight=0.5)
    cloud._edges[("B", "C")] = CloudEdge("B", "C", weight=0.6, ref_weight=0.6, prev_weight=0.6)

    return cloud


# ── 场景 a：内化后 effective_gamma ─────────────────────────────────────────────

def test_internalized_edge_has_reduced_gamma():
    """内化后 effective_gamma = 0.08。"""
    edge = CloudEdge(
        "A", "B", internalized=True,
        effective_gamma=0.08,
    )
    assert edge.effective_gamma == 0.08
    assert edge.internalized is True


# ── 场景 b：内化后 effective_K_d ───────────────────────────────────────────────

def test_internalized_edge_has_increased_K_d(default_config):
    """内化后 effective_K_d = config.K_d * 10。"""
    target_K_d = default_config.K_d * default_config.internalized_K_d_multiplier
    assert target_K_d == pytest.approx(0.2, abs=1e-9)

    edge = CloudEdge(
        "A", "B", internalized=True,
        effective_K_d=target_K_d,
    )
    assert edge.effective_K_d == pytest.approx(0.2, abs=1e-9)


# ── 场景 c：内化后 forget_factor ───────────────────────────────────────────────

def test_internalized_edge_has_low_forget_factor(default_config):
    """内化后 forget_factor = 0.02。"""
    assert default_config.internalized_forget_factor == 0.02

    edge = CloudEdge(
        "A", "B", internalized=True,
        forget_factor=0.02,
    )
    assert edge.forget_factor == 0.02


# ── 场景 d：内化边衰减远慢于普通边 ─────────────────────────────────────────────

def test_internalized_edge_decays_slower(default_config, internalized_edge):
    """同样输入下，内化边的权重变化远小于普通边。

    普通边：gamma=0.01, K_d=0.02, forget_factor=1.0
    内化边：gamma=0.08, K_d=0.2, forget_factor=0.02

    对普通边，衰减项 = 0.01 * (w_old - w_ref) + 0.02 * (w_old - w_prev)
    对静止边 (w_old==w_ref==w_prev)，衰减项 = 0，两边权重变化相同。

    使用偏离锚点的边来测试：
      w_old=0.7, w_ref=0.5, w_prev=0.7
      普通边衰减 = 0.01 * 0.2 * 1.0 + 0.02 * 0 * 1.0 = 0.002
      内化边衰减 = 0.08 * 0.2 * 0.02 + 0.2 * 0 * 0.02 = 0.00032
      内化边衰减 ≈ 普通边的 16%
    """
    # 普通边
    normal = CloudEdge("A", "B", weight=0.7, ref_weight=0.5, prev_weight=0.7)
    normal_before = normal.weight
    normal_after = update_weight(normal, 0.0, 0.0, default_config)
    normal_change = abs(normal_after.weight - normal_before)

    # 内化边
    internalized = CloudEdge(
        "A", "B", weight=0.7, ref_weight=0.5, prev_weight=0.7,
        internalized=True, forget_factor=0.02,
        effective_gamma=0.08, effective_K_d=0.2,
    )
    internalized_before = internalized.weight
    internalized_after = update_weight(internalized, 0.0, 0.0, default_config)
    internalized_change = abs(internalized_after.weight - internalized_before)

    # 内化边变化量应远小于普通边
    assert internalized_change < normal_change


def test_internalized_edge_decays_slower_with_hebbian(default_config):
    """即使有赫布项，内化边仍衰减更慢。

    普通边衰减 = 0.01*(0.7-0.5)*1.0 + 0.02*(0.7-0.7)*1.0 = 0.002
    内化边衰减 = 0.08*(0.7-0.5)*0.02 + 0.2*(0.7-0.7)*0.02 = 0.00032
    """
    # 普通边
    normal = CloudEdge("A", "B", weight=0.7, ref_weight=0.5, prev_weight=0.7)
    normal_after = update_weight(normal, 1.0, 1.0, default_config)

    # 内化边
    internalized = CloudEdge(
        "A", "B", weight=0.7, ref_weight=0.5, prev_weight=0.7,
        internalized=True, forget_factor=0.02,
        effective_gamma=0.08, effective_K_d=0.2,
    )
    internalized_after = update_weight(internalized, 1.0, 1.0, default_config)

    # 赫布项相同，但内化边衰减更小 → 内化边权重更高
    assert internalized_after.weight > normal_after.weight


# ── 场景 e：check_internalization 标记所有边 ───────────────────────────────────

def test_check_internalization_marks_all_edges(cloud_with_edges, default_config):
    """check_internalization 标记交互中的所有边。"""
    interaction = [("A", "B"), ("B", "C")]

    result = cloud_with_edges.check_internalization(interaction, config=default_config)

    assert len(result) == 2

    edge_ab = cloud_with_edges.get_edge("A", "B")
    assert edge_ab is not None
    assert edge_ab.internalized is True
    assert edge_ab.internalization_count == 1
    assert edge_ab.forget_factor == default_config.internalized_forget_factor
    assert edge_ab.effective_gamma == default_config.internalized_gamma

    edge_bc = cloud_with_edges.get_edge("B", "C")
    assert edge_bc is not None
    assert edge_bc.internalized is True
    assert edge_bc.internalization_count == 1


def test_check_internalization_skips_nonexistent_edges(cloud_with_edges, default_config):
    """不存在的边被跳过，不创建新边。"""
    interaction = [("A", "B"), ("ghost", "phantom")]

    result = cloud_with_edges.check_internalization(interaction, config=default_config)

    assert len(result) == 1
    assert cloud_with_edges.get_edge("ghost", "phantom") is None


def test_check_internalization_empty_list(cloud_with_edges, default_config):
    """空边列表返回空列表。"""
    result = cloud_with_edges.check_internalization([], config=default_config)
    assert result == []


# ── 场景 f：系统纠正时效果减弱 ─────────────────────────────────────────────────

def test_check_internalization_system_correction_lower_effect(cloud_with_edges, default_config):
    """系统纠正时 effective_gamma 和 effective_K_d 更接近 config 默认值。"""
    interaction = [("A", "B")]

    result = cloud_with_edges.check_internalization(
        interaction, user_correction=False, config=default_config
    )

    assert len(result) == 1
    edge = cloud_with_edges.get_edge("A", "B")
    assert edge is not None

    # system correction: correction_factor = 0.1
    # effective_gamma = 0.08 * 0.1 + 0.01 * 0.9 = 0.008 + 0.009 = 0.017
    expected_gamma = (
        default_config.internalized_gamma * 0.1
        + default_config.gamma * 0.9
    )
    assert edge.effective_gamma == pytest.approx(expected_gamma, abs=1e-9)

    # effective_K_d = (0.02 * 10) * 0.1 + 0.02 * 0.9 = 0.2 * 0.1 + 0.018 = 0.02 + 0.018 = 0.038
    target_K_d = default_config.K_d * default_config.internalized_K_d_multiplier
    expected_K_d = target_K_d * 0.1 + default_config.K_d * 0.9
    assert edge.effective_K_d == pytest.approx(expected_K_d, abs=1e-9)


# ── 场景 g：未内化边使用 config 默认值 ─────────────────────────────────────────

def test_non_internalized_edge_uses_config_defaults(default_config, normal_edge):
    """未内化边（effective_gamma=None, effective_K_d=None）使用 config 默认值。"""
    result = update_weight(normal_edge, 1.0, 1.0, default_config)

    # 权重应正常更新，使用 config.gamma 和 config.K_d
    assert result.weight > 0.5  # 赫布增强
    assert result.weight < 1.0  # 在合法范围内


def test_non_internalized_edge_forget_factor_is_one(default_config):
    """普通边的 forget_factor 默认为 1.0。"""
    edge = CloudEdge("A", "B")
    assert edge.forget_factor == 1.0
    assert edge.internalized is False
    assert edge.effective_gamma is None
    assert edge.effective_K_d is None


# ── 场景 h：多次内化 internalization_count 递增 ────────────────────────────────

def test_internalization_count_increments(cloud_with_edges, default_config):
    """多次内化 internalization_count 递增。"""
    interaction = [("A", "B")]

    # 第一次内化
    cloud_with_edges.check_internalization(interaction, config=default_config)
    edge = cloud_with_edges.get_edge("A", "B")
    assert edge is not None
    assert edge.internalization_count == 1

    # 第二次内化
    cloud_with_edges.check_internalization(interaction, config=default_config)
    assert edge.internalization_count == 2

    # 第三次内化
    cloud_with_edges.check_internalization(interaction, config=default_config)
    assert edge.internalization_count == 3


# ── CloudEdge __repr__ 包含内化信息 ────────────────────────────────────────────

def test_repr_shows_internalization():
    """内化边的 __repr__ 包含内化状态。"""
    edge = CloudEdge(
        "A", "B", weight=0.5,
        internalized=True, internalization_count=3, forget_factor=0.02,
    )
    rep = repr(edge)
    assert "internalized" in rep
    assert "count=3" in rep
    assert "ff=0.020" in rep


def test_repr_normal_edge_no_internalization():
    """普通边的 __repr__ 不包含内化信息。"""
    edge = CloudEdge("A", "B", weight=0.5)
    rep = repr(edge)
    assert "internalized" not in rep


# ── IntentCloudConfig 内化参数默认值 ───────────────────────────────────────────

def test_config_default_internalized_values():
    """验证 IntentCloudConfig 内化参数默认值。"""
    cfg = IntentCloudConfig()
    assert cfg.internalized_gamma == 0.08
    assert cfg.internalized_K_d_multiplier == 10.0
    assert cfg.internalized_forget_factor == 0.02


def test_config_internalized_validation():
    """内化参数负值抛出 ValueError。"""
    with pytest.raises(ValueError):
        IntentCloudConfig(internalized_gamma=-0.01)

    with pytest.raises(ValueError):
        IntentCloudConfig(internalized_K_d_multiplier=-1.0)

    with pytest.raises(ValueError):
        IntentCloudConfig(internalized_forget_factor=-0.01)