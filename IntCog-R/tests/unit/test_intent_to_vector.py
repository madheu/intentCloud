"""意图到向量映射层单元测试。

覆盖：
  a. 四种边类型的对比对生成（refines/contrasts/evokes/constrains）
  b. 缺失模板配置时抛出 ValueError
  c. 空状态（无边）降级为 None
  d. 无 extractor 时降级为零向量
  e. map_intent_to_vector 端到端（含强度计算）
  f. 节点语义查找回退
"""

from __future__ import annotations

import pytest

from core.intent_to_vector import (
    generate_contrastive_pair,
    map_intent_to_vector,
    _get_node_semantic,
    load_templates,
)
from core.intent_cloud import IntentCloud, CloudEdge
from core.anchor_system import AnchorSystem, AnchorNodeConfig, AnchorEdgeConfig
from core.models import ImmutableKernel, Identity, Constraint


# ── 共享 fixture ──────────────────────────────────────────────────────────────

@pytest.fixture
def sample_templates() -> dict[str, dict[str, str]]:
    """测试用的对比对模板。"""
    return {
        "refines": {
            "positive": "深化{concept}的内涵，聚焦{target}的特质",
            "negative": "对{concept}进行最浅层的字面描述",
        },
        "contrasts": {
            "positive": "强化{concept}与{target}的对立关系",
            "negative": "将{concept}推向与{target}完全相反的方向",
        },
        "evokes": {
            "positive": "通过{concept}延伸丰富的联想链，关联到{target}",
            "negative": "看到{concept}就只是看到{concept}，切断所有联想",
        },
        "constrains": {
            "positive": "严格遵守{concept}的约束，确保{target}在规则范围内",
            "negative": "无视{concept}的限制，让{target}突破所有约束",
        },
    }


@pytest.fixture
def cloud_with_edges() -> IntentCloud:
    """创建一个有非锚点边的 IntentCloud。"""
    kernel = ImmutableKernel(
        identity=Identity(name="TestBot", role="test", immutable=True),
        safety_constraints=[Constraint(id="c1", text="safe", level="absolute")],
    )
    anchors = AnchorSystem(
        nodes={
            "self": AnchorNodeConfig(id="self", activation=1.0, label="系统自身"),
        },
        edges={},
    )
    cloud = IntentCloud(kernel=kernel, anchor_system=anchors)

    # 添加非锚点边
    cloud._edges[("self", "task")] = CloudEdge(
        "self", "task", weight=0.8, edge_type="refines"
    )
    cloud._edges[("self", "goal")] = CloudEdge(
        "self", "goal", weight=0.6, edge_type="contrasts"
    )
    cloud._edges[("task", "goal")] = CloudEdge(
        "task", "goal", weight=0.9, edge_type="evokes"
    )

    return cloud


# ── 场景 a：四种边类型的对比对生成 ────────────────────────────────────────────

def test_generate_refines_pair(sample_templates):
    """refines 边类型生成深化 vs 浅层描述。"""
    pos, neg = generate_contrastive_pair("refines", "诗歌", "存在", sample_templates)
    assert "深化诗歌的内涵" in pos
    assert "聚焦存在的特质" in pos
    assert "对诗歌进行最浅层的字面描述" == neg


def test_generate_contrasts_pair(sample_templates):
    """contrasts 边类型生成强化对立 vs 取反方向。"""
    pos, neg = generate_contrastive_pair("contrasts", "苦难", "享乐", sample_templates)
    assert "强化苦难与享乐的对立关系" == pos
    assert "将苦难推向与享乐完全相反的方向" == neg


def test_generate_evokes_pair(sample_templates):
    """evokes 边类型生成联想延伸 vs 切断联想。"""
    pos, neg = generate_contrastive_pair("evokes", "月亮", "故乡", sample_templates)
    assert "通过月亮延伸丰富的联想链，关联到故乡" == pos
    assert "看到月亮就只是看到月亮，切断所有联想" == neg


def test_generate_constrains_pair(sample_templates):
    """constrains 边类型生成遵守约束 vs 突破约束。"""
    pos, neg = generate_contrastive_pair("constrains", "字数限制", "回答", sample_templates)
    assert "严格遵守字数限制的约束，确保回答在规则范围内" == pos
    assert "无视字数限制的限制，让回答突破所有约束" == neg


# ── 场景 b：缺失配置报错 ──────────────────────────────────────────────────────

def test_generate_unknown_edge_type_raises(sample_templates):
    """不存在的边类型应抛出 ValueError。"""
    with pytest.raises(ValueError, match="No contrastive template"):
        generate_contrastive_pair("unknown_type", "A", "B", sample_templates)


def test_generate_empty_templates_raises():
    """空模板字典应抛出 ValueError。"""
    with pytest.raises(ValueError, match="No contrastive template"):
        generate_contrastive_pair("refines", "A", "B", {})


# ── 场景 c：空状态降级 ─────────────────────────────────────────────────────────

def test_map_intent_to_vector_no_edges():
    """无边时返回 None。"""
    kernel = ImmutableKernel(
        identity=Identity(name="TestBot", role="test", immutable=True),
        safety_constraints=[],
    )
    cloud = IntentCloud(kernel=kernel)
    # 不添加任何非锚点边

    result = map_intent_to_vector(cloud)
    assert result is None


def test_map_intent_to_vector_only_anchor_edges():
    """只有锚点边时返回 None。"""
    kernel = ImmutableKernel(
        identity=Identity(name="TestBot", role="test", immutable=True),
        safety_constraints=[],
    )
    anchors = AnchorSystem(
        nodes={"self": AnchorNodeConfig(id="self", activation=1.0)},
        edges={
            ("self", "user"): AnchorEdgeConfig("self", "user", weight=1.0),
        },
    )
    cloud = IntentCloud(kernel=kernel, anchor_system=anchors)

    result = map_intent_to_vector(cloud)
    assert result is None


# ── 场景 d：无 extractor 降级为零向量 ──────────────────────────────────────────

def test_map_intent_to_vector_no_extractor(cloud_with_edges):
    """无 extractor 时返回零向量，但强度正确计算。"""
    result = map_intent_to_vector(
        cloud_with_edges,
        extractor=None,
        base_strength=1.0,
        trust_score=0.8,
    )

    assert result is not None
    steering_vec, strength = result

    # 零向量
    assert steering_vec.norm == 0.0
    assert steering_vec.label == "zero_fallback"

    # 强度 = base_strength × edge_weight × trust_score
    # 权重最高的边是 task->goal，weight=0.9
    assert strength == pytest.approx(1.0 * 0.9 * 0.8, abs=1e-9)


# ── 场景 e：端到端映射（含强度计算）───────────────────────────────────────────

def test_map_intent_to_vector_strength_calculation(cloud_with_edges):
    """强度计算公式验证：base_strength × edge_weight × trust_score。"""
    result = map_intent_to_vector(
        cloud_with_edges,
        extractor=None,
        base_strength=2.0,
        trust_score=0.5,
    )

    assert result is not None
    _, strength = result

    # 权重最高的边是 task->goal，weight=0.9
    expected = 2.0 * 0.9 * 0.5
    assert strength == pytest.approx(expected, abs=1e-9)


def test_map_intent_to_vector_selects_highest_weight_edge(cloud_with_edges):
    """应选择权重最高的非锚点边。"""
    result = map_intent_to_vector(cloud_with_edges, extractor=None)
    assert result is not None
    _, strength = result
    # 权重最高的是 task->goal (0.9)，不是 self->task (0.8) 或 self->goal (0.6)
    assert strength == pytest.approx(0.9, abs=1e-9)


# ── 场景 f：节点语义查找 ──────────────────────────────────────────────────────

def test_get_node_semantic_from_shell(cloud_with_edges):
    """优先从 _shell 获取节点文本。"""
    # 添加一个 shell 节点
    from core.models import IntentNode
    node = IntentNode(id="shell_001", text="用户请求写诗")
    cloud_with_edges._shell["shell_001"] = node

    assert _get_node_semantic(cloud_with_edges, "shell_001") == "用户请求写诗"


def test_get_node_semantic_from_anchor(cloud_with_edges):
    """_shell 中不存在时从锚点获取 label。"""
    assert _get_node_semantic(cloud_with_edges, "self") == "系统自身"


def test_get_node_semantic_fallback_to_id(cloud_with_edges):
    """都不存在时回退到节点 ID。"""
    assert _get_node_semantic(cloud_with_edges, "unknown_node") == "unknown_node"


# ── 模板加载 ──────────────────────────────────────────────────────────────────

def test_load_templates_from_config():
    """从 config.yaml 加载模板。"""
    templates = load_templates()
    assert "refines" in templates
    assert "contrasts" in templates
    assert "evokes" in templates
    assert "constrains" in templates
    assert "positive" in templates["refines"]
    assert "negative" in templates["refines"]


# ── 对比对模板占位符填充 ──────────────────────────────────────────────────────

def test_template_format_with_special_chars(sample_templates):
    """模板占位符正确填充，包含特殊字符。"""
    pos, neg = generate_contrastive_pair(
        "refines",
        "存在主义哲学",
        "荒诞与自由",
        sample_templates,
    )
    assert "存在主义哲学" in pos
    assert "荒诞与自由" in pos
