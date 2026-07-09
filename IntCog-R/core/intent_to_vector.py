"""意图云到引导向量的映射层。

将意图云的活跃状态（激活值 + 边权重）映射为可用于 LLM 注入的 steering_vector。

流程：
  1. 从意图云中找到权重最高的活跃非锚点边（意图路径）
  2. 根据边类型从配置中读取对比对模板
  3. 用模板填充节点的语义概念，生成正反例提示
  4. 复用 SteeringVectorExtractor 提取归一化方向向量
  5. 计算强度：final_strength = base_strength × edge_weight × trust_score

降级策略：
  - 无边时 → 返回零向量 + 零强度
  - 无 extractor 时 → 返回零向量 + 零强度
  - 缺失模板配置时 → 抛出 ValueError
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from core.config_loader import load_config
from core.intent_cloud import IntentCloud

if TYPE_CHECKING:
    from core.steering import SteeringVector, SteeringVectorExtractor


# ── 模板加载 ──────────────────────────────────────────────────────────────────

def load_templates(path: str | None = None) -> dict[str, dict[str, str]]:
    """从 config.yaml 加载对比对生成模板。

    配置格式：
      contrastive_templates:
        refines:
          positive: "深化{concept}的内涵..."
          negative: "对{concept}进行最浅层描述..."
        ...
    """
    cfg = load_config(path)
    templates = cfg.get("contrastive_templates", {})
    return templates


# ── 对比对生成 ─────────────────────────────────────────────────────────────────

def generate_contrastive_pair(
    edge_type: str,
    concept: str,
    target: str,
    templates: dict[str, dict[str, str]] | None = None,
) -> tuple[str, str]:
    """根据边类型和模板生成对比提示对。

    Args:
        edge_type: 边的语义类型（refines/contrasts/evokes/constrains/...）
        concept: 源节点的语义概念（填充 {concept}）
        target: 目标节点的语义概念（填充 {target}）
        templates: 模板字典，若为 None 则从 config.yaml 加载

    Returns:
        (正例提示, 反例提示)

    Raises:
        ValueError: 如果边类型在模板中不存在
    """
    if templates is None:
        templates = load_templates()

    template = templates.get(edge_type)
    if template is None:
        raise ValueError(
            f"No contrastive template for edge type '{edge_type}'. "
            f"Available types: {list(templates.keys())}"
        )

    positive = template["positive"].format(concept=concept, target=target)
    negative = template["negative"].format(concept=concept, target=target)

    return positive, negative


# ── 节点语义查找 ──────────────────────────────────────────────────────────────

def _get_node_semantic(cloud: IntentCloud, node_id: str) -> str:
    """获取节点的语义描述文本。

    查找顺序：
      1. 外壳节点（_shell）的 text 字段
      2. 锚点节点的 label 字段
      3. 回退到节点 ID 本身
    """
    # 1. 外壳节点
    if node_id in cloud._shell:
        return cloud._shell[node_id].text

    # 2. 锚点节点
    if node_id in cloud.anchor_system._nodes:
        return cloud.anchor_system._nodes[node_id].label

    # 3. 回退到 ID
    return node_id


# ── 意图到向量映射 ────────────────────────────────────────────────────────────

def map_intent_to_vector(
    cloud: IntentCloud,
    extractor: SteeringVectorExtractor | None = None,
    base_strength: float = 1.0,
    trust_score: float = 1.0,
) -> tuple[SteeringVector, float] | None:
    """将意图云的活跃状态映射为引导向量和注入强度。

    步骤：
      1. 找到权重最高的非锚点边（意图路径）
      2. 获取源/目标节点的语义文本
      3. 根据边类型生成对比提示对
      4. 用 extractor 提取 steering_vector
      5. 计算 final_strength = base_strength × edge_weight × trust_score

    Args:
        cloud: 意图云实例（应先调用 process_interaction 完成扩散和权重更新）
        extractor: SteeringVectorExtractor 实例（可为 None，降级为零向量）
        base_strength: 基础引导强度
        trust_score: 信任分数（通常来自节点的 trust 字段，默认 1.0）

    Returns:
        (SteeringVector, final_strength) 元组。
        如果没有活跃非锚点边，返回 None。
        如果没有 extractor，返回零向量和计算后的强度。
    """
    # ── 步骤 1：找到权重最高的非锚点边 ──────────────────────────────────
    active_edge: Any = None
    max_weight = -float("inf")

    for (src, tgt), edge in cloud._edges.items():
        # 跳过锚点边
        if not cloud.anchor_system.is_edge_mutable(src, tgt):
            continue
        if edge.weight > max_weight:
            max_weight = edge.weight
            active_edge = edge

    if active_edge is None:
        # 无活跃非锚点边，降级
        return None

    edge = active_edge

    # ── 步骤 2：获取节点语义文本 ────────────────────────────────────────
    concept = _get_node_semantic(cloud, edge.source_id)
    target = _get_node_semantic(cloud, edge.target_id)

    # ── 步骤 3：生成对比提示对 ──────────────────────────────────────────
    templates = load_templates()
    try:
        positive, negative = generate_contrastive_pair(
            edge.edge_type, concept, target, templates
        )
    except ValueError:
        # 如果边类型没有对应模板，使用默认的 "connects" 模板
        positive, negative = generate_contrastive_pair(
            "connects", concept, target, templates
        )

    # ── 步骤 4：提取 steering_vector ───────────────────────────────────
    # 延迟导入，避免在测试环境中 torch 不可用时报错
    from core.steering import SteeringVector

    if extractor is None:
        # 降级：零向量
        try:
            import torch
            zero_vec = torch.zeros(1)
        except ImportError:
            # 测试环境无 torch，使用 numpy 降级
            import numpy as np
            zero_vec = np.zeros(1)
        steering_vec = SteeringVector(zero_vec, label="zero_fallback")
    else:
        steering_vec = extractor.extract_steering_vector(
            positive_prompts=[positive],
            negative_prompts=[negative],
            label=f"{edge.edge_type}:{concept[:20]}",
        )

    # ── 步骤 5：计算强度 ────────────────────────────────────────────────
    final_strength = base_strength * edge.weight * trust_score

    return steering_vec, final_strength