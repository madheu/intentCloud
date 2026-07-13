"""常识加载器。

从 JSON 文件批量加载常识节点和边到意图云，不依赖 LLM，纯数据加载，可立即测试。

用法：
    cloud = IntentCloud()
    n_nodes, n_edges = load_common_sense(cloud, "data/common_sense.json")
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from core.models import IntentNode


def load_common_sense(
    cloud,
    data_path: str = "data/common_sense.json",
    model: nn.Module | None = None,
    tokenizer=None,
    layer_idx: int | None = None,
) -> tuple[int, int]:
    """从 JSON 文件加载常识节点和边到意图云。

    步骤：
      1. 读取 JSON 文件
      2. 对每个节点：
         a. 创建 IntentNode 并加入 cloud._shell
         b. 如果 JSON 中有 embedding，直接使用
         c. 如果 JSON 中没有 embedding 但提供了 model/tokenizer/layer_idx，实时查询
         d. 都没有则 llm_embedding = None（降级）
      3. 对每条边：
         a. 创建 CloudEdge 并加入 cloud._edges
         b. ref_weight = 初始 weight（作为锚定基准）
      4. 返回 (节点数, 边数)

    Args:
        cloud: IntentCloud 实例
        data_path: JSON 数据文件路径
        model: 用于实时查询 embedding 的 LLM 模型（可选）
        tokenizer: 对应的 tokenizer（可选）
        layer_idx: 目标层索引（可选）

    Returns:
        (节点数, 边数)

    Raises:
        FileNotFoundError: 数据文件不存在
        json.JSONDecodeError: JSON 格式错误
    """
    data_path = Path(data_path)

    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    from core.intent_cloud import CloudEdge

    # ── 步骤 2：加载节点 ──────────────────────────────────────────────────
    node_count = 0
    for node_data in data.get("nodes", []):
        node_id = node_data["id"]
        text = node_data.get("text", node_id)
        trust = node_data.get("trust", 0.5)

        # 处理 llm_embedding
        llm_embedding = None
        if "llm_embedding" in node_data and node_data["llm_embedding"] is not None:
            llm_embedding = torch.tensor(node_data["llm_embedding"])
        elif model is not None and tokenizer is not None and layer_idx is not None:
            from core.node_embedding import get_node_embedding
            llm_embedding = get_node_embedding(text, model, tokenizer, layer_idx)

        node = IntentNode(
            id=node_id,
            text=text,
            trust=trust,
            llm_embedding=llm_embedding,
        )
        cloud._shell[node_id] = node
        node_count += 1

    # ── 步骤 3：加载边 ────────────────────────────────────────────────────
    edge_count = 0
    for edge_data in data.get("edges", []):
        source = edge_data["source"]
        target = edge_data["target"]
        weight = edge_data.get("weight", 0.5)
        edge_type = edge_data.get("type", "connects")

        edge = CloudEdge(
            source_id=source,
            target_id=target,
            weight=weight,
            ref_weight=weight,
            prev_weight=weight,
            edge_type=edge_type,
        )
        cloud._edges[(source, target)] = edge
        edge_count += 1

    return node_count, edge_count