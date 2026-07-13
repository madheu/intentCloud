#!/usr/bin/env python3
"""H10.2：常识生成脚本。

一次性脚本，用 LLM 生成一批常识概念 + 关系，查询每个概念的中间层
embedding，保存为 data/common_sense.json 供 H10.1 的加载器使用。

用法：
    python scripts/inject_common_sense.py [--model MODEL_NAME] [--layer-idx N]

默认使用 microsoft/phi-2（~2.7B，适合 CPU/低显存场景）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.node_embedding import get_node_embedding

# ── LLM Prompt ────────────────────────────────────────────────────────────────

COMMON_SENSE_PROMPT = """你是一个语义网络构建助手。请生成一批基础常识概念及其关系。

要求：
1. 生成 20-30 个基础概念节点，覆盖以下类别：
   - 时间（如：过去、现在、未来、永恒）
   - 空间（如：这里、那里、远方、边界）
   - 情感（如：喜悦、悲伤、愤怒、平静）
   - 逻辑（如：因果、对比、递进、转折）
   - 社交（如：合作、冲突、信任、怀疑）
   - 自然（如：天空、海洋、生命、变化）

2. 为每对有关系的概念生成边，标注关系类型：
   - connects / refines / contrasts / evokes / constrains

3. 输出格式（JSON），直接输出 JSON，不要解释。

输出格式示例：
{
  "nodes": [
    {"id": "time_past", "text": "过去", "category": "时间"},
    {"id": "time_future", "text": "未来", "category": "时间"}
  ],
  "edges": [
    {"source": "time_past", "target": "time_future", "type": "contrasts", "weight": 0.2}
  ]
}

注意：
- 每个节点的 id 使用 {category}_{pinyin_or_english} 格式（如 time_past, emotion_joy）
- 边权重按类型：contrasts 0.1~0.3, connects 0.3~0.5, refines 0.5~0.7, evokes 0.2~0.4, constrains 0.4~0.6
- 直接输出 JSON，不要有任何其他文字
"""

# ── 权重映射 ──────────────────────────────────────────────────────────────────

WEIGHT_RANGES = {
    "contrasts": (0.1, 0.3),
    "connects": (0.3, 0.5),
    "refines": (0.5, 0.7),
    "evokes": (0.2, 0.4),
    "constrains": (0.4, 0.6),
}


def _normalize_weight(weight: float, edge_type: str) -> float:
    """将 LLM 返回的权重裁剪到该类型的合理范围。"""
    lo, hi = WEIGHT_RANGES.get(edge_type, (0.2, 0.6))
    return max(lo, min(hi, weight))


def _parse_json_response(text: str) -> dict:
    """从 LLM 返回的文本中提取 JSON，容错处理。

    支持：
      - 纯 JSON 文本
      - markdown ```json ... ``` 包裹
      - 前后有额外文字
    """
    text = text.strip()

    # 尝试提取 markdown code block
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start)
        text = text[start:end].strip()
    elif "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        text = text[start:end].strip()

    # 尝试找到第一个 { 和最后一个 }
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1:
        text = text[brace_start : brace_end + 1]

    return json.loads(text)


def _load_model(model_name: str):
    """加载 LLM 模型和 tokenizer。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"加载模型: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print(f"  模型加载完成, device={next(model.parameters()).device}")
    return model, tokenizer


def _generate_concepts(model, tokenizer) -> dict:
    """向 LLM 发送 prompt 生成常识概念。"""
    print("生成常识概念...")
    messages = [{"role": "user", "content": COMMON_SENSE_PROMPT}]

    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=2048,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs.shape[1] :], skip_special_tokens=True)
    print(f"  LLM 返回 {len(response)} 字符")

    data = _parse_json_response(response)
    print(f"  解析到 {len(data.get('nodes', []))} 个节点, {len(data.get('edges', []))} 条边")
    return data


def _query_embeddings(
    data: dict,
    model,
    tokenizer,
    layer_idx: int,
) -> dict:
    """对每个节点查询中间层 embedding。"""
    nodes = data.get("nodes", [])
    print(f"查询 embedding (layer_idx={layer_idx})...")

    for i, node in enumerate(nodes):
        text = node["text"]
        print(f"  [{i + 1}/{len(nodes)}] {node['id']}: {text} ...", end=" ", flush=True)
        try:
            emb = get_node_embedding(text, model, tokenizer, layer_idx)
            node["llm_embedding"] = emb.tolist()
            print(f"ok ({len(node['llm_embedding'])} dims)")
        except Exception as e:
            print(f"失败: {e}")
            node["llm_embedding"] = None

    return data


def _assign_weights(data: dict) -> dict:
    """为边分配或规范化权重。"""
    edges = data.get("edges", [])
    for edge in edges:
        edge_type = edge.get("type", "connects")
        if "weight" not in edge:
            lo, hi = WEIGHT_RANGES.get(edge_type, (0.3, 0.5))
            edge["weight"] = round((lo + hi) / 2, 2)
        else:
            edge["weight"] = _normalize_weight(edge["weight"], edge_type)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="H10.2: 常识生成脚本"
    )
    parser.add_argument(
        "--model",
        default="microsoft/phi-2",
        help="LLM 模型名称 (默认: microsoft/phi-2)",
    )
    parser.add_argument(
        "--layer-idx",
        type=int,
        default=None,
        help="embedding 查询的目标层索引 (默认: num_hidden_layers // 2)",
    )
    parser.add_argument(
        "--output",
        default="data/common_sense.json",
        help="输出文件路径 (默认: data/common_sense.json)",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="跳过 embedding 查询（仅生成 JSON 结构）",
    )
    args = parser.parse_args()

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 步骤 1：加载模型
    model, tokenizer = _load_model(args.model)

    # 步骤 2：生成概念
    data = _generate_concepts(model, tokenizer)
    data["version"] = "1.0"

    # 步骤 3：为边分配权重
    data = _assign_weights(data)

    # 步骤 4：查询 embedding
    if not args.skip_embeddings:
        layer_idx = args.layer_idx
        if layer_idx is None:
            layer_idx = _get_num_layers(model) // 2
        data = _query_embeddings(data, model, tokenizer, layer_idx)

    # 步骤 5：保存
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n已保存到 {output_path}")
    print(f"  节点: {len(data.get('nodes', []))}")
    print(f"  边:   {len(data.get('edges', []))}")
    print(f"  带 embedding 的节点: {sum(1 for n in data.get('nodes', []) if n.get('llm_embedding'))}")

    # 验证：用加载器加载
    print("\n验证加载...")
    from core.intent_cloud import IntentCloud
    cloud = IntentCloud()
    n_nodes, n_edges = cloud.load_common_sense(str(output_path))
    print(f"  加载了 {n_nodes} 个节点, {n_edges} 条边")


def _get_num_layers(model) -> int:
    """获取模型层数。"""
    from core.node_embedding import _get_num_layers
    return _get_num_layers(model)


if __name__ == "__main__":
    main()