#!/usr/bin/env python3
"""H12: 双海波对照实验——验证拓扑演化驱动 LLM 生成方向。

核心：两个完全相同的海波，通过不同方向的交互演化，产生不同的边权重，
然后驱动同一个 Qwen 模型生成文本，看是否产生不同风格的输出。
"""

from __future__ import annotations
import sys, time, copy, json, math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class Tee:
    def __init__(self, t, f):
        self.t = t; self.f = f
    def write(self, s):
        self.t.write(s); self.f.write(s); self.f.flush()
    def flush(self):
        self.t.flush(); self.f.flush()

# ── 配置 ──────────────────────────────────────────────────────────────────
PROMPT = "写大海"
MAX_NEW = 300
EVOLVE_ROUNDS = 8
EVOLVE_STRENGTH = 0.8

# ── 模型加载 ──────────────────────────────────────────────────────────────
def load_model(model_name: str):
    print(f"[加载模型] {model_name}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb,
                                                  device_map="auto", trust_remote_code=True)
    model.eval()
    print(f"          用时 {time.time()-t0:.1f}s, device={next(model.parameters()).device}")
    return model, tokenizer

# ── 海波工具 ──────────────────────────────────────────────────────────────
def load_cloud(data_path: str, model, tokenizer):
    """加载常识数据并初始化海波实例。"""
    from core.intent_cloud import IntentCloud
    cloud = IntentCloud()
    layer_idx = model.config.num_hidden_layers // 2
    n, e = cloud.load_common_sense(str(data_path), model, tokenizer, layer_idx)
    return cloud, layer_idx

def evolve_edge(cloud, src_node: str, tgt_node: str, direction: str = "up",
                rounds: int = EVOLVE_ROUNDS, strength: float = EVOLVE_STRENGTH):
    """通过多次交互演化海波中一条边的权重。

    direction='up': 强化 src→tgt（同时激活两端）
    direction='down': 弱化 src→tgt（仅激活一端）
    """
    from core.intent_cloud_config import IntentCloudConfig
    cfg = IntentCloudConfig()

    print(f"  [演化] {src_node} → {tgt_node} ({direction}), {rounds}轮, 强度={strength}")

    for r in range(1, rounds + 1):
        # 构造输入信号
        if direction == "up":
            # 同时激活两端 → 赫布项增强
            signals = {src_node: strength, tgt_node: strength}
        else:
            # 弱化: 手动衰减边权重（参考拉回太慢，且扩散会使另一端也被激活）
            signals = {src_node: strength}  # 仅一端激活，避免共现增强
            edge = cloud._edges.get((src_node, tgt_node))
            if edge:
                edge.weight = max(0.0, edge.weight * 0.9)  # 每轮衰减10%

        # process_interaction = 激活扩散 → 权重更新
        converged = cloud.process_interaction(signals, cfg)

        # 读取关键边权重
        edge = cloud._edges.get((src_node, tgt_node))
        if edge:
            w = edge.weight
            src_act = converged.get(src_node, 0)
            tgt_act = converged.get(tgt_node, 0)
            print(f"    轮{r:2d}: 边权重={w:.4f}  |  src_act={src_act:.3f}  tgt_act={tgt_act:.3f}")

    # 最终状态
    edge = cloud._edges.get((src_node, tgt_node))
    if edge:
        print(f"  [结果] {src_node}→{tgt_node} 权重: {edge.weight:.4f} → {edge.weight:.4f}")


# ── 生成 ──────────────────────────────────────────────────────────────────
def generate_with_cloud(cloud, prompt: str, model, tokenizer,
                        scale: float = 3.0, max_new: int = MAX_NEW, seed: int = 42):
    """用海波的激活节点驱动 BilingualInjector 生成。"""
    from core.intent_cloud_config import IntentCloudConfig
    from core.bilingual_injector import BilingualInjector

    torch.manual_seed(seed)

    # 先对 prompt 做意图提取（关键词匹配）
    input_signals = {}
    for nid, node in cloud._shell.items():
        if node.text and any(kw in prompt for kw in [node.text[:2], node.text[:3]]):
            # 简单关键词匹配：prompt 中包含节点文本的前几个字
            continue  # 我们先手动激活 ocean 节点
        if nid == "nature_ocean":
            input_signals[nid] = 0.8

    # 如果 prompt 中包含了节点文本
    for nid, node in cloud._shell.items():
        if node.text and node.text[:2] in prompt:
            input_signals[nid] = 0.8

    if not input_signals:
        input_signals = {"nature_ocean": 0.8}

    # 激活扩散
    cfg = IntentCloudConfig()
    converged = cloud.process_interaction(input_signals, cfg)

    # 打印激活节点
    active = {k: v for k, v in sorted(converged.items(), key=lambda x: x[1], reverse=True)[:8]}
    print(f"  [激活] {len(active)}/{len(converged)} 节点:")
    for nid, v in active.items():
        nd = cloud._shell.get(nid)
        txt = nd.text if nd else nid
        print(f"         {nid}: {txt} = {v:.4f}")

    # 双语者注入生成
    injector = BilingualInjector(model, tokenizer, activation_threshold=0.0)
    output = injector.generate_with_injection(
        prompt, activations=converged, cloud=cloud,
        max_new_tokens=max_new, temperature=0.7, top_p=0.9,
    )
    return output.strip()


# ── 主流程 ────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="E:\\models\\Qwen2.5-7B-Instruct")
    parser.add_argument("--data", default="data/common_sense.json")
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--rounds", type=int, default=EVOLVE_ROUNDS)
    parser.add_argument("--scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.log_file:
        p = Path(args.log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        fh = open(p, "w", encoding="utf-8")
        sys.stdout = Tee(sys.stdout, fh)
        sys.stderr = Tee(sys.stderr, fh)

    model, tokenizer = load_model(args.model)

    print("=" * 70)
    print("H12: 双海波对照实验")
    print(f"Prompt: \"{PROMPT}\"")
    print(f"演化轮次: {args.rounds}, 注入强度: ×{args.scale}")
    print("=" * 70)
    print()

    # ── 1. 初始化两个相同的海波 ──────────────────────────────────────────
    print("[步骤1] 初始化海波 A 和 B ...")
    cloud_a, layer_idx = load_cloud(args.data, model, tokenizer)
    # 深拷贝海波 B（需要序列化再反序列化）
    import pickle
    cloud_b = pickle.loads(pickle.dumps(cloud_a))
    print(f"  海波 A: {len(cloud_a._shell)}节点, {len(cloud_a._edges)}边")
    print(f"  海波 B: {len(cloud_b._shell)}节点, {len(cloud_b._edges)}边")
    # 检查关键边的初始权重
    for src, tgt in [("nature_ocean", "emotion_awe"), ("nature_ocean", "emotion_fear")]:
        e = cloud_a._edges.get((src, tgt))
        print(f"  初始 {src}→{tgt}: weight={e.weight if e else 'N/A'}")
    print()

    # ── 2. 演化海波 A：强化大海→敬畏 ────────────────────────────────────
    print("[步骤2] 演化海波 A：强化 大海→敬畏，弱化 大海→恐惧")
    evolve_edge(cloud_a, "nature_ocean", "emotion_awe", "up", rounds=args.rounds)
    evolve_edge(cloud_a, "nature_ocean", "emotion_fear", "down", rounds=args.rounds)
    print()

    # ── 3. 演化海波 B：强化大海→恐惧 ────────────────────────────────────
    print("[步骤3] 演化海波 B：强化 大海→恐惧，弱化 大海→敬畏")
    evolve_edge(cloud_b, "nature_ocean", "emotion_fear", "up", rounds=args.rounds)
    evolve_edge(cloud_b, "nature_ocean", "emotion_awe", "down", rounds=args.rounds)
    print()

    # ── 4. 打印演化后的边权重对比 ────────────────────────────────────────
    print("[步骤4] 演化结果对比:")
    for src, tgt in [("nature_ocean", "emotion_awe"), ("nature_ocean", "emotion_fear")]:
        ea = cloud_a._edges.get((src, tgt))
        eb = cloud_b._edges.get((src, tgt))
        wa = ea.weight if ea else 0
        wb = eb.weight if eb else 0
        print(f"  {src}→{tgt}:  海波A={wa:.4f}  |  海波B={wb:.4f}")
    print()

    # ── 5. 生成对比文本 ──────────────────────────────────────────────────
    print("[步骤5] 生成对比文本 (seed固定，排除随机性)")
    print()

    # 每组生成 3 次，取平均观察
    for run in range(1, 4):
        seed = args.seed + run
        print(f"{'─' * 60}")
        print(f"[生成 Run {run}/3] seed={seed}")
        print(f"{'─' * 60}")

        # 海波 A 生成
        print(f"\n▸ 海波A (敬畏方向):")
        text_a = generate_with_cloud(cloud_a, PROMPT, model, tokenizer,
                                     scale=args.scale, seed=seed)

        # 海波 B 生成
        print(f"\n▸ 海波B (恐惧方向):")
        text_b = generate_with_cloud(cloud_b, PROMPT, model, tokenizer,
                                     scale=args.scale, seed=seed + 100)

        # 打印完整文本
        print(f"\n  ── 文本A (敬畏) ──")
        for line in text_a.split('\n')[:15]:
            print(f"  {line}")
        if len(text_a.split('\n')) > 15:
            print(f"  ...（共{len(text_a)}字）")

        print(f"\n  ── 文本B (恐惧) ──")
        for line in text_b.split('\n')[:15]:
            print(f"  {line}")
        if len(text_b.split('\n')) > 15:
            print(f"  ...（共{len(text_b)}字）")
        print()

    print("=" * 70)
    print("实验完成")
    print("=" * 70)

    if args.log_file:
        sys.stdout.flush(); sys.stderr.flush(); fh.close()
        print(f"\n日志: {args.log_file}")

if __name__ == "__main__":
    main()
