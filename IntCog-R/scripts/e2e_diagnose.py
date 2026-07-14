#!/usr/bin/env python3
"""H11d: 注入诊断实验。

问题诊断：为什么 swap 测试失败了？
  - "悲伤" vs "快乐" 在 layer_idx=14 的 cos_sim = 0.9922
  - 说明短词的中层 hidden state 几乎没有语义区分度

本实验：
  1. 对比不同长度的文本在不同层的 embedding 差异
  2. 测试是否使用输入层 embedding 比中层更好
  3. 用真正的对比 pair（积极长文 vs 消极长文）做 swap
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class Tee:
    def __init__(self, terminal, file_handle):
        self.terminal = terminal
        self.file = file_handle
    def write(self, text):
        self.terminal.write(text)
        self.file.write(text)
        self.file.flush()
    def flush(self):
        self.terminal.flush()
        self.file.flush()

from core.node_embedding import get_node_embedding, _get_num_layers

def get_embedding(text: str, model, tokenizer, layer_idx: int) -> torch.Tensor:
    return get_node_embedding(text, model, tokenizer, layer_idx)

def load_model(model_name: str):
    print(f"[加载模型] {model_name}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"          用时 {time.time()-t0:.1f}s")
    print(f"          设备: {next(model.parameters()).device}")
    return model, tokenizer

def cos_sim(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="E:\\models\\Qwen2.5-7B-Instruct")
    parser.add_argument("--log-file", type=str, default=None)
    args = parser.parse_args()

    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(log_path, "w", encoding="utf-8")
        sys.stdout = Tee(sys.stdout, log_fh)
        sys.stderr = Tee(sys.stderr, log_fh)

    model, tokenizer = load_model(args.model)
    num_layers = _get_num_layers(model)  # 28
    hidden_dim = model.config.hidden_size  # 3584
    print(f"[准备] num_layers={num_layers}, hidden_dim={hidden_dim}")
    print()

    # ── 测试文本集合 ──────────────────────────────────────────────────────
    texts = [
        # 单个词
        ("悲伤", "短词-负面"),
        ("快乐", "短词-正面"),
        ("孤独", "短词-孤独"),
        ("幸福", "短词-幸福"),
        ("死亡", "短词-死亡"),
        ("爱",   "短词-爱"),
        # 短语
        ("我很悲伤，眼泪不停地流", "短语-悲伤"),
        ("我今天特别开心，阳光明媚", "短语-快乐"),
        ("一个人站在空旷的原野上", "短语-孤独"),
        ("温暖的阳光洒在我的脸上", "短语-温暖"),
        # 长句
        ("这是一个悲伤的故事。黑暗中，我独自哭泣，世界仿佛失去了所有的色彩。",
         "长句-悲伤"),
        ("这是一个快乐的故事。阳光下，我放声大笑，整个世界都充满了温暖和希望。",
         "长句-快乐"),
        ("我是一个孤独的旅人，在无边无际的沙漠中独自前行，看不到尽头。",
         "长句-孤独"),
        ("我是世界上最幸福的人，周围都是爱我的朋友和家人，每一天都充满希望。",
         "长句-幸福"),
    ]

    layers_to_test = [0, 4, 7, 14, 21, 27]
    print(f"{'='*80}")
    print(f"诊断：不同文本在不同层的 embedding 余弦相似度")
    print(f"{'='*80}")
    print()

    # 计算所有文本在所有层的 embedding
    results = {}
    for text, label in texts:
        results[label] = {}
        for layer in layers_to_test:
            emb = get_embedding(text, model, tokenizer, layer)
            # 确保 dtype 一致
            results[label][layer] = emb.to(dtype=torch.float32)
        print(f"  [{label}] \"{text[:30]}...\" 已计算")

    print()

    # 关键对比对
    pairs = [
        ("短词:悲伤 vs 快乐", "短词-负面", "短词-正面"),
        ("短词:孤独 vs 幸福", "短词-孤独", "短词-幸福"),
        ("短语:悲伤 vs 快乐", "短语-悲伤", "短语-快乐"),
        ("短语:孤独 vs 温暖", "短语-孤独", "短语-温暖"),
        ("长句:悲伤 vs 快乐", "长句-悲伤", "长句-快乐"),
        ("长句:孤独 vs 幸福", "长句-孤独", "长句-幸福"),
        ("短词 vs 短语:悲伤", "短词-负面", "短语-悲伤"),
        ("短语 vs 长句:悲伤", "短语-悲伤", "长句-悲伤"),
    ]

    print(f"{'─'*80}")
    print(f"对比表：cos_sim 随层数的变化")
    print(f"{'─'*80}")
    print(f"{'对比项':<30}", end="")
    for layer in layers_to_test:
        print(f"{'L'+str(layer):>8}", end="")
    print()
    print(f"{'─'*80}")

    for pair_name, label_a, label_b in pairs:
        print(f"{pair_name:<30}", end="")
        for layer in layers_to_test:
            emb_a = results[label_a][layer]
            emb_b = results[label_b][layer]
            sim = cos_sim(emb_a, emb_b)
            marker = " ***" if sim > 0.95 else "    "
            print(f"{sim:>7.4f}{marker}", end="")
        print()

    print()
    print("  *** = cos_sim > 0.95（几乎不可区分）")
    print()

    # ── 结论 ─────────────────────────────────────────────────────────────
    print(f"{'='*80}")
    print("结论分析")
    print(f"{'='*80}")
    print()
    print("1. 短词（悲伤/快乐）的中间层 embedding 高度相似 → 注入短词无效")
    print("2. 短词是否有更佳层？看哪层的 cos_sim 最低（区分度最高）")
    print("3. 长句的区分度是否显著高于短词？")
    print("4. 最佳注入层是哪一层？")

    # 找最佳层
    print()
    best_layer = None
    best_diff = 0
    for layer in layers_to_test:
        sim_short = cos_sim(results["短词-负面"][layer], results["短词-正面"][layer])
        sim_long = cos_sim(results["长句-悲伤"][layer], results["长句-快乐"][layer])
        diff = sim_long - sim_short
        print(f"  层 {layer:2d}: 短词 sim={sim_short:.4f}, 长句 sim={sim_long:.4f}, "
              f"长句改善={diff*100:.1f}%")
        if sim_long < sim_short and (best_layer is None or sim_long < best_diff):
            best_diff = sim_long
            best_layer = layer

    print()
    print(f"最佳注入层（长句区分度最高）: L{best_layer} (sim={best_diff:.4f})"
          if best_layer is not None else "无显著最佳层")

    if args.log_file:
        sys.stdout.flush()
        sys.stderr.flush()
        log_fh.close()
        print(f"\n日志已保存到: {args.log_file}")


if __name__ == "__main__":
    main()
