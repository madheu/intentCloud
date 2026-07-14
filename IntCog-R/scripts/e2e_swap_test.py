#!/usr/bin/env python3
"""H11c: swap 注入对比实验——因果检验。

核心思路：固定 prompt，只换注入的 embedding 内容，看输出是否跟随变化。

实验设计：
  条件 A：无注入（基线）
  条件 B：注入 "悲伤" 的 embedding（正常双语者注入）
  条件 C：注入 "快乐" 的 embedding（把"悲伤"节点换成"快乐"的 embedding）
  条件 D：注入 "随机噪声"（对照——如果也有效果说明是 token 数量起作用而非语义）

如果条件 B → 悲伤风格，条件 C → 快乐风格，条件 D → 乱码/无意义，
就证明了：是海波的 embedding 在控制方向，不是 LLM 在随机采样。

更进一步：如果 B/C/D 三样都差不多，说明注入太弱，需要调参。
所以这个实验同时测试 ×1、×5、×10 三个强度等级。
"""

from __future__ import annotations

import sys
import time
import json
import math
from pathlib import Path
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# ── 日志 ──────────────────────────────────────────────────────────────────

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


# ── 模型加载 ──────────────────────────────────────────────────────────────

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
    hidden_dim = model.config.hidden_size
    print(f"          hidden_dim: {hidden_dim}")
    return model, tokenizer, hidden_dim


# ── 核心：获取 token 的中间层 embedding ──────────────────────────────────

from core.node_embedding import get_node_embedding, _get_num_layers


def get_text_embedding(text: str, model, tokenizer, layer_idx: int) -> torch.Tensor:
    """获取一段文本在 LLM 中间层的 hidden state embedding。"""
    return get_node_embedding(text, model, tokenizer, layer_idx)


# ── 手动注入生成（绕过 IntentCloud，直接构造虚拟 token）───────────────────

def inject_and_generate(
    model, tokenizer,
    prompt: str,
    virtual_embeds: torch.Tensor | None,
    scale: float = 1.0,
    max_new_tokens: int = 300,
    temperature: float = 0.7,
    seed: int = 42,
) -> str:
    """带虚拟 token 注入的生成。scale 控制注入强度。"""
    torch.manual_seed(seed)
    device = next(model.parameters()).device

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    embed_layer = model.get_input_embeddings()
    input_embeds = embed_layer(input_ids)  # [1, seq_len, hidden_dim]

    if virtual_embeds is None:
        # 无注入
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)

    # 强度缩放
    scaled = virtual_embeds * scale

    # 可选的：重复 token 以增强信号
    num_virtual = scaled.shape[0]
    combined_embeds = torch.cat([scaled.unsqueeze(0), input_embeds], dim=1)
    total_len = combined_embeds.shape[1]
    attention_mask = torch.ones((1, total_len), dtype=torch.long, device=device)

    # 生成
    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    real_start = num_virtual + input_ids.shape[1]
    return tokenizer.decode(outputs[0][real_start:], skip_special_tokens=True)


# ── 实验 ──────────────────────────────────────────────────────────────────

PROMPT = "写一首关于孤独的诗"

@dataclass
class Condition:
    name: str
    description: str
    virtual_embeds: torch.Tensor | None  # None = 无注入


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="H11c: swap 注入对比实验")
    parser.add_argument("--model", default="E:\\models\\Qwen2.5-7B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--num-runs", type=int, default=2,
                        help="每个条件重复次数（固定种子不同值，排除随机性干扰）")
    parser.add_argument("--scales", type=str, default="1,5,10",
                        help="注入强度倍数，逗号分隔")
    args = parser.parse_args()

    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(log_path, "w", encoding="utf-8")
        sys.stdout = Tee(sys.stdout, log_fh)
        sys.stderr = Tee(sys.stderr, log_fh)

    scales = [float(s) for s in args.scales.split(",")]

    # ── 加载模型 ──────────────────────────────────────────────────────────
    model, tokenizer, hidden_dim = load_model(args.model)
    layer_idx = model.config.num_hidden_layers // 2  # 14
    print(f"[准备] layer_idx={layer_idx}, prompt=\"{PROMPT}\"")
    print(f"        scales={scales}, runs per condition={args.num_runs}")
    print()

    # ── 预计算三种 embedding ──────────────────────────────────────────────
    print("[计算注入 embedding ...]")
    emb_sadness = get_text_embedding("悲伤", model, tokenizer, layer_idx)  # [hidden_dim]
    emb_joy = get_text_embedding("快乐", model, tokenizer, layer_idx)      # [hidden_dim]
    emb_random = torch.randn(hidden_dim, device=emb_sadness.device)
    emb_random = emb_random / emb_random.norm() * 0.1  # 归一化到小幅度

    # 形状检查
    print(f"  悲伤 embedding: shape={emb_sadness.shape}, norm={emb_sadness.norm():.4f}")
    print(f"  快乐 embedding: shape={emb_joy.shape}, norm={emb_joy.norm():.4f}")
    print(f"  随机 embedding: norm={emb_random.norm():.4f}")

    # 计算相似度
    cos_sim = torch.nn.functional.cosine_similarity(emb_sadness, emb_joy, dim=0)
    print(f"  悲伤 vs 快乐 cos_sim: {cos_sim.item():.4f}")
    print()

    # ── 构建条件矩阵 ─────────────────────────────────────────────────────
    # 条件 × 强度 矩阵
    cond_templates = [
        ("A-基线", "无注入", None),
        ("B-悲伤", "注入「悲伤」embedding", emb_sadness),
        ("C-快乐", "注入「快乐」embedding（swap）", emb_joy),
        ("D-随机", "注入「随机噪声」embedding（对照）", emb_random),
    ]

    print("=" * 70)
    print(f"Swap 注入对比实验")
    print(f"Prompt: \"{PROMPT}\"")
    print(f"种子: 在每个条件下用 run{{1..N}} 种子")
    print("=" * 70)
    print()

    for cond_id, cond_desc, embed in cond_templates:
        print(f"{'─' * 70}")
        print(f"【{cond_id}】{cond_desc}")
        print()

        for scale in scales:
            print(f"  强度 ×{scale}:", end="")

            for run in range(1, args.num_runs + 1):
                seed = run * 100  # 100, 200 等不同种子
                t0 = time.time()

                if embed is None:
                    vb = None
                else:
                    vb = embed.clone().unsqueeze(0)  # [1, hidden_dim]

                output = inject_and_generate(
                    model, tokenizer, PROMPT, vb,
                    scale=scale, max_new_tokens=args.max_new_tokens,
                    seed=seed,
                )
                elapsed = time.time() - t0

                first_line = output.strip().split('\n')[0][:60]
                word_count = len(output)
                print(f" run{run}({elapsed:.0f}s)", end="")

            print()

            # 用最后一个 run 的结果展示完整内容
            print(f"  {'─' * 50}")
            for line in output.strip().split('\n')[:12]:
                print(f"  {line}")
            if len(output.strip().split('\n')) > 12:
                print(f"  ...（共 {word_count} 字）")
            print()

    # ── 关键对比：同一强度 ×5 下三个 embedding 的差异 ────────────────────
    ref_scale = 5
    print(f"{'=' * 70}")
    print(f"关键对比：强度 ×{ref_scale} 下，固定种子，只换 embedding")
    print(f"{'=' * 70}")
    print()

    fixed_seed = 42
    for cond_id, cond_desc, embed in cond_templates:
        print(f"【{cond_id}】{cond_desc} (×{ref_scale}, seed={fixed_seed}):")

        if embed is None:
            vb = None
        else:
            vb = embed.clone().unsqueeze(0)

        output = inject_and_generate(
            model, tokenizer, PROMPT, vb,
            scale=ref_scale, max_new_tokens=args.max_new_tokens,
            seed=fixed_seed,
        )

        for line in output.strip().split('\n')[:10]:
            print(f"  {line}")
        if len(output.strip().split('\n')) > 10:
            print(f"  ...（共 {len(output)} 字）")
        print()

    if args.log_file:
        sys.stdout.flush()
        sys.stderr.flush()
        log_fh.close()
        print(f"\n日志已保存到: {args.log_file}", file=sys.__stdout__)


if __name__ == "__main__":
    main()
