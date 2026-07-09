# [NEW] 真实 Activation Steering 演示 (GPT-2 + 对比对) - 2026-07-08
#
#!/usr/bin/env python3
"""真正的 Activation Steering — 使用对比对提取语义引导向量。

流程：
  1. 加载 GPT-2
  2. 通过对比提示对提取 steering vector（有语义的方向）
  3. 注入中间层
  4. 对比 无引导 vs 有引导 的生成

运行方式：
    python mvp_real_steering.py
    python mvp_real_steering.py --strength 3.0 --prompt "Once upon a time"
"""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.steering import (
    SteeringInjector,
    SteeringVector,
    SteeringVectorExtractor,
    compute_cosine_shift,
)


# ── 1. 加载 GPT-2 ───────────────────────────────────────────────────────────

def load_gpt2():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("[模型] 加载 GPT-2 (124M)...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()
    n_layers = model.config.n_layer
    d_model = model.config.hidden_size
    print(f"  ✓ 层数={n_layers}, hidden={d_model}, "
          f"参数={sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    return model, tokenizer


# ── 2. 生成函数 ──────────────────────────────────────────────────────────────

def generate_text(model, tokenizer, prompt, max_new=50, temperature=0.8):
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new, do_sample=True,
            temperature=temperature, top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            output_logits=True, return_dict_in_generate=True,
        )
    text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    if outputs.logits:
        last_logits = outputs.logits[-1][0]
    else:
        last_logits = torch.zeros(model.config.vocab_size)
    return text, last_logits


# ── 3. 对比对定义 ──────────────────────────────────────────────────────────

STEERING_PAIRS = {
    "creative": {
        "label": "创意/想象力方向",
        "positive": [
            "The ocean is a vast, mysterious world full of ancient secrets and hidden treasures.",
            "The waves whispered stories of distant lands as they kissed the shore.",
            "Deep beneath the surface, colorful coral cities teem with exotic life.",
        ],
        "negative": [
            "The ocean is a large body of salt water that covers most of the Earth.",
            "Water is composed of two hydrogen atoms and one oxygen atom.",
            "The average depth of the ocean is about 3688 meters according to scientific measurements.",
        ],
    },
    "poetic": {
        "label": "诗歌/浪漫方向",
        "positive": [
            "The moonlight danced upon the gentle waves, painting silver pathways to infinity.",
            "Each sunset paints the horizon in hues of gold and crimson over the endless sea.",
            "The ocean's heartbeat echoes in every shell that washes ashore.",
        ],
        "negative": [
            "The ocean is used for commercial shipping and transportation of goods.",
            "Fishing is a major industry that depends on ocean resources.",
            "Ocean currents affect global weather patterns and climate systems.",
        ],
    },
    "safety_careful": {
        "label": "安全/谨慎方向",
        "positive": [
            "One must always be careful near the ocean and respect its power.",
            "Before entering the water, check weather conditions and warning signs.",
            "The ocean can be dangerous; always swim at patrolled beaches.",
        ],
        "negative": [
            "Jump into the ocean without looking back.",
            "Ignore all warnings and swim far out into the sea.",
            "The ocean is completely safe and nothing bad ever happens.",
        ],
    },
}


# ── 4. 对比对向量提取 ──────────────────────────────────────────────────────

def extract_contrastive_vector(model, tokenizer, layer_idx,
                               positive_prompts, negative_prompts,
                               label=""):
    """使用 mean(act(正例)) - mean(act(反例)) 提取 steering vector。"""
    extractor = SteeringVectorExtractor(model, tokenizer, layer_idx)

    # 分别提取正例和反例的激活
    pos_acts = []
    for p in positive_prompts:
        act = extractor.extract_activation(p)
        pos_acts.append(act)

    neg_acts = []
    for n in negative_prompts:
        act = extractor.extract_activation(n)
        neg_acts.append(act)

    pos_mean = torch.stack(pos_acts).mean(dim=0)
    neg_mean = torch.stack(neg_acts).mean(dim=0)
    direction = pos_mean - neg_mean

    norm = torch.norm(direction)
    if norm > 0:
        direction = direction / norm

    sv = SteeringVector(direction, label=label)
    return sv, {
        "pos_norm": torch.norm(torch.stack(pos_acts).mean(dim=0)).item(),
        "neg_norm": torch.norm(torch.stack(neg_acts).mean(dim=0)).item(),
        "diff_norm": norm.item(),
        "mean_shift": direction.mean().item(),
        "std_shift": direction.std().item(),
    }


# ── 5. 主流程 ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="真正的对比对 Activation Steering")
    parser.add_argument("--prompt", default="The ocean is", help="生成提示")
    parser.add_argument("--strength", type=float, default=3.0, help="引导强度")
    parser.add_argument("--layer", type=int, default=6, help="注入层 (GPT-2: 0-11)")
    parser.add_argument("--mode", choices=list(STEERING_PAIRS.keys()),
                        default="creative", help="引导方向")
    args = parser.parse_args()

    model, tokenizer = load_gpt2()
    n_layers = model.config.n_layer
    layer_idx = min(args.layer, n_layers - 1)
    pair = STEERING_PAIRS[args.mode]

    print(f"\n[配置] 注入层: layer_{layer_idx}, 强度: {args.strength}, "
          f"方向: {pair['label']}")
    print(f"\n[蓝图] 引导方向: {pair['label']}")
    print(f"       正例: {pair['positive'][0][:60]}...")
    print(f"       反例: {pair['negative'][0][:60]}...")

    # ── 对比对向量提取 ──
    print(f"\n[提取] 计算 steering_vector = mean(正例) - mean(反例)...")
    steering_vector, stats = extract_contrastive_vector(
        model, tokenizer, layer_idx,
        pair["positive"], pair["negative"], label=pair["label"],
    )
    summary = steering_vector.summary(top_k=5)
    print(f"       正例均值范数: {stats['pos_norm']:.4f}")
    print(f"       反例均值范数: {stats['neg_norm']:.4f}")
    print(f"       方向向量范数: {stats['diff_norm']:.4f}")
    print(f"       前5维: {summary['top_dims']}")

    # ── 基线生成 ──
    print(f"\n{'='*70}")
    print("  📝 [BASELINE] 无引导生成")
    print("="*70)
    text_base, logits_base = generate_text(model, tokenizer, args.prompt)
    print(f"  prompt: {args.prompt}")
    print(f"  output: {text_base}")

    # ── 引导生成 ──
    print(f"\n  📝 [STEERED] 有引导生成 ({pair['label']})")
    print("="*70)
    injector = SteeringInjector(model, layer_idx)
    injector.inject(steering_vector, strength=args.strength)
    text_steer, logits_steer = generate_text(model, tokenizer, args.prompt)
    injector.remove()
    print(f"  prompt: {args.prompt}")
    print(f"  output: {text_steer}")

    # ── Logits 对比 ──
    cos_sim = compute_cosine_shift(logits_base, logits_steer)
    print(f"\n{'='*70}")
    print("  📊 Logits 对比")
    print("="*70)
    print(f"  余弦相似度: {cos_sim:.6f}")
    print(f"  语义偏移:   {(1-cos_sim)*100:.2f}%")

    # ── 所有层扫描 ──
    print(f"\n{'='*70}")
    print("  🔬 多层注入效果扫描")
    print("="*70)
    for tl in range(min(12, n_layers)):
        ti = SteeringInjector(model, tl)
        ti.inject(steering_vector, strength=args.strength)
        _, tl_logits = generate_text(model, tokenizer, args.prompt, max_new=10)
        ti.remove()
        tc = compute_cosine_shift(logits_base, tl_logits)
        shift = (1 - tc) * 100
        bar = "█" * int(shift * 10)
        print(f"  layer_{tl:2d}: cos={tc:.4f}  shift={shift:.2f}%  {bar}")

    print(f"\n{'='*70}")
    print("  ✅ 对比对 Activation Steering 演示完成")
    print("="*70)


if __name__ == "__main__":
    main()
