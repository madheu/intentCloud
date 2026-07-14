#!/usr/bin/env python3
"""H11e: 注入方式对比实验。

三合一验证：
  1. 修复随机噪声 dtype bug → 确认扰动 vs 方向引导
  2. 对比 Embedding 层注入 (BilingualInjector) vs 中间层注入 (SteeringInjector)
  3. 使用短语节点 (修复后的 common_sense.json)
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
    def __init__(self, terminal, fh):
        self.terminal = terminal
        self.fh = fh
    def write(self, s):
        self.terminal.write(s)
        self.fh.write(s)
        self.fh.flush()
    def flush(self):
        self.terminal.flush()
        self.fh.flush()


PROMPT = "写一首关于孤独的诗"

# 两个方向：使用节点 ID（短语文本会在加载时自动取 embedding）
SAD_DIRECTION = {"emotion_sadness": 1.0}   # "深深的悲伤和失落感"
JOY_DIRECTION = {"emotion_joy": 1.0}      # "发自内心的喜悦和快乐"


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
    print(f"          用时 {time.time()-t0:.1f}s, device={next(model.parameters()).device}")
    return model, tokenizer


def load_cloud(model, tokenizer, data_path: str):
    """加载 IntentCloud + 短语常识数据。"""
    from core.intent_cloud import IntentCloud
    cloud = IntentCloud()
    layer_idx = model.config.num_hidden_layers // 2
    p = Path(data_path)
    n, e = cloud.load_common_sense(str(p), model, tokenizer, layer_idx)
    print(f"  常识: {n}节点, {e}边, layer_idx={layer_idx}")
    return cloud, layer_idx


# ── 方法1: BilingualInjector（embedding 层虚拟 token prepend）───────────────

def gen_bilingual(model, tokenizer, cloud, prompt: str,
                  activations: dict[str, float] | None,
                  scale: float = 1.0, max_new: int = 300, seed: int = 42) -> str:
    from core.bilingual_injector import BilingualInjector
    torch.manual_seed(seed)
    injector = BilingualInjector(model, tokenizer, activation_threshold=0.0)

    if activations is None:
        # 基线：直接生成
        device = next(model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(input_ids=inputs["input_ids"],
                                 max_new_tokens=max_new,
                                 temperature=0.7, top_p=0.9, do_sample=True,
                                 pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # 缩放激活值
    scaled = {k: v * scale for k, v in activations.items()}

    # 激活扩散（为了从 cloud 中拿到 node embedding）
    from core.intent_cloud_config import IntentCloudConfig
    converged = cloud.process_interaction(scaled)

    # 注入生成
    return injector.generate_with_injection(
        prompt, activations=converged, cloud=cloud,
        max_new_tokens=max_new, temperature=0.7, top_p=0.9)


def gen_bilingual_noise(model, tokenizer, cloud, prompt: str,
                        num_tokens: int = 3, hidden_dim: int = 3584,
                        scale: float = 3.0, max_new: int = 300, seed: int = 42) -> str:
    """随机噪声注入（embedding 层），修复 dtype bug。"""
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    # 关键修复：使用 model.dtype（bfloat16）
    model_dtype = next(model.parameters()).dtype
    noise = torch.randn(num_tokens, hidden_dim, device=device, dtype=model_dtype) * 0.1 * scale

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    embed = model.get_input_embeddings()(input_ids)  # [1, seq, dim]

    combined = torch.cat([noise.unsqueeze(0), embed], dim=1)
    attn = torch.ones((1, combined.shape[1]), dtype=torch.long, device=device)

    with torch.no_grad():
        out = model.generate(inputs_embeds=combined, attention_mask=attn,
                             max_new_tokens=max_new, temperature=0.7,
                             top_p=0.9, do_sample=True,
                             pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][num_tokens + input_ids.shape[1]:], skip_special_tokens=True)


# ── 方法2: SteeringInjector（中间层 hidden state 注入）───────────────────────

def gen_steering(model, tokenizer, cloud, prompt: str,
                 activations: dict[str, float] | None,
                 layer_idx: int = 14, scale: float = 1.0,
                 max_new: int = 300, seed: int = 42) -> str:
    """用 SteeringInjector 在中间层注入。"""
    from core.steering import SteeringInjector, SteeringVector
    torch.manual_seed(seed)

    if activations is None:
        # 基线
        device = next(model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(input_ids=inputs["input_ids"],
                                 max_new_tokens=max_new, temperature=0.7,
                                 top_p=0.9, do_sample=True,
                                 pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # 构造 steering vector：取激活节点 embedding 的加权平均
    vectors = []
    total_act = 0.0
    for node_id, act in activations.items():
        node = cloud._shell.get(node_id)
        if node is None or node.llm_embedding is None:
            continue
        emb = node.llm_embedding
        if not isinstance(emb, torch.Tensor):
            emb = torch.tensor(emb)
        vectors.append(emb * act)
        total_act += act

    if not vectors:
        return "(无有效节点)"

    sv = torch.stack(vectors).sum(dim=0) / max(total_act, 1e-8)
    # 修复: L2归一化，使注入向量尺度与hidden states匹配
    sv_norm = torch.norm(sv)
    if sv_norm > 0:
        sv = sv / sv_norm * (sv.shape[0] ** 0.5)
    steering_vec = SteeringVector(sv, label="方向引导")

    # 注入并生成
    injector = SteeringInjector(model, layer_idx)
    try:
        injector.inject(steering_vec, strength=scale * 0.05)
        device = next(model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(input_ids=inputs["input_ids"],
                                 max_new_tokens=max_new, temperature=0.7,
                                 top_p=0.9, do_sample=True,
                                 pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    finally:
        injector.remove()


def gen_steering_noise(model, tokenizer, prompt: str,
                       hidden_dim: int = 3584, layer_idx: int = 14,
                       scale: float = 3.0, max_new: int = 300, seed: int = 42) -> str:
    """随机噪声在中间层注入（对照）。"""
    from core.steering import SteeringInjector, SteeringVector
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    # 修复 dtype
    model_dtype = next(model.parameters()).dtype
    noise = torch.randn(hidden_dim, device=device, dtype=model_dtype) * 0.1
    noise_vec = SteeringVector(noise, label="随机噪声")

    injector = SteeringInjector(model, layer_idx)
    try:
        injector.inject(noise_vec, strength=scale)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(input_ids=inputs["input_ids"],
                                 max_new_tokens=max_new, temperature=0.7,
                                 top_p=0.9, do_sample=True,
                                 pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    finally:
        injector.remove()


# ── 主流程 ─────────────────────────────────────────────────────────────────

def run_condition(label: str, gen_fn, *args, **kwargs) -> str:
    print(f"  [{label}]", end=" ", flush=True)
    t0 = time.time()
    out = gen_fn(*args, **kwargs)
    elapsed = time.time() - t0
    print(f"({elapsed:.0f}s)")
    for line in out.strip().split('\n')[:8]:
        print(f"    {line}")
    if len(out.strip().split('\n')) > 8:
        print(f"    ...（共 {len(out)} 字）")
    print()
    return out


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="E:\\models\\Qwen2.5-7B-Instruct")
    parser.add_argument("--data", default="data/common_sense.json")
    parser.add_argument("--max-new", type=int, default=300)
    parser.add_argument("--log-file", type=str, default=None)
    args = parser.parse_args()

    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(log_path, "w", encoding="utf-8")
        sys.stdout = Tee(sys.stdout, log_fh)
        sys.stderr = Tee(sys.stderr, log_fh)

    model, tokenizer = load_model(args.model)
    cloud, layer_idx = load_cloud(model, tokenizer, args.data)
    hidden_dim = model.config.hidden_size
    print(f"  hidden_dim={hidden_dim}")
    print()

    print("=" * 70)
    print(f"注入方式对比实验")
    print(f"Prompt: \"{PROMPT}\"")
    print(f"数据: 短语节点 (修复版 common_sense.json)")
    print("=" * 70)
    print()

    # ── 实验矩阵 ──────────────────────────────────────────────────────────
    experiments = [
        # (label, 注入方式, 方向, scale)
        # ---- 方法 A: Embedding 层注入 (BilingualInjector) ----
        ("A1-Base",     "bilingual", None,       1),
        ("A2-悲伤×1",   "bilingual", SAD_DIRECTION, 1),
        ("A2-悲伤×5",   "bilingual", SAD_DIRECTION, 5),
        ("A3-快乐×1",   "bilingual", JOY_DIRECTION, 1),
        ("A3-快乐×5",   "bilingual", JOY_DIRECTION, 5),
        ("A4-噪声×3",   "bilingual-noise", None,  3),

        # ---- 方法 B: 中间层注入 (SteeringInjector) ----
        ("B1-Base",     "steering", None,       1),
        ("B2-悲伤×1",   "steering", SAD_DIRECTION, 1),
        ("B2-悲伤×5",   "steering", SAD_DIRECTION, 5),
        ("B3-快乐×1",   "steering", JOY_DIRECTION, 1),
        ("B3-快乐×5",   "steering", JOY_DIRECTION, 5),
        ("B4-噪声×3",   "steering-noise", None, 3),
    ]

    for label, method, direction, scale in experiments:
        print(f"{'─' * 70}")
        print(f"【{label}】 method={method}, scale={scale}")

        if method == "bilingual":
            run_condition(label, gen_bilingual,
                          model, tokenizer, cloud, PROMPT,
                          activations=direction, scale=scale,
                          max_new=args.max_new, seed=42)

        elif method == "bilingual-noise":
            run_condition(label, gen_bilingual_noise,
                          model, tokenizer, cloud, PROMPT,
                          num_tokens=3, hidden_dim=hidden_dim,
                          scale=scale, max_new=args.max_new, seed=42)

        elif method == "steering":
            run_condition(label, gen_steering,
                          model, tokenizer, cloud, PROMPT,
                          activations=direction, layer_idx=layer_idx,
                          scale=scale, max_new=args.max_new, seed=42)

        elif method == "steering-noise":
            run_condition(label, gen_steering_noise,
                          model, tokenizer, PROMPT,
                          hidden_dim=hidden_dim, layer_idx=layer_idx,
                          scale=scale, max_new=args.max_new, seed=42)

    print("=" * 70)
    print("实验结束")
    print("=" * 70)
    print()
    print(f"Prompt: \"{PROMPT}\"")
    print(f"条件数: {len(experiments)}")

    if args.log_file:
        sys.stdout.flush()
        sys.stderr.flush()
        log_fh.close()
        print(f"日志: {args.log_file}")


if __name__ == "__main__":
    main()
