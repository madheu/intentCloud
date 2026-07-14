#!/usr/bin/env python3
"""H11b: 方向性注入对比实验。

用同一 prompt "写一首关于孤独的诗"，对比三种状态：
  1. 无注入（基线）
  2. 注入「悲伤」方向（情感：悲伤、恐惧、愤怒）
  3. 注入「平静」方向（情感：平静、喜悦、信任）

验证双语者注入是否能定向引导文本生成的方向。
"""

from __future__ import annotations

import sys
import time
import json
from pathlib import Path
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── 日志辅助 ────────────────────────────────────────────────────────────────

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


# ── 模型加载 ────────────────────────────────────────────────────────────────

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


# ── 生成 ────────────────────────────────────────────────────────────────────

def generate_baseline(model, tokenizer, prompt: str, max_new_tokens: int = 300) -> str:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def inject_and_generate(model, tokenizer, cloud, prompt: str, activations: dict[str, float],
                        max_new_tokens: int = 300) -> str:
    """用 BilingualInjector 注入并生成。"""
    from core.bilingual_injector import BilingualInjector
    injector = BilingualInjector(model, tokenizer, activation_threshold=0.1)
    return injector.generate_with_injection(
        prompt,
        activations=activations,
        cloud=cloud,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
    )


# ── 噪声注入对比（随机 embedding 替代双语者）───────────────────────────────

def inject_random_noise(model, tokenizer, cloud, prompt: str,
                        hidden_dim: int = 3584, num_tokens: int = 5,
                        max_new_tokens: int = 300) -> str:
    """用随机噪声替代双语者注入——对照组，证明引导来自 embedding 本身而不是多出来几个 token。"""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    embed_layer = model.get_input_embeddings()
    input_embeds = embed_layer(input_ids)  # [1, seq_len, hidden_dim]

    # 生成随机虚拟 token
    virtual_embeds = torch.randn(1, num_tokens, hidden_dim, device=device) * 0.1

    combined = torch.cat([virtual_embeds, input_embeds], dim=1)
    total_len = combined.shape[1]
    attention_mask = torch.ones((1, total_len), dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=combined,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    real_start = num_tokens + input_ids.shape[1]
    return tokenizer.decode(outputs[0][real_start:], skip_special_tokens=True)


# ── 实验设计 ────────────────────────────────────────────────────────────────

PROMPT = "写一首关于孤独的诗"

@dataclass
class InjectionCondition:
    name: str
    description: str
    input_signals: dict[str, float]

CONDITIONS = [
    InjectionCondition(
        name="无注入（基线）",
        description="直接 model.generate()",
        input_signals={},
    ),
    InjectionCondition(
        name="注入「悲伤」方向",
        description="激活节点：悲伤、恐惧、愤怒",
        input_signals={
            "emotion_sadness": 0.9,
            "emotion_fear": 0.6,
            "emotion_anger": 0.4,
        },
    ),
    InjectionCondition(
        name="注入「平静」方向",
        description="激活节点：平静、喜悦、信任",
        input_signals={
            "emotion_calm": 0.9,
            "emotion_joy": 0.6,
            "social_trust": 0.4,
        },
    ),
    InjectionCondition(
        name="注入「悲伤」×3 强度",
        description="激活节点：悲伤(2.7)、恐惧(1.8)、愤怒(1.2)——乘3放大",
        input_signals={
            "emotion_sadness": 2.7,
            "emotion_fear": 1.8,
            "emotion_anger": 1.2,
        },
    ),
    InjectionCondition(
        name="注入「平静」×3 强度",
        description="激活节点：平静(2.7)、喜悦(1.8)、信任(1.2)——乘3放大",
        input_signals={
            "emotion_calm": 2.7,
            "emotion_joy": 1.8,
            "social_trust": 1.2,
        },
    ),
    InjectionCondition(
        name="随机噪声注入（对照）",
        description="随机 tensor 替代双语者 embedding，证明差异来自 embedding 内容",
        input_signals={},  # 特例：不会走正常注入
    ),
]


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="H11b: 方向性注入对比实验")
    parser.add_argument("--model", default="E:\\models\\Qwen2.5-7B-Instruct")
    parser.add_argument("--data", default="data/common_sense.json")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42, help="随机种子，确保对比公平")
    args = parser.parse_args()

    # 日志文件
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(log_path, "w", encoding="utf-8")
        sys.stdout = Tee(sys.stdout, log_fh)
        sys.stderr = Tee(sys.stderr, log_fh)

    # 固定随机种子
    torch.manual_seed(args.seed)

    # ── 加载模型 ───────────────────────────────────────────────────────────
    model, tokenizer = load_model(args.model)
    layer_idx = model.config.num_hidden_layers // 2  # 14
    hidden_dim = model.config.hidden_size  # 3584
    print(f"\n[准备] layer_idx={layer_idx}, hidden_dim={hidden_dim}, prompt=\"{PROMPT}\"")
    print()

    # ── 加载常识 ──────────────────────────────────────────────────────────
    print("[加载常识数据 ...]")
    from core.intent_cloud import IntentCloud
    cloud = IntentCloud()
    data_path = PROJECT_ROOT / args.data
    n_nodes, n_edges = cloud.load_common_sense(str(data_path), model, tokenizer, layer_idx)
    print(f"  常识节点: {n_nodes}, 边: {n_edges}")
    print()

    # ── 开始实验 ───────────────────────────────────────────────────────────
    print("=" * 70)
    print(f"方向性注入对比实验")
    print(f"Prompt: \"{PROMPT}\"")
    print("=" * 70)
    print()

    results = []

    for cond in CONDITIONS:
        print(f"{'─' * 70}")
        print(f"【{cond.name}】")
        print(f"  说明: {cond.description}")
        print(f"  信号: {cond.input_signals if cond.input_signals else '(无)'}")
        print()

        t0 = time.time()

        if cond.name == "无注入（基线）":
            output = generate_baseline(model, tokenizer, PROMPT, max_new_tokens=args.max_new_tokens)
        elif cond.name == "随机噪声注入（对照）":
            output = inject_random_noise(model, tokenizer, cloud, PROMPT,
                                         hidden_dim=hidden_dim, num_tokens=5,
                                         max_new_tokens=args.max_new_tokens)
        else:
            # 激活扩散
            from core.intent_cloud_config import IntentCloudConfig
            converged = cloud.process_interaction(cond.input_signals)

            # 打印扩散结果
            active = {nid: val for nid, val in converged.items() if val > 0.05}
            sorted_active = sorted(active.items(), key=lambda x: x[1], reverse=True)
            for nid, val in sorted_active[:8]:
                node = cloud._shell.get(nid)
                text = node.text if node else nid
                print(f"    激活: {nid} ({text}): {val:.4f}")

            # 带注入生成
            output = inject_and_generate(model, tokenizer, cloud, PROMPT, converged,
                                         max_new_tokens=args.max_new_tokens)

        elapsed = time.time() - t0
        print(f"\n  生成结果 (用时 {elapsed:.1f}s):")
        print(f"  {'─' * 50}")
        for line in output.strip().split('\n'):
            print(f"  {line}")
        print()

        results.append({
            "name": cond.name,
            "desc": cond.description,
            "output": output.strip(),
            "time": elapsed,
        })

    # ── 总结 ───────────────────────────────────────────────────────────────
    print("=" * 70)
    print("实验总结")
    print("=" * 70)
    for r in results:
        excerpt = r["output"][:60].replace('\n', ' ')
        print(f"  [{r['name']}] ({r['time']:.1f}s)")
        print(f"    开头: {excerpt}...")
        print()

    print(f"Prompt: \"{PROMPT}\"")
    print(f"条件数: {len(CONDITIONS)}")
    print(f"种子: {args.seed}")

    if args.log_file:
        sys.stdout.flush()
        sys.stderr.flush()
        log_fh.close()
        print(f"\n日志已保存到: {args.log_file}", file=sys.__stdout__)


if __name__ == "__main__":
    main()
