#!/usr/bin/env python3
"""MVP 激活引导演示：走通「蓝图 → 引导向量 → 注入 → 生成」完整数据流。

自包含实现：不依赖 HuggingFace 模型下载，使用从零构建的微型 Transformer。

运行方式：
    python mvp_steering_demo.py

中间产物：
  1. 蓝图 JSON（identity, core_task, deep_goal, constraints, concepts）
  2. steering_vector 数值摘要（前 10 维、L2 范数）
  3. 注入前后生成文本对比
  4. 注入前后 logits 余弦相似度
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.steering import SteeringInjector, SteeringVector, compute_cosine_shift


# ── 1. 微型 Transformer（自包含，无外部依赖）───────────────────────────────────

class TinyTransformer(nn.Module):
    """一个极简的 Transformer 解码器，用于演示激活引导机制。

    结构：Embedding → N 层 TransformerBlock → LM Head
    每层在 forward 中暴露 hook 点，供 SteeringInjector 注入。
    """

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        max_seq_len: int = 64,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """前向传播，返回 logits。"""
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)

        x = self.embed(input_ids) + self.pos_embed(positions)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.lm_head(x)

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 20,
        temperature: float = 0.8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """自回归生成，返回 (token_ids, last_logits)。"""
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.forward(input_ids)
                last_logits = logits[:, -1, :] / temperature
                probs = F.softmax(last_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
        return input_ids, last_logits


class TransformerBlock(nn.Module):
    """单个 Transformer 解码器块。"""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 因果自注意力
        attn_out, _ = self.attn(x, x, x, is_causal=True)
        x = self.ln1(x + attn_out)
        mlp_out = self.mlp(x)
        x = self.ln2(x + mlp_out)
        return x


# ── 2. 字符级编码器 ──────────────────────────────────────────────────────────

class CharTokenizer:
    """简单的字符级 tokenizer，vocab_size=256，映射到连续 ID。"""

    def __init__(self) -> None:
        self.bos_id = 0
        self.eos_id = 1
        self.pad_id = 2
        self.vocab_size = 256

    def encode(self, text: str) -> list[int]:
        ids = [min(ord(c) % 253 + 4, 255) for c in text]
        return ids

    def decode(self, ids: list[int] | torch.Tensor) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        chars = []
        for i in ids:
            if i == self.eos_id or i == self.pad_id:
                break
            if i >= 4:
                # 反向映射：尝试找到原始字符
                chars.append(chr((i - 4) % 253 + (i - 4) // 253 * 253))
        return "".join(chars)

    def to_tensor(self, text: str) -> torch.Tensor:
        ids = self.encode(text)
        return torch.tensor([ids], dtype=torch.long)


# ── 3. 蓝图生成 ──────────────────────────────────────────────────────────────

def generate_blueprint(user_input: str) -> dict:
    """生成意图蓝图 JSON。"""
    return {
        "identity": "Qwythos",
        "core_task": "帮助用户创作一首关于大海的诗",
        "deep_goal": "满足用户的审美需求，提供情感共鸣",
        "constraints": ["使用中文", "保持诗歌的意象美感", "不超过 8 行"],
        "concepts": ["大海", "诗歌", "意象", "浪漫"],
        "trust_score": 0.9,
    }


# ── 4. 蓝图 → 引导向量 ──────────────────────────────────────────────────────

def blueprint_to_steering_vector(
    blueprint: dict,
    hidden_dim: int,
    device: torch.device,
) -> SteeringVector:
    """从蓝图生成引导向量（确定性伪随机）。"""
    import hashlib

    core_task = blueprint.get("core_task", "")
    deep_goal = blueprint.get("deep_goal", "")
    identity = blueprint.get("identity", "")

    seed_text = f"{identity}|{core_task}|{deep_goal}"
    seed = int(hashlib.sha256(seed_text.encode()).hexdigest(), 16) % (2**31)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    vector = torch.randn(hidden_dim, generator=gen, device=device)
    vector = vector / torch.norm(vector)

    return SteeringVector(vector, label=f"blueprint:{core_task[:30]}")


# ── 5. 主流程 ────────────────────────────────────────────────────────────────

def main() -> None:
    user_input = "帮我写一首关于大海的诗"

    # ── 打印蓝图 ──
    blueprint = generate_blueprint(user_input)
    print("=" * 60)
    print("  📋 意图蓝图 (Blueprint JSON)")
    print("=" * 60)
    print(json.dumps(blueprint, ensure_ascii=False, indent=2))
    print()

    # ── 创建模型 ──
    device = torch.device("cpu")
    d_model = 128
    n_layers = 4
    layer_idx = 2  # 注入到第 2 层

    model = TinyTransformer(d_model=d_model, n_layers=n_layers).to(device)
    model.eval()
    tokenizer = CharTokenizer()

    print(f"[模型] 自包含微型 Transformer")
    print(f"  layers={n_layers}, d_model={d_model}, heads=4, vocab=256")
    print(f"  注入层: layer_{layer_idx}")
    print(f"  参数总量: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # ── 蓝图 → 引导向量 ──
    steering_vector = blueprint_to_steering_vector(blueprint, d_model, device)
    print("=" * 60)
    print("  🧭 引导向量 (Steering Vector) 摘要")
    print("=" * 60)
    summary = steering_vector.summary(top_k=10)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print()

    # ── 创建注入器：绑定到 model.layers[layer_idx] ──
    injector = SteeringInjector(model, layer_idx)
    # 覆盖 _get_layer 以适配 TinyTransformer 结构
    injector._get_layer = lambda: model.layers[layer_idx]

    prompt = "大海"
    prompt_ids = tokenizer.to_tensor(prompt)

    # ── 无引导生成 ──
    print("=" * 60)
    print("  📝 生成对比：无引导 vs 有引导")
    print("=" * 60)

    ids_no_steer, logits_no_steer = model.generate(prompt_ids, max_new_tokens=15)
    text_no_steer = tokenizer.decode(ids_no_steer[0])
    print(f"\n  [无引导]  prompt: {prompt}")
    print(f"  [无引导]  output: {text_no_steer}")

    # ── 注入引导并生成 ──
    injector.inject(steering_vector, strength=1.5)
    status = injector.get_status()
    print(f"\n  [注入状态] {json.dumps(status, ensure_ascii=False)}")

    ids_steered, logits_steered = model.generate(prompt_ids, max_new_tokens=15)
    text_steered = tokenizer.decode(ids_steered[0])
    print(f"\n  [有引导]  prompt: {prompt}")
    print(f"  [有引导]  output: {text_steered}")

    # 移除注入
    injector.remove()

    # ── Logits 余弦相似度 ──
    cos_sim = compute_cosine_shift(logits_no_steer, logits_steered)
    print()
    print("=" * 60)
    print("  📊 注入前后 Logits 余弦相似度")
    print("=" * 60)
    print(f"  cosine_similarity = {cos_sim:.6f}")
    shift_pct = (1 - cos_sim) * 100
    print(f"  logits 偏移量 = {shift_pct:.2f}%")
    print(f"  (数值越大，引导效果越显著)")
    print()

    # ── 验证：多层注入对比 ──
    print("=" * 60)
    print("  🔬 多层注入效果对比")
    print("=" * 60)
    for test_layer in range(n_layers):
        test_injector = SteeringInjector(model, test_layer)
        test_injector._get_layer = lambda l=test_layer: model.layers[l]
        test_injector.inject(steering_vector, strength=1.5)

        test_ids = tokenizer.to_tensor(prompt)
        _, test_logits = model.generate(test_ids, max_new_tokens=10)
        test_cos = compute_cosine_shift(logits_no_steer, test_logits)
        test_injector.remove()

        bar = "█" * int((1 - test_cos) * 50)
        print(f"  layer_{test_layer}: cos={test_cos:.4f}  shift={bar}")

    print()
    print("=" * 60)
    print("  ✅ MVP 激活引导数据流演示完成")
    print("=" * 60)


if __name__ == "__main__":
    main()