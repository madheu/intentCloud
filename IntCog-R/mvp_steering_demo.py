#!/usr/bin/env python3
"""MVP 激活引导演示：走通「蓝图 → 引导向量 → 注入 → 生成」完整数据流。

支持加载本地真实模型（如 Qwythos），也可使用自包含微型 Transformer。

运行方式：
    python mvp_steering_demo.py                           # 使用自包含模型
    python mvp_steering_demo.py --model qwythos           # 加载本地 Qwythos 模型
    python mvp_steering_demo.py --model /path/to/model    # 加载指定路径模型

中间产物：
  1. 蓝图 JSON（identity, core_task, deep_goal, constraints, concepts）
  2. steering_vector 数值摘要（前 10 维、L2 范数）
  3. 注入前后生成文本对比
  4. 注入前后 logits 余弦相似度
"""
from __future__ import annotations

import argparse
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


# ── 1. 本地模型加载（优先）─────────────────────────────────────────────────────

def load_local_model(model_path: str):
    """加载本地预训练模型（如 Qwythos）。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[模型] 尝试加载本地模型: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        print(f"  ✓ 加载成功")
        print(f"    模型: {model.__class__.__name__}")
        print(f"    层数: {model.config.n_layer}")
        print(f"    隐藏维度: {model.config.hidden_size}")
        print(f"    设备: {next(model.parameters()).device}")
        return model, tokenizer
    except Exception as e:
        print(f"  ✗ 加载失败: {e}")
        return None, None


# ── 2. 自包含微型 Transformer（Fallback）──────────────────────────────────────

class TinyTransformer(nn.Module):
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
        self.config = type('Config', (), {'n_layer': n_layers, 'hidden_size': d_model})()

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
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
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.forward(input_ids)
                last_logits = logits[:, -1, :] / temperature
                probs = F.softmax(last_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
        return input_ids, last_logits


class TransformerBlock(nn.Module):
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
        attn_out, _ = self.attn(x, x, x, is_causal=True)
        x = self.ln1(x + attn_out)
        mlp_out = self.mlp(x)
        x = self.ln2(x + mlp_out)
        return x


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
                chars.append(chr((i - 4) % 253 + (i - 4) // 253 * 253))
        return "".join(chars)

    def to_tensor(self, text: str) -> torch.Tensor:
        ids = self.encode(text)
        return torch.tensor([ids], dtype=torch.long)

    def __call__(self, text: str, return_tensors: str = None) -> dict:
        """模拟 HuggingFace tokenizer 的调用接口。"""
        ids = self.encode(text)
        result = {"input_ids": torch.tensor([ids], dtype=torch.long)}
        if return_tensors == "pt":
            return result
        return {"input_ids": ids}


# ── 3. 蓝图生成 ──────────────────────────────────────────────────────────────

def generate_blueprint(user_input: str) -> dict:
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


# ── 5. 模型层定位器（适配不同模型结构）───────────────────────────────────────

def get_layer(model: nn.Module, layer_idx: int) -> nn.Module:
    """获取模型指定层，适配不同模型结构。"""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]
    if hasattr(model, "layers"):
        return model.layers[layer_idx]
    raise ValueError(f"无法定位模型第 {layer_idx} 层，请检查模型结构")


# ── 6. 生成函数 ──────────────────────────────────────────────────────────────

def generate_text(
    model: nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 30,
    temperature: float = 0.8,
) -> tuple[str, torch.Tensor]:
    """生成文本，返回 (文本, last_logits)。"""
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            output_logits=True,
            return_dict_in_generate=True,
        )

    generated_ids = outputs.sequences[0]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    if hasattr(outputs, "logits") and outputs.logits:
        last_logits = outputs.logits[-1][0]
    else:
        last_logits = torch.zeros(model.config.vocab_size)

    return text, last_logits


# ── 7. 主流程 ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="MVP 激活引导演示")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="本地模型路径（如 qwythos 或 /path/to/model）",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=2,
        help="注入引导向量的层索引",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=1.5,
        help="引导强度系数",
    )
    args = parser.parse_args()

    user_input = "帮我写一首关于大海的诗"

    # ── 打印蓝图 ──
    blueprint = generate_blueprint(user_input)
    print("=" * 60)
    print("  📋 意图蓝图 (Blueprint JSON)")
    print("=" * 60)
    print(json.dumps(blueprint, ensure_ascii=False, indent=2))
    print()

    # ── 加载模型 ──
    model, tokenizer = None, None
    if args.model:
        model, tokenizer = load_local_model(args.model)

    if model is None:
        print("[模型] 使用自包含微型 Transformer（Fallback）")
        device = torch.device("cpu")
        d_model = 128
        n_layers = 4
        model = TinyTransformer(d_model=d_model, n_layers=n_layers).to(device)
        model.eval()
        tokenizer = CharTokenizer()
    else:
        device = next(model.parameters()).device
        d_model = model.config.hidden_size
        n_layers = model.config.n_layer

    layer_idx = min(args.layer, n_layers - 1)
    print(f"  层数: {n_layers}, 隐藏维度: {d_model}, 注入层: layer_{layer_idx}")
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

    # ── 创建注入器 ──
    injector = SteeringInjector(model, layer_idx)
    injector._get_layer = lambda: get_layer(model, layer_idx)

    prompt = "大海"
    print("=" * 60)
    print("  📝 生成对比：无引导 vs 有引导")
    print("=" * 60)

    # ── 无引导生成 ──
    text_no_steer, logits_no_steer = generate_text(model, tokenizer, prompt)
    print(f"\n  [无引导]  prompt: {prompt}")
    print(f"  [无引导]  output: {text_no_steer}")

    # ── 注入引导并生成 ──
    injector.inject(steering_vector, strength=args.strength)
    status = injector.get_status()
    print(f"\n  [注入状态] {json.dumps(status, ensure_ascii=False)}")

    text_steered, logits_steered = generate_text(model, tokenizer, prompt)
    print(f"\n  [有引导]  prompt: {prompt}")
    print(f"  [有引导]  output: {text_steered}")

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

    # ── 多层注入效果对比 ──
    print("=" * 60)
    print("  🔬 多层注入效果对比")
    print("=" * 60)
    for test_layer in range(min(4, n_layers)):
        test_injector = SteeringInjector(model, test_layer)
        test_injector._get_layer = lambda l=test_layer: get_layer(model, l)
        test_injector.inject(steering_vector, strength=args.strength)

        _, test_logits = generate_text(model, tokenizer, prompt, max_new_tokens=10)
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