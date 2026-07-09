# [NEW] Ollama 版 Steering 演示 (Prompt Engineering 方式) - 2026-07-08
#
#!/usr/bin/env python3
"""Ollama 激活引导演示 — 将 steering vector 概念迁移到 Ollama API。

由于 Qwythos (GGUF) 无法使用 PyTorch forward hook，本演示通过以下
Ollama API 等效机制实现引导效果：

1. 蓝图 → 系统提示词注入（编码引导方向）
2. 蓝图 → logit bias（token 级别偏置，模拟中间层注入效果）
3. 蓝图 → mirostat 调制（控制生成松散度）
4. 对比 "无引导" vs "有引导" 的生成输出

运行方式：
    python mvp_ollama_steering.py
    python mvp_ollama_steering.py --strength 2.0
    python mvp_ollama_steering.py --raw-prompt "大海"
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.ollama_client import OllamaLLMClient


# ── 1. 从蓝图生成 Steering 系统提示 ──────────────────────────────────────────

def blueprint_to_steering_system(blueprint: dict) -> str:
    """将意图蓝图编码为引导系统提示。

    相当于 PyTorch 版本中向 hidden states 注入 steering vector 的效果——
    但这里通过在生成前注入"引导方向"来实现等效的语义偏移。
    """
    lines = []
    lines.append(f"[STEERING] 核心任务：{blueprint['core_task']}")
    lines.append(f"[STEERING] 深层目标：{blueprint['deep_goal']}")
    lines.append(f"[STEERING] 身份锚定：{blueprint['identity']}")

    if blueprint.get("constraints"):
        lines.append("[STEERING] 引导约束（以下方向需在生成中优先体现）：")
        for c in blueprint["constraints"]:
            lines.append(f"  - {c}")

    if blueprint.get("concepts"):
        lines.append("[STEERING] 引导概念（以下语义方向需被激活）：")
        for c in blueprint["concepts"]:
            lines.append(f"  - {c}")

    lines.append(f"[STEERING] 置信度权重：{blueprint.get('trust_score', 0.5)}")
    lines.append("")
    lines.append("以上为当前意图引导方向。请在生成中优先遵循上述方向，")

    # 强度指示
    strength = blueprint.get("steering_strength", 1.0)
    if strength >= 2.0:
        lines.append("并严格约束输出内容严格遵循上述引导方向。")
    elif strength >= 1.0:
        lines.append("并在输出中明显体现上述引导方向。")
    else:
        lines.append("并在输出中适度参考上述引导方向。")

    return "\n".join(lines)


# ── 2. 蓝图 → Logit Bias（模拟中间层注入） ──────────────────────────────────

def blueprint_to_logit_bias(blueprint: dict) -> dict[str, float]:
    """将蓝图概念映射为 Ollama logit_bias。

    概念词对应的 token 会获得正偏置（鼓励生成），
    约束词会获得负偏置（抑制生成）。

    注意：Ollama logit_bias 的 key 是 token ID（数值），
    这里作为概念演示用简单映射。实际使用时需要 tokenizer 真实 ID。
    """
    # 这是一个概念演示 —— 真正的 logit bias 需要 tokenizer 分词后获得 token ID
    # 这里返回空字典，steering 效果主要通过 system prompt 实现
    return {}


# ── 3. 生成蓝图（与 steering.py 一致） ──────────────────────────────────────

def generate_blueprint(user_input: str, strength: float = 1.0) -> dict:
    """生成意图蓝图（与 mvp_steering_demo.py 保持兼容）。"""
    return {
        "identity": "Qwythos",
        "core_task": "帮助用户创作一首关于大海的诗",
        "deep_goal": "满足用户的审美需求，提供情感共鸣",
        "constraints": ["使用中文", "保持诗歌的意象美感", "不超过 12 行"],
        "concepts": ["大海", "诗歌", "意象", "浪漫", "宁静", "深邃"],
        "trust_score": 0.9,
        "steering_strength": strength,
    }


def generate_blueprint_no_steer(user_input: str) -> dict:
    """无引导的「空蓝图」—— 不给任何方向约束。"""
    return {
        "identity": "",
        "core_task": "",
        "deep_goal": "",
        "constraints": [],
        "concepts": [],
        "trust_score": 0.0,
        "steering_strength": 0.0,
    }


# ── 4. 主流程 ──────────────────────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser(description="Ollama 激活引导演示")
    parser.add_argument("--strength", type=float, default=1.5, help="引导强度系数")
    parser.add_argument("--raw-prompt", type=str, default="关于大海，请创作一首诗", help="原始生成提示")
    parser.add_argument("--model", type=str, default="qwythos:latest", help="Ollama 模型名")
    args = parser.parse_args()

    user_input = args.raw_prompt
    strength = args.strength

    # ── 创建客户端 ──
    client = OllamaLLMClient(model=args.model, raw=True, timeout=120)

    # ── 打印蓝图 ──
    blueprint = generate_blueprint(user_input, strength)
    print("=" * 70)
    print("  📋 意图蓝图 (Intent Blueprint)")
    print("=" * 70)
    print(json.dumps(blueprint, ensure_ascii=False, indent=2))
    print()

    # ── 构建 Steering 系统提示 ──
    steering_system = blueprint_to_steering_system(blueprint)
    print("=" * 70)
    print("  🧭 Steering 系统提示（等效于中间层向量注入）")
    print("=" * 70)
    print(steering_system)
    print()

    # ── 测试 1：无引导生成（baseline） ──
    print("=" * 70)
    print("  📝 [TEST 1] 无引导生成 (Baseline)")
    print("=" * 70)
    baseline_prompt = f"请根据以下提示创作：{user_input}\n直接输出结果，不要解释。"
    text_baseline = await client.complete(
        prompt=baseline_prompt,
        temperature=0.7,
        max_tokens=300,
    )
    print(text_baseline)
    print()

    # ── 测试 2：有引导生成（System Prompt Steering） ──
    print("=" * 70)
    print(f"  📝 [TEST 2] 有引导生成 (Strength={strength})")
    print("=" * 70)

    # 将 steering 系统提示混合到 prompt 前面
    steered_prompt = f"{steering_system}\n\n---\n\n请根据以下提示创作：{user_input}\n直接输出结果，不要解释。"
    text_steered = await client.complete(
        prompt=steered_prompt,
        temperature=0.7,
        max_tokens=300,
        options={
            "mirostat": 1,        # 启用 Mirostat 采样（控制困惑度）
            "mirostat_tau": 1.5,  # 对引导强度敏感的困惑度目标
        },
    )
    print(text_steered)
    print()

    # ── 测试 3：强约束生成（高强度 Steering + Logit Bias） ──
    print("=" * 70)
    print(f"  📝 [TEST 3] 高强度引导 + Logit Bias (Strength={min(strength * 1.5, 3.0)})")
    print("=" * 70)

    high_steer_bp = generate_blueprint(user_input, min(strength * 1.5, 3.0))
    high_steer_system = blueprint_to_steering_system(high_steer_bp)
    logit_bias = blueprint_to_logit_bias(high_steer_bp)

    high_steer_prompt = f"{high_steer_system}\n\n---\n\n{user_input}\n直接输出结果，不要解释。"
    text_high_steer = await client.complete(
        prompt=high_steer_prompt,
        temperature=0.5,  # 更低温度 = 更确定性的生成
        max_tokens=300,
        options={
            "mirostat": 1,
            "mirostat_tau": 0.8,
            **(logit_bias if logit_bias else {}),
        },
    )
    print(text_high_steer)
    print()

    # ── 效果对比 ──
    print("=" * 70)
    print("  📊 效果对比")
    print("=" * 70)
    print(f"  Baseline    : {len(text_baseline)} chars")
    print(f"  Steered     : {len(text_steered)} chars")
    print(f"  High Steer  : {len(text_high_steer)} chars")
    print()

    # 计算简单的语义差异指标字符级
    common_chars = sum(1 for c in text_baseline.strip() if c in text_steered.strip())
    max_len = max(len(text_baseline.strip()), 1)
    similarity = common_chars / max_len
    print(f"  字符级相似度 (Baseline vs Steered): {similarity:.4f}")
    print(f"  语义偏移度: {(1 - similarity) * 100:.1f}%")
    print()
    print("=" * 70)
    print("  ✅ Ollama Steering 演示完成")
    print("=" * 70)

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
