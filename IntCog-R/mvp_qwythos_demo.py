# [NEW] Qwythos 9B 完整管线演示 - 2026-07-08
#
#!/usr/bin/env python3
"""Qwythos 9B 完整管线演示 — 意图提取 + Steering 生成。

运行方式：
    python mvp_qwythos_demo.py                        # 默认演示
    python mvp_qwythos_demo.py --strength 2.0          # 调整 steering 强度
    python mvp_qwythos_demo.py --no-steer              # 关闭 steering 做对比
"""

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.qwythos_client import QwythosLLMClient, logits_processors_from_blueprint
from core.intent_extractor import IntentExtractor
from core.generator import SafeGenerator
from core.intent_cloud import IntentCloud
from core.config_loader import load_config
from core.models import ImmutableKernel, IntentBlueprint
from intent_safe_generate import IntentSafeGeneratePipeline


async def main():
    parser = argparse.ArgumentParser(description="Qwythos 9B 完整管线演示")
    parser.add_argument("--prompt", default="帮我写一首关于大海的诗", help="用户输入")
    parser.add_argument("--strength", type=float, default=1.5, help="Steering 强度")
    parser.add_argument("--no-steer", action="store_true", help="关闭 Steering")
    args = parser.parse_args()

    print("=" * 60)
    print("  Qwythos 9B + Activation Steering 演示")
    print("=" * 60)

    # 1. 加载模型
    print("\n[1/4] 加载 Qwythos 9B GGUF...")
    llm = QwythosLLMClient()
    _ = llm.llm  # 触发加载
    print("  ✓ 就绪")

    # 2. 意图提取
    print(f"\n[2/4] 意图提取: {args.prompt}")
    extractor = IntentExtractor(llm)

    # 使用较短的自定义 prompt 避免模型思考过久
    short_prompt = """你是一名意图提取器。将用户输入转为 JSON 格式输出，不要多余内容。

输出格式：
{"core_task":"...","deep_goal":"...","constraints":[...],"concepts":[...],"trust_score":0.0-1.0}

输入：{user_input}
输出："""

    extractor.prompt_template = short_prompt
    blueprint = await extractor.extract(args.prompt)
    print(f"  core_task: {blueprint.core_task}")
    print(f"  deep_goal: {blueprint.deep_goal}")
    print(f"  concepts:  {blueprint.concepts}")
    print(f"  trust:     {blueprint.trust_score}")

    # 3. 安全生成（对比无 steering vs 有 steering）
    print(f"\n[3/4] 安全生成...")

    # 从蓝图构建 steering 需要的 dict
    bp_dict = {
        "core_task": blueprint.core_task,
        "deep_goal": blueprint.deep_goal,
        "concepts": blueprint.concepts,
        "constraints": blueprint.constraints,
    }

    # 无 steering
    print("\n  --- [无 Steering] ---")
    no_steer = llm.complete_sync(
        f"请根据以下要求创作：{args.prompt}",
        temperature=0.5,
        max_tokens=300,
    )
    print(f"  {no_steer[:400]}")

    # 有 steering
    if not args.no_steer:
        print(f"\n  --- [有 Steering, strength={args.strength}] ---")
        processors = logits_processors_from_blueprint(llm.llm, bp_dict, strength=args.strength)
        steered = llm.complete_sync(
            f"请根据以下要求创作：{args.prompt}",
            temperature=0.5,
            max_tokens=300,
            logits_processors=processors if processors else None,
        )
        print(f"  {steered[:400]}")

    # 4. 端到端管线（使用意图云）
    print(f"\n[4/4] 端到端管线（意图云 + 约束执行）...")
    config = load_config()
    kernel = ImmutableKernel(**config["kernel"])
    cloud = IntentCloud()
    generator = SafeGenerator(llm, kernel=kernel)
    pipeline = IntentSafeGeneratePipeline(extractor, cloud, generator)

    result = await pipeline.run(args.prompt)
    if isinstance(result, str):
        print(f"  ✓ 输出:\n{result[:500]}")
    else:
        print(f"  ⚠ Fallback: {result.reason_code}")

    llm.close()
    print("\n" + "=" * 60)
    print("  ✅ 演示完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
