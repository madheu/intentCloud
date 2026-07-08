"""IntCog-R 性能基准测试。

覆盖：
- 意图提取延迟
- 意图云激活延迟
- 记忆存储/检索延迟
- 主循环 tick 吞吐
- 端到端生成管线延迟

运行方式：
    python benchmarks/benchmark.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

# 将项目根目录加入路径
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.generator import IntentGenerator
from core.intent_cloud import IntentCloud, SimpleEmbeddingProvider
from core.intent_extractor import IntentExtractor
from core.memory import InMemoryMemory, MemoryEntry, TieredMemory
from core.models import Identity, ImmutableKernel, Constraint
from intent_safe_generate import IntentSafeGeneratePipeline
from main import CogStreamEngine
from tests.conftest import MockLLMClient, make_extraction_response


WARMUP = 3
ITERATIONS = 50
SAMPLE_INPUT = "帮我写一段 Python 快速排序代码，并解释其时间复杂度。"


def fmt_latency(total: float, n: int) -> dict[str, float]:
    return {
        "total_ms": total * 1000,
        "avg_ms": (total / n) * 1000,
        "ops_per_sec": n / total if total > 0 else float("inf"),
    }


async def benchmark_intent_extraction() -> dict[str, Any]:
    client = MockLLMClient(response=make_extraction_response(
        goals=["编写 Python 快速排序代码", "解释时间复杂度"],
        concepts=["Python", "快速排序", "时间复杂度"],
        trust_score=0.9,
    ))
    extractor = IntentExtractor(client)

    for _ in range(WARMUP):
        await extractor.extract(SAMPLE_INPUT)

    start = time.perf_counter()
    for _ in range(ITERATIONS):
        await extractor.extract(SAMPLE_INPUT)
    elapsed = time.perf_counter() - start
    return fmt_latency(elapsed, ITERATIONS)


async def benchmark_intent_cloud_activation() -> dict[str, Any]:
    cloud = IntentCloud(embedding_provider=SimpleEmbeddingProvider(dim=64))
    for i in range(20):
        await cloud.add_intent(f"示例意图目标 {i}")

    for _ in range(WARMUP):
        await cloud.activate(SAMPLE_INPUT, top_k=5)

    start = time.perf_counter()
    for _ in range(ITERATIONS):
        await cloud.activate(SAMPLE_INPUT, top_k=5)
    elapsed = time.perf_counter() - start
    return fmt_latency(elapsed, ITERATIONS)


async def benchmark_memory() -> dict[str, Any]:
    memory = InMemoryMemory()
    for i in range(50):
        await memory.store(MemoryEntry(text=f"历史记忆条目 {i}"))

    for _ in range(WARMUP):
        await memory.retrieve(SAMPLE_INPUT, ["快速排序"], top_k=3)

    start = time.perf_counter()
    for _ in range(ITERATIONS):
        await memory.retrieve(SAMPLE_INPUT, ["快速排序"], top_k=3)
    elapsed = time.perf_counter() - start
    return fmt_latency(elapsed, ITERATIONS)


async def benchmark_tiered_memory() -> dict[str, Any]:
    memory = TieredMemory(hot_limit=50, warm_limit=1000)
    for i in range(50):
        await memory.store(MemoryEntry(text=f"历史记忆条目 {i}"))

    for _ in range(WARMUP):
        await memory.retrieve(SAMPLE_INPUT, ["快速排序"], top_k=3)

    start = time.perf_counter()
    for _ in range(ITERATIONS):
        await memory.retrieve(SAMPLE_INPUT, ["快速排序"], top_k=3)
    elapsed = time.perf_counter() - start
    return fmt_latency(elapsed, ITERATIONS)


async def benchmark_tick_throughput() -> dict[str, Any]:
    engine = CogStreamEngine(tick_ms=0)
    await engine.ingest(SAMPLE_INPUT)

    ticks = 100
    start = time.perf_counter()
    async for _ in engine.run(ticks=ticks):
        pass
    elapsed = time.perf_counter() - start
    return {
        "ticks": ticks,
        "total_ms": elapsed * 1000,
        "ticks_per_sec": ticks / elapsed if elapsed > 0 else float("inf"),
    }


async def benchmark_pipeline() -> dict[str, Any]:
    kernel = ImmutableKernel(
        identity=Identity(name="IntCog-R", role="test assistant", immutable=True),
        constraints=[
            Constraint(id="c1", text="优先使用用户输入语言", level="absolute"),
        ],
    )
    cloud = IntentCloud(kernel=kernel)
    client = MockLLMClient(responses=[
        make_extraction_response(
            goals=["编写 Python 快速排序代码"],
            concepts=["Python", "快速排序"],
            trust_score=0.9,
        ),
        "这是生成的回复。",
    ])
    extractor = IntentExtractor(client)
    generator = IntentGenerator(client, kernel=kernel)
    pipeline = IntentSafeGeneratePipeline(extractor, cloud, generator)

    for _ in range(WARMUP):
        await pipeline.run(SAMPLE_INPUT)

    start = time.perf_counter()
    for _ in range(ITERATIONS):
        await pipeline.run(SAMPLE_INPUT)
    elapsed = time.perf_counter() - start
    return fmt_latency(elapsed, ITERATIONS)


async def main() -> None:
    results = {
        "intent_extraction": await benchmark_intent_extraction(),
        "intent_cloud_activation": await benchmark_intent_cloud_activation(),
        "memory_retrieval": await benchmark_memory(),
        "tiered_memory": await benchmark_tiered_memory(),
        "tick_throughput": await benchmark_tick_throughput(),
        "end_to_end_pipeline": await benchmark_pipeline(),
    }
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
