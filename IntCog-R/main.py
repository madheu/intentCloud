"""IntCog-R 异步主循环。

快通道（每 200 ms）：
- 接收输入/感知帧
- 注意竞争
- 记忆检索
- 约束检查
- 广播当前帧

慢通道（异步，结果可丢弃）：
- LLM 意图提取
- 内部言语规划
- 骨架生成

慢通道结果作为候选帧参与后续 tick 的注意竞争。
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any

from core.attention import AttentionRouter
from core.config_loader import load_config
from core.generator import SafeGenerator
from core.intent_cloud import IntentCloud
from core.intent_extractor import IntentExtractor
from core.internal_speech import InternalSpeechPlanner
from core.memory import ChromaMemory, MemoryEntry, MemoryProvider
from core.metacognition import MetacognitionMonitor
from core.models import ConsciousFrame, FallbackBlueprint, FrameModality, FrameType, IntentBlueprint
from core.safety import SafetyFilter
from core.workspace import GlobalWorkspace
from intent_safe_generate import IntentSafeGeneratePipeline


class CogStreamEngine:
    """CogStream 意识流引擎。"""

    def __init__(
        self,
        llm_client: Any | None = None,
        tick_ms: int = 200,
        max_slow_task_age_ticks: int = 5,
    ) -> None:
        self.tick_ms = tick_ms
        self.max_slow_task_age_ticks = max_slow_task_age_ticks

        config = load_config()
        self.workspace = GlobalWorkspace(capacity=1)
        self.intent_cloud = IntentCloud()
        self.attention = AttentionRouter(self.intent_cloud)
        self.memory: MemoryProvider = ChromaMemory()
        self.planner = InternalSpeechPlanner(llm_client=llm_client, use_llm=False)
        self.metacog = MetacognitionMonitor()
        self.safety = SafetyFilter(config=config)

        # 意图提取与生成（慢通道）
        self.extractor = IntentExtractor(llm_client=llm_client) if llm_client else None
        self.generator = SafeGenerator(llm_client=llm_client, kernel=self.intent_cloud.kernel) if llm_client else None
        self.safe_pipeline = (
            IntentSafeGeneratePipeline(self.extractor, self.intent_cloud, self.generator, self.safety)
            if self.extractor and self.generator
            else None
        )

        self.input_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=128)
        self.output_queue: asyncio.Queue[str | FallbackBlueprint] = asyncio.Queue(maxsize=256)
        self.pending_slow_tasks: set[asyncio.Task[list[ConsciousFrame]]] = set()
        self.logs: list[dict[str, Any]] = []
        self.running = False

    async def ingest(self, text: str) -> bool:
        """用户输入进入感知队列；队列满时丢弃最旧输入并记录。

        Returns:
            True if the input was accepted without dropping, False if an older entry was dropped.
        """
        dropped = False
        if self.input_queue.full():
            try:
                self.input_queue.get_nowait()
                self._log_dropped_input("queue_overflow")
                dropped = True
            except asyncio.QueueEmpty:
                pass
        await self.input_queue.put(text)
        return not dropped

    def _log_dropped_input(self, reason: str) -> None:
        self.logs.append(
            {
                "event": "input_dropped",
                "reason": reason,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    async def tick(self) -> ConsciousFrame | None:
        """执行一个意识 tick。"""
        await self.workspace.next_tick()
        tick_id = self.workspace.tick_id

        candidates: list[ConsciousFrame] = []

        # 1. 快通道：感知输入
        if not self.input_queue.empty():
            user_input = await self.input_queue.get()
            perception = ConsciousFrame(
                modality=FrameModality.PERCEPTION,
                frame_type=FrameType.USER_INPUT,
                data={"text": user_input},
                confidence=1.0,
                tick_id=tick_id,
            )
            candidates.append(perception)
            # 启动慢通道：意图提取 + 内部言语规划
            self._spawn_slow_task(user_input, perception)

        # 2. 快通道：收集已完成的慢通道结果
        candidates.extend(await self._collect_slow_results(tick_id))

        # 3. 快通道：记忆检索（基于当前工作空间帧）
        context = self._context_text()
        blueprint = await self.intent_cloud.build_blueprint(context)
        memories = await self.memory.retrieve(context, blueprint.goals, top_k=3)
        if memories:
            candidates.append(
                ConsciousFrame(
                    modality=FrameModality.INTERNAL_SPEECH,
                    frame_type=FrameType.MEMORY_RETRIEVAL,
                    data={"memories": [m.text for m in memories]},
                    confidence=0.45,
                    intent_refs=blueprint.goals,
                    tick_id=tick_id,
                )
            )

        # 4. 元认知监控
        current = self.workspace.current()
        if current:
            metacog_frame = self.metacog.check(current, blueprint)
            if metacog_frame:
                candidates.append(metacog_frame)

        # 5. 注意竞争
        winner = await self.attention.route(candidates, context_text=context)
        if winner is None:
            winner = self.workspace.current()

        if winner:
            await self.workspace.publish(winner)
            await self.workspace.broadcast()
            await self._react(winner, blueprint, memories)

        self._log(tick_id, winner, candidates)
        return winner

    def _context_text(self) -> str:
        current = self.workspace.current()
        if current is None:
            return ""
        if isinstance(current.data, dict) and "text" in current.data:
            return str(current.data["text"])
        return str(current.data)

    def _spawn_slow_task(self, user_input: str, perception_frame: ConsciousFrame) -> None:
        """启动慢通道任务：意图提取、记忆检索、内部言语规划。"""
        task = asyncio.create_task(
            self._slow_channel(user_input, perception_frame),
            name=f"slow-{self.workspace.tick_id}",
        )
        self.pending_slow_tasks.add(task)
        task.add_done_callback(self.pending_slow_tasks.discard)

    async def _slow_channel(
        self,
        user_input: str,
        perception_frame: ConsciousFrame,
    ) -> list[ConsciousFrame]:
        """慢通道：异步生成候选思维帧。"""
        blueprint = await self.intent_cloud.build_blueprint(user_input)
        memories = await self.memory.retrieve(user_input, blueprint.goals, top_k=3)
        planned = await self.planner.plan(perception_frame, blueprint, memories)
        for frame in planned:
            frame.timestamp = datetime.now(timezone.utc)
        return planned

    async def _collect_slow_results(self, current_tick: int) -> list[ConsciousFrame]:
        """收集已完成的慢通道结果，丢弃过期的；记录异常。"""
        results: list[ConsciousFrame] = []
        done_tasks: set[asyncio.Task[list[ConsciousFrame]]] = set()
        for task in self.pending_slow_tasks:
            if task.done():
                done_tasks.add(task)
                try:
                    frames = task.result()
                except Exception as exc:
                    self.logs.append(
                        {
                            "event": "slow_task_error",
                            "exc_type": type(exc).__name__,
                            "exc": str(exc),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                    continue
                for frame in frames:
                    if current_tick - frame.tick_id <= self.max_slow_task_age_ticks:
                        results.append(frame)
        for task in done_tasks:
            self.pending_slow_tasks.discard(task)
        return results

    async def _react(
        self,
        frame: ConsciousFrame,
        blueprint: IntentBlueprint,
        memories: list[MemoryEntry],
    ) -> None:
        """根据当前帧类型做出反应。"""
        # 将当前帧存入情节记忆
        await self.memory.store(
            MemoryEntry(
                text=self._frame_to_text(frame),
                metadata={
                    "tick_id": frame.tick_id,
                    "frame_type": frame.frame_type.value,
                    "modality": frame.modality.value,
                },
            )
        )

        # 若是用户输入帧且存在安全生成管线，生成回复
        if frame.frame_type == FrameType.USER_INPUT and self.safe_pipeline:
            text = frame.data.get("text", "") if isinstance(frame.data, dict) else ""
            output = await self.safe_pipeline.run(text)
            await self.output_queue.put(output)

    def _frame_to_text(self, frame: ConsciousFrame) -> str:
        if isinstance(frame.data, dict):
            return str(frame.data.get("text", frame.data))
        return str(frame.data)

    def _log(
        self,
        tick_id: int,
        winner: ConsciousFrame | None,
        candidates: list[ConsciousFrame],
    ) -> None:
        self.logs.append(
            {
                "tick": tick_id,
                "winner": winner.model_dump() if winner else None,
                "candidate_count": len(candidates),
                "pending_slow_tasks": len(self.pending_slow_tasks),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    async def run(self, ticks: int | None = None) -> AsyncIterator[ConsciousFrame]:
        """持续运行主循环。"""
        self.running = True
        count = 0
        while self.running:
            winner = await self.tick()
            if winner:
                yield winner
            count += 1
            if ticks is not None and count >= ticks:
                break
            await asyncio.sleep(self.tick_ms / 1000.0)

    def stop(self) -> None:
        self.running = False

    def snapshot(self) -> dict[str, Any]:
        return {
            "tick_id": self.workspace.tick_id,
            "workspace": self.workspace.snapshot(),
            "pending_slow_tasks": len(self.pending_slow_tasks),
            "output_queue_size": self.output_queue.qsize(),
            "input_queue_size": self.input_queue.qsize(),
        }


async def main() -> None:
    """命令行入口：运行 20 个 tick 的意识流演示。"""
    engine = CogStreamEngine(tick_ms=200)
    await engine.ingest("你好，请介绍一下自己。")

    async for winner in engine.run(ticks=20):
        print(f"tick={engine.workspace.tick_id} modality={winner.modality.value} type={winner.frame_type.value}")
        if winner.frame_type == FrameType.USER_INPUT:
            print(f"  input: {winner.data}")
        if winner.modality == FrameModality.INTERNAL_SPEECH:
            print(f"  thought: {winner.data}")


if __name__ == "__main__":
    asyncio.run(main())
