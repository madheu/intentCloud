"""全局工作空间（Global Workspace）。

容量为 1；每帧占据工作空间 200 ms。新帧只有具备足够置信度或是感知/用户输入时，
才能覆写当前帧；否则被丢弃。所有模块通过监听器异步接收广播。
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from datetime import datetime, timezone
from typing import Any

from core.models import ConsciousFrame, FrameModality


Listener = Callable[[ConsciousFrame], Coroutine[Any, Any, None]]


class GlobalWorkspace:
    """容量为 1 的全局工作空间。"""

    def __init__(self, capacity: int = 1) -> None:
        if capacity != 1:
            raise ValueError("GlobalWorkspace capacity must be 1")
        self.capacity: int = capacity
        self.current_frame: ConsciousFrame | None = None
        self.tick_id: int = 0
        self._listeners: list[Listener] = []
        self._lock = asyncio.Lock()

    def subscribe(self, listener: Listener) -> None:
        """订阅工作空间广播。"""
        self._listeners.append(listener)

    def unsubscribe(self, listener: Listener) -> None:
        """取消订阅。"""
        if listener in self._listeners:
            self._listeners.remove(listener)

    async def publish(self, frame: ConsciousFrame) -> ConsciousFrame:
        """尝试将帧写入工作空间；返回最终占据工作空间的帧。"""
        async with self._lock:
            frame.tick_id = self.tick_id
            frame.timestamp = datetime.now(timezone.utc)

            if self.current_frame is None:
                self.current_frame = frame
                return frame

            current = self.current_frame
            # 感知/用户输入具有高突显性，允许覆写
            if frame.modality in (FrameModality.PERCEPTION,):
                self.current_frame = frame
                return frame

            # 静默帧若未结束，则不被普通帧覆写
            if current.modality == FrameModality.SILENCE:
                silence_state = current.data
                if silence_state.elapsed_frames < silence_state.duration_frames:
                    return current

            # 按置信度竞争
            if frame.confidence >= current.confidence:
                self.current_frame = frame
                return frame

            # 丢弃
            return current

    async def broadcast(self) -> None:
        """将当前帧广播给所有监听器。"""
        if self.current_frame is None:
            return
        frame = self.current_frame
        await asyncio.gather(
            *(listener(frame) for listener in self._listeners),
            return_exceptions=True,
        )

    async def next_tick(self) -> None:
        """进入下一 tick；静默帧推进持续时间。"""
        async with self._lock:
            self.tick_id += 1
            if self.current_frame and self.current_frame.modality == FrameModality.SILENCE:
                silence_state = self.current_frame.data
                silence_state.elapsed_frames += 1

    def current(self) -> ConsciousFrame | None:
        return self.current_frame

    def snapshot(self) -> dict[str, Any]:
        """结构化快照，用于观测面板。"""
        return {
            "tick_id": self.tick_id,
            "current_frame": self.current_frame.model_dump() if self.current_frame else None,
            "listener_count": len(self._listeners),
        }
