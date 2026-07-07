import pytest

from main import CogStreamEngine


@pytest.mark.asyncio
async def test_engine_runs_ticks():
    engine = CogStreamEngine(tick_ms=10)
    await engine.ingest("hello")

    winners = []
    async for winner in engine.run(ticks=5):
        winners.append(winner)

    assert len(winners) == 5
    # 第一个 winner 应是用户输入感知帧
    assert any(w.frame_type.value == "user_input" for w in winners)


@pytest.mark.asyncio
async def test_engine_stores_memories():
    engine = CogStreamEngine(tick_ms=10)
    await engine.ingest("hello")

    async for _ in engine.run(ticks=3):
        pass

    assert len(engine.memory._fallback.entries) > 0


@pytest.mark.asyncio
async def test_engine_snapshot():
    engine = CogStreamEngine(tick_ms=10)
    await engine.ingest("hi")
    async for _ in engine.run(ticks=2):
        pass
    snap = engine.snapshot()
    assert snap["tick_id"] >= 2
    assert "workspace" in snap
