import pytest

from core.memory import InMemoryMemory, MemoryEntry, TieredMemory


@pytest.mark.asyncio
async def test_store_and_retrieve():
    mem = InMemoryMemory()
    await mem.store(MemoryEntry(text="I like Python"))
    await mem.store(MemoryEntry(text="I enjoy hiking"))
    results = await mem.retrieve("Python", ["coding"], top_k=2)
    assert len(results) == 2
    assert "Python" in results[0].text


@pytest.mark.asyncio
async def test_retrieve_respects_intent_goals():
    mem = InMemoryMemory()
    await mem.store(MemoryEntry(text="Python is great for AI"))
    await mem.store(MemoryEntry(text="I had coffee this morning"))
    results = await mem.retrieve("coding", ["Python", "AI"], top_k=1)
    assert "Python" in results[0].text


@pytest.mark.asyncio
async def test_time_decay():
    from datetime import datetime, timedelta, timezone

    mem = InMemoryMemory(time_decay_lambda=10.0)
    old = MemoryEntry(text="old memory", timestamp=datetime.now(timezone.utc) - timedelta(seconds=10))
    new = MemoryEntry(text="new memory")
    await mem.store(old)
    await mem.store(new)
    results = await mem.retrieve("memory", [], top_k=2)
    assert results[0].text == "new memory"


@pytest.mark.asyncio
async def test_tiered_memory_hot_to_warm_overflow():
    """三层缓存：热层溢出到温层。"""
    mem = TieredMemory(hot_limit=3, warm_limit=100)
    for i in range(5):
        await mem.store(MemoryEntry(text=f"memory {i}"))
    assert len(mem.hot) == 3
    assert len(mem.warm) == 2


@pytest.mark.asyncio
async def test_tiered_memory_warm_to_disk_overflow():
    """三层缓存：温层溢出到磁盘。"""
    import tempfile, os

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        disk_path = f.name

    try:
        mem = TieredMemory(hot_limit=2, warm_limit=3, disk_path=disk_path)
        for i in range(6):
            await mem.store(MemoryEntry(text=f"memory {i}"))
        assert len(mem.hot) == 2
        assert len(mem.warm) == 3
        import json
        with open(disk_path) as f:
            cold = json.load(f)
        assert len(cold) >= 1
    finally:
        os.unlink(disk_path)


@pytest.mark.asyncio
async def test_tiered_memory_retrieve_across_tiers():
    """三层缓存：跨热层和温层检索。"""
    mem = TieredMemory(hot_limit=2, warm_limit=5)
    await mem.store(MemoryEntry(text="I like Python programming"))
    await mem.store(MemoryEntry(text="I enjoy hiking in mountains"))
    await mem.store(MemoryEntry(text="Python is great for AI"))
    await mem.store(MemoryEntry(text="Coffee is my favorite drink"))
    results = await mem.retrieve("Python", ["coding", "AI"], top_k=2)
    assert len(results) == 2
    assert any("Python" in r.text for r in results)
