import pytest

from core.memory import InMemoryMemory, MemoryEntry


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
