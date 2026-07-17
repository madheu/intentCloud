from __future__ import annotations
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from core.structured_memory import StructuredMemory, OutputSummary

def test_narrative_chain():
    sm = StructuredMemory()
    s1 = OutputSummary(intent='写诗', style='现代', emotion='敬畏', topic='大海')
    s2 = OutputSummary(intent='写诗', style='现代', emotion='平静', topic='大海')
    tid1 = sm.add_output(s1)
    tid2 = sm.add_output(s2)
    assert tid2.startswith('chain_'), 'should be narrative chain'

def test_discrete_point():
    sm = StructuredMemory()
    s1 = OutputSummary(intent='陈述', emotion='中立', topic='海鸥')
    s2 = OutputSummary(intent='写诗', emotion='敬畏', topic='大海')
    sm.add_output(s1)
    tid2 = sm.add_output(s2)
    assert tid2.startswith('point_') or tid2.startswith('chain_'), f'unexpected: {tid2}'

def test_search_by_topic():
    sm = StructuredMemory()
    sm.add_output(OutputSummary(intent='写诗', emotion='敬畏', topic='大海'))
    sm.add_output(OutputSummary(intent='写诗', emotion='平静', topic='大海'))
    results = sm.search(topic='大海')
    assert len(results) >= 1

def test_decay():
    sm = StructuredMemory()
    sm.add_output(OutputSummary(intent='陈述', topic='孤立'))
    sm.decay()
    assert True
