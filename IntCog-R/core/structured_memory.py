# H17 R5: 结构化记忆
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

@dataclass
class OutputSummary:
    intent: str = ''; style: str = ''; emotion: str = ''
    constraints: list[str] = field(default_factory=list)
    length: int = 0; rhyme: bool = False; topic: str = ''
    timestamp: float = 0.0

@dataclass
class NarrativeChain:
    chain_id: str; items: list[OutputSummary] = field(default_factory=list)
    causal_anchor: str = ''; created: float = 0.0
    def add(self, item: OutputSummary): self.items.append(item)
    def weight(self) -> float: return sum(1 for _ in self.items) * 0.3

@dataclass
class DiscretePoint:
    summary: OutputSummary; weight: float = 0.5; timestamp: float = 0.0

class StructuredMemory:
    def __init__(self):
        self.chains: list[NarrativeChain] = []
        self.points: list[DiscretePoint] = []
        self._chain_counter = 0

    def add_output(self, summary: OutputSummary):
        now = datetime.now().timestamp()
        summary.timestamp = now
        # 尝试关联到已有叙事链
        for chain in self.chains:
            if self._is_related(summary, chain):
                chain.add(summary); return chain.chain_id
        # 尝试关联到离散点（升级为叙事链）
        for pt in self.points:
            if self._is_related(summary, pt.summary):
                self._chain_counter += 1
                new_chain = NarrativeChain(chain_id=f'chain_{self._chain_counter}',
                                           causal_anchor=summary.topic or 'auto',
                                           created=now)
                new_chain.add(pt.summary); new_chain.add(summary)
                self.points.remove(pt); self.chains.append(new_chain)
                return new_chain.chain_id
        # 作为离散点
        self._chain_counter += 1
        pt = DiscretePoint(summary=summary, weight=0.5, timestamp=now)
        self.points.append(pt); return f'point_{self._chain_counter}'

    def _is_related(self, a: OutputSummary, b: OutputSummary) -> bool:
        if a.topic and b.topic and a.topic == b.topic: return True
        if a.intent and b.intent and a.intent == b.intent: return True
        return False

    def search(self, topic: str = '', intent: str = '') -> list[OutputSummary]:
        results = []
        for chain in self.chains:
            for item in chain.items:
                if (topic and item.topic == topic) or (intent and item.intent == intent):
                    results.append((item, chain.weight()))
        results.sort(key=lambda x: x[1], reverse=True)
        for pt in self.points:
            if (topic and pt.summary.topic == topic) or (intent and pt.summary.intent == intent):
                results.append((pt.summary, pt.weight))
        return [r[0] for r in results]

    def decay(self):
        now = datetime.now().timestamp()
        for pt in self.points[:]:
            age = now - pt.timestamp
            pt.weight = max(0.0, pt.weight - age / 86400 * 0.1)

    def clear(self): self.chains.clear(); self.points.clear()
