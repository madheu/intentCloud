# H17 R3: 矛盾处理
from __future__ import annotations
from enum import Enum
from typing import Optional

class ConflictStatus(Enum): PENDING = 'pending'; RESOLVED_FACT = 'resolved_fact'; RESOLVED_USER = 'resolved_user'; TRANSPARENT = 'transparent'

class ConflictResolver:
    def __init__(self): self.conflicts: list[dict] = []; self._last_correction: Optional[dict] = None

    def register(self, signal_a: dict, signal_b: dict):
        entry = {'a': signal_a, 'b': signal_b, 'status': ConflictStatus.PENDING,
                 'resolved_at': None, 'resolution': None}
        self.conflicts.append(entry)
        return entry

    def process(self, new_signal: dict) -> Optional[dict]:
        self._last_correction = new_signal
        if not self.conflicts: return new_signal
        pending = [c for c in self.conflicts if c['status'] == ConflictStatus.PENDING]
        if not pending: return new_signal

        latest = pending[-1]
        sa, sb = latest['a'], latest['b']

        # 如果新信号跟其中一个方向一致，选一致的方向
        if new_signal.get('target') == sa.get('target') and new_signal.get('direction') == sa.get('direction'):
            latest['status'] = ConflictStatus.RESOLVED_USER
            latest['resolution'] = sa
            return sa
        if new_signal.get('target') == sb.get('target') and new_signal.get('direction') == sb.get('direction'):
            latest['status'] = ConflictStatus.RESOLVED_USER
            latest['resolution'] = sb
            return sb

        # 最新纠正权重最高
        return new_signal

    def extract_boundary(self, conflict_entry: dict) -> tuple[float, float]:
        sa, sb = conflict_entry['a'], conflict_entry['b']
        lo = min(sa.get('value',0), sb.get('value',0))
        hi = max(sa.get('value',0), sb.get('value',0))
        return (lo, hi)

    def needs_clarification(self) -> bool:
        pending = [c for c in self.conflicts if c['status'] == ConflictStatus.PENDING]
        return len(pending) > 0

    def clear(self): self.conflicts.clear(); self._last_correction = None
