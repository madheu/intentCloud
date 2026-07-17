# H17 R2: 纠正与共情
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

class CorrectionType(Enum): EMOTIONAL = 'emotional'; CASUAL = 'casual'; CONTRADICTORY = 'contradictory'; SELF_CORRECT = 'self_correct'
class CorrectionDirection(Enum): REDUCE = 'reduce'; INCREASE = 'increase'; SWITCH = 'switch'

EMOTIONAL_MARKERS = ['心情不好','难受','伤心','难过','崩溃','受不了','不舒服','压抑','心情','太悲伤']
CASUAL_MARKERS = ['对了','顺便','挺好的','还可以','还行','差不多']
SELF_CORRECT_MARKERS = ['算了','说错了','刚才说错了','当我没说','不是','刚才不算']
CONTRADICTORY_PATTERN = ['但是','不过','然而','可是','却']

@dataclass
class CorrectionSignal:
    ctype: CorrectionType; target: str; value: float; direction: CorrectionDirection
    confidence: float = 0.5; source_text: str = ''

class CorrectionAnalyzer:
    @staticmethod
    @staticmethod
    def classify(text: str) -> tuple[CorrectionType, float]:
        if any(m in text for m in SELF_CORRECT_MARKERS): return CorrectionType.SELF_CORRECT, 0.9
        contradicted = sum(1 for m in CONTRADICTORY_PATTERN if m in text)
        if contradicted >= 1 and ('太' in text or '再' in text): return CorrectionType.CONTRADICTORY, 0.7
        if any(m in text for m in EMOTIONAL_MARKERS): return CorrectionType.EMOTIONAL, 0.6
        if any(m in text for m in CASUAL_MARKERS): return CorrectionType.CASUAL, 0.4
        return CorrectionType.CASUAL, 0.3
    @staticmethod
    def extract_target(text: str) -> tuple[str, CorrectionDirection, float]:
        for word, val in [('悲伤',0.3),('恐惧',0.3),('敬畏',0.3),('喜悦',0.3),('平静',0.3),('愤怒',0.3)]:
            if f'太{word}了' in text: return f'emotion_{word}', CorrectionDirection.REDUCE, val
            if f'再{word}' in text or f'喜欢{word}' in text: return f'emotion_{word}', CorrectionDirection.INCREASE, val
        return '', CorrectionDirection.REDUCE, 0.0

    @staticmethod
    def extract_boundary(text: str) -> tuple[float, float]:
        lo, hi = 0.0, 1.0
        for marker, val in [('太',-0.3),('再',+0.3),('别太',-0.2),('一点',+0.1),('夸张',+0.2),('收敛',-0.2)]:
            if marker in text:
                if val < 0: lo = max(lo, 1.0 + val)
                else: hi = min(hi, max(0.5, val))
        return lo, hi

    @staticmethod
    def generate_emotional_response() -> str:
        return '我理解你现在心情不太好。我没有情绪，但我能通过数据知道你的感受。要不要换一个更轻松的话题？'

    @staticmethod
    def generate_casual_clarification(text: str) -> str:
        return '你是觉得太长了想缩短一点？还是有其他方面想调整？可以告诉我具体方向。'

    @staticmethod
    def generate_clarification() -> str:
        return '我没太理解你的意思，能再说得具体一点吗？'
