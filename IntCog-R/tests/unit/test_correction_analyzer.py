from __future__ import annotations
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from core.correction_analyzer import CorrectionAnalyzer, CorrectionType

def test_classify_emotional():
    ct, conf = CorrectionAnalyzer.classify('太悲伤了。我今天心情本来就不好')
    assert ct == CorrectionType.EMOTIONAL

def test_classify_casual():
    ct, conf = CorrectionAnalyzer.classify('挺好的。对了，那首诗好像有点长')
    assert ct == CorrectionType.CASUAL

def test_classify_contradictory():
    ct, conf = CorrectionAnalyzer.classify('太悲伤了，但是再悲伤一点')
    assert ct == CorrectionType.CONTRADICTORY

def test_classify_self_correct():
    ct, conf = CorrectionAnalyzer.classify('太敬畏了……算了，还是悲伤一点吧')
    assert ct == CorrectionType.SELF_CORRECT

def test_extract_boundary():
    lo, hi = CorrectionAnalyzer.extract_boundary('太悲伤了，但是再悲伤一点')
    assert lo >= 0.0 and hi <= 1.0

def test_generate_emotional():
    resp = CorrectionAnalyzer.generate_emotional_response()
    assert len(resp) > 10

def test_generate_clarification():
    resp = CorrectionAnalyzer.generate_clarification()
    assert '?' in resp or '吗' in resp
