from __future__ import annotations
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from core.conflict_resolver import ConflictResolver, ConflictStatus

def test_register_conflict():
    cr = ConflictResolver()
    a = {'target':'emotion_fear','direction':'reduce','value':0.3}
    b = {'target':'emotion_fear','direction':'increase','value':0.5}
    cr.register(a, b)
    assert len(cr.conflicts) == 1

def test_process_new_signal():
    cr = ConflictResolver()
    a = {'target':'emotion_fear','direction':'reduce','value':0.3}
    b = {'target':'emotion_fear','direction':'increase','value':0.5}
    cr.register(a, b)
    result = cr.process({'target':'emotion_fear','direction':'reduce','value':0.6})
    assert result['direction'] == 'reduce'

def test_extract_boundary():
    cr = ConflictResolver()
    a = {'target':'emotion_sadness','direction':'reduce','value':0.7}
    b = {'target':'emotion_sadness','direction':'increase','value':0.9}
    entry = cr.register(a, b)
    lo, hi = cr.extract_boundary(entry)
    assert lo == 0.7 and hi == 0.9

def test_needs_clarification():
    cr = ConflictResolver()
    assert cr.needs_clarification() == False
    cr.register({'target':'a'},{'target':'a'})
    assert cr.needs_clarification() == True
