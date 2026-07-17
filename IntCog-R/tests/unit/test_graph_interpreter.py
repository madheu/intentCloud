# H19 unit: GraphInterpreter
from __future__ import annotations
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from core.graph_interpreter import GraphInterpreter

def test_ocean_awe_emotion():
    gi = GraphInterpreter()
    d = gi.interpret("\u5199\u5927\u6d77", {"nature_ocean":0.8,"emotion_awe":0.7})
    assert d.emotion_bias == "awe", f"got {d.emotion_bias}"
    assert d.persona == "haibo"

def test_no_activation():
    gi = GraphInterpreter()
    d = gi.interpret("\u4f60\u597d", {})
    assert d.intent == "greeting", f"got {d.intent}"
    assert d.emotion_bias == "neutral"

def test_no_node_text_in_decision():
    gi = GraphInterpreter()
    d = gi.interpret("\u6211\u5f88\u60b2\u4f24", {"emotion_sadness":0.9})
    s = str(d.__dict__)
    assert "\u60b2\u4f24" not in s, f"found node text in: {s}"

def test_farewell():
    gi = GraphInterpreter()
    d = gi.interpret("\u518d\u89c1", {})
    assert d.intent == "farewell"
    assert "short" in d.constraints[0] or "\u77ed" in d.constraints[0] or "\u7b80" in d.constraints[0]
