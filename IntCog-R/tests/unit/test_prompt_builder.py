# H19 unit: PromptBuilder
from __future__ import annotations
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from core.graph_interpreter import Decision
from core.prompt_builder import PromptBuilder

def test_no_node_text():
    pb = PromptBuilder(); d = Decision()
    d.intent = "chat"; d.topic = "\u6d77\u6d0b"; d.emotion_bias = "awe"
    sp = pb.build(d)
    assert "\u5927\u6d77" not in sp, f"found ocean in: {sp}"
    assert "\u656c\u754f" not in sp, f"found awe in: {sp}"

def test_haibo_identity():
    pb = PromptBuilder()
    sp = pb.build(Decision())
    assert "\u6d77\u6ce2" in sp, f"identity missing in: {sp}"

def test_no_underlying_model():
    pb = PromptBuilder()
    sp = pb.build(Decision())
    assert "\u4e0d\u8981\u66b4\u9732" in sp, f"no expose constraint in: {sp}"

def test_under_200_tokens():
    pb = PromptBuilder(); d = Decision()
    d.topic = "\u5929\u7a7a"; d.emotion_bias = "calm"; d.tone = "calm"
    d.constraints = ["\u5305\u542b\u6b63\u5411\u5f15\u5bfc"]
    sp = pb.build(d)
    tok = pb.token_count(sp)
    assert tok <= 200, f"token count {tok} > 200"
