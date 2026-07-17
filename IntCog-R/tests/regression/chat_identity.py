# H19 R5: 身份回归测试
from __future__ import annotations
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from core.graph_interpreter import GraphInterpreter
from core.prompt_builder import PromptBuilder

p=f=t=0
def chk(nm, ok, dt=""):
    global p,f,t; t+=1
    if ok: p+=1; print(f"  OK [{nm}]")
    else: f+=1; print(f"  FAIL [{nm}] {dt}")

gi = GraphInterpreter(); pb = PromptBuilder()

# 输入你好 -> Decision 不含重复追问
d = gi.interpret("\u4f60\u597d", {})
chk("greeting_short", d.intent=="greeting" and len(d.constraints)>=1)
sp = pb.build(d)
chk("greeting_no_repeat", "hello" not in sp and "hello" not in sp)

# 输入你是谁 -> Persona haibo
d2 = gi.interpret("\u4f60\u662f\u8c01", {})
chk("whoami_persona", d2.persona=="haibo")
sp2 = pb.build(d2)
chk("whoami_no_qwen", "\u901a\u4e49\u5343\u95ee" not in sp2 and "\u963f\u91cc\u4e91" not in sp2)
chk("whoami_is_haibo", "\u6d77\u6ce2" in sp2)

# 输入我又来了 -> intent=chat
d3 = gi.interpret("\u6211\u53c8\u6765\u4e86", {})
chk("context_aware", d3.intent=="chat")

# 输入天空 -> topic=天空
d4 = gi.interpret("\u5929\u7a7a\u662f\u600e\u6837\u5b58\u5728\u7684", {"nature_sky":0.8})
chk("sky_topic", d4.topic=="\u5929\u7a7a")
sp4 = pb.build(d4)
chk("sky_no_user_repeat", "\u5929\u7a7a\u662f\u600e\u6837\u5b58\u5728\u7684" not in sp4)

print(f"\\nResults: {p}/{t} pass, {f} fail")
