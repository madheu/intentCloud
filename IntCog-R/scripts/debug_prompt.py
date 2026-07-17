# H19 R4: Prompt Inspection
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.graph_interpreter import GraphInterpreter
from core.prompt_builder import PromptBuilder

if __name__ == "__main__":
    gi = GraphInterpreter(); pb = PromptBuilder()
    inp = sys.argv[1] if len(sys.argv) > 1 else "你好"
    d = gi.interpret(inp, {})
    sp = pb.build(d)
    print("="*60)
    print("DECISION:")
    print(f"  intent={d.intent}, topic={d.topic}, tone={d.tone}")
    print(f"  emotion_bias={d.emotion_bias}, persona={d.persona}")
    print(f"  constraints={d.constraints}")
    print()
    print("SYSTEM PROMPT:")
    print(f"  {sp}")
    print(f"  (token: ~{pb.token_count(sp)})")
    print("="*60)
