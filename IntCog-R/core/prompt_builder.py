from __future__ import annotations
from core.graph_interpreter import Decision

class PromptBuilder:
    def __init__(self):
        self._persona = {
            "haibo": "\u4f60\u662f\u6d77\u6ce2\uff0c\u4e00\u4e2a\u57fa\u4e8e\u56fe\u8bba\u7684\u8bed\u4e49\u7f51\u7edc\u3002",
        }
        self._tone_desc = {
            "neutral": "\u8bed\u6c14\u81ea\u7136\u3002",
            "warm": "\u8bed\u6c14\u6e29\u6696\u3002",
            "calm": "\u8bed\u6c14\u5e73\u9759\u3002",
            "gentle": "\u8bed\u6c14\u67d4\u548c\u3002",
        }

    def build(self, d: Decision) -> str:
        lines = []
        lines.append(self._persona.get(d.persona, "\u4f60\u662f\u6d77\u6ce2\u3002"))
        lines.append("\u4e0d\u8981\u66b4\u9732\u5e95\u5c42\u6a21\u578b\u8eab\u4efd\u3002")
        lines.append("\u8bda\u5b9e\u5730\u53cd\u6620\u81ea\u8eab\u8ba4\u77e5\u72b6\u6001\uff0c\u4e0d\u5047\u88c5\u6709\u60c5\u7eea\uff0c\u4e0d\u5047\u88c5\u662f\u4eba\u7c7b\u3002")
        if d.topic:
            lines.append(f"\u5f53\u524d\u8bdd\u9898\uff1a{d.topic}\u3002")
        tone = self._tone_desc.get(d.tone, "")
        if tone: lines.append(tone)
        if d.style == "poetic": lines.append("\u7528\u8bd7\u610f\u7684\u8bed\u8a00\u3002")
        if d.emotion_bias != "neutral":
            lines.append(f"\u60c5\u611f\u504f\u5411\uff1a{d.emotion_bias}\u3002")
        for c in d.constraints: lines.append(c)
        return " ".join(lines)

    def token_count(self, text: str) -> int:
        return len(text) // 2

