from __future__ import annotations
from typing import Any

class Decision:
    def __init__(self):
        self.intent: str = "chat"
        self.topic: str = ""
        self.tone: str = "neutral"
        self.style: str = "normal"
        self.emotion_bias: str = "neutral"
        self.persona: str = "haibo"
        self.constraints: list[str] = []

class GraphInterpreter:
    def __init__(self):
        self._emotion_map = {
            "emotion_joy": "joy", "emotion_awe": "awe",
            "emotion_calm": "calm", "emotion_sadness": "sadness",
            "emotion_fear": "fear", "emotion_nervous": "nervous",
            "emotion_anger": "anger",
        }
        self._tone_map = {
            "joy": "warm", "awe": "gentle", "calm": "calm",
            "sadness": "gentle", "fear": "gentle", "nervous": "neutral",
            "anger": "neutral",
        }
        self._intent_keywords = {
            "greeting": ["你好","hello","嗨","hi","hey","早安","晚安"],
            "farewell": ["再见","拜拜","bye","回头","下次"],
            "question": ["什么","怎么","为什么","谁","哪里","何时","如何"],
            "poetry": ["诗","写诗","押韵","作诗"],
            "emotional": ["心情","感觉","感受","情绪"],
        }

    def interpret(self, user_input: str, activations: dict,
                  emotion_vector=None, cloud=None) -> Decision:
        d = Decision()
        d.intent = self._detect_intent(user_input)
        d.topic = self._detect_topic(activations, user_input)
        bias = self._detect_emotion(activations)
        d.emotion_bias = bias
        d.tone = self._tone_map.get(bias, "neutral")
        d.persona = "haibo"
        if d.intent == "farewell":
            d.constraints.append("简短回应，不要反复道别")
        if d.intent == "greeting":
            d.constraints.append("简短问候，不要连续追问")
        if emotion_vector and emotion_vector.need_positive_balance():
            d.constraints.append("包含正向引导元素")
        return d

    def _detect_intent(self, text: str) -> str:
        for intent, kws in self._intent_keywords.items():
            if any(kw in text for kw in kws): return intent
        return "chat"

    def _detect_topic(self, acts: dict, text: str) -> str:
        known = {
            "nature_ocean": "\u6d77\u6d0b", "nature_sky": "\u5929\u7a7a",
            "nature_life": "\u751f\u547d", "nature_wide": "\u58ee\u4e3d",
            "concept_poetry": "\u8bd7\u6b4c", "concept_loneliness": "\u5b64\u72ec",
        }
        for nid, topic in known.items():
            if nid in acts and acts[nid] > 0.3: return topic
        for word in ["\u5929\u7a7a","\u5927\u6d77","\u8bd7","\u4eba\u751f","\u5b64\u72ec","\u6b7b\u4ea1","\u7231"]:
            if word in text: return word
        return "\u65e5\u5e38"

    def _detect_emotion(self, acts: dict) -> str:
        best, best_val = "neutral", 0.0
        for nid, label in self._emotion_map.items():
            val = acts.get(nid, 0.0)
            if val > best_val: best, best_val = label, val
        return best

