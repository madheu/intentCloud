# H17 R4: 情绪球
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import math

@dataclass
class EmotionVector:
    valence: float = 0.0  # -1 to 1 (negative to positive)
    arousal: float = 0.0  # 0 to 1 (calm to excited)
    magnitude: float = 0.0

    def is_stable(self) -> bool: return self.magnitude < 0.7
    def is_excited(self) -> bool: return self.magnitude >= 0.7

class EmotionSphere:
    def __init__(self):
        self.current: EmotionVector = EmotionVector()
        self.history: list[EmotionVector] = []
        self.primary_weight: float = 0.7
        self.secondary_weights: list[float] = [0.2, 0.1]

    def compute(self, activations: dict[str, float]) -> EmotionVector:
        fear = activations.get('emotion_fear', 0)
        nervous = activations.get('emotion_nervous', 0)
        calm = activations.get('emotion_calm', 0)
        joy = activations.get('emotion_joy', 0)
        awe = activations.get('emotion_awe', 0)
        sad = activations.get('emotion_sadness', 0)
        anger = activations.get('emotion_anger', 0)

        # Negative emotions weight
        neg = fear * 0.9 + sad * 0.7 + anger * 0.6 + nervous * 0.5
        pos = joy * 0.9 + calm * 0.3 + awe * 0.5

        valence = max(-1.0, min(1.0, pos - neg))
        arousal = max(0.0, min(1.0, fear * 0.6 + nervous * 0.7 + joy * 0.5 + anger * 0.8))
        magnitude = math.sqrt(valence**2 + arousal**2)

        vec = EmotionVector(valence=valence, arousal=arousal, magnitude=magnitude)
        self.current = vec
        self.history.append(vec)
        if len(self.history) > 20: self.history.pop(0)
        return vec

    def need_positive_balance(self) -> bool:
        return self.current.valence < -0.3 and self.current.arousal > 0.5

    def get_component_vector(self, activations: dict) -> dict[str, float]:
        components = {}
        for key in ['emotion_joy','emotion_calm','emotion_awe','emotion_fear','emotion_sadness','emotion_nervous','emotion_anger']:
            if key in activations and activations[key] > 0.1:
                components[key] = activations[key]
        return components

    def clear(self): self.current = EmotionVector(); self.history.clear()
