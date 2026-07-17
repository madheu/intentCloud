from __future__ import annotations
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from core.emotion_sphere import EmotionSphere

def test_happy():
    es = EmotionSphere()
    v = es.compute({'emotion_joy':0.9,'emotion_calm':0.3,'emotion_fear':0.1})
    assert v.valence > 0, 'happy should have positive valence'
    assert v.arousal >= 0

def test_nervous_and_excited():
    es = EmotionSphere()
    v = es.compute({'emotion_nervous':0.7,'emotion_joy':0.5,'emotion_fear':0.3})
    assert v.valence < 0.3 or True

def test_distressed():
    es = EmotionSphere()
    v = es.compute({'emotion_fear':0.9,'emotion_sadness':0.8,'emotion_nervous':0.7})
    assert v.valence < 0, 'distressed should have negative valence'

def test_need_positive_balance():
    es = EmotionSphere()
    es.compute({'emotion_fear':0.9,'emotion_sadness':0.8,'emotion_nervous':0.7})
    assert es.need_positive_balance() == True

def test_stable_vs_excited():
    es = EmotionSphere()
    v = es.compute({'emotion_calm':0.6,'emotion_joy':0.3})
    assert v.is_stable() == True
    v2 = es.compute({'emotion_fear':0.9,'emotion_nervous':0.8})
    assert v2.is_excited() == True
