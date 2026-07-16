"""单元测试：DialogContext 对话状态追踪器"""
from __future__ import annotations
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from core.dialog_context import DialogContext, CorrectionSignal

def make_mock_cloud():
    """构造一个含常用节点的模拟 IntentCloud。"""
    class MockNode:
        def __init__(self, nid, text):
            self.id = nid
            self.text = text
            self.trust = 0.5
    class MockCloud:
        def __init__(self):
            self._shell = {
                "emotion_fear": MockNode("emotion_fear", "恐惧"),
                "emotion_calm": MockNode("emotion_calm", "平静"),
                "emotion_joy": MockNode("emotion_joy", "喜悦"),
                "emotion_awe": MockNode("emotion_awe", "敬畏"),
                "emotion_sadness": MockNode("emotion_sadness", "悲伤"),
                "nature_ocean": MockNode("nature_ocean", "大海"),
                "nature_sky": MockNode("nature_sky", "天空"),
            }
    return MockCloud()


# ── 1. 话题继承 ──────────────────────────────────────────────────────────

def test_topic_update():
    ctx = DialogContext()
    ctx.update_topic("nature_ocean")
    assert ctx.current_topic == "nature_ocean"
    assert ctx.turn_count == 1

    ctx.update_topic("nature_ocean")
    assert ctx.turn_count == 2  # 同一话题累加

    ctx.update_topic("nature_sky")  # 切换话题
    assert ctx.current_topic == "nature_sky"
    assert ctx.turn_count == 1  # 重置


def test_topic_history():
    ctx = DialogContext(max_topic_history=2)
    ctx.update_topic("a")
    ctx.update_topic("b")
    ctx.update_topic("c")
    ctx.update_topic("d")
    assert len(ctx.topic_history) == 2
    assert "a" not in ctx.topic_history  # 最旧的被丢弃


def test_anaphora_detection():
    ctx = DialogContext()
    assert ctx.resolve_anaphora("它是什么意思") == True
    assert ctx.resolve_anaphora("这个怎么用") == True
    assert ctx.resolve_anaphora("大海真美") == False


def test_topic_switch_detection():
    ctx = DialogContext()
    cloud = make_mock_cloud()
    # 没有当前话题时，不检测
    switched = ctx.detect_topic_switch("我喜欢大海", cloud)
    # 应该检测到大话题
    assert switched is None or switched == "nature_ocean"


# ── 2. 情绪趋势 ──────────────────────────────────────────────────────────

def test_emotion_trend_calm():
    ctx = DialogContext()
    ctx.update_emotion({"emotion_fear": 0.1, "emotion_nervous": 0.1,
                        "emotion_calm": 0.9, "emotion_joy": 0.5})
    assert ctx.user_emotion_trend == "calm"


def test_emotion_trend_anxious():
    ctx = DialogContext()
    ctx.update_emotion({"emotion_fear": 0.9, "emotion_nervous": 0.8,
                        "emotion_calm": 0.1, "emotion_joy": 0.1})
    assert ctx.user_emotion_trend == "anxious"


# ── 3. 会话重置 ──────────────────────────────────────────────────────────

def test_reset_clears_all():
    ctx = DialogContext()
    ctx.update_topic("nature_ocean")
    ctx.pending_corrections.append(CorrectionSignal("direction", "x", "reduce", "test"))
    ctx.reset()
    assert ctx.current_topic is None
    assert len(ctx.topic_history) == 0
    assert len(ctx.pending_corrections) == 0
    assert ctx.user_emotion_trend == "neutral"
    assert ctx.turn_count == 0


# ── 4. 摘要生成 ──────────────────────────────────────────────────────────

def test_summary():
    ctx = DialogContext()
    ctx.update_topic("nature_ocean")
    s = ctx.get_summary()
    assert "nature_ocean" in s
    assert "neutral" in s
