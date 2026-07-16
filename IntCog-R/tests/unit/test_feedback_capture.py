"""单元测试：反馈信号捕获与内化门控"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from core.dialog_context import DialogContext, CorrectionSignal, INTERNALIZE_KEYWORDS, CLEAR_KEYWORDS

def make_mock_cloud():
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
            }
    return MockCloud()


# ── 1. 纠正信号检测 ──────────────────────────────────────────────────────

def test_detect_too_sad():
    ctx = DialogContext()
    signals = ctx.detect_correction("太悲伤了", make_mock_cloud())
    assert len(signals) >= 1
    assert any(s.type == "direction" and s.direction == "reduce" for s in signals)


def test_detect_not_only():
    ctx = DialogContext()
    signals = ctx.detect_correction("不只是恐惧，还有敬畏", make_mock_cloud())
    assert any(s.type == "perspective" for s in signals)


def test_detect_negation():
    ctx = DialogContext()
    signals = ctx.detect_correction("不要悲伤", make_mock_cloud())
    assert any(s.direction == "reduce" and s.target_node == "emotion_sadness"
               for s in signals)


def test_detect_affirmation():
    ctx = DialogContext()
    signals = ctx.detect_correction("喜欢敬畏", make_mock_cloud())
    assert any(s.direction == "increase" and s.target_node == "emotion_awe"
               for s in signals)


# ── 2. 暂存不自动内化 ────────────────────────────────────────────────────

def test_correction_only_enqueued():
    """纠正信号只入队，不触发权重更新。"""
    cloud = make_mock_cloud()
    ctx = DialogContext()
    old_trust = cloud._shell["emotion_sadness"].trust

    ctx.detect_correction("太悲伤了", cloud)

    # 权重不变
    assert cloud._shell["emotion_sadness"].trust == old_trust
    # 但队列有内容
    assert ctx.pending_count > 0


# ── 3. 确认关键词触发内化 ────────────────────────────────────────────────

def test_internalize_keyword_detection():
    ctx = DialogContext()
    for kw in INTERNALIZE_KEYWORDS:
        assert ctx.is_internalize_command(f"请{kw}") == True, f"should detect: {kw}"


def test_internalize_pending_updates_weights():
    cloud = make_mock_cloud()
    ctx = DialogContext()
    ctx.detect_correction("太悲伤了", cloud)
    ctx.detect_correction("喜欢敬畏", cloud)

    old_sad = cloud._shell["emotion_sadness"].trust
    old_awe = cloud._shell["emotion_awe"].trust

    log = ctx.internalize_pending(cloud)

    # 权重确实变了
    assert cloud._shell["emotion_sadness"].trust < old_sad
    assert cloud._shell["emotion_awe"].trust > old_awe
    # 队列已清空
    assert ctx.pending_count == 0


# ── 4. 清理指令 ──────────────────────────────────────────────────────────

def test_clear_keyword_detection():
    ctx = DialogContext()
    for kw in CLEAR_KEYWORDS:
        assert ctx.is_clear_command(kw) == True, f"should detect: {kw}"


def test_clear_pending_no_weight_change():
    cloud = make_mock_cloud()
    ctx = DialogContext()
    ctx.detect_correction("太悲伤了", cloud)
    old_trust = cloud._shell["emotion_sadness"].trust

    ctx.clear_pending()

    assert ctx.pending_count == 0
    assert cloud._shell["emotion_sadness"].trust == old_trust  # 权重不变


# ── 5. 队列上限保护 ──────────────────────────────────────────────────────

def test_queue_max_limit():
    ctx = DialogContext(max_pending=3)
    for i in range(5):
        ctx._enqueue(CorrectionSignal("direction", f"node_{i}", "reduce", str(i)))
    assert ctx.pending_count == 3  # 最旧的2条被丢弃
    # 最旧的应该是 node_2 (因为前2条被丢)
    assert ctx.pending_corrections[0].target_node == "node_2"
