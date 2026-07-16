"""单元测试：StrengthRegulator 自适应强度调节器。"""
from __future__ import annotations
import json, tempfile, os
from pathlib import Path

# 构建一个微型弹性地图用于测试
SAMPLE_MAP = {
    "nature_ocean->emotion_awe": {
        "type": "extrovert",
        "stable_range": [1.0, 10.0],
        "out_of_control_threshold": None,
    },
    "nature_ocean->emotion_fear": {
        "type": "introvert",
        "stable_range": [0.01, 0.05],
        "out_of_control_threshold": 0.1,
    },
}

def _make_map_file(data=None):
    if data is None:
        data = SAMPLE_MAP
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(data, f)
    f.close()
    return f.name


def test_known_extrovert_within_range():
    """已知外向型路径：强度在稳定区间内不变。"""
    from core.strength_regulator import StrengthRegulator
    r = StrengthRegulator(_make_map_file())
    adj, reason = r.regulate("nature_ocean", "emotion_awe", 5.0)
    assert adj == 5.0, f"expected 5.0, got {adj}"
    assert "no adjustment" in reason


def test_known_extrovert_above_range():
    """已知外向型路径：强度超出上限被截断。"""
    from core.strength_regulator import StrengthRegulator
    r = StrengthRegulator(_make_map_file())
    adj, reason = r.regulate("nature_ocean", "emotion_awe", 20.0)
    assert adj == 10.0, f"expected 10.0, got {adj}"
    assert "clamped" in reason


def test_known_extrovert_below_range():
    """已知外向型路径：强度低于下限被拉高。"""
    from core.strength_regulator import StrengthRegulator
    r = StrengthRegulator(_make_map_file())
    adj, reason = r.regulate("nature_ocean", "emotion_awe", 0.01)
    assert adj == 1.0, f"expected 1.0, got {adj}"
    assert "raised" in reason


def test_known_introvert_within_range():
    """已知内向型路径（恐惧）：强度在稳定区间内，但安全系数缩小上限。"""
    from core.strength_regulator import StrengthRegulator
    r = StrengthRegulator(_make_map_file(), safety_margin=0.8)
    adj, reason = r.regulate("nature_ocean", "emotion_fear", 0.03)
    assert adj == 0.03, f"expected 0.03, got {adj}"
    assert "within" in reason


def test_known_introvert_above_range():
    """已知内向型路径：强度超出上限（含安全系数）被截断。"""
    from core.strength_regulator import StrengthRegulator
    r = StrengthRegulator(_make_map_file(), safety_margin=0.8)
    adj, reason = r.regulate("nature_ocean", "emotion_fear", 5.0)
    # effective_hi = 0.05 * 0.8 = 0.04
    assert round(adj, 6) == 0.04, f"expected 0.04, got {adj}"
    assert "clamped" in reason


def test_unknown_path():
    """未知路径：使用保守默认值。"""
    from core.strength_regulator import StrengthRegulator
    r = StrengthRegulator(_make_map_file())
    adj, reason = r.regulate("unknown_src", "unknown_tgt", 10.0)
    assert adj == 0.5, f"expected 0.5, got {adj}"
    assert "unknown path" in reason


def test_unknown_path_weak_signal():
    """未知路径弱信号：默认最小值保护。"""
    from core.strength_regulator import StrengthRegulator
    r = StrengthRegulator(_make_map_file())
    adj, reason = r.regulate("unknown_src", "unknown_tgt", 0.001)
    assert adj == 0.01, f"expected 0.01, got {adj}"
    assert "unknown path" in reason


def test_map_file_missing():
    """弹性地图文件缺失时：降级为全部使用默认值。"""
    from core.strength_regulator import StrengthRegulator
    r = StrengthRegulator("/nonexistent/path.json")
    assert r.is_loaded == False
    adj, reason = r.regulate("anything", "anything", 10.0)
    assert adj == 0.5, f"expected 0.5, got {adj}"


def test_introvert_safety_margin():
    """验证内向型路径的安全系数生效。"""
    from core.strength_regulator import StrengthRegulator
    r = StrengthRegulator(_make_map_file(), safety_margin=0.5)
    adj, reason = r.regulate("nature_ocean", "emotion_fear", 0.05)
    # effective_hi = 0.05 * 0.5 = 0.025
    assert adj == 0.025, f"expected 0.025, got {adj}"


def test_custom_defaults():
    """自定义默认值。"""
    from core.strength_regulator import StrengthRegulator
    r = StrengthRegulator(_make_map_file(), default_min=0.1, default_max=1.0)
    adj, reason = r.regulate("unknown", "unknown", 10.0)
    assert adj == 1.0, f"expected 1.0, got {adj}"
