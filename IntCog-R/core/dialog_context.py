"""对话状态追踪器 + 反馈信号捕获器 + 内化门控。

H16a: 多轮对话上下文管理（话题继承、情绪趋势、指代消解）
H16b: 反馈信号检测与暂存、内化门控（安全核心）
"""
from __future__ import annotations
import re
from collections import deque
from typing import Any

# ── 内化/清理关键词 ──────────────────────────────────────────────────────
INTERNALIZE_KEYWORDS = ["记住这个", "保存这个", "内化", "记住了", "记下来"]
CLEAR_KEYWORDS = ["重置", "清理记忆", "忘掉", "忘掉刚才说的", "重新开始"]
ANAPHORA_WORDS = {"它", "这个", "那个", "这些", "那些", "这里", "那里", "刚才", "上次"}

# ── 纠正信号 ─────────────────────────────────────────────────────────────
CORRECTION_PATTERNS = [
    # (类型, 正则, 方向)
    ("direction", re.compile(r"太[^了]+了"), "reduce"),
    ("intensity", re.compile(r"再[^一]+一?点"), "increase"),
    ("perspective", re.compile(r"不只是.+?还有"), "mix"),
    ("topic", re.compile(r"不是[^，,!！]+而是"), "switch"),
]

# ── 数据结构 ──────────────────────────────────────────────────────────────

class CorrectionSignal:
    """单条纠正信号。"""
    def __init__(self, ctype: str, target_node: str | None,
                 direction: str, raw_input: str):
        self.type = ctype
        self.target_node = target_node
        self.direction = direction
        self.raw_input = raw_input

    def __repr__(self) -> str:
        return (f"Correction(type={self.type}, target={self.target_node}, "
                f"dir={self.direction})")


class DialogContext:
    """对话状态追踪器。

    会话级，结束时必须调用 reset() 清空全部。
    """

    def __init__(
        self,
        max_topic_history: int = 5,
        max_emotion_history: int = 5,
        max_pending: int = 10,
    ):
        self.max_topic_history = max_topic_history
        self.max_emotion_history = max_emotion_history
        self.max_pending = max_pending
        self.reset()

    # ── 状态字段 ──────────────────────────────────────────────────────────

    def reset(self) -> None:
        """清空全部会话状态。会话结束时必须调用。"""
        self.current_topic: str | None = None
        self.topic_history: list[str] = []
        self.turn_count: int = 0
        self.user_emotion_trend: str = "neutral"
        self.emotion_history: list[str] = []
        self.pending_corrections: deque[CorrectionSignal] = deque()
        self.last_user_input: str = ""
        self.last_assistant_summary: str = ""

    # ── 话题管理 ──────────────────────────────────────────────────────────

    def update_topic(self, node_id: str | None) -> None:
        """更新当前话题。"""
        if node_id and node_id != self.current_topic:
            if self.current_topic:
                self.topic_history.append(self.current_topic)
                if len(self.topic_history) > self.max_topic_history:
                    self.topic_history.pop(0)
            self.current_topic = node_id
            self.turn_count = 0  # 新话题重置轮次
        self.turn_count += 1

    def resolve_anaphora(self, text: str) -> bool:
        """检测输入中是否存在指代词。"""
        for word in ANAPHORA_WORDS:
            if word in text:
                return True
        return False

    def detect_topic_switch(self, text: str, cloud: Any) -> str | None:
        """检测是否切换到新话题。

        遍历 shell 中的所有节点，如果文本中包含某个节点的文本片段
        且不是当前话题，则视为话题切换候选。
        """
        candidates = []
        for nid, node in cloud._shell.items():
            if node.text and node.text[:2] in text and nid != self.current_topic:
                candidates.append((nid, len(node.text)))
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
        return None

    # ── 情绪追踪 ──────────────────────────────────────────────────────────

    def update_emotion(self, activations: dict[str, float]) -> None:
        """基于当前激活值更新情绪趋势。"""
        fear = activations.get("emotion_fear", 0)
        nervous = activations.get("emotion_nervous", 0)
        calm = activations.get("emotion_calm", 0)
        joy = activations.get("emotion_joy", 0)
        awe = activations.get("emotion_awe", 0)

        anxiety_score = fear + nervous
        peace_score = calm + joy + awe

        if anxiety_score > peace_score + 0.3:
            trend = "anxious"
        elif peace_score > anxiety_score + 0.3:
            trend = "calm"
        else:
            trend = "neutral"

        if trend != self.user_emotion_trend:
            self.emotion_history.append(trend)
            if len(self.emotion_history) > self.max_emotion_history:
                self.emotion_history.pop(0)
        self.user_emotion_trend = trend

    # ── 反馈捕获（H16b 核心）─────────────────────────────────────────────

    def detect_correction(self, text: str, cloud: Any) -> list[CorrectionSignal]:
        """检测输入中的纠正信号，存入队列（不自动内化）。"""
        signals = []

        for ctype, pattern, direction in CORRECTION_PATTERNS:
            if pattern.search(text):
                target = self._extract_target(text, ctype, cloud)
                sig = CorrectionSignal(ctype, target, direction, text)
                signals.append(sig)

        # 否定/肯定词匹配
        for node_id, node in cloud._shell.items():
            if node.text is None:
                continue
            for neg in ["不要", "不喜欢", "不想", "别"]:
                if f"{neg}{node.text[:2]}" in text or f"{neg} {node.text[:2]}" in text:
                    signals.append(CorrectionSignal("direction", node_id, "reduce", text))
            for pos in ["喜欢", "要", "想要", "多"]:
                if f"{pos}{node.text[:2]}" in text or f"{pos} {node.text[:2]}" in text:
                    signals.append(CorrectionSignal("direction", node_id, "increase", text))

        # 去重
        seen = set()
        unique = []
        for s in signals:
            key = (s.type, s.target_node, s.direction)
            if key not in seen:
                seen.add(key)
                unique.append(s)

        for s in unique:
            self._enqueue(s)

        return unique

    def _extract_target(self, text: str, ctype: str, cloud: Any) -> str | None:
        """从输入中提取目标节点。"""
        for nid, node in cloud._shell.items():
            if node.text and node.text[:2] in text:
                return nid
        return None

    def _enqueue(self, signal: CorrectionSignal) -> None:
        """入队，超上限自动丢弃最旧。"""
        self.pending_corrections.append(signal)
        if len(self.pending_corrections) > self.max_pending:
            self.pending_corrections.popleft()

    def internalize_pending(self, cloud: Any, strength: float = 0.3) -> str:
        """将队列中的纠正信号打包写入海波权重。

        仅当用户明确说出内化关键词时才调用。
        """
        if not self.pending_corrections:
            return "无待内化的纠正信号"

        from core.intent_cloud_config import IntentCloudConfig
        cfg = IntentCloudConfig()

        logs = []
        while self.pending_corrections:
            sig = self.pending_corrections.popleft()
            if sig.target_node and sig.target_node in cloud._shell:
                target_node = cloud._shell[sig.target_node]
                if sig.direction == "increase":
                    # 提高节点初始trust
                    target_node.trust = min(1.0, target_node.trust + 0.1)
                    logs.append(f"增强 {sig.target_node}: trust={target_node.trust:.2f}")
                elif sig.direction == "reduce":
                    target_node.trust = max(0.0, target_node.trust - 0.1)
                    logs.append(f"减弱 {sig.target_node}: trust={target_node.trust:.2f}")

        self.clear_pending()
        return "; ".join(logs) if logs else "内化完成（无有效目标）"

    def clear_pending(self) -> str:
        """清空待处理纠正队列（权重不变）。"""
        count = len(self.pending_corrections)
        self.pending_corrections.clear()
        return f"已清空 {count} 条待处理纠正"

    # ── 对话上下文检测 ──────────────────────────────────────────────────

    def is_internalize_command(self, text: str) -> bool:
        return any(kw in text for kw in INTERNALIZE_KEYWORDS)

    def is_clear_command(self, text: str) -> bool:
        return any(kw in text for kw in CLEAR_KEYWORDS)

    # ── 摘要 ──────────────────────────────────────────────────────────────

    def get_summary(self) -> str:
        """生成当前对话上下文摘要。"""
        parts = []
        if self.current_topic:
            parts.append(f"话题: {self.current_topic}")
        parts.append(f"情绪: {self.user_emotion_trend}")
        parts.append(f"轮次: {self.turn_count}")
        if self.pending_corrections:
            parts.append(f"待处理纠正: {len(self.pending_corrections)}条")
        return " | ".join(parts)

    @property
    def pending_count(self) -> int:
        return len(self.pending_corrections)
