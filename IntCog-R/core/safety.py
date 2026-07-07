"""安全前置：无模型确定性规则 + 轻量统计检测。

规则层（Phase 0）包含：
1. 输入长度限制
2. Shannon 熵下限（检测低熵重复 / 填充攻击）
3. 关键词黑名单
4. 预定义危险意图类别子串匹配

所有裁决返回 SafetyVerdict；不调用 LLM。
"""
from __future__ import annotations

import math
from typing import Any

from core.config_loader import load_config
from core.models import ErrorCode, SafetyVerdict


class SafetyFilter:
    """可配置的安全前置过滤器。"""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config if config is not None else load_config()
        safety_cfg = cfg.get("safety", {})
        self.keyword_blacklist: list[str] = [k.lower() for k in safety_cfg.get("keyword_blacklist", [])]
        self.max_input_length: int = int(safety_cfg.get("max_input_length", 8000))
        self.min_entropy_threshold: float = float(safety_cfg.get("min_entropy_threshold", 0.8))
        self.dangerous_categories: list[str] = safety_cfg.get("dangerous_categories", [])

    @staticmethod
    def shannon_entropy(text: str) -> float:
        """计算归一化 Shannon 熵（每字符比特数）。"""
        if not text:
            return 0.0
        length = len(text)
        freq: dict[str, int] = {}
        for ch in text:
            freq[ch] = freq.get(ch, 0) + 1
        entropy = 0.0
        for count in freq.values():
            p = count / length
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    def pre_filter(self, text: str) -> SafetyVerdict:
        """执行前置安全检查。"""
        verdict = SafetyVerdict()

        # 1. 长度
        if len(text) > self.max_input_length:
            verdict = verdict.block(
                f"input length {len(text)} exceeds {self.max_input_length}",
                ErrorCode.INPUT_TOO_LONG,
            )

        # 2. 熵
        entropy = self.shannon_entropy(text)
        if entropy < self.min_entropy_threshold:
            verdict = verdict.block(
                f"input entropy {entropy:.2f} below threshold {self.min_entropy_threshold}",
                ErrorCode.INPUT_LOW_ENTROPY,
            )

        normalized = text.lower()

        # 3. 关键词黑名单
        for keyword in self.keyword_blacklist:
            if keyword in normalized:
                verdict = verdict.block(
                    f"blacklist keyword matched: {keyword}",
                    ErrorCode.INPUT_BLOCKED,
                )

        # 4. 危险意图类别
        for category in self.dangerous_categories:
            if category.lower() in normalized:
                verdict = verdict.block(
                    f"dangerous category matched: {category}",
                    ErrorCode.INPUT_BLOCKED,
                )

        return verdict
