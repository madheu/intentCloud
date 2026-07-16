"""视角转换器（认知重评机制）。

H15a: 基础视角扩展（适用所有路径）
H15b: 恐惧链专属视角转换（用生命/壮丽/敬畏对冲恐惧）
"""
from __future__ import annotations
from typing import Any
from core.strength_regulator import StrengthRegulator

# 恐惧链专属视角混合比例
FEAR_CHAIN_WEIGHTS = {
    "original": 0.40,   # 恐惧保留 40%
    "life": 0.25,       # 生命视角 25%
    "wide": 0.20,       # 壮丽视角 20%
    "awe": 0.15,        # 敬畏视角 15%
}

# 恐惧链强制视角映射：源头节点 → 应该引入的对冲节点
FEAR_PERSPECTIVE_MAP = {
    "nature_ocean": ["nature_life", "nature_wide", "emotion_awe"],
    "concept_loneliness": ["emotion_calm", "nature_sky", "emotion_joy"],
}


class PerspectiveConverter:
    """视角转换器。

    接收当前激活节点和边权重，生成多维引导向量。
    对恐惧路径执行专属深度转换，对其他路径执行基础扩展。
    """

    def __init__(
        self,
        regulator: StrengthRegulator | None = None,
        mix_ratio: float = 0.7,
        fear_chain_weights: dict[str, float] | None = None,
    ):
        self.regulator = regulator or StrengthRegulator()
        self.mix_ratio = mix_ratio
        self.fear_chain_weights = fear_chain_weights or dict(FEAR_CHAIN_WEIGHTS)

    def convert(
        self,
        source_node: str,
        target_node: str,
        base_activations: dict[str, float],
        cloud: Any,
        requested_strength: float,
        mode: str = "auto",
    ) -> dict[str, Any]:
        """执行视角转换。

        Args:
            source_node: 源节点 ID
            target_node: 目标节点 ID
            base_activations: process_interaction 的收敛激活值
            cloud: IntentCloud 实例
            requested_strength: 请求的注入强度

        Returns:
            { "activations": 调整后的激活值字典,
              "strength": 调节后的注入强度,
              "log": 转换日志 }
        """
        path_key = f"{source_node}->{target_node}"
        path_info = self.regulator._map.get(path_key, {})
        path_type = path_info.get("type", "unknown")

        # 强度调节
        adjusted_strength, strength_reason = self.regulator.regulate(
            source_node, target_node, requested_strength
        )

        is_fear = (path_type == "introvert" and source_node in FEAR_PERSPECTIVE_MAP)
        if mode == "fear_chain" or (mode == "auto" and is_fear):
            return self._fear_chain_convert(
                source_node, target_node, base_activations, cloud,
                adjusted_strength
            )
        else:
            return self._basic_convert(
                source_node, target_node, base_activations, cloud,
                adjusted_strength, path_key
            )

    def _basic_convert(
        self, src, tgt, activations, cloud, strength, path_key
    ) -> dict[str, Any]:
        """H15a: 基础视角扩展。"""
        result = dict(activations)
        log_parts = [f"path={path_key}"]

        # 查找替代视角节点
        alternatives = []
        for nid in cloud._shell:
            if nid == tgt:
                continue
            edge = cloud._edges.get((src, nid))
            if edge and edge.weight > 0.2:
                alternatives.append((nid, edge.weight))

        alternatives.sort(key=lambda x: x[1], reverse=True)

        if alternatives:
            mix_weight = 1.0 - self.mix_ratio
            top_alt = alternatives[:2]
            alt_count = len(top_alt)
            for i, (alt_id, w) in enumerate(top_alt):
                alt_activation = mix_weight / alt_count * 2.0
                if alt_id in result:
                    result[alt_id] = max(result[alt_id], alt_activation)
                else:
                    result[alt_id] = alt_activation
                log_parts.append(f"alt={alt_id}({alt_activation:.2f})")

        log_parts.append(f"strength={strength:.3f}")

        # 主方向保持原值
        if tgt in result and result[tgt] > self.mix_ratio:
            result[tgt] = self.mix_ratio * result[tgt]

        return {
            "activations": result,
            "strength": strength,
            "log": " | ".join(log_parts),
        }

    def _fear_chain_convert(
        self, src, tgt, activations, cloud, strength
    ) -> dict[str, Any]:
        """H15b: 恐惧链专属视角转换。"""
        result = dict(activations)
        log_parts = [f"path={src}->{tgt} (fear_chain)"]

        # 压低声恐惧节点激活值
        if tgt in result:
            orig_val = result[tgt]
            result[tgt] = orig_val * self.fear_chain_weights["original"]
            log_parts.append(f"fear_reduced: {orig_val:.3f}->{result[tgt]:.3f}")

        # 引入对冲视角
        perspectives = FEAR_PERSPECTIVE_MAP.get(src, [])
        for i, pid in enumerate(perspectives):
            weight_key = list(self.fear_chain_weights.keys())[i + 1]
            weight = self.fear_chain_weights.get(weight_key, 0.15)
            node = cloud._shell.get(pid)
            if node:
                if pid in result:
                    result[pid] = max(result[pid], weight * 2.0)
                else:
                    result[pid] = weight * 2.0
                log_parts.append(f"{pid}({weight:.2f})")

        log_parts.append(f"strength={strength:.3f}")
        return {
            "activations": result,
            "strength": strength,
            "log": " | ".join(log_parts),
        }
