"""自适应强度调节器。

在双语者注入前调用，查询弹性地图，限制注入强度在安全区间内。
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

# 默认保守值
DEFAULT_MIN_STRENGTH = 0.01
DEFAULT_MAX_STRENGTH = 0.5
SAFETY_MARGIN = 0.8  # 窄路额外缩小系数


class StrengthRegulator:
    """注入强度调节器。

    用法:
        regulator = StrengthRegulator()
        adjusted, reason = regulator.regulate(
            source="nature_ocean", target="emotion_fear",
            requested_strength=5.0
        )
        # adjusted = 0.04, reason = "clamped to stable range [0.01, 0.05]"
    """

    def __init__(
        self,
        map_path: str | Path | None = None,
        default_min: float = DEFAULT_MIN_STRENGTH,
        default_max: float = DEFAULT_MAX_STRENGTH,
        safety_margin: float = SAFETY_MARGIN,
    ):
        self.default_min = default_min
        self.default_max = default_max
        self.safety_margin = safety_margin
        self._map: dict[str, Any] = {}

        if map_path is not None:
            self.load_map(map_path)

    def load_map(self, map_path: str | Path) -> bool:
        """加载弹性地图 JSON 文件。"""
        p = Path(map_path)
        if not p.exists():
            self._map = {}
            return False
        try:
            with open(p, "r", encoding="utf-8") as f:
                self._map = json.load(f)
            return True
        except (json.JSONDecodeError, IOError):
            self._map = {}
            return False

    def get_path_key(self, source: str, target: str) -> str:
        """构造路径键。"""
        return f"{source}->{target}"

    def regulate(
        self, source: str, target: str, requested_strength: float
    ) -> tuple[float, str]:
        """调节注入强度。

        Args:
            source: 源节点 ID
            target: 目标节点 ID
            requested_strength: 请求的注入强度

        Returns:
            (调整后的强度, 调整原因描述)
        """
        path_key = self.get_path_key(source, target)
        entry = self._map.get(path_key)

        if entry is None:
            # 未知路径：使用保守默认值
            adjusted = max(self.default_min, min(requested_strength, self.default_max))
            if adjusted != requested_strength:
                return adjusted, (
                    f"unknown path {path_key}, "
                    f"clamped to default [{self.default_min}, {self.default_max}]"
                )
            return adjusted, f"unknown path, within default [{self.default_min}, {self.default_max}]"

        stable = entry.get("stable_range")
        if stable is None:
            # 路径已知但无稳定区间：极端保守
            adjusted = max(self.default_min, min(requested_strength, self.default_max))
            return adjusted, f"path {path_key} has no stable range, clamped to defaults"

        lo, hi = stable

        # 对窄路应用安全系数
        path_type = entry.get("type", "")
        if path_type == "introvert":
            effective_hi = hi * self.safety_margin
        else:
            effective_hi = hi

        if requested_strength < lo:
            adjusted = lo
            return adjusted, (
                f"path {path_key} below stable range [{lo:.4f}, {hi:.4f}], "
                f"raised to lower bound"
            )
        elif requested_strength > effective_hi:
            adjusted = effective_hi
            return adjusted, (
                f"path {path_key} above stable range [{lo:.4f}, {hi:.4f}], "
                f"clamped to {effective_hi:.4f}"
            )
        else:
            return requested_strength, (
                f"path {path_key} within stable range [{lo:.4f}, {hi:.4f}], "
                f"no adjustment needed"
            )

    def get_map_summary(self) -> str:
        """返回弹性地图摘要。"""
        if not self._map:
            return "No map loaded"
        lines = []
        for pid, info in sorted(self._map.items()):
            sr = info.get("stable_range")
            oc = info.get("out_of_control_threshold")
            sr_s = f"[{sr[0]:.3f}, {sr[1]:.3f}]" if sr else "N/A"
            oc_s = f"{oc:.3f}" if oc else ">max"
            lines.append(f"  {pid:40s} stable={sr_s:20s}  threshold={oc_s}")
        return "\n".join(lines)

    @property
    def is_loaded(self) -> bool:
        return bool(self._map)

    @property
    def path_count(self) -> int:
        return len(self._map)
