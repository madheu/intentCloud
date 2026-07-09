"""意图云拓扑演化配置。

所有可调参数集中在此 dataclass，确保参数可追溯、可单元测试。
每个参数都有明确的物理含义和默认值，默认值经过七份控制论分析的交叉验证。

快子系统（激活扩散）参数：
  - alpha: 激活更新步长
  - epsilon: 收敛阈值（相邻迭代激活变化量 < ε 视为收敛）
  - max_iter: 最大扩散迭代次数（防止无限循环）

慢子系统（权重更新）参数：
  - eta: 赫布学习率
  - gamma: 参考模型拉回系数
  - K_d: 动量阻尼系数
  - delta: 死区阈值
  - w_min/w_max: 权重投影边界
  - a_max: 激活值饱和上界
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IntentCloudConfig:
    """意图云拓扑演化参数配置。

    所有参数均为冻结不可变，避免运行中被意外修改。
    需要不同配置时，创建新实例即可。

    Attributes:
        alpha: 快子系统激活更新步长，控制每次迭代的激活变化幅度。
            过大可能导致振荡，过小则收敛太慢。
        epsilon: 快子系统收敛阈值，当所有节点的激活变化量的最大值
            小于此值时，认为扩散达到稳态。
        max_iter: 快子系统最大扩散迭代次数，防止扩散过程无限循环。
        eta: 赫布学习率，控制共现相关性对权重的影响强度。
            过高会导致权重爆炸，过低则学习太慢。
        gamma: 参考模型拉回系数，控制权重向 w_ref 回归的速度。
            相当于"弹性系数"，越大锚定越强。
        K_d: 动量阻尼系数，控制对高频振荡的抑制强度。
            越大惯性越强，对突变越不敏感。
        delta: 死区阈值，激活乘积的绝对值低于此值时跳过赫布项。
            用于过滤噪声共现。
        w_min: 权重下界，投影算子的硬裁剪下限。
        w_max: 权重上界，投影算子的硬裁剪上限。
        a_max: 激活值饱和上界，输入激活值被截断到此值。
    """

    # ── 快子系统参数（激活扩散）───────────────────────────────────────────

    alpha: float = 0.1
    """快子系统激活更新步长 α。"""

    epsilon: float = 0.001
    """快子系统收敛阈值 ε。"""

    max_iter: int = 100
    """快子系统最大扩散迭代次数。"""

    # ── 慢子系统参数（权重更新）───────────────────────────────────────────

    eta: float = 0.1
    """赫布学习率 η。"""

    gamma: float = 0.01
    """参考模型拉回系数 γ。"""

    K_d: float = 0.3
    """动量阻尼系数 K_d。"""

    delta: float = 0.01
    """死区阈值 δ。"""

    w_min: float = 0.0
    """权重下界。"""

    w_max: float = 1.0
    """权重上界。"""

    a_max: float = 1.0
    """激活值饱和上界。"""

    def __post_init__(self) -> None:
        """构造后校验：确保参数在合理范围内。"""
        # 快子系统参数校验
        if self.alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {self.alpha}")
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {self.epsilon}")
        if self.max_iter <= 0:
            raise ValueError(f"max_iter must be > 0, got {self.max_iter}")

        # 慢子系统参数校验
        if self.eta < 0:
            raise ValueError(f"eta must be >= 0, got {self.eta}")
        if self.gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {self.gamma}")
        if self.K_d < 0:
            raise ValueError(f"K_d must be >= 0, got {self.K_d}")
        if self.delta < 0:
            raise ValueError(f"delta must be >= 0, got {self.delta}")
        if self.w_min > self.w_max:
            raise ValueError(
                f"w_min ({self.w_min}) must be <= w_max ({self.w_max})"
            )
        if self.a_max < 0:
            raise ValueError(f"a_max must be >= 0, got {self.a_max}")