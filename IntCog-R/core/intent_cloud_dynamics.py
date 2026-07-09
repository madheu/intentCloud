"""意图云动力学系统：快慢分离的双层时间尺度演化。

阶段 1（快子系统）：激活扩散
  - 在固定权重下迭代扩散激活值
  - 概念锚点节点激活值保持不变
  - 激活值始终在 [0, a_max] 范围内
  - 收敛条件：max(|Δa|) < ε 或达到 max_iter

阶段 2（慢子系统）：权重更新
  - 激活值保持不变
  - 仅更新非锚点边的权重
  - 直接调用 Harness 6 的 update_weight() 函数

核心约束：快子系统必须先收敛才能触发慢子系统。
"""

from __future__ import annotations

from typing import Any


def diffuse_activation(
    cloud: "IntentCloud",
    input_signals: dict[str, float],
    config: "IntentCloudConfig",
) -> dict[str, float]:
    """快子系统：激活扩散。

    在固定权重下迭代扩散激活值，直到收敛或达到最大迭代次数。

    扩散公式：
        new_act[i] = old_act[i] + α × (input_signal[i] + sum(w[j→i] × act[j]))
        激活值被裁剪到 [0, a_max]

    收敛条件：
        max(|new_act[i] - old_act[i]| for all i) < ε

    Args:
        cloud: IntentCloud 实例
        input_signals: 外部输入信号（节点 ID → 初始激活值）
        config: IntentCloudConfig 配置（含 alpha、epsilon、max_iter）

    Returns:
        收敛后的各节点激活值字典

    Raises:
        ValueError: 如果输入信号包含 NaN 或无穷大值
    """
    # ── 输入校验 ─────────────────────────────────────────────────────────
    for node_id, act in input_signals.items():
        if isinstance(act, (float, int)):
            if __import__("math").isnan(act) or __import__("math").isinf(act):
                raise ValueError(
                    f"Input signal for '{node_id}' must not be NaN or infinite: {act}"
                )
        else:
            raise ValueError(
                f"Input signal for '{node_id}' must be numeric, got {type(act)}"
            )

    # ── 初始化激活值 ─────────────────────────────────────────────────────
    # 所有节点的激活值初始化为 0，然后被输入信号覆盖
    # 概念锚点节点的激活值从锚点系统获取
    activations: dict[str, float] = {}

    # 初始化锚点节点的固定激活值
    for node_id in cloud.anchor_system.node_ids:
        act = cloud.anchor_system.get_node_activation(node_id)
        if act is not None:
            activations[node_id] = act

    # 初始化其他节点为 0
    # 收集所有边涉及的节点
    all_nodes = set(activations.keys())
    for (src, tgt) in cloud._edges.keys():
        all_nodes.add(src)
        all_nodes.add(tgt)
    # 添加输入信号中的节点
    for node_id in input_signals.keys():
        all_nodes.add(node_id)

    # 非锚点节点初始化为 0
    for node_id in all_nodes:
        if node_id not in activations:
            activations[node_id] = 0.0

    # 输入信号只覆盖非锚点节点
    for node_id, act in input_signals.items():
        # 概念锚点节点的激活值不被输入信号覆盖
        if cloud.anchor_system.is_node_mutable(node_id):
            activations[node_id] = min(max(0.0, float(act)), config.a_max)

    # ── 迭代扩散 ─────────────────────────────────────────────────────────
    for _ in range(config.max_iter):
        new_activations = dict(activations)
        max_delta = 0.0

        for node_id in activations.keys():
            # 概念锚点节点激活值保持不变
            if not cloud.anchor_system.is_node_mutable(node_id):
                continue

            # 当前激活值
            old_act = activations[node_id]

            # 计算来自所有入边的激活输入
            incoming_sum = 0.0
            for (src, tgt), edge in cloud._edges.items():
                if tgt == node_id:
                    incoming_sum += edge.weight * activations.get(src, 0.0)

            # 输入信号
            input_signal = input_signals.get(node_id, 0.0)

            # 扩散更新：Δa = α × (input + incoming_sum - old_act)
            # 这样稳态解是：a[i] = input[i] + sum(w[j→i] × a[j])
            # 没有边时 incoming_sum=0，稳态时 a[i] = input[i]
            delta = config.alpha * (input_signal + incoming_sum - old_act)
            new_act = old_act + delta

            # 饱和限幅：裁剪到 [0, a_max]
            new_act = max(0.0, min(new_act, config.a_max))

            # 更新激活值
            new_activations[node_id] = new_act

            # 记录最大变化量
            max_delta = max(max_delta, abs(new_act - old_act))

        # 更新激活值
        activations = new_activations

        # 检查收敛条件
        if max_delta < config.epsilon:
            break

    return activations


def update_weights_after_diffusion(
    cloud: "IntentCloud",
    activations: dict[str, float],
    config: "IntentCloudConfig",
) -> None:
    """慢子系统：权重更新。

    在激活值收敛后，遍历所有非锚点边，调用 update_weight() 进行权重更新。
    拓扑锚点边被跳过，不会被修改。

    Args:
        cloud: IntentCloud 实例
        activations: 收敛后的各节点激活值（来自 diffuse_activation 的输出）
        config: IntentCloudConfig 配置（含 eta、gamma、K_d 等）

    Returns:
        None
    """
    # 遍历所有边，跳过锚点边
    for (src, tgt), edge in cloud._edges.items():
        # 拓扑锚点边跳过
        if not cloud.anchor_system.is_edge_mutable(src, tgt):
            continue

        # 获取激活值，未知节点默认为 0
        a_i = activations.get(src, 0.0)
        a_j = activations.get(tgt, 0.0)

        # 调用 Harness 6 的 update_weight() 函数
        from core.intent_cloud import update_weight

        update_weight(edge, a_i, a_j, config)