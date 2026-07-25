---
tags: [haibo, 决策, 激活扩散, 改造]
---

# 激活扩散改造清单

> 基于 9 篇论文方法论，形成围合计数的完整改造方案

---

## 当前状态

```
score(义项) = Σ_j w_ij · activation(node_j)
w_ij = 1/deg(i)
```

| 维度 | 当前做法 | 问题 |
|------|---------|------|
| 邻居权重 | 等权 1/deg(i) | 度大的节点主导激活 |
| 传播层数 | 单层（1跳） | 错过2跳语义 |
| 轮次 | 单轮 | 无法细化邻域结构 |
| 边类型 | 无区分 | 上下位/属性/因果混为一谈 |

## 改造项（按优先级）

| 编号 | 改造 | 来源论文 | 工期 |
|:----:|------|---------|:----:|
| R1 | 对称重整化归一化 | GCN | 3h |
| R2 | 自身/邻居分开归一化 | GraphSAGE | 2h |
| R3 | 固定邻居采样 | GraphSAGE | 4h |
| R4 | 多层传播（2跳） | GCN/MPNN | 6h |
| R5 | 多头围合 | GAT | 8h |
| R6 | 边类型加权 | FrameNet | 6h |

## 关联

- [[haibo/semantics-graph/activation-counting]]
- [[literature/knowledge-graph/GCN]]
- [[literature/knowledge-graph/GraphSAGE]]
- [[literature/knowledge-graph/GAT]]
- [[literature/knowledge-graph/MPNN]]

*最后更新：2026-07-23*
