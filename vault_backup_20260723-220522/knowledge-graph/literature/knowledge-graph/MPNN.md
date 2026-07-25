---
tags: [literature, 知识图谱, MPNN, 论文]
---

# MPNN（消息传递神经网络）论文笔记

> **核心贡献**：将各种 GNN 统一到 Message → Update → Readout 三段式框架

---

## MPNN 统一框架

**消息传递阶段**（T 步迭代）：

```
m_v^{t+1} = Σ_{w∈N(v)} M_t(h_v^t, h_w^t, e_{vw})   // Message
h_v^{t+1} = U_t(h_v^t, m_v^{t+1})                     // Update
```

**读出阶段**：

```
ŷ = R({ h_v^T | v ∈ G })                              // Readout
```

## 覆盖的模型

GCN、GraphSAGE、GG-NN、Interaction Networks 等全部是 MPNN 的特例，区别只在 M_t / U_t / R 的具体实现。

## 与 Haibo 的关系

MPNN 框架给海波语义图一个清晰的定位：**海波是 Message 函数为"围合计数"、Update 函数为"概念激活"、Readout 为"最高激活义项"的特例**。

## 来源

Gilmer et al. (2017). Neural Message Passing for Quantum Chemistry. ICML 2017.
- [[MPNN论文中文详读]]（原 vault 笔记）

## 关联

- [[literature/knowledge-graph/GCN]]
- [[literature/knowledge-graph/GraphSAGE]]
- [[literature/knowledge-graph/GAT]]
- [[haibo/semantics-graph/activation-counting]]

*最后更新：2026-07-23*
