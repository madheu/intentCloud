---
tags: [literature, 知识图谱, GAT, 论文]
---

# GAT（图注意力网络）论文笔记

> **核心贡献**：在图上引入自注意力机制，让节点动态学习哪些邻居更重要

---

## 要解决的问题

GCN 和 GraphSAGE 的共同盲点：**等权重假设**——所有邻居一视同仁。

| 方法 | 邻居权重策略 | 能区分谁更重要？ |
|------|-------------|:--------------:|
| GCN | 固定的归一化权重 | ❌ |
| GraphSAGE | 均匀采样 + 聚合 | ❌ |
| **GAT** | **每个邻居动态计算注意力权重 α_ij** | ✅ |

## 核心方法

1. **线性变换**：节点特征通过共享权重 W 变换
2. **注意力系数**：计算每一对相邻节点的相关性
3. **Softmax 归一化**：得到权重 α_ij
4. **多头注意力**：K 组独立的注意力拼接/平均

## 与 Haibo 的关系

围合计数目前是**等权相加**。GAT 的思路值得借鉴——但 GAT 靠反向传播学重要性，海波需要用赫布式局部信号来估计。

## 来源

Veličković et al. (2018). Graph Attention Networks. ICLR 2018.
- [[GAT论文中文详读]]（原 vault 笔记）

## 关联

- [[literature/knowledge-graph/GCN]]
- [[literature/knowledge-graph/GraphSAGE]]
- [[haibo/semantics-graph/activation-counting]]

*最后更新：2026-07-23*
