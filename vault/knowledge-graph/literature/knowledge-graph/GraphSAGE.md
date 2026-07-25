---
tags: [literature, 知识图谱, GraphSAGE, 论文]
---

# GraphSAGE（图采样与聚合）论文笔记

> **核心关注**：动态图的邻居采样方法

---

## 问题

GCN 需要整张图的邻接矩阵，无法处理动态增长的图。

## GraphSAGE 的方法

- **固定邻居数采样**：每个节点采样固定数量的邻居
- **聚合函数**：Mean / LSTM / Pooling
- **自身与邻居分开归一化**

## 与 Haibo 的关系

语义图需要动态扩展（新概念 → 新节点），GraphSAGE 的采样策略直接适用。

## 行动项

- ✅ 推荐先做：固定邻居数采样
- ✅ 推荐先做：自身 + 邻居分开归一化

## 来源

Hamilton, Ying & Leskovec (2017). Inductive Representation Learning on Large Graphs.
- [[GraphSAGE论文中文详读]]（原 vault 笔记）

*最后更新：2026-07-22*
