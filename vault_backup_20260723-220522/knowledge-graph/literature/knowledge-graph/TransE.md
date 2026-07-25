---
tags: [literature, 知识图谱, TransE, 论文]
---

# TransE（知识图谱嵌入）论文笔记

> 将知识图谱中的实体和关系嵌入到连续向量空间

---

## 核心思想

对于知识图谱中的三元组 (h, r, t)：
```
h + r ≈ t
```
头实体向量 + 关系向量 ≈ 尾实体向量。

## 例子

(北京, 首都, 中国) → 北京向量 + 首都向量 ≈ 中国向量

## 对 Haibo 的价值

- 语义图中的关系（上下位、因果）可以用嵌入向量表示 —— 但海波不需要学习嵌入，而是利用 HowNet 的预定义关系类型
- **关系类型系数传播**：TransE 的 h+r≈t 启发海波为每种关系类型分配差异化的传播系数 α(r)
- **关系基数分类**：1-to-1/1-to-Many/Many-to-1/Many-to-Many 分类可用于优化围合策略

## 更详细的阅读

[[TransE论文中文详读]] — 完整的中文详读笔记（含方法论决策卡片、实验方案、与海波语义图的深层联系分析）

## 来源

Bordes et al. (2013). Translating Embeddings for Modeling Multi-relational Data. NIPS 2013.

## 关联

- [[haibo/overview/hypothesis-chain|假设链 H-SG-3]]
- [[literature/knowledge-graph/_index|知识图谱]]
- [[TransE论文中文详读]]

*最后更新：2026-07-22*
