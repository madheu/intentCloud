---
tags: [literature, 知识图谱, WL, 图同构, 论文]
---

# WL（Weisfeiler-Lehman 图核）论文笔记

> **核心贡献**：将 WL 图同构测试变为实用的图特征提取器

---

## WL 算法（1维）

每轮迭代：
1. **多集标签确定** — 收集邻居标签
2. **排序拼接** — 自身标签 + 排序后的邻居标签串
3. **标签压缩** — 相同串 → 相同新标签
4. **重新标记** — 更新节点标签

**复杂度**：O(hm)，h 轮迭代，m 条边。

## 与 Haibo 的惊人联系

第 i 轮压缩标签 lᵢ(v) 恰好对应**以 v 为根、高度为 i 的子树模式**。

论文原文明确指出：
> "the compressed labels lᵢ(v) correspond to subtree patterns of height i rooted at v"

这意味着：**WL 围合计数与 Haibo 的围合计数在数学上同构**。

## 对 Haibo 的验证价值

WL 的数学保证：如果语义图上两个节点的邻域结构不同，围合计数一定不同。这为围合消歧提供了坚实的理论基础。

## 来源

Shervashidze et al. (2011). Weisfeiler-Lehman Graph Kernels. JMLR.
- [[WL图同构中文详读]]（原 vault 笔记）
- **阅读目的**：探索 WL 算法与 Haibo 围合计数之间的结构等价性

## 关联

- [[haibo/semantics-graph/activation-counting]]
- [[literature/knowledge-graph/GCN]]
- [[literature/knowledge-graph/MPNN]]
- [[haibo/experiments/blind-spot-hsg1]]

*最后更新：2026-07-23*
