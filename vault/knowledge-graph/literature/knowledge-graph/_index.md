---
tags: [literature, 知识图谱]
---

# 知识表示与知识图谱

> 语义网络、知识图谱、图神经网络、图上推理

---

## 子域

### [[literature/knowledge-graph/GCN|GCN（图卷积网络）]]
图上的消息传递机制。→ Haibo 关注其归一化公式。

### [[literature/knowledge-graph/GraphSAGE|GraphSAGE（图采样与聚合）]]
动态图的邻居采样方法。→ 固定邻居数采样 + 分开归一化。

### [[literature/knowledge-graph/GAT|GAT（图注意力网络）]]
动态注意力权重取代等权相加。→ 多头围合思路来源。

### [[literature/knowledge-graph/MPNN|MPNN（消息传递框架）]]
统一 GNN 三段式框架。→ 海波是 MPNN 特例。

### [[literature/knowledge-graph/TransE|TransE（知识图谱嵌入）]]
将知识图谱中的实体和关系嵌入到连续向量空间。

### [[literature/knowledge-graph/WL|WL（Weisfeiler-Lehman 图核）]]
图同构测试的线性时间算法。→ 围合计数的数学基础。

### [[literature/knowledge-graph/WordNet|WordNet（词汇数据库）]]
英文词汇的语义网络，概念的上下位关系组织。

## 对 Haibo 的价值

- GCN → H-SG-4 持久状态的图学习基础
- GraphSAGE → 动态图扩展的采样策略
- GAT → 差异化邻居权重的思路
- MPNN → 统一框架的定位
- TransE → H-SG-3 关系形成的嵌入方法
- WL → 围合计数的数学同构验证
- WordNet → 概念层级组织的参考

*最后更新：2026-07-23*
