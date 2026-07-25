---
tags: [haibo, 学术地图, 定位]
---

# 学术地图：海波在语义研究中的位置

> 三轴坐标系定位海波与其他方法的关系

---

## 轴1：训练需求（无训练 ←→ 需要训练）

```
无训练 ←——————————————→ 需要训练
  │                            │
  WL   HowNet  海波    Word2Vec  TransE  GCN   LLM
```

**海波**：最左端，跟 WL、HowNet 同侧——这是根本立场。

## 轴2：语义表示方式（离散 ←→ 连续）

```
离散 ←——————————————→ 连续
  │                        │
 HowNet  WL  海波    Word2Vec  GCN/TransE  LLM
```

**海波**：偏左——概念节点离散，激活强度连续（混合）。

## 轴3：上下文敏感度（静态 ←→ 动态）

**海波**：居中偏动态——围合计数随上下文词改变，但图结构是静态的。

## 对 Haibo 的定位意义

海波在学术地图上的独特位置是：**最左端方法中唯一做动态消歧的**——WL 和 HowNet 都是静态分析，海波引入了上下文驱动。

## 关联

- [[haibo/overview/core-question]]
- [[haibo/overview/hypothesis-chain]]
- [[literature/knowledge-graph/WL]]
- [[literature/comp-ling/hownet/_index]]

*最后更新：2026-07-23*
