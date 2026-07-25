---
tags: [literature, 计算语言学, HowNet, 论文]
---

# HowNet 义原综述笔记

> **核心主张**：义原（sememe）是 NLP 的「元素周期表」——不可再分的语义基元

---

## 什么是义原

HowNet（董振东，1999）用约 **2,000 个**义原描述所有汉语/英语词的概念。

例子：
- 「苹果（水果）」= {水果, 可食用, 圆, 甜/酸}
- 「苹果（公司）」= {公司, 科技, 品牌}

## 五个研究层级

```
义原数据库构建 (HowNet)
    ↓
义原表示学习 (Sememe Embedding)
    ↓
义原辅助词汇语义理解 (WSD, SSE)
    ↓
义原辅助句子级理解 (SRL, SPINN)
    ↓
义原应用 (情感分析, MT, KG)
```

## 与 Haibo 的关系

海波语义图与 HowNet 完全同构——都在寻找不可再分的「概念原子」。HowNet 的义原可以直接作为海波概念图的原子节点。

P0-B-1 尝试用 HowNet 做消歧（11%），问题不在 HowNet 本身，而在匹配算法。

## 来源

Liu et al. (2020). A Survey on Sememe-Based Natural Language Understanding.
- [[HowNet综述中文详读]]（原 vault 笔记）

## 关联

- [[literature/comp-ling/hownet/_index]]
- [[literature/comp-ling/wsd/_index]]
- [[haibo/semantics-graph/concept-nodes]]

*最后更新：2026-07-23*
