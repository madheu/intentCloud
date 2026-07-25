---
tags: [haibo, 实验, 盲区扫描, H-SG-1]
---

# 盲区扫描：H-SG-1 实验前的问题

> 动手之前需要先回答的关键问题

---

## 盲区1：海波没有 BriLLM，怎么做 Token 活动模式？

H-SG-1 原文提到"Token 活动模式"，但海波的实际能力是**围合特征向量**，不是 Token 活动向量聚类。

**回答**：用围合特征向量做。BriLLM Token 活动 = 海波围合计数，数学同构（WL 证明）。但 H-SG-1 的表述需澄清。

## 盲区2：静态 HowNet 图是否存在足够的结构差异？

WL 证明邻域结构不同 → 可以区分。

但 HowNet 中近义词的语义图可能**结构太相似**：

```
苹果[公司]：苹果 ─公司─企业─上市─产品
苹果[水果]：苹果 ─水果─食物─吃─甜
```

→ 重叠路径可能存在，需实验验证。

## 盲区3：上下文窗口选多大？

围合需要上下文词。太长引入噪声，太短信息不足。需要实验确定最优窗口。

## 关联

- [[haibo/overview/hypothesis-chain]]
- [[literature/knowledge-graph/WL]]
- [[literature/nlp-llm/brillm/billm-vs-haibo]]
- [[haibo/decisions/theory-review-sg-v0]]

*最后更新：2026-07-23*
