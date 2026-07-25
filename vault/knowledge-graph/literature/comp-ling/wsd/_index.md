---
tags: [literature, 计算语言学, WSD]
---

# 词义消歧（WSD）

> Word Sense Disambiguation — 区分词在不同语境中的含义

---

## 关键论文

### [[literature/comp-ling/wsd/glossbert|GlossBERT]]
用 BERT 做 context-gloss 匹配。与 Haibo 的围合计数在数学上同构。

## 对 Haibo 的价值

WSD 直接对应 H-SG-1（语义分区）：如果上下文的词能告诉我们一个歧义词在哪个语义分区，那 WSD 就是验证语义分区假设的最直接工具。

## 关联

- [[haibo/overview/hypothesis-chain|假设链 H-SG-1]]
- [[haibo/semantics-graph/activation-counting|激活计数（与 GlossBERT 同构）]]

*最后更新：2026-07-22*
