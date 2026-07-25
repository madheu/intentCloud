---
tags: [literature, 计算语言学, WSD, 论文]
---

# GlossBERT 论文笔记

> **核心发现**：context-gloss 匹配与 Haibo 围合计数在数学上同构

---

## 问题

用 BERT 判断一个歧义词在给定 context 中对应哪个 gloss（义项定义）。

## 方法

- 构造 (context, gloss) 对
- 用 BERT 编码并计算匹配得分
- 选择得分最高的 gloss 作为预测

## 与 Haibo 的关系

围合计数（上下文词 → 概念归属 → 激活计数）跟 gloss 匹配在数学上同构。
**结论**：这条路没有走歪。

## 来源

Huang et al. (2019). GlossBERT: BERT for Word Sense Disambiguation with Gloss Knowledge.
- [[GlossBERT论文中文详读]]（原 vault 笔记）

*最后更新：2026-07-22*
