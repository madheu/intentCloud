---
tags: [haibo, 决策, 语义图]
---

# 节点类型定义：概念 vs 词

> 语义图的第一设计决策

---

## 教训

P0.8（词 Token 节点 + 共现频率边）→ 全部"不确定" ❌
P0.9（概念节点 + 层级归属）→ 10/11 正确 ✅

## 决策

语义图的节点 = **语义概念**（科技类/水果类），**不是**词 Token。
Word → Concept 是归属关系（belonging），不是等价关系。

## 含义

- 一个词可能归属多个概念（歧义来源）
- 一个概念可能包含多个词
- 需要在 word 层和 concept 层之间建立映射表

## 关联

- [[haibo/semantics-graph/concept-nodes|概念节点设计]]
- [[haibo/semantics-graph/word-concept-mapping|Word→Concept 映射]]
- [[literature/nlp-llm/brillm/billm-vs-haibo|Haibo vs BriLLM]]

*最后更新：2026-07-22*
