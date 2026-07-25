---
tags: [literature, NLP, TokenGraph]
---

# Token Graph

> 以 Token 为节点的图结构分析

---

## 核心概念

- 每个 Token 是图中的一个节点
- 边表示 Token 之间的注意力关系或共现关系
- 分析 LLM 内部表示路径的图结构

## 与 Haibo 的关系

Token Graph 是粒度更细的表示。Haibo 在其之上做**抽象跃迁**：
Token → Semantic Node（概念级节点）。

## 关联

- [[literature/nlp-llm/brillm/billm-vs-haibo|Haibo vs BriLLM]]
- [[literature/nlp-llm/_index|NLP-LLM 表示空间]]

*最后更新：2026-07-22*
