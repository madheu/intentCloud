---
tags: [concepts, NLP技术, 语义表示]
---

# 语义表示

> 如何在计算机中表示"意义"

---

## 三种范式

### 1. 静态词向量
每个词一个固定向量（Word2Vec, GloVe, 腾讯词向量）
- 优点：计算快，覆盖广
- 缺点：无法处理歧义（一个词一个向量）

### 2. 上下文表示
每个词在不同上下文中不同向量（BERT, GPT）
- 优点：可以区分歧义
- 缺点：计算量大，解释性差

### 3. 概念表示
用概念节点（不是词）作为语义单元
- Haibo 走的这条路
- 优点：结构化、可解释
- 挑战：概念库的构建和维护

## 关联

- [[haibo/overview/core-question|Haibo 的核心问题]]
- [[literature/nlp-llm/_index|NLP-LLM 表示空间]]
- [[concepts/nlp-techniques/word-embeddings|词嵌入]]

*最后更新：2026-07-22*
