---
tags: [concepts, NLP技术]
---

# NLP 核心技术

> 自然语言处理的核心技术概念

---

## 核心概念

### 词义消歧（WSD）
确定一个歧义词在给定上下文中的正确义项。
→ [[literature/comp-ling/wsd/_index|WSD 文献索引]]

### 语义表示
如何在计算机中表示词、句子、文本的语义。
→ 静态词向量（Word2Vec / GloVe）
→ 上下文表示（BERT / GPT）
→ 概念表示（Concept Net / HowNet）

### 词嵌入（Word Embeddings）
将词汇映射到密集向量空间的技术。
→ 腾讯词向量、Word2Vec、GloVe

### 评估方法论
如何正确评估 NLP 系统。
→ 训练/开发/测试集划分
→ 基线（MFS）
→ 统计显著性

## 对 Haibo 的价值

P0 系列的教训表明：理解和避免 NLP 评估中的常见陷阱（数据泄漏、基线比较）至关重要。

*最后更新：2026-07-22*
