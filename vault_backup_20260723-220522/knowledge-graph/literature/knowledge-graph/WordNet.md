---
tags: [literature, 知识图谱, WordNet]
---

# WordNet（词汇数据库）

> 英文词汇的语义网络，按概念（synset）组织

---

## 核心结构

- **Synset（同义词集）**：一个概念，包含一组同义词
- **上下位关系**：hypernym（上位）/ hyponym（下位）
- **其他关系**：部分整体、反义、蕴含等

## 例子

```
{apple, edible_fruit} → hypernym: {fruit}
{apple, electronics_brand} → hypernym: {company}
```

## 对 Haibo 的价值

WordNet 的 synset 结构与 Haibo 的**概念节点**设计高度相似：
- Synset = 概念节点
- Word → Synset 归属 = Word → Concept 映射
- 上下位关系 = 概念层级结构

## 关联

- [[haibo/semantics-graph/concept-nodes|概念节点设计]]
- [[literature/knowledge-graph/_index|知识图谱]]

*最后更新：2026-07-22*
