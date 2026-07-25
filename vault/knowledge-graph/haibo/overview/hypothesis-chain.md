---
tags: [haibo, 假设链]
---

# 假设链 H-SG-1~5

> Haibo 2.0 的五个核心假设，逐步递进

---

```
H-SG-1 语义分区     ← 计算语言学(WSD) + NLP(表示空间)
H-SG-2 节点固化     ← 认知语言学(原型) + 知识表示(语义网络)
H-SG-3 关系形成     ← 知识表示(知识图谱嵌入) + 计算语言学(FrameNet)
H-SG-4 持久状态     ← 知识表示(图学习)
H-SG-5 下游价值     ← NLP(LLM 接口)
```

## H-SG-1：语义分区
Token 活动模式中，不同语义的概念占据不同的子空间区域。
→ [[literature/comp-ling/wsd/_index|WSD]] + [[literature/nlp-llm/_index|表示空间]]

## H-SG-2：节点固化
反复出现的语义分区会固化为稳定的概念节点（原型效应）。
→ [[literature/cog-ling/barsalou-perceptual-symbols|知觉符号]] + [[literature/knowledge-graph/_index|语义网络]]

## H-SG-3：关系形成
概念节点之间形成语义关系（上下位、因果等）。
→ [[literature/knowledge-graph/TransE|知识图谱嵌入]] + [[literature/comp-ling/framenet/_index|FrameNet]]

## H-SG-4：持久状态
语义图在无监督条件下维持稳定状态（不发散）。
→ [[literature/knowledge-graph/GCN|图学习]]

## H-SG-5：下游价值
形成的语义图能提升 LLM 的消歧或生成质量。
→ NLP(LLM 接口)

## 关联

- [[haibo/overview/core-question|核心问题]]
- [[文献库索引]]（原 vault 文献索引）

*最后更新：2026-07-22*
