---
tags: [knowledge-graph, 入口, 导航]
---

# 知识图谱入口

> 原子知识碎片（Atomic Knowledge Graph）
> 每个文件只讲一件事，通过 `[[双向链接]]` 串联

---

## 四大知识域

### [[haibo/_index|Haibo 2.0 项目]]
核心假设链、实验记录、设计决策、语义图设计、学术定位
- [[haibo/overview/core-question|核心问题]]
- [[haibo/overview/hypothesis-chain|假设链 H-SG-1~5]]
- [[haibo/overview/academic-map|学术地图]]
- [[haibo/experiments/p0-series/_index|P0 系列实验]]
- [[haibo/experiments/blind-spot-hsg1|盲区扫描]]
- [[haibo/decisions/_index|关键决策]]
- [[haibo/decisions/activation-diffusion-refactor|激活扩散改造清单]]

### [[literature/_index|学术文献]]
四个支柱 + 新补论文的原子笔记
- 计算语言学（WSD / FrameNet / HowNet / [[literature/comp-ling/hownet/survey|HowNet综述]]）
- 认知语言学（Barsalou / Lakoff / Rosch）
- 知识表示与知识图谱（GCN / GAT / GraphSAGE / MPNN / TransE / WL / WordNet）
- NLP-LLM 表示空间（BriLLM / Token Graph）

### [[ai-dev/_index|AI 开发知识库]]
33 本书的原子化条目，按类别组织
- 数学基础 / 控制论 / AI 核心 / AI 工程 / 编程工具

### [[concepts/_index|跨域基础概念]]
认知科学、NLP 技术的核心概念
- 感知符号系统、原型理论、隐喻
- 词义消歧、语义表示、嵌入

---

## 链接关系速查

| 从 | 到 | 关系 |
|:---|:---|:-----|
| [[haibo/overview/hypothesis-chain]] | [[literature/cog-ling/barsalou-perceptual-symbols]] | H-SG-2 的认知语言学支撑 |
| [[haibo/overview/hypothesis-chain]] | [[literature/knowledge-graph/GCN]] | H-SG-4 的图学习支撑 |
| [[haibo/experiments/p0-series/p0.4-irrelevant-concepts]] | [[haibo/decisions/p0-series-conclusion]] | 实验结论 → 决策 |
| [[haibo/semantics-graph/concept-nodes]] | [[literature/nlp-llm/brillm/billm-vs-haibo]] | Haibo 与 BriLLM 的边界 |
| [[concepts/cognitive-science/perceptual-symbol-systems]] | [[haibo/decisions/perceptual-weight]] | 理论 → 设计决策 |
| [[haibo/semantics-graph/activation-counting]] | [[literature/knowledge-graph/WL]] | 围合计数与 WL 数学同构 |
| [[haibo/experiments/blind-spot-hsg1]] | [[literature/nlp-llm/brillm/billm-vs-haibo]] | 盲区1：BriLLM 与围合的关系 |
| [[haibo/decisions/activation-diffusion-refactor]] | [[literature/knowledge-graph/GAT]] | R5 多头围合借鉴 GAT |

---

*最后更新：2026-07-23*
