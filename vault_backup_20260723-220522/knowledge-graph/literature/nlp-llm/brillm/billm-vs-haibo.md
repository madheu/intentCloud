---
tags: [literature, NLP, BriLLM, Haibo对比]
---

# Haibo 与 BriLLM 的边界与关系

> 明确两个项目的分工与创新边界

---

## 各自覆盖

| 层面 | BriLLM | Haibo |
|:----:|:------:|:-----:|
| 节点 | Token | 语义概念 |
| 边 | 全连接信号流 | 语义关系 |
| 目标 | 分析 LLM 表示 | 形成可复用语义结构 |
| 粒度 | 细（Token 级） | 粗（概念级） |

## Haibo 的创新

**Token → Semantic Node 跃迁**。BriLLM 没有这一步。

## 关联

- [[haibo/overview/core-question|核心问题]]
- [[haibo/decisions/token-vs-concept-node|节点类型定义]]

*最后更新：2026-07-22*
