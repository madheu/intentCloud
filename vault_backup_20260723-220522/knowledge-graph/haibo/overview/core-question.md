---
tags: [haibo, 核心问题]
---

# 核心问题：区分歧义的最小维度是什么？

> Haibo 2.0 的核心驱动力问题

---

## 问题表述

能否从 Token 活动路径中**无监督**形成稳定、可复用、持久的 **Semantic Node**？

具体化为：**区分歧义的最小维度是什么？**

## 背景

1.0 路线的 embedding 注入无法控制 LLM（类间/类内比 0.93 和 0.99），2026-07-18 审查委员会决定关闭 1.0，转向 2.0。

## 关键区分

- **BriLLM** — 已覆盖 Token 节点传播（信号全连接流 SiFu）
- **Haibo 创新** — Token → Semantic Node 跃迁（不是 Token Graph）
- 图节点 = **语义概念**（科技类/水果类），**不是**词 Token
- Word → Concept 是归属关系（belonging），不是等价关系

## 关联

- [[haibo/overview/philosophy|核心理念]]
- [[haibo/overview/hypothesis-chain|假设链]]
- [[literature/nlp-llm/brillm/billm-vs-haibo|Haibo 与 BriLLM 的边界]]

*最后更新：2026-07-22*
