---
tags: [haibo, 研究概述]
---

# Haibo (IntentCloud) Research Overview

> 最后更新：2026-07-22 | 阶段：P0-R3 草案阶段
> 1.0 路线关闭，2.0 核心问题：**区分歧义的最小维度是什么？**

---

## Project Location
- **Root:** `E:\intentCloud\`
- **Codebase:** `E:\intentCloud\IntCog-R\`
- **Research:** `E:\intentCloud\Haibo-Research\`
- **Obsidian Vault:** `E:\intentCloud\vault\`
- **Decision Log:** `E:\intentCloud\Haibo-Research\05_Decision_Log.md`

## Core Philosophy
> **语言不是思维本身，是思维在低维空间的投影。**

## Current Direction (Haibo 2.0)

**Core Question:** 能否从 Token 活动路径中无监督形成稳定、可复用、持久的 Semantic Node？

**具体化为**：区分歧义的最小维度是什么？

### Key Distinction
- **BriLLM** — 已覆盖 Token 节点传播（信号全连接流 SiFu）
- **Haibo innovation** — Token → Semantic Node 跃迁（不是 Token Graph）
- **Graph nodes = semantic concepts**（科技类/水果类），**not** word tokens
- Word → concept is a belonging relationship

### Hypothesis Chain (H-SG-1~5)
详见 `Haibo-Research/01_Core_Hypothesis.md`
```
H-SG-1 语义分区 ← 计算语言学(WSD) + NLP(表示空间)
H-SG-2 节点固化 ← 认知语言学(原型) + 知识表示(语义网络)
H-SG-3 关系形成 ← 知识图谱嵌入 + FrameNet
H-SG-4 持久状态 ← 图学习
H-SG-5 下游价值 ← LLM 接口
```

---

## Haibo 1.0 (Closed)

**路线**：手编 36 节点语义图 → Embedding 注入 → LLM 生成方向控制

**关闭原因**（2026-07-18 审查委员会结论）：
| 问题 | 证据 |
|:----|:-----|
| embedding 注入无法控制 LLM | 类间/类内比 0.93 和 0.99 |
| 赫布学习 + 扩散不可共存 | 3 节点全连通图发散，ρ(A)=1.099 |
| 注入信号强度差距 | 3 轮审查一致认定 |

**关闭不等于失败**——排除了"从外部控制 LLM 内部表示"这条路线。

---

## 实验进度

### 已完结
- **P0 系列**（7 轮）：词向量 90° 正交，词级不够 ❌
- **P0-A 系列**（6 轮）：50% 天花板，未超 MFS 基线 ❌

### 进行中
- **P0-R 系列**（评测修复）：
  - R0 58.3% ❌ oracle MFS 泄漏
  - R1 61.7% ✅ L4 词性过滤有效（L5 全死）
  - R2 63.3% ❌ 评分泄漏
  - R2.1 63.3% ❌ 评分泄漏
  - ⏳ R3 草案：120 句全新语料 + 严格盲测管道

详见 [[P0 Series Experiments]]

---

## What Remains Valuable from Haibo 1.0
- 扩散引擎（`intent_cloud_dynamics.py` 快慢分离设计）
- 36 节点常识图作为语义资源
- H17 模块群（语义理解层：指代消解、纠正分析、情绪球等）
- Haibo Memory MCP Server

## Core Principles
1. 不重新发明语义学——采用已有理论（Frame Semantics, HowNet, Ontology, Semantic Role）
2. Haibo 的创新在**计算机制**（如何组织、学习、演化图结构），不在语言学
3. **可验证假设 → 最小实现 → 对照实验 → 量化测量 → 证伪或支持**
4. **改一行代码 → 跑一次 → 看两个数字**
5. 诚实报告失败 > 正面结论

---

**关联笔记**：[[P0 Series Experiments]]、[[文献库索引]]、[[感知特征权重与语义消歧]]
