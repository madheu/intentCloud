---
tags: [haibo, 决策, 理论审查, 语义图]
---

# 理论审查：Semantics Graph v0

> 基于 9 篇论文的方法论卡片，审查 Haibo 2.0 项目假设

---

## 发现1：代码存在阶段断层

IntCog-R 核心代码是 **Haibo 1.0 遗产**，Semantics Graph v0 **不存在任何实现代码**。

**严重性**：⚠️ 中等 — 正常现象（P0 系列先实验后实现），但需注意 P0-B 实验脚本仍是 1.0 思路延续。

## 发现2：P0-B 实验脚本继承自 P0-A 的错误范式

`p0_b1_hownet.py` 的核心逻辑（HowNet 义原 → 手工映射 4 面 → 点积消歧）与论文结论存在冲突。

## 关联

- [[haibo/decisions/activation-diffusion-refactor]]
- [[haibo/experiments/blind-spot-hsg1]]
- [[haibo/semantics-graph/_index]]

*最后更新：2026-07-23*
