---
tags: [haibo, 实验, 图设计]
---

# P_Resolve：图设计探索

> **核心教训**：节点类型定义是语义图设计的第一决策

---

## 版本对比

| 版本 | 节点类型 | 边 | 结果 |
|:----:|---------|------|:----:|
| P0.8 | 词 Token | 共现频率 | 全部"不确定"❌ |
| P0.9 | 概念节点 | 层级归属 | 10/11 正确 ✅ |

## 关键发现

P0.9 使用概念节点（不是词节点）在 11 句测试用例中达到 10/11 正确率。这验证了 [[haibo/decisions/token-vs-concept-node|概念节点决策]]。

## 关联

- [[haibo/semantics-graph/_index|语义图设计]]
- [[haibo/experiments/p0-series/_index|P0 系列总览]]
- [[haibo/experiments/p0-a-series/_index|P0-A 消歧实验]]

*最后更新：2026-07-22*
