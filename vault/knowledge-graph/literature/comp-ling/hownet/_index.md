---
tags: [literature, 计算语言学, HowNet]
---

# HowNet 义原系统

> 董振东的 HowNet 知识库：用细粒度义原作为语义最小单位

---

## 核心概念

- **Sememe（义原）**：不可再分的语义最小单位
- HowNet 用约 2000 个义原描述所有中文词的语义
- 每个词 = 一组义原的组合（义原表示式）

## 例子

"苹果（水果）" = {fruit|水果, edible|能食, sweet|甜, ...}
"苹果（公司）" = {company|公司, electronic|电子, product|产品, ...}

## 在 Haibo 中的应用

P0-B-1 尝试用 HowNet 义原作消歧，但词定位失败（11%）。问题不在 HowNet 本身，而是匹配算法。

详见 [[literature/comp-ling/hownet/survey|HowNet 义原综述]]。

## 关联

- [[literature/comp-ling/_index|计算语言学]]
- [[haibo/experiments/p0-a-series/_index|P0-A 系列实验]]
- [[haibo/semantics-graph/concept-nodes|概念节点（与义原类似的概念）]]

*最后更新：2026-07-22*
