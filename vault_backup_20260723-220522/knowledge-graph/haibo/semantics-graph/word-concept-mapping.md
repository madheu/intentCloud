---
tags: [haibo, 语义图, 映射]
---

# Word→Concept 映射

> 词到概念的归属关系表

---

## 结构

一张归属表记录每个词属于哪些概念：

```
"苹果" → { "水果类苹果", "科技类苹果" }
"甜"   → { "水果类苹果", "糖果", "甜品" }
"音质" → { "科技类苹果", "音响设备" }
```

## 作用

- 输入词 → 查表 → 候选概念集
- 结合上下文 → 激活最匹配的概念
- 实现消歧

## 构建方式

- 初始版本：手动构建（已验证可行）
- 未来：从大规模语料无监督学习

## 关联

- [[haibo/semantics-graph/concept-nodes|概念节点]]
- [[haibo/semantics-graph/activation-counting|激活计数]]
- [[haibo/experiments/p-resolve/_index|P_Resolve 验证]]

*最后更新：2026-07-22*
