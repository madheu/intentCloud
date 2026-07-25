---
tags: [haibo, 语义图, 激活]
---

# 激活计数机制

> 概念激活的推理逻辑

---

## 基本思路

1. 输入句子中的词触发其归属的概念节点
2. 每个概念节点累计激活计数
3. 激活数最高的概念 = 当前语境最匹配的概念
4. 该概念下的词 → 消歧后的语义

## 与 GlossBERT 的关系

GlossBERT 做 context-gloss 匹配。围合计数跟 gloss 匹配在数学上同构，说明"查表消歧"这条路没有走歪。

## 关联

- [[haibo/semantics-graph/word-concept-mapping|Word→Concept 映射]]
- [[literature/comp-ling/wsd/glossbert|GlossBERT 论文]]
- [[haibo/experiments/p-resolve/_index|P_Resolve 验证]]

*最后更新：2026-07-22*
