---
tags: [literature, 知识图谱, GCN, 论文]
---

# GCN（图卷积网络）论文笔记

> **核心关注**：归一化公式的对称重整化

---

## 问题

图上的消息传递：每个节点聚合邻居信息来更新自己的表示。

## 关键公式

GCN 的对称重整化：
```
H^{(l+1)} = σ( D^{-1/2} A D^{-1/2} H^{(l)} W^{(l)} )
```
其中 D 是度矩阵，A 是邻接矩阵。

## 与 Haibo 的关系

1.0 用的归一化是 `度^{-1}`（不对称），GCN 用的是对称重整化。
[[haibo/decisions/no-hebbian-update|当前不做图学习]]，但未来会回来参考 GCN。

## 行动项

✅ 推荐先做：改归一化公式（度^{-1} → 对称重整化）

## 来源

Kipf & Welling (2017). Semi-Supervised Classification with Graph Convolutional Networks.
- [[GCN论文中文详读]]（原 vault 笔记）

*最后更新：2026-07-22*
