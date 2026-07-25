---
tags: [文献索引, NLP, LLM, 表示空间]
---

# NLP-LLM 表示空间文献

> Transformer 内部表示、Token 活动模式、BriLLM、表示空间几何
> 与 [[索引]]、[[计算语言学文献]]、[[认知语言学文献]] 并列

---

## 一、BriLLM — 核心论文

| 标题 | 作者 | 年 | 核心贡献 |
|------|------|:--:|----------|
| *BriLLM: Brain-inspired Large Language Model* | Hai Zhao, Hongqiu Wu 等 | 2025 | Token→专用节点 + 动态信号传播，全模型可解释 |

**与海波的关系**：BriLLM 是 Haibo 的**上游邻居**——BriLLM 做 Token 节点信号传播，Haibo 做 Token→Semantic Node 跃迁。不重复。
详细分析见 [[BriLLM论文中文详读]]（含完整中文摘要）。

---

### 可下载全文的论文（arXiv）

| 论文 | arXiv ID | 在线地址 |
|------|:--------:|----------|
| BriLLM (Zhao 2025) | 2503.11299 | https://arxiv.org/abs/2503.11299 |
| Word2Vec (Mikolov 2013) | 1301.3781 | https://arxiv.org/abs/1301.3781 |

---

## 二、Token Graph 相关工作

| 标题 | 作者 | 年 | 核心贡献 |
|------|------|:--:|----------|
| *Towards Improved Sentence Representations using Token Graphs* | Mantri 等 | 2026 | 预训练 LM 上构建 Token 图增强句子表示 |
| *Lightweight Token Graphs from Transformers* | Pimentel 等 | 2026 | 从注意力输出直接构造轻量 Token 图 |
| *Tokenized Graph Transformer (TokenGT)* | Kim 等 | 2022 | 图节点和边当 token 输入标准 Transformer |
| *NAGphormer* | Chen 等 | 2023 | Hop2Token—多跳邻域→token 序列 |

**与海波的关系**：这些方法把"图结构"还给 Transformer。海波的 Semantics Graph 可以作为**先验结构**注入这些框架。

---

## 三、Transformer 内部表示空间分析

| 标题 | 作者 | 年 | 核心贡献 |
|------|------|:--:|----------|
| *Geometry of Reason* | Noël | 2026 | 注意力矩阵视为加权 Token 图，谱特征区分推理 |
| *Self-Attention as Token Graph Diffusion* | Roffo 等 | 2026 | 自注意力=Token 图上的扩散过程 |
| *Mixture of Hidden-Dimensions* | Chen 等 | 2025 | 仅部分 hidden 维度建模跨 Token 交互 |
| *Correcting Influence with Orthogonal Spaces* | Yu 等 | 2026 | 正交潜在空间分解 LLM 输出 |

**与海波的关系**："注意力矩阵 = Token 图"是海波"围合"直觉的直接数学证据。自注意力本身就是一种围合。

---

## 四、Token 活动模式

| 标题 | 作者 | 年 | 核心贡献 |
|------|------|:--:|----------|
| *A Single Neuron Bypasses Safety* | Kazemi 等 | 2026 | Per-Token 激活模式揭示单神经元即可绕过安全 |
| *Cross-Layer Clustering for Parameter Decomposition* | Seshadri 等 | 2025 | 谱聚类发现跨层一致的 Token 激活模式 |

**与海波的关系**：H-SG-1 的实证基础——Token 活动模式确实存在可聚类的稳定性。

---

## 五、Embedding 空间几何

| 标题 | 作者 | 年 | 核心贡献 |
|------|------|:--:|----------|
| *Riemannian Geometry for Graph Intelligence* | Yu & Sun | 2026 | 欧氏 embedding 空间局限，黎曼几何建模 |
| *Semantic-Space Exploration in RLVR* | Huang 等 | 2026 | 直接对 Transformer 语义空间做 RL |

**与海波的关系**：海波的 P0.5-7 发现"词级静态向量空间不够"，跟这里"欧氏空间局限"的判断一致。

---

### 跨学科链接
- BriLLM 的 Token 信号传播 ↔ [[知识表示与知识图谱文献#GNN 消息传递]]
- 注意力矩阵=Token 图 ↔ [[计算语言学文献#Frame Semantics]]（框架作为图结构）
- Token 活动模式聚类 ↔ [[认知语言学文献#Rosch 原型理论]]（自然范畴形成）
