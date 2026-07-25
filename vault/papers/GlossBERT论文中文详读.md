---
tags: [论文, GlossBERT, WSD, 词义消歧]
---

# GlossBERT 论文中文详读

> **GlossBERT: BERT for Word Sense Disambiguation with Gloss Knowledge**  
> 黄璐瑶、孙迟、邱锡鹏、黄萱菁 | 复旦大学  
> EMNLP-IJCNLP 2019 | arXiv:1908.07245

---

## 一、这篇论文在做什么

GlossBERT 做的正是 Haibo 的"围合消歧"在深度学习时代的对标方案。

**核心思路极简**：对于句子中的一个歧义词，把它的候选义项的定义（gloss，来自 WordNet）跟句子原文拼在一起，用 BERT 判断哪个义项最匹配。

---

## 二、核心方法

### 2.1 Context-Gloss Pair

对每一个歧义词的每一个候选义项，构造一对输入：

```
[CLS] 句子上下文 [SEP] 义项定义 (gloss) [SEP]
```

比如：

```
"苹果比上一代便宜了五百块"
→ 候选义项1: "苹果=科技公司" → gloss: "a multinational technology company"
→ 候选义项2: "苹果=水果" → gloss: "a fruit with red or green skin"
```

BERT 对每个 pair 输出一个匹配分，取分数最高的义项。

### 2.2 三种模型变体

| 变体 | 做法 | 效果 |
|------|------|------|
| **GlossBERT (CLS-Context)** | 上述简单拼接，只用 [CLS] 做分类 | 效果好 |
| **GlossBERT (CLS-Gloss)** | 把句子和 gloss 分开过 BERT 再匹配 | 稍差 |
| **GlossBERT (Context-Gloss)** | 带注意力交互的拼接 | **最佳** |

### 2.3 训练

- 在 SemCor 3.0 上微调 BERT
- 负采样策略：随机选其他义项作为负样本
- 二分类任务（匹配/不匹配）

---

## 三、实验结果

- 在 SemEval 2007/2013/2015 等标准 WSD 评测集上 **超过当时所有 SOTA**
- 证明了：**gloss（义项定义）作为语义先验是有效的**
- 简单拼接就比复杂的词专家系统好

---

## 四、与 Haibo 的直接关系

这是目前看到跟 Haibo "围合消歧"最像的方法。

### 结构对比如下

```
GlossBERT:
  句子上下文 → BERT → 与每个 gloss 匹配 → 选最高分

Haibo Semantics Graph v0:
  句子输入词 → 查 word→concept 表 → 概念激活计数 → 选最多激活
```

### 区别

| 维度 | GlossBERT | Haibo v0 |
|------|-----------|----------|
| 语义先验来源 | WordNet gloss（自然语言定义） | 手写 word→concept 归属表 |
| 匹配方式 | BERT 深层语义匹配 | 简单计数（词是否属于该概念） |
| 是否需训练 | ✅ 需要 BERT 微调 | ❌ 不需要训练 |
| 可解释性 | 黑箱（为什么匹配不知道） | **完全透明**（什么词激活了什么概念） |
| 依赖 | BERT + WordNet | 无外部依赖 |
| 算力 | GPU | CPU 即可 |

### Haibo 能从 GlossBERT 学到什么

1. **词汇覆盖决定上限** — GlossBERT 的弱点是词汇覆盖不全时不知道。Haibo 的 word→concept 表也一样
2. **负采样策略** — GlossBERT 用随机负采样，Haibo 可以借鉴"概念间区分度"的负例设计
3. **混合信号有效** — GlossBERT 证明"上下文 + 语义先验"的组合优于纯上下文或纯先验。Haibo 的围合正是这种混合

---

## 五、论文基本信息

- **arXiv**: https://arxiv.org/abs/1908.07245
- **PDF**: https://arxiv.org/pdf/1908.07245
- **发表**: EMNLP-IJCNLP 2019
- **代码**: 见论文内引用

---

**所属方向**: [[计算语言学文献]]（WSD 方法）
**关联概念**: [[Haibo-Research/07_Implementation/semantics_graph_design_v0|Semantics Graph v0 设计]]
