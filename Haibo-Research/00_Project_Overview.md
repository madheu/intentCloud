# 海波 2.0 — 项目概览

> 更新：2026-07-18 | 阶段：Phase 0（方向修正期）
> 上一版本（Haibo 1.0）核心路线已被实验证伪。

---

## 项目目标（修正版）

研究 Semantic Emergence（语义涌现）—— **概念是否能够从 Token 关系中自然形成**。

不是"控制 LLM"，也不是"认知架构"。核心问题是：

> Meaning 是如何形成的？

---

## 当前状态：Haibo 1.0 死亡确认

| 核心路线 | 状态 | 证据 |
|---------|:----:|------|
| IntentCloud → Embedding Injection → LLM | **证伪** | 两轮 pilot：I(a*; y_LLM)≈0 |
| 赫布学习 + 扩散的双层动力学 | **证伪** | Control Scientist 反例（ρ(A)=1.099） |
| "图控制 LLM 生成方向" | **不成立** | 注入信号在统计上不可检测 |

**非致命损失**：
- 图扩散机制本身有意义（H12 在 GPT-2 上验证过，Decision 产出合理）
- GraphInterpreter → Decision 的数据结构有用（道岔，没有实验证明它有用）
- 36 节点常识图作为语义资源有复用价值

---

## Haibo 2.0 新方向

BriLLM（上海交大）的 Token Graph 路线与 Haibo 关注的问题不同：

| | BriLLM | Haibo 2.0 |
|---|---|---|
| 核心 | Token Graph | Semantic Graph |
| 问题 | Token 如何传播 | Semantic 如何形成 |
| 方法 | 图上的信号流 | 多维方向围合形成局部稳定结构 |
| 输出 | Next Token | Context → Intent → Language |

新架构假设：

```
Input Text
    ↓
Token Graph（需要构造）
    ↓
Semantic Emergence（当前黑箱）
    ↓
Semantic Graph（概念关系拓扑）
    ↓
Context（多个 Semantic 的稳定状态）
    ↓
Intent（Context 长期稳定后的高层状态）
    ↓
Language Generation
```

---

## Phase 0（当前）

**目标**：验证语义信号在数据中天然存在（不依赖任何图架构）。

**实验**：词袋 + PCA 检验"苹果科技类"和"苹果水果类"句子是否自然可分。

**代码**：`06_Experiment/p0_bow_apple.py`

**通过标准**：类间/类内余弦距离比 > 1.2

---

## 已归档（不继续推进）

| 模块 | 状态 | 原因 |
|------|:----:|------|
| BilingualInjector（embedding 注入） | **废弃** | 实验证伪 |
| SteeringInjector（中间层 hook） | **归档** | 未测试，不属于 2.0 方向 |
| H20（编译层/元认知/节点生长） | **已取消** | 基于 Haibo 1.0 假设 |
| "LLM 只是嘴巴"哲学主张 | **待修订** | 与工程实际存在系统性错位 |

## 当前允许/禁止

| | 状态 |
|--|------|
| 跑 Phase 0 实验 | ✅ 进行中 |
| 读 BriLLM 论文做对比分析 | ✅ 建议 |
| 设计 Token Graph | 🟡 Phase 0 通过后 |
| 定义 Semantic Emergence 机制 | 🟡 Phase 0 通过后 |
| 继续 Embedding Injection 调优 | ❌ 路线已死 |
| 声称"认知架构" | ❌ 与当前工程规模不匹配 |
