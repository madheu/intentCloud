---
tags: [论文, MPNN, 图神经网络, 消息传递]
---

# Neural Message Passing for Quantum Chemistry (MPNN) 中文详读

> **原文信息**：Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., & Dahl, G. E. (2017). Neural Message Passing for Quantum Chemistry. *ICML 2017*. arXiv:1704.01212. Google Brain / Google DeepMind. **引用量**：8000+（截至2025年）

---

## 一、开门：用一个比喻理解论文

**如果把 GNN 比作一个「化学家会议」：**

- 每个原子（节点）是一位化学家，坐在会议室里（分子图）
- 化学键（边）是他们之间的电话线，每种键类型有不同的通话费率（边特征）
- **Message Passing 阶段**：T 轮电话会议，每轮每位化学家：
  1. 从邻居收到消息（**Message**）
  2. 结合自己的当前状态更新笔记（**Update**）
  3. 下一轮继续
- **Readout 阶段**：T 轮结束后，主持人收集所有人的最终笔记，画一张「这张分子的整体画像」——预测一个化学性质数值

**为什么这篇论文重要**：它不是说「我们发明了一种新 GNN」，而是说「你们所有人用的各种 GNN 其实都长一样——都是 Message → Update → Readout 三段式」。它把 GCN、GraphSAGE、GG-NN、Interaction Networks、Deep Tensor Neural Networks 等全部装进一个框架，让研究者可以系统性地比较和优化。

---

## 二、核心要解决的问题

| 问题 | 描述 |
|------|------|
| 主问题 | 用量子化学计算预测分子性质（如 HOMO/LUMO 能级、偶极矩等）太慢（DFT 单分子 ~1小时），能否用 GNN 学出一个快 30 万倍的近似器？ |
| 框架问题 | 已有的多种图神经网络（GCN/GG-NN/Interaction Nets 等）看似不同，但能否统一到一个框架下系统研究？ |
| 工程问题 | 如何让 GNN 在大图/高维隐层上跑得更快？ |

**本质**：这是一篇 **统一框架 + 工程实证** 论文，不是一篇理论创新论文。它的价值在于抽象出 Message → Update → Readout 三段式，让后续的 GNN 研究有了一个公共语言。

---

## 三、核心方法

### 3.1 MPNN 统一框架

MPNN 的前向传播分为两个阶段：**消息传递阶段** 和 **读出阶段**。

**消息传递阶段**（运行 T 步）：

$$
m_v^{t+1} = \sum_{w \in N(v)} M_t(h_v^t, h_w^t, e_{vw}) \tag{1}
$$

$$
h_v^{t+1} = U_t(h_v^t, m_v^{t+1}) \tag{2}
$$

其中：
- $h_v^t$：节点 v 在第 t 步的隐藏状态
- $e_{vw}$：边 (v, w) 的特征向量
- $M_t$：消息函数（可微、可学习）
- $U_t$：更新函数（可微、可学习）
- $N(v)$：节点 v 的邻居集合

**读出阶段**：

$$
\hat{y} = R(\{h_v^T | v \in G\}) \tag{3}
$$

其中 $R$ 是读出函数，必须对节点排列不敏感（permutation invariant），以确保 MPNN 对图同构不敏感。

### 3.2 论文考察的已有模型作为 MPNN 特例

| 模型 | 消息函数 $M_t$ | 更新函数 $U_t$ | 读出 $R$ |
|------|---------------|---------------|---------|
| Duvenaud et al. (2015) | 拼接$(h_w, e_{vw})$ → 求和 | $\sigma(H^{deg(v)}_t m_v^{t+1})$ | 每步 softmax 求和 |
| GG-NN (Li et al. 2016) | $A_{e_{vw}} h_w$（每种边类型一个矩阵） | GRU$(h_v^t, m_v^{t+1})$ | $i(h_v^T, h_v^0) \odot j(h_v^T)$ |
| Interaction Networks (Battaglia et al. 2016) | NN$(h_v, h_w, e_{vw})$ | NN$(h_v, x_v, m_v)$ | $f(\sum h_v^T)$，$T=1$ |
| Kearnes et al. (2016) | 边状态 $e_{vw}^t$ 作为消息 | 节点 + 边双向更新 | — |
| Deep Tensor NN (Schütt et al. 2017) | $ \tanh(W^{fc}((W^{cf}h_w+b_1) \odot (W^{df}e_{vw}+b_2)))$ | $h_v^t + m_v^{t+1}$ | $\sum \text{NN}(h_v^T)$ |
| GCN (Kipf & Welling 2017) | $c_{vw} h_w$，$c_{vw} = (deg(v)deg(w))^{-1/2}A_{vw}$ | ReLU$(W^t m_v^{t+1})$ | — |

### 3.3 论文提出的 MPNN 变体

论文在 GG-NN 基础上探索了多种变体：

**消息函数变体**：
1. **Matrix Multiplication**（GG-NN 基线）：$M(h_v, h_w, e_{vw}) = A_{e_{vw}} h_w$（每种离散边类型一个矩阵）
2. **Edge Network（enn）**：$M(h_v, h_w, e_{vw}) = A(e_{vw}) h_w$，其中 $A(e_{vw})$ 是一个将边向量映射到 $d \times d$ 矩阵的神经网络。**支持连续边特征**，对海波最相关。
3. **Pair Message**：$M(h_v, h_w, e_{vw}) = f(h_w, h_v, e_{vw})$ — 消息同时依赖源节点和目标节点。但实验效果差（平均错误率 3.98 vs enn 的 1.53），作者怀疑训练困难。

**虚拟图元素**：
1. **Virtual Edge**：为不相连的节点对添加虚拟边类型，让信息在传播阶段可以走长距离
2. **Master Node**：一个与所有节点相连的"主节点"，作为全局读写空间，每个消息传递步所有节点都向其读写

**读出函数变体**：
1. **GG-NN 读出**：$R = \sum_v \sigma(i(h_v^T, h_v^0)) \odot j(h_v^T)$
2. **Set2Set**（Vinyals et al. 2015）：专门为集合设计，表达能力比简单求和更强

**Multiple Towers**（计算加速）：
将 $d$ 维嵌入拆成 $k$ 组 $d/k$ 维，分别做消息传递后由神经网络混合。复杂度从 $O(n^2 d^2)$ 降到 $O(n^2 d^2/k)$。$k=8$ 时看到 2 倍加速。

### 3.4 输入表示

| 特征类型 | 内容 |
|---------|------|
| 原子特征 | 原子类型（one-hot）、原子序数、受体/供体、芳香性、杂化类型、氢原子数 |
| 边特征 | 化学图模式：单/双/三键/芳香键；距离分箱模式：10 个距离区间；原始距离模式：5维向量（欧氏距离 + 键类型 one-hot）|

**关键实验发现**：
- 显式包含氢原子作为节点 → 大幅提升性能（但训练慢 10 倍）
- 包含空间距离信息 → 大幅提升（平均错误率 0.68 vs 无空间信息的 2.57）
- 包含完整边特征向量（键类型 + 空间距离）→ 对很多任务至关重要

### 3.5 训练细节

- 数据集：QM9（134k 分子，13 个回归任务）
- 超参数搜索：每个目标 50 次随机搜索
- T 的范围：$3 \leq T \leq 8$（任意 $T \geq 3$ 都有效）
- 优化器：ADAM，batch size 20，3M steps（~540 epoch）
- 学习率：$1e-5 \sim 5e-4$，线性衰减
- 权重共享：所有时间步共享参数（weight tying）优于不共享

---

## 四、关键结果

### 主要结果

| 目标 | 最佳基线 | enn-s2s（单模型） | enn-s2s-ens5（集成） | 化学精度阈值 |
|:----:|:--------:|:----------------:|:-------------------:|:----------:|
| μ (偶极矩) | 0.70 | **0.30** | **0.20** | 1.0 |
| α (极化率) | 1.75 | **0.92** | **0.68** | 1.0 |
| HOMO | 1.17 | **0.99** | **0.74** | 1.0 |
| LUMO | 1.08 | **0.87** | **0.65** | 1.0 |
| Δε (能隙) | 1.70 | **1.60** | **1.23** | 1.0 |
| ⟨R²⟩ (电子空间范围) | 1.35 | **0.15** | **0.14** | 1.0 |
| ZPVE (零点能) | 1.91 | **1.27** | **1.10** | 1.0 |
| U0 | 0.58 | **0.45** | **0.33** | 1.0 |
| 其他... | — | — | — | — |
| **平均** | 1.35 | **0.68** | **0.52** | — |

**数据形式**：表中数值为 **MAE ratio = 模型 MAE / 化学精度阈值**，比值 < 1 表示达到化学精度。

**关键陈述**：最佳 MPNN 在 11/13 个目标上达到化学精度，所有 13 个目标达到 SOTA。无空间信息条件下，5/13 个目标达到化学精度。

---

## 五、与项目的深层联系（海波语义图）

### 5.1 结构同构分析：Message → Update 循环 ≈ 激活扩散循环

这是四篇论文中 **最直接的结构映射**。MPNN 的 Message → Update 循环和 Haibo 的激活扩散循环几乎是一一对应的：

| MPNN 组件 | Haibo 对应物 | 映射强度 |
|-----------|-------------|:--------:|
| $h_v^t$（节点隐藏状态） | 概念节点的**当前激活值** | ✅ 完全同构 |
| $e_{vw}$（边特征） | 海波的**有向边类型**（围合关系/上下位/同义等） | ✅ 完全同构 |
| $M_t(h_v^t, h_w^t, e_{vw})$（消息函数） | **从邻居 w 到 v 的信号传递**，受边类型 e 调制 | ✅ 完全同构 |
| $U_t(h_v^t, m_v^{t+1})$（更新函数） | **激活更新**：旧激活 + 传入信号 → 新激活 | ✅ 完全同构 |
| $m_v^{t+1} = \sum_{w \in N(v)} M_t(\dots)$（聚合） | **求和/聚合所有邻居的传入信号** | ✅ 完全同构 |
| $T$ 步固定迭代 | 海波的"扩散 N 轮" | ✅ 结构同构（但海波目前用"直到稳定"）|
| Readout $R$（全图输出） | 海波**不需要**全图读出 | ❌ 不相关 |
| 可学参数（$M_t, U_t, R$ 中的权重）| 海波的**非参数化赫布学习** | ❌ 核心冲突 |

### 5.2 核心差异

| 维度 | MPNN | Haibo |
|------|------|-------|
| **训练方式** | 监督学习（MSE loss + 反向传播） | 无监督赫布学习（无反向传播、无标签） |
| **参数更新** | $M_t, U_t$ 含可学权重，通过 SGD 优化 | 纯关联强度更新（赫布规则），无可学权重 |
| **图结构** | 静态分子图（固定结构） | 动态语义图（概念可增减、边可变化） |
| **消息内容** | 实值向量（基于可学习权重变换邻居状态） | 实值标量（加权邻居激活值） |
| **边特征处理** | 通过可学神经网络将边特征映射为 $d\times d$ 矩阵 | 边类型编码直接作为 **调制权重**（非可学，预定义） |
| **更新函数** | GRU（复杂非线性 RNN） | 简单的激活更新（旧激活 × decay + 新信号 × 调制因子） |
| **目标** | 预测一个全图输出（分子性质回归） | 节点激活模式的**无监督涌现**（词义消歧是下游任务） |
| **消息范围** | 所有邻居（对 9 原子分子可稠密连接） | 局部邻域 / 语义关联概念 |

### 5.3 可以借用的东西（带优先级）

**P1（最优先先做）—— Message 和 Update 的职责分离**

MPNN 最大的贡献是把「从邻居收消息」和「更新自己」拆成两个独立函数。Haibo 目前的激活扩散代码里这两步常常混在一起（比如在一个循环里同时计算邻居贡献和更新自身）。**建议重构**：

```
# 当前（混在一起）：
for each node v:
    signal = sum(neighbors * weight * edge_type_factor)
    activation[v] = activation[v] * decay + signal

# 改造（按 MPNN 分离）：
# Step 1: Message — 仅计算邻居信号
for each edge (w->v):
    msg[w->v] = M(activation[w], edge_type[e])   # 消息与目标节点无关

# Step 2: Update — 仅更新节点
for each node v:
    agg_msg = sum(msg[w->v] for w in N(v))
    activation[v] = U(activation[v], agg_msg)      # 更新只依赖当前激活 + 聚合消息
```

这个分离让调试更容易（可以单独检查谁发了什么消息）、也方便未来引入不同的消息函数。

**P1（最优先先做）—— 边特征融入消息函数**

MPNN 的 Edge Network 消息函数 $M(h_v, h_w, e_{vw}) = A(e_{vw}) h_w$ 对海波有直接启发：**边类型不应该只在聚合后加权，而应该在消息生成时就参与运算**。

```
# 当前做法（边类型只在聚合后加权）：
msg = neighbor_activation[w]
agg_msg = sum(msg * edge_type_weight[e])  # 边类型作为权重

# 改造（边类型在消息生成时参与）：
msg = neighbor_activation[w] * edge_embedding[e]  # 或更复杂
agg_msg = sum(msg)
```

关键点：**无需学习 $A(e_{vw})$**，可以直接用预定义的边类型编码向量（如海波的围合边 = [1, 0, 0]、上下位边 = [0, 1, 0] 等）与邻居激活值做元素级乘法。

**P2（先跑个实验看看）—— 固定 T 步替代「扩散直到稳定」**

MPNN 的 $3 \leq T \leq 8$ 固定步数策略值得实验。当前 Haibo 的做法是"扩散直到激活收敛"，在 2-3 跳内到达稳定。实验可问：**固定 T=3 是否比"直到稳定"更快且结果近似？**

```
# 实验设计：
# A 组：until_stable(activation) — 当前做法
# B 组：diffuse(activation, T=3) — 固定 3 步
# C 组：diffuse(activation, T=5) — 固定 5 步
# 比较：稳定性（是否震荡）、收敛速度（步数）、消歧结果（最终分类准确率）
```

**P3（以后再说）—— Multiple Towers 用于大图**

如果海波的语义图扩大到数千个概念节点，Multiple Towers 策略（将嵌入拆分成 $k$ 份独立传播后再混合）可以加速。

**P3（以后再说）—— Master Node 作为全局上下文**

在有长距离语义依赖的场景（如一篇长文的整体主题对局部消歧的影响），可以尝试给海波图加一个"主题节点"或"上下文节点"，与所有概念节点相连，类似 MPNN 的 Master Node。

### 5.4 绝对不能借用的东西（铁律）

| 铁律 | 原因 |
|------|------|
| ❌ **不可学习的消息/更新函数** | MPNN 的 $M_t, U_t$ 都是可学函数（含神经网络权重）。海波的核心原则是 **无反向传播、无监督、无训练**。任何引入可学参数的做法都会打破方法论一致。 |
| ❌ **不能为了性能提升而引入反向传播** | 即使 MPNN 的 SOTA 结果令人印象深刻（11/13 任务达化学精度），海波不能为性能牺牲无监督原则。引以为戒的是 MPNN 论文说"we find that training one model per target consistently outperformed jointly training on all 13 targets"——这是一种典型的监督学习思维，与海波格格不入。 |
| ❌ **GRU 更新函数** | 虽然 GG-NN 用 GRU 作为更新函数 $U_t$ 效果好，但 GRU 有可学门控参数。海波的更新应该是非参数化的：$h_v^{t+1} = \text{decay} \cdot h_v^t + (1 - \text{decay}) \cdot \text{activation\_func}(m_v^{t+1})$。 |
| ❌ **全图读出 / 图级别监督信号** | MPNN 的 Readout 函数是为分子性质回归设计的。海波不需要全图输出，它的目标是节点级别的激活模式涌现。不要把全图损失引入海波系统。 |

### 5.5 演进路线图

```
GCN (Kipf 2017):     归一化邻接矩阵   × 直推式       × 共享卷积核     × 无边特征
                      ↓ 采样 + 归纳    ↓
GraphSAGE (2017):    固定 K 邻居采样   ✓ 归纳式 ✓    多个聚合器选择   × 无边特征
                      ↓ 注意力权重     ↓
GAT (Velickovic 2018): 多头自注意力    ✓ 归纳式 ✓    可学注意力参数   × 无边特征
                      ↓ 框架统一       ↓ 引入边特征    ↓ 分开 M 和 U    ↓
MPNN (Gilmer 2017):  Message→Update   ✓ 框架级统一    ✓ 可学/不可学    ✓ 边特征明确
                      ↓ 赫布约束       ↓ 动态图         ↓ 无监督         ↓
海波（本系统）:       激活扩散循环     ✓ 动态语义图    ✓ 非参数化      ✓ 边类型编码
```

**MPNN 在前三篇中的位置**：
- 它不引入 GCN 的谱方法、不是 GraphSAGE 的采样技巧、不是 GAT 的注意力机制
- 它做的是 **抽象和统一**——声明所有 GNN 都是 Message→Update→Readout 三段式
- 对海波而言，MPNN 的价值不是"学什么参数"，而是 **提供了一个比前三篇更干净的框架语言来描述激活扩散**

---

## 六、一个具体的实验想法

### 实验：MPNN 式消息-更新分离对海波的影响

**基线**：海波当前的激活扩散代码（消息和更新在同一循环中）

**方法**：
1. 将 Message 和 Update 分离为两个独立函数
2. 在消息函数中显式引入边类型嵌入（而非仅在聚合后加权）
3. 用固定 T 步（T=3,5,7）替代"直到稳定"
4. 在相同的消歧任务上比较

**比较指标**：
- 消歧准确率（最终哪个 sense 被激活）
- 收敛所需步数
- 每一步的激活稳定性

**失败条件**：分离后的消息-更新框架比原始方法在准确率上下降超过 5 个百分点。

---

## 七、批判性评价

### 优点

1. **框架统一价值巨大**：Message→Update→Readout 三段式成为后续所有 GNN 论文的共同语言，这是该论文引用量高的根本原因。
2. **系统性的变体探索**：不是只提出一种新架构，而是在统一框架下系统探索了多个消息函数、更新函数、读出函数的组合。
3. **实战导向**：在 QM9 上做到 SOTA 并在 11/13 任务上达到化学精度，展示了 GNN 在化学领域的实用价值。
4. **工程技巧实用**：Multiple Towers、Master Node、Virtual Edge 等技巧都有实际加速或性能提升。
5. **诚实的实验报告**：报告了 Pair Message 效果差、Towers + Edge Network 组合不成功等负面结果，这在 2017 年不常见。

### 弱点

1. **理论深度有限**：论文主要是框架统一 + 工程优化，没有理论分析（比如 MPNN 的表达能力界限，2 年后 Xu et al. 2019 才补充）。
2. **仅适用于小图**：QM9 分子最多 29 个节点（含氢），论文明确说"additional improvements will be needed to scale to much larger graphs"。这限制了方法的通用性。
3. **没有讨论归纳/直推式问题**：与 GraphSAGE 明确讨论归纳式学习不同，MPNN 没有区分直推式和归纳式设置。论文假设所有图在推理时都是完整可用的。
4. **没有跨图大小泛化**：论文承认"generalizing to larger molecules seems particularly challenging when using spatial information"——这是一个已知但未解决的问题。
5. **框架的局限**：MPNN 框架能表达大多数 GNN，但并非所有——一些基于随机游走的模型（如 Node2Vec）和 Transformer 式图模型不完全落入这个框架。

---

## 八、论文基本信息

- **标题**：Neural Message Passing for Quantum Chemistry
- **作者**：Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, George E. Dahl
- **机构**：Google Brain, Google, Google DeepMind
- **会议**：ICML 2017
- **arXiv**：1704.01212
- **PDF 链接**：https://arxiv.org/pdf/1704.01212.pdf
- **所属方向**：图神经网络 / 计算化学
- **关联笔记**：[[GCN论文中文详读]], [[GraphSAGE论文中文详读]], [[GAT论文中文详读]]

---

## 九、相关文献路线

MPNN 在 GNN 演进中处于特殊位置：

- **前驱**：
  - GG-NN (Li et al. 2016) → MPNN 的直接基线（消息函数+GRU 更新）
  - Interaction Networks (Battaglia et al. 2016) → 「消息依赖源和目标节点」这个设计来源于此
  - Duvenaud et al. (2015) → 最早的可微分子指纹

- **后继**：
  - GIN (Xu et al. 2019) → 在 MPNN 框架下证明求和聚合器是表达能力上限
  - Message Passing 的思想被扩展到 3D 图（SchNet, DimeNet, GemNet）
  - ForceNet / EGNN → 将等变约束引入 MPNN

---

## 十、一句话总结

> **MPNN 统一了所有 GNN 为 Message → Update → Readout 三段式，对海波而言最大的价值是提供了一个比 GCN/GraphSAGE/GAT 更干净的框架语言来描述激活扩散循环，但它的监督训练方式与海波的无监督方法论依然冲突。**

---

---

## 附录A：方法论决策卡片

### A1 问题映射

| MPNN 的问题 | → | 海波对应问题 |
|------------|:-:|-------------|
| 从分子图预测量子化学性质 | → | 从语义图做概念消歧（不同领域的下游任务） |
| 设计一个统一的消息传递框架 | → | 需要一个统一的激活扩散框架 |
| 在消息中融入边特征 | → | 在激活扩散中融入边类型编码 |
| 固定 T 步消息传递 | → | 用固定步数替代"直到稳定" |
| 学习图结构上的不变表示 | → | 从图结构产生稳定的激活模式 |

**判定**：问题结构在 Message→Update 循环层面 **高度同构** ✅；在训练目标和输出形式上 **不相关** ❌

### A2 算法对齐

| 步骤 | MPNN 做法 | 海波当前做法 |
|------|----------|-------------|
| 消息生成 | $m_{w\to v} = M_t(h_w, h_v, e_{vw})$ | `msg = activation[w] * edge_weight[e_type]` |
| 消息聚合 | $m_v = \sum_{w\in N(v)} m_{w\to v}$ | `agg = sum(msg for all neighbors)` |
| 节点更新 | $h_v^{t+1} = U_t(h_v^t, m_v^{t+1})$ | `activation[v] = decay * activation[v] + (1-decay) * act_func(agg)` |
| 边特征使用 | $e_{vw}$ 作为 $M_t$ 的输入参数 | $e_{type}$ 作为消息权重的索引 |
| 迭代轮数 | 固定 $T$ 步（3-8） | "直到稳定"或手动设定轮数 |

**判定**：底层操作 **同构** ✅，但海波的消息函数中没有显式使用目标节点 $h_v$，且边特征以更简单的方式使用。

### A3 差异分析

| 维度 | MPNN | 海波 |
|------|------|------|
| 训练目标 | 监督学习（MSE loss） | 无监督赫布学习 |
| 参数可学性 | $M_t, U_t, R$ 含可学参数 | 所有运算非参数化 |
| 图结构 | 静态分子图 | 动态语义图 |
| 边特征处理 | 神经网络映射到矩阵 | 预定义编码向量 |
| 更新函数复杂性 | GRU（带门控 RNN） | 简单加权和 |
| 目标输出 | 全图标量（回归） | 节点激活向量（涌现） |
| 是否依赖反向传播 | 是 | 否（赫布规则） |
| 消息是否依赖目标节点 | 部分模型依赖 | 目前不依赖 |

**判定**：差异集中在 **训练和参数化层面**，前向传播的 Message→Update 结构 **高度同构** ✅。这意味着可以借用框架结构，但不能借用训练机制。

### A4 改造方案

#### 方案 A：Message-Update 职责分离（风险低，2h）

```
# 当前（混在一起）：
def diffuse_one_step(graph, activations):
    new_activations = {}
    for v in graph.nodes:
        signal = sum(activations[w] * graph.edge_weight[(w,v)] 
                     for w in graph.neighbors(v))
        new_activations[v] = (1 - decay) * activations[v] + decay * np.tanh(signal)
    return new_activations

# 改造后（MPNN 风格分离）：
def message(graph, activations):
    messages = {}
    for (w, v, e_type) in graph.edges:
        # 消息可以独立于目标节点 h_v
        messages[(w,v)] = activations[w] * edge_embedding(e_type)
    return messages

def update(activations, aggregated_messages):
    new_activations = {}
    for v in graph.nodes:
        agg = sum(messages[(w,v)] for w in graph.neighbors(v))
        new_activations[v] = (1 - decay) * activations[v] + decay * np.tanh(agg)
    return new_activations

# 主循环：
for t in range(T):
    msgs = message(graph, activations)
    activations = update(activations, msgs)
```

**优点**：更清晰，易于调试（可以 inspect 每一条消息的内容），易于扩展（未来可换消息函数）
**风险**：对现有代码改动小，几乎无风险

#### 方案 B：边类型嵌入在消息生成时参与（风险低，1h）

```
# 当前：边类型在聚合后作为权重
msg = activations[w]
agg = sum(msg * edge_type_weight[e])    # 边类型作权重

# 改造：边类型编码在消息生成时作为向量参与
edge_vec = edge_type_embedding[e]       # 预定义编码 [1,0,0] 或 [0,1,0] 等
msg = activations[w] * edge_vec         # 元素级乘法
agg = sum(msg)                          # 聚合
```

**优点**：更忠实于 MPNN 的思想，边特征在消息层面参与而非聚合层面
**风险**：需要测试编码向量的维度与激活值维度是否匹配

#### 方案 C：固定 T 步实验（风险低，3h）

```
# 实验三个版本：
for T in [3, 5, 7]:
    for _ in range(T):
        activations = message_update_step(activations, graph)
    # 比较消歧结果
```

**优点**：明确好调参，不会出现"一直震荡不收敛"
**风险**：固定 T 步可能 T 不够大导致扩散不充分

### A5 决策建议

| 编号 | 改造项 | 类型 | 优先级 | 工期 | 决策 |
|:----:|--------|:----:|:------:|:----:|:----:|
| R1 | Message-Update 职责分离 | 直接采用 | P1 | 2h | ✅ **推荐** — 代码整洁、易调试、为未来扩展铺路 |
| R2 | 边类型编码在消息生成时参与 | 改后用 | P1 | 1h | ✅ **推荐** — 简单修改，更忠实于 MPNN 框架设计 |
| R3 | 固定 T 步实验 | 先跑实验 | P2 | 3h | 🔶 **待 pilot** — 需要对比"T=3 vs until_stable"的消歧准确率 |
| R4 | 引入可学消息/更新函数 | ❌ | — | — | ❌ **不采用** — 与方法论核心冲突（无反向传播） |
| R5 | GRU 更新函数 | ❌ | — | — | ❌ **不采用** — 引入可学参数 |
| R6 | Readout / 全图输出 | ❌ | — | — | ❌ **不采用** — 海波不需要全图输出 |
| R7 | Master Node / Virtual Edge | 原理借鉴 | P3 | 1d | ⏸ **暂缓** — 目前图规模小（<100 节点），不需要长距离传播加速 |
| R8 | Multiple Towers 加速 | 原理借鉴 | P3 | 1d | ⏸ **暂缓** — 同理，图规模小不需要 |

### A6 一句话决策

> **MPNN 的 Message→Update 框架可以直接作为海波激活扩散的重构蓝图（R1, R2），边特征在消息层面参与的设计比当前"聚合后加权"更干净；但所有可学参数、反向传播、全图读出的部分一律摒弃。**

### A7 关联实验（决策 → 执行闭环）

| 决策编号 | 对应实验文件/代码位置 | 预计工期 | 状态 |
|:--------:|---------------------|:--------:|:----:|
| R1 | `haibo/activation_diffusion.py` — 拆分为 `diffuse_message()` 和 `diffuse_update()` | 2h | 🟡 待排期 |
| R2 | `haibo/graph_edge.py:edge_type_embedding` — 边类型编码参与消息生成 | 1h | 🟡 待排期 |
| R3 | `experiments/fixed_T_vs_convergence.py` — T=3/5/7 vs until_stable | 3h | ⚪ 待 pilot |
| R4-R6 | 记录到 [[Haibo Design Decisions]] 铁律清单 | — | 🔴 拒绝 |
| R7-R8 | 记录到 [[Haibo Design Decisions]] 低优先级清单 | — | ⏸ 暂缓 |

---

*决策卡片生成日期：2026-07-22*
*关联决策日志：[[Haibo Design Decisions]]*
*关联实验追踪：[[Experiments Log]]*
*所属系列：GNN 论文精读系列 — MPNN（第四篇）*
*前一篇：[[GAT论文中文详读]]*
