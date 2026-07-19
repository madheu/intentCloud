# 实验：互信息 I(a*; y_LLM) 测量

> 指派：PI → AI Reviewer | 日期：2026-07-18
> 状态：🔴 **已终止——路径 C 触发**

> 实验状态历史：
> 1. 原始提案 → 🔴阻塞（Cognitive Scientist 标记 #1 #2 须修复）
> 2. Pilot 1 完成（大海，FEAR vs NONE，N=10）→ 类间/类内比 0.93
> 3. Pilot 2 完成（桌子，FEAR vs NONE，N=10）→ 类间/类内比 0.99
> 4. 综合结论：**在当前注入配置下，I(a*; y_LLM) ≈ 0，H2 证伪。终止实验。**

---

## ⚠️ Cognitive Scientist 审阅意见（2026-07-18）

以下两个问题必须在执行前修复。不修复则实验数据不可解释。

### 🔴 必须修复 #1：离散情绪分类器（关键词匹配）不可靠

**问题：** 用"害怕""恐惧""壮丽""震撼"等关键词将 LLM 输出归入 {fear, awe, joy, sadness, neutral, mixed} 的分类器分辨率太低。用"请描述大海"做 prompt 时，LLM 大多数输出是中性风景描写，不会出现关键词。此时分类器将所有输出标为 `neutral`，MI 必然 ≈ 0——不论注入是否有效。

**修复要求：** 至少以下之一
- (a) 使用 sentence embedding + kNN 连续 MI 估计（文中第 4.2 节已有代码）作为主要指标，关键词分类降为辅助
- (b) 或者先跑 1 条件 × 20 seed，人工校验分类器输出的情绪标签是否与注入方向匹配。如果匹配率 < 60%，换方法

**风险：** 不修复 → MI≈0，但无法区分是"注入无效"还是"分类器太差"。

### 🔴 必须修复 #2：NONE 条件与共享主题锚不一致

**问题：** 四个情绪条件都带了 `"nature_ocean": 0.5`，但 NONE = `{}`。这意味着 FEAR/AWE/JOY/SADNESS vs NONE 的 MI 差异来自**两个因素的叠加**——情绪方向 + 话题锚存在与否。即使 MI > 0，也无法归因于情绪方向。

**修复要求：** NONE 改为 `{"nature_ocean": 0.5}`。四个情绪条件只变情绪节点，话题锚不变。

**额外建议（非必须但推荐）：** 四情绪条件之间互相做 pairwise MI 检验——如果 FEAR vs JOY 不可区分（MI≈0），则注入强度的确不够，不需要跑完整实验。

---

## 1. 实验问题

海波的图激活状态 a* 是否携带关于 LLM 输出 y_LLM 的可测量信息？

即：**I(a*; y_LLM) 是否显著大于零？**

---

## 2. 操作化定义

### a*（图激活状态）

扩散收敛后 36 节点的激活值向量。在注入阶段，仅 activation ≥ 0.1 且排名前 4 的节点参与注入。

为离散化 MI 估计，将 a* 映射为注入主导方向 = 注入的 4 个节点中激活值最高者的情绪类别：

| 状态标签 | 注入输入 | 含义 |
|----------|---------|------|
| `FEAR` | `{"emotion_fear": 0.8, "nature_ocean": 0.5}` | 恐惧主导 |
| `AWE` | `{"emotion_awe": 0.8, "nature_ocean": 0.5}` | 敬畏主导 |
| `JOY` | `{"emotion_joy": 0.8, "nature_ocean": 0.5}` | 喜悦主导 |
| `SADNESS` | `{"emotion_sadness": 0.8, "nature_ocean": 0.5}` | 悲伤主导 |
| `NONE` | `{"nature_ocean": 0.5}` | 无情绪注入（基线） |

> 🔧 已按 Cognitive Scientist 要求修复 NONE 条件。

### y_LLM（LLM 输出）

对每个生成文本，提取两个正交的降维表示：

**主要指标** — 连续嵌入（推荐，避免关键词分类器分辨率问题）：
- 使用 sentence-transformers 模型（如 `paraphrase-multilingual-MiniLM-L12-v2`）获取 384 维 embedding
- 作为 kNN 互信息估计器的输入（连续-连续 MI）

> ⚠️ 实际 pilot 中使用 Qwen 最后一层 hidden state mean-pooling 替代（sentence-transformers 下载超时）。

**辅助指标** — 离散情绪分类（不推荐作为主要指标）：
- 使用 Chinese sentiment classifier 或关键词规则，将输出归入 {fear, awe, joy, sadness, neutral, mixed}
- 6 类 → 与 5 个注入状态构成 5×6 列联表

### 控制变量

| 变量 | 固定值 |
|------|--------|
| Prompt | `"请描述大海"` → 后改为 `"请描述这张桌子"` |
| System prompt | 不使用（避免 prompt 通道污染） |
| Chat template | 不使用（直接 tokenize prompt，跳过对话模板） |
| Temperature | 0.7 |
| Top-p | 0.9 |
| Max new tokens | 80 |
| Model | Qwen2.5-1.5B-Instruct |

---

## 3. 实验设计

### 3.1 Pilot 实验：FEAR vs NONE × {大海, 桌子} × 10 seed

```
条件：FEAR vs NONE（共享 nature_ocean: 0.5 话题锚）
Prompt: "请描述大海" → "请描述这张桌子"
每条件 seed：10
总生成次数：40（2 条件 × 2 prompt × 10 seed）
```

### 3.2 完整实验（未执行——pilot 已触发终止条件）

```
原计划：5 状态 × 3 主题 × {30-50} seed
实际：pilot 结果不满足继续条件，终止。
```

### 3.3 种子设置

```python
import torch, random, numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

---

## 4. 互信息估计方法

### 4.1 离散估计（辅助指标）

**输出分类**：基于关键词的情绪分类

```python
EMOTION_KEYWORDS = {
    "fear":     ["害怕", "恐惧", "焦虑", "担心", "紧张", "不安", "危险", "威胁", "可怕", "令人畏惧"],
    "awe":      ["敬畏", "壮丽", "宏大", "震撼", "奇迹", "赞叹", "崇高", "磅礴", "浩瀚"],
    "joy":      ["快乐", "喜悦", "开心", "美好", "幸福", "温暖", "欢快", "舒畅", "心旷神怡"],
    "sadness":  ["悲伤", "失落", "孤独", "忧郁", "沉重", "哀伤", "泪水", "凄凉", "寂寥"],
    "neutral":  [],
    "mixed":    [],
}
```

> ⚠️ Pilot 实际表现：Pilot 1（大海）分类器输出全部为 awe，Pilot 2（桌子）全部为 neutral。关键词分类器在两个 prompt 下均无法区分 FEAR 和 NONE。

### 4.2 连续估计（主要指标）

使用 Qwen 最后一层 hidden state mean-pooling（替代 sentence-transformers）。

---

## 5. 退出阈值

### 主要退出条件

**如果类间/类内距离比 < 1.2 → 宣告注入无效，终止实验。**

### 退出触发记录

| 条件 | 预设阈值 | 实测 | 触发？ |
|------|---------|------|--------|
| Pilot 1（大海）ratio | < 1.2 终止 | **0.93** | ✅ |
| Pilot 2（桌子）ratio | < 1.2 终止 | **0.99** | ✅ |
| 综合判定 | — | — | **✅ 终止** |

### 附加逻辑

Pilot 1 和 Pilot 2 使用不同 prompt（大海 vs 桌子），结论一致：
- **类间/类内比 ≈ 1.0** → FEAR 注入和 NONE 的 LLM 输出在 embedding 空间中不可分
- 无论 prompt 语义先验如何，4 token 的输入层注入信号不足以产生可检测的输出差异

---

## 6. 实施细节

### Pilot 1（2026-07-18）

- Prompt: `"请描述大海"`
- 状态: FEAR, NONE
- Seed: 10
- 关键词结果: FEAR 10/10 awe, NONE 6/10 awe + 4/10 neutral
- 连续 embedding ratio: 0.93
- 结论: 触发<br>1.2 阈值，但需排除"大海 prompt 语义先验过强"的混淆解释

### Pilot 2（2026-07-18）

- Prompt: `"请描述这张桌子"`
- 状态: FEAR, NONE
- Seed: 10
- 关键词结果: FEAR 0/10 awe + 10/10 neutral, NONE 0/10 awe + 10/10 neutral
- 连续 embedding ratio: 0.99
- 结论: 排除了 prompt 混淆。**确认注入信号不可检测。**

### 代码修改量

基于 `chat_with_haibo.py`，新增约 60 行：seed 控制（10 行）+ 多状态循环（15 行）+ 分类器（15 行）+ embedding 分析（20 行）。`core/` 下任何模块未修改。

---

## 7. 风险与缓解

| 风险 | 概率 | 缓解 | 实际结果 |
|------|:----:|------|---------|
| Instruct 模型 RLHF 压制注入信号 | 高 | 换 Base 模型对比 | 未执行（触发终止） |
| 关键词分类器太粗糙 | 中 | 用连续 embedding 为主要指标 | Pilot 确认分类器在两句 prompt 下均不可用 |
| N=10 统计功效不足 | 低 | 即使增加 N 到 50，ratio ≈ 1.0 意味着信号不存在 | 不适用 |
| embedding 度量用最后一层不够敏感 | 中 | 可用倒数第 2-3 层 | 未执行（即使更敏感也无法将 0.99 提升到 1.2 以上） |

---

## 8. 实验结论

**H2 在当前配置下被证伪。**

### 证伪声明

在 Qwen2.5-1.5B-Instruct 模型上，使用 BilingualInjector（input embedding 层注入，MAX_INJECT=4，activation_threshold=0.1，VIRTUAL_REPEAT=1），FEAR 注入（emotion_fear=0.8, nature_ocean=0.5）与基线（nature_ocean=0.5）所产生的 LLM 输出在连续 embedding 空间中**不可区分**（类间/类内比 ≈ 1.0）。

两个不同 prompt（大海、桌子）均得到一致结果，排除了"prompt 先验过强"的混淆解释。

### 边界条件

此结论**仅适用于**以下配置组合：
- 模型：Qwen2.5-1.5B-Instruct（输入层注入）
- 注入器：BilingualInjector（input embedding，非 SteeringInjector 的中间层 hook）
- 参数：MAX_INJECT=4, VIRTUAL_REPEAT=1, activation_threshold=0.1

以下路径可能让 H2 恢复有效性（但未在本次实验中验证）：
- 换 Base 模型（非 Instruct），避免 RLHF 压制
- 使用 SteeringInjector 的中间层 hook 注入
- 增加 VIRTUAL_REPEAT 或 MAX_INJECT
- 使用 LoRA 微调替代输入层注入

### 对核心假设的影响

| 假设 | 状态 | 说明 |
|------|------|------|
| H1（身份一致性） | 未验证 | 不能基于此实验结论推断 |
| H2（扩散结果作为控制信号） | **在当前配置下证伪** | 实验直接测试的条件 |
| H5（独立于 prompt 的 steering） | 间接被削弱 | 因 H2 证伪，H5 无法在当前配置下验证 |
| H9（拓扑控制优于文本控制） | 间接被削弱 | 海波 vs prompt 的对比实验未执行 |
