Cognitive Scientist 你好，

Haibo 项目经过一轮审查实验（Resolve 系列，7 轮）后，核心发现如下：

1. **原路线（图 → embedding 注入 → 控制 LLM）被实验证伪**，I(a*; y_LLM) ≈ 0。Control Scientist 的 pilot 数据确认 class separation ratio ≈ 0.93，等价于 MI ≈ 0。

2. **新方向假设**：语义应由词序列中的"围合"关系定义，而非静态词向量。直觉是：一个词的意义不由它自身携带，由同时出现的其他词围合出来。

3. **P_Resolve-07**（Tencent 200d 词向量空间）：验证了"语义面朝向"假设——科技类词集合与水果类词集合在 200 维空间中各自有独立朝向，但科技-水果歧义面的夹角（~86°）与无关控制组（鲸鱼-汽车，~84°）几乎相同。结论：词级分辨率不足以区分语义歧义，所有面朝向 ≈ 90° 正交。

4. **P_Resolve-08**（Token Graph）：转向新设计。30 节点，用维基百科句子的词共现建边，1 步扩散做语境消歧。目前写好了最简实现（`p_resolve_08_token_graph.py`），手写测试用例 11 条（5 科技 / 5 水果 / 1 边界），设计文档在 `07_Implementation/token_graph_design_v0.md`。

现在需要在理论上确认以下问题：

---

### 1. "围合"在认知科学/语言学中有没有对应的理论框架？

直觉：一个词的语义由它的"邻居词集"决定（不是由向量坐标决定）。`"苹果" + {手机, 芯片, 系统, 屏幕, 应用}` → 科技。`"苹果" + {甜, 树, 采摘, 种植, 营养}` → 水果。

这是不是等价于某种分布语义学（distributional semantics / Firth 1957: "you shall know a word by the company it keeps"）？还是说有什么更特殊的结构在这里？

语言学的"语境选择"（contextual selection / semantic priming）在这个方向下能提供什么已知陷阱？

### 2. 序列级别信号 vs 静态向量信号——这个区分有理论意义吗？

Resolve 系列的核心发现是：**静态词向量（无论是 200d Tencent 还是 TF-IDF）都分不开歧义**，但**序列中的多词联合约束（P_Resolve-08 的扩散 1 步）似乎能分开**。

这在认知科学上成立吗？人处理歧义时依赖的是"这个词本身的意思"还是"这个词在句子里的位置和邻居"？有没有已知的神经科学证据支持"围合式消歧"？

### 3. 已知陷阱

从工程实验角度我们已经踩过的坑：
- 扩散步数 > 2 会淹没问题（感受野太宽，所有歧义坍缩到同一方向）
- 词表覆盖率不够时围合检测退化为关键词匹配（和原海波一样的问题）
- 手写边的泛化能力未知

认知科学上有没有类似的已知陷阱，比如"围合词集大小与消歧精度的关系"、"语境窗口长度的最优范围"？

### 4. 值得继续投入吗？

从你的角度：序列级别信号 vs 静态向量信号的区别，是否值得继续投入？还是说这只是换个方式做向量检索，最终会碰到和原海波一样的瓶颈？

---

**核心文件**：
- 新设计：`Haibo-Research/07_Implementation/token_graph_design_v0.md`
- 实验代码：`Haibo-Research/06_Experiment/p_resolve_08_token_graph.py`
- 词向量面朝向实验：`Haibo-Research/06_Experiment/p_resolve_07_tencent_embedding.py`
- 完整决策记录：`Haibo-Research/05_Decision_Log.md`

—— PI
