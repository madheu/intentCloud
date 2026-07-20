# 执行者任务指令

## 角色

你是海波项目的**研究执行者**。你的职责是严格按照实验设计运行代码、收集数据、返回原始结果。不做理论判断，不改实验设计，不加功能。

---

## 项目背景（帮你理解你在做什么）

海波是一个研究"语义如何从词序列中涌现"的项目。我们已经完成：

- **Haibo 1.0**：手写 36 节点语义图 + embedding 注入 LLM → 实验证伪（控制信号太弱）
- **P0 系列**：词级向量不能区分语义歧义 → 90° 正交结论
- **P0-A 系列**：尝试概念域、自动扩展、布尔 4 轴消歧 → 手工词表不扩展，jieba 不可控

当前研究方向：**用 HowNet（知网）的义原（sememe）体系作为 Token Graph 的语义基础。** 同时用 LLM 生成的高质量语境句作为训练数据。

---

## 当前任务

### 并行路径 A：P0-B-1 HowNet 义原接入

**目标**：从 HowNet 获取 6 个歧义词的义原定义，用义原重叠度做消歧。

#### Step A1：获取 HowNet 数据

检查以下路径是否有 HowNet 数据：

```python
import os
paths_to_check = [
    r"E:\intentCloud\models\hownet\hownet.dat",
    r"E:\intentCloud\models\hownet\howNet.xml",
]
for p in paths_to_check:
    print(f"{p}: {'✅ found' if os.path.exists(p) else '❌ not found'}")
```

如果不存在，安装 OpenHowNet：

```bash
pip install OpenHowNet
```

然后用 OpenHowNet 加载数据：

```python
from OpenHowNet import HowNetDict
hownet = HowNetDict()
# 测试：获取"苹果"的定义
senses = hownet.get_senses("苹果")
print(senses)
```

如果 OpenHowNet 也无法加载，从认知架构研究者的资源库获取 hownet.dat 文件。

#### Step A2：提取 6 个歧义词的义原

歧义词和义项：

| 词 | 义项 1 | 义项 2 | 义项 3 |
|:--:|:-------|:-------|:-------|
| 苹果 | 科技公司 | 水果 | — |
| 纸 | 文书/办公 | 包装/日用 | — |
| 光 | 光线/物理 | 修辞/评价 | — |
| 花 | 植物 | 消费 | — |
| 行 | 评价/可以 | 排列/行列 | 行业/银行 |
| 口 | 嘴巴/人体 | 出入口/空间 | — |

对每个词，提取 HowNet 中每个义项的 DEF（定义）和义原列表。

输出格式（每个词一个条目）：

```python
SENSE_SEMEMES = {
    "苹果": {
        "科技公司": {"Company", "Produce", "ElectronicProduct", ...},
        "水果": {"Fruit", "Edible", "Plant", ...},
    },
    # ... 其他词
}
```

注意：HowNet 可能没有精确匹配我们的义项名称。用最接近的义项。如果某个歧义词的所有义项在 HowNet 中没有区分，记录这个缺口（"HowNet 未区分该词的歧义"）。

#### Step A3：定义义原→语义面映射

从 HowNet 的常见义原中，建立一个映射表：

| 语义面 | 包含的义原 |
|:------|:----------|
| **物**（实体） | Thing, Entity, Object, Animal, Plant, Fruit, Tool, Artifact, Material, Body, Organ, ... |
| **事**（事件） | Event, Action, Process, Activity, Behavior, Produce, Create, Change, Move, ... |
| **质**（属性） | Property, Attribute, Quality, Color, Size, Value, Degree, Speed, ... |
| **序**（关系） | Relation, Time, Space, Location, Part, Possession, Cause, Purpose, Manner, ... |

需要覆盖 HowNet 中 Top-200 的常见义原。如果某个义原不知归到哪个面，归到最接近的；如果无法归类，跳过（不参与匹配）。

输出文件：`Haibo-Research/07_Implementation/hownet_sememe_to_face_mapping.md`

#### Step A4：改写消歧逻辑

```python
def hownet_disambiguate(sentence, ambiguous_word):
    """
    HowNet-based disambiguation.
    
    1. Tokenize sentence (substring matching — NOT jieba)
    2. For each token found, look up its HowNet sememes
    3. Collect all sememes from context words (excluding the ambiguous word itself)
    4. For the ambiguous word, look at each candidate sense's sememe set
    5. Score each sense = count of overlapping sememes with context
    6. If top score >= second_score × 1.5: return top sense's domain label
       Else: return "不确定"
    """
```

注意：
- 用**子串匹配**分词（不是 jieba）。P0-A-5 证明子串匹配在词表不全时更鲁棒
- 只匹配词表中已有的词（HowNet 已有义原的词）
- 歧义词本身不贡献到语境义原集（避免自激活）

#### Step A5：跑测试

用 P0-A-1 的全部 38 句测试。输出每个测试句的：
- 分词结果
- 命中的语境词及其义原
- 歧义词各候选义项的得分
- 最终判定

#### Step A6：结果对比

| 版本 | 苹果 | 纸 | 光 | 花 | 行 | 口 | 总计 |
|:----:|:---:|:--:|:--:|:--:|:--:|:--:|:---:|
| 手写概念域 | 82% | 17% | 33% | 83% | 0% | 75% | 50% |
| 布尔 4 轴 | 45% | 83% | 50% | 50% | 40% | 25% | 50% |
| **HowNet 义原** | **?** | **?** | **?** | **?** | **?** | **?** | **?** |

---

### 并行路径 B：P0-B-2 整合 LLM 生成的语境句

（这部分的数据由 PI 用最强 LLM 生成。你负责整合进测试框架。）

#### Step B1：接收 LLM 数据

接收 PI 提供的约 **20 个歧义词 × 每个义项 50 句** 的 LLM 生成数据。

文件格式预期：

```json
{
  "苹果": {
    "科技公司": ["苹果发布了搭载新芯片的手机。", "苹果应用商店的规则更新了。", ...],
    "水果": ["今年的苹果果肉很甜。", "苹果的采摘季节到了。", ...]
  },
  "杜鹃": {
    "花": [...],
    "鸟": [...]
  },
  ...
}
```

#### Step B2：从 LLM 句中提取义原

对 LLM 生成的所有句子，跑 HowNet 义原提取（复用 Step A2/A4 的逻辑）。

#### Step B3：对比 LLM 数据 vs. 原始测试句

比较：
- 原始测试句（P0-A-1 的 38 句）在 HowNet 消歧下的正确率
- LLM 生成句在 HowNet 消歧下的正确率

如果 LLM 数据上的正确率显著高于原始测试句，说明 LLM 训练数据有效。如果两者相近，说明 HowNet 本身的质量比数据源更重要。

---

## 交付物

1. `Haibo-Research/06_Experiment/p0_b1_hownet.py` — 完整脚本（含 HowNet 加载、义原提取、消歧逻辑、测试）
2. `Haibo-Research/07_Implementation/hownet_sememe_to_face_mapping.md` — 义原→语义面映射表
3. 控制台输出：
   - 6 个歧义词各自的义原义项提取结果
   - 38 句测试逐句结果（分词、匹配、得分、判定）
   - 对比表（HowNet vs 布尔 vs 手写概念域）
4. 对 HowNet 的质量评估：
   - 覆盖了多少测试词？
   - 歧义词的义项是否完整？
   - 是否有明显错误标注？

---

## 注意事项

- **不要修改实验设计**：如果 HowNet 不可用（数据缺失、义原未覆盖测试词），记录问题，不要绕过
- **不要加功能**：不要尝试"优化"消歧算法，严格按 Step A4 实现
- **遇到错误就停**：如果 HowNet 加载失败、义原提取返回空值等，记录错误并停止，不要自行找替代方案
- **子串匹配，不是 jieba**：P0-A-5 和 P0-A-6 已经证明 jieba 整词匹配带来不可控的复合词问题

---

## 参考文件

- 测试用例：`Haibo-Research/06_Experiment/p0_a1_multi_ambiguous.py`
- 布尔 4 轴参考实现：`Haibo-Research/06_Experiment/p0_a5_boolean.py`
- 手写概念域参考：`Haibo-Research/06_Experiment/p_resolve_10_hierarchical.py`
- 认知架构研究者分析：`Haibo-Research/04_Review/Cognitive_Reviewer/question_20260719.md`
- 预计算法复杂度：加载 HowNet 约 30s，每个词的义原提取 < 1s，38 句消歧 < 1s
