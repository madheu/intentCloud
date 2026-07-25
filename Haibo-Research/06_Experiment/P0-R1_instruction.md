# P0-R1 实验指令 — 给执行者

## 角色
研究执行者。实现以下5层基线，严格按约束执行。破坏任何约束 = 实验无效。

## 核心约束

| # | 约束 | 违规后果 |
|:--|:-----|:--------|
| 1 | 预测脚本不得读取任何金标文件（dev 或 blind） | 实验无效 |
| 2 | MFS 基线只能从开发集金标计算 | 实验无效 |
| 3 | 每层必须输出多个不同标签。如果某层对所有样本输出同一标签 → 脚本报错终止 | — |
| 4 | 目标跨度使用字符位置匹配（`target_word` + 句内出现次数编号） | — |
| 5 | 级联使用 Layer3→4→5→MFS，不使用 Layer2 | — |

## 文件

| 文件 | 用途 | 可读取？ |
|:----|:-----|:--------|
| `p0_r1_predict.py` | 读取输入数据，输出预测文件 | ✅ |
| `p0_r1_score.py` | 读取预测文件 + 金标，输出正确率 | ✅ |
| `P0-R_gold_labels.json`（开发集38句） | 计算 MFS 时的参考 | ✅（只用于 MFS 计算） |
| `P0-R_blind_gold_labels.json`（盲测集60句） | 评分依据 | ❌ 预测时不可读 |

## 输入格式

`p0_r1_predict.py` 从 `data/dev_input.json` 和 `data/blind_input.json` 读取：

```json
[
  {
    "id": 1,
    "sentence": "苹果发布了新款手机。",
    "target_word": "苹果",
    "target_occurrence": 1,
    "candidate_senses": ["科技公司", "水果"]
  }
]
```

`target_occurrence`：目标词在句中第几次出现（1-based）。`target_span` = 句中找到第 N 个目标词的起止字符位置。

**注意**：`candidate_senses` 是 HowNet sense ID 或义项名。预测脚本输出时使用此列表中的值，不要自创标签。

## 5层基线

### Layer 1：MFS（开发集多数标签）

**方法**：从开发集金标统计每个歧义词的多数义项。对新词使用全局多数（所有目标词中的最频义项，从开发集算）。

**约束**：MFS 值必须从开发集独立计算，不能读盲测金标。**使用盲测标签计算 MFS = 实验无效。**

### Layer 2：HowNet 第一义项

**方法**：使用 OpenHowNet 获取每个目标词在 HowNet 中的第一个义项，作为预测。

**注意**：HowNet 义项名称和 candidate_senses 中的名称可能不一致。用字符包含/模糊匹配选择最接近的候选。

### Layer 3：复合词优先（目标锚定）

**方法**：
```
1. 定位目标词在句中的 target_span（字符起止位置）
2. 用 jieba 分词，找到覆盖该位置的最长 jieba 词
3. 如果覆盖词在 HowNet 中只有一个义项：
   - 将义项名映射到 candidate_senses 中最近的一个
   - 输出该候选
4. 如果无法确定 → 递交给 Layer 4
```

**注意**：不要走回 P0-A 的老路——只看指定 target_span 覆盖的词，不扫描全句找歧义词。

### Layer 4：词性过滤

**方法**：用 jieba(posseg) 获取目标词的词性。每个 candidate_sense 预设一个允许的词性列表。匹配时过滤掉不匹配的候选。

候选义项→允许词性映射示例：
- 苹果(科技公司) → ["n", "nr", "nt", "nz"]（名词/专名）
- 苹果(水果) → ["n"]
- 光(光线) → ["n"]
- 光(修辞) → ["v", "a"]（光说不做...）
- 花(植物) → ["n"]
- 花(消费) → ["v"]
- 行(评价) → ["v", "a"]
- 行(排列) → ["n", "q"]（量词）
- 口(人体) → ["n"]
- 口(空间) → ["n"]

如果 pos 过滤不匹配任何候选 → 回退到 Layer 5。

### Layer 5：HowNet 义原重叠评分

**方法**：对 target_word 每个候选 sense，计算义原重叠度：
```
context_words = 句中除歧义词外的所有 jieba 词
context_sememes = 所有 context_words 的 HowNet 义原的并集
score(sense) = |sense的义原 ∩ context_sememes| / |sense的义原|
```

**注意**：每个候选 sense 独立计算。选择 score 最高的候选。如果最高分 < 第二高分 × 1.2 → 不判定。

### 级联

```
Layer 3（复合词）→ 如果确定 → 输出
                → 如果未确定 → Layer 4（词性过滤）
                              → 如果确定 → 输出
                              → 如果未确定 → Layer 5（义原重叠）
                                            → 如果确定 → 输出
                                            → 如果未确定 → MFS 回退
```

**MFS 回退值从开发集计算，不能从盲测集计算。**

## 输出格式

`p0_r1_predict.py` 生成两个文件：

**`output/dev_predictions.json`** 和 **`output/blind_predictions.json`**：

```json
[
  {
    "id": 1,
    "prediction": "科技公司",
    "resolved_by": "L3",
    "debug": {
      "MFS": "水果",
      "L2_howNet_first": "科技公司",
      "L3_compound": "苹果",
      "L3_mapped": "科技公司",
      "L4_pos": "n",
      "L4_filtered": ["科技公司", "水果"],
      "L5_scores": {"科技公司": 0.67, "水果": 0.33},
      "L5_winner": "科技公司",
      "cascade_path": "L3"
    }
  }
]
```

## 自检要求

实现后，跑以下自检：

```python
# 自检 1：每层输出多样性
assert len(set(layer_outputs)) >= 2, f"Layer X 只输出单一标签"

# 自检 2：预测文件中无盲测标签
# grep -E "A|B|C" blind_predictions.json（预测值不是 A/B/C，是义项名）

# 自检 3：级联路径覆盖
assert set(cascade_paths) in [{"L3"}, {"L3","L4"}, {"L3","L4","L5"}, {"L3","L4","L5","MFS"}]
```

## 输出要求

1. **`06_Experiment/p0_r1_predict.py`** — 完整脚本
2. **`06_Experiment/p0_r1_score.py`** — 独立评分脚本
3. **`06_Experiment/data/dev_input.json`** — 从 38 句金标构造
4. **`06_Experiment/data/blind_input.json`** — 从 60 句构造（不含标签）
5. 跑完 `predict.py` → 生成预测文件 → 跑 `score.py` 对比金标

## 通过标准

- 预测脚本**从未读取**盲测金标（执行者确认）
- 级联输出有多个不同标签
- 每层自检全部通过
- 层间递进趋势可观测（至少有一层比 MFS 好）