# P0-R3 角色 B 指令：模型开发、冻结与盲测预测

> 角色：模型执行者（Role B）  
> 日期：2026-07-21  
> 本文件可独立执行，无需承担数据整理或人工标注工作。  
> 主实验协议：`P0-R3_instruction.md`

---

## 1. 你的唯一任务

实现并冻结一套严格的中文语义消歧预测管线，在收到无标签盲测输入后不改代码地运行一次预测。

你负责：

- 清理旧开发数据格式；
- 实现输入/输出验证器；
- 实现 MFS、HowNet 第一义项、L3、L4、L5；
- 实现两个级联；
- 构建腾讯 embedding 义项原型；
- 在开发集上冻结阈值和配置；
- 提交代码/配置/原型哈希；
- 收到无标签盲测输入后运行一次预测并提交预测哈希。

你不负责：

- 收集或生成 120 句新盲测语料；
- 人工标注盲测句；
- 创建、读取或保管盲测金标；
- 查看数据整理者的选句标签、来源分类或私下说明；
- 在揭盲后修改预测；
- 解释最终显著性或决定项目方向。

---

## 2. 绝对隔离规则

以下任一发生，本轮实验立即标记为 **INVALID**：

1. 读取任何 P0-R3 盲测金标或已填写的标注模板。
2. 在冻结 manifest 后修改预测代码、配置、原型、阈值或测试。
3. 在收到新盲测输入后根据句子修改规则或词表。
4. 用旧 60 句的 63.3% 作为通过标准或调参目标。
5. 将非法预测默认映射为第一个候选、A 或任意 sense。
6. 让 L3/L4 提前返回，从而跳过 L5 独立诊断。
7. 输出表面词、A/B/C 或不在候选集中的值。

禁止读取或搜索：

- `P0-R3_blind_gold.json`
- 已填写的 `P0-R3_blind_label_template.md`
- 数据整理者私下保存的 intended sense 或来源分类
- 任何名称含 `p0_r3` 且包含 `gold` 的盲测文件

允许读取：

- 旧 38 句和旧 60 句开发/诊断数据；
- `p0_b2_llm_contexts.json`；
- 腾讯 200d 词向量；
- HowNet；
- 本文件和 `P0-R3_instruction.md`；
- 冻结完成后由 PI 提供的无标签 `blind_input.json`。

---

## 3. 固定义项表

所有预测必须使用以下 sense ID。不得使用 A/B/C，不得自创别名。

| 目标 | 完整候选 sense ID |
|:---:|:---|
| 苹果 | `apple.company`, `apple.fruit` |
| 小米 | `xiaomi.company`, `xiaomi.grain` |
| 杜鹃 | `cuckoo.flower`, `cuckoo.bird` |
| 花 | `hua.plant`, `hua.spend` |
| 光 | `guang.physical`, `guang.figurative` |
| 行 | `xing.approval`, `xing.row`, `xing.industry` |
| 口 | `kou.body`, `kou.space` |
| 头 | `tou.body`, `tou.leader`, `tou.beginning` |
| 结 | `jie.concrete`, `jie.abstract` |

将此表写入只读配置，例如 `p0_r3_config.json`。输入验证器必须要求每条候选集与对应目标的完整候选集完全一致。

---

## 4. 工作阶段与停止点

### 阶段 B1：只用开发数据开发

在此阶段不得接触新 120 句盲测输入。完成：

1. 开发数据转换；
2. MFS 和 HowNet 第一义项映射；
3. L3/L4/L5；
4. embedding 原型；
5. 阈值选择；
6. 输入/输出验证器；
7. 严格 scorer；
8. 单元测试；
9. 开发集诊断报告。

### 阶段 B2：冻结并停止

计算并提交 `P0-R3_frozen_manifest.sha256` 后，**立即停止**，等待 PI 提供无标签盲测输入。

### 阶段 B3：一次性盲测预测

收到 `blind_input.json` 后：

1. 核对冻结 manifest；
2. 运行输入验证；
3. 不改代码直接预测；
4. 运行输出验证；
5. 生成预测文件和 SHA-256；
6. 向 PI 交付后停止。

你不执行最终揭盲评分。PI 会在私有环境中运行 scorer。

---

## 5. 开发数据规范

旧 38 句和旧 60 句可作为开发材料，但必须先清洗：

- 转换为本文件固定 sense ID；
- 每条包含准确 `target_start`、`target_end`；
- 每条包含该目标的完整候选集；
- 删除标签不一致、目标 occurrence 不明确或不属于固定义项表的句子；
- 旧数据结果不得称为盲测成绩。

开发输入格式：

```json
{
  "id": "dev-0001",
  "sentence": "苹果发布了新系统。",
  "target_surface": "苹果",
  "target_start": 0,
  "target_end": 2,
  "candidate_sense_ids": ["apple.company", "apple.fruit"]
}
```

开发金标单独保存：

```json
{"id": "dev-0001", "gold_sense_id": "apple.company"}
```

---

## 6. MFS 与 HowNet 基线

### MFS

- 只能从清洗后的开发金标计算。
- 每个目标词分别得到一个 sense ID。
- 平局或开发集缺失时，使用冻结的 HowNet 第一义项映射。
- 禁止使用跨目标的全局 A/B/C 或全局 sense ID。

### HowNet 第一义项

- 将 HowNet 第一义项映射到候选 sense ID。
- 映射表必须在开发阶段写入配置并冻结。
- 映射失败时返回 `None`，禁止默认到第一个候选。
- 仅作为独立基线，不进入主级联。

---

## 7. 统一语境 token 提取

原型构建和推理必须调用同一个函数：

```python
extract_context_tokens(sentence, target_start, target_end, kv) -> list[str]
```

固定规则：

1. 使用 jieba 产生带字符跨度的 token。
2. 删除与目标 span 有任何重叠的 token。
3. 删除标点、纯数字和冻结停用词。
4. 只保留存在于腾讯词表中的 token。
5. 单字只有在 jieba 将其切为独立 token 且存在于腾讯词表时保留。
6. 保留自然出现频次，不做 set 去重。
7. 禁止 `for w in sentence` 按字符构建原型。
8. 禁止遍历整个 `kv.key_to_index` 扫描句子子串。

对同一句开发样本，训练路径与推理路径必须提取出完全相同的 token。

---

## 8. L3：目标跨度覆盖复合词

对每条样本独立运行：

1. 只检查覆盖 `target_start:target_end` 的 jieba/HowNet 合法复合词。
2. 取最长覆盖词。
3. 获取覆盖词的 HowNet 义项和义原。
4. 获取目标每个候选 sense 的义原。
5. 逐候选计算 Jaccard 相似度。
6. 仅在唯一最高分、最高分 > 0 且 `best - second >= 0.10` 时输出候选 sense ID。
7. 否则返回 `None`。

L3 只能返回：

- `candidate_sense_ids` 中的值；或
- `None`。

严禁返回“门径、纸杯、入口”等表面复合词。

---

## 9. L4：目标词性过滤

1. 使用目标 span 定位对应 jieba token 和 POS。
2. 根据冻结配置中的 sense→允许 POS 映射过滤候选。
3. 只剩一个候选时输出该 sense ID。
4. 零个或多个候选时返回 `None`。
5. 目标被包在复合词中且无法获得可靠目标 POS 时必须返回 `None`。

POS 映射只能根据 HowNet、语言学定义和开发数据建立，盲测输入到达后不得修改。

---

## 10. L5：腾讯 embedding 义项原型

### 10.1 原型数据映射

使用 `p0_b2_llm_contexts.json`，固定映射：

| 原目标/义项 | sense ID |
|:---|:---|
| 苹果/科技公司 | `apple.company` |
| 苹果/水果 | `apple.fruit` |
| 小米/科技 | `xiaomi.company` |
| 小米/谷物 | `xiaomi.grain` |
| 杜鹃/花 | `cuckoo.flower` |
| 杜鹃/鸟 | `cuckoo.bird` |
| 花/植物 | `hua.plant` |
| 花/消费 | `hua.spend` |
| 光/物理光线 | `guang.physical` |
| 光/修辞评价 | `guang.figurative` |
| 行/评价 | `xing.approval` |
| 行/排列 | `xing.row` |
| 行/行业 | `xing.industry` |
| 口/人体 | `kou.body` |
| 口/空间 | `kou.space` |
| 头/人体 | `tou.body` |
| 头/领导 | `tou.leader` |
| 头/起始 | `tou.beginning` |
| 结/具体 | `jie.concrete` |
| 结/抽象 | `jie.abstract` |

### 10.2 原型构建

每个训练句：

1. 调用统一 `extract_context_tokens()`；
2. 平均有效 token 向量；
3. 将句向量做 L2 归一化；
4. 同一 sense 下平均全部有效句向量；
5. 将 sense 原型再次 L2 归一化。

不得使用盲测结果做 top-k 去噪或删除训练句。

### 10.3 两种 L5 输出

对每条样本都计算：

- `L5_forced`：只要语境可计算，就输出 cosine 最高的候选；
- `L5_selective`：只有置信门槛通过才输出，否则 `None`。

置信门槛：

```text
best_score >= min_similarity
best_score - second_score >= min_margin
```

不得使用 cosine 比值阈值。

### 10.4 阈值选择

只在开发集搜索：

- `min_similarity ∈ {0.00, 0.10, 0.20, 0.30}`
- `min_margin ∈ {0.00, 0.02, 0.05, 0.10}`

选择目标词宏平均准确率最高且 selective coverage ≥20% 的组合。并列时优先更大的 `min_margin`，再优先更大的 `min_similarity`。

阈值写入冻结配置。收到盲测输入后禁止修改。

---

## 11. 独立诊断与主级联

必须先对每条样本独立计算 MFS、L2、L3、L4、L5，再构建级联。不得因 L3/L4 已决定就跳过 L5。

两个主级联：

```text
Cascade-A: L3 → L4 → MFS
Cascade-B: L3 → L4 → L5-selective → MFS
```

盲测前层顺序固定，不得按目标词或句子临时改变。

---

## 12. 预测输出格式

输出：

`06_Experiment/output/p0_r3/blind_predictions.json`

```json
{
  "id": "p0r3-0001",
  "prediction": "apple.company",
  "resolved_by": "L5",
  "independent": {
    "MFS": "apple.fruit",
    "L2": "apple.company",
    "L3": null,
    "L4": null,
    "L5_forced": "apple.company",
    "L5_selective": "apple.company"
  },
  "debug": {
    "target_span_valid": true,
    "covering_compound": null,
    "target_pos": "nz",
    "context_tokens": ["发布", "开发者", "系统"],
    "context_oov_tokens": [],
    "L5_computable": true,
    "L5_scores": {
      "apple.company": 0.61,
      "apple.fruit": 0.32
    },
    "L5_best_margin": 0.29,
    "cascade_A": "apple.fruit",
    "cascade_B": "apple.company"
  }
}
```

每条都必须包含完整独立诊断。空 `L5_scores` 只能表示 L5 已执行但无有效语境向量。

---

## 13. 输入验证器

实现 `p0_r3_validate_input.py`。收到盲测输入后必须验证：

- 恰好 120 条且 ID 唯一；
- 只包含固定 9 个目标和 20 个 sense ID；
- 每条候选集与该目标完整候选集完全一致；
- target span 精确匹配；
- 每句目标字符串恰好出现 1 次；
- 不含 `label`、`gold`、`sense_name`、`intended_sense`；
- 不与旧 38/60 句精确重复。

任何失败立即终止，不得自行修复盲测文件。

---

## 14. 输出验证器

预测后必须验证：

- 输入与输出 ID 集合完全相同；
- 无缺失、重复或额外预测；
- 最终 prediction 属于该条候选集；
- L2/L3/L4/L5 的非空输出均属于候选集；
- 不存在 A/B/C、表面复合词或未知字符串；
- 全部 120 条都包含 L5 独立诊断。

任何非法预测必须报错。禁止默认映射。

---

## 15. 严格 scorer

实现 `p0_r3_score.py`，但角色 B 只用开发金标测试它。最终由 PI 在私有环境运行。

核心评分只能是：

```python
correct = prediction_sense_id == gold_sense_id
```

禁止：

- `LABEL_MAP`；
- A/B/C 转换；
- 模糊匹配；
- 字符包含；
- `.get(key, default)`；
- 未知预测默认正确类别。

scorer 必须先验证 gold ID 集与输入一致，且 gold sense 属于完整候选集。

---

## 16. 冻结前强制测试

必须提供自动测试并展示通过日志：

1. 未知 sense ID 被拒绝。
2. A/B/C 被拒绝。
3. “门径、纸杯”等表面词被拒绝。
4. 单候选输入被拒绝。
5. 不完整候选集被拒绝。
6. 缺失、重复或额外 ID 被拒绝。
7. 错误 target span 被拒绝。
8. L3 只能返回合法 sense ID 或 `None`。
9. L4 只能返回合法 sense ID 或 `None`。
10. L5 在 L3/L4 已判定时仍执行独立诊断。
11. 原型与推理调用同一 token 提取函数。
12. “专业、能力、提升、指导、专门”能从本地腾讯词表识别，不得再次误报为全部 OOV。
13. predictor 源码和运行日志不包含 blind gold 路径。
14. scorer 对非法预测抛错而非返回分数。

---

## 17. 冻结 manifest

在接触新盲测输入前，对以下内容计算 SHA-256：

- `p0_r3_predict.py`
- `p0_r3_validate_input.py`
- 输出验证器
- `p0_r3_score.py`
- `p0_r3_config.json`
- embedding 原型文件
- 测试文件
- 开发测试日志

写入：

`06_Experiment/P0-R3_frozen_manifest.sha256`

提交 manifest 给 PI 后停止。后续任何改动都必须改为 P0-R4，不能覆盖本轮。

---

## 18. 盲测运行与交付

只有 PI 明确提供无标签 `blind_input.json` 后才能继续：

1. 验证冻结文件哈希未变；
2. 运行输入验证器；
3. 运行一次预测；
4. 运行输出验证器；
5. 保存完整无金标诊断；
6. 计算预测文件 SHA-256；
7. 向 PI 交付并停止。

交付：

- `p0_r3_predict.py`
- `p0_r3_validate_input.py`
- 输出验证器
- `p0_r3_score.py`
- 固定配置和 embedding 原型
- 自动测试与通过日志
- `P0-R3_frozen_manifest.sha256`
- `output/p0_r3/blind_predictions.json`
- `output/p0_r3/blind_predictions.sha256`
- 不含金标的层覆盖统计

角色 B 不得索取金标，不得运行最终盲测评分，不得在看到任何最终正确率后修改本轮产物。

---

## 19. 你现在应该做什么

现在只执行阶段 B1：

1. 阅读本文件；
2. 检查开发资源；
3. 实现代码、配置、验证器和测试；
4. 在旧开发数据上完成诊断；
5. 生成冻结 manifest；
6. 报告“已冻结，等待 blind_input”；
7. 停止。

在 PI 提供无标签 `blind_input.json` 之前，不得进入阶段 B3。
