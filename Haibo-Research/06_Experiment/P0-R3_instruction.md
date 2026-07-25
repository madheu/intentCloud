# P0-R3 实验指令：Clean-room 中文语义消歧基准

> 版本：1.0  
> 日期：2026-07-21  
> 状态：待执行  
> 目标：在无金标泄漏、目标跨度明确、候选义项完整、严格评分的条件下，首次可信地测量 L3 复合词、L4 词性和 L5 腾讯 embedding 的独立贡献与级联增益。

---

## 一、核心科学问题

> 对明确标记的中文多义目标词，复合词、词性和腾讯 embedding 义项原型是否分别提供可复现的语义消歧信号？将 L5 加入 L3→L4→MFS 后，是否在全新自然语料盲测上产生显著净增益？

本轮只验证 WSD 评测和三类信号，不测试 Semantic Emergence，不接入 BriLLM，不调整 Semantic Graph 架构。

## 二、为什么必须重做

P0-R2.1 的 63.3% 无效，原因包括：

1. `blind_input.json` 每条只包含金标对应的唯一候选义项。
2. L3 返回“门径、纸杯”等表面复合词，而不是候选 sense ID。
3. scorer 将所有未知预测静默映射为 A。
4. L3/L4 提前返回后，L5 只执行 1/60；空 debug 被误读为 OOV。
5. embedding 原型按字符构建，推理按词构建，特征单位不一致。
6. 原盲测数据已经被读取、评分和反复调参，不能再次作为盲测。

P0-R3 不继承任何旧成绩。旧 38 句和旧 60 句只能作为开发、诊断和单元测试材料。

---

## 三、角色隔离

本实验必须由三个逻辑角色完成。

### 角色 A：数据整理者

- 从自然中文来源抽取候选句。
- 标记目标字符跨度并生成空白标注模板。
- 不开发或修改预测算法。
- 不查看任何模型预测。

### 角色 B：模型执行者

- 实现验证器、L3、L4、L5、级联和预测输出。
- 只能使用开发数据和无标签 `blind_input.json`。
- 永远不能读取完整盲测金标、已填标注模板或数据整理者的选句说明。
- 收到盲测输入后不得修改代码、配置、原型或阈值。

### 角色 C：PI / 金标保管者

- 由用户本人担任。
- 私下完成或组织双人标注。
- 在预测文件和哈希冻结前不公开金标。
- 最终运行严格评分器。

**角色 A 与角色 B 不得由同一个人、同一个 AI 线程或共享完整上下文的执行者承担。** 如资源有限，先让角色 B 完成并冻结代码，再用独立新线程承担角色 A。

---

## 四、固定目标词与义项表

盲测使用 9 个目标词、20 个义项，每个义项 6 句，共 120 句。

| 目标词 | sense ID | 中文说明 |
|:---:|:---|:---|
| 苹果 | `apple.company` | 科技公司、品牌或公司行为 |
| 苹果 | `apple.fruit` | 水果、种植、食用或营养 |
| 小米 | `xiaomi.company` | 科技公司、品牌或产品 |
| 小米 | `xiaomi.grain` | 谷物、种植、烹饪或食用 |
| 杜鹃 | `cuckoo.flower` | 杜鹃花、植物或花期 |
| 杜鹃 | `cuckoo.bird` | 杜鹃鸟、鸟类行为或生态 |
| 花 | `hua.plant` | 花朵、植物、生长或观赏 |
| 花 | `hua.spend` | 花费金钱、时间或资源 |
| 光 | `guang.physical` | 物理光线、光谱、照射或传播 |
| 光 | `guang.figurative` | 非物理用法，包括“只、用尽、光荣”等现有粗粒度集合 |
| 行 | `xing.approval` | 可以、可行、同意或评价 |
| 行 | `xing.row` | 行列、排列或文本中的一行 |
| 行 | `xing.industry` | 行业、行当或同业 |
| 口 | `kou.body` | 嘴、口腔或人体器官 |
| 口 | `kou.space` | 入口、出口、开口等空间开口 |
| 头 | `tou.body` | 头部、头顶等身体部位 |
| 头 | `tou.leader` | 头领、负责人或领导者 |
| 头 | `tou.beginning` | 开头、起点或开始部分 |
| 结 | `jie.concrete` | 绳结、打结等具体结构或动作 |
| 结 | `jie.abstract` | 结论、结果、总结等抽象结束或产物 |

不在表中的其他词义不得强行归类。例如“口”作为人口量词、“头”作为量词、无法区分的真实歧义句均应排除并替换。

---

## 五、盲测语料构建

### 5.1 数量与平衡

- 每个 sense ID 恰好 6 句。
- 总计恰好 120 句。
- 同一来源文章最多贡献 2 句。
- 不得使用旧 38 句、旧 60 句或 `p0_b2_llm_contexts.json` 中的句子。

### 5.2 来源要求

允许：

- 中文 Wikipedia 原文；
- 正规中文新闻原文；
- 已公开中文书面语料；
- 项目已有但从未用于 P0-R 开发和调参的真实语料。

禁止：

- LLM 直接生成或改写最终盲测句；
- 为增加提示而追加同义词；
- 字典式释义句；
- 将多个明显提示词人工堆在一句中；
- 从旧测试句轻微改写得到的新句。

LLM 只能帮助查找可能来源，最终句子必须能追溯到真实中文原文。

### 5.3 句子质量门槛

每句必须同时满足：

1. 长度为 8-60 个中文字符，不含标题残片或表格残片。
2. 目标字符串在句中恰好出现 1 次。
3. `sentence[target_start:target_end] == target_surface`。
4. 目标 occurrence 在该句中只能对应一个 sense ID。
5. 去除目标词后，句子仍保留自然上下文，但不得人工堆叠答案提示。
6. 不与其他盲测句近似重复。
7. 不包含由排版、OCR 或分词错误造成的伪歧义。

### 5.4 盲测输入格式

角色 A 生成：

`06_Experiment/data/p0_r3/blind_input.json`

```json
[
  {
    "id": "p0r3-0001",
    "sentence": "苹果发布了面向开发者的新系统。",
    "target_surface": "苹果",
    "target_start": 0,
    "target_end": 2,
    "candidate_sense_ids": [
      "apple.company",
      "apple.fruit"
    ],
    "source_id": "source-001"
  }
]
```

要求：

- 每条必须包含该目标词的**完整候选集**，候选顺序按本指令固定。
- 不含 `label`、`gold`、`sense_name`、`intended_sense` 或任何答案字段。
- `source_id` 只用于去重；盲测输入不包含可能直接暴露答案的来源标题。

### 5.5 空白标注模板

角色 A 同时生成：

`P0-R3_blind_label_template.md`

每条格式：

```markdown
### p0r3-0001

- 句子：苹果发布了面向开发者的新系统。
- 目标：苹果（0:2）
- 候选：
  - apple.company：科技公司
  - apple.fruit：水果
- 金标：[待 PI 私下填写]
```

角色 A 交付空白模板和 `blind_input.json` 后停止工作，不参与后续预测。

---

## 六、标注与金标保管

1. PI 将空白模板复制到 `E:\intentCloud` 之外、角色 B 无法读取的位置。
2. 两名标注者独立选择 sense ID，不查看模型预测。
3. 计算原始一致率和 Cohen's kappa。
4. 要求 kappa ≥ 0.80；低于 0.80 时先修订义项说明并重新标注整批数据。
5. 分歧项由第三方仲裁；无法达成一致的句子必须替换，保持每义项 6 句。
6. 最终金标保存为私有 `P0-R3_blind_gold.json`，不得写入项目工作区。
7. 在模型预测前计算金标文件 SHA-256，并只公开哈希，不公开内容。

私有金标格式：

```json
[
  {"id": "p0r3-0001", "gold_sense_id": "apple.company"}
]
```

---

## 七、开发数据与原型数据

### 7.1 开发数据

- 旧 38 句和旧 60 句可用于调试，但必须转换为本指令的 sense ID。
- 每条开发输入也必须包含完整候选集和明确 target span。
- 标签不一致或不属于固定义项表的旧句应删除，不得强行映射。
- 开发成绩只用于选择固定配置，不进入最终结论。

### 7.2 MFS

- MFS 只能从清理后的开发金标计算。
- MFS 的值是每个目标词对应的 sense ID，不得继续使用通用 A/B/C。
- 某目标在开发集平局或缺失时，回退到预注册的 HowNet 第一义项映射。
- 禁止使用跨目标的“全局 A”或全局 sense ID，因为不同目标的义项不可互换。

### 7.3 Embedding 原型数据

- 使用 `p0_b2_llm_contexts.json` 作为原型训练数据，不作为盲测。
- 固定映射：

| 原数据目标/义项 | P0-R3 sense ID |
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

---

## 八、统一语境特征提取

原型构建和盲测推理必须调用同一个函数：

```python
extract_context_tokens(sentence, target_start, target_end, kv) -> list[str]
```

固定规则：

1. 使用 jieba 产生带字符跨度的 token。
2. 删除与目标 span 有任何重叠的 token。
3. 删除纯标点、纯数字和停用功能词。
4. 保留存在于腾讯词表中的 token。
5. 单字只有在 jieba 将其切为独立 token 且存在于腾讯词表时才保留。
6. 不遍历 `for w in sentence` 构建字符原型。
7. 不扫描整个 `kv.key_to_index` 做全句子串匹配。
8. 输出去重前的自然 token 序列；同一词重复出现应保留频次。

必须对同一句开发句证明：原型路径和推理路径提取出的 token 完全相同。

---

## 九、各层算法

所有层必须对每条样本独立计算，即使级联前层已经判定。级联只能在全部层完成诊断后选择最终输出。

### 9.1 L1：开发集 MFS

输出每个目标词在开发集中的 MFS sense ID。它是冻结基线和最终级联回退，不读取盲测金标。

### 9.2 L2：HowNet 第一义项

- 将 HowNet 第一义项映射到固定 sense ID。
- 映射表在开发阶段完成并写入配置。
- 映射失败时弃权，禁止默认到第一个候选。
- L2 只作为基线，不进入主级联。

### 9.3 L3：目标跨度覆盖复合词

1. 只查找覆盖 `target_start:target_end` 的 jieba/HowNet 合法复合词。
2. 取最长覆盖词。
3. 获取覆盖词的 HowNet 义项与义原。
4. 分别和目标的每个候选 sense 义原计算 Jaccard 相似度。
5. 只有唯一最高分、最高分 > 0 且 `best - second >= 0.10` 时判定。
6. 输出必须是 `candidate_sense_ids` 中的 sense ID。
7. 否则弃权。

严禁返回“门径、纸杯、入口”等表面词。

### 9.4 L4：目标词性过滤

1. 从目标 span 对应的 jieba token 取得 POS。
2. 根据冻结配置中的 sense→允许 POS 映射过滤候选。
3. 只剩一个候选时判定。
4. 零个或多个候选时弃权。
5. 如果目标被包在复合词中且无法取得可靠目标 POS，L4 必须弃权，不得使用整句其他词的 POS。

### 9.5 L5：腾讯 embedding 原型

#### 原型构建

对每个训练句：

1. 使用统一 `extract_context_tokens()`。
2. 取语境 token 的腾讯向量。
3. 先求句子平均向量并做 L2 归一化。
4. 同一 sense 下平均所有有效句向量，再做 L2 归一化。
5. 不做查看盲测结果后的 top-k 去噪。

#### 推理

1. 使用同一函数得到语境 token。
2. 无有效 token 时标记 `computable=false` 并弃权。
3. 计算归一化语境向量与每个候选原型的 cosine。
4. 始终输出 forced-choice 最佳候选用于独立诊断。
5. selective 判定同时满足：
   - `best_score >= min_similarity`
   - `best_score - second_score >= min_margin`
6. 未满足阈值时弃权。

#### 阈值冻结

- 只能使用开发数据选择 `min_similarity` 和 `min_margin`。
- 网格固定为：
  - `min_similarity ∈ {0.00, 0.10, 0.20, 0.30}`
  - `min_margin ∈ {0.00, 0.02, 0.05, 0.10}`
- 目标：最大化开发集目标词宏平均准确率，且 selective coverage ≥ 20%。
- 并列时选择较大的 `min_margin`，再选择较大的 `min_similarity`。
- 选定值写入只读配置并计入冻结哈希。

### 9.6 两个主级联

```text
Cascade-A: L3 → L4 → MFS
Cascade-B: L3 → L4 → L5-selective → MFS
```

判定规则：前层返回合法 sense ID 时采用，否则进入下一层。不得根据句子或目标词临时改变层顺序。

---

## 十、预测输出

角色 B 生成：

`06_Experiment/output/p0_r3/blind_predictions.json`

```json
[
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
]
```

每条都必须包含 L3/L4/L5 独立结果。空 `L5_scores` 只能表示 L5 已执行但不可计算，不得表示“因为前层返回所以没运行”。

---

## 十一、严格验证器与评分器

### 11.1 预测前验证

必须提供 `p0_r3_validate_input.py` 并验证：

- 恰好 120 条、ID 唯一；
- 9 个目标词、20 个 sense ID；
- 每条候选集与固定义项表完全一致；
- target span 非空且精确匹配；
- 每句目标字符串恰好出现一次；
- 无任何 gold/label 字段；
- 无旧测试句精确重复。

任何失败立即终止。

### 11.2 预测输出验证

必须验证：

- 输入与输出 ID 集合完全相同；
- `prediction` 必须属于该条 `candidate_sense_ids`；
- L2/L3/L4/L5 的非空输出也必须属于候选集；
- 不允许缺失、重复或额外预测；
- 不允许 A/B/C 或表面复合词作为预测；
- 每条均执行 L5 独立诊断。

未知预测必须报错，严禁默认到任一 sense。

### 11.3 独立评分

`p0_r3_score.py` 只能执行：

```python
correct = prediction_sense_id == gold_sense_id
```

不得使用 `LABEL_MAP`、模糊匹配、字符包含或 `.get(key, default)`。

评分前先验证金标：

- ID 集合与输入完全一致；
- `gold_sense_id` 属于该条完整候选集；
- 每个 sense ID 恰好 6 句。

---

## 十二、冻结与揭盲流程

### 阶段 1：开发

角色 B 在旧开发数据上完成：

- 代码；
- 单元测试；
- 固定义项/POS/HowNet 映射配置；
- embedding 原型；
- L5 阈值；
- 严格输入/输出验证器；
- scorer 的开发集测试。

### 阶段 2：冻结

在收到新 `blind_input.json` 前，对以下文件计算 SHA-256：

- `p0_r3_predict.py`
- `p0_r3_validate_input.py`
- `p0_r3_score.py`
- 固定配置文件；
- embedding 原型文件；
- 所有单元测试。

将哈希写入 `P0-R3_frozen_manifest.sha256` 并交给 PI。

### 阶段 3：盲测预测

1. PI/角色 A 提供无标签 `blind_input.json`。
2. 角色 B 先运行输入验证器。
3. 角色 B 不修改任何冻结文件，直接运行预测。
4. 运行输出验证器。
5. 计算 `blind_predictions.json` 的 SHA-256 并交给 PI。
6. 到此角色 B 停止，不得查看金标或修改预测。

### 阶段 4：揭盲评分

1. PI 核对冻结哈希与预测哈希。
2. PI 在私有环境中运行 scorer。
3. PI 保存完整结果和逐句差异。
4. 任何盲测后改动只能进入下一实验版本，不能覆盖 P0-R3。

---

## 十三、必须提交的指标

### 13.1 每个独立层

- computable coverage；
- selective decision coverage；
- coverage 内准确率；
- 全集准确率（弃权按错）；
- 每目标词准确率；
- 9 个目标词宏平均；
- L5 forced-choice 准确率；
- OOV token 比例和至少一个有效语境词的句子比例。

### 13.2 基线与级联

必须列出：

| 方法 | micro accuracy | macro by target | coverage |
|:---|:---:|:---:|:---:|
| MFS |  |  | 100% |
| HowNet first sense |  |  |  |
| L3 independent |  |  |  |
| L4 independent |  |  |  |
| L5 forced |  |  | computable |
| L5 selective |  |  | selective |
| Cascade-A：L3→L4→MFS |  |  | 100% |
| Cascade-B：L3→L4→L5→MFS |  |  | 100% |

### 13.3 配对统计

主要比较：

1. Cascade-B vs MFS；
2. Cascade-B vs Cascade-A；
3. L5 forced vs MFS（仅 L5 computable 的共同样本）。

报告：

- 正确率差值；
- 两方向 discordant counts；
- 配对 McNemar exact p-value；
- 按目标词聚类 bootstrap 的 95% 置信区间。

---

## 十四、有效性闸门与通过标准

### 14.1 有效性闸门

以下任一失败，整轮标记为 **INVALID**，不得报告性能结论：

- 角色 B 接触盲测金标；
- 代码/配置/原型在收到盲测输入后发生变化；
- 候选集不完整或泄漏正确答案；
- 目标跨度错误；
- 非法预测被默认映射；
- L5 没有对全部 120 条独立运行；
- 盲测句由 LLM 生成或从旧测试改写；
- 预测后修改阈值、原型或测试数据。

### 14.2 性能通过标准

在有效性闸门全部通过后，P0-R3 判为性能通过需同时满足：

1. Cascade-B 比 MFS 高至少 5.0 个百分点；
2. Cascade-B vs MFS 的 McNemar exact `p < 0.05`；
3. L5 computable coverage ≥ 80%；
4. L5 selective decision coverage ≥ 20%；
5. Cascade-B 相对 Cascade-A 的净增正确数大于净增错误数。

未达到性能标准不代表实验无效，只能得出“当前方法没有可靠增益”。

---

## 十五、强制单元测试

角色 B 在冻结前必须证明：

1. scorer 对未知 sense ID 抛错。
2. scorer 对 A/B/C 抛错。
3. scorer 对“门径、纸杯”等表面词抛错。
4. 任何单候选 blind input 被验证器拒绝。
5. 缺失或多余 ID 被拒绝。
6. 错误 target span 被拒绝。
7. L3 只能返回候选 sense ID 或 `None`。
8. L4 只能返回候选 sense ID 或 `None`。
9. L5 对被 L3/L4 判定的样本仍产生独立 debug。
10. 原型和推理调用同一 `extract_context_tokens()`。
11. 对“专业能力提升后，进阶门径需要专门指导”进行词表诊断时，至少识别本地腾讯词表实际覆盖的“专业、能力、提升、指导、专门”，不得再次声称全部 OOV。
12. predictor 源码和运行日志中不存在 blind gold 文件路径。

---

## 十六、执行者交付物

### 角色 A

- `data/p0_r3/blind_input.json`
- 空白 `P0-R3_blind_label_template.md`
- 私下交给 PI 的来源清单，不进入角色 B 工作区
- 数据验证报告

### 角色 B

- `p0_r3_predict.py`
- `p0_r3_validate_input.py`
- `p0_r3_score.py`
- 固定配置与原型文件
- 单元测试及完整通过日志
- `P0-R3_frozen_manifest.sha256`
- `output/p0_r3/blind_predictions.json`
- `output/p0_r3/blind_predictions.sha256`
- 无金标预测日志和各层覆盖统计

### 角色 C / PI

- 私有 `P0-R3_blind_gold.json`
- 标注一致性报告
- 金标 SHA-256
- 最终评分报告
- P0-R3 有效/无效判定

---

## 十七、执行顺序

1. 将本指令分别发给角色 A 和角色 B。
2. 角色 B 只用开发数据完成代码、测试、配置、原型和阈值。
3. 角色 B 提交冻结 manifest。
4. 独立角色 A 构建 120 句自然盲测和空白标注模板。
5. PI 私下完成双人标注、仲裁、金标文件与哈希。
6. PI 只向角色 B 提供 `blind_input.json`。
7. 角色 B 验证、预测、验证输出并提交预测哈希，然后停止。
8. PI 独立评分并出具结果。

执行者不得合并步骤、跳过停止点，或以“先跑一下看看”为理由提前读取金标。
