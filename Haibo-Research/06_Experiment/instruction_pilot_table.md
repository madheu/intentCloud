# 实验指令：确认性实验 — 验证"大海"prompt 的 awe 基座是否混淆了 MI 测量

## 背景

Pilot 实验用"请描述大海"作为 prompt，发现：
- FEAR 注入条件下，10/10 输出 awe 关键词（"壮丽""敬畏""浩瀚"等）
- NONE 条件下，7/10 输出 awe 关键词
- 关键词分类器 FEAR vs NONE 的离散 MI=0.139（虚假信号——两类全部坍缩到 awe）
- 连续 embedding 的类间/类内距离比 ≈ 0.93（等价于 MI≈0）

**问题**：awe 坍缩来自"大海"prompt 的内在情绪基座，不是来自注入信号。需要排除这个混淆因素。

## Control Scientist 的最终预测

> 替换 prompt 为中性主题（桌子/天气）后，类间/类内距离比 < 1.2。注入仍然无效——不是 prompt 问题，是接口类型问题。

## 你只需执行

### 步骤 1：修改 prompt

将 `PROMPT` 从 `"请描述大海"` 改为 **`"请描述一张桌子"`**。

理由：桌子无情绪基座，任何输出差异只能来自注入信号。

### 步骤 2：跑 pilot（仅 FEAR vs NONE）

复用现有脚本 `pilot_mi_classifier.py`，只改 PROMPT 变量。

```
条件：FEAR（{"emotion_fear": 0.8, "nature_ocean": 0.5}）vs NONE（{"nature_ocean": 0.5}）
种子：10
模型：Qwen2.5-1.5B-Instruct
Prompt：请描述一张桌子
```  

### 步骤 3：看两个数字

运行脚本后，直接看输出中的两个关键数字：

**数字 A：关键词分类器的分布**
- FEAR 条件中，输出关键词含"害怕/恐惧/焦虑/紧张"的有几个？还是全部 neutral？
- 如果 FEAR 和 NONE 都坍缩到 neutral → 注入无效

**数字 B：类间/类内距离比**
- `cross / inner` 这个比值 > 1.5 还是 < 1.2？
- Control Scientist 预测 < 1.2

### 步骤 4：写入结果文件

`Haibo-Research/06_Experiment/pilot_table_results.json`

格式见 `pilot_mi_results.json`。

### 步骤 5：停止——不需要跑完整 5 条件 × 50 seed 实验

**不要跑主实验。** 这个 pilot 只需要回答一个问题：换了 prompt 后 MI 是否 > 0？

如果类间/类内比仍 < 1.2，Control Scientist 的预测被实验确认，路径 C 锁定。不需要更多数据。

---

## 技术细节

- 脚本路径：`Haibo-Research/06_Experiment/pilot_mi_classifier.py`
- 修改：第 36 行 `PROMPT = "请描述大海"` → `PROMPT = "请描述一张桌子"`
- 运行：`python Haibo-Research/06_Experiment/pilot_mi_classifier.py`
- 运行环境：CUDA，RTX 4060 8GB
- 预计耗时：~2 分钟（10 seed × 2 条件 × 3-5 秒/次）
- Python 路径：`python` 或 `C:\Python313\python.exe`

## 注意

- 不要改任何其他代码
- 不要加功能、不要改注入参数、不要增加条件
- 只需要跑一次，看结果
- **不需要写报告。** 把原始输出贴回 haibo-review.txt 或直接告诉我两个数字
