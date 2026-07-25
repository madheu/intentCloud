# P0-R2 实验指令 — 给执行者

## 背景
P0-R1 盲测 61.7% ✅。L4（词性过滤）贡献最大（20/60），L5（义原重叠）因 HowNet 词汇覆盖不足而 0/60 触发。本实验：**保留 L3+L4，用腾讯 200d embedding 替换 L5。**

## 数据
- 腾讯词向量：`E:\intentCloud\models\text2vec-word2vec-tencent-chinese\light_Tencent_AILab_ChineseEmbedding.bin`（143,613 词 × 200 维）
- 加载方式：`from gensim.models import KeyedVectors; kv = KeyedVectors.load_word2vec_format(path, binary=True)`
- LLM 语境数据：`06_Experiment/p0_b2_llm_contexts.json`（20 词 × 各义项 50 句 = 2100 句）
- 输入文件：`data/dev_input.json`（38 句）和 `data/blind_input.json`（60 句）——**不变**

## 修改

### 新增 L5-embedding：义项原型向量比对

**训练阶段（基于 2100 句 LLM 数据）：**

```python
def build_prototypes(llm_data, kv):
    """
    为每个歧义词的每个义项计算原型向量。

    llm_data: p0_b2_llm_contexts.json 的结构
              {"苹果": {"科技公司": ["句1", "句2", ...], "水果": [...]}, ...}

    kv: gensim KeyedVectors (腾讯 200d)

    返回:
      prototypes["苹果"]["科技公司"] = np.array([0.12, -0.34, ...])  # 200d
    """
    prototypes = {}
    for word, senses in llm_data.items():
        prototypes[word] = {}
        for sense, sentences in senses.items():
            all_vecs = []
            for sent in sentences:
                # 提取语境词（排陈目标词）
                context_words = [w for w in sent if w != word and w in kv]
                if not context_words:
                    continue
                # 取所有语境词的腾讯向量平均
                vecs = [kv[w] for w in context_words]
                all_vecs.append(np.mean(vecs, axis=0))
            if all_vecs:
                # 全 50 句的平均 → 义项原型
                prototypes[word][sense] = np.mean(all_vecs, axis=0)
    return prototypes
```

**推理阶段（对新句子）：**

```python
def predict_embedding(target_word, sentence, occurrence, candidate_senses, prototypes, kv):
    """
    1. 提取句中语境词（子串匹配，排陈目标词）
    2. 对能找到腾讯向量的语境词取平均 → 语境向量
    3. 语境向量与候选义项原型向量做余弦相似度
    4. 最高分且 > 第二高分 × 1.2 → 输出，否则 None
    """
    # 语境词提取（子串匹配）
    context_words = []
    for w in kv.key_to_index:
        if w != target_word and w in sentence:
            context_words.append(w)

    if not context_words:
        return None

    context_vec = np.mean([kv[w] for w in context_words], axis=0)

    scores = {}
    for cs in candidate_senses:
        proto = prototypes.get(target_word, {}).get(cs)
        if proto is None:
            scores[cs] = 0.0
        else:
            scores[cs] = np.dot(context_vec, proto) / (
                np.linalg.norm(context_vec) * np.linalg.norm(proto) + 1e-8
            )

    if not scores:
        return None

    sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
    best, best_score = sorted_scores[0]
    second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0

    if second_score == 0 or best_score >= second_score * 1.2:
        return best
    return None
```

### 级联（L3 → L4 → L5-embedding → MFS）

```
L3 复合词优先（不变）
  → 如果确定 → 输出
  → 否则 → L4 词性过滤（不变）
            → 如果确定 → 输出
            → 否则 → L5-embedding（新增）
                      → 如果确定 → 输出
                      → 否则 → MFS 回退（不变）
```

## 约束（继承 P0-R 所有约束）
- 预测脚本不读任何金标
- MFS 只从开发集算
- 每层输出多个标签
- 子串匹配，不用 jieba 整词（避免 P0-A 教训）
- 输出 debug 字段包含 L5-embedding 的各候选得分

## 输出

1. `p0_r2_predict.py`（复制 `p0_r1_predict.py`，改 L5 + 加原型构建）
2. `p0_r2_score.py`（复制 `p0_r1_score.py`，不变）
3. 跑完后的对比表：

| 版本 | L3 | L4 | L5 | MFS回退 | 级联 |
|:----|:--:|:--:|:--:|:------:|:----:|
| P0-R1（义原） | 0/60 | 20/60 | 0/60 | 40/60 | 61.7% |
| **P0-R2（embedding）** | **?** | **?** | **?** | **?** | **?** |

## 通过标准
- 级联盲测 ≥ P0-R1 的 61.7%
- L5-embedding 触发数 > 0/60（至少比死掉的义原好）
- L5-embedding 得分有区分度（多候选得分不完全平坦）
