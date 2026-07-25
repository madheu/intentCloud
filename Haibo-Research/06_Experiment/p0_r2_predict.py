"""P0-R2: 目标锚定消歧（L5 替换为腾讯 200d embedding 原型比对）
"""
import json, os, sys, numpy as np
sys.path = [p for p in sys.path if 'Diviner' not in p]
from collections import Counter
import jieba
import jieba.posseg as pseg
from gensim.models import KeyedVectors

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
OUTPUT = os.path.join(BASE, "output")
KV_PATH = r"E:\intentCloud\models\text2vec-word2vec-tencent-chinese\light_Tencent_AILab_ChineseEmbedding.bin"
LLM_DATA = os.path.join(BASE, "p0_b2_llm_contexts.json")

# ── 加载腾讯词向量 ──
print("加载腾讯词向量...")
kv = KeyedVectors.load_word2vec_format(KV_PATH, binary=True)
print(f"  {len(kv.index_to_key)} 词, {kv.vector_size} 维")

# ── 构建义项原型向量 ──
print("构建义项原型...")
with open(LLM_DATA, "r") as f:
    llm_data = json.load(f)

prototypes = {}
for word, senses in llm_data.items():
    prototypes[word] = {}
    for sense, sentences in senses.items():
        all_vecs = []
        for sent in sentences:
            ctx_words = [w for w in sent if w != word and w in kv]
            if not ctx_words:
                continue
            vecs = [kv[w] for w in ctx_words]
            all_vecs.append(np.mean(vecs, axis=0))
        if all_vecs:
            prototypes[word][sense] = np.mean(all_vecs, axis=0)
            print(f"  {word}/{sense}: {len(all_vecs)}句原型")

# ── 加载开发集计算 MFS ──
with open(os.path.join(DATA, "dev_gold.json"), "r") as f:
    dev_gold = json.load(f)

word_mfs = {}
for item in dev_gold:
    w = item["target_word"]
    label = item["_gold_label"]
    word_mfs.setdefault(w, []).append(label)

mfs = {}
dev_all_labels = []
for w, labels in word_mfs.items():
    c = Counter(labels)
    most = c.most_common(1)[0][0]
    mfs[w] = most
    dev_all_labels.extend(labels)
global_mfs = Counter(dev_all_labels).most_common(1)[0][0]
print(f"\nMFS: {len(mfs)} 词, 全局={global_mfs}")

# ── Layer 1: MFS ──
def layer1(word):
    return mfs.get(word, global_mfs)

# ── Layer 2: (不使用) ──

# ── Layer 3: 复合词优先（同P0-R1）──
def find_target_span(sentence, target_word, occurrence=1):
    start = 0; count = 0
    while True:
        pos = sentence.find(target_word, start)
        if pos == -1: return None
        count += 1
        if count == occurrence: return (pos, pos + len(target_word))
        start = pos + 1

def layer3(sentence, target_word, target_occurrence, candidate_senses):
    span = find_target_span(sentence, target_word, target_occurrence)
    if not span: return None
    words = list(pseg.cut(sentence))
    covering = []; char_pos = 0
    for w, pos in words:
        word_end = char_pos + len(w)
        if char_pos <= span[0] and word_end >= span[1] and len(w) > len(target_word):
            covering.append((w, pos))
        char_pos = word_end
    if not covering: return None
    return covering[0][0]  # 有覆盖词就输出（简化）

# ── Layer 4: 词性过滤（同P0-R1）──
POS_MAP = {
    "科技公司": ["n","nr","nt","nz"], "水果": ["n"],
    "文书办公": ["n"], "日用包装": ["n"],
    "物理光线": ["n"], "修辞评价": ["v","a","d"],
    "植物": ["n"], "消费": ["v"],
    "评价可以": ["v","a"], "排列行列": ["n","q"],
    "行业银行": ["n","nt"],
    "人体嘴巴": ["n"], "出入口空间": ["n"],
}

def layer4(sentence, target_word, target_occurrence, candidate_senses):
    span = find_target_span(sentence, target_word, target_occurrence)
    if not span: return None
    words = list(pseg.cut(sentence)); char_pos = 0
    target_pos = None
    for w, pos in words:
        if char_pos == span[0] and w == target_word:
            target_pos = pos; break
        char_pos += len(w)
    if not target_pos: return None
    filtered = [c for c in candidate_senses if any(target_pos.startswith(p) for p in POS_MAP.get(c, ["n","v","a"]))]
    return filtered[0] if len(filtered) == 1 else None

# ── L5-embedding: 腾讯 200d 原型比对 ──
def layer5_embedding(sentence, target_word, target_occurrence, candidate_senses):
    """用腾讯 200d 原型向量做语义消歧"""
    if target_word not in prototypes:
        return None, {}
    
    # 子串匹配找语境词
    context_words = []
    for w in kv.key_to_index:
        if w != target_word and w in sentence and len(w) >= 2:
            context_words.append(w)
    
    if not context_words:
        return None, {}
    
    context_vec = np.mean([kv[w] for w in context_words], axis=0)
    
    scores = {}
    for cs in candidate_senses:
        proto = prototypes.get(target_word, {}).get(cs)
        if proto is None:
            scores[cs] = 0.0
        else:
            scores[cs] = float(np.dot(context_vec, proto) / (np.linalg.norm(context_vec) * np.linalg.norm(proto) + 1e-8))
    
    sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
    best, best_score = sorted_scores[0]
    second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0
    
    if second_score == 0 or best_score >= second_score * 1.2:
        return best, scores
    return None, scores

# ── 级联 ──
def predict_one(item):
    sent = item["sentence"]; word = item["target_word"]; occ = item["target_occurrence"]
    cands = item["candidate_senses"]
    
    # L3
    l3 = layer3(sent, word, occ, cands)
    if l3: return l3, {"cascade_path": "L3"}
    
    # L4
    l4 = layer4(sent, word, occ, cands)
    if l4: return l4, {"cascade_path": "L4"}
    
    # L5-embedding
    l5, l5_scores = layer5_embedding(sent, word, occ, cands)
    if l5: return l5, {"cascade_path": "L5", "L5_scores": l5_scores}
    
    # MFS
    return layer1(word), {"cascade_path": "MFS", "MFS_pick": layer1(word)}

# ── 主流程 ──
def main():
    os.makedirs(OUTPUT, exist_ok=True)
    for set_name in ["dev", "blind"]:
        with open(os.path.join(DATA, f"{set_name}_input.json")) as f:
            items = json.load(f)
        results = []; layer_counts = Counter()
        for item in items:
            pred, debug = predict_one(item)
            layer_counts[debug["cascade_path"]] += 1
            results.append({"id": item["id"], "prediction": pred, "resolved_by": debug["cascade_path"]})
        with open(os.path.join(OUTPUT, f"{set_name}_predictions.json"), "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"{set_name}: {len(items)}条 层分布={dict(layer_counts)}")

if __name__ == "__main__":
    main()
