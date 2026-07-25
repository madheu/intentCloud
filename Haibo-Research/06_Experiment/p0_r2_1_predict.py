"""P0-R2.1: L5-embedding 优化（阈值1.05 + 原型去噪 + jieba补语境 + debug）
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

kv = KeyedVectors.load_word2vec_format(KV_PATH, binary=True)
print(f"腾讯词向量: {len(kv.index_to_key)} 词")

# ── 修改2: 去噪声原型构建 (top-20) ──
print("构建去噪声原型(top-20)...")
with open(LLM_DATA) as f:
    llm_data = json.load(f)

prototypes = {}
for word, senses in llm_data.items():
    prototypes[word] = {}
    for sense, sentences in senses.items():
        sent_vecs = []
        for sent in sentences:
            ctx = [w for w in sent if w != word and w in kv]
            if len(ctx) < 2: continue
            sent_vecs.append(np.mean([kv[w] for w in ctx], axis=0))
        if not sent_vecs: continue
        mean_vec = np.mean(sent_vecs, axis=0)
        # 按与均值向量的余弦排序，取 top-20
        sims = [np.dot(v, mean_vec) / (np.linalg.norm(v)*np.linalg.norm(mean_vec)+1e-8) for v in sent_vecs]
        idxs = np.argsort(sims)[-20:][::-1] if len(sims) > 20 else list(range(len(sims)))
        top_vecs = [sent_vecs[i] for i in idxs]
        prototypes[word][sense] = np.mean(top_vecs, axis=0)
        print(f"  {word}/{sense}: {len(top_vecs)}句原型")

# ── 开发集 MFS ──
with open(os.path.join(DATA, "dev_gold.json")) as f:
    dev_gold = json.load(f)
word_labels = {}
for it in dev_gold:
    word_labels.setdefault(it["target_word"], []).append(it["_gold_label"])
mfs = {w: Counter(ls).most_common(1)[0][0] for w, ls in word_labels.items()}
all_l = [l for ls in word_labels.values() for l in ls]
global_mfs = Counter(all_l).most_common(1)[0][0]
print(f"MFS: {len(mfs)}词, 全局={global_mfs}")

def layer1(w): return mfs.get(w, global_mfs)

# ── L3: 复合词优先 ──
def find_span(s, t, occ=1):
    p, c = 0, 0
    while True:
        i = s.find(t, p)
        if i == -1: return None
        c += 1
        if c == occ: return (i, i+len(t))
        p = i + 1

def layer3(sent, word, occ, cands):
    sp = find_span(sent, word, occ)
    if not sp: return None
    words = list(pseg.cut(sent)); cp = 0
    for w, pos in words:
        we = cp + len(w)
        if cp <= sp[0] and we >= sp[1] and len(w) > len(word):
            return w
        cp = we
    return None

# ── L4: 词性过滤 ──
POS_MAP = {"科技公司":["n","nr","nt","nz"],"水果":["n"],"文书办公":["n"],"日用包装":["n"],"物理光线":["n"],"修辞评价":["v","a","d"],"植物":["n"],"消费":["v"],"评价可以":["v","a"],"排列行列":["n","q"],"行业银行":["n","nt"],"人体嘴巴":["n"],"出入口空间":["n"]}
def layer4(sent, word, occ, cands):
    sp = find_span(sent, word, occ)
    if not sp: return None
    words = list(pseg.cut(sent)); cp = 0; tp = None
    for w, pos in words:
        if cp == sp[0] and w == word: tp = pos; break
        cp += len(w)
    if not tp: return None
    f = [c for c in cands if any(tp.startswith(p) for p in POS_MAP.get(c,["n","v","a"]))]
    return f[0] if len(f) == 1 else None

# ── L5-embedding (修改1+3+4) ──
def layer5_emb(sent, word, occ, cands):
    if word not in prototypes: return None, {}
    # 修改3: 子串匹配 + jieba 补全
    sub_ctx = [w for w in kv.key_to_index if w != word and w in sent and len(w) >= 2]
    jieba_ctx = [w for w in jieba.lcut(sent) if w in kv and w != word and len(w) >= 2]
    ctx_words = list(set(sub_ctx) | set(jieba_ctx))
    if not ctx_words: return None, {}
    ctx_vec = np.mean([kv[w] for w in ctx_words], axis=0)
    scores = {}
    for cs in cands:
        proto = prototypes.get(word, {}).get(cs)
        if proto is None: scores[cs] = 0.0
        else: scores[cs] = float(np.dot(ctx_vec, proto) / (np.linalg.norm(ctx_vec)*np.linalg.norm(proto)+1e-8))
    ss = sorted(scores.items(), key=lambda x: -x[1])
    best, bs = ss[0]; sec = ss[1][1] if len(ss) > 1 else 0
    # 修改1: 阈值 1.05
    if sec == 0 or bs >= sec * 1.05:
        return best, scores  # 修改4: 返回完整 scores
    return None, scores

# ── 级联 ──
def predict(item):
    s, w, o, cs = item["sentence"], item["target_word"], item["target_occurrence"], item["candidate_senses"]
    l3 = layer3(s, w, o, cs)
    if l3: return l3, {"path": "L3"}
    l4 = layer4(s, w, o, cs)
    if l4: return l4, {"path": "L4"}
    l5, sc = layer5_emb(s, w, o, cs)
    if l5: return l5, {"path": "L5", "L5_detail": sc}
    return layer1(w), {"path": "MFS"}

# ── 主流程 ──
os.makedirs(OUTPUT, exist_ok=True)
for name in ["dev", "blind"]:
    with open(os.path.join(DATA, f"{name}_input.json")) as f:
        items = json.load(f)
    results = []; lc = Counter()
    for it in items:
        pred, debug = predict(it)
        lc[debug["path"]] += 1
        results.append({"id": it["id"], "prediction": pred, "resolved_by": debug["path"],
                        "debug": {"L5_detail": debug.get("L5_detail", {})}})
    with open(os.path.join(OUTPUT, f"{name}_predictions.json"), "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"{name}: {len(items)}条, 层={dict(lc)}")
