"""P0-R4 构建: 开发数据 + MFS + 义原映射 + 原型（一次性运行, 产物冻结）"""
import json, os, sys, numpy as np
sys.path = [p for p in sys.path if "Diviner" not in p]
from collections import Counter
import jieba, jieba.posseg as pseg
from gensim.models import KeyedVectors
import OpenHowNet

BASE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(BASE, "output")
os.makedirs(OUTPUT, exist_ok=True)

# 配置
with open(os.path.join(BASE, "p0_r4_config.json")) as f:
    CONFIG = json.load(f)
SENSE_IDS = CONFIG["sense_ids"]
KV_PATH = r"E:\intentCloud\models\text2vec-word2vec-tencent-chinese\light_Tencent_AILab_ChineseEmbedding.bin"
LLM_DATA = os.path.join(BASE, "p0_b2_llm_contexts.json")

kv = KeyedVectors.load_word2vec_format(KV_PATH, binary=True)
hownet = OpenHowNet.HowNetDict()
assert hownet.get_sememes_by_word("手机") and any(s.get("sememes") for s in hownet.get_sememes_by_word("手机"))

# ── 1. 正确的标签映射 ──
# 旧 38 句: A/B/C + 纯文本
LABEL_38 = {  # 目标词 -> (金标A/B/C) -> sense_id
    "苹果": {"A": "apple.company", "B": "apple.fruit"},
    "光": {"A": "guang.physical", "B": "guang.figurative"},
    "花": {"A": "hua.plant", "B": "hua.spend"},
    "行": {"A": "xing.approval", "B": "xing.row", "C": "xing.industry"},
    "口": {"A": "kou.body", "B": "kou.space"},
}
# 旧 60 句: 用 sense_name（中文名）映射
LABEL_60 = {  # sense_name -> sense_id
    "科技公司": "apple.company", "水果": "apple.fruit",
    "科技": "xiaomi.company", "谷物": "xiaomi.grain",
    "花": "cuckoo.flower", "鸟": "cuckoo.bird",
    "植物": "hua.plant", "消费": "hua.spend",
    "物理光线": "guang.physical", "修辞评价": "guang.figurative",
    "评价": "xing.approval", "排列": "xing.row", "行业": "xing.industry",
    "人体": "kou.body", "空间": "kou.space",
    "人体": "tou.body", "领导": "tou.leader", "起始": "tou.beginning",
    "具体": "jie.concrete", "抽象": "jie.abstract",
}

dev_items = []; idx = 0
# 从旧 38
with open(os.path.join(BASE, "P0-R_gold_labels.json")) as f:
    old38 = json.load(f)
for word, sents in old38.items():
    if word not in SENSE_IDS: continue
    for sent, label_str in sents.items():
        idx += 1
        gold = LABEL_38.get(word, {}).get(label_str)
        if gold is None or gold not in SENSE_IDS[word]:
            raise ValueError(f"旧38句 {idx}: {word}/{label_str} → {gold} 非法")
        ts = sent.find(word)
        assert ts != -1, f"旧38句 {idx}: 无法找到'{word}'"
        dev_items.append({"id": f"dev-{idx:04d}", "sentence": sent, "target_surface": word,
                          "target_start": ts, "target_end": ts+len(word),
                          "candidate_sense_ids": SENSE_IDS[word], "_gold_sense_id": gold})

# 从旧 60
label_60_sense_name = {  # 按词分别映射以防同名冲突
    "苹果": {"科技公司": "apple.company", "水果": "apple.fruit"},
    "小米": {"科技": "xiaomi.company", "谷物": "xiaomi.grain"},
    "杜鹃": {"花": "cuckoo.flower", "鸟": "cuckoo.bird"},
    "花": {"植物": "hua.plant", "消费": "hua.spend"},
    "光": {"物理光线": "guang.physical", "修辞评价": "guang.figurative"},
    "行": {"评价": "xing.approval", "排列": "xing.row"},
    "口": {"人体": "kou.body", "空间": "kou.space"},
    "头": {"人体": "tou.body", "领导": "tou.leader", "起始": "tou.beginning"},
    "结": {"具体": "jie.concrete", "抽象": "jie.abstract"},
}
with open(os.path.join(BASE, "P0-R_blind_gold_labels.json")) as f:
    old60 = json.load(f)
for word, entries in old60.items():
    if word not in SENSE_IDS: continue
    for e in entries:
        idx += 1
        sense_name = e.get("sense_name", e.get("label", ""))
        gold = label_60_sense_name.get(word, {}).get(sense_name)
        if gold is None or gold not in SENSE_IDS[word]:
            raise ValueError(f"旧60句 {idx}: {word}/{sense_name} → {gold} 非法")
        sent = e["sentence"]; ts = sent.find(word)
        assert ts != -1, f"旧60句 {idx}: 无法找到'{word}'"
        dev_items.append({"id": f"dev-{idx:04d}", "sentence": sent, "target_surface": word,
                          "target_start": ts, "target_end": ts+len(word),
                          "candidate_sense_ids": SENSE_IDS[word], "_gold_sense_id": gold})

# 验证所有 gold 都合法
for it in dev_items:
    assert it["_gold_sense_id"] in it["candidate_sense_ids"], f"{it['id']}: gold {it['_gold_sense_id']} 不在 {it['candidate_sense_ids']}"
print(f"开发数据: {len(dev_items)} 条, 覆盖 {len(set(it['target_surface'] for it in dev_items))} 词, 全部 gold 合法")

# ── 2. MFS ──
wl = {}
for it in dev_items:
    wl.setdefault(it["target_surface"], []).append(it["_gold_sense_id"])
mfs_map = {w: Counter(ls).most_common(1)[0][0] for w, ls in wl.items()}
CONFIG["mfs_map"] = mfs_map
print(f"MFS: {mfs_map}")

# ── 3. 按候选区分的义原映射 ──
# 思路: HowNet 对每个目标词有多个 sense, 确定每个 sense 对应哪个 candidate
# 手工: 对每个 candidate, 指定它在 HowNet 中的 sense 序号 或 义项英文名
CANDIDATE_SENSE_HOWNET = {
    "apple.company": {"query": "苹果", "sense_idx": 0},  # Apple Inc.
    "apple.fruit": {"query": "苹果", "sense_idx": 1},    # fruit
    "xiaomi.company": {"query": "小米", "sense_idx": 4},
    "xiaomi.grain": {"query": "小米", "sense_idx": 0},
    "cuckoo.flower": {"query": "杜鹃", "sense_idx": 0},
    "cuckoo.bird": {"query": "杜鹃", "sense_idx": 1},
    "hua.plant": {"query": "花", "sense_idx": 0},
    "hua.spend": {"query": "花", "sense_idx": 4},
    "guang.physical": {"query": "光", "sense_idx": 6},
    "guang.figurative": {"query": "光", "sense_idx": 0},
    "xing.approval": {"query": "行", "sense_idx": 19},
    "xing.row": {"query": "行", "sense_idx": 11},
    "xing.industry": {"query": "行", "sense_idx": 3},
    "kou.body": {"query": "口", "sense_idx": 3},
    "kou.space": {"query": "口", "sense_idx": 2},
    "tou.body": {"query": "头", "sense_idx": 0},
    "tou.leader": {"query": "头", "sense_idx": 2},
    "tou.beginning": {"query": "头", "sense_idx": 3},
    "jie.concrete": {"query": "结", "sense_idx": 0},
    "jie.abstract": {"query": "结", "sense_idx": 4},
}

sememe_map = {}
for sid, cfg in CANDIDATE_SENSE_HOWNET.items():
    data = hownet.get_sememes_by_word(cfg["query"])
    sems = set()
    if data and cfg["sense_idx"] < len(data):
        entry = data[cfg["sense_idx"]]
        for s in entry.get("sememes", []):
            sems.add(str(s).split("|")[0].strip())
    sememe_map[sid] = list(sems)
    print(f"  {sid}: {len(sems)} 义原 (sense_idx={cfg['sense_idx']})")

with open(os.path.join(OUTPUT, "p0_r4_sememe_map.json"), "w") as f:
    json.dump(sememe_map, f, ensure_ascii=False)
print(f"义原映射已保存")

# ── 4. 阈值搜索 (macro accuracy) ──
def extract_ctx(sent, ts, te):
    words = []; cp = 0
    for w, pos in pseg.cut(sent):
        we = cp+len(w)
        if cp < te and we > ts: cp = we; continue
        if all(c in "，。、！？：；""''（）【】《》——…·,.:;!?()[]{}" for c in w) or w.isdigit() or w in CONFIG["stop_words"]: cp = we; continue
        if w in kv: words.append(w)
        cp = we
    return words

def layer5(sent, ts, te, cands, protos):
    toks = extract_ctx(sent, ts, te)
    if not toks: return None, {}, None, None
    cv = np.mean([kv[t] for t in toks], axis=0)
    cv = cv / (np.linalg.norm(cv) + 1e-8)
    sc = {s: float(np.dot(cv, protos[s])) if s in protos else 0. for s in cands}
    ss = sorted(sc.items(), key=lambda x: -x[1])
    return ss[0][0], sc, ss[0][0], (ss[0][0] if len(sc) > 1 and ss[0][1] >= 0. and (ss[0][1] - ss[1][1]) >= 0. else None)

# 构建原型
from collections import defaultdict
with open(LLM_DATA) as f: llm_data = json.load(f)
llm_map = {"苹果/科技公司":"apple.company","苹果/水果":"apple.fruit","小米/科技":"xiaomi.company","小米/谷物":"xiaomi.grain",
           "杜鹃/花":"cuckoo.flower","杜鹃/鸟":"cuckoo.bird","花/植物":"hua.plant","花/消费":"hua.spend",
           "光/物理光线":"guang.physical","光/修辞评价":"guang.figurative","行/评价":"xing.approval","行/排列":"xing.row",
           "行/行业":"xing.industry","口/人体":"kou.body","口/空间":"kou.space","头/人体":"tou.body","头/领导":"tou.leader",
           "头/起始":"tou.beginning","结/具体":"jie.concrete","结/抽象":"jie.abstract"}
mapped = {}
for word, senses in llm_data.items():
    for sense, sts in senses.items():
        sid = llm_map.get(f"{word}/{sense}")
        if sid: mapped[sid] = sts
protos = {}
for sid, sts in mapped.items():
    svs = []
    for st in sts:
        for t, cands in SENSE_IDS.items():
            if sid in cands:
                ts = st.find(t)
                if ts == -1: continue
                toks = extract_ctx(st, ts, ts+len(t))
                if not toks: continue
                v = np.mean([kv[t] for t in toks], axis=0)
                svs.append(v / (np.linalg.norm(v) + 1e-8)); break
    if svs:
        p = np.mean(svs, axis=0)
        protos[sid] = p / (np.linalg.norm(p) + 1e-8)

np.savez(os.path.join(OUTPUT, "p0_r4_prototypes.npz"), **{k: v for k, v in protos.items()})
print(f"原型已保存: {len(protos)} 个")

# 阈值搜索
best_macro = -1; best_cfg = None
for ms in [0.0, 0.1, 0.2, 0.3]:
    for mm in [0.0, 0.02, 0.05, 0.10]:
        wr = {w: {"c": 0, "t": 0} for w in SENSE_IDS}
        for it in dev_items:
            w = it["target_surface"]
            _, _, _, sel = layer5(it["sentence"], it["target_start"], it["target_end"],
                                  it["candidate_sense_ids"], protos)
            if sel:
                wr[w]["t"] += 1
                if sel == it["_gold_sense_id"]: wr[w]["c"] += 1
        l5_all = sum(v["t"] for v in wr.values())
        cov = l5_all / len(dev_items) * 100
        if cov < 20: continue
        wa = [v["c"]/max(v["t"],1) for w, v in wr.items() if v["t"] > 0]
        macro = np.mean(wa) if wa else 0
        if macro > best_macro or (macro == best_macro and mm > (best_cfg[1] if best_cfg else -1)) or \
           (macro == best_macro and mm == (best_cfg[1] if best_cfg else -1) and ms > (best_cfg[0] if best_cfg else -1)):
            best_macro = macro; best_cfg = (ms, mm, macro, cov)
        print(f"  ms={ms:.1f} mm={mm:.2f}: macro={macro:.3f} cov={cov:.0f}%")

if best_cfg:
    CONFIG["l5_min_similarity"] = best_cfg[0]
    CONFIG["l5_min_margin"] = best_cfg[1]
print(f"阈值: ms={CONFIG['l5_min_similarity']} mm={CONFIG['l5_min_margin']} macro={best_macro:.3f}")

# ── 保存配置 ──
with open(os.path.join(BASE, "p0_r4_config.json"), "w") as f:
    json.dump(CONFIG, f, ensure_ascii=False, indent=2)
print("配置已写入")
print(f"\n可冻结产物: config.json + prototypes.npz + sememe_map.json")
