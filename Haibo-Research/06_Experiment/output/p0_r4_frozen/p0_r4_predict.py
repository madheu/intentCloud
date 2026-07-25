"""P0-R4 预测管线 — 命令行入口: python p0_r4_predict.py <input.json>"""
import json, os, sys, numpy as np
sys.path = [p for p in sys.path if "Diviner" not in p]
import jieba, jieba.posseg as pseg
from gensim.models import KeyedVectors

BASE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(BASE, "output")

with open(os.path.join(BASE, "p0_r4_config.json")) as f:
    CONFIG = json.load(f)
SENSE_IDS = CONFIG["sense_ids"]
MFS_MAP = CONFIG["mfs_map"]
HOWNET_FIRST = CONFIG["hownet_first"]
STOP_WORDS = set(CONFIG["stop_words"])
MS = CONFIG["l5_min_similarity"]
MM = CONFIG["l5_min_margin"]
KV_PATH = r"E:\intentCloud\models\text2vec-word2vec-tencent-chinese\light_Tencent_AILab_ChineseEmbedding.bin"

kv = KeyedVectors.load_word2vec_format(KV_PATH, binary=True)

# 加载冻结产物
protos_data = np.load(os.path.join(OUTPUT, "p0_r4_prototypes.npz"), allow_pickle=True)
protos = {}
for k in protos_data.files:
    v = protos_data[k]
    if v.ndim == 1 and v.shape[0] == 200:
        protos[k] = v

with open(os.path.join(OUTPUT, "p0_r4_sememe_map.json")) as f:
    SEMEME_MAP = json.load(f)

# ── 工具函数 ──
def extract_ctx(sent, ts, te):
    words = []; cp = 0
    for w, pos in pseg.cut(sent):
        we = cp + len(w)
        if cp < te and we > ts: cp = we; continue
        if all(c in "，。、！？：；""''（）【】《》——…·,.:;!?()[]{}" for c in w) or w.isdigit() or w in STOP_WORDS: cp = we; continue
        if w in kv: words.append(w)
        cp = we
    return words

def get_spans(s):
    ws = []; cp = 0
    for w, pos in pseg.cut(s):
        ws.append((w, pos, cp, cp+len(w))); cp += len(w)
    return ws

# ── L3 ──
def layer3(sent, ts, te, cands):
    covering = [(w, pos) for w, pos, cs, ce in get_spans(sent) if cs <= ts and ce >= te and len(w) > (te-ts)]
    if not covering: return None
    cp_word = max(covering, key=lambda x: len(x[0]))[0]
    cp_sems = set()
    data = hownet.get_sememes_by_word(cp_word) if cp_word else None
    if data:
        for entry in data:
            for s in entry.get("sememes", []):
                cp_sems.add(str(s).split("|")[0].strip())
    if not cp_sems: return None
    sc = {}
    for sid in cands:
        cand_sems = set(SEMEME_MAP.get(sid, []))
        if not cand_sems: sc[sid] = 0.0
        else:
            inter = len(cp_sems & cand_sems)
            sc[sid] = inter / max(len(cp_sems | cand_sems), 1)
    ss = sorted(sc.items(), key=lambda x: -x[1])
    if ss[0][1] > 0 and (ss[0][1] - (ss[1][1] if len(ss) > 1 else 0)) >= 0.10:
        return ss[0][0]
    return None

# ── L4 ──
def layer4(sent, ts, te, cands):
    tp = next((pos for w, pos, cs, ce in get_spans(sent) if cs == ts and ce == te), None)
    if not tp: return None
    fl = [s for s in cands if any(tp.startswith(p) for p in CONFIG["sense_pos_map"].get(s, ["n", "v", "a"]))]
    return fl[0] if len(fl) == 1 else None

# ── L5 ──
def layer5(sent, ts, te, cands):
    toks = extract_ctx(sent, ts, te)
    if not toks: return None, {}, None, None
    cv = np.mean([kv[t] for t in toks], axis=0)
    cv = cv / (np.linalg.norm(cv) + 1e-8)
    sc = {s: float(np.dot(cv, protos[s])) if s in protos else 0. for s in cands}
    ss = sorted(sc.items(), key=lambda x: -x[1])
    sel = ss[0][0] if ss[0][1] >= MS and (ss[0][1] - (ss[1][1] if len(ss) > 1 else 0)) >= MM else None
    return ss[0][0], sc, ss[0][0], sel

# ── MFS 回退 ──
def fallback(w):
    r = MFS_MAP.get(w)
    if r: return r
    return HOWNET_FIRST.get(w)

# ── 级联 ──
def cascade_a(sent, ts, te, cands):
    l3 = layer3(sent, ts, te, cands)
    if l3: return l3, "L3"
    l4 = layer4(sent, ts, te, cands)
    if l4: return l4, "L4"
    return fallback(sent[ts:te]), "MFS"

def cascade_b(sent, ts, te, cands):
    l3 = layer3(sent, ts, te, cands)
    if l3: return l3, "L3"
    l4 = layer4(sent, ts, te, cands)
    if l4: return l4, "L4"
    _, _, _, sel = layer5(sent, ts, te, cands)
    if sel: return sel, "L5"
    return fallback(sent[ts:te]), "MFS"

# ── 单条预测 ──
def predict_entry(item):
    w, sent = item["target_surface"], item["sentence"]
    ts, te = item["target_start"], item["target_end"]
    cands = item["candidate_sense_ids"]
    ind = {"MFS": MFS_MAP.get(w), "L2": HOWNET_FIRST.get(w), "L3": None, "L4": None, "L5_forced": None, "L5_selective": None}
    ind["L3"] = layer3(sent, ts, te, cands)
    ind["L4"] = layer4(sent, ts, te, cands)
    best, sc, f5, s5 = layer5(sent, ts, te, cands)
    ind["L5_forced"] = f5; ind["L5_selective"] = s5
    toks = extract_ctx(sent, ts, te)
    comp = next((w for w, pos, cs, ce in get_spans(sent) if cs <= ts and ce >= te and len(w) > (te-ts)), None)
    tp = next((pos for w, pos, cs, ce in get_spans(sent) if cs == ts and ce == te), None)
    db = {"target_span_valid": sent[ts:te] == w, "covering_compound": comp, "target_pos": tp,
          "context_tokens": toks, "context_oov_tokens": [], "L5_computable": len(toks) > 0,
          "L5_scores": sc, "L5_best_margin": max(sc.values()) - min(sc.values()) if sc else 0}
    ca, _ = cascade_a(sent, ts, te, cands)
    cb, cb_layer = cascade_b(sent, ts, te, cands)
    db["cascade_A"] = ca; db["cascade_B"] = cb
    return {"id": item["id"], "prediction": cb, "resolved_by": cb_layer, "independent": ind, "debug": db}

# ── 主入口 ──
def main():
    if len(sys.argv) != 2:
        print("用法: python p0_r4_predict.py <input.json>", file=sys.stderr)
        sys.exit(1)
    input_path = sys.argv[1]
    with open(input_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    results = [predict_entry(it) for it in items]
    set_name = "blind" if "blind" in os.path.basename(input_path).lower() else "dev"
    out_path = os.path.join(OUTPUT, f"p0_r4_{set_name}_predictions.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"预测完成: {len(results)} 条 → {out_path}", file=sys.stderr)

if __name__ == "__main__":
    main()
