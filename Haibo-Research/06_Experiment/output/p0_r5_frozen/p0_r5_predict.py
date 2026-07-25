"""P0-R5 预测管线 — 自包含冻结入口: python p0_r5_predict.py <input.json>"""
import json, os, sys, hashlib, numpy as np
sys.path = [p for p in sys.path if "Diviner" not in p]
import jieba, jieba.posseg as pseg
from gensim.models import KeyedVectors

FROZEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "p0_r5_frozen")
if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "p0_r5_config.json")):
    FROZEN = os.path.dirname(os.path.abspath(__file__))
OUT_RUN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "p0_r5_run")

with open(os.path.join(FROZEN, "p0_r5_config.json"), encoding="utf-8") as f:
    C = json.load(f)
SENSE_IDS = C["sense_ids"]; MFS = C["mfs_map"]; HF = C["hownet_first"]
STOP = set(C["stop_words"]); MS = C["l5_min_similarity"]; MM = C["l5_min_margin"]

# 修复2: 加载旧句集合
with open(os.path.join(FROZEN, "p0_r5_old_sentences.json"), encoding="utf-8") as f:
    OLD_SENTENCES = set(json.load(f))

kv = KeyedVectors.load_word2vec_format(r"E:\intentCloud\models\text2vec-word2vec-tencent-chinese\light_Tencent_AILab_ChineseEmbedding.bin", binary=True)
proto_data = np.load(os.path.join(FROZEN, "p0_r5_prototypes.npz"), allow_pickle=True)
protos = {k: v for k, v in proto_data.items() if v.ndim == 1 and v.shape[0] == 200}
with open(os.path.join(FROZEN, "p0_r5_sememe_map.json"), encoding="utf-8") as f:
    SEMEME_MAP = json.load(f)["sememes"]

def extract_ctx(sent, ts, te):
    ws = []; oov = []; cp = 0
    for w, pos in pseg.cut(sent):
        we = cp + len(w)
        if cp < te and we > ts: cp = we; continue
        if all(c in "，。、！？：；""''（）【】《》——…·,.:;!?()[]{}" for c in w) or w.isdigit() or w in STOP: cp = we; continue
        if w in kv:
            ws.append(w)
        else:
            oov.append(w)
        cp = we
    return ws, oov

def get_sp(s):
    ws = []; cp = 0
    for w, pos in pseg.cut(s):
        ws.append((w, pos, cp, cp+len(w))); cp += len(w)
    return ws

def layer3(sent, ts, te, cands):
    covering = [(w,p) for w,p,cs,ce in get_sp(sent) if cs <= ts and ce >= te and len(w) > (te-ts)]
    if not covering: return None
    cp_w = max(covering, key=lambda x: len(x[0]))[0]
    import OpenHowNet; hn = OpenHowNet.HowNetDict()
    data = hn.get_sememes_by_word(cp_w); cps = set()
    if data:
        for entry in data:
            for s in entry.get("sememes", []): cps.add(str(s).split("|")[0].strip())
    if not cps: return None
    sc = {}
    for sid in cands:
        cs2 = set(SEMEME_MAP.get(sid, []))
        if not cs2: sc[sid] = 0.0
        else: sc[sid] = len(cps & cs2) / max(len(cps | cs2), 1)
    ss = sorted(sc.items(), key=lambda x: -x[1])
    if ss[0][1] > 0 and (ss[0][1] - (ss[1][1] if len(ss) > 1 else 0)) >= 0.10: return ss[0][0]
    return None

def layer4(sent, ts, te, cands):
    tp = next((p for w,p,cs,ce in get_sp(sent) if cs == ts and ce == te), None)
    if not tp: return None
    fl = [s for s in cands if any(tp.startswith(p) for p in C["sense_pos_map"].get(s, ["n","v","a"]))]
    return fl[0] if len(fl) == 1 else None

def layer5(sent, ts, te, cands):
    toks, oov = extract_ctx(sent, ts, te)
    if not toks: return None, {}, None, None, oov
    cv = np.mean([kv[t] for t in toks], axis=0)
    cv = cv / (np.linalg.norm(cv) + 1e-8)
    sc = {s: float(np.dot(cv, protos[s])) if s in protos else 0. for s in cands}
    ss = sorted(sc.items(), key=lambda x: -x[1])
    sel = ss[0][0] if ss[0][1] >= MS and (ss[0][1] - (ss[1][1] if len(ss) > 1 else 0)) >= MM else None
    return ss[0][0], sc, ss[0][0], sel, oov

def fallback(w): return MFS.get(w) or HF.get(w)

def cascade_a(sent, ts, te, cands):
    r = layer3(sent, ts, te, cands)
    if r: return r, "L3"
    r = layer4(sent, ts, te, cands)
    if r: return r, "L4"
    return fallback(sent[ts:te]), "MFS"

def cascade_b(sent, ts, te, cands):
    r = layer3(sent, ts, te, cands)
    if r: return r, "L3"
    r = layer4(sent, ts, te, cands)
    if r: return r, "L4"
    _, _, _, sel, _ = layer5(sent, ts, te, cands)
    if sel: return sel, "L5"
    return fallback(sent[ts:te]), "MFS"

def predict_one(item):
    w,st = item["target_surface"], item["sentence"]; ts,te = item["target_start"], item["target_end"]; cs = item["candidate_sense_ids"]
    ind = {"MFS": MFS.get(w), "L2": HF.get(w), "L3": None, "L4": None, "L5_forced": None, "L5_selective": None}
    ind["L3"] = layer3(st,ts,te,cs); ind["L4"] = layer4(st,ts,te,cs)
    b,sc,f5,s5,oov = layer5(st,ts,te,cs); ind["L5_forced"] = f5; ind["L5_selective"] = s5
    toks, _ = extract_ctx(st,ts,te)
    comp = next((w for w,p,cs2,ce in get_sp(st) if cs2 <= ts and ce >= te and len(w) > (te-ts)), None)
    tp = next((p for w,p,cs2,ce in get_sp(st) if cs2 == ts and ce == te), None)
    db = {"target_span_valid": st[ts:te] == w, "covering_compound": comp, "target_pos": tp,
          "context_tokens": toks, "context_oov_tokens": oov, "L5_computable": len(toks) > 0,
          "L5_scores": sc, "L5_best_margin": max(sc.values())-min(sc.values()) if sc else 0}
    ca,_ = cascade_a(st,ts,te,cs); cb,lb = cascade_b(st,ts,te,cs)
    db["cascade_A"] = ca; db["cascade_B"] = cb
    return {"id": item["id"], "prediction": cb, "resolved_by": lb, "independent": ind, "debug": db}

def validate_input(data):
    import p0_r5_validate_input as vi
    vi.validate(data, OLD_SENTENCES)
    print("  输入验证: ✅", file=sys.stderr)

def validate_output(inp, pred):
    import p0_r5_validate_output as vo
    vo.validate(inp, pred)
    print("  输出验证: ✅", file=sys.stderr)

def main():
    if len(sys.argv) != 2:
        print("用法: python p0_r5_predict.py <input.json>", file=sys.stderr); sys.exit(1)
    ip = sys.argv[1]
    with open(ip, encoding="utf-8") as f: items = json.load(f)
    print(f"加载: {len(items)} 条", file=sys.stderr)
    validate_input(items)
    os.makedirs(OUT_RUN, exist_ok=True)
    results = [predict_one(it) for it in items]
    base = os.path.splitext(os.path.basename(ip))[0]
    op = os.path.join(OUT_RUN, f"{base}_predictions.json")
    with open(op, "w", encoding="utf-8") as f: json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"预测 → {op}", file=sys.stderr)
    validate_output(items, results)
    with open(op, "rb") as f: h = hashlib.sha256(f.read()).hexdigest()
    with open(op + ".sha256", "w") as f: f.write(h + "\n")
    print(f"SHA-256: {h}", file=sys.stderr)
    print(f"P0-R5 预测完成: {len(results)} 条", file=sys.stderr)

if __name__ == "__main__":
    main()
