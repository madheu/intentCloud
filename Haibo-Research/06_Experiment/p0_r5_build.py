"""P0-R5 构建 — 自包含冻结包"""
import json, os, sys, numpy as np
sys.path = [p for p in sys.path if "Diviner" not in p]
from collections import Counter
import jieba, jieba.posseg as pseg
from gensim.models import KeyedVectors
import OpenHowNet

BASE = os.path.dirname(os.path.abspath(__file__))
FROZEN = os.path.join(BASE, "output", "p0_r5_frozen")
OUT_RUN = os.path.join(BASE, "output", "p0_r5_run")
os.makedirs(FROZEN, exist_ok=True)
os.makedirs(OUT_RUN, exist_ok=True)

KV_PATH = r"E:\intentCloud\models\text2vec-word2vec-tencent-chinese\light_Tencent_AILab_ChineseEmbedding.bin"
LLM_DATA = os.path.join(BASE, "p0_b2_llm_contexts.json")

kv = KeyedVectors.load_word2vec_format(KV_PATH, binary=True)
hownet = OpenHowNet.HowNetDict()
assert hownet.get_sememes_by_word("苹果")

# ── 1. 开发数据 — 保证所有金标合法 ──
SENSE_IDS = {"苹果":["apple.company","apple.fruit"],"小米":["xiaomi.company","xiaomi.grain"],
             "杜鹃":["cuckoo.flower","cuckoo.bird"],"花":["hua.plant","hua.spend"],
             "光":["guang.physical","guang.figurative"],"行":["xing.approval","xing.row","xing.industry"],
             "口":["kou.body","kou.space"],"头":["tou.body","tou.leader","tou.beginning"],
             "结":["jie.concrete","jie.abstract"]}

LABEL_38 = {"苹果":{"A":"apple.company","B":"apple.fruit"},"光":{"A":"guang.physical","B":"guang.figurative"},
            "花":{"A":"hua.plant","B":"hua.spend"},"行":{"A":"xing.approval","B":"xing.row","C":"xing.industry"},
            "口":{"A":"kou.body","B":"kou.space"}}
LABEL_60 = {"苹果":{"科技公司":"apple.company","水果":"apple.fruit"},"小米":{"科技":"xiaomi.company","谷物":"xiaomi.grain"},
            "杜鹃":{"花":"cuckoo.flower","鸟":"cuckoo.bird"},"花":{"植物":"hua.plant","消费":"hua.spend"},
            "光":{"物理光线":"guang.physical","修辞评价":"guang.figurative"},"行":{"评价":"xing.approval","排列":"xing.row"},
            "口":{"人体":"kou.body","空间":"kou.space"},"头":{"人体":"tou.body","领导":"tou.leader","起始":"tou.beginning"},
            "结":{"具体":"jie.concrete","抽象":"jie.abstract"}}

dev_items = []; idx = 0; old_norm = set()
for path, label_src in [("P0-R_gold_labels.json", LABEL_38), ("P0-R_blind_gold_labels.json", LABEL_60)]:
    with open(os.path.join(BASE, path), encoding="utf-8") as f: data = json.load(f)
    for word, entries in data.items():
        if word not in SENSE_IDS: continue
        # P0-R_gold_labels: {sentence: label}; P0-R_blind_gold_labels: list of {sentence, label, sense_name}
        if isinstance(entries, dict):
            for sent, lbl in entries.items():
                gold = label_src.get(word, {}).get(lbl)
                if gold is None or gold not in SENSE_IDS[word]:
                    raise ValueError(f"{path}: {word}/{lbl} → {gold} 非法")
                ts = sent.find(word)
                assert ts != -1, f"找不到'{word}'"
                idx += 1; te = ts + len(word)
                dev_items.append({"id":f"dev-{idx:04d}","sentence":sent,"target_surface":word,
                                  "target_start":ts,"target_end":te,"candidate_sense_ids":SENSE_IDS[word],"_gold_sense_id":gold})
                old_norm.add(sent.replace(" ","").replace("　",""))
        else:
            for e in entries:
                sent = e["sentence"]
                lbl = e.get("sense_name", e.get("label", ""))
                gold = label_src.get(word, {}).get(lbl)
                if gold is None or gold not in SENSE_IDS[word]:
                    raise ValueError(f"{path}: {word}/{lbl} → {gold} 非法")
                ts = sent.find(word)
                assert ts != -1, f"找不到'{word}'"
                idx += 1; te = ts + len(word)
                dev_items.append({"id":f"dev-{idx:04d}","sentence":sent,"target_surface":word,
                                  "target_start":ts,"target_end":te,"candidate_sense_ids":SENSE_IDS[word],"_gold_sense_id":gold})
                old_norm.add(sent.replace(" ","").replace("　",""))
            # 循环结束
assert len(dev_items) == 58, f"开发集应为58条, 实际{len(dev_items)}"
print(f"开发数据: {len(dev_items)} 条, 0 非法")

# 冻结旧句集
with open(os.path.join(FROZEN, "p0_r5_old_sentences.json"), "w", encoding="utf-8") as f:
    json.dump(list(old_norm), f, ensure_ascii=False)
print(f"旧句去重集: {len(old_norm)} 句")

# ── 2. MFS + HOWNET_FIRST ──
wl = {}
for it in dev_items:
    wl.setdefault(it["target_surface"], []).append(it["_gold_sense_id"])
mfs_map = {w: Counter(ls).most_common(1)[0][0] for w, ls in wl.items()}

HOWNET_FIRST = {"苹果":"apple.company","小米":"xiaomi.grain","杜鹃":"cuckoo.flower",
                "花":"hua.plant","光":"guang.physical","行":"xing.approval",
                "口":"kou.body","头":"tou.body","结":"jie.concrete"}

# ── 3. 义原映射 — 带 HowNet sense No + 英文义项 ──
# 手工核对: 用 HowNet get_sense 获取 sense list, 记录 No + 英文
HOWNET_SENSE_MAP = {
    "apple.company": {"query":"苹果","sense_no":"244398","gloss":"IPHONE|苹果"},
    "apple.fruit": {"query":"苹果","sense_no":"244397","gloss":"apple|苹果"},
    "xiaomi.company": {"query":"小米","sense_no":"135733","gloss":"cell phone|小米"},
    "xiaomi.grain": {"query":"小米","sense_no":"242635","gloss":"millet|小米"},
    "cuckoo.flower": {"query":"杜鹃","sense_no":"61599","gloss":"azalea|杜鹃"},
    "cuckoo.bird": {"query":"杜鹃","sense_no":"61593","gloss":"cuckoo|杜鹃"},
    "hua.plant": {"query":"花","sense_no":"242326","gloss":"flower|花"},
    "hua.spend": {"query":"花","sense_no":"242332","gloss":"spend|花"},
    "guang.physical": {"query":"光","sense_no":"48873","gloss":"ray|光"},
    "guang.figurative": {"query":"光","sense_no":"48867","gloss":"glory|光"},
    "xing.approval": {"query":"行","sense_no":"250604","gloss":"competent|行"},
    "xing.row": {"query":"行","sense_no":"250607","gloss":"line|行"},
    "xing.industry": {"query":"行","sense_no":"250592","gloss":"business firm|行"},
    "kou.body": {"query":"口","sense_no":"73573","gloss":"mouth|口"},
    "kou.space": {"query":"口","sense_no":"73572","gloss":"entrance|口"},
    "tou.body": {"query":"头","sense_no":"206669","gloss":"head|头"},
    "tou.leader": {"query":"头","sense_no":"206672","gloss":"chief|头"},
    "tou.beginning": {"query":"头","sense_no":"206677","gloss":"beginning|头"},
    "jie.concrete": {"query":"结","sense_no":"42686","gloss":"knot|结"},
    "jie.abstract": {"query":"结","sense_no":"42689","gloss":"conclusion|结"},
}

sememe_map = {}
for sid, cfg in HOWNET_SENSE_MAP.items():
    data = hownet.get_sememes_by_word(cfg["query"])
    sems = set()
    if data:
        # 按 sense_no 找到对应义项
        for entry in data:
            s_str = str(entry.get("sense",""))
            if cfg["sense_no"] in s_str:
                for s in entry.get("sememes", []):
                    sems.add(str(s).split("|")[0].strip())
    if not sems:
        print(f"  ⚠️ {sid}: 义原为空 (sense_no={cfg['sense_no']})")
    sememe_map[sid] = list(sems)

with open(os.path.join(FROZEN, "p0_r5_sememe_map.json"), "w", encoding="utf-8") as f:
    json.dump({"sense_map": HOWNET_SENSE_MAP, "sememes": sememe_map}, f, ensure_ascii=False, indent=2)
print(f"义原映射: {sum(1 for v in sememe_map.values() if v)}/20 非空")

# ── 4. 原型 ──
with open(LLM_DATA, encoding="utf-8") as f: llm_data = json.load(f)
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

def extract_ctx(sent, ts, te, stop_words):
    words = []; cp = 0
    for w, pos in pseg.cut(sent):
        we = cp + len(w)
        if cp < te and we > ts: cp = we; continue
        if all(c in "，。、！？：；""''（）【】《》——…·,.:;!?()[]{}" for c in w) or w.isdigit() or w in stop_words: cp = we; continue
        if w in kv: words.append(w)
        cp = we
    return words

protos = {}
for sid, sts in mapped.items():
    svs = []
    for st in sts:
        for t, cands in SENSE_IDS.items():
            if sid in cands:
                ts = st.find(t)
                if ts == -1: continue
                toks = extract_ctx(st, ts, ts+len(t), set())
                if not toks: continue
                v = np.mean([kv[t] for t in toks], axis=0)
                svs.append(v / (np.linalg.norm(v)+1e-8)); break
    if svs:
        p = np.mean(svs, axis=0); protos[sid] = p / (np.linalg.norm(p)+1e-8)
np.savez(os.path.join(FROZEN, "p0_r5_prototypes.npz"), **{k:v for k,v in protos.items()})
print(f"原型: {len(protos)}/20")

# ── 5. 阈值搜索 (macro accuracy, 并列规则) ──
CONFIG = {"sense_ids":SENSE_IDS,"sense_pos_map":{"apple.company":["n","nr","nt","nz"],"apple.fruit":["n"],
  "xiaomi.company":["n","nr","nt","nz"],"xiaomi.grain":["n"],"cuckoo.flower":["n"],"cuckoo.bird":["n"],
  "hua.plant":["n"],"hua.spend":["v"],"guang.physical":["n"],"guang.figurative":["v","a","d"],
  "xing.approval":["v","a"],"xing.row":["n","q"],"xing.industry":["n","nt"],"kou.body":["n"],"kou.space":["n"],
  "tou.body":["n"],"tou.leader":["n","nr"],"tou.beginning":["n"],"jie.concrete":["n"],"jie.abstract":["n","v","a"]},
  "hownet_first":HOWNET_FIRST,"mfs_map":mfs_map,"stop_words":["的","了","是","在","和","有","不","就","也","都",
  "而","且","或","与","比","这款","这个","那个","一个","没有","可以","会","要","能","让","上","下","来","去",
  "说","看","做","用","知道","怎么","什么","这","那","它","他","她","我","你","们","自己","人","为","被","把",
  "从","对","到","得","很","还","又","再","才","只","但","如果","因为","所以","虽然","然后"],
  "version":"p0-r5-001"}

def layer5(sent, ts, te, cands, protos, ms, mm):
    toks = extract_ctx(sent, ts, te, set(CONFIG["stop_words"]))
    if not toks: return None,{},None,None
    cv = np.mean([kv[t] for t in toks], axis=0)
    cv = cv / (np.linalg.norm(cv)+1e-8)
    sc = {s: float(np.dot(cv, protos[s])) if s in protos else 0. for s in cands}
    ss = sorted(sc.items(), key=lambda x: -x[1])
    sel = ss[0][0] if ss[0][1] >= ms and (ss[0][1]-(ss[1][1] if len(ss)>1 else 0)) >= mm else None
    return ss[0][0], sc, ss[0][0], sel

best_macro = -1; best_cfg = None
for ms in [0.0, 0.1, 0.2, 0.3]:
    for mm in [0.0, 0.02, 0.05, 0.10]:
        wr = {w:{"c":0,"t":0} for w in SENSE_IDS}
        for it in dev_items:
            w=it["target_surface"]; _,_,_,sel=layer5(it["sentence"],it["target_start"],it["target_end"],it["candidate_sense_ids"],protos,ms,mm)
            if sel: wr[w]["t"]+=1
            if sel==it["_gold_sense_id"]: wr[w]["c"]+=1
        l5_all=sum(v["t"] for v in wr.values()); cov=l5_all/len(dev_items)*100
        if cov<20: continue
        wa=[v["c"]/max(v["t"],1) for w,v in wr.items() if v["t"]>0]
        macro=np.mean(wa) if wa else 0
        # 并列规则: 先 macro, 再 margin, 再 similarity
        better=False
        if macro>best_macro or abs(macro-best_macro)<1e-6 and mm>(best_cfg[1] if best_cfg else -1) or \
           abs(macro-best_macro)<1e-6 and abs(mm-(best_cfg[1] if best_cfg else -1))<1e-6 and ms>(best_cfg[0] if best_cfg else -1):
            better=True
        if better or best_cfg is None:
            best_macro=macro; best_cfg=(ms,mm)
        print(f"  ms={ms:.1f} mm={mm:.2f}: macro={macro:.3f} cov={cov:.0f}%{' ←最佳' if better else ''}")

CONFIG["l5_min_similarity"]=best_cfg[0]; CONFIG["l5_min_margin"]=best_cfg[1]
print(f"阈值: ms={best_cfg[0]} mm={best_cfg[1]} macro={best_macro:.3f}")

with open(os.path.join(FROZEN, "p0_r5_config.json"), "w", encoding="utf-8") as f:
    json.dump(CONFIG, f, ensure_ascii=False, indent=2)

# ── 6. 开发诊断报告 ──
def cascade_b(sent,ts,te,cands):
    # L3
    covering=[(w,pos,cs,ce) for w,pos,cs,ce in [(lambda w,p,cs,ce:(w,p,cs,ce))(*x) for x in []] if False]
    sps=[(w,pos,cs,ce) for w,pos,cs,ce in [(lambda:s.__setitem__(j,0) or s)({}) for _ in [0]] and []]
    # simplified

report={"targets":{},"summary":{}}
for w,cands in SENSE_IDS.items():
    its=[it for it in dev_items if it["target_surface"]==w]
    n=len(its); golds=[it["_gold_sense_id"] for it in its]
    mfs_correct=sum(1 for it in its if mfs_map.get(w)==it["_gold_sense_id"])
    report["targets"][w]={"samples":n,"gold_dist":dict(Counter(golds)),"mfs_accuracy":mfs_correct/max(n,1)}
    print(f"  {w}: {n}条 MFS_acc={mfs_correct/n:.2%}")
report["summary"]["total_samples"]=len(dev_items)
report["summary"]["overall_mfs_accuracy"]=sum(v["mfs_accuracy"]*v["samples"] for v in report["targets"].values())/len(dev_items)
with open(os.path.join(FROZEN, "p0_r5_dev_report.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(f"诊断报告已保存")
print(f"\n自包含冻结包就绪: {FROZEN}")
