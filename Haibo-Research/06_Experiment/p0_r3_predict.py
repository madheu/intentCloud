"""P0-R3 预测管线 — 完整版含 B3 入口"""
import json, os, sys, numpy as np
sys.path = [p for p in sys.path if 'Diviner' not in p]
from collections import Counter
import jieba, jieba.posseg as pseg
from gensim.models import KeyedVectors
import OpenHowNet

hownet = OpenHowNet.HowNetDict()
assert hownet.get_sememes_by_word("手机") and any(s.get("sememes") for s in hownet.get_sememes_by_word("手机")), "OpenHowNet 不可用"

BASE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE, "p0_r3_config.json")
KV_PATH = r"E:\intentCloud\models\text2vec-word2vec-tencent-chinese\light_Tencent_AILab_ChineseEmbedding.bin"
LLM_DATA = os.path.join(BASE, "p0_b2_llm_contexts.json")
with open(CONFIG_PATH) as f: CONFIG = json.load(f)
SENSE_IDS = CONFIG["sense_ids"]; STOP_WORDS = set(CONFIG["stop_words"])

kv = KeyedVectors.load_word2vec_format(KV_PATH, binary=True)

# ── 冻结义原映射 ──
FROZEN_SEMEMES = {}
for surface, cands in SENSE_IDS.items():
    data = hownet.get_sememes_by_word(surface)
    sems = set()
    if data:
        for entry in data:
            for s in entry.get("sememes", []):
                sems.add(str(s).split("|")[0].strip())
    for sid in cands:
        FROZEN_SEMEMES[sid] = sems

# ── HOWNET_FIRST ──
HOWNET_FIRST = {"苹果":"apple.company","小米":"xiaomi.grain","杜鹃":"cuckoo.flower",
                "花":"hua.plant","光":"guang.physical","行":"xing.approval",
                "口":"kou.body","头":"tou.body","结":"jie.concrete"}

def extract_context_tokens(sentence, ts, te):
    ws = []; cp = 0
    for w, pos in pseg.cut(sentence):
        we = cp+len(w)
        if cp < te and we > ts: cp = we; continue
        if all(c in "，。、！？：；""''（）【】《》——…·,.:;!?()[]{}" for c in w) or w.isdigit() or w in STOP_WORDS:
            cp = we; continue
        if w in kv: ws.append(w)
        cp = we
    return ws

def get_spans(s):
    ws = []; cp = 0
    for w, pos in pseg.cut(s):
        ws.append((w, pos, cp, cp+len(w))); cp += len(w)
    return ws
def compute_mfs(d):
    wl={}
    for it in d: wl.setdefault(it["target_surface"],[]).append(it["_gold_sense_id"])
    return {w: Counter(ls).most_common(1)[0][0] for w,ls in wl.items()}

def layer1_mfs(w, m): return m.get(w)
def layer2_hownet(w): return HOWNET_FIRST.get(w)

def layer3(sent, ts, te, cands):
    covering = [(w,pos) for w,pos,cs,ce in get_spans(sent) if cs<=ts and ce>=te and len(w)>(te-ts)]
    if not covering: return None
    cp = max(covering, key=lambda x: len(x[0]))[0]
    cs = FROZEN_SEMEMES.get(cp) or set()
    if not cs: return None
    sc = {sid: len(cs & FROZEN_SEMEMES.get(sid,set()))/max(len(cs|FROZEN_SEMEMES.get(sid,set())),1) for sid in cands}
    ss = sorted(sc.items(), key=lambda x:-x[1])
    return ss[0][0] if ss[0][1] > 0 and (ss[0][1]-(ss[1][1] if len(ss)>1 else 0)) >= 0.10 else None

def layer4(sent, ts, te, cands):
    tp = next((pos for w,pos,cs,ce in get_spans(sent) if cs==ts and ce==te), None)
    if not tp: return None
    fl = [s for s in cands if any(tp.startswith(p) for p in CONFIG["sense_pos_map"].get(s,["n","v","a"]))]
    return fl[0] if len(fl)==1 else None

def build_prototypes():
    with open(LLM_DATA) as f: d=json.load(f)
    mp={"苹果/科技公司":"apple.company","苹果/水果":"apple.fruit","小米/科技":"xiaomi.company","小米/谷物":"xiaomi.grain",
        "杜鹃/花":"cuckoo.flower","杜鹃/鸟":"cuckoo.bird","花/植物":"hua.plant","花/消费":"hua.spend",
        "光/物理光线":"guang.physical","光/修辞评价":"guang.figurative","行/评价":"xing.approval","行/排列":"xing.row",
        "行/行业":"xing.industry","口/人体":"kou.body","口/空间":"kou.space","头/人体":"tou.body","头/领导":"tou.leader",
        "头/起始":"tou.beginning","结/具体":"jie.concrete","结/抽象":"jie.abstract"}
    m={}; pr={}
    for w,ss in d.items():
        for sn,sts in ss.items():
            sid=mp.get(f"{w}/{sn}")
            if sid: m[sid]=sts
    for sid,sts in m.items():
        svs=[]
        for st in sts:
            for t,cands in SENSE_IDS.items():
                if sid in cands:
                    ts=st.find(t)
                    if ts==-1: continue
                    toks=extract_context_tokens(st,ts,ts+len(t))
                    if not toks: continue
                    v=np.mean([kv[t] for t in toks],axis=0)
                    svs.append(v/(np.linalg.norm(v)+1e-8)); break
        if svs:
            p=np.mean(svs,axis=0); pr[sid]=p/(np.linalg.norm(p)+1e-8)
    return pr

def layer5(sent,ts,te,cands,pr):
    toks=extract_context_tokens(sent,ts,te)
    if not toks: return None,{},None,None
    cv=np.mean([kv[t] for t in toks],axis=0); cv=cv/(np.linalg.norm(cv)+1e-8)
    sc={s:float(np.dot(cv,pr[s])) if s in pr else 0. for s in cands}
    ss=sorted(sc.items(),key=lambda x:-x[1])
    mg=ss[0][1]-(ss[1][1] if len(ss)>1 else 0)
    ms=CONFIG.get("l5_min_similarity",0.); mm=CONFIG.get("l5_min_margin",0.)
    sel=ss[0][0] if (ss[0][1]>=ms and mg>=mm) else None
    return ss[0][0],sc,ss[0][0],sel

def _mfs_or_hownet(w,m):
    r=layer1_mfs(w,m)
    return r if r else layer2_hownet(w)

def cascade_a(sent,ts,te,cands,m):
    l3=layer3(sent,ts,te,cands)
    if l3: return l3,"L3"
    l4=layer4(sent,ts,te,cands)
    if l4: return l4,"L4"
    return _mfs_or_hownet(sent[ts:te],m),"MFS"

def cascade_b(sent,ts,te,cands,m,pr):
    l3=layer3(sent,ts,te,cands)
    if l3: return l3,"L3"
    l4=layer4(sent,ts,te,cands)
    if l4: return l4,"L4"
    _,_,_,sel=layer5(sent,ts,te,cands,pr)
    if sel: return sel,"L5"
    return _mfs_or_hownet(sent[ts:te],m),"MFS"

# ── B3: 完整单条预测 ──
def predict_entry(item, mfs_map, protos):
    w,sent=item["target_surface"],item["sentence"]; ts,te=item["target_start"],item["target_end"]; cands=item["candidate_sense_ids"]
    ind={"MFS":layer1_mfs(w,mfs_map),"L2":layer2_hownet(w),"L3":None,"L4":None,"L5_forced":None,"L5_selective":None}
    ind["L3"]=layer3(sent,ts,te,cands)
    ind["L4"]=layer4(sent,ts,te,cands)
    _,sc,f5,s5=layer5(sent,ts,te,cands,protos); ind["L5_forced"]=f5; ind["L5_selective"]=s5
    toks=extract_context_tokens(sent,ts,te)
    sps=get_spans(sent)
    comp=next((w for w,p,cs,ce in sps if cs<=ts and ce>=te and len(w)>(te-ts)),None)
    tp=next((pos for w,pos,cs,ce in sps if cs==ts and ce==te),None)
    db={"target_span_valid":sent[ts:te]==w,"covering_compound":comp,"target_pos":tp,
        "context_tokens":toks,"context_oov_tokens":[],"L5_computable":len(toks)>0,
        "L5_scores":sc,"L5_best_margin":max(sc.values())-min(sc.values()) if sc else 0}
    ca,la=cascade_a(sent,ts,te,cands,mfs_map); cb,lb=cascade_b(sent,ts,te,cands,mfs_map,protos)
    db["cascade_A"]=ca; db["cascade_B"]=cb
    return {"id":item["id"],"prediction":cb,"resolved_by":lb,"independent":ind,"debug":db}
