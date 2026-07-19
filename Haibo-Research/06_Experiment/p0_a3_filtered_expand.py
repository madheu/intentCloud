"""P0-A-3: 三种跨类重叠过滤方法对比
=====================================
在自动扩展后加过滤，去除跨类共享的近邻词。
"""
import sys, math
sys.path = [p for p in sys.path if 'Diviner' not in p]
import numpy as np
from gensim.models import KeyedVectors
from collections import defaultdict

KV_PATH = r"E:\intentCloud\models\text2vec-word2vec-tencent-chinese\light_Tencent_AILab_ChineseEmbedding.bin"
print("加载腾讯词向量...")
kv = KeyedVectors.load_word2vec_format(KV_PATH, binary=True)
print(f"  共 {len(kv.index_to_key)} 词，{kv.vector_size} 维\n")

# ── 配置 ──
TOP_K = 20
SEEDS = {
    "科技_消费电子": ["手机", "芯片", "系统", "屏幕", "应用", "发布", "商店", "生态"],
    "科技_产品": ["产品", "品牌", "营收", "市场", "新款", "版本", "销量"],
    "科技_研发": ["芯片", "研发", "专利", "自研", "性能", "升级", "架构"],
    "水果_种植": ["树", "种植", "采摘", "开花", "修剪", "丰收", "成熟"],
    "水果_品质": ["甜", "营养", "含糖量", "口感", "新鲜", "果汁", "果肉"],
    "水果_品种": ["品种", "特产", "产地", "上市", "季节性"],
    "自然_光线": ["太阳", "明亮", "照射", "反射", "折射", "阳光", "光芒", "耀眼", "紫外线"],
    "自然_植物": ["叶子", "根", "种子", "开花", "生长", "盆栽", "绿植", "园林"],
    "生活_消费": ["钱", "消费", "付款", "购买", "花费", "折扣", "预算", "便宜"],
    "生活_日常": ["吃饭", "睡觉", "上班", "回家", "出门", "做饭", "洗", "休息"],
    "生活_建筑": ["门", "窗户", "墙", "房间", "入口", "出口", "走廊", "楼梯"],
    "语言_修辞": ["比喻", "象征", "夸张", "拟人", "排比", "修辞", "形象"],
    "语言_评价": ["好", "坏", "不错", "优秀", "差", "满意", "一般"],
    "语言_抽象": ["概念", "定义", "逻辑", "因果", "关系", "本质", "结构"],
}
DOMAINS = {
    "科技": ["科技_消费电子", "科技_产品", "科技_研发"],
    "水果": ["水果_种植", "水果_品质", "水果_品种"],
    "自然": ["自然_光线", "自然_植物"],
    "生活": ["生活_消费", "生活_日常", "生活_建筑"],
    "语言": ["语言_修辞", "语言_评价", "语言_抽象"],
}
AMBIGUOUS = {"苹果":["科技","水果"],"纸":["科技","生活"],"光":["自然","语言"],"花":["自然","生活"],"行":["生活","语言"],"口":["生活","生活"]}
NEUTRAL = {"比","这款","这个","那个","的","了","是","很","也","而且","和","在","有","不","就","把","被","从"}

# ── 基本扩展 ──
def expand_base():
    raw = {}
    for concept, seeds in SEEDS.items():
        ws = set(seeds)
        for s in seeds:
            if s in kv:
                try:
                    ws.update(w for w, _ in kv.most_similar(s, topn=TOP_K))
                except:
                    pass
        raw[concept] = ws
    return raw

def word_concept_count(raw):
    """统计每个词出现在几个概念中"""
    wc = defaultdict(set)
    for c, ws in raw.items():
        for w in ws:
            wc[w].add(c)
    return {w: len(cs) for w, cs in wc.items()}, wc

# ── 过滤方法 ──
def filter_hard(raw, threshold):
    """方法A: 硬阈值"""
    cnt, _ = word_concept_count(raw)
    filtered = {}
    removed = 0
    for c, ws in raw.items():
        kept = {w for w in ws if cnt.get(w, 0) < threshold}
        removed += len(ws) - len(kept)
        filtered[c] = kept
    return filtered, removed

def filter_tfidf(raw, filter_threshold):
    """方法B: TF-IDF风格过滤"""
    n = len(raw)
    _, wc = word_concept_count(raw)
    filtered = {}
    removed = 0
    for c, ws in raw.items():
        kept = set()
        for w in ws:
            df = len(wc[w])
            idf = math.log(n / max(df, 1))
            if idf >= filter_threshold:
                kept.add(w)
            else:
                removed += 1
        filtered[c] = kept
    return filtered, removed

def filter_best_friend(raw, top_k=3, max_foreign=1):
    """方法C: best-friend过滤"""
    # 先找到每个概念的种子词在词向量空间中的近邻
    concept_centroids = {}
    for c, seeds in SEEDS.items():
        vecs = []
        for s in seeds:
            if s in kv:
                vecs.append(kv[s])
        if vecs:
            concept_centroids[c] = np.mean(vecs, axis=0)
    
    filtered = {}
    removed = 0
    for c, ws in raw.items():
        kept = set()
        for w in ws:
            if w not in kv:
                kept.add(w)
                continue
            # 找这个词的 top_k 最近邻
            try:
                nn = [nn_w for nn_w, _ in kv.most_similar(w, topn=top_k)]
            except:
                kept.add(w)
                continue
            # 统计近邻中属于其他概念的数量
            foreign = 0
            for nn_w in nn:
                for other_c in raw:
                    if other_c != c and nn_w in raw[other_c]:
                        foreign += 1
                        break
            if foreign <= max_foreign:
                kept.add(w)
            else:
                removed += 1
        filtered[c] = kept
    return filtered, removed

# ── 重叠率 ──
def overlap_rate(filtered):
    total = 0
    overlap = 0
    concepts = list(filtered.keys())
    for i in range(len(concepts)):
        for j in range(i+1, len(concepts)):
            common = filtered[concepts[i]] & filtered[concepts[j]]
            if common:
                overlap += len(common)
                total += min(len(filtered[concepts[i]]), len(filtered[concepts[j]]))
    return overlap / max(total, 1)

# ── 分词 + 推理 ──
def make_resolver(WORD_TO_SUB):
    def tokenize(sentence):
        found = set()
        for w in sorted(WORD_TO_SUB.keys(), key=len, reverse=True):
            if w in sentence:
                found.add(w)
        for aw in AMBIGUOUS:
            if aw in sentence:
                found.add(aw)
        return found

    def resolve(sentence, ambiguous_word):
        words = tokenize(sentence)
        ds = {d: 0 for d in DOMAINS}
        for w in words:
            if w in NEUTRAL: continue
            if w in AMBIGUOUS:
                for d in AMBIGUOUS[w]:
                    if d in ds: ds[d] += 1
            elif w in WORD_TO_SUB:
                for sc in WORD_TO_SUB[w]:
                    for d, subs in DOMAINS.items():
                        if sc in subs: ds[d] += 1
        sd = sorted(ds.items(), key=lambda x: -x[1])
        top, top_s = sd[0]
        sec_s = sd[1][1] if len(sd) > 1 else 0
        if sec_s > 0 and top_s >= sec_s * 1.5: return top
        if sec_s == 0 and top_s > 0: return top
        # layer2
        tied = [d for d, s in ds.items() if s > 0]
        words2 = tokenize(sentence)
        ss = {}
        for d in tied:
            for sub in DOMAINS[d]: ss[sub] = 0
        for w in words2:
            if w in AMBIGUOUS:
                for d in AMBIGUOUS[w]:
                    if d in tied:
                        for sub in DOMAINS[d]: ss[sub] += 1
            elif w in WORD_TO_SUB:
                for sc in WORD_TO_SUB[w]:
                    if sc in ss: ss[sc] += 1
        if not ss: return "不确定"
        ss_sorted = sorted(ss.items(), key=lambda x: -x[1])
        top_sub, top_sub_s = ss_sorted[0]
        sec_sub_s = ss_sorted[1][1] if len(ss_sorted) > 1 else 0
        if sec_sub_s > 0 and top_sub_s >= sec_sub_s * 1.5:
            for d, subs in DOMAINS.items():
                if top_sub in subs: return d
            return top_sub
        if sec_sub_s == 0 and top_sub_s > 0:
            for d, subs in DOMAINS.items():
                if top_sub in subs: return d
            return top_sub
        return "不确定"
    return resolve

# ── 测试 ──
TEST_CASES = {
    "苹果": [("这个苹果比上一代便宜了五百块","科技"),("苹果发布了新款手机","科技"),("今年苹果的芯片性能提升很大","科技"),("苹果的屏幕显示效果很好","科技"),("苹果应用商店的规则更新了","科技"),("这个苹果比上一代更甜","水果"),("苹果的采摘季节到了","水果"),("今年的苹果果肉很甜","水果"),("苹果正在开花","水果"),("苹果的含糖量很高","水果"),("这个苹果很好吃，而且设计也很漂亮","不确定")],
    "纸": [("打印机没纸了","科技"),("这张纸的质量很好","科技"),("把协议落实到纸面上","科技"),("论文的摘要写在一张纸上","科技"),("用纸包住花束","不确定"),("纸抽用完了","不确定")],
    "光": [("太阳的光很强烈","自然"),("光在水面发生折射","自然"),("钱都花光了","语言"),("光说不做","语言"),("他为国争光","语言"),("光的速度是每秒钟三十万公里","自然")],
    "花": [("花园里的花开了","自然"),("这盆花需要浇水","自然"),("花了很多钱","生活"),("花钱如流水","生活"),("花时间学习","生活"),("花开得很鲜艳","自然")],
    "行": [("往前走三行","生活"),("行，就这样吧","语言"),("这个办法行不通","语言"),("他在银行工作","科技"),("一行代码","科技")],
    "口": [("他张开了口","生活"),("入口在左边","生活"),("三口之家","生活"),("伤口很深","不确定")],
}

def run_test(resolve, label=""):
    results = {}
    total_correct = 0; total_all = 0
    for word, cases in TEST_CASES.items():
        correct = sum(1 for s, e in cases if resolve(s, word) == e)
        results[word] = (correct, len(cases), correct/len(cases)*100)
        total_correct += correct; total_all += len(cases)
    results["total"] = (total_correct, total_all, total_correct/total_all*100)
    return results

# ── 构建 WORD_TO_SUB ──
def build_word_to_sub(filtered):
    wts = {}
    for concept, words in filtered.items():
        for w in words:
            wts.setdefault(w, []).append(concept)
    return wts

# ── 主流程 ──
raw = expand_base()
configs = [
    ("无过滤(基线)", raw, None),
]

# A: 硬阈值
for t in [2, 3, 4]:
    f, r = filter_hard(raw, t)
    configs.append((f"硬阈值(t={t})", f, r))

# B: TF-IDF
for th in [0.3, 0.5]:
    f, r = filter_tfidf(raw, th)
    configs.append((f"TF-IDF(th={th})", f, r))

# C: best-friend
f, r = filter_best_friend(raw, 3, 1)
configs.append((f"BestFriend(k=3,mf=1)", f, r))

# 跑全部
print(f"{'配置':25s} {'总正确率':10s} {'苹果':8s} {'纸':8s} {'光':8s} {'花':8s} {'行':8s} {'口':8s} {'词表':6s} {'重叠':6s}")
print("-" * 100)

all_results = []
for name, filtered, removed in configs:
    wts = build_word_to_sub(filtered)
    resolve = make_resolver(wts)
    res = run_test(resolve)
    total = res["total"]
    wc = sum(len(ws) for ws in filtered.values())
    ov = overlap_rate(filtered)
    r_str = f"({removed})" if removed is not None else ""
    print(f"{name:25s} {total[0]:>2d}/{total[1]:<3d}({total[2]:.0f}%)"
          f"  {res['苹果'][2]:>5.0f}%  {res['纸'][2]:>5.0f}%  {res['光'][2]:>5.0f}%  {res['花'][2]:>5.0f}%  {res['行'][2]:>5.0f}%  {res['口'][2]:>5.0f}%"
          f"  {wc:>4d}  {ov:.0%}")
    all_results.append((name, total[2], res['苹果'][2], res['花'][2], res['光'][2]))

print("\n" + "=" * 100)
print("通过标准检查：")
for name, total_pct, apple_pct, flower_pct, light_pct in all_results:
    check_total = total_pct > 50
    check_flower = flower_pct >= 83
    check_light = light_pct >= 33
    flags = []
    if check_total: flags.append("总>50%")
    if check_flower: flags.append("花≥83%")
    if check_light: flags.append("光≥33%")
    passed = all([check_total, check_flower, check_light])
    status = "✅" if passed else "❌"
    print(f"  {name:25s} {status} ({', '.join(flags) if flags else '无一通过'})")
