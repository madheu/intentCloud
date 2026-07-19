"""P0-A-4: 修复单字噪声 + 结巴分词
===================================
修复1: 过滤单字词（len >= 2）
修复2: 结巴分词替代子串匹配
修复3: 补充中性词表
"""
import sys; sys.path = [p for p in sys.path if 'Diviner' not in p]
import jieba
from gensim.models import KeyedVectors
from collections import defaultdict
import math

KV_PATH = r"E:\intentCloud\models\text2vec-word2vec-tencent-chinese\light_Tencent_AILab_ChineseEmbedding.bin"
print("加载腾讯词向量...")
kv = KeyedVectors.load_word2vec_format(KV_PATH, binary=True)
print(f"  共 {len(kv.index_to_key)} 词，{kv.vector_size} 维\n")

TOP_K = 20
SEEDS = {
    "科技_消费电子": ["手机","芯片","系统","屏幕","应用","发布","商店","生态"],
    "科技_产品": ["产品","品牌","营收","市场","新款","版本","销量"],
    "科技_研发": ["芯片","研发","专利","自研","性能","升级","架构"],
    "水果_种植": ["树","种植","采摘","开花","修剪","丰收","成熟"],
    "水果_品质": ["甜","营养","含糖量","口感","新鲜","果汁","果肉"],
    "水果_品种": ["品种","特产","产地","上市","季节性"],
    "自然_光线": ["太阳","明亮","照射","反射","折射","阳光","光芒","耀眼","紫外线"],
    "自然_植物": ["叶子","根","种子","开花","生长","盆栽","绿植","园林"],
    "生活_消费": ["钱","消费","付款","购买","花费","折扣","预算","便宜"],
    "生活_日常": ["吃饭","睡觉","上班","回家","出门","做饭","洗","休息"],
    "生活_建筑": ["门","窗户","墙","房间","入口","出口","走廊","楼梯"],
    "语言_修辞": ["比喻","象征","夸张","拟人","排比","修辞","形象"],
    "语言_评价": ["好","坏","不错","优秀","差","满意","一般"],
    "语言_抽象": ["概念","定义","逻辑","因果","关系","本质","结构"],
}
DOMAINS = {"科技":["科技_消费电子","科技_产品","科技_研发"],"水果":["水果_种植","水果_品质","水果_品种"],"自然":["自然_光线","自然_植物"],"生活":["生活_消费","生活_日常","生活_建筑"],"语言":["语言_修辞","语言_评价","语言_抽象"]}
AMBIGUOUS = {"苹果":["科技","水果"],"纸":["科技","生活"],"光":["自然","语言"],"花":["自然","生活"],"行":["生活","语言"],"口":["生活","生活"]}
# 修复3: 补充中性词表
NEUTRAL = {"比","这款","这个","那个","的","了","是","很","也","而且","和","在","有","不","就","把","被","从",
           "一个","没有","可以","会","都","要","能","让","上","下","来","去","说","看","做","用","知道"}

# ── 基本扩展 ──
raw = {}
for concept, seeds in SEEDS.items():
    ws = set(seeds)
    for s in seeds:
        if s in kv:
            try: ws.update(w for w,_ in kv.most_similar(s, topn=TOP_K))
            except: pass
    # 修复1: 过滤单字词
    ws = {w for w in ws if len(w) >= 2}
    raw[concept] = ws

# 建 WORD_TO_SUB
WORD_TO_SUB = {}
for c, ws in raw.items():
    for w in ws:
        WORD_TO_SUB.setdefault(w, []).append(c)

print(f"扩展后词表大小: {len(WORD_TO_SUB)} 词（已过滤单字）")

# ── 修复2: 结巴分词 ──
def tokenize(sentence):
    words_in_sent = set(jieba.lcut(sentence))
    found = set()
    # 按词长从长到短匹配
    all_vocab = sorted(WORD_TO_SUB.keys(), key=len, reverse=True)
    for w in all_vocab:
        if w in words_in_sent:
            found.add(w)
    for aw in AMBIGUOUS:
        if aw in words_in_sent:
            found.add(aw)
    return found

# ── 推理 ──
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

# ── 测试 ──
TEST_CASES = {
    "苹果": [("这个苹果比上一代便宜了五百块","科技"),("苹果发布了新款手机","科技"),("今年苹果的芯片性能提升很大","科技"),("苹果的屏幕显示效果很好","科技"),("苹果应用商店的规则更新了","科技"),("这个苹果比上一代更甜","水果"),("苹果的采摘季节到了","水果"),("今年的苹果果肉很甜","水果"),("苹果正在开花","水果"),("苹果的含糖量很高","水果"),("这个苹果很好吃，而且设计也很漂亮","不确定")],
    "纸": [("打印机没纸了","科技"),("这张纸的质量很好","科技"),("把协议落实到纸面上","科技"),("论文的摘要写在一张纸上","科技"),("用纸包住花束","不确定"),("纸抽用完了","不确定")],
    "光": [("太阳的光很强烈","自然"),("光在水面发生折射","自然"),("钱都花光了","语言"),("光说不做","语言"),("他为国争光","语言"),("光的速度是每秒钟三十万公里","自然")],
    "花": [("花园里的花开了","自然"),("这盆花需要浇水","自然"),("花了很多钱","生活"),("花钱如流水","生活"),("花时间学习","生活"),("花开得很鲜艳","自然")],
    "行": [("往前走三行","生活"),("行，就这样吧","语言"),("这个办法行不通","语言"),("他在银行工作","科技"),("一行代码","科技")],
    "口": [("他张开了口","生活"),("入口在左边","生活"),("三口之家","生活"),("伤口很深","不确定")],
}

BASELINE = {"苹果":"82%","纸":"17%","光":"33%","花":"33%","行":"0%","口":"75%"}

print(f"\n{'='*60}")
print(f"P0-A-4: 修复后测试")
print(f"{'='*60}\n")

total_correct = 0; total_all = 0
results = {}

for word, cases in TEST_CASES.items():
    correct = 0
    fails = []
    for sent, expected in cases:
        result = resolve(sent, word)
        match = result == expected
        if match: correct += 1
        else: fails.append((sent, expected, result))
    rate = correct/len(cases)*100
    results[word] = (correct, len(cases), rate, fails)
    total_correct += correct; total_all += len(cases)
    
    print(f"「{word}」: {correct}/{len(cases)} ({rate:.0f}%)  手写版基线: {BASELINE[word]}")
    for sent, exp, got in fails:
        print(f"  ❌ 预期={exp}  →  {got}  |  {sent}")
    print()

print(f"{'='*60}")
print(f"总计: {total_correct}/{total_all} ({total_correct/total_all*100:.0f}%)")
print(f"手写版基线总计: 17/38 (45%)")
print(f"P0-A-2 自动扩展总计: 19/38 (50%)")
print(f"{'='*60}")

# 通过标准
print("\n通过标准检查：")
passed = True
for word, req in [("光", 33), ("苹果", 80), ("花", 75)]:
    c, t, r, _ = results[word]
    ok = r >= req
    if not ok: passed = False
    print(f"  {'✅' if ok else '❌'} {word}: {r:.0f}% >= {req}%")

total_rate = total_correct/total_all*100
ok_total = total_rate >= 55
if not ok_total: passed = False
print(f"  {'✅' if ok_total else '❌'} 总计: {total_rate:.0f}% >= 55%")
print(f"\n  {'✅✅ 全部通过' if passed else '❌ 未全部通过'}")
