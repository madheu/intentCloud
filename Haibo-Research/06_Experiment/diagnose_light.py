"""诊断：为什么"太阳的光很强烈"被判为语言？"""
import sys; sys.path = [p for p in sys.path if 'Diviner' not in p]
from gensim.models import KeyedVectors
from collections import defaultdict
import math

KV_PATH = r"E:\intentCloud\models\text2vec-word2vec-tencent-chinese\light_Tencent_AILab_ChineseEmbedding.bin"
kv = KeyedVectors.load_word2vec_format(KV_PATH, binary=True)

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
NEUTRAL = {"比","这款","这个","那个","的","了","是","很","也","而且","和","在","有","不","就","把","被","从"}

# 扩展
raw = {}
for concept, seeds in SEEDS.items():
    ws = set(seeds)
    for s in seeds:
        if s in kv:
            try: ws.update(w for w,_ in kv.most_similar(s, topn=20))
            except: pass
    raw[concept] = ws

# 建表
WORD_TO_SUB = {}
for c, ws in raw.items():
    for w in ws:
        WORD_TO_SUB.setdefault(w, []).append(c)

print(f"总词数: {len(WORD_TO_SUB)}")

# 诊断 "太阳的光很强烈"
sent = "太阳的光很强烈"
for w in sorted(WORD_TO_SUB.keys(), key=len, reverse=True):
    if w in sent:
        concepts = WORD_TO_SUB[w]
        domains = set()
        for c in concepts:
            for d, subs in DOMAINS.items():
                if c in subs: domains.add(d)
        neutral = "⭐中性" if w in NEUTRAL else ""
        print(f"  找到: '{w}' → 概念{concepts} → 域{domains} {neutral}")

print()
# 模式2：光的速度是每秒钟三十万公里
sent2 = "光的速度是每秒钟三十万公里"
print(f"\n诊断2: \"{sent2}\"")
for w in sorted(WORD_TO_SUB.keys(), key=len, reverse=True):
    if w in sent2:
        concepts = WORD_TO_SUB[w]
        domains = set()
        for c in concepts:
            for d, subs in DOMAINS.items():
                if c in subs: domains.add(d)
        neutral = "⭐中性" if w in NEUTRAL else ""
        print(f"  找到: '{w}' → 概念{concepts} → 域{domains} {neutral}")
