"""P0-A-2: 自动扩展词表 Semantics Graph
=========================================
用腾讯词向量（200d）从种子词自动扩展概念词表。
"""
import sys
sys.path = [p for p in sys.path if 'Diviner' not in p]

from gensim.models import KeyedVectors
import numpy as np

# ── 加载词向量 ──
KV_PATH = r"E:\intentCloud\models\text2vec-word2vec-tencent-chinese\light_Tencent_AILab_ChineseEmbedding.bin"
print("加载腾讯词向量...")
kv = KeyedVectors.load_word2vec_format(KV_PATH, binary=True)
print(f"  共 {len(kv.index_to_key)} 词，{kv.vector_size} 维")

# ── 种子词定义 ──
SEEDS = {
    # 科技域
    "科技_消费电子": ["手机", "芯片", "系统", "屏幕", "应用", "发布", "商店", "生态"],
    "科技_产品": ["产品", "品牌", "营收", "市场", "新款", "版本", "销量"],
    "科技_研发": ["芯片", "研发", "专利", "自研", "性能", "升级", "架构"],
    # 水果域
    "水果_种植": ["树", "种植", "采摘", "开花", "修剪", "丰收", "成熟"],
    "水果_品质": ["甜", "营养", "含糖量", "口感", "新鲜", "果汁", "果肉"],
    "水果_品种": ["品种", "特产", "产地", "上市", "季节性"],
    # 自然_光线
    "自然_光线": ["太阳", "明亮", "照射", "反射", "折射", "阳光", "光芒", "耀眼", "紫外线"],
    # 自然_植物
    "自然_植物": ["叶子", "根", "种子", "开花", "生长", "盆栽", "绿植", "园林"],
    # 生活_消费
    "生活_消费": ["钱", "消费", "付款", "购买", "花费", "折扣", "预算", "便宜"],
    # 生活_日常
    "生活_日常": ["吃饭", "睡觉", "上班", "回家", "出门", "做饭", "洗", "休息"],
    # 生活_建筑
    "生活_建筑": ["门", "窗户", "墙", "房间", "入口", "出口", "走廊", "楼梯"],
    # 语言_修辞
    "语言_修辞": ["比喻", "象征", "夸张", "拟人", "排比", "修辞", "形象"],
    # 语言_评价
    "语言_评价": ["好", "坏", "不错", "优秀", "差", "满意", "一般"],
    # 语言_抽象
    "语言_抽象": ["概念", "定义", "逻辑", "因果", "关系", "本质", "结构"],
}

# 大类划分
DOMAINS = {
    "科技":   ["科技_消费电子", "科技_产品", "科技_研发"],
    "水果":   ["水果_种植", "水果_品质", "水果_品种"],
    "自然":   ["自然_光线", "自然_植物"],
    "生活":   ["生活_消费", "生活_日常", "生活_建筑"],
    "语言":   ["语言_修辞", "语言_评价", "语言_抽象"],
}

# 歧义词 → 候选大类
AMBIGUOUS = {
    "苹果": ["科技", "水果"],
    "纸":   ["科技", "生活"],
    "光":   ["自然", "语言"],
    "花":   ["自然", "生活"],
    "行":   ["生活", "语言"],
    "口":   ["生活", "生活"],
}

# 中性词（不过滤，但在推理中权重降低）
NEUTRAL = {"比", "这款", "这个", "那个", "的", "了", "是", "很", "也", "而且", "和", "在", "有", "不", "就", "把", "被", "从"}

# ── 近邻扩展 ──
TOP_K = 20

print("\n扩展词表...")
EXPANDED = {}
for concept, seeds in SEEDS.items():
    words_set = set(seeds)
    for s in seeds:
        if s in kv:
            try:
                neighbors = [w for w, _ in kv.most_similar(s, topn=TOP_K)]
                words_set.update(neighbors)
            except:
                pass
    # 去重，去掉自身
    words_set.discard(s)
    EXPANDED[concept] = words_set
    print(f"  {concept:12s}: {len(seeds):2d}种子 → {len(words_set):3d}扩展")

# ── 跨类重叠检查 ──
print("\n跨类重叠检查（有无词同时属于两个类）：")
all_concepts = list(EXPANDED.keys())
overlap_count = 0
for i in range(len(all_concepts)):
    for j in range(i+1, len(all_concepts)):
        c1, c2 = all_concepts[i], all_concepts[j]
        overlap = EXPANDED[c1] & EXPANDED[c2]
        if overlap:
            overlap_count += len(overlap)
            rate = len(overlap) / min(len(EXPANDED[c1]), len(EXPANDED[c2])) * 100
            print(f"  {c1} ↔ {c2}: {len(overlap)}词重叠 ({rate:.0f}%)")
print(f"  总重叠词数: {overlap_count}")

# ── 构建 word→concept 表 ──
WORD_TO_SUB = {}
for concept, words in EXPANDED.items():
    for w in words:
        if w not in WORD_TO_SUB:
            WORD_TO_SUB[w] = []
        WORD_TO_SUB[w].append(concept)

# ── 分词 ──
def tokenize(sentence):
    found = set()
    for w in sorted(WORD_TO_SUB.keys(), key=len, reverse=True):
        if w in sentence:
            found.add(w)
    for aw in AMBIGUOUS:
        if aw in sentence:
            found.add(aw)
    return found

# ── 推理 ──
def layer1(sentence):
    words = tokenize(sentence)
    domain_scores = {d: 0 for d in DOMAINS}
    for w in words:
        if w in NEUTRAL:
            continue
        if w in AMBIGUOUS:
            for d in AMBIGUOUS[w]:
                if d in domain_scores:
                    domain_scores[d] += 1
        elif w in WORD_TO_SUB:
            for sub_c in WORD_TO_SUB[w]:
                for domain, subs in DOMAINS.items():
                    if sub_c in subs:
                        domain_scores[domain] += 1
    sorted_d = sorted(domain_scores.items(), key=lambda x: -x[1])
    top_name, top_score = sorted_d[0]
    second_score = sorted_d[1][1] if len(sorted_d) > 1 else 0
    if second_score > 0 and top_score >= second_score * 1.5:
        return top_name, domain_scores, True
    elif second_score == 0 and top_score > 0:
        return top_name, domain_scores, True
    else:
        return None, domain_scores, False

def resolve(sentence, ambiguous_word):
    winner, scores1, decided1 = layer1(sentence)
    if decided1:
        return winner
    tied = [d for d, s in scores1.items() if s > 0]
    words = tokenize(sentence)
    sub_scores = {}
    for d in tied:
        for sub in DOMAINS[d]:
            sub_scores[sub] = 0
    for w in words:
        if w in AMBIGUOUS:
            for d in AMBIGUOUS[w]:
                if d in tied:
                    for sub in DOMAINS[d]:
                        sub_scores[sub] += 1
        elif w in WORD_TO_SUB:
            for sub_c in WORD_TO_SUB[w]:
                if sub_c in sub_scores:
                    sub_scores[sub_c] += 1
    if not sub_scores:
        return "不确定"
    sorted_s = sorted(sub_scores.items(), key=lambda x: -x[1])
    top_name, top_score = sorted_s[0]
    second_score = sorted_s[1][1] if len(sorted_s) > 1 else 0
    if second_score > 0 and top_score >= second_score * 1.5:
        for d, subs in DOMAINS.items():
            if top_name in subs:
                return d
        return top_name
    elif second_score == 0 and top_score > 0:
        for d, subs in DOMAINS.items():
            if top_name in subs:
                return d
        return top_name
    else:
        return "不确定"

# ── 测试用例 ──
TEST_CASES = {
    "苹果": [
        ("这个苹果比上一代便宜了五百块", "科技"),
        ("苹果发布了新款手机", "科技"),
        ("今年苹果的芯片性能提升很大", "科技"),
        ("苹果的屏幕显示效果很好", "科技"),
        ("苹果应用商店的规则更新了", "科技"),
        ("这个苹果比上一代更甜", "水果"),
        ("苹果的采摘季节到了", "水果"),
        ("今年的苹果果肉很甜", "水果"),
        ("苹果正在开花", "水果"),
        ("苹果的含糖量很高", "水果"),
        ("这个苹果很好吃，而且设计也很漂亮", "不确定"),
    ],
    "纸": [
        ("打印机没纸了", "科技"),
        ("这张纸的质量很好", "科技"),
        ("把协议落实到纸面上", "科技"),
        ("论文的摘要写在一张纸上", "科技"),
        ("用纸包住花束", "不确定"),
        ("纸抽用完了", "不确定"),
    ],
    "光": [
        ("太阳的光很强烈", "自然"),
        ("光在水面发生折射", "自然"),
        ("钱都花光了", "语言"),
        ("光说不做", "语言"),
        ("他为国争光", "语言"),
        ("光的速度是每秒钟三十万公里", "自然"),
    ],
    "花": [
        ("花园里的花开了", "自然"),
        ("这盆花需要浇水", "自然"),
        ("花了很多钱", "生活"),
        ("花钱如流水", "生活"),
        ("花时间学习", "生活"),
        ("花开得很鲜艳", "自然"),
    ],
    "行": [
        ("往前走三行", "生活"),
        ("行，就这样吧", "语言"),
        ("这个办法行不通", "语言"),
        ("他在银行工作", "科技"),
        ("一行代码", "科技"),
    ],
    "口": [
        ("他张开了口", "生活"),
        ("入口在左边", "生活"),
        ("三口之家", "生活"),
        ("伤口很深", "不确定"),
    ],
}

# ── 测试 ──
print("\n" + "=" * 60)
print("P0-A-2: 自动扩展 Semantics Graph 测试")
print("=" * 60)

total_correct = 0
total_all = 0
word_results = []
word_baseline = {"苹果": "82%", "纸": "17%", "光": "33%", "花": "33%", "行": "0%", "口": "75%"}

for word, cases in TEST_CASES.items():
    print(f"\n{'='*50}")
    print(f"歧义词: 「{word}」 (手写基线: {word_baseline.get(word, '?')})")
    print(f"{'='*50}")
    correct = 0
    for sent, expected in cases:
        result = resolve(sent, word)
        match = result == expected
        if match:
            correct += 1
        icon = "✅" if match else "❌"
        print(f"  {icon} 预期={expected:4s}  →  {result}")
        print(f"      {sent}")
    rate = correct / len(cases) * 100
    total_correct += correct
    total_all += len(cases)
    word_results.append((word, correct, len(cases), rate))
    print(f"  → {correct}/{len(cases)} ({rate:.0f}%)")

print("\n" + "=" * 60)
print("结果对比")
print("=" * 60)
print(f"{'歧义词':6s} {'手写版':8s} {'自动扩展':8s}")
print("-" * 25)
for word, c, t, r in word_results:
    baseline = word_baseline.get(word, "?")
    print(f"  {word:4s}    {baseline:>7s}    {c}/{t} ({r:.0f}%)")

print(f"\n总正确率: {total_correct}/{total_all} ({total_correct/total_all*100:.0f}%)")
print(f"手写版总正确率: 17/38 (45%)")

if total_correct > 17:
    print("\n✅ 自动扩展提升了正确率！词向量扩展策略有效。")
elif total_correct == 17:
    print("\n⚠️ 持平。自动扩展没有损害性能，但也没有提升。")
else:
    print(f"\n❌ 自动扩展后正确率下降。需要检查：种子词质量、近邻噪声、跨类重叠。")
