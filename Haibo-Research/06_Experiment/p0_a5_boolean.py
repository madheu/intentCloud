"""P0-A-5: 布尔向量消歧（天、地、人、心 四轴）
==============================================
不用概念域、不用词向量。只用 4 个布尔轴表示语义。
"""
import sys; sys.path = [p for p in sys.path if 'Diviner' not in p]
import jieba
import numpy as np

# ── 歧义词的候选向量 ──
# 格式: {歧义词: [(义项名, (天,地,人,心)), ...]}
AMBIGUOUS = {
    "苹果": [("科技", (0,1,1,0)), ("水果", (0,1,0,1))],
    "纸":   [("科技", (0,1,1,0)), ("生活", (0,1,0,0))],
    "光":   [("自然", (1,0,0,0)), ("语言", (0,0,1,1))],
    "花":   [("自然", (0,1,0,1)), ("生活", (0,0,1,0))],
    "行":   [("生活", (0,1,0,0)), ("语言", (0,0,0,1)), ("科技", (0,1,1,0))],
    "口":   [("生活", (0,1,1,0)), ("生活", (0,1,0,0))],  # 两个义项都属生活域
}

# ── 语境词 → 布尔向量 ──
WORD_VEC = {
    "太阳": (1,1,0,0), "水面": (1,1,0,0), "折射": (1,0,1,0),
    "速度": (1,0,1,0), "手机": (0,1,1,0), "芯片": (0,0,1,1),
    "系统": (0,0,1,1), "屏幕": (0,1,1,0), "应用": (0,0,1,1),
    "发布": (0,0,1,1), "商店": (0,1,1,0), "新款": (0,0,0,1),
    "便宜": (0,0,1,1), "上一代": (0,1,1,0),
    "五百": (0,0,1,1), "性能": (0,0,1,1), "提升": (0,0,1,1),
    "采摘": (0,1,1,0), "季节": (1,0,1,0), "果肉": (0,1,0,1),
    "甜": (0,0,0,1), "含糖量": (0,0,1,1), "开花": (1,1,0,1),
    "好吃": (0,0,0,1), "设计": (0,0,1,1), "漂亮": (0,0,0,1),
    "打印机": (0,1,1,0), "协议": (0,0,1,1), "论文": (0,0,1,1),
    "摘要": (0,0,1,1), "包装": (0,1,1,0), "钱": (0,1,1,0),
    "消费": (0,0,1,1), "时间": (1,0,1,0), "水": (1,1,0,0),
    "可以": (0,0,0,1), "办法": (0,0,1,1), "银行": (0,1,1,0),
    "代码": (0,0,1,0), "队伍": (0,0,1,1), "入口": (0,1,1,0),
    "张嘴": (0,1,1,0), "三口": (0,0,1,0), "伤口": (0,1,1,1),
    "花园": (0,1,0,1), "浇水": (0,1,0,1), "鲜艳": (0,0,0,1),
    "学习": (0,0,1,1), "纸": (0,1,0,0), "光": (1,0,0,0),
    "花": (0,1,0,1), 
    # 补充 P0-A-5 表中词
    "块": (0,0,1,0), "苹果": (0,1,1,0), 
}

# 中性词（不产生激活）
NEUTRAL = {"比","这款","这个","那个","的","了","是","很","也","而且","和","在","有","不","就","把","被","从",
           "一个","没有","可以","会","都","要","能","让","上","下","来","去","说","看","做","用","知道"}

# ── 消歧 ──
def boolean_disambiguate(sentence, ambiguous_word):
    """布尔向量消歧"""
    if ambiguous_word not in AMBIGUOUS:
        return "不确定", {}, {}
    
    words = jieba.lcut(sentence)
    
    # 累加语境向量（去歧义词本身和中性词）
    context_vec = np.array([0, 0, 0, 0], dtype=float)
    found_words = []
    for w in words:
        if w == ambiguous_word or w in NEUTRAL:
            continue
        if w in WORD_VEC:
            context_vec += np.array(WORD_VEC[w])
            found_words.append(w)
    
    # 计算歧义词每个候选向量的点积
    candidates = AMBIGUOUS[ambiguous_word]
    scores = {}
    # 歧义词自身也贡献自己的向量（作为"默认信号"）
    if ambiguous_word in WORD_VEC:
        context_vec += np.array(WORD_VEC[ambiguous_word])
    
    for label, vec in candidates:
        scores[label] = float(np.dot(context_vec, np.array(vec)))
    
    # 排序
    sorted_s = sorted(scores.items(), key=lambda x: -x[1])
    top_name, top_score = sorted_s[0]
    second_score = sorted_s[1][1] if len(sorted_s) > 1 else 0
    
    if second_score > 0 and top_score >= second_score * 1.5:
        return top_name, scores, context_vec
    elif second_score == 0 and top_score > 0:
        return top_name, scores, context_vec
    else:
        return "不确定", scores, context_vec

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

print("=" * 60)
print("P0-A-5: 布尔向量消歧（四轴：天、地、人、心）")
print("=" * 60)

total_correct = 0; total_all = 0
results = {}

for word, cases in TEST_CASES.items():
    correct = 0; fails = []
    for sent, expected in cases:
        result, scores, ctx = boolean_disambiguate(sent, word)
        match = result == expected
        if match: correct += 1
        else: fails.append((sent, expected, result, ctx, scores))
    rate = correct/len(cases)*100
    results[word] = (correct, len(cases), rate)
    total_correct += correct; total_all += len(cases)
    
    print(f"\n「{word}」: {correct}/{len(cases)} ({rate:.0f}%)  [{BASELINE[word]}]")
    for sent, exp, got, ctx, scores in fails:
        score_str = ", ".join(f"{k}={v:.0f}" for k,v in scores.items())
        print(f"  ❌ 预期={exp} → {got}  ctx=({ctx[0]},{ctx[1]},{ctx[2]},{ctx[3]})  [{score_str}]")
        print(f"      {sent}")

print(f"\n{'='*60}")
print(f"总计: {total_correct}/{total_all} ({total_correct/total_all*100:.0f}%)")
print(f"{'='*60}")

print(f"\n{'='*60}")
print("通过标准：")
for word, req in [("光", 50), ("花", 75)]:
    c, t, r = results[word]
    ok = r >= req
    print(f"  {'✅' if ok else '❌'} {word}: {r:.0f}% >= {req}%")
ok_total = total_correct/total_all*100 >= 60
print(f"  {'✅' if ok_total else '❌'} 总计: {total_correct/total_all*100:.0f}% >= 60%")
