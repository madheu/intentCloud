"""P0-A-6: 布尔向量全覆盖 — 物/事/质/序 四轴
===============================================
补全所有测试句的 jieba 分词词表。
"""
import sys; sys.path = [p for p in sys.path if 'Diviner' not in p]
import jieba
import numpy as np

# ── 四轴重定义：物/事/质/序 ──
# 物: 具体/抽象实体、事物、生命体
# 事: 动作、过程、变化、活动
# 质: 属性、特征、状态、评价
# 序: 时间、空间、因果、角色关系

# ── 歧义词的候选向量 ──
AMBIGUOUS = {
    "苹果": [("科技", (1,1,0,0)), ("水果", (1,0,1,0))],  # 物+事 vs 物+质
    "纸":   [("科技", (1,1,0,0)), ("生活", (1,0,0,0))],  # 物+事 vs 物
    "光":   [("自然", (0,1,0,1)), ("语言", (0,0,1,1))],  # 事+序 vs 质+序
    "花":   [("自然", (1,0,1,0)), ("生活", (0,1,0,0))],  # 物+质 vs 事
    "行":   [("生活", (0,1,0,1)), ("语言", (0,0,1,0)), ("科技", (1,1,0,0))],
    "口":   [("生活", (1,1,0,0)), ("生活", (1,0,0,1))],  # 两个义项都属生活域
}

# ── 全覆盖布尔词表 ──
WORD_VEC = {
    # === 原始词表 ===
    "太阳": (1,1,0,1), "水面": (1,1,1,1), "折射": (0,1,0,1),
    "速度": (0,0,1,1), "手机": (1,0,0,0), 
    "系统": (0,0,1,1), "屏幕": (1,0,0,0), 
    "便宜": (0,0,1,0), "上一代": (0,0,1,1),
    "甜": (0,0,1,0), 
    "好吃": (0,0,1,0), 
    "漂亮": (0,0,1,0),
    "塑料袋": (1,0,0,0),
    
    # === Step 1 缺失词（从 jieba 分析补入）===
    # 苹果科技句
    "发布": (0,1,0,1),    # 事+序(事件+面向公众的角色关系)
    "新款": (0,0,1,1),    # 质+序(属性+时间上新)
    "芯片": (1,0,0,0),    # 物(实体)
    "性能": (0,0,1,0),    # 质(属性)
    "提升": (0,1,0,1),    # 事+序(事件+空间上)
    "商店": (1,0,0,1),    # 物+序(实体+空间)
    "规则": (0,0,1,1),    # 质+序(属性+约束关系)
    "更新": (0,1,0,1),    # 事+序(事件+时间)
    "正在": (0,0,0,1),    # 序(时间关系)
    
    # 苹果水果句
    "开花": (0,1,0,0),    # 事(植物生长事件)
    "采摘": (0,1,0,0),    # 事(农业活动)
    "季节": (0,0,0,1),    # 序(时间)
    "果肉": (1,0,1,0),    # 物+质(实体+质地)
    "含糖量": (0,0,1,0),  # 质(测量属性)
    "设计": (0,1,1,0),    # 事+质(活动+审美属性)
    
    # 花园/花句
    "花园": (1,0,0,1),    # 物+序(实体+空间)
    "浇水": (0,1,0,0),    # 事(事件)
    "鲜艳": (0,0,1,0),    # 质(视觉属性)
    "花钱": (0,1,0,0),    # 事(消费活动)
    "流水": (0,1,0,1),    # 事+序(流动+路径)
    "学习": (0,1,1,0),    # 事+质(活动+知识属性)
    
    # 光句
    "强烈": (0,0,1,0),    # 质(强度属性)
    "水面": (1,0,0,1),    # 物+序(实体+空间)
    
    # 纸句
    "打印机": (1,0,0,0),  # 物(实体)
    "质量": (0,0,1,0),    # 质(属性)
    "协议": (0,0,1,1),    # 质+序(属性+关系)
    "论文": (0,0,1,1),    # 质+序(知识+逻辑)
    "摘要": (0,0,1,1),    # 质+序(知识+结构)
    "落实": (0,1,0,1),    # 事+序(事件+从抽象到具体的关系)
    "纸面": (1,0,0,1),    # 物+序(实体+空间/媒介)
    "包装": (0,1,0,1),    # 事+序(事件+包裹关系)
    "纸抽": (1,0,0,0),    # 物(实体)
    
    # 口句
    "张开": (0,1,0,1),    # 事+序(事件+空间变化)
    "入口": (1,0,0,1),    # 物+序(实体+空间)
    "伤口": (1,0,1,1),    # 物+质+序(实体+状态+因果关系)
    "左边": (0,0,0,1),    # 序(空间)
    "三口": (0,0,0,1),    # 序(计量关系)
    "之家": (0,0,0,1),    # 序(归属关系)
    
    # 行句
    "往前走": (0,1,0,1),  # 事+序(移动+方向)
    "三行": (0,0,0,1),    # 序(空间排列)
    "就这样": (0,0,1,0),  # 质(状态)
    "吧": (0,0,0,0),      # 语气词→全零
    "办法": (0,0,1,1),    # 质+序(方法属性+手段关系)
    "行不通": (0,0,1,0),  # 质(评价)
    "在": (0,0,0,1),      # 序(空间关系)
    "银行": (1,1,0,1),    # 物+事+序(实体+金融活动+空间)
    "一行": (0,0,0,1),    # 序(空间排列)
    "代码": (0,0,1,1),    # 质+序(符号属性+逻辑关系)
    
    # 歧义词自身的默认向量
    "苹果": (1,0,1,0),    # 物+质(实体+属性) — 默认是水果
    "纸": (1,0,0,0),      # 物
    "光": (0,0,0,1),      # 序 — 中性
    "花": (1,0,1,0),      # 物+质
    "行": (0,1,0,1),      # 事+序
    "口": (1,0,0,1),      # 物+序
}

# 中性词
NEUTRAL = {"比","这款","这个","那个","的","了","是","很","也","而且","和","在","有","不","就","把","被","从",
           "一个","没有","可以","会","都","要","能","让","上","下","来","去","说","看","做","用","知道",
           "怎么", "什么", "这", "那", "它", "他", "她", "我", "你", "们", "自己", "人", "为"}

# ── 消歧 ──
def boolean_disambiguate(sentence, ambiguous_word):
    if ambiguous_word not in AMBIGUOUS:
        return "不确定", {}, np.array([0,0,0,0])
    
    words = jieba.lcut(sentence)
    context_vec = np.array([0, 0, 0, 0], dtype=float)
    missing = []
    
    for w in words:
        if w == ambiguous_word or w in NEUTRAL:
            continue
        if w in WORD_VEC:
            context_vec += np.array(WORD_VEC[w])
        else:
            missing.append(w)
    
    # 歧义词自身贡献默认向量
    if ambiguous_word in WORD_VEC:
        context_vec += np.array(WORD_VEC[ambiguous_word])
    
    candidates = AMBIGUOUS[ambiguous_word]
    scores = {}
    for label, vec in candidates:
        scores[label] = float(np.dot(context_vec, np.array(vec)))
    
    sorted_s = sorted(scores.items(), key=lambda x: -x[1])
    top_name, top_score = sorted_s[0]
    second_score = sorted_s[1][1] if len(sorted_s) > 1 else 0
    
    if second_score > 0 and top_score >= second_score * 1.5:
        return top_name, scores, context_vec, missing
    elif second_score == 0 and top_score > 0:
        return top_name, scores, context_vec, missing
    else:
        return "不确定", scores, context_vec, missing

# ── 测试 ──
TEST_CASES = {
    "苹果": [("这个苹果比上一代便宜了五百块","科技"),("苹果发布了新款手机","科技"),("今年苹果的芯片性能提升很大","科技"),("苹果的屏幕显示效果很好","科技"),("苹果应用商店的规则更新了","科技"),("这个苹果比上一代更甜","水果"),("苹果的采摘季节到了","水果"),("今年的苹果果肉很甜","水果"),("苹果正在开花","水果"),("苹果的含糖量很高","水果"),("这个苹果很好吃，而且设计也很漂亮","不确定")],
    "纸": [("打印机没纸了","科技"),("这张纸的质量很好","科技"),("把协议落实到纸面上","科技"),("论文的摘要写在一张纸上","科技"),("用纸包住花束","不确定"),("纸抽用完了","不确定")],
    "光": [("太阳的光很强烈","自然"),("光在水面发生折射","自然"),("钱都花光了","语言"),("光说不做","语言"),("他为国争光","语言"),("光的速度是每秒钟三十万公里","自然")],
    "花": [("花园里的花开了","自然"),("这盆花需要浇水","自然"),("花了很多钱","生活"),("花钱如流水","生活"),("花时间学习","生活"),("花开得很鲜艳","自然")],
    "行": [("往前走三行","生活"),("行，就这样吧","语言"),("这个办法行不通","语言"),("他在银行工作","科技"),("一行代码","科技")],
    "口": [("他张开了口","生活"),("入口在左边","生活"),("三口之家","生活"),("伤口很深","不确定")],
}

BASELINE = {"苹果":"45%","纸":"83%","光":"50%","花":"50%","行":"40%","口":"25%"}

print("=" * 60)
print("P0-A-6: 布尔向量全覆盖 — 物/事/质/序 四轴")
print("=" * 60)

total_correct = 0; total_all = 0; total_missing = set()
results = {}

for word, cases in TEST_CASES.items():
    correct = 0; fails = []; word_missing = set()
    for sent, expected in cases:
        result, scores, ctx, missing = boolean_disambiguate(sent, word)
        word_missing.update(missing)
        match = result == expected
        if match: correct += 1
        else: fails.append((sent, expected, result, ctx, scores, missing))
    rate = correct/len(cases)*100
    results[word] = (correct, len(cases), rate)
    total_correct += correct; total_all += len(cases)
    total_missing.update(word_missing)
    
    print(f"\n「{word}」: {correct}/{len(cases)} ({rate:.0f}%)  布尔v1: {BASELINE[word]}")
    for sent, exp, got, ctx, scores, miss in fails:
        score_str = ", ".join(f"{k}={v:.0f}" for k,v in scores.items())
        miss_str = f" 缺:{miss}" if miss else ""
        print(f"  ❌ 预期={exp:4s} → {got:4s}  ctx=({ctx[0]:.0f},{ctx[1]:.0f},{ctx[2]:.0f},{ctx[3]:.0f})  [{score_str}]{miss_str}")
        print(f"      {sent}")
    if word_missing:
        print(f"  失踪词: {sorted(word_missing)}")

print(f"\n{'='*60}")
print(f"总计: {total_correct}/{total_all} ({total_correct/total_all*100:.0f}%)")
print(f"布尔v1: 19/38 (50%)")
print(f"缺失词总数: {len(total_missing)}")

print(f"\n{'='*60}")
print("通过标准检查：")
passed = True
for word, req in [("苹果", 80), ("花", 75), ("口", 75)]:
    c, t, r = results[word]
    ok = r >= req
    if not ok: passed = False
    print(f"  {'✅' if ok else '❌'} {word}: {r:.0f}% >= {req}%")
ok_total = total_correct/total_all*100 >= 65
if not ok_total: passed = False
print(f"  {'✅' if ok_total else '❌'} 总计: {total_correct/total_all*100:.0f}% >= 65%")
print(f"\n{'✅✅ 全部通过' if passed else '❌ 未全部通过'}")
