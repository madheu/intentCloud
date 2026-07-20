"""P0-B-1: HowNet 义原消歧（修复版 API）
=========================================
用 get_sememes_by_word() 提取义原 → 映射为 4 面向量 → 点积消歧。
"""
import sys; sys.path = [p for p in sys.path if 'Diviner' not in p]
import OpenHowNet
import numpy as np
import re

hownet = OpenHowNet.HowNetDict()
print("HowNet 加载成功")

# ── 义原 → 语义面映射 ──
SEMEME_FACE = {
    # 物
    "Thing": "物", "Entity": "物", "Object": "物", "Physical": "物",
    "Material": "物", "Substance": "物", "Solid": "物", "Liquid": "物",
    "Animal": "物", "Plant": "物", "Fruit": "物", "Food": "物", "Vegetable": "物",
    "Tool": "物", "Implement": "物", "Instrument": "物", "Machine": "物", "Device": "物",
    "Artifact": "物", "Product": "物", "Building": "物", "Container": "物",
    "Part": "物", "Body": "物", "Organ": "物", "Face": "物", "Head": "物",
    "Land": "物", "Stone": "物", "Metal": "物", "Wood": "物", "Paper": "物",
    "Book": "物", "Document": "物", "Money": "物", "Currency": "物",
    "Vehicle": "物", "Computer": "物", "Phone": "物",
    "FlowerGrass": "物", "celestial": "物", "tree": "物",
    "livestock": "物", "facilities": "物",
    "part": "物", "component": "物", "symbol": "物",
    "fund": "物", "money": "物", "merchandise": "物",
    "software": "物", "electronic": "物",
    # 事
    "Event": "事", "Action": "事", "Process": "事", "Activity": "事", "Behavior": "事",
    "Produce": "事", "Create": "事", "Make": "事", "Build": "事", "Grow": "事",
    "Change": "事", "Move": "事", "Transfer": "事", "Give": "事", "Take": "事",
    "Use": "事", "Operate": "事", "Control": "事", "Manage": "事",
    "Eat": "事", "Drink": "事", "Cook": "事", "Wash": "事",
    "Speak": "事", "Say": "事", "Tell": "事", "Write": "事",
    "Buy": "事", "Sell": "事", "Trade": "事", "Pay": "事", "Spend": "事",
    "Work": "事", "Study": "事", "Learn": "事",
    "Emit": "事", "Travel": "事", "Flow": "事", "Fly": "事",
    "Cause": "事", "Effect": "事", "Influence": "事",
    "Social": "事", "Communication": "事", "Education": "事",
    "announce": "事", "plans": "事", "plan": "事", "planting": "事",
    "draw": "事", "write": "事", "wrap": "事", "GoOut": "事",
    "display": "事", "compile": "事", "communicate": "事",
    "bring": "事", "finance": "事", "commerce": "事",
    "TakeBack": "事", "spend": "事", "split": "事",
    "forming": "事", "Behave": "事", "act": "事", "rest": "事",
    "sleep": "事", "Act": "事",
    # 质
    "Property": "质", "Attribute": "质", "Quality": "质", "Feature": "质",
    "Color": "质", "Shape": "质", "Size": "质", "Weight": "质",
    "Value": "质", "Price": "质", "Cost": "质",
    "Speed": "质", "Strength": "质", "Power": "质", "Temperature": "质",
    "State": "质", "Status": "质", "Condition": "质",
    "Ability": "质", "Skill": "质", "Knowledge": "质",
    "Appearance": "质", "Beauty": "质", "Ugly": "质",
    "Good": "质", "Bad": "质", "Correct": "质", "Wrong": "质",
    "High": "质", "Low": "质", "Big": "质", "Small": "质", "Long": "质", "Short": "质",
    "New": "质", "Old": "质", "Young": "质",
    "Sweet": "质", "Sour": "质", "Bitter": "质",
    "Bright": "质", "Dark": "质", "Light": "质",
    "Able": "质", "Feeling": "质", "Emotion": "质", "Evaluation": "质",
    "joyful": "质", "BeWell": "质", "beautiful": "质",
    "BehaviorValue": "质", "AptTo": "质", "blurred": "质",
    "Pattern": "质", "Glory": "质", "glorious": "质",
    "Age": "质",
    # 序
    "Relation": "序", "Time": "序", "Space": "序", "Location": "序",
    "Direction": "序", "Position": "序", "Place": "序",
    "Sequence": "序", "Order": "序", "Rank": "序",
    "Group": "序", "Set": "序", "Category": "序", "Type": "序",
    "PartWhole": "序", "Whole": "序", "Member": "序",
    "Possession": "序", "Own": "序", "Belong": "序",
    "Source": "序", "Target": "序", "Goal": "序", "Purpose": "序",
    "Means": "序", "Method": "序", "Way": "序",
    "Role": "序", "Function": "序", "Job": "序", "Occupation": "序",
    "Context": "序", "Situation": "序", "Circumstance": "序",
    "Pattern": "序", "Rule": "序", "Law": "序", "System": "序",
    "human": "序", "group": "序", "NounUnit": "序", "Unit": "序",
    "exposure": "序", "surplus": "序", "all": "序", "neg": "序",
    "InstitutePlace": "序", "ResidentialArea": "序",
    "location": "序", "position": "序", "linear": "序",
    "ResultFrom": "序",
}

def extract_sememe_names(sememe_data):
    """从 get_sememes_by_word 返回的数据中提取义原英文名"""
    sememe_names = set()
    if not sememe_data:
        return sememe_names
    for entry in sememe_data:
        sememes = entry.get('sememes', [])
        for s in sememes:
            if isinstance(s, str):
                # 格式: "tool|用具" 或 "fruit|水果"
                name = s.split('|')[0].strip()
                sememe_names.add(name)
    return sememe_names

def get_face_vector(word):
    """获取一个词的语义面向量"""
    data = hownet.get_sememes_by_word(word)
    names = extract_sememe_names(data)
    vec = [0, 0, 0, 0]
    for n in names:
        face = SEMEME_FACE.get(n)
        if face == "物": vec[0] += 1
        elif face == "事": vec[1] += 1
        elif face == "质": vec[2] += 1
        elif face == "序": vec[3] += 1
    return tuple(vec), names

# ── 歧义词候选（沿用布尔版已验证的候选向量）──
AMBIGUOUS_CANDIDATES = {
    "苹果": {"科技": (1,1,0,0), "水果": (1,0,1,0)},
    "纸":   {"科技": (1,1,0,0), "生活": (1,0,0,0)},
    "光":   {"自然": (0,0,0,1), "语言": (0,0,1,1)},
    "花":   {"自然": (1,0,1,0), "生活": (0,1,0,0)},
    "行":   {"生活": (0,0,0,1), "语言": (0,0,1,0), "科技": (1,1,0,0)},
    "口":   {"生活": (1,1,0,0)},
}
NEUTRAL = {"比","这款","这个","那个","的","了","是","很","也","而且","和","在","有","不","就","把","被","从"}

# ── 预缓存所有测试句中词的义原（避免重复调用 HowNet API）──
CACHE = {}
def cached_face_vector(word):
    if word not in CACHE:
        CACHE[word] = get_face_vector(word)
    return CACHE[word]

# ── 消歧 ──
def disambiguate(sentence, ambiguous_word):
    if ambiguous_word not in AMBIGUOUS_CANDIDATES:
        return "不确定", {}, []
    
    # 子串匹配找词
    found = set()
    # 对每个测试句，用滑动窗口找 HowNet 中存在的词
    for length in range(4, 0, -1):
        for i in range(len(sentence) - length + 1):
            seg = sentence[i:i+length]
            if seg in NEUTRAL:
                continue
            if seg == ambiguous_word:
                continue
            data = hownet.get_sememes_by_word(seg)
            if data and any(s.get('sememes') for s in data):
                found.add(seg)
    
    # 累加语境义原向量的均化版本
    context_vec = np.array([0.0, 0.0, 0.0, 0.0])
    found_words = []
    for w in found:
        vec, sememes = cached_face_vector(w)
        if any(v > 0 for v in vec):
            context_vec += np.array(vec, dtype=float)
            found_words.append(f"{w}({vec})")
    
    # 歧义词自身的义原
    ambig_vec, ambig_sems = cached_face_vector(ambiguous_word)
    context_vec += np.array(ambig_vec, dtype=float) * 0.5  # 歧义词自己贡献减半
    
    # 点积
    candidates = AMBIGUOUS_CANDIDATES[ambiguous_word]
    scores = {}
    for label, vec in candidates.items():
        scores[label] = float(np.dot(context_vec, np.array(vec)))
    
    sorted_s = sorted(scores.items(), key=lambda x: -x[1])
    top_name, top_score = sorted_s[0]
    second_score = sorted_s[1][1] if len(sorted_s) > 1 else 0
    
    if second_score > 0 and top_score >= second_score * 1.5:
        return top_name, scores, found_words
    elif second_score == 0 and top_score > 0:
        return top_name, scores, found_words
    else:
        return "不确定", scores, found_words

# ── 测试 ──
TEST_CASES = {
    "苹果": [("这个苹果比上一代便宜了五百块","科技"),("苹果发布了新款手机","科技"),("今年苹果的芯片性能提升很大","科技"),("苹果的屏幕显示效果很好","科技"),("苹果应用商店的规则更新了","科技"),("这个苹果比上一代更甜","水果"),("苹果的采摘季节到了","水果"),("今年的苹果果肉很甜","水果"),("苹果正在开花","水果"),("苹果的含糖量很高","水果"),("这个苹果很好吃，而且设计也很漂亮","不确定")],
    "纸": [("打印机没纸了","科技"),("这张纸的质量很好","科技"),("把协议落实到纸面上","科技"),("论文的摘要写在一张纸上","科技"),("用纸包住花束","不确定"),("纸抽用完了","不确定")],
    "光": [("太阳的光很强烈","自然"),("光在水面发生折射","自然"),("钱都花光了","语言"),("光说不做","语言"),("他为国争光","语言"),("光的速度是每秒钟三十万公里","自然")],
    "花": [("花园里的花开了","自然"),("这盆花需要浇水","自然"),("花了很多钱","生活"),("花钱如流水","生活"),("花时间学习","生活"),("花开得很鲜艳","自然")],
    "行": [("往前走三行","生活"),("行，就这样吧","语言"),("这个办法行不通","语言"),("他在银行工作","科技"),("一行代码","科技")],
    "口": [("他张开了口","生活"),("入口在左边","生活"),("三口之家","生活"),("伤口很深","不确定")],
}

print(f"\n{'='*60}")
print("P0-B-1: HowNet 义原消歧")
print("="*60)

total_correct = 0; total_all = 0
for word, cases in TEST_CASES.items():
    correct = 0; fails = []
    for sent, expected in cases:
        result, scores, found = disambiguate(sent, word)
        match = result == expected
        if match: correct += 1
        else: fails.append((sent, expected, result, scores))
    rate = correct/len(cases)*100
    total_correct += correct; total_all += len(cases)
    print(f"\n「{word}」: {correct}/{len(cases)} ({rate:.0f}%)")
    for sent, exp, got, scores in fails[:3]:
        score_str = ", ".join(f"{k}={v:.0f}" for k,v in scores.items())
        print(f"  ❌ 预期={exp:4s} → {got:4s}  [{score_str}]")

print(f"\n{'='*60}")
print(f"总计: {total_correct}/{total_all} ({total_correct/total_all*100:.0f}%)")
print("对比: 手写概念域=50%, 布尔4轴=50%")
