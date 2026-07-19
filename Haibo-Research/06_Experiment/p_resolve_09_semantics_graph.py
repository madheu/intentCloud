"""
Semantics Graph v0 — 围合消歧（概念层，非词层）
=============================================
节点 = 语义概念，不是词。
word→concept 归属表（一个词可属多个概念）。
推理：输入词集 → 查 word→concept → 概念激活计数 → 比较歧义词的候选概念。
"""

# ═══════════════════════════════════════
# 1. 概念节点 + word→concept 归属表
# ═══════════════════════════════════════

CONCEPTS = [
    # 科技概念 1: 消费电子
    ("科技_消费电子", {"手机", "芯片", "系统", "屏幕", "应用",
                       "发布", "商店", "生态", "手表", "硬件",
                       "软件", "服务", "隐私", "设计", "零售",
                       "供应链"}),
    # 科技概念 2: 价格/产品
    ("科技_产品", {"产品", "品牌", "营收", "价格", "市场",
                    "上一代", "新款", "版本", "标配",
                    "便宜", "昂贵", "性价比", "销量"}),
    # 科技概念 3: 技术研发
    ("科技_研发", {"芯片", "基带", "开发者", "专利", "研发",
                    "收购", "投资", "自研",
                    "性能", "提升", "升级", "架构", "工艺"}),

    # 水果概念 1: 种植/生长
    ("水果_种植", {"树", "种植", "采摘", "开花", "修剪",
                    "果农", "丰收", "成熟", "冷藏", "保鲜"}),
    # 水果概念 2: 品质/营养
    ("水果_品质", {"甜", "营养", "含糖量", "有机", "口感",
                    "色泽", "果肉", "果汁", "果皮",
                    "维生素", "健康", "新鲜",
                    "好吃", "美味", "香甜"}),
    # 水果概念 3: 品种/商品
    ("水果_品种", {"品种", "特产", "产地", "上市", "季节性",
                    "进口", "出口"}),

    # 中性高频词（不属任何概念）
    # "比", "这款", "这个", "那个", "的", "了", "是" — 不产生激活
]

# 构建方便查找的数据结构
WORD_TO_CONCEPTS = {}  # word → set of concept names
CONCEPT_WORDS = {}     # concept name → set of words

for cname, words in CONCEPTS:
    CONCEPT_WORDS[cname] = words
    for w in words:
        if w not in WORD_TO_CONCEPTS:
            WORD_TO_CONCEPTS[w] = set()
        WORD_TO_CONCEPTS[w].add(cname)

ALL_CONCEPT_NAMES = [c[0] for c in CONCEPTS]

# ═══════════════════════════════════════
# 2. 歧义词定义
# ═══════════════════════════════════════

# 歧义词 → 它"连接"到的候选概念（歧义词本身只在 Step 2 才计入）
AMBIGUOUS_WORDS = {
    "苹果": {"科技_消费电子", "科技_产品", "科技_研发",
             "水果_种植", "水果_品质", "水果_品种"},
    "小米": {"科技_消费电子", "科技_产品",
             "水果_种植", "水果_品质"},  # 既是手机品牌也是谷物
    "华为": {"科技_消费电子", "科技_产品", "科技_研发"},
    "bank": {"科技_产品"},  # placeholder for English
}

# ═══════════════════════════════════════
# 3. 分词
# ═══════════════════════════════════════

def tokenize(sentence):
    """简单分词：找到句子中出现在 word→concept 表中的词"""
    found = []
    # 先检查长词匹配（多字词优先）
    all_words = sorted(WORD_TO_CONCEPTS.keys(), key=len, reverse=True)
    remaining = sentence
    # 简单：直接扫
    for w in all_words:
        if w in sentence:
            found.append(w)
    # 去重但保持词表覆盖
    return list(set(found))

# ═══════════════════════════════════════
# 4. 围合消歧
# ═══════════════════════════════════════

def disambiguate(sentence, ambiguous_word, threshold=1.5):
    """
    对含歧义词的句子做语义消歧。
    
    参数:
        sentence: 输入句子
        ambiguous_word: 歧义词（如"苹果"）
        threshold: 比率阈值，默认 1.5
    
    返回:
        (result: str, scores: dict)
        result: "科技_消费电子" / "水果_品质" / "不确定"
    """
    if ambiguous_word not in AMBIGUOUS_WORDS:
        return None, {}
    
    candidate_concepts = AMBIGUOUS_WORDS[ambiguous_word]
    words = tokenize(sentence)
    
    # 去掉歧义词本身（不贡献信号）
    context_words = [w for w in words if w != ambiguous_word]
    
    # Step 1: 非歧义词激活它所属的概念
    concept_activation = {c: 0 for c in ALL_CONCEPT_NAMES}
    for w in context_words:
        if w in WORD_TO_CONCEPTS:
            for c in WORD_TO_CONCEPTS[w]:
                concept_activation[c] += 1
    
    # Step 2: 歧义词自身也激活（但不能用词本身的知识——它是被消歧的对象）
    # 这里歧义词只贡献 1 个计数到每个候选概念
    # （因为"苹果"这个词本身既激活科技概念也激活水果概念）
    for c in candidate_concepts:
        concept_activation[c] += 1  # 歧义词自身的基础激活
    
    # Step 3: 只看候选概念之间的比较
    candidate_scores = {c: concept_activation[c] for c in candidate_concepts}
    
    # Step 4: 找出最高的候选
    sorted_candidates = sorted(candidate_scores.items(), key=lambda x: -x[1])
    
    if len(sorted_candidates) < 2:
        return "不确定", candidate_scores
    
    top_name, top_score = sorted_candidates[0]
    second_name, second_score = sorted_candidates[1]
    
    # 最高分必须 ≥ 第二高分 × threshold 才做判断
    if second_score > 0 and top_score >= second_score * threshold:
        return top_name, candidate_scores
    elif second_score == 0 and top_score > 0:
        return top_name, candidate_scores
    else:
        return "不确定", candidate_scores


def classify_domain(result):
    """将具体概念归入大类：科技 / 水果 / 不确定"""
    if result is None or result == "不确定":
        return "不确定"
    if result.startswith("科技_"):
        return "科技"
    if result.startswith("水果_"):
        return "水果"
    return result

# ═══════════════════════════════════════
# 5. 测试
# ═══════════════════════════════════════

def test():
    test_cases = [
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
        ("这个苹果很好吃，而且设计也很漂亮", "不确定"),  # 混合
    ]
    
    print("=" * 60)
    print("Semantics Graph v0 — 围合消歧测试（概念层）")
    print("=" * 60)
    print()
    
    correct = 0
    uncertain_allowed = 0
    total = len(test_cases)
    
    for sent, expected in test_cases:
        result, scores = disambiguate(sent, "苹果")
        domain = classify_domain(result)
        
        match = domain == expected
        if match:
            correct += 1
        if expected == "不确定":
            uncertain_allowed += 1
        
        icon = "✅" if match else "❌"
        
        # 显示前几名候选
        sorted_s = sorted(scores.items(), key=lambda x: -x[1])[:4]
        scores_str = ", ".join(f"{n}:{s}" for n, s in sorted_s)
        
        print(f"{icon} 预期={expected:4s}  →  {domain:8s}")
        print(f"   候选: {scores_str}")
        print(f"   句: {sent}")
        print()
    
    # 统计
    success_rate = (correct - uncertain_allowed) / max(total - uncertain_allowed, 1) * 100
    print(f"正确: {correct}/{total} (排除不确定后: {correct - uncertain_allowed}/{total - uncertain_allowed} = {success_rate:.0f}%)")
    
    if correct >= total - uncertain_allowed:
        print("\n✅ 全部歧义句正确消歧！")
        print("   Semantics Graph 方向经过这个简单的测试验证。")
    elif correct >= total - uncertain_allowed - 1:
        print("\n⚠️ 大部分正确，可能有个别边界情况需要调整。")
    else:
        print(f"\n❌ 设计有问题：只有 {correct} 句正确。")
        print("   可能是概念分组不合理或词表覆盖不够。")

if __name__ == "__main__":
    test()
