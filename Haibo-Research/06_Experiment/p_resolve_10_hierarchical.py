"""
P_Resolve-10: 分层比对验证 — "四维一次比对，超出就拓层"
=======================================================
核心想法：不把 6 个概念拉平了比。分层。
每次只比 2-4 个，如果平局就展开下一层。
"""

# ═════════════════════════════
# 1. 概念分层结构
# ═════════════════════════════

# 第一层：大类（2个）
DOMAINS = {
    "科技": ["科技_消费电子", "科技_产品", "科技_研发"],
    "水果": ["水果_种植", "水果_品质", "水果_品种"],
}

# 第二层：子概念 → 归属词
SUB_CONCEPTS = {
    "科技_消费电子": {"手机", "芯片", "系统", "屏幕", "应用",
                      "发布", "商店", "生态", "手表", "硬件",
                      "软件", "服务", "隐私", "设计", "零售",
                      "供应链"},
    "科技_产品": {"产品", "品牌", "营收", "价格", "市场",
                  "上一代", "新款", "版本", "标配",
                  "便宜", "昂贵", "性价比", "销量"},
    "科技_研发": {"芯片", "基带", "开发者", "专利", "研发",
                  "收购", "投资", "自研",
                  "性能", "提升", "升级", "架构", "工艺"},

    "水果_种植": {"树", "种植", "采摘", "开花", "修剪",
                  "果农", "丰收", "成熟", "冷藏", "保鲜"},
    "水果_品质": {"甜", "营养", "含糖量", "有机", "口感",
                  "色泽", "果肉", "果汁", "果皮",
                  "维生素", "健康", "新鲜",
                  "好吃", "美味", "香甜"},
    "水果_品种": {"品种", "特产", "产地", "上市", "季节性",
                  "进口", "出口"},
}

# 中性词（不属任何概念）
NEUTRAL = {"比", "这款", "这个", "那个", "的", "了", "是", "很", "也", "而且"}

# 歧义词
AMBIGUOUS = {
    "苹果": {"科技", "水果"},  # 歧义词只标记它属于哪些大类，不标记子概念
}

# 构建 word → sub_concepts 的映射
WORD_TO_SUB = {}
for concept, words in SUB_CONCEPTS.items():
    for w in words:
        if w not in WORD_TO_SUB:
            WORD_TO_SUB[w] = []
        WORD_TO_SUB[w].append(concept)

# ═════════════════════════════
# 2. 分层推理
# ═════════════════════════════

def tokenize(sentence):
    """找句子中的所有已知词"""
    found = set()
    # 先长后短匹配
    all_words = sorted(WORD_TO_SUB.keys(), key=len, reverse=True)
    for w in all_words:
        if w in sentence:
            found.add(w)
    # 加上歧义词
    for aw in AMBIGUOUS:
        if aw in sentence:
            found.add(aw)
    return found


def layer1_compare(sentence):
    """
    第一层：比大类（科技 vs 水果）
    返回 (winner_domain, scores, is_decided)
    """
    words = tokenize(sentence)
    
    # 大类激活
    domain_scores = {d: 0 for d in DOMAINS}
    
    for w in words:
        if w in NEUTRAL:
            continue
        if w in AMBIGUOUS:
            # 歧义词：激活它标记的所有大类（各 +1）
            for d in AMBIGUOUS[w]:
                domain_scores[d] += 1
        elif w in WORD_TO_SUB:
            # 非歧义词：看它属于哪个子概念，子概念归哪个大类
            for sub_c in WORD_TO_SUB[w]:
                for domain, subs in DOMAINS.items():
                    if sub_c in subs:
                        domain_scores[domain] += 1
    
    # 判断
    sorted_d = sorted(domain_scores.items(), key=lambda x: -x[1])
    top_name, top_score = sorted_d[0]
    second_score = sorted_d[1][1] if len(sorted_d) > 1 else 0
    
    threshold = 1.5
    if second_score > 0 and top_score >= second_score * threshold:
        return top_name, domain_scores, True
    elif second_score == 0 and top_score > 0:
        return top_name, domain_scores, True
    else:
        return None, domain_scores, False


def layer2_compare(sentence, tied_domains):
    """
    第二层：在平局的大类下，比子概念
    返回 (winner_sub_concept, sub_scores, is_decided)
    """
    words = tokenize(sentence)
    
    # 只计算 tied_domains 下的子概念
    sub_scores = {}
    for d in tied_domains:
        for sub in DOMAINS[d]:
            sub_scores[sub] = 0
    
    for w in words:
        if w in AMBIGUOUS:
            # 歧义词：激活它标记的大类下的所有子概念
            for d in AMBIGUOUS[w]:
                if d in tied_domains:
                    for sub in DOMAINS[d]:
                        sub_scores[sub] += 1
        elif w in WORD_TO_SUB:
            for sub_c in WORD_TO_SUB[w]:
                if sub_c in sub_scores:
                    sub_scores[sub_c] += 1
    
    if not sub_scores:
        return None, sub_scores, False
    
    sorted_s = sorted(sub_scores.items(), key=lambda x: -x[1])
    top_name, top_score = sorted_s[0]
    second_score = sorted_s[1][1] if len(sorted_s) > 1 else 0
    
    threshold = 1.5
    if second_score > 0 and top_score >= second_score * threshold:
        return top_name, sub_scores, True
    elif second_score == 0 and top_score > 0:
        return top_name, sub_scores, True
    else:
        return None, sub_scores, False


def resolve(sentence):
    """完整分层推理"""
    print(f"句: {sentence}")
    print()
    
    # Layer 1
    winner, scores1, decided1 = layer1_compare(sentence)
    print(f"  [第一层] 大类激活: ", end="")
    for d, s in sorted(scores1.items(), key=lambda x: -x[1]):
        print(f"{d}={s} ", end="")
    print()
    
    if decided1:
        print(f"  ✅ 第一层已决定 → {winner}")
        return winner
    
    # Layer 1 平局 → 展开
    tied = [d for d, s in scores1.items() if s > 0]
    print(f"  ⚠️ 平局（比率 < 1.5），展开: {tied}")
    print()
    
    # Layer 2
    winner2, scores2, decided2 = layer2_compare(sentence, tied)
    print(f"  [第二层] 子概念激活: ", end="")
    for s, sc in sorted(scores2.items(), key=lambda x: -x[1]):
        print(f"{s}={sc} ", end="")
    print()
    
    if decided2:
        # 映射回大类
        for d, subs in DOMAINS.items():
            if winner2 in subs:
                print(f"  ✅ 第二层决定 → {d}（{winner2}）")
                return d
        print(f"  ✅ 第二层决定 → {winner2}")
        return winner2
    else:
        print(f"  ❌ 第二层仍平局，无法决定")
        return "不确定"


# ═════════════════════════════
# 3. 测试
# ═════════════════════════════

def test():
    test_cases = [
        ("这个苹果比上一代便宜了五百块", "科技"),
        ("苹果发布了新款手机", "科技"),
        ("今年苹果的芯片性能提升很大", "科技"),
        ("苹果的屏幕显示效果很好", "科技"),
        ("苹果应用商店的规则更新了", "科技"),
        ("这个苹果比上一代更甜", "水果"),  # ← 之前失败的那个
        ("苹果的采摘季节到了", "水果"),
        ("今年的苹果果肉很甜", "水果"),
        ("苹果正在开花", "水果"),
        ("苹果的含糖量很高", "水果"),
        ("这个苹果很好吃，而且设计也很漂亮", "不确定"),
    ]
    
    print("=" * 60)
    print("P_Resolve-10: 分层比对验证")
    print("=" * 60)
    print()
    
    correct = 0
    total = len(test_cases)
    layer1_only = 0
    
    for sent, expected in test_cases:
        result = resolve(sent)
        match = result == expected
        if match:
            correct += 1
        icon = "✅" if match else "❌"
        
        # 判断是第一层还是第二层决定的
        # 重新走一遍来判断层级（有点冗余但清晰）
        _, _, d1 = layer1_compare(sent)
        
        print(f" {icon} 预期={expected} → 结果={result}")
        if d1:
            layer1_only += 1
            print(f"    第一层直接解决")
        else:
            print(f"    需要第二层展开")
        print()
    
    print(f"正确: {correct}/{total}（{correct/total*100:.0f}%）")
    print(f"第一层解决: {layer1_only}/{total}")
    print(f"需要第二层: {total - layer1_only}/{total}")


if __name__ == "__main__":
    test()
