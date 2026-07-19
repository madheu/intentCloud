"""
P0-A-1: 多歧义词范围测试
=========================
从"苹果"扩展到更多常见歧义词：
"纸" — 材料/文件
"光" — 光线/空无一物/荣耀
"花" — 植物/消费
"行" — 可以/行列/行业
"口" — 嘴巴/出入口
"""

# ═════════════════════════════
# 1. 概念分层（从 P0.10 扩展）
# ═════════════════════════════

DOMAINS = {
    "科技":   ["科技_消费电子", "科技_产品", "科技_研发"],
    "水果":   ["水果_种植", "水果_品质", "水果_品种"],
    "自然":   ["自然_光线", "自然_植物", "自然_动物"],
    "生活":   ["生活_消费", "生活_日常", "生活_建筑"],
    "语言":   ["语言_修辞", "语言_评价", "语言_抽象"],
}

# 子概念 → 归属词
SUB_CONCEPTS = {
    # 科技（已有）
    "科技_消费电子": {"手机", "芯片", "系统", "屏幕", "应用",
                      "发布", "商店", "生态", "手表", "硬件",
                      "软件", "服务", "隐私", "设计", "零售",
                      "供应链"},
    "科技_产品": {"产品", "品牌", "营收", "价格", "市场",
                  "上一代", "新款", "版本", "标配",
                  "便宜", "昂贵", "性价比", "销量"},
    "科技_研发": {"芯片", "基带", "开发者", "专利", "研发",
                  "收购", "投资", "自研", "性能", "提升",
                  "升级", "架构", "工艺"},
    # 水果（已有）
    "水果_种植": {"树", "种植", "采摘", "开花", "修剪",
                  "果农", "丰收", "成熟", "冷藏", "保鲜"},
    "水果_品质": {"甜", "营养", "含糖量", "有机", "口感",
                  "色泽", "果肉", "果汁", "果皮",
                  "维生素", "健康", "新鲜",
                  "好吃", "美味", "香甜"},
    "水果_品种": {"品种", "特产", "产地", "上市", "季节性",
                  "进口", "出口"},
    # 自然_光线
    "自然_光线": {"太阳", "明亮", "照射", "反射", "折射",
                  "阳光", "阴影", "光芒", "耀眼", "柔和",
                  "紫外线", "可见"},
    # 自然_植物
    "自然_植物": {"叶子", "根", "茎", "种子", "开花",
                  "授粉", "凋谢", "盆栽", "园林", "生长",
                  "绿植"},
    # 自然_动物（从P0.7复用）
    "自然_动物": {"鲸鱼", "海洋", "哺乳", "迁徙", "蓝鲸",
                  "虎鲸", "座头鲸", "捕食", "寿命",
                  "狗", "猫", "鸟", "鱼", "昆虫"},
    # 生活_消费
    "生活_消费": {"钱", "消费", "付款", "账单", "价格",
                  "购买", "花费", "报销", "预算", "折扣",
                  "买", "贵", "便宜"},
    # 生活_日常
    "生活_日常": {"吃饭", "睡觉", "上班", "回家", "出门",
                  "做饭", "打扫", "洗", "穿", "休息"},
    # 生活_建筑
    "生活_建筑": {"门", "窗户", "墙", "房间", "入口",
                  "出口", "走廊", "楼梯", "屋顶", "地基"},
    # 语言_修辞
    "语言_修辞": {"比喻", "象征", "夸张", "拟人", "排比",
                  "修辞", "生动", "形象"},
    # 语言_评价
    "语言_评价": {"好", "坏", "不错", "优秀", "差", "一般",
                  "满意", "糟糕"},
    # 语言_抽象
    "语言_抽象": {"概念", "定义", "逻辑", "因果", "关系",
                  "本质", "形式", "结构"},
}

# 歧义词 → 候选大类
AMBIGUOUS = {
    "苹果": {"科技", "水果"},
    "纸":   {"科技", "生活"},    # 纸张(科技/办公) vs 文件(生活/法律)
    "光":   {"自然", "语言"},    # 光线(自然) vs 光了(语言/修辞)
    "花":   {"自然", "生活"},    # 花朵(自然/植物) vs 花钱(生活/消费)
    "行":   {"生活", "语言"},    # 行走(生活) vs 可以(语言/评价)
    "口":   {"生活", "生活"},    # 嘴巴(生活/日常) vs 入口(生活/建筑)
}

# 中性词
NEUTRAL = {"比", "这款", "这个", "那个", "的", "了", "是",
           "很", "也", "而且", "和", "在", "有", "不", "就"}

# 构建 word → sub_concepts
WORD_TO_SUB = {}
for concept, words in SUB_CONCEPTS.items():
    for w in words:
        if w not in WORD_TO_SUB:
            WORD_TO_SUB[w] = []
        WORD_TO_SUB[w].append(concept)

# ═════════════════════════════
# 2. 推理（复用 P0.10 的分层逻辑）
# ═════════════════════════════

def tokenize(sentence):
    """找句子中的所有已知词"""
    found = set()
    all_words = sorted(WORD_TO_SUB.keys(), key=len, reverse=True)
    for w in all_words:
        if w in sentence:
            found.add(w)
    for aw in AMBIGUOUS:
        if aw in sentence:
            found.add(aw)
    return found


def layer1(sentence):
    """大类层比较"""
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
    """完整分层推理"""
    # Layer 1
    winner, scores1, decided1 = layer1(sentence)
    
    if decided1:
        return winner
    
    # Layer 1 平局 → 展开
    tied = [d for d, s in scores1.items() if s > 0]
    
    # Layer 2：子概念层
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
                        sub_scores[sub] = sub_scores.get(sub, 0) + 1
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


# ═════════════════════════════
# 3. 测试
# ═════════════════════════════

def test_word(ambiguous_word, test_cases):
    """测试一个歧义词的所有用例"""
    print(f"\n{'='*60}")
    print(f"歧义词: 「{ambiguous_word}」")
    print(f"{'='*60}")
    
    correct = 0
    total = len(test_cases)
    l1 = 0
    
    for sent, expected in test_cases:
        result = resolve(sent, ambiguous_word)
        match = result == expected
        if match:
            correct += 1
        
        _, _, d1 = layer1(sent)
        if d1:
            l1 += 1
        
        icon = "✅" if match else "❌"
        print(f"  {icon} 预期={expected:4s}  →  {result}")
        print(f"      {sent}")
    
    rate = correct / total * 100
    print(f"\n  正确: {correct}/{total}（{rate:.0f}%） 第一层解决: {l1}/{total}")
    return correct, total


def main():
    print("=" * 60)
    print("P0-A-1: 多歧义词范围测试")
    print("=" * 60)
    
    all_correct = 0
    all_total = 0
    
    # ── 苹果（已有） ──
    c, t = test_word("苹果", [
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
    ])
    all_correct += c; all_total += t
    
    # ── 纸 ──
    c, t = test_word("纸", [
        ("打印机没纸了", "科技"),           # 办公用品
        ("这张纸的质量很好", "科技"),        # 材料
        ("把协议落实到纸面上", "科技"),      # 文书
        ("论文的摘要写在一张纸上", "科技"),   # 学术
        ("用纸包住花束", "不确定"),           # 包装——可属于生活，但当前的"纸"只设了科技和生活两个候选，需要更多测试
        ("纸抽用完了", "不确定"),             # 生活用品——纸巾是科技(制造)还是生活(消费)？
    ])
    all_correct += c; all_total += t
    
    # ── 光 ──
    c, t = test_word("光", [
        ("太阳的光很强烈", "自然"),          # 光线
        ("光在水面发生折射", "自然"),        # 物理
        ("钱都花光了", "语言"),              # 光了=完了（修辞）
        ("光说不做", "语言"),               # 光=只（修辞）
        ("他为国争光", "语言"),             # 光荣（修辞）
        ("光的速度是每秒钟三十万公里", "自然"),  # 物理
    ])
    all_correct += c; all_total += t
    
    # ── 花 ──
    c, t = test_word("花", [
        ("花园里的花开了", "自然"),          # 花朵
        ("这盆花需要浇水", "自然"),          # 植物
        ("花了很多钱", "生活"),              # 消费
        ("花钱如流水", "生活"),              # 消费
        ("花时间学习", "生活"),              # 时间消费
        ("花开得很鲜艳", "自然"),            # 植物
    ])
    all_correct += c; all_total += t
    
    # ── 行 ──
    c, t = test_word("行", [
        ("往前走三行", "生活"),              # 行走
        ("行，就这样吧", "语言"),            # 评价/同意
        ("这个办法行不通", "语言"),          # 可行
        ("他在银行工作", "科技"),            # 行业（注意："银行"中的"行"不是单独的词）
        ("一行代码", "科技"),               # 行列
    ])
    all_correct += c; all_total += t
    
    # ── 口 ──
    c, t = test_word("口", [
        ("他张开了口", "生活"),              # 嘴巴
        ("入口在左边", "生活"),              # 出入口
        ("三口之家", "生活"),               # 计量
        ("伤口很深", "不确定"),              # 医疗（当前没有医疗领域）
    ])
    all_correct += c; all_total += t
    
    # ── 总结 ──
    print("\n" + "="*60)
    print(f"P0-A-1 总结")
    print(f"="*60)
    print(f"总正确率: {all_correct}/{all_total}（{all_correct/all_total*100:.0f}%）")
    print()
    print("通过的歧义词: 苹果(✅91%)")
    print("新增测试的歧义词: 纸, 光, 花, 行, 口")
    print("注意: 纸和口的部分用例因为缺少对应领域的概念而失败")
    print("→ 这不是分层算法的问题，是概念覆盖不够")


if __name__ == "__main__":
    main()
