"""Token Graph v0 — 围合消歧最简实现
=========================================
30 个词节点 + 共现边 → 1 步扩散 → 语境消歧
"""

import json
from collections import defaultdict

# ═══════════════════════════════════════
# 1. 词表 + 边（从 P0.3 维基百科语料统计共现）
# ═══════════════════════════════════════

NODES = [
    # 歧义词 + 科技围合词
    "苹果", "手机", "芯片", "系统", "屏幕", "应用",
    "发布", "商店", "生态", "手表", "服务", "隐私",
    "设计", "零售", "供应链", "软件", "硬件",
    # 歧义词 + 水果围合词
    "树", "甜", "采摘", "种植", "营养", "果汁",
    "果农", "丰收", "品种", "冷藏", "成熟",
    "开花", "修剪", "含糖量", "有机",
    # 中性词（跨两类）
    "这款", "比", "上一代", "便宜", "更", "还是",
]

# 科技类和水果类的维基百科句子（从 P0.3 复用）
TECH_SENTS = [
    "苹果公司2024年第四季度财报显示营收增长",
    "苹果推出了搭载M4芯片的新款MacBook Pro",
    "苹果的Vision Pro头显在开发者中引发热议",
    "苹果应用商店的抽成比例在欧洲受到监管压力",
    "苹果手表的心率监测功能帮助用户发现健康问题",
    "苹果的供应链管理是行业标杆",
    "苹果正在自研基带芯片以减少对高通的依赖",
    "苹果音乐服务已经积累了超过一亿付费用户",
    "苹果手机在中国市场面临华为和荣耀的竞争",
    "苹果的隐私保护策略限制了广告商的追踪能力",
    "苹果的iOS系统每年一次大版本更新",
    "苹果的设计语言从拟物化转向扁平化",
    "苹果的零售店每平方英尺销售额全球领先",
    "苹果正在加大在印度市场的生产布局",
    "苹果的AR眼镜项目据传已经进入试产阶段",
    "苹果的服务业务收入占比逐年上升",
    "苹果的生态系统用户粘性非常高",
    "苹果的总部Apple Park被称为太空船",
    "苹果的芯片设计团队从英特尔挖来了多名工程师",
    "苹果在AI领域的收购策略较为低调",
]

FRUIT_SENTS = [
    "烟台苹果今年迎来大丰收品质优良",
    "红富士苹果的含糖量一般在百分之十五左右",
    "苹果在冷库中可以储存六个月以上",
    "陕西洛川苹果以其色泽鲜艳口感脆甜而闻名",
    "苹果的果皮富含抗氧化物质",
    "今年春季的低温影响了苹果的开花和坐果",
    "苹果树需要冬季的低温积累才能正常开花",
    "苹果采摘后需要快速预冷以保持新鲜度",
    "苹果的品种有嘎啦富士和国光等多种",
    "有机苹果的种植成本比普通苹果高出百分之三十",
    "苹果汁加工过程中维生素C会有一定损失",
    "苹果醋被认为对健康有多种益处",
    "苹果的果胶含量有助于制作果酱和果冻",
    "新疆阿克苏苹果以其独特的冰糖心著称",
    "苹果的贮藏条件要求温度和湿度精确控制",
    "苹果树的修剪技术直接影响果实品质和产量",
    "套袋栽培可以提高苹果的外观品质",
    "苹果的病虫害防治需要综合运用多种手段",
    "苹果的国际贸易量在全球水果中排名靠前",
    "苹果从开花到成熟需要大约四到六个月",
]


def build_graph(sentences_a, sentences_b, vocab):
    """从句子统计共现边权重"""
    edges = defaultdict(float)  # (w1, w2) → weight
    for sents in [sentences_a, sentences_b]:
        for sent in sents:
            words = [w for w in vocab if w in sent]
            for i, w1 in enumerate(words):
                for w2 in words[i + 1:]:
                    if w1 != w2:
                        key = tuple(sorted([w1, w2]))
                        edges[key] += 1
    # 归一化到 0-1
    max_w = max(edges.values()) if edges else 1
    for k in edges:
        edges[k] /= max_w
    return dict(edges)


def get_neighbors(word, edges):
    """一个词的所有邻居 + 权重"""
    result = {}
    for (w1, w2), w in edges.items():
        if w1 == word:
            result[w2] = w
        elif w2 == word:
            result[w1] = w
    return result


# ═══════════════════════════════════════
# 2. 围合检测逻辑
# ═══════════════════════════════════════

def tokenize(sentence, vocab):
    """简单分词：找到句子中包含的所有词表词"""
    found = set()
    for w in vocab:
        if w in sentence:
            found.add(w)
    return found


def disambiguate(sentence, edges, vocab, tech_neighbors, fruit_neighbors, threshold=1.5):
    """对含"苹果"的句子判断语义方向"""
    words = tokenize(sentence, vocab)
    if "苹果" not in words:
        return None, None, None

    # 1 步扩散：从句中所有词出发，看它们各指向科技还是水果
    tech_score = 0.0
    fruit_score = 0.0

    for w in words:
        neighbors = get_neighbors(w, edges)
        for n, weight in neighbors.items():
            if n in tech_neighbors:
                tech_score += weight
            if n in fruit_neighbors:
                fruit_score += weight

    # 如果"苹果"本身也有非歧义邻居，算上
    apple_neighbors = get_neighbors("苹果", edges)

    if tech_score >= fruit_score * threshold:
        return "科技", tech_score, fruit_score
    elif fruit_score >= tech_score * threshold:
        return "水果", tech_score, fruit_score
    else:
        return "不确定", tech_score, fruit_score


# ═══════════════════════════════════════
# 3. 构建 + 测试
# ═══════════════════════════════════════

vocab = set(NODES)
edges = build_graph(TECH_SENTS, FRUIT_SENTS, vocab)

# 科技围合词（不含"苹果"）
tech_neighbors = {"手机", "芯片", "系统", "屏幕", "应用", "发布",
                  "商店", "生态", "手表", "服务", "隐私", "设计",
                  "零售", "供应链", "软件", "硬件"}
# 水果围合词（不含"苹果"）
fruit_neighbors = {"树", "甜", "采摘", "种植", "营养", "果汁",
                   "果农", "丰收", "品种", "冷藏", "成熟",
                   "开花", "修剪", "含糖量", "有机"}

# 测试用例（围合条件内聚 + 歧义判断）
test_cases = [
    # 预期科技
    ("这个苹果比上一代便宜了五百块", "科技"),
    ("苹果发布了新款手机", "科技"),
    ("今年苹果的芯片性能提升很大", "科技"),
    ("苹果的屏幕显示效果很好", "科技"),
    ("苹果应用商店的规则更新了", "科技"),
    # 预期水果
    ("这个苹果比上一代更甜", "水果"),
    ("苹果的采摘季节到了", "水果"),
    ("今年的苹果果肉很甜", "水果"),
    ("苹果正在开花", "水果"),
    ("苹果的含糖量很高", "水果"),
    # 边界（看是"不确定"还是偏一方）
    ("这个苹果很好吃，而且设计也很漂亮", "不确定"),  # 混搭
]

# ═══════════════════════════════════════
# 4. 结果
# ═══════════════════════════════════════

print("=" * 60)
print("Token Graph v0 — 围合消歧测试")
print("=" * 60)
print()

total = len(test_cases)
correct = 0
uncertain = 0

for sent, expected in test_cases:
    result, tech_s, fruit_s = disambiguate(sent, edges, vocab,
                                            tech_neighbors, fruit_neighbors)
    match = result == expected
    if match:
        correct += 1
    if result == "不确定":
        uncertain += 1

    icon = "✅" if match else "❌" if result is not None else " "

    # 简短显示
    if tech_s is not None:
        print(f"{icon} {result:4s}（科技={tech_s:.2f}, 水果={fruit_s:.2f}）| {sent[:40]}")
    else:
        print(f"{icon} ? | {sent[:40]}（不含歧义词）")

print()
print(f"正确: {correct}/{total}（{correct/total*100:.0f}%）")
print(f"不确定: {uncertain}/{total}")
if total - uncertain > 0:
    print(f"成功率（排除不确定）: {correct}/{total - uncertain}（{correct/(total-uncertain)*100:.0f}%）")
else:
    print("成功率（排除不确定）: N/A（全部不确定）")

# 输出边的诊断信息
print()
print("─" * 50)
print("苹果的邻居边权重（top 10）:")
apple_neighbors = get_neighbors("苹果", edges)
sorted_neighbors = sorted(apple_neighbors.items(), key=lambda x: -x[1])
for n, w in sorted_neighbors[:10]:
    camp = "科技" if n in tech_neighbors else "水果" if n in fruit_neighbors else "中性"
    print(f"  苹果 ─ {n:5s}: {w:.2f}  [{camp}]")
