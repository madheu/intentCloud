"""诊断：Token Graph 边为什么全零"""
import sys
sys.path = [p for p in sys.path if 'Diviner' not in p]
from collections import defaultdict

NODES = [
    "苹果", "手机", "芯片", "系统", "屏幕", "应用",
    "发布", "商店", "生态", "手表", "服务", "隐私",
    "设计", "零售", "供应链", "软件", "硬件",
    "树", "甜", "采摘", "种植", "营养", "果汁",
    "果农", "丰收", "品种", "冷藏", "成熟",
    "开花", "修剪", "含糖量", "有机",
    "这款", "比", "上一代", "便宜", "更", "还是",
]
vocab = set(NODES)

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

# 检查：每个句子中找到多少 vocab 词
print("=== 句子词汇覆盖率 ===")
for label, sents in [("科技", TECH_SENTS), ("水果", FRUIT_SENTS)]:
    for i, s in enumerate(sents):
        found = [w for w in vocab if w in s]
        if len(found) >= 2:
            print(f"  [{label} {i}] +{len(found)}词: {found} | {s[:40]}")

# 建图
edges = defaultdict(float)
for sents in [TECH_SENTS, FRUIT_SENTS]:
    for sent in sents:
        words = [w for w in sent if w in vocab]
        for i, w1 in enumerate(words):
            for w2 in words[i + 1:]:
                if w1 != w2:
                    key = tuple(sorted([w1, w2]))
                    edges[key] += 1
max_w = max(edges.values()) if edges else 1
for k in edges:
    edges[k] /= max_w

print(f"\n=== 总边数: {len(edges)} ===")
print(f"最大权重: {max_w}")

# 苹果的邻居
apple_edges = {}
for (w1, w2), w in edges.items():
    if w1 == "苹果": apple_edges[w2] = w
    elif w2 == "苹果": apple_edges[w1] = w

print(f"\n=== 苹果的邻居: {len(apple_edges)} 个 ===")
for n, w in sorted(apple_edges.items(), key=lambda x: -x[1])[:15]:
    print(f"  苹果 ─ {n}: {w:.2f}")
