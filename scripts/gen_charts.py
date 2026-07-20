"""生成海波项目预览图"""
from PIL import Image, ImageDraw, ImageFont
import os

OUT_DIR = r"E:\intentCloud\assets"
os.makedirs(OUT_DIR, exist_ok=True)

# 尝试加载字体
def get_font(size):
    try:
        return ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", size)  # 微软雅黑
    except:
        try:
            return ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", size)
        except:
            return ImageFont.load_default()

# ============================================================
# 图1: 方法对比柱状图
# ============================================================
W, H = 900, 460
img = Image.new('RGB', (W, H), '#0f0f23')
draw = ImageDraw.Draw(img)
font_t = get_font(18)
font_s = get_font(13)
font_xs = get_font(11)

# 数据
methods = ["手写概念域", "自动扩展", "布尔4轴"]
words = ["苹果", "纸", "光", "花", "行", "口"]
data = [
    [82, 73, 45],   # 苹果
    [17, 17, 83],   # 纸
    [33, 17, 50],   # 光
    [33, 83, 50],   # 花
    [0,  20, 40],   # 行
    [75, 75, 25],   # 口
]
colors = ['#4a9eff', '#f39c12', '#2ecc71']

# 标题
draw.text((20, 12), "各方法消歧正确率对比", fill='#c8d6e5', font=font_t)
draw.text((20, 36), "P0-A 系列 | 6 歧义词 × 3 方法", fill='#5a6a7e', font=font_xs)

# 柱状图区域
chart_x, chart_y = 80, 80
bar_w = 28
gap = 6
group_gap = 30
chart_h = 300

# 网格线
for pct in range(0, 101, 20):
    y = chart_y + chart_h - int(pct / 100 * chart_h)
    draw.line([(chart_x, y), (W - 30, y)], fill='#1e1e3a', width=1)
    draw.text((10, y - 7), f"{pct}%", fill='#5a6a7e', font=font_xs)

# 柱状图
for gi, word in enumerate(words):
    gx = chart_x + gi * (bar_w * 3 + gap * 2 + group_gap)
    best_val = max(data[gi])
    for mi, val in enumerate(data[gi]):
        x = gx + mi * (bar_w + gap)
        bh = int(val / 100 * chart_h)
        cy = chart_y + chart_h - bh
        is_best = (val == best_val and val > 0)
        c = colors[mi] if not is_best else '#e94560'
        draw.rectangle([x, cy, x + bar_w, chart_y + chart_h], fill=c)
        # 数值
        if val > 0:
            draw.text((x + 2, cy - 16), f"{val}%", fill='#c8d6e5' if not is_best else '#e94560', font=font_xs)

# X 轴标签
for gi, word in enumerate(words):
    gx = chart_x + gi * (bar_w * 3 + gap * 2 + group_gap) + (bar_w * 3 + gap * 2) // 2 - 12
    draw.text((gx, chart_y + chart_h + 6), word, fill='#c8d6e5', font=font_s)

# 图例
lx = chart_x
ly = chart_y + chart_h + 40
for mi, m in enumerate(methods):
    draw.rectangle([lx, ly, lx + 14, ly + 14], fill=colors[mi])
    draw.text((lx + 20, ly - 2), m, fill='#c8d6e5', font=font_xs)
    lx += 140

img.save(os.path.join(OUT_DIR, "chart_comparison.png"))
print(f"✓ chart_comparison.png")

# ============================================================
# 图2: 研究路径演化图
# ============================================================
W2, H2 = 800, 500
img2 = Image.new('RGB', (W2, H2), '#0f0f23')
draw2 = ImageDraw.Draw(img2)

draw2.text((20, 12), "研究路径演化", fill='#c8d6e5', font=font_t)
draw2.text((20, 36), "Haibo 1.0 → P0 → P0-A → 布尔向量", fill='#5a6a7e', font=font_xs)

# 节点
nodes = [
    ("Haibo 1.0", "36节点语义图\nembedding注入LLM", (150, 120), "#e74c3c"),
    ("P0 系列", "词级向量空间\n7 轮实验 90°正交", (350, 120), "#e74c3c"),
    ("Token Graph", "词节点+共现边\n全部不确定", (100, 280), "#f39c12"),
    ("Semantics Graph", "概念节点+分层\n10/11 (91%)", (300, 280), "#2ecc71"),
    ("自动扩展", "词向量扩词表\n总 50% → 花 83%", (500, 280), "#2ecc71"),
    ("布尔 4 轴", "天/地/人/心\n纸 83% 光 50%", (640, 280), "#4a9eff"),
]

for label, desc, (x, y), color in nodes:
    # 圆角矩形
    r = 10
    bw, bh = 130, 70
    draw2.rounded_rectangle([x, y, x + bw, y + bh], radius=r, fill='#111128', outline=color, width=2)
    # 标题
    tw = draw2.textlength(label, font=font_xs)
    draw2.text((x + (bw - tw) // 2, y + 6), label, fill=color, font=font_xs)
    # 描述
    desc_lines = desc.split('\n')
    for li, line in enumerate(desc_lines):
        dw = draw2.textlength(line, font=font_xs)
        draw2.text((x + (bw - dw) // 2, y + 26 + li * 18), line, fill='#8899aa', font=font_xs)

# 连线
lines = [
    ((215, 190), (350, 120)),    # Haibo 1.0 → P0
    ((215, 190), (100, 280)),    # Haibo 1.0 → Token Graph
    ((415, 190), (300, 280)),    # P0 → Semantics Graph (off-ramp)
    ((165, 350), (300, 280)),    # Token → Semantics
    ((365, 350), (500, 280)),    # Semantics → 自动扩展
    ((565, 350), (640, 280)),    # 自动扩展 → 布尔
]

for (x1, y1), (x2, y2) in lines:
    draw2.line([(x1, y1), (x2, y2)], fill='#2a2a4a', width=2)

img2.save(os.path.join(OUT_DIR, "chart_evolution.png"))
print(f"✓ chart_evolution.png")

# ============================================================
# 图3: 4轴布尔向量示意图
# ============================================================
W3, H3 = 700, 420
img3 = Image.new('RGB', (W3, H3), '#0f0f23')
draw3 = ImageDraw.Draw(img3)

draw3.text((20, 12), "布尔语义 4 轴", fill='#c8d6e5', font=font_t)
draw3.text((20, 36), "天 · 地 · 人 · 心 — 每个词用 4 bits 编码", fill='#5a6a7e', font=font_xs)

# 4个轴区域
axes = [
    ("天", "自然现象 · 物理规律 · 光线", "#4a9eff", 60, 90),
    ("地", "物理实体 · 物体 · 空间 · 物质", "#2ecc71", 230, 90),
    ("人", "人类活动 · 社会关系 · 技术 · 人造物", "#f39c12", 400, 90),
    ("心", "情感 · 评价 · 审美 · 认知", "#e94560", 570, 90),
]

for label, desc, color, x, y in axes:
    # 轴框
    draw3.rounded_rectangle([x, y, x + 140, y + 70], radius=8, fill='#111128', outline=color, width=2)
    # 大字
    lw = draw3.textlength(label, font=get_font(28))
    draw3.text((x + (140 - lw) // 2, y + 6), label, fill=color, font=get_font(28))
    # 描述
    dw = draw3.textlength(desc, font=get_font(10))
    draw3.text((x + (140 - dw) // 2, y + 48), desc, fill='#5a6a7e', font=get_font(10))

# 示例词
examples = [
    ("太阳: [1,1,0,0]", "天+地", (100, 200)),
    ("手机: [0,1,1,0]", "地+人", (250, 200)),
    ("甜:   [0,0,0,1]", "心", (400, 200)),
    ("花(消费): [0,0,1,0]", "人", (100, 270)),
    ("花(植物): [0,1,0,1]", "地+心", (250, 270)),
    ("光(光线): [1,1,0,0]", "天+地", (400, 270)),
    ("光(修辞): [0,0,1,1]", "人+心", (100, 340)),
    ("苹果(科技): [0,1,1,0]", "地+人", (250, 340)),
    ("苹果(水果): [0,1,0,1]", "地+心", (400, 340)),
]

for text, tag, (x, y) in examples:
    draw3.rounded_rectangle([x, y, x + 145, y + 28], radius=4, fill='#111128', outline='#2a2a4a', width=1)
    draw3.text((x + 6, y + 4), text, fill='#c8d6e5', font=font_xs)
    tw = draw3.textlength(tag, font=get_font(10))
    draw3.text((x + 145 - tw - 6, y + 5), tag, fill='#5a6a7e', font=get_font(10))

img3.save(os.path.join(OUT_DIR, "chart_boolean_4axis.png"))
print(f"✓ chart_boolean_4axis.png")

print(f"\n全部图片生成完成: {OUT_DIR}")
