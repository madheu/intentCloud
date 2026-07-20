"""生成 P0-R 可视化图"""
from PIL import Image, ImageDraw, ImageFont
import os

OUT_DIR = r"E:\intentCloud\assets"
os.makedirs(OUT_DIR, exist_ok=True)

def get_font(size):
    try: return ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", size)
    except: return ImageFont.load_default()

# ============================================================
#   P0-R 5层基线柱状图
# ============================================================
W, H = 800, 400
img = Image.new('RGB', (W, H), '#0f0f23')
draw = ImageDraw.Draw(img)
ft = get_font(16)
fs = get_font(12)

layers = ["MFS", "HowNet\n第一义项", "复合词\n优先", "词性\n过滤", "义原\n重叠", "级联\n组合"]
dev  = [50.0, 47.4, 47.4, 50.0, 47.4, 50.0]
blind = [55.0, 46.7, 46.7, 46.7, 46.7, 58.3]

draw.text((20, 10), "P0-R 5层基线评测", fill='#c8d6e5', font=ft)
draw.text((20, 32), "开发集(38句) vs 盲测集(60句)", fill='#5a6a7e', font=fs)

chart_x, chart_y = 80, 60
bar_w, gap = 24, 8
chart_h = 260

# 网格
for pct in [40, 45, 50, 55, 60]:
    y = chart_y + chart_h - int((pct-35) / 30 * chart_h)
    draw.line([(chart_x, y), (W-30, y)], fill='#1e1e3a', width=1)
    draw.text((10, y-6), f"{pct}%", fill='#5a6a7e', font=fs)

for i in range(len(layers)):
    gx = chart_x + i * (bar_w*2 + gap + 20)
    # 开发集
    bh = int((dev[i]-35) / 30 * chart_h)
    draw.rectangle([gx, chart_y+chart_h-bh, gx+bar_w, chart_y+chart_h], fill='#4a9eff' if dev[i]!=max(dev) else '#e94560')
    if dev[i] > 0:
        draw.text((gx-8, chart_y+chart_h-bh-16), f"{dev[i]:.1f}%", fill='#4a9eff', font=fs)
    # 盲测集
    bh2 = int((blind[i]-35) / 30 * chart_h)
    draw.rectangle([gx+bar_w+gap, chart_y+chart_h-bh2, gx+bar_w*2+gap, chart_y+chart_h], fill='#2ecc71' if blind[i]!=max(blind) else '#e94560')
    if blind[i] > 0:
        draw.text((gx+bar_w+gap-4, chart_y+chart_h-bh2-16), f"{blind[i]:.1f}%", fill='#2ecc71', font=fs)
    # X标签
    lines = layers[i].split('\n')
    for j, ln in enumerate(lines):
        lw = draw.textlength(ln, font=get_font(10))
        draw.text((gx + bar_w + gap//2 - lw//2, chart_y+chart_h+6 + j*14), ln, fill='#8899aa', font=get_font(10))

# 图例
lx, ly = chart_x, chart_y + chart_h + 42
draw.rectangle([lx, ly, lx+12, ly+12], fill='#4a9eff')
draw.text((lx+16, ly-2), "开发集", fill='#c8d6e5', font=fs)
draw.rectangle([lx+80, ly, lx+92, ly+12], fill='#2ecc71')
draw.text((lx+96, ly-2), "盲测集", fill='#c8d6e5', font=fs)
draw.rectangle([lx+160, ly, lx+172, ly+12], fill='#e94560')
draw.text((lx+176, ly-2), "该层最高", fill='#c8d6e5', font=fs)

img.save(os.path.join(OUT_DIR, "chart_p0r_layers.png"))
print("✓ chart_p0r_layers.png")

# ============================================================
#   P0 全系列 vs P0-R 对比
# ============================================================
W2, H2 = 750, 380
img2 = Image.new('RGB', (W2, H2), '#0f0f23')
draw2 = ImageDraw.Draw(img2)

draw2.text((20, 10), "P0 系列 → P0-R 进化", fill='#c8d6e5', font=ft)
draw2.text((20, 32), "正确的评估设计让信号从噪声中分离", fill='#5a6a7e', font=fs)

# 时间线箭头
draw2.line([(60, 200), (W2-40, 200)], fill='#2a2a4a', width=3)

# 节点
nodes = [
    ("P0 手写\n概念域", "45%", "#4a9eff", 80),
    ("P0-A-2\n自动扩展", "50%", "#4a9eff", 180),
    ("P0-A-5\n布尔4轴", "50%", "#4a9eff", 280),
    ("⚠ P0-R\n修复评测", "MFS 55%", "#f39c12", 420),
    ("P0-R\n级联组合", "58.3%", "#2ecc71", 560),
]

for label, val, color, x in nodes:
    y_top = 120
    draw2.rounded_rectangle([x-40, y_top, x+40, y_top+40], radius=6, fill='#111128', outline=color, width=2)
    lines = label.split('\n')
    for j, ln in enumerate(lines):
        lw = draw2.textlength(ln, font=get_font(11))
        draw2.text((x-lw//2, y_top+6+j*14), ln, fill=color, font=get_font(11))
    # 值
    vw = draw2.textlength(val, font=get_font(10))
    draw2.text((x-vw//2, y_top+42), val, fill='#c8d6e5', font=get_font(10))

# 注释
draw2.text((160, 260), "MFS基线 = 52.6%\n14轮实验未超过", fill='#e74c3c', font=get_font(10))
draw2.text((420, 260), "新建60句盲测集\n修复目标污染", fill='#f39c12', font=get_font(10))
draw2.text((520, 260), "级联 > MFS\n+3.3pp信号", fill='#2ecc71', font=get_font(10))

img2.save(os.path.join(OUT_DIR, "chart_p0_to_p0r.png"))
print("✓ chart_p0_to_p0r.png")

# ============================================================
#   研究阶段全景图
# ============================================================
W3, H3 = 780, 300
img3 = Image.new('RGB', (W3, H3), '#0f0f23')
draw3 = ImageDraw.Draw(img3)

draw3.text((20, 10), "海波项目全景进展", fill='#c8d6e5', font=ft)

phases = [
    ("Haibo 1.0\n注入路线", "关闭", "#e74c3c", "embedding注入\n被实验证伪", 60),
    ("P0 系列\n7轮词级实验", "完成", "#e74c3c", "90°正交\n词向量不行", 190),
    ("P0-A 系列\n6轮消歧实验", "完成", "#f39c12", "50%正确率\n≤MFS基线", 320),
    ("P0-R 阶段\n评测修复", "✅ 通过", "#2ecc71", "58.3%\n级联>MFS", 480),
    ("下一步\nSemantic Graph", "待定", "#4a9eff", "BriLLM底座\n+语义涌现", 630),
]

for label, status, color, detail, x in phases:
    # 圆
    draw3.ellipse([x, 60, x+16, 76], fill=color)
    # 标题
    lines = label.split('\n')
    for j, ln in enumerate(lines):
        lw = draw3.textlength(ln, font=get_font(11))
        draw3.text((x+8-lw//2, 84+j*15), ln, fill=color, font=get_font(11))
    # 状态
    sw = draw3.textlength(status, font=get_font(10))
    draw3.text((x+8-sw//2, 116), status, fill=color, font=get_font(10))
    # 详情
    dl = detail.split('\n')
    for j, dln in enumerate(dl):
        dw = draw3.textlength(dln, font=get_font(10))
        draw3.text((x+8-dw//2, 146+j*14), dln, fill='#5a6a7e', font=get_font(10))

# 连线
for i in range(len(phases)-1):
    draw3.line([(phases[i][4]+16, 68), (phases[i+1][4], 68)], fill='#2a2a4a', width=2)

img3.save(os.path.join(OUT_DIR, "chart_roadmap.png"))
print("✓ chart_roadmap.png")
print(f"\n全部3张图生成完成: {OUT_DIR}")
