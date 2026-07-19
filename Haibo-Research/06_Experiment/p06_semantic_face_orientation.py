"""P0.6：语义面的朝向 — 词的向量集形成的平面的方向
=====================================================
原初直觉（P0.5 测错了，测试了从苹果指向句子的方向）：
语义面由词构成。科技面（苹果+手机+芯片+...）和水果面（苹果+树+甜+...）
是两个不同的平面。它们的朝向不同——这是区分的方式。

修正后：
对每个语义面，取构成该面的所有词的 char n-gram 向量，
计算这些向量的主成分方向（PC1 即平面的主朝向）。
比较两个面的 PC1 夹角。
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ── 语义面：词构成的集合 ──
# 注意：这些是词，不是句子。
# "苹果"出现在两个面中——这是交线。

TECH_FACE = [
    "苹果", "手机", "芯片", "系统", "屏幕",
    "应用", "发布", "商店", "生态", "手表",
    "音乐", "服务", "隐私", "设计", "零售",
    "头显", "基带", "芯片设计", "供应链",
    "开发者", "软件", "硬件", "产品",
    "销售", "市场", "品牌", "营收",
]

FRUIT_FACE = [
    "苹果", "树", "甜", "采摘", "种植",
    "营养", "果汁", "果农", "园", "丰收",
    "品种", "冷藏", "果胶", "苹果醋", "果皮",
    "成熟", "开花", "修剪", "套袋", "含糖量",
    "有机", "汁", "果酱", "冷冻",
    "口感", "色泽", "保鲜",
]

# ── 单独出现的词（用来验证面向量是否来自语义而非共现）──
CONTROL_FACE_WHALE = [
    "鲸鱼", "海洋", "哺乳", "呼吸", "迁徙",
    "蓝鲸", "虎鲸", "座头鲸", "喷水", "回声",
    "须鲸", "齿鲸", "捕食", "搁浅", "寿命",
]

CONTROL_FACE_CAR = [
    "汽车", "发动机", "轮胎", "刹车", "悬挂",
    "电动", "变速", "涡轮", "底盘", "气囊",
    "方向盘", "制动", "减震", "车漆", "照明",
]

# ── 向量化 ──
# 把所有词一起向量化，然后在同一空间中取每个面的子集
all_words = TECH_FACE + FRUIT_FACE + CONTROL_FACE_WHALE + CONTROL_FACE_CAR
vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 2))
X = vec.fit_transform(all_words).toarray()

n_tech = len(TECH_FACE)
n_fruit = len(FRUIT_FACE)

X_tech = X[:n_tech]
X_fruit = X[n_tech:n_tech+n_fruit]
X_whale = X[n_tech+n_fruit:n_tech+n_fruit+len(CONTROL_FACE_WHALE)]
X_car = X[n_tech+n_fruit+len(CONTROL_FACE_WHALE):]

def get_face_orientation(vectors):
    """对一个面的词向量集，计算主朝向（PC1）"""
    pca = PCA(n_components=3)
    pca.fit(vectors)
    # PC1 是方差最大的方向 = 这个面的主朝向
    pc1 = pca.components_[0]  # 长度为 hidden_dim
    # 方差解释率
    variance_ratio = pca.explained_variance_ratio_[0]
    return pc1, variance_ratio, pca

def angle_between(v1, v2):
    """两个向量之间的夹角"""
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

# ── 分析 ──
print("=" * 60)
print("语义面朝向分析（词向量集的主成分方向）")
print("=" * 60)
print()

# 各面朝向
pc1_tech, var_tech, _ = get_face_orientation(X_tech)
pc1_fruit, var_fruit, _ = get_face_orientation(X_fruit)
pc1_whale, var_whale, _ = get_face_orientation(X_whale)
pc1_car, var_car, _ = get_face_orientation(X_car)

print("各面 PC1 方差解释率（越大说明面越平整）：")
print(f"  科技面: {var_tech:.2%}")
print(f"  水果面: {var_fruit:.2%}")
print(f"  (控制)鲸鱼面: {var_whale:.2%}")
print(f"  (控制)汽车面: {var_car:.2%}")
print()

# 面朝向两两夹角
a_tf = angle_between(pc1_tech, pc1_fruit)
a_tw = angle_between(pc1_tech, pc1_whale)
a_tc = angle_between(pc1_tech, pc1_car)
a_fw = angle_between(pc1_fruit, pc1_whale)
a_fc = angle_between(pc1_fruit, pc1_car)
a_wc = angle_between(pc1_whale, pc1_car)

print("面主朝向（PC1）夹角（°）：")
print(f"  科技面 vs 水果面: {a_tf:.1f}°  ← 这是你要测的")
print(f"  科技面 vs 鲸鱼面: {a_tw:.1f}°")
print(f"  科技面 vs 汽车面: {a_tc:.1f}°")
print(f"  水果面 vs 鲸鱼面: {a_fw:.1f}°")
print(f"  水果面 vs 汽车面: {a_fc:.1f}°")
print(f"  鲸鱼面 vs 汽车面: {a_wc:.1f}°  ← 完全不相关面的夹角基线")
print()

print("─" * 50)
print("判断：")
if a_tf > 30:
    print(f"✅ 科技面 vs 水果面 = {a_tf:.0f}° > 30° → 朝向不同，可区分")
elif a_tf > 15:
    print(f"⚠️ 科技面 vs 水果面 = {a_tf:.0f}° → 有差异但不大")
else:
    print(f"❌ 科技面 vs 水果面 = {a_tf:.0f}° → 面朝向几乎相同")

print()
if a_tf > a_wc:
    print(f"   歧义词面的朝向差异（{a_tf:.0f}°）> 无关面差异（{a_wc:.0f}°）→ 语义面朝向优于随机")
else:
    print(f"   歧义词面的朝向差异（{a_tf:.0f}°）< 无关面差异（{a_wc:.0f}°）→ 朝向差异部分来自随机")
