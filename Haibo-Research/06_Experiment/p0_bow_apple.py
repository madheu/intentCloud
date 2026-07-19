"""Phase 0: 词袋实验 — 同一语境下的词是否自然相似？
====================================================
问题：海波 2.0 想从 Token 关系中涌现语义。
      但前提是：语义信号在数据中本身就存在。
      
方法：不需要任何图系统，只用 sklearn 做词袋 + PCA。
      如果词袋就能分开"苹果科技"和"苹果水果"两类，
      说明语义信号存在于词的共现统计中，值得用图去捕捉。
      如果词袋都分不开，图也救不了。
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances

# ── 语料 ──
# 25 句科技类（Apple Inc.）
TECH = [
    "苹果发布了新款iPhone",
    "苹果开发者大会在六月举行",
    "苹果的M4芯片性能大幅提升",
    "苹果Vision Pro头显正式发售",
    "苹果应用商店的审核规则更新",
    "苹果和谷歌在AI领域竞争激烈",
    "苹果的生态系统非常封闭",
    "苹果手表销量持续增长",
    "苹果音乐服务用户突破一亿",
    "苹果的供应链遍布全球",
    "苹果发布会邀请了多家媒体",
    "苹果手机在中国市场份额下滑",
    "苹果的隐私保护政策受到称赞",
    "苹果正在研发折叠屏设备",
    "苹果电脑的MacBook Air轻薄便携",
    "苹果AirPods成为最受欢迎的耳机",
    "苹果服务业务收入创新高",
    "苹果在设计上一直追求极简风格",
    "苹果的零售店体验是行业标杆",
    "苹果健康功能可以监测心率",
    "苹果正在加大对原创内容的投入",
    "苹果的芯片设计是核心竞争力",
    "苹果产品定价策略引发讨论",
    "苹果在中国市场的表现影响股价",
    "苹果正在开发智能家居产品",
]

# 25 句水果类（apple fruit）
FRUIT = [
    "苹果是一种常见的水果",
    "红富士苹果又甜又脆",
    "苹果树需要充足的阳光才能结果",
    "苹果富含维生素C和膳食纤维",
    "今天在菜市场买了几斤苹果",
    "苹果可以做成苹果派和苹果酱",
    "秋天的苹果园挂满了果实",
    "苹果汁是很好的天然饮品",
    "苹果削皮后容易氧化变黄",
    "这个品种的苹果酸甜适中",
    "苹果在冰箱里可以保存更久",
    "苹果采摘的季节是秋天",
    "苹果树开花时非常漂亮",
    "每天一个苹果医生远离我",
    "苹果的营养价值很高",
    "苹果被榨汁后果渣可以做肥料",
    "苹果在水果中销量一直排名靠前",
    "苹果的种植技术已经很成熟",
    "苹果从开花到成熟需要几个月",
    "苹果的颜色有红色绿色和黄色",
    "山西的苹果品质非常好",
    "苹果放在阴凉处可以保存很久",
    "苹果的含糖量因品种而异",
    "苹果树需要定期修剪",
    "水果店的苹果按斤卖",
]

# ── 构建数据集 ──
texts = TECH + FRUIT
labels = ["科技"] * len(TECH) + ["水果"] * len(FRUIT)

# ── TF-IDF 向量化 ──
vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 2), max_features=500)
X = vectorizer.fit_transform(texts).toarray()  # (50, 500)
print(f"TF-IDF 矩阵形状: {X.shape}")

# ── PCA 降维到 2D ──
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)
explained = pca.explained_variance_ratio_.sum()
print(f"前 2 个主成分解释方差: {explained:.2%}")

# ── 类间/类内距离 ──
tech_vecs = X_2d[:25]
fruit_vecs = X_2d[25:]

inner_tech = np.mean(cosine_distances(tech_vecs))
inner_fruit = np.mean(cosine_distances(fruit_vecs))
cross = np.mean(cosine_distances(tech_vecs, fruit_vecs))

print(f"\n类内距离（科技）: {inner_tech:.4f}")
print(f"类内距离（水果）: {inner_fruit:.4f}")
print(f"类间距离:          {cross:.4f}")
print(f"类间/类内比:       {cross / ((inner_tech + inner_fruit) / 2):.2f}")

if cross > max(inner_tech, inner_fruit):
    print("\n✅ 结论：类间距离 > 类内距离 → 两类自然可分 → 语义信号存在")
    print("   海波 2.0 的 Token Graph 方向有基础。继续。")
else:
    print("\n❌ 结论：类间距离 ≈ 类内距离 → 两类不可分 → 语义不在词共现中")
    print("   海波 2.0 的方向需要重新审视。")

# ── 打印每类的质心坐标（用于后续对比实际图系统的输出）──
tech_center = tech_vecs.mean(axis=0)
fruit_center = fruit_vecs.mean(axis=0)
print(f"\n科技类质心: ({tech_center[0]:.4f}, {tech_center[1]:.4f})")
print(f"水果类质心: ({fruit_center[0]:.4f}, {fruit_center[1]:.4f})")
