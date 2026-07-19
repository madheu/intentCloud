"""实验：歧义词的"朝向"能否区分语义？
======================================
核心直觉：同一歧义词（如"苹果"）在不同义项语境中，
          从该词指向语境质心的向量方向不同。
          
方法：不用预训练词向量，只用我们已有的 TF-IDF 矩阵。
      1. 对每句含"苹果"的句子，计算句中所有词的 TF-IDF 向量均值作为"语境向量"
      2. 计算"苹果"本身的 TF-IDF 向量
      3. 语境向量 - 苹果向量 = "从苹果指向语境的方向向量"
      4. 比较科技类方向向量 vs 水果类方向向量
      -> 如果两类方向向量的夹角显著 > 0 且类内夹角 < 类间夹角 → 朝向成立了
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
from sklearn.decomposition import PCA
import numpy as np

# ── 语料（维基百科来源，P0.3 的苹果 + 鲸鱼/汽车控制组）──
APPLE_TECH = [
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

APPLE_FRUIT = [
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

# ── 工具函数 ──
def get_tfidf_vectors(texts, apple_texts_a, apple_texts_b):
    """把苹果句子、科技文本、水果文本一起向量化"""
    all_texts = texts + apple_texts_a + apple_texts_b
    vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 2), max_features=500)
    X = vec.fit_transform(all_texts).toarray()
    
    # 取每类区段
    n_texts = len(texts)
    X_texts = X[:n_texts]
    X_a = X[n_texts:n_texts+len(apple_texts_a)]
    X_b = X[n_texts+len(apple_texts_a):n_texts+len(apple_texts_a)+len(apple_texts_b)]
    
    return X_a, X_b, X_texts, vec

# ── 获取“苹果”这个词本身的向量 ──
# 我们手写一个含"苹果"单独出现的句子来近似
apple_solo = "苹果"
solo_sentences = ["苹果。"]

# 整合所有语料
tech_texts = APPLE_TECH
fruit_texts = APPLE_FRUIT

all_texts = tech_texts + fruit_texts + solo_sentences
vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 2), max_features=500)
X = vec.fit_transform(all_texts).toarray()

n_tech = len(tech_texts)
n_fruit = len(fruit_texts)

X_tech = X[:n_tech]
X_fruit = X[n_tech:n_tech+n_fruit]
X_solo = X[n_tech+n_fruit:]  # "苹果。"的向量

# 苹果向量
apple_vec = X_solo[0].reshape(1, -1)  # shape (1, 500)

# 语境质心
tech_centroid = X_tech.mean(axis=0).reshape(1, -1)  # (1, 500)
fruit_centroid = X_fruit.mean(axis=0).reshape(1, -1)

# 方向向量 = 语境质心 - 苹果向量
dir_tech = tech_centroid - apple_vec
dir_fruit = fruit_centroid - apple_vec

# 对方向向量归一化（只保留方向信息）
dir_tech_norm = dir_tech / (np.linalg.norm(dir_tech) + 1e-8)
dir_fruit_norm = dir_fruit / (np.linalg.norm(dir_fruit) + 1e-8)

# 夹角
cos_sim = np.dot(dir_tech_norm[0], dir_fruit_norm[0])
angle_rad = np.arccos(np.clip(cos_sim, -1.0, 1.0))
angle_deg = np.degrees(angle_rad)

print("=" * 60)
print("朝向实验：从「苹果」指向语境质心的向量方向")
print("=" * 60)
print()
print(f"科技面质心 - 苹果向量 的模:  {np.linalg.norm(dir_tech):.4f}")
print(f"水果面质心 - 苹果向量 的模:  {np.linalg.norm(dir_fruit):.4f}")
print()
print(f"两个方向向量的余弦相似度:    {cos_sim:.4f}")
print(f"两个方向向量的夹角:          {angle_deg:.2f}°")
print()

# ── 做一个更强的版本：不用质心，看每句的朝向分布 ──
# 每句的朝向 = 该句向量 - 苹果向量
tech_dirs = X_tech - apple_vec  # (n_tech, 500)
fruit_dirs = X_fruit - apple_vec  # (n_fruit, 500)

# 归一化
tech_dirs_norm = tech_dirs / (np.linalg.norm(tech_dirs, axis=1, keepdims=True) + 1e-8)
fruit_dirs_norm = fruit_dirs / (np.linalg.norm(fruit_dirs, axis=1, keepdims=True) + 1e-8)

# 类内朝向一致性（同组句子之间的朝向夹角均值）
inner_tech_angles = []
for i in range(len(tech_dirs_norm)):
    for j in range(i+1, len(tech_dirs_norm)):
        sim = np.dot(tech_dirs_norm[i], tech_dirs_norm[j])
        inner_tech_angles.append(np.degrees(np.arccos(np.clip(sim, -1.0, 1.0))))

inner_fruit_angles = []
for i in range(len(fruit_dirs_norm)):
    for j in range(i+1, len(fruit_dirs_norm)):
        sim = np.dot(fruit_dirs_norm[i], fruit_dirs_norm[j])
        inner_fruit_angles.append(np.degrees(np.arccos(np.clip(sim, -1.0, 1.0))))

# 类间朝向差异（跨组的句子之间的朝向夹角均值）
cross_angles = []
for i in range(len(tech_dirs_norm)):
    for j in range(len(fruit_dirs_norm)):
        sim = np.dot(tech_dirs_norm[i], fruit_dirs_norm[j])
        cross_angles.append(np.degrees(np.arccos(np.clip(sim, -1.0, 1.0))))

print(f"科技组内朝向夹角（内聚性）:     {np.mean(inner_tech_angles):.1f}°")
print(f"水果组内朝向夹角（内聚性）:     {np.mean(inner_fruit_angles):.1f}°")
print(f"跨组朝向夹角（分离度）:        {np.mean(cross_angles):.1f}°")
print()

ratio = np.mean(cross_angles) / ((np.mean(inner_tech_angles) + np.mean(inner_fruit_angles)) / 2)
print(f"跨组/组内夹角比:               {ratio:.2f}")

# ── 控制组：鲸鱼 vs 汽车用在同样的方法上 ──
WHALE = [
    "鲸鱼是哺乳动物用肺呼吸",
    "蓝鲸是地球上最大的动物体重可达一百八十吨",
    "鲸鱼通过喷水孔呼吸空气",
    "虎鲸是高度社会化的海洋哺乳动物",
    "座头鲸以其复杂的歌声闻名",
    "鲸鱼每几年迁徙数千公里",
    "鲸鱼通过回声定位感知环境",
    "鲸鱼群有复杂的社会结构",
    "抹香鲸可以潜入深海捕食巨型乌贼",
    "鲸鱼的祖先是从陆地返回海洋的",
    "须鲸通过鲸须过滤海水获取食物",
    "齿鲸用牙齿捕食鱼类和鱿鱼",
    "鲸鱼的身体流线型适应水生生活",
    "鲸鱼和人类一样需要浮出水面呼吸",
    "鲸鱼的叫声可以传播很远的距离",
    "鲸鱼在海洋生态系统中处于食物链顶端",
    "白鲸生活在北极附近的海域",
    "鲸鱼搁浅后需要人类的救助",
    "露脊鲸曾因捕鲸而濒临灭绝",
    "鲸鱼的寿命可以达到一百年以上",
]

CAR = [
    "汽车由发动机底盘车身和电气系统组成",
    "内燃机汽车主要使用汽油或柴油作为燃料",
    "自动变速箱使驾驶更加简便",
    "电动汽车使用电池组驱动电动机",
    "汽车的制动系统包括刹车盘和刹车片",
    "悬挂系统影响汽车的操控性和舒适性",
    "安全气囊在碰撞时保护乘客",
    "涡轮增压器可以提高发动机的功率输出",
    "汽车的空气动力学设计影响燃油效率",
    "四轮驱动系统适合复杂路况行驶",
    "汽车的减震器可以过滤路面的颠簸",
    "方向盘转向系统使汽车可以改变方向",
    "汽车的空调系统调节车内温度",
    "防抱死制动系统防止车轮在急刹时锁死",
    "汽车的车身结构包括车架和覆盖件",
    "汽车轮胎的胎压影响行驶安全和油耗",
    "车载信息娱乐系统集成了导航和多媒体功能",
    "汽车的灯光系统包括前照灯和尾灯",
    "发动机的活塞在气缸内往复运动",
    "汽车的车漆由底漆色漆和清漆多层组成",
]

# 控制组的"鲸鱼"和"汽车"语义朝向
ctrl_texts = WHALE + CAR + ["鲸鱼。"]
ctrl_vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 2), max_features=500)
X_ctrl = ctrl_vec.fit_transform(ctrl_texts).toarray()

n_whale = len(WHALE)
X_whale = X_ctrl[:n_whale]
X_car = X_ctrl[n_whale:n_whale+len(CAR)]
X_whale_solo = X_ctrl[n_whale+len(CAR):]

whale_vec = X_whale_solo[0].reshape(1, -1)
whale_centroid = X_whale.mean(axis=0).reshape(1, -1)
car_centroid = X_car.mean(axis=0).reshape(1, -1)

dir_whale = whale_centroid - whale_vec
dir_car = car_centroid - whale_vec

dir_whale_norm = dir_whale / (np.linalg.norm(dir_whale) + 1e-8)
dir_car_norm = dir_car / (np.linalg.norm(dir_car) + 1e-8)

cos_ctrl = np.dot(dir_whale_norm[0], dir_car_norm[0])
angle_ctrl = np.degrees(np.arccos(np.clip(cos_ctrl, -1.0, 1.0)))

print()
print("─" * 60)
print("控制组：鲸鱼 → 海洋 vs 汽车 → 机械")
print(f"两个方向向量的夹角:          {angle_ctrl:.2f}°")
print()

# ── 结果解读 ──
print("=" * 60)
print("判断：")
if angle_deg > 30:
    print(f"✅ 苹果科技 vs 水果的朝向夹角 = {angle_deg:.0f}° > 30°")
    print("   「朝不同方向走」的直觉成立。朝向可以区分语义。")
elif angle_deg > 10:
    print(f"⚠️ 苹果科技 vs 水果的朝向夹角 = {angle_deg:.0f}°")
    print("   方向有差异但不大，需要在更精确的向量空间中验证。")
else:
    print(f"❌ 苹果科技 vs 水果的朝向夹角 = {angle_deg:.0f}°")
    print("   「朝不同方向走」的直觉在当前语料中不成立。")

print()
if ratio > 1.3:
    print(f"✅ 跨组/组内夹角比 = {ratio:.2f} > 1.3，朝向可区分性强。")
else:
    print(f"⚠️ 跨组/组内夹角比 = {ratio:.2f}，朝向区分度有限。")
