"""Resolve-07：词向量空间下的语义面朝向（P0系列终局）
=============================================================
P_Resolve-06 在 char n-gram 空间中失败了——高维诅咒让所有面正交。
现在换到 200 维密集词向量（Tencent Chinese Embedding），
期望：语义面在密集空间中不再正交，有真实朝向差异。

如果词向量空间也分不开 → 面朝向在当前词级粒度下不成立。
如果词向量空间能分开 → Resolve 系列结束，进入 Token Graph 设计。
"""

import numpy as np
import os
import sys
from pathlib import Path

# ── 配置 ──
EMBEDDING_BIN = r"E:\intentCloud\models\text2vec-word2vec-tencent-chinese\light_Tencent_AILab_ChineseEmbedding.bin"

# ── 语义面（与 P_Resolve-06 相同的词集合）──
TECH_FACE = [
    "苹果", "手机", "芯片", "系统", "屏幕",
    "应用", "发布", "商店", "生态", "手表",
    "音乐", "服务", "隐私", "设计", "零售",
    "头显", "基带", "供应链", "开发者",
    "软件", "硬件", "产品", "销售", "市场", "品牌", "营收",
]

FRUIT_FACE = [
    "苹果", "树", "甜", "采摘", "种植",
    "营养", "果汁", "果农", "园", "丰收",
    "品种", "冷藏", "果胶", "苹果醋", "果皮",
    "成熟", "开花", "修剪", "套袋", "含糖量",
    "有机", "汁", "果酱", "冷冻", "口感", "色泽", "保鲜",
]

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

FACES = {
    "科技": TECH_FACE,
    "水果": FRUIT_FACE,
    "鲸鱼": CONTROL_FACE_WHALE,
    "汽车": CONTROL_FACE_CAR,
}


def load_tencent_embeddings():
    """加载腾讯中文词向量（.bin 格式）"""
    if not os.path.exists(EMBEDDING_BIN):
        print("=" * 60)
        print("需要下载腾讯中文词向量模型")
        print("=" * 60)
        print()
        print(f"期望路径: {EMBEDDING_BIN}")
        print()
        print("下载命令：")
        print(f"  huggingface-cli download shibing624/text2vec-word2vec-tencent-chinese --local-dir {os.path.dirname(EMBEDDING_BIN)}")
        print()
        sys.exit(1)

    print(f"加载: {EMBEDDING_BIN}")
    from gensim.models import KeyedVectors
    kv = KeyedVectors.load_word2vec_format(EMBEDDING_BIN, binary=True)
    embeddings = {word: kv[word] for word in kv.index_to_key}
    print(f"  共 {len(embeddings)} 词，每词 {kv.vector_size} 维")
    return embeddings


def get_face_vectors(face_words: list, embeddings: dict) -> np.ndarray:
    """获取一个面的词向量矩阵（只取在词表中存在的词）"""
    vectors = []
    found = 0
    missing = []
    for w in face_words:
        if w in embeddings:
            vectors.append(embeddings[w])
            found += 1
        else:
            missing.append(w)
    if missing:
        print(f"  ⚠️ 未找到的词: {missing}")
    print(f"  找到 {found}/{len(face_words)} 个词")
    return np.array(vectors)


def face_orientation(vectors: np.ndarray) -> tuple:
    """PC1 方向向量 + 方差解释率"""
    mean = vectors.mean(axis=0)
    centered = vectors - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # 取最大特征值对应的特征向量（PC1）
    pc1 = eigvecs[:, -1]
    var_ratio = eigvals[-1] / eigvals.sum()
    return pc1, var_ratio


def angle_between(v1, v2):
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


# ── 主逻辑 ──
def main():
    embeddings = load_tencent_embeddings()
    dim = list(embeddings.values())[0].shape[0]
    print(f"  嵌入维度: {dim}")
    print()

    # 各面的词向量
    face_vectors = {}
    for name, words in FACES.items():
        print(f"面「{name}」:")
        vecs = get_face_vectors(words, embeddings)
        if len(vecs) < 3:
            print(f"  ❌ 词太少（{len(vecs)}个），跳过")
            continue
        face_vectors[name] = vecs

    if len(face_vectors) < 2:
        print("❌ 至少需要两个面")
        return

    print()
    print("=" * 60)
    print("各面朝向（PC1）分析")
    print("=" * 60)
    print()

    orients = {}
    for name, vecs in face_vectors.items():
        pc1, var = face_orientation(vecs)
        orients[name] = (pc1, var)
        print(f"  {name}面: PC1 方差解释率 = {var:.2%}")

    print()
    print("─" * 50)
    print("面朝向两两夹角（°）：")
    print("─" * 50)

    names = list(face_vectors.keys())
    angles = {}
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            n1, n2 = names[i], names[j]
            ang = angle_between(orients[n1][0], orients[n2][0])
            angles[(n1, n2)] = ang

    for (n1, n2), ang in angles.items():
        mark = ""
        if n1 == "科技" and n2 == "水果":
            mark = "  ← 歧义面"
        elif n1 == "鲸鱼" and n2 == "汽车":
            mark = "  ← 无关面（控制基线）"
        print(f"  {n1:4s} vs {n2:4s}: {ang:6.1f}°{mark}")

    print()
    print("=" * 60)
    print("判断：")

    tf_angle = angles.get(("科技", "水果"), None)
    wc_angle = angles.get(("鲸鱼", "汽车"), None)

    if tf_angle is None:
        print("  ❌ 缺少歧义面对比")
        return

    if wc_angle is None:
        print("  ❌ 缺少控制组对比")
        return

    if tf_angle > 45:
        print(f"  ✅ 科技 vs 水果 = {tf_angle:.0f}° → 朝向差异显著，区分可行")
    elif tf_angle > 20:
        print(f"  ⚠️ 科技 vs 水果 = {tf_angle:.0f}° → 有差异但中等")
    else:
        print(f"  ❌ 科技 vs 水果 = {tf_angle:.0f}° → 朝向几乎相同，不可区分")
        print("     词向量空间也分不开→面朝向假设在词级粒度不成立")

    if tf_angle > wc_angle * 1.2:
        print(f"  ✅ 歧义面差异（{tf_angle:.0f}°）> 无关面差异（{wc_angle:.0f}°）→ 含真实语义信号")
    elif tf_angle > wc_angle * 0.8:
        print(f"  ⚠️ 歧义面差异 ≈ 无关面差异 → 信号弱")
    else:
        print(f"  ⚠️ 歧义面差异 < 无关面差异 → 控制组反而更大，信号被噪声覆盖")

    if tf_angle > 45 and tf_angle > wc_angle * 1.2:
        print()
        print("✅✅ 面朝向假设成立。Resolve 系列结束。进入 Token Graph 设计。")
    else:
        print()
        print("⚠️ 面朝向假设在词向量空间未通过。")
        print("   Resolve 系列结论：词级粒度不足以区分语义歧义。")
        print("   可能需要更细粒度的 Token Graph 级信号。")


if __name__ == "__main__":
    main()
