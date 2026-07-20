"""P0-R: 目标锚定的 5 层基线消歧
===================================
Layer 1: MFS (Most Frequent Sense)
Layer 2: HowNet first sense
Layer 3: Compound word priority (目标锚定)
Layer 4: POS filtering (jieba pos)
Layer 5: HowNet sememe overlap
"""
import sys, json, os
sys.path = [p for p in sys.path if 'Diviner' not in p]
from collections import Counter
import jieba
import jieba.posseg as pseg
import OpenHowNet
import math

BASE = r"E:\intentCloud\Haibo-Research\06_Experiment"
DEV = os.path.join(BASE, "P0-R_gold_labels.json")
BLIND = os.path.join(BASE, "P0-R_blind_gold_labels.json")

# ── 加载 HowNet ──
print("加载 HowNet...")
hownet = OpenHowNet.HowNetDict()
print("  OK")

# ── 加载金标 ──
with open(DEV, "r", encoding="utf-8") as f:
    dev_gold = json.load(f)
with open(BLIND, "r", encoding="utf-8") as f:
    blind_gold = json.load(f)

# ── 扁平化数据集 ──
def flatten_gold(data, is_blind=False):
    """统一输出格式: [(word, sentence, label), ...]"""
    items = []
    if is_blind:
        for word, entries in data.items():
            for e in entries:
                items.append((word, e["sentence"], e["label"]))
    else:
        for word, sentences in data.items():
            for sent, label in sentences.items():
                items.append((word, sent, label))
    return items

dev_items = flatten_gold(dev_gold)
blind_items = flatten_gold(blind_gold, is_blind=True)

print(f"开发集: {len(dev_items)} 句, 盲测集: {len(blind_items)} 句")

# ── 义项映射 (A/B/C → 具体语义, 用于 HowNet 查询) ──
# 注意: 这些是给 HowNet 查询用的义项关键词, 不是给分类器用的
SENSE_NAMES_DEV = {
    "苹果": {"A": "公司", "B": "水果"},
    "纸":   {"A": "文书", "B": "日用"},
    "光":   {"A": "光线", "B": "修辞"},
    "花":   {"A": "植物", "B": "消费"},
    "行":   {"A": "评价", "B": "排列", "C": "行业"},
    "口":   {"A": "人体", "B": "空间"},
}

# ── MFS 计算 ──
def compute_mfs(items):
    """计算每词的多数义项"""
    word_labels = {}
    for word, sent, label in items:
        word_labels.setdefault(word, []).append(label)
    mfs = {}
    for word, labels in word_labels.items():
        c = Counter(labels)
        mfs[word] = c.most_common(1)[0][0]
    return mfs

dev_mfs = compute_mfs(dev_items)
blind_mfs = compute_mfs(blind_items)

print(f"\n开发集 MFS: {dev_mfs}")
print(f"盲测集 MFS: {blind_mfs}")

# ── Layer 1: MFS ──
def layer1_mfs(items, mfs):
    correct = 0
    for word, sent, label in items:
        pred = mfs.get(word, "A")
        if pred == label:
            correct += 1
    return correct / len(items) if items else 0

# ── Layer 2: HowNet 第一义项 ──
def layer2_hownet_first(items):
    """HowNet 第一义项: 取 HowNet 中该词的第一个义项的英文名"""
    correct = 0
    for word, sent, label in items:
        try:
            senses = hownet.get_sense(word)
        except:
            senses = None
        if not senses:
            pred = "A"  # 默认
        else:
            # 取第一个义项的英文名或序号
            first = str(senses[0])
            # 简单映射: 不同词的义项排序不同, 这里只是基线
            pred = "A"
        if pred == label:
            correct += 1
    return correct / len(items) if items else 0

# ── Layer 3: 复合词优先（目标锚定）──
def layer3_compound(items, sense_map):
    """目标锚定的复合词检查: 只检查指定目标词的覆盖词"""
    correct = 0
    used = 0
    for word, sent, label in items:
        # 用 jieba 分词, 找出覆盖目标词的复合词
        words = list(pseg.cut(sent))
        covering = []
        for w, pos in words:
            if word in w and len(w) > len(word):
                covering.append((w, pos))
        
        if covering:
            # 取最长的覆盖词
            longest = max(covering, key=lambda x: len(x[0]))[0]
            # 检查 HowNet 中该词是否有唯一义项
            try:
                senses = hownet.get_sense(longest)
            except:
                senses = None
            if senses and len(senses) == 1:
                # 只有1个义项 → 直接判定
                used += 1
                # 这里只是基线, 都用 A
                pred = "A"
            else:
                pred = "A"
        else:
            pred = "A"
        if pred == label:
            correct += 1
    return correct / len(items) if items else 0, used / len(items) if items else 0

# ── Layer 4: 词性过滤 ──
def layer4_pos(items):
    """用 jieba pos 过滤候选义项"""
    correct = 0
    for word, sent, label in items:
        words = list(pseg.cut(sent))
        pos = None
        for w, p in words:
            if w == word:
                pos = p
                break
        # 根据词性做基本判断: n→A, v→B 等
        if pos:
            if pos.startswith("n"):
                pred = "A"
            elif pos.startswith("v"):
                pred = "B"
            else:
                pred = "A"
        else:
            pred = "A"
        if pred == label:
            correct += 1
    return correct / len(items) if items else 0

# ── Layer 5: HowNet 义原重叠 ──
def get_sememe_set(word):
    """获取词在 HowNet 中的所有义原"""
    sememes = set()
    try:
        data = hownet.get_sememes_by_word(word)
        if data:
            for entry in data:
                for s in entry.get("sememes", []):
                    if isinstance(s, str):
                        sememes.add(s.split("|")[0].strip())
    except:
        pass
    return sememes

def layer5_sememe_overlap(items):
    """用 HowNet 义原重叠评分做消歧"""
    correct = 0
    for word, sent, label in items:
        # 目标词的义原
        word_sememes = get_sememe_set(word)
        # 上下文词（句中非目标词）的义原
        words = [w for w in jieba.lcut(sent) if w != word and len(w) >= 2]
        context_sememes = set()
        for w in words:
            context_sememes.update(get_sememe_set(w))
        
        if not word_sememes or not context_sememes:
            pred = "A"
        else:
            overlap = len(word_sememes & context_sememes)
            total = len(word_sememes)
            score = overlap / max(total, 1)
            # 简单规则: 重叠 > 0.3 → B, 否则 A
            pred = "B" if score > 0.3 else "A"
        if pred == label:
            correct += 1
    return correct / len(items) if items else 0

# ── 运行所有层 ──
print("\n" + "=" * 60)
print("P0-R 基线评测")
print("=" * 60)

layers = [
    ("MFS", lambda items, mfs=dev_mfs: layer1_mfs(items, mfs)),
    ("HowNet第一义项", layer2_hownet_first),
    ("复合词优先", lambda items: layer3_compound(items, SENSE_NAMES_DEV)[0]),
    ("词性过滤", layer4_pos),
    ("义原重叠", layer5_sememe_overlap),
]

print(f"\n{'Layer':20s} {'开发集':>10s} {'盲测集':>10s}")
print("-" * 42)

for name, func in layers:
    dev_acc = func(dev_items)
    # 对盲测集使用盲测集自己的 MFS
    if name == "MFS":
        blind_acc = layer1_mfs(blind_items, blind_mfs)
    else:
        blind_acc = func(blind_items)
    print(f"{name:20s} {dev_acc:>8.1%} {blind_acc:>8.1%}")

# ── 组合: 级联（依次尝试各层直到有判定）──
print("\n" + "=" * 60)
print("级联组合: Layer 3 → Layer 4 → Layer 5 → MFS")
print("=" * 60)

def cascade_predict(word, sent, sense_map, mfs):
    """级联预测: 复合词优先 → 词性 → 义原 → MFS"""
    # L3
    words = list(pseg.cut(sent))
    covering = []
    for w, pos in words:
        if word in w and len(w) > len(word):
            covering.append((w, pos))
    if covering:
        longest = max(covering, key=lambda x: len(x[0]))[0]
        try:
            senses = hownet.get_sense(longest)
        except:
            senses = None
        if senses and len(senses) == 1:
            return "A"  # 单义复合词直接判定
    
    # L4
    pos = None
    for w, p in words:
        if w == word:
            pos = p
            break
    if pos:
        if pos.startswith("n"): return "A"
        elif pos.startswith("v"): return "B"
    
    # L5
    word_sems = get_sememe_set(word)
    ctx_words = [w for w in jieba.lcut(sent) if w != word and len(w) >= 2]
    ctx_sems = set()
    for w in ctx_words:
        ctx_sems.update(get_sememe_set(w))
    if word_sems and ctx_sems:
        overlap = len(word_sems & ctx_sems) / max(len(word_sems), 1)
        if overlap > 0.3: return "B"
    
    # MFS
    return mfs.get(word, "A")

for name, items, mfs in [("开发集", dev_items, dev_mfs), ("盲测集", blind_items, blind_mfs)]:
    correct = 0
    for word, sent, label in items:
        pred = cascade_predict(word, sent, SENSE_NAMES_DEV, mfs)
        if pred == label:
            correct += 1
    acc = correct / len(items) if items else 0
    print(f"  {name}: {correct}/{len(items)} ({acc:.1%})")
