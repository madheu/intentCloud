"""P0-R1: 目标锚定 5 层消歧预测 (predict 阶段)
约束: 不读取任何盲测金标文件
"""
import json, os, sys
sys.path = [p for p in sys.path if 'Diviner' not in p]
from collections import Counter
import jieba
import jieba.posseg as pseg
import OpenHowNet

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
OUTPUT = os.path.join(BASE, "output")

# ── 加载 HowNet ──
print("加载 HowNet...")
hownet = OpenHowNet.HowNetDict()
print("  OK")

# ── 加载开发集计算 MFS ──
with open(os.path.join(DATA, "dev_gold.json"), "r") as f:
    dev_gold = json.load(f)

# 计算 MFS（只能从开发集）
word_mfs = {}
for item in dev_gold:
    w = item["target_word"]
    label = item["_gold_label"]
    word_mfs.setdefault(w, []).append(label)

mfs = {}
dev_all_labels = []
for w, labels in word_mfs.items():
    c = Counter(labels)
    most = c.most_common(1)[0][0]
    mfs[w] = most
    dev_all_labels.extend(labels)

# 全局多数（用于未见词）
global_mfs = Counter(dev_all_labels).most_common(1)[0][0]
print(f"MFS 计算完成: {len(mfs)} 词, 全局多数={global_mfs}")

# ── 候选义项 → HowNet sense ID 映射（用于 L2）──
# 注意：先做简单映射，后续可以扩展
SENSE_ALIAS = {
    "科技公司": ["apple", "Apple", "公司", "科技公司"],
    "水果": ["fruit", "Fruit", "水果", "苹果"],
    "文书办公": ["paper", "Paper", "文书", "document"],
    "日用包装": ["wrap", "包装", "日用", "材料"],
    "物理光线": ["ray", "光线", "light", "物理"],
    "修辞评价": ["glory", "光荣", "修辞", "exhausted", "评价"],
    "植物": ["flower", "flower|花", "花草", "植物"],
    "消费": ["spend", "花费", "消费", "花钱"],
    "评价可以": ["behaviour", "行为", "conduct", "可以", "competent", "行"],
    "排列行列": ["line", "排", "行列", "排列"],
    "行业银行": ["bank", "银行", "profession", "行业"],
    "人体嘴巴": ["mouth", "mouth|口", "人体", "嘴巴"],
    "出入口空间": ["entrance", "entrance|入口", "空间", "开口"],
}

def match_hownet_sense(word, candidate_senses):
    """对歧义词词, 返回 HowNet 中与候选列表最匹配的义项索引"""
    try:
        senses = hownet.get_sense(word)
    except:
        return None
    if not senses:
        return None
    for i, s in enumerate(senses):
        s_str = str(s).lower()
        for cand in candidate_senses:
            aliases = SENSE_ALIAS.get(cand, [cand])
            for alias in aliases:
                if alias.lower() in s_str:
                    return i, s
    return None

# ── Layer 1: MFS ──
def layer1(word, dev_mfs, global_fallback):
    return dev_mfs.get(word, global_fallback)

# ── Layer 2: HowNet 第一义项 ──
def layer2(word, candidate_senses):
    result = match_hownet_sense(word, candidate_senses)
    if result:
        idx, sense_obj = result
        # 取第一个义项的索引, 映射到候选列表
        s_str = str(sense_obj).lower()
        for cand in candidate_senses:
            aliases = SENSE_ALIAS.get(cand, [cand])
            for alias in aliases:
                if alias.lower() in s_str:
                    return cand
    return None

# ── Layer 3: 复合词优先（目标锚定）──
def find_target_span(sentence, target_word, occurrence=1):
    """找目标词第 N 次出现的字符起止位置"""
    start = 0
    count = 0
    while True:
        pos = sentence.find(target_word, start)
        if pos == -1:
            return None
        count += 1
        if count == occurrence:
            return (pos, pos + len(target_word))
        start = pos + 1

def layer3(sentence, target_word, target_occurrence, candidate_senses):
    span = find_target_span(sentence, target_word, target_occurrence)
    if not span:
        return None
    
    # 用 jieba 分词, 找覆盖 span 的复合词
    words = list(pseg.cut(sentence))
    covering = []
    char_pos = 0
    for w, pos in words:
        word_end = char_pos + len(w)
        # 检查 jieba 词是否覆盖目标 span
        if char_pos <= span[0] and word_end >= span[1] and len(w) > len(target_word):
            covering.append((w, pos, char_pos, word_end))
        char_pos = word_end
    
    if not covering:
        return None
    
    # 取最长覆盖词
    longest = max(covering, key=lambda x: len(x[0]))
    compound = longest[0]
    
    # 检查 HowNet 中该词是否有唯一义项
    try:
        compound_senses = hownet.get_sense(compound)
    except:
        compound_senses = None
    
    if not compound_senses or len(compound_senses) != 1:
        return None
    
    # 单一义项 → 映射到候选列表
    s_str = str(compound_senses[0]).lower()
    for cand in candidate_senses:
        aliases = SENSE_ALIAS.get(cand, [cand])
        for alias in aliases:
            if alias.lower() in s_str:
                return cand
    return None

# ── Layer 4: 词性过滤 ──
POS_MAP = {
    "科技公司": ["n", "nr", "nt", "nz"],
    "水果": ["n"],
    "文书办公": ["n"],
    "日用包装": ["n"],
    "物理光线": ["n"],
    "修辞评价": ["v", "a", "d"],
    "植物": ["n"],
    "消费": ["v"],
    "评价可以": ["v", "a"],
    "排列行列": ["n", "q"],
    "行业银行": ["n", "nt"],
    "人体嘴巴": ["n"],
    "出入口空间": ["n"],
}

def layer4(sentence, target_word, target_occurrence, candidate_senses):
    span = find_target_span(sentence, target_word, target_occurrence)
    if not span:
        return None
    
    # 找目标词在 jieba 中的词性
    words = list(pseg.cut(sentence))
    char_pos = 0
    target_pos = None
    for w, pos in words:
        if char_pos == span[0] and w == target_word:
            target_pos = pos
            break
        char_pos += len(w)
    
    if not target_pos:
        return None
    
    # 过滤候选
    filtered = []
    for cand in candidate_senses:
        allowed = POS_MAP.get(cand, ["n", "v", "a"])
        if any(target_pos.startswith(p) for p in allowed):
            filtered.append(cand)
    
    if len(filtered) == 1:
        return filtered[0]
    return None

# ── Layer 5: HowNet 义原重叠 ──
def get_sememes(word):
    sems = set()
    try:
        data = hownet.get_sememes_by_word(word)
        if data:
            for entry in data:
                for s in entry.get("sememes", []):
                    if isinstance(s, str):
                        sems.add(s.split("|")[0].strip())
    except:
        pass
    return sems

def layer5(sentence, target_word, target_occurrence, candidate_senses):
    """HowNet 义原重叠评分, 返回 (赢家, 详细得分)"""
    words = jieba.lcut(sentence)
    context_words = [w for w in words if w != target_word and len(w) >= 2]
    
    import logging
    ctx_sems = set()
    for w in context_words:
        sems = get_sememes(w)
        if sems:
            pass  # OK
        ctx_sems.update(sems)
    
    if not ctx_sems:
        return None, {}
    
    # 对每个候选义项, 获取该义项相关的义原
    cand_sememes = {}
    for cand in candidate_senses:
        # 候选义项对应的义原: 用歧义词看所有义原, 按候选名投影
        all_sems = get_sememes(target_word)
        cand_sememes[cand] = all_sems  # 使用全部义原
    
    # 计算每个候选的得分 = 候选义原与语境义原的交集/候选义原数
    scores = {}
    for cand, sems in cand_sememes.items():
        if not sems:
            scores[cand] = 0.0
        else:
            overlap = len(sems & ctx_sems) / max(len(sems), 1)
            scores[cand] = overlap
    
    if not scores:
        return None, scores
    
    sorted_s = sorted(scores.items(), key=lambda x: -x[1])
    best_name, best_score = sorted_s[0]
    second_score = sorted_s[1][1] if len(sorted_s) > 1 else 0
    
    # 修改 1: 阈值从 1.2 降为 1.05
    threshold = 1.05
    if second_score == 0 or best_score >= second_score * threshold:
        # 同时检查绝对阈值
        if best_score >= 0.05:
            return best_name, scores
    return None, scores

# ── 级联预测 ──
def predict_one(item):
    sent = item["sentence"]
    word = item["target_word"]
    occ = item["target_occurrence"]
    candidates = item["candidate_senses"]
    
    debug = {"L1_MFS": layer1(word, mfs, global_mfs),
             "L2_howNet": layer2(word, candidates),
             "L3_compound_raw": None, "L3_mapped": None,
             "L4_pos": None, "L4_filtered": candidates,
             "L5_overlap": None, "L5_scores": {}, "L5_winner": None,
             "cascade_path": "MFS"}
    
    # L3
    l3_result = layer3(sent, word, occ, candidates)
    if l3_result:
        debug["cascade_path"] = "L3"
        return l3_result, debug
    
    # L4
    l4_result = layer4(sent, word, occ, candidates)
    if l4_result:
        debug["cascade_path"] = "L4"
        return l4_result, debug
    
    # L5（始终记录分数用于诊断）
    l5_result, l5_scores = layer5(sent, word, occ, candidates)
    debug["L5_scores"] = l5_scores
    debug["L5_winner"] = l5_result
    if l5_result:
        debug["cascade_path"] = "L5"
        return l5_result, debug
    
    # MFS 回退
    return debug["L1_MFS"], debug

# ── 主流程 ──
def main():
    os.makedirs(OUTPUT, exist_ok=True)
    
    for set_name in ["dev", "blind"]:
        input_file = os.path.join(DATA, f"{set_name}_input.json")
        output_file = os.path.join(OUTPUT, f"{set_name}_predictions.json")
        
        with open(input_file, "r") as f:
            items = json.load(f)
        
        results = []
        layer_counts = Counter()
        for item in items:
            pred, debug = predict_one(item)
            layer = debug["cascade_path"]
            layer_counts[layer] += 1
            results.append({
                "id": item["id"],
                "prediction": pred,
                "resolved_by": layer,
                "debug": {
                    "MFS": debug["L1_MFS"],
                    "L2_howNet_first": debug["L2_howNet"],
                    "L3_compound": debug["L3_compound_raw"],
                    "L3_mapped": debug["L3_mapped"],
                    "L4_pos": debug["L4_pos"],
                    "L4_filtered": debug["L4_filtered"],
                    "L5_scores": debug["L5_scores"],
                    "L5_winner": debug["L5_winner"],
                    "cascade_path": debug["cascade_path"]
                }
            })
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n{set_name}: {len(items)} 条")
        print(f"  层分布: {dict(layer_counts)}")
        
        # 自检 1: 每层输出多样性
        preds = [r["prediction"] for r in results]
        unique = set(preds)
        assert len(unique) >= 2, f"{set_name}: 只输出单一标签 {unique}"
        print(f"  ✅ 自检1: 输出 {len(unique)} 种不同标签")
        
        # 自检 3: 级联路径
        paths = set(r["resolved_by"] for r in results)
        print(f"  ✅ 自检3: 级联路径={paths}")
        
        # L5 诊断
        l5_attempted = 0
        l5_near_miss = 0
        for r in results:
            scores = r.get("debug", {}).get("L5_scores", {})
            if scores:
                vals = list(scores.values())
                if max(vals) > 0:
                    l5_attempted += 1
                    if max(vals) > 0.02 and r["resolved_by"] != "L5":
                        l5_near_miss += 1
        print(f"  L5诊断: {l5_attempted}句有非零分, {l5_near_miss}句差一点解决")

if __name__ == "__main__":
    main()
