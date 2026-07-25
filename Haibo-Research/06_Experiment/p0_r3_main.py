"""P0-R3 主驱动 — 开发运行 + 阈值冻结（修复版）"""
import json, os, sys, numpy as np
sys.path = [p for p in sys.path if 'Diviner' not in p]
from collections import Counter
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from p0_r3_predict import *

OLD_DEV = os.path.join(BASE, "P0-R_gold_labels.json")
OLD_BLIND = os.path.join(BASE, "P0-R_blind_gold_labels.json")

# ── Fix5: 完整 9 词开发数据 ──
# 旧 38 句 + 旧 60 句中共有 9 个目标词的句子
LABEL_MAP = {
    "苹果": {"A": "apple.company", "B": "apple.fruit"},
    "光": {"A": "guang.physical", "B": "guang.figurative"},
    "花": {"A": "hua.plant", "B": "hua.spend"},
    "行": {"A": "xing.approval", "B": "xing.row", "C": "xing.industry"},
    "口": {"A": "kou.body", "B": "kou.space"},
}
BLIND_LABEL_MAP = {
    "苹果": {"科技公司": "apple.company", "水果": "apple.fruit"},
    "花": {"消费": "hua.spend", "植物": "hua.plant"},
    "光": {"修辞评价": "guang.figurative", "物理光线": "guang.physical"},
    "行": {"排列": "xing.row", "评价": "xing.approval"},
    "口": {"人体": "kou.body", "空间": "kou.space"},
    "头": {"人体": "tou.body", "领导": "tou.leader", "起始": "tou.beginning"},
    "结": {"具体": "jie.concrete", "抽象": "jie.abstract"},
    "小米": {"谷物": "xiaomi.grain", "科技": "xiaomi.company"},
    "杜鹃": {"花": "cuckoo.flower", "鸟": "cuckoo.bird"},
}

def build_all_dev():
    items = []; idx = 0
    # 旧 38 句
    with open(OLD_DEV) as f: old = json.load(f)
    for word, sents in old.items():
        if word not in SENSE_IDS: continue
        for sent, label in sents.items():
            idx += 1
            ts = sent.find(word)
            if ts == -1: continue
            gold = LABEL_MAP.get(word, {}).get(label)
            if gold is None: continue
            items.append({"id": f"dev-{idx:04d}", "sentence": sent, "target_surface": word,
                          "target_start": ts, "target_end": ts+len(word),
                          "candidate_sense_ids": SENSE_IDS[word], "_gold_sense_id": gold})
    # 旧 60 句
    with open(OLD_BLIND) as f: old_blind = json.load(f)
    for word, entries in old_blind.items():
        if word not in SENSE_IDS: continue
        for e in entries:
            idx += 1
            sent = e["sentence"]; ts = sent.find(word)
            if ts == -1: continue
            gold = BLIND_LABEL_MAP.get(word, {}).get(e["label"], e.get("sense_name"))
            if gold is None: continue
            items.append({"id": f"dev-{idx:04d}", "sentence": sent, "target_surface": word,
                          "target_start": ts, "target_end": ts+len(word),
                          "candidate_sense_ids": SENSE_IDS[word], "_gold_sense_id": gold})
    return items

dev_data = build_all_dev()
print(f"开发集: {len(dev_data)} 条, 覆盖词: {set(it['target_surface'] for it in dev_data)}")

# MFS
mfs_map = compute_mfs(dev_data)
print(f"MFS: {mfs_map}")
# 验证每个词都有 MFS 或 HOWNET_FIRST
for w in SENSE_IDS:
    assert mfs_map.get(w) or HOWNET_FIRST.get(w), f"{w}: 无 MFS 也无 HOWNET_FIRST"
print("  ✅ 所有词合法回退")

# 原型
print("\n构建原型...")
protos = build_prototypes()

# ── Fix6: Macro accuracy 阈值搜索 ──
print("\n阈值搜索 (macro accuracy, coverage>=20%):")
best_macro = -1; best_cfg = None
for ms in [0.0, 0.1, 0.2, 0.3]:
    for mm in [0.0, 0.02, 0.05, 0.10]:
        CONFIG["l5_min_similarity"] = ms
        CONFIG["l5_min_margin"] = mm
        # 按目标词分组计算 acc
        word_results = {w: {"correct": 0, "total": 0, "l5": 0} for w in SENSE_IDS}
        for it in dev_data:
            w = it["target_surface"]
            _, _, _, sel = layer5(it["sentence"], it["target_start"], it["target_end"],
                                  it["candidate_sense_ids"], protos)
            if sel:
                word_results[w]["l5"] += 1
                if sel == it["_gold_sense_id"]:
                    word_results[w]["correct"] += 1
                word_results[w]["total"] += 1
        # 总 coverage ≥ 20%
        l5_all = sum(v["total"] for v in word_results.values())
        cov = l5_all / len(dev_data) * 100
        if cov < 20: continue
        # macro accuracy
        word_accs = []
        for w, v in word_results.items():
            if v["total"] > 0:
                word_accs.append(v["correct"] / max(v["total"], 1))
        macro = np.mean(word_accs) if word_accs else 0
        print(f"  ms={ms:.1f} mm={mm:.2f}: macro={macro:.3f} cov={cov:.0f}%")
        if macro > best_macro:
            best_macro = macro; best_cfg = (ms, mm, macro, cov)

if best_cfg:
    CONFIG["l5_min_similarity"] = best_cfg[0]
    CONFIG["l5_min_margin"] = best_cfg[1]
else:
    CONFIG["l5_min_similarity"] = 0.0
    CONFIG["l5_min_margin"] = 0.0
print(f"\n最佳阈值: ms={CONFIG['l5_min_similarity']} mm={CONFIG['l5_min_margin']} (macro={best_macro:.3f} cov={best_cfg[3]:.0f}%如果找到)")

# 写入配置
with open(CONFIG_PATH, "w") as f:
    json.dump(CONFIG, f, ensure_ascii=False, indent=2)
print(f"配置已写入: {CONFIG_PATH}")
