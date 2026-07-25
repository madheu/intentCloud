"""构建 dev_input.json 和 blind_input.json"""
import json, os

BASE = r"E:\intentCloud\Haibo-Research\06_Experiment"

with open(os.path.join(BASE, "P0-R_gold_labels.json"), "r") as f:
    dev_gold = json.load(f)
with open(os.path.join(BASE, "P0-R_blind_gold_labels.json"), "r") as f:
    blind_gold = json.load(f)

# 开发集 sense 映射（人工构建）
DEV_SENSES = {
    "苹果": ["科技公司", "水果"],
    "纸": ["文书办公", "日用包装"],
    "光": ["物理光线", "修辞评价"],
    "花": ["植物", "消费"],
    "行": ["评价可以", "排列行列", "行业银行"],
    "口": ["人体嘴巴", "出入口空间"],
}

def build_input(gold, is_blind=False):
    items = []
    idx = 0
    if is_blind:
        for word, entries in gold.items():
            for e in entries:
                idx += 1
                sent = e["sentence"]
                # 找 candidate_senses
                label = e["label"]
                sense_name = e["sense_name"]
                # 通用候选（根据词确定）
                candidates = [sense_name]  # fallback
                items.append({
                    "id": idx,
                    "sentence": sent,
                    "target_word": word,
                    "target_occurrence": 1,
                    "candidate_senses": candidates,
                    "_gold_label": label  # 不会被 predict 脚本使用
                })
    else:
        for word, sentences in gold.items():
            senses = DEV_SENSES.get(word, ["A", "B", "C"][:len(set(sentences.values()))])
            for sent, label in sentences.items():
                idx += 1
                items.append({
                    "id": idx,
                    "sentence": sent,
                    "target_word": word,
                    "target_occurrence": 1,
                    "candidate_senses": senses,
                    "_gold_label": label
                })
    return items

dev_input = build_input(dev_gold)
blind_input = build_input(blind_gold, is_blind=True)

os.makedirs(os.path.join(BASE, "data"), exist_ok=True)

# 保存无标签版本（用于预测）
dev_no_label = [{k:v for k,v in d.items() if k != "_gold_label"} for d in dev_input]
blind_no_label = [{k:v for k,v in d.items() if k != "_gold_label"} for d in blind_input]

with open(os.path.join(BASE, "data", "dev_input.json"), "w", encoding="utf-8") as f:
    json.dump(dev_no_label, f, ensure_ascii=False, indent=2)
with open(os.path.join(BASE, "data", "blind_input.json"), "w", encoding="utf-8") as f:
    json.dump(blind_no_label, f, ensure_ascii=False, indent=2)

# 保存有标签版本（用于评分）
with open(os.path.join(BASE, "data", "dev_gold.json"), "w", encoding="utf-8") as f:
    json.dump(dev_input, f, ensure_ascii=False, indent=2)
with open(os.path.join(BASE, "data", "blind_gold.json"), "w", encoding="utf-8") as f:
    json.dump(blind_input, f, ensure_ascii=False, indent=2)

print(f"dev_input: {len(dev_input)} 条, blind_input: {len(blind_input)} 条")
print("保存到 data/ 目录")
