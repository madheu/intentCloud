"""P0-R3 严格评分器 — 只做精确匹配"""
import json, sys

def score(gold_path, pred_path):
    with open(gold_path) as f:
        gold = json.load(f)
    with open(pred_path) as f:
        pred = json.load(f)
    assert len(gold) == len(pred), f"数量不匹配: {len(gold)} vs {len(pred)}"
    pred_map = {p["id"]: p["prediction"] for p in pred}
    gold_map = {g["id"]: g["gold_sense_id"] for g in gold}
    assert pred_map.keys() == gold_map.keys(), "ID 集合不匹配"
    correct = 0
    errors = []
    for pid, p in pred_map.items():
        g = gold_map[pid]
        assert g in [s for sg in gold if sg["id"] == pid][0], "gold 在候选集中"
        if p == g:
            correct += 1
        else:
            errors.append({"id": pid, "gold": g, "pred": p})
    acc = correct / len(pred) * 100
    print(f"正确: {correct}/{len(pred)} ({acc:.1f}%)")
    for e in errors[:10]:
        print(f"  ❌ {e['id']}: 金标={e['gold']}, 预测={e['pred']}")
    return acc

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python p0_r3_score.py gold.json predictions.json")
        sys.exit(1)
    score(sys.argv[1], sys.argv[2])
