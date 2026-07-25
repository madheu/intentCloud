"""P0-R4 严格评分器 — 三文件输入(input+gold+prediction), 精确 sense ID 比较"""
import json, sys

def score(input_path, gold_path, pred_path):
    inp = json.load(open(input_path, "r", encoding="utf-8"))
    gold = json.load(open(gold_path, "r", encoding="utf-8"))
    pred = json.load(open(pred_path, "r", encoding="utf-8"))
    assert len(inp) == len(gold) == len(pred), "文件长度不一致"
    inp_ids = sorted(d["id"] for d in inp)
    gold_ids = sorted(d["id"] for d in gold)
    pred_ids = sorted(d["id"] for d in pred)
    assert inp_ids == gold_ids == pred_ids, "ID 集合不一致"
    gold_map = {d["id"]: d["gold_sense_id"] for d in gold}
    pred_map = {d["id"]: d["prediction"] for d in pred}
    cand_map = {d["id"]: d["candidate_sense_ids"] for d in inp}
    correct = 0
    for pid, p in pred_map.items():
        g = gold_map[pid]
        assert g in cand_map[pid], f"{pid}: gold '{g}' 不在候选 {cand_map[pid]} 中"
        assert p in cand_map[pid], f"{pid}: pred '{p}' 不在候选 {cand_map[pid]} 中"
        if p == g:
            correct += 1
    acc = correct / len(pred) * 100
    print(f"正确: {correct}/{len(pred)} ({acc:.1f}%)")
    return acc

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("用法: python p0_r4_score.py input.json gold.json predictions.json", file=sys.stderr)
        sys.exit(1)
    score(sys.argv[1], sys.argv[2], sys.argv[3])
