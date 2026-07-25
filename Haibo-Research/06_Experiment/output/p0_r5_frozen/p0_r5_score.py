"""P0-R5 严格评分器 — 三文件, 精确 sense ID 比较"""
import json

def score(input_path, gold_path, pred_path):
    inp = json.load(open(input_path, encoding="utf-8"))
    gold = json.load(open(gold_path, encoding="utf-8"))
    pred = json.load(open(pred_path, encoding="utf-8"))
    # ID 唯一性断言
    for name, data in [("input", inp), ("gold", gold), ("prediction", pred)]:
        ids = [d["id"] for d in data]
        assert len(ids) == len(set(ids)), f"{name} ID 不唯一"
    # ID 集合一致
    ids = sorted(d["id"] for d in inp)
    assert ids == sorted(d["id"] for d in gold), "inp/gold ID 不一致"
    assert ids == sorted(d["id"] for d in pred), "inp/pred ID 不一致"
    # 建图
    cand_map = {d["id"]: d["candidate_sense_ids"] for d in inp}
    gold_map = {d["id"]: d["gold_sense_id"] for d in gold}
    pred_map = {d["id"]: d["prediction"] for d in pred}
    # 验证 gold/pred 属于候选
    for pid in ids:
        c = cand_map[pid]
        assert gold_map[pid] in c, f"{pid}: gold '{gold_map[pid]}' 不在候选 {c} 中"
        assert pred_map[pid] in c, f"{pid}: pred '{pred_map[pid]}' 不在候选 {c} 中"
    correct = sum(1 for pid in ids if pred_map[pid] == gold_map[pid])
    acc = correct / len(ids) * 100
    print(f"正确: {correct}/{len(ids)} ({acc:.1f}%)")
    return acc
