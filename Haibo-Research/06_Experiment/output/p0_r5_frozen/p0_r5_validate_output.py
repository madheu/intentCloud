"""P0-R5 输出验证器 — 逐条检查"""
import json

ALLOWED = {"apple.company","apple.fruit","xiaomi.company","xiaomi.grain","cuckoo.flower","cuckoo.bird",
           "hua.plant","hua.spend","guang.physical","guang.figurative","xing.approval","xing.row",
           "xing.industry","kou.body","kou.space","tou.body","tou.leader","tou.beginning",
           "jie.concrete","jie.abstract"}

def validate(inp, pred):
    assert len(inp) == len(pred), f"数量 {len(inp)} vs {len(pred)}"
    inp_map = {d["id"]: d for d in inp}
    assert len(inp_map) == len(inp), "输入 ID 不唯一"
    pred_map = {}
    for p in pred:
        assert p["id"] not in pred_map, f"重复预测 ID: {p['id']}"
        pred_map[p["id"]] = p
    assert set(inp_map.keys()) == set(pred_map.keys()), "ID 集合不匹配"
    for pid, p in pred_map.items():
        cands = inp_map[pid]["candidate_sense_ids"]
        # prediction
        assert p["prediction"] in cands, f"{pid}: 预测 '{p['prediction']}' 不在候选 {cands} 中"
        # resolved_by
        assert p["resolved_by"] in ["L3","L4","L5","MFS"], f"{pid}: resolved_by 非法"
        # independent
        ind = p.get("independent", {})
        for key in ["MFS","L2","L3","L4","L5_forced","L5_selective"]:
            assert key in ind, f"{pid}: 缺少 independent.{key}"
            v = ind[key]
            if v is not None:
                assert v in cands, f"{pid}: independent.{key}='{v}' 不在候选 {cands} 中"
        # debug
        db = p.get("debug", {})
        for f in ["target_span_valid","covering_compound","target_pos","context_tokens",
                   "context_oov_tokens","L5_computable","L5_scores","L5_best_margin",
                   "cascade_A","cascade_B"]:
            assert f in db, f"{pid}: 缺少 debug.{f}"
        # cascade_A/B 属于候选
        for cf in ["cascade_A","cascade_B"]:
            cv = db[cf]
            assert cv in cands or cv is None, f"{pid}: {cf}='{cv}' 不在候选 {cands} 中"
        # cascade_B == prediction
        assert db["cascade_B"] == p["prediction"], f"{pid}: cascade_B={db['cascade_B']} ≠ prediction={p['prediction']}"
        # resolved_by 与 cascade_B 来源一致: 如果 cascade_B 由某层解决, resolved_by 必须是那层
        # L5 scores keys
        l5s = db["L5_scores"]
        if l5s:
            for k in l5s:
                assert k in cands, f"{pid}: L5_scores 键 '{k}' 不在候选 {cands} 中"
        # L5_computable 与 context_tokens 一致
        if db["L5_computable"]:
            assert len(db["context_tokens"]) > 0
        else:
            assert len(db["context_tokens"]) == 0
    print(f"  输出验证: ✅ {len(pred)} 条")
