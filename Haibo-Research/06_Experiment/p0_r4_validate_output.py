"""P0-R4 输出验证器 — 逐条检查"""
import json, sys

ALLOWED = {"apple.company","apple.fruit","xiaomi.company","xiaomi.grain","cuckoo.flower","cuckoo.bird",
           "hua.plant","hua.spend","guang.physical","guang.figurative",
           "xing.approval","xing.row","xing.industry","kou.body","kou.space",
           "tou.body","tou.leader","tou.beginning","jie.concrete","jie.abstract"}

def validate(input_path, pred_path):
    inp = json.load(open(input_path, "r", encoding="utf-8"))
    pred = json.load(open(pred_path, "r", encoding="utf-8"))
    assert len(inp) == len(pred), f"输入 {len(inp)} 条 vs 预测 {len(pred)} 条"
    inp_map = {d["id"]: d for d in inp}
    pred_map = {d["id"]: d for d in pred}
    assert set(inp_map.keys()) == set(pred_map.keys()), "ID 集合不匹配"
    for pid, p in pred_map.items():
        assert p["prediction"] in ALLOWED, f"{pid}: 非法预测 '{p['prediction']}'"
        candidates = inp_map[pid]["candidate_sense_ids"]
        assert p["prediction"] in candidates, f"{pid}: '{p['prediction']}' 不在候选 {candidates} 中"
        indep = p.get("independent", {})
        for key in ["MFS", "L2", "L3", "L4", "L5_forced", "L5_selective"]:
            v = indep.get(key)
            if v is not None:
                assert v in candidates or v in ALLOWED, f"{pid}: independent.{key}='{v}' 非法"
        debug = p.get("debug", {})
        for field in ["target_span_valid", "covering_compound", "target_pos", "context_tokens",
                       "context_oov_tokens", "L5_computable", "L5_scores", "L5_best_margin",
                       "cascade_A", "cascade_B"]:
            assert field in debug, f"{pid}: 缺少 debug.{field}"
        assert "resolved_by" in p, f"{pid}: 缺少 resolved_by"
        assert p["resolved_by"] in ["L3", "L4", "L5", "MFS"], f"{pid}: resolved_by='{p['resolved_by']}'"
    # 无重复预测
    assert len(set(p["id"] for p in pred)) == len(pred), "有重复 ID"
    print(f"输出验证通过: {len(pred)} 条")

if __name__ == "__main__":
    validate(sys.argv[1], sys.argv[2])
