"""P0-R3 输出验证器"""
import json, sys

allowed_senses = {"apple.company","apple.fruit","xiaomi.company","xiaomi.grain","cuckoo.flower","cuckoo.bird",
                  "hua.plant","hua.spend","guang.physical","guang.figurative",
                  "xing.approval","xing.row","xing.industry","kou.body","kou.space",
                  "tou.body","tou.leader","tou.beginning","jie.concrete","jie.abstract"}

def validate(input_path, pred_path):
    with open(input_path) as f:
        inp = json.load(f)
    with open(pred_path) as f:
        pred = json.load(f)
    in_ids = set(d["id"] for d in inp)
    pred_ids = set(d["id"] for d in pred)
    assert in_ids == pred_ids, f"ID 不匹配: 输入={len(in_ids)}, 预测={len(pred_ids)}"
    for p in pred:
        sid = p["prediction"]
        assert sid in allowed_senses, f"{p['id']}: 非法预测 {sid}（不在全局 sense 表中）"
        # 验证 prediction 属于该条候选集
        inp_item = next(d for d in inp if d["id"] == p["id"])
        assert sid in inp_item["candidate_sense_ids"], f"{p['id']}: {sid} 不在候选集中 {inp_item['candidate_sense_ids']}"
        # 验证独立诊断字段
        indep = p.get("independent", {})
        for key in ["MFS", "L2", "L3", "L4", "L5_forced", "L5_selective"]:
            val = indep.get(key)
            if val is not None:
                assert val in allowed_senses or val in inp_item["candidate_sense_ids"], f"{p['id']}: 非法 {key}={val}"
        # 验证 debug 字段
        debug = p.get("debug", {})
        assert "context_tokens" in debug, f"{p['id']}: 缺少 context_tokens"
        assert "L5_computable" in debug, f"{p['id']}: 缺少 L5_computable"
        assert "cascade_A" in debug, f"{p['id']}: 缺少 cascade_A"
        assert "cascade_B" in debug, f"{p['id']}: 缺少 cascade_B"
    print(f"  ✅ 输出验证通过: {len(pred)} 条")

if __name__ == "__main__":
    validate(sys.argv[1], sys.argv[2])
