"""P0-R3 输入验证器"""
import json, sys

def validate(input_path, allowed_senses):
    with open(input_path) as f:
        data = json.load(f)
    assert isinstance(data, list), "必须是数组"
    assert len(data) > 0, "至少一条"
    ids = [d["id"] for d in data]
    assert len(ids) == len(set(ids)), "ID 必须唯一"
    for item in data:
        assert "id" in item, f"缺少 id: {item}"
        assert "sentence" in item, f"缺少 sentence: {item['id']}"
        assert "target_surface" in item, f"缺少 target_surface: {item['id']}"
        assert "target_start" in item, f"缺少 target_start: {item['id']}"
        assert "target_end" in item, f"缺少 target_end: {item['id']}"
        assert "candidate_sense_ids" in item, f"缺少 candidate_sense_ids: {item['id']}"
        w = item["target_surface"]
        assert w in allowed_senses, f"非法目标词: {w}"
        cands = item["candidate_sense_ids"]
        expected = allowed_senses[w]
        assert cands == expected, f"{item['id']}: 候选集不完整, 期望={expected}, 实际={cands}"
        ts, te = item["target_start"], item["target_end"]
        sent = item["sentence"]
        assert sent[ts:te] == w, f"{item['id']}: span 不匹配, 期望'{w}' 实际'{sent[ts:te]}'"
        assert sent.count(w) == 1, f"{item['id']}: {w} 出现次数 != 1"
        # 禁止金标字段
        for forbid in ["label", "gold", "sense_name", "intended_sense"]:
            assert forbid not in item, f"{item['id']}: 包含禁止字段 {forbid}"
    print(f"  ✅ 验证通过: {len(data)} 条")

if __name__ == "__main__":
    allowed = {"苹果":["apple.company","apple.fruit"],"小米":["xiaomi.company","xiaomi.grain"],
               "杜鹃":["cuckoo.flower","cuckoo.bird"],"花":["hua.plant","hua.spend"],
               "光":["guang.physical","guang.figurative"],"行":["xing.approval","xing.row","xing.industry"],
               "口":["kou.body","kou.space"],"头":["tou.body","tou.leader","tou.beginning"],
               "结":["jie.concrete","jie.abstract"]}
    validate(sys.argv[1], allowed)
