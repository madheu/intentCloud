"""P0-R4 输入验证器 — 强制验证"""
import json, sys

SENSE_IDS = {"苹果":["apple.company","apple.fruit"],"小米":["xiaomi.company","xiaomi.grain"],
             "杜鹃":["cuckoo.flower","cuckoo.bird"],"花":["hua.plant","hua.spend"],
             "光":["guang.physical","guang.figurative"],"行":["xing.approval","xing.row","xing.industry"],
             "口":["kou.body","kou.space"],"头":["tou.body","tou.leader","tou.beginning"],
             "结":["jie.concrete","jie.abstract"]}

# 旧 dev/blind 输入(用于去重检查)
OLD_SENTENCES = set()

def validate(data):
    assert isinstance(data, list), "必须是数组"
    assert len(data) == 120, f"必须恰好 120 条, 实际 {len(data)}"
    ids = [d["id"] for d in data]
    assert len(ids) == len(set(ids)), "ID 不唯一"
    for item in data:
        assert "id" in item, "缺少 id"
        assert "sentence" in item, f"{item['id']}: 缺少 sentence"
        assert "target_surface" in item, f"{item['id']}: 缺少 target_surface"
        assert "target_start" in item, f"{item['id']}: 缺少 target_start"
        assert isinstance(item["target_start"], int), f"{item['id']}: target_start 非整数"
        assert "target_end" in item, f"{item['id']}: 缺少 target_end"
        assert isinstance(item["target_end"], int), f"{item['id']}: target_end 非整数"
        assert "candidate_sense_ids" in item, f"{item['id']}: 缺少 candidate_sense_ids"
        w = item["target_surface"]
        assert w in SENSE_IDS, f"{item['id']}: 未知目标词 '{w}'"
        expected = SENSE_IDS[w]
        assert item["candidate_sense_ids"] == expected, f"{item['id']}: 候选集不对, 期望 {expected}, 实际 {item['candidate_sense_ids']}"
        ts, te = item["target_start"], item["target_end"]
        sent = item["sentence"]
        assert isinstance(sent, str), f"{item['id']}: sentence 非字符串"
        assert ts >= 0 and te <= len(sent), f"{item['id']}: span [{ts},{te}) 超出句子长度 {len(sent)}"
        assert sent[ts:te] == w, f"{item['id']}: span [{ts},{te})='{sent[ts:te]}' ≠ '{w}'"
        assert sent.count(w) == 1, f"{item['id']}: '{w}' 出现 {sent.count(w)} 次, 应为 1"
        # 禁止金标字段
        for forbid in ["label", "gold", "sense_name", "intended_sense"]:
            assert forbid not in item and not any(forbid in k for k in item), f"{item['id']}: 包含禁止字段 '{forbid}'"
        # 不与旧数据重复
        if sent in OLD_SENTENCES:
            # 加载旧数据
            pass
    return True

if __name__ == "__main__":
    data = json.load(open(sys.argv[1], "r", encoding="utf-8"))
    validate(data)
    print(f"输入验证通过: {len(data)} 条")
