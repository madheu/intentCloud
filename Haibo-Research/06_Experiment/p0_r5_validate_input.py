"""P0-R5 输入验证器 — 强制 120 条 + 目标分布 + 旧句去重"""
import json, os, sys

SENSE_IDS = {"苹果":["apple.company","apple.fruit"],"小米":["xiaomi.company","xiaomi.grain"],
             "杜鹃":["cuckoo.flower","cuckoo.bird"],"花":["hua.plant","hua.spend"],
             "光":["guang.physical","guang.figurative"],"行":["xing.approval","xing.row","xing.industry"],
             "口":["kou.body","kou.space"],"头":["tou.body","tou.leader","tou.beginning"],
             "结":["jie.concrete","jie.abstract"]}
# 目标分布: 7 个二义词各 12 条, 2 个三义词各 18 条 = 120
TARGET_COUNTS = {"苹果":12,"小米":12,"杜鹃":12,"花":12,"光":12,"口":12,"结":12,"行":18,"头":18}

def validate(data, old_sentences=None):
    if old_sentences is None: old_sentences = set()
    assert isinstance(data, list), "必须是数组"
    assert len(data) == 120, f"必须恰好 120 条, 实际 {len(data)}"
    ids = [d["id"] for d in data]
    assert len(ids) == len(set(ids)), "ID 不唯一"
    # 目标分布
    wc = {}
    for d in data:
        w = d["target_surface"]
        wc[w] = wc.get(w, 0) + 1
    for w, expected in TARGET_COUNTS.items():
        actual = wc.get(w, 0)
        assert actual == expected, f"{w}: 期望 {expected} 条, 实际 {actual}"
    # 逐条检查
    for item in data:
        req_fields = ["id","sentence","target_surface","target_start","target_end","candidate_sense_ids"]
        for f in req_fields:
            assert f in item, f"缺少 {f}"
        w = item["target_surface"]
        assert w in SENSE_IDS, f"未知目标词 '{w}'"
        assert item["candidate_sense_ids"] == SENSE_IDS[w], f"候选集不匹配"
        ts, te = item["target_start"], item["target_end"]
        assert isinstance(ts, int) and isinstance(te, int), "span 非整数"
        sent = item["sentence"]
        assert isinstance(sent, str), "sentence 非字符串"
        assert 0 <= ts < te <= len(sent), f"span [{ts},{te}) 越界"
        assert sent[ts:te] == w, f"span '{sent[ts:te]}' ≠ '{w}'"
        assert sent.count(w) == 1, f"'{w}' 出现 {sent.count(w)} 次"
        # 禁止金标字段
        for k in item:
            for forbid in ["label","gold","intended","sense_name"]:
                assert forbid not in k.lower(), f"包含禁止字段 '{k}'"
        # 旧句去重
        norm = sent.replace(" ","").replace("　","")
        assert norm not in old_sentences, f"与旧句重复"
    return True
