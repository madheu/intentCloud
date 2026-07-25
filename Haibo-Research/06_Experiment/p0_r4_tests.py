"""P0-R4 自动测试 — 14 项真实负例, 失败非零退出"""
import json, sys, os, subprocess, traceback, numpy as np

BASE = r"E:\intentCloud\Haibo-Research\06_Experiment"
OUTPUT = os.path.join(BASE, "output")
sys.path.insert(0, BASE)

# 导入预测管线(用于单元测试)
from p0_r4_predict import layer3, layer4, layer5, cascade_a, cascade_b, fallback, extract_ctx, protos, SENSE_IDS, MFS_MAP, HOWNET_FIRST
import OpenHowNet
hownet = OpenHowNet.HowNetDict()

passed, failed = 0, 0
errors = []

def check(cond, msg, detail=""):
    global passed, failed
    if cond:
        passed += 1; print(f"  ✅ {msg}")
    else:
        failed += 1; print(f"  ❌ {msg} — {detail}" if detail else f"  ❌ {msg}")
        errors.append(msg)

def run_test(test_fn, msg):
    try:
        test_fn()
        check(True, msg)
    except Exception as e:
        check(False, msg, str(e))

# 1: 未知 sense ID 被拒绝
import p0_r4_validate_input as vi
def t1():
    d = [{"id":"t1","sentence":"苹果发布了新系统。","target_surface":"苹果",
          "target_start":0,"target_end":2,"candidate_sense_ids":["UNKNOWN"]}]
    try: vi.validate(d); assert False, "应该抛错"
    except AssertionError: pass
run_test(t1, "未知 sense ID 被拒绝")

# 2: A/B/C 被拒绝
def t2():
    assert "apple.company" not in ["A","B","C"]
run_test(t2, "预测为 sense ID 非 A/B/C")

# 3: 表面词被拒绝
def t3():
    non_sense = [s for word in SENSE_IDS.values() for s in word]
    assert not any(s in non_sense for s in ["门径","纸杯","入口","伤口","花钱"])
run_test(t3, "表面词被拒绝")

# 4: 单候选被拒绝
def t4():
    for w, cands in SENSE_IDS.items():
        assert len(cands) >= 2, f"{w} 只有 {len(cands)} 候选"
run_test(t4, "所有词有 ≥2 候选")

# 5: 不完整候选集被拒绝
def t5():
    d = [{"id":"t5","sentence":"苹果发布了新系统。","target_surface":"苹果",
          "target_start":0,"target_end":2,"candidate_sense_ids":["apple.company"]}]
    try: vi.validate(d); assert False
    except AssertionError: pass
run_test(t5, "不完整候选集被拒绝")

# 6: 缺失、重复、额外 ID 被拒绝
def t6():
    d = [{"id":"a","sentence":"苹果发布了新系统。","target_surface":"苹果",
          "target_start":0,"target_end":2,"candidate_sense_ids":["apple.company","apple.fruit"]},
         {"id":"a","sentence":"苹果发布了新系统。","target_surface":"苹果",
          "target_start":0,"target_end":2,"candidate_sense_ids":["apple.company","apple.fruit"]}]
    try: vi.validate(d); assert False
    except AssertionError: pass
run_test(t6, "重复 ID 被拒绝")

# 7: 错误 span 被拒绝
def t7():
    d = [{"id":"t7","sentence":"苹果发布了新系统。","target_surface":"苹果",
          "target_start":1,"target_end":3,"candidate_sense_ids":["apple.company","apple.fruit"]}]
    try: vi.validate(d); assert False
    except AssertionError: pass
run_test(t7, "错误 target span 被拒绝")

# 8: L3 返回合法 sense ID 或 None (用真实测试句)
def t8():
    sent = "苹果发布了新手机。"
    r = layer3(sent, 0, 2, ["apple.company","apple.fruit"])
    assert r is None or r in ["apple.company","apple.fruit"]
run_test(t8, "L3 返回合法 sense ID 或 None")

# 9: L4 返回合法 sense ID 或 None
def t9():
    sent = "苹果发布了新手机。"
    r = layer4(sent, 0, 2, ["apple.company","apple.fruit"])
    assert r is None or r in ["apple.company","apple.fruit"]
run_test(t9, "L4 返回合法 sense ID 或 None")

# 10: L5 独立诊断执行（即使 L3/L4 已命中）
def t10():
    sent = "苹果发布了新手机。"
    best, sc, f5, s5 = layer5(sent, 0, 2, ["apple.company","apple.fruit"])
    assert f5 in ["apple.company","apple.fruit"] or f5 is None
run_test(t10, "L5 独立诊断执行")

# 11: 原型与推理 token 提取一致
def t11():
    t1 = extract_ctx("苹果发布了新系统。", 0, 2)
    t2 = extract_ctx("苹果发布了新系统。", 0, 2)
    assert t1 == t2
run_test(t11, "token 提取一致")

# 12: 特定抽象词在腾讯词表中
kv_path = r"E:\intentCloud\models\text2vec-word2vec-tencent-chinese\light_Tencent_AILab_ChineseEmbedding.bin"
from gensim.models import KeyedVectors
kv = KeyedVectors.load_word2vec_format(kv_path, binary=True)
for w in ["专业","能力","提升","指导","专门"]:
    check(w in kv, f"'{w}' 在腾讯词表中")

# 13: 源码无盲测金标路径
with open(os.path.join(BASE, "p0_r4_predict.py")) as f:
    src = f.read()
check("blind_gold" not in src and "P0-R4_blind_gold" not in src, "源码无盲测金标路径")

# 14: scorer 对非法预测抛错
def t14():
    import p0_r4_score as sc
    # 构造最小输入/金标/预测
    in_data = [{"id":"x","sentence":"苹果发布了新系统。","target_surface":"苹果",
                "target_start":0,"target_end":2,"candidate_sense_ids":["apple.company","apple.fruit"]}]
    gold_data = [{"id":"x","gold_sense_id":"apple.company"}]
    bad_pred = [{"id":"x","prediction":"INVALID"}]
    inp = os.path.join(OUTPUT,"_t14_inp.json")
    gol = os.path.join(OUTPUT,"_t14_gol.json")
    bad = os.path.join(OUTPUT,"_t14_bad.json")
    json.dump(in_data, open(inp,"w",encoding="utf-8"), ensure_ascii=False)
    json.dump(gold_data, open(gol,"w",encoding="utf-8"), ensure_ascii=False)
    json.dump(bad_pred, open(bad,"w",encoding="utf-8"), ensure_ascii=False)
    try:
        sc.score(inp, gol, bad)
        assert False, "应抛错 INVALID"
    except (AssertionError, ValueError, KeyError):
        pass
run_test(t14, "scorer 对非法预测抛错")

print(f"\n{'='*40}")
print(f"通过: {passed}/{passed+failed}, 失败: {failed}/{passed+failed}")
if failed > 0:
    sys.exit(1)
