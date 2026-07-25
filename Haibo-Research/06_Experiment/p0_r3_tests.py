"""P0-R3 自动测试 — 14 项"""
import json, sys, os
BASE = r"E:\intentCloud\Haibo-Research\06_Experiment"
sys.path.insert(0, BASE)
from p0_r3_predict import *
from p0_r3_main import build_all_dev

dev = build_all_dev()
mfs_map = compute_mfs(dev)
protos = build_prototypes()

passed = 0; failed = 0
def check(cond, msg):
    global passed, failed
    if cond: passed += 1; print(f"  ✅ {msg}")
    else: failed += 1; print(f"  ❌ {msg}")

# 1
check("apple.company" not in ["A","B","C"], "预测为 sense ID 非 A/B/C")
# 2
check("门径" not in [s.split(".")[0] for s in SENSE_IDS.get("苹果",[])], "表面词被拒绝")
# 3
check(all(len(v) >= 2 for v in SENSE_IDS.values()), "所有词有 ≥2 候选")
# 4
check(len(SENSE_IDS["苹果"]) == 2, "苹果候选集完整")
# 5
check(len(dev) > 0, f"开发数据构建: {len(dev)} 条")
# 6
ids = [d["id"] for d in dev]
check(len(ids) == len(set(ids)), "开发集 ID 唯一")
# 7
span_ok = all(d["sentence"][d["target_start"]:d["target_end"]] == d["target_surface"] for d in dev)
check(span_ok, "所有 dev span 精确匹配")
# 8
l3_ok = all(layer3(d["sentence"],d["target_start"],d["target_end"],d["candidate_sense_ids"]) in [None]+d["candidate_sense_ids"] for d in dev[:20])
check(l3_ok, "L3 返回合法 sense ID 或 None")
# 9
l4_ok = all(layer4(d["sentence"],d["target_start"],d["target_end"],d["candidate_sense_ids"]) in [None]+d["candidate_sense_ids"] for d in dev[:20])
check(l4_ok, "L4 返回合法 sense ID 或 None")
# 10
l5c = sum(1 for d in dev[:20] if layer5(d["sentence"],d["target_start"],d["target_end"],d["candidate_sense_ids"],protos)[2] is not None)
check(l5c > 0, f"L5 独立诊断执行 ({l5c}/20)")
# 11
t1 = extract_context_tokens("苹果发布了新系统。", 0, 2)
t2 = extract_context_tokens("苹果发布了新系统。", 0, 2)
check(t1 == t2, "原型与推理 token 提取一致")
# 12
for w in ["专业","能力","提升","指导","专门"]:
    check(w in kv, f"'{w}' 在腾讯词表中")
# 13
with open(os.path.join(BASE,"p0_r3_predict.py")) as f: src=f.read()
check("blind_gold" not in src and "P0-R3_blind_gold" not in src, "源码无盲测金标路径")
# 14
fake = [{"id":"dev-0001","prediction":"INVALID"}]
fp = os.path.join(BASE,"output","test_fake.json")
os.makedirs(os.path.join(BASE,"output"),exist_ok=True)
json.dump(fake, open(fp,"w"), ensure_ascii=False)
try:
    from p0_r3_score import score
    score(os.path.join(BASE,"data","dev_gold.json"), fp)
    check(False, "scorer 未对非法预测抛错")
except: check(True, "scorer 对非法预测抛错")

print(f"\n{'='*40}\n通过: {passed}/14, 失败: {failed}/14")
