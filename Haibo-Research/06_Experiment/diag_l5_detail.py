"""诊断 P0-R2.1 L5"""
import json
with open(r"E:\intentCloud\Haibo-Research\06_Experiment\output\blind_predictions.json") as f:
    data = json.load(f)

l5s = [d for d in data if d["resolved_by"] == "L5"]
print(f"L5 触发: {len(l5s)}")
for d in l5s:
    print(f"  id={d['id']} pred={d['prediction']}")

print("\n所有非零 L5 分:")
for d in data:
    det = d.get("debug", {}).get("L5_detail", {})
    if det:
        vals = list(det.values())
        mv = max(vals)
        if mv > 0.01:
            print(f"  id={d['id']:3d} -> {d['prediction']:12s} ({d['resolved_by']:4s}) L5={mv:.3f} score={det}")
