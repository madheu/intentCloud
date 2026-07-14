#!/usr/bin/env python3
"""H13: 海波注入强度校准实验。"""
from __future__ import annotations
import sys, time, csv
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class Tee:
    def __init__(self, t, f): self.t = t; self.f = f
    def write(self, s): self.t.write(s); self.f.write(s); self.f.flush()
    def flush(self): self.t.flush(); self.f.flush()

PROMPT = "写大海"
MAX_NEW = 200
OCEAN_KW = {"大海","海洋","海","波涛","浪","水","蓝","深","广阔","无垠","浩瀚","深邃","海浪","海岸","沙滩","潮","鱼","船","航"}
EXAM_KW = {"答案","选择题","改错","解析","主观题","语法错误","战略","习近平","社会主义","制度"}

def load_model(name):
    print(f"[加载模型] {name}")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    m = AutoModelForCausalLM.from_pretrained(name, quantization_config=bnb,
                                              device_map="auto", trust_remote_code=True)
    m.eval()
    print(f"          {time.time()-t0:.1f}s")
    return m, tok

def evolve_clouds(data_path, model, tokenizer, rounds=8):
    from core.intent_cloud import IntentCloud
    from core.intent_cloud_config import IntentCloudConfig
    layer_idx = model.config.num_hidden_layers // 2
    cfg = IntentCloudConfig()
    s = 0.8
    ca = IntentCloud()
    ca.load_common_sense(str(data_path), model, tokenizer, layer_idx)
    for _ in range(rounds):
        ca.process_interaction({"nature_ocean": s, "emotion_awe": s}, cfg)
        e = ca._edges.get(("nature_ocean", "emotion_fear"))
        if e: e.weight = max(0.0, e.weight * 0.9)
        ca.process_interaction({"nature_ocean": s}, cfg)
    cb = IntentCloud()
    cb.load_common_sense(str(data_path), model, tokenizer, layer_idx)
    for _ in range(rounds):
        cb.process_interaction({"nature_ocean": s, "emotion_fear": s}, cfg)
        e = cb._edges.get(("nature_ocean", "emotion_awe"))
        if e: e.weight = max(0.0, e.weight * 0.9)
        cb.process_interaction({"nature_ocean": s}, cfg)
    for label, c in [("A(敬畏)", ca), ("B(恐惧)", cb)]:
        w1 = c._edges.get(("nature_ocean", "emotion_awe"))
        w2 = c._edges.get(("nature_ocean", "emotion_fear"))
        w1s = f"{w1.weight:.4f}" if w1 else "?"
        w2s = f"{w2.weight:.4f}" if w2 else "?"
        print(f"  海波{label}: awe={w1s}, fear={w2s}")
    return ca, cb, layer_idx

def generate(cloud, prompt, model, tokenizer, scale, layer_idx, seed=42, max_new=MAX_NEW):
    from core.intent_cloud_config import IntentCloudConfig
    from core.bilingual_injector import BilingualInjector
    torch.manual_seed(seed)
    cfg = IntentCloudConfig()
    conv = cloud.process_interaction({"nature_ocean": 0.8}, cfg)
    inj = BilingualInjector(model, tokenizer, activation_threshold=0.0)
    return inj.generate_with_injection(prompt, activations=conv, cloud=cloud,
                                       max_new_tokens=max_new, temperature=0.7, top_p=0.9)

def gen_baseline(model, tokenizer, prompt, seed=42, max_new=MAX_NEW):
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(input_ids=inputs["input_ids"],
                             max_new_tokens=max_new, temperature=0.7,
                             top_p=0.9, do_sample=True,
                             pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

def evaluate(text):
    preview = text.strip().replace("\n", " ")[:200]
    low = preview.lower()
    ocean_hits = sum(1 for kw in OCEAN_KW if kw in low)
    exam_hits = sum(1 for kw in EXAM_KW if kw in low)
    if exam_hits >= 2: status = "失控-考题模式"
    elif ocean_hits == 0 and len(low) > 20: status = "失控-离题"
    elif ocean_hits >= 2: status = "稳定-大海相关"
    elif ocean_hits == 1: status = "临界-弱关联"
    else: status = "无效-无意义"
    return {"preview": preview, "ocean_hits": ocean_hits,
            "exam_hits": exam_hits, "status": status, "length": len(text)}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="E:\\models\\Qwen2.5-7B-Instruct")
    parser.add_argument("--data", default="data/common_sense.json")
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--csv", type=str, default="injection_strength_curve.csv")
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--strengths", type=str, default="0,0.01,0.05,0.1,0.5,1,2,3,5,8,10,15,20")
    args = parser.parse_args()
    if args.log_file:
        p = Path(args.log_file); p.parent.mkdir(parents=True, exist_ok=True)
        fh = open(p, "w", encoding="utf-8")
        sys.stdout = Tee(sys.stdout, fh); sys.stderr = Tee(sys.stderr, fh)
    strengths = [float(s) for s in args.strengths.split(",")]
    print("=" * 70)
    print("H13: 注入强度校准实验")
    print(f"Prompt: \"{PROMPT}\"")
    print(f"强度列表: {strengths}")
    print("=" * 70)
    model, tokenizer = load_model(args.model)
    print("[步骤1] 创建并演化海波...")
    cloud_a, cloud_b, layer_idx = evolve_clouds(args.data, model, tokenizer, args.rounds)
    print()
    print("[步骤2] 基线:")
    results = []
    for s in range(1, args.seeds + 1):
        txt = gen_baseline(model, tokenizer, PROMPT, seed=s*100)
        ev = evaluate(txt)
        results.append({"cloud": "baseline", "strength": 0, "seed": s*100, **ev})
        print(f"  baseline sd={s*100}: {ev['status']} ocean={ev['ocean_hits']}")
    print()
    print("[步骤3] 恐惧方向 B 强度扫描:")
    print(f"{'strength':>7} | {'seed':>4} | {'status':<14} | {'ocean':>4} | preview")
    print("-" * 100)
    for sc in strengths:
        if sc == 0: continue
        for s in range(1, args.seeds + 1):
            sd = s * 100 + int(sc * 10) % 100
            txt = generate(cloud_b, PROMPT, model, tokenizer, sc, layer_idx, seed=sd)
            ev = evaluate(txt)
            results.append({"cloud": "B(恐惧)", "strength": sc, "seed": sd, **ev})
            pv = ev["preview"][:55]
            print(f"{sc:>7.2f} | {sd:>4} | {ev['status']:<14} | {ev['ocean_hits']:>4} | {pv}")
    print()
    print("[步骤4] 敬畏方向 A 对照:")
    key_scales = [sc for sc in strengths if sc in [0.1, 1.0, 3.0, 5.0, 10.0]]
    for sc in key_scales:
        for s in range(1, args.seeds + 1):
            sd = s * 100 + int(sc * 10) % 100
            txt = generate(cloud_a, PROMPT, model, tokenizer, sc, layer_idx, seed=sd)
            ev = evaluate(txt)
            results.append({"cloud": "A(敬畏)", "strength": sc, "seed": sd, **ev})
            pv = ev["preview"][:55]
            print(f"{sc:>7.2f} | {sd:>4} | {ev['status']:<14} | {ev['ocean_hits']:>4} | {pv}")
    print()
    csv_path = Path(args.csv)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["cloud","strength","seed","status","ocean_hits","exam_hits","length","preview"])
        w.writeheader(); w.writerows(results)
    print(f"[CSV] {len(results)} records -> {csv_path}")
    print()
    print("=" * 70)
    print("控制曲线分析 (恐惧方向)")
    print("=" * 70)
    by_sc = defaultdict(list)
    for r in results:
        if r["cloud"] == "B(恐惧)": by_sc[r["strength"]].append(r)
    stable_zone = []
    print(f"{'scale':>6} | stable | crit | out | avg_ocean")
    print("-" * 50)
    for sc in sorted(by_sc.keys()):
        entries = by_sc[sc]
        n = len(entries)
        st = sum(1 for e in entries if e["status"] == "稳定-大海相关")
        cr = sum(1 for e in entries if "临界" in e["status"])
        out = sum(1 for e in entries if "失控" in e["status"])
        ao = sum(e["ocean_hits"] for e in entries) / n
        print(f"{sc:>6.2f} | {st:>5} | {cr:>3} | {out:>3} | {ao:>8.1f}")
        if st > cr + out and st > 0: stable_zone.append(sc)
    if stable_zone:
        print(f"\n-> 稳定引导区间: [{min(stable_zone):.2f}, {max(stable_zone):.2f}]")
        out_scales = sorted(set(r["strength"] for r in results if r["cloud"] == "B(恐惧)" and "失控" in r["status"]))
        if out_scales: print(f"-> 失控临界: {min(out_scales):.2f}")
        else: print("-> 未发现失控")
    else:
        print("\n-> 未找到稳定区间")
    if args.log_file:
        sys.stdout.flush(); sys.stderr.flush(); fh.close()
        print(f"\n日志: {args.log_file}")

if __name__ == "__main__": main()
