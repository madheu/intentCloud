#!/usr/bin/env python3
"""H14a: 引导弹性地图构建器

为海波的主要意图路径扫描稳定区间，输出 elasticity_map.json。
支持分批运行（--paths 参数）和恢复（--json 参数）。
"""
from __future__ import annotations
import sys, time, json, math, os, random
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# 测试路径定义
PATHS = [
    {"id": "nature_ocean->emotion_awe",  "src": "nature_ocean", "tgt": "emotion_awe",
     "type": "extrovert", "prompt": "写大海", "strengths": [0.1, 0.5, 1, 2, 5, 8, 10, 15, 20],
     "notes": "外向型，指向外部对象，天然稳定"},
    {"id": "nature_ocean->emotion_fear", "src": "nature_ocean", "tgt": "emotion_fear",
     "type": "introvert", "prompt": "写大海",
     "strengths": [0.005, 0.01, 0.02, 0.05, 0.1, 0.3, 0.5, 1, 2],
     "notes": "内向型，自激递归，稳定区间极窄"},
    {"id": "nature_ocean->emotion_joy",  "src": "nature_ocean", "tgt": "emotion_joy",
     "type": "extrovert", "prompt": "写大海", "strengths": [0.1, 0.5, 1, 2, 5, 8, 10, 15, 20],
     "notes": "正向情感，预期稳定"},
    {"id": "nature_ocean->emotion_sadness", "src": "nature_ocean", "tgt": "emotion_sadness",
     "type": "introvert", "prompt": "写大海",
     "strengths": [0.005, 0.01, 0.02, 0.05, 0.1, 0.3, 0.5, 1, 2],
     "notes": "负向情感，预期窄路"},
    {"id": "concept_loneliness->emotion_calm", "src": "concept_loneliness", "tgt": "emotion_calm",
     "type": "extrovert", "prompt": "写孤独",
     "strengths": [0.1, 0.5, 1, 2, 5, 8, 10, 15, 20],
     "notes": "情绪调节方向，预期稳定"},
    {"id": "concept_loneliness->emotion_nervous", "src": "concept_loneliness", "tgt": "emotion_nervous",
     "type": "introvert", "prompt": "写孤独",
     "strengths": [0.005, 0.01, 0.02, 0.05, 0.1, 0.3, 0.5, 1, 2],
     "notes": "焦虑方向，预期窄路自激"},
    {"id": "concept_poetry->style_formal", "src": "concept_poetry", "tgt": "style_formal",
     "type": "style", "prompt": "写一首诗",
     "strengths": [0.1, 0.5, 1, 2, 5, 8, 10, 15, 20],
     "notes": "风格型，非情感路径"},
    {"id": "concept_poetry->style_casual", "src": "concept_poetry", "tgt": "style_casual",
     "type": "style", "prompt": "写一首诗",
     "strengths": [0.1, 0.5, 1, 2, 5, 8, 10, 15, 20],
     "notes": "风格型，对比"},
]

OCEAN_KW = {"大海","海洋","海","波涛","浪","水","蓝","广阔","无垠","浩瀚","深邃"}
LONELY_KW = {"孤独","寂寞","独","一人","寂寥","夜晚","寂静"}
POEM_KW = {"诗","诗歌","赋","韵","词","句","章"}

PROMPT_KW = {
    "写大海": OCEAN_KW,
    "写孤独": LONELY_KW,
    "写一首诗": POEM_KW,
}

def load_model(name):
    print(f"[模型] {name}", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    m = AutoModelForCausalLM.from_pretrained(name, quantization_config=bnb,
                                              device_map="auto", trust_remote_code=True)
    m.eval()
    print(f"  {time.time()-t0:.1f}s", flush=True)
    return m, tok

def setup_cloud(data_path, model, tokenizer, test_path):
    """创建海波，激活 src→tgt 边为高权重。"""
    from core.intent_cloud import IntentCloud
    from core.intent_cloud_config import IntentCloudConfig
    layer_idx = model.config.num_hidden_layers // 2
    cloud = IntentCloud()
    cloud.load_common_sense(str(data_path), model, tokenizer, layer_idx)
    cfg = IntentCloudConfig()
    s = "up"
    src, tgt = test_path["src"], test_path["tgt"]
    for _ in range(5):
        cloud.process_interaction({src: 0.8, tgt: 0.8}, cfg)
    e = cloud._edges.get((src, tgt))
    if e:
        print(f"  边权重: {e.weight:.4f}", flush=True)
    return cloud, layer_idx

def generate(cloud, prompt, model, tokenizer, scale, layer_idx, src, seed=42):
    from core.intent_cloud_config import IntentCloudConfig
    from core.bilingual_injector import BilingualInjector
    torch.manual_seed(seed)
    cfg = IntentCloudConfig()
    conv = cloud.process_interaction({src: 0.8}, cfg)
    inj = BilingualInjector(model, tokenizer, activation_threshold=0.0)
    return inj.generate_with_injection(prompt, activations=conv, cloud=cloud,
                                       max_new_tokens=150, temperature=0.7, top_p=0.9)

def evaluate(text, prompt):
    kw = PROMPT_KW.get(prompt, set())
    low = text.lower()[:200]
    hits = sum(1 for k in kw if k in low)
    if hits >= 2:
        return "stable"
    elif hits == 1:
        return "weak"
    else:
        return "out_of_control"

def scan_path(data_path, model, tokenizer, test_path, result, seeds=2):
    pid = test_path["id"]
    print(f"\n{'='*50}", flush=True)
    print(f"[扫描] {pid} ({test_path['type']})", flush=True)
    print(f"  prompt: {test_path['prompt']}", flush=True)
    print(f"  强度: {test_path['strengths']}", flush=True)

    cloud, layer_idx = setup_cloud(data_path, model, tokenizer, test_path)
    prompt = test_path["prompt"]
    entry = {
        "id": pid,
        "type": test_path["type"],
        "src": test_path["src"],
        "tgt": test_path["tgt"],
        "prompt": prompt,
        "notes": test_path["notes"],
        "records": [],
    }

    for sc in test_path["strengths"]:
        if sc == 0:
            continue
        for s in range(1, seeds + 1):
            sd = s * 100 + int(sc * 10) % 100
            t0 = time.time()
            try:
                txt = generate(cloud, prompt, model, tokenizer, sc, layer_idx, test_path["src"], seed=sd)
                status = evaluate(txt, prompt)
                elapsed = time.time() - t0
                entry["records"].append({
                    "strength": sc, "seed": sd, "status": status,
                    "preview": txt.strip().replace("\n", " ")[:80],
                    "time": round(elapsed, 1),
                })
            except Exception as e:
                status = "error"
                entry["records"].append({
                    "strength": sc, "seed": sd, "status": "error",
                    "preview": str(e)[:80], "time": 0,
                })
            preview_str = status
            print(f"  s={sc:>6.3f} sd={sd:>4} {preview_str:15s} ({time.time()-t0:.1f}s)", flush=True)

    # 分析稳定区间
    by_scale = defaultdict(list)
    for r in entry["records"]:
        by_scale[r["strength"]].append(r)

    stable_scales = []
    for sc in sorted(by_scale.keys()):
        entries = by_scale[sc]
        st = sum(1 for e in entries if e["status"] == "stable")
        out = sum(1 for e in entries if e["status"] == "out_of_control")
        n = len(entries)
        if st > n // 2:
            stable_scales.append(sc)

    if stable_scales:
        entry["stable_range"] = [min(stable_scales), max(stable_scales)]
        # 找失控临界
        out_scales = sorted(set(r["strength"] for r in entry["records"]
                                if r["status"] == "out_of_control"))
        entry["out_of_control_threshold"] = min(out_scales) if out_scales else None
    else:
        # 退回到基于记录的判定
        stable_recs = [r for r in entry["records"] if r["status"] == "stable"]
        if stable_recs:
            scs = sorted(set(r["strength"] for r in stable_recs))
            entry["stable_range"] = [min(scs), max(scs)]
        else:
            entry["stable_range"] = None
        entry["out_of_control_threshold"] = min(
            (r["strength"] for r in entry["records"] if r["status"] == "out_of_control"),
            default=None)

    print(f"  稳定区间: {entry.get('stable_range')}", flush=True)
    print(f"  失控临界: {entry.get('out_of_control_threshold')}", flush=True)

    result[pid] = entry
    return entry

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="E:\\models\\Qwen2.5-7B-Instruct")
    parser.add_argument("--data", default="data/common_sense.json")
    parser.add_argument("--output", default="data/elasticity_map.json")
    parser.add_argument("--json", default=None, help="已有结果JSON，仅扫描缺失路径")
    parser.add_argument("--paths", type=str, default=None, help="逗号分隔的路径ID，如: path1,path2")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--log-file", type=str, default=None)
    args = parser.parse_args()

    if args.log_file:
        p = Path(args.log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        fh = open(p, "w", encoding="utf-8")
        old_stdout = sys.stdout
        class Tee:
            def __init__(self, t, f): self.t = t; self.f = f
            def write(self, s): self.t.write(s); self.f.write(s); self.f.flush()
            def flush(self): self.t.flush(); self.f.flush()
        sys.stdout = Tee(sys.stdout, fh)
        sys.stderr = Tee(sys.stderr, fh)

    # 加载已有结果
    result = {}
    if args.json and Path(args.json).exists():
        with open(args.json, "r", encoding="utf-8") as f:
            result = json.load(f)
        print(f"[恢复] 已加载 {len(result)} 条路径结果")

    # 筛选路径
    to_scan = PATHS
    if args.paths:
        selected = set(args.paths.split(","))
        to_scan = [p for p in PATHS if p["id"] in selected]
        print(f"[筛选] 仅扫描 {len(to_scan)} 条路径: {selected}")
    
    # 跳过已有结果
    to_scan = [p for p in to_scan if p["id"] not in result]
    if not to_scan:
        print("[完成] 所有路径已有结果")
        json.dump(result, open(args.output, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        return

    model, tokenizer = load_model(args.model)
    for test_path in to_scan:
        scan_path(args.data, model, tokenizer, test_path, result, seeds=args.seeds)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n[保存] {args.output} ({len(result)} 条路径)")

    # 打印汇总
    print("\n" + "=" * 60)
    print("弹性地图汇总")
    print("=" * 60)
    for pid, info in sorted(result.items()):
        sr = info.get("stable_range")
        oc = info.get("out_of_control_threshold")
        sr_str = f"[{sr[0]:.3f}, {sr[1]:.3f}]" if sr else "N/A"
        oc_str = f"{oc:.3f}" if oc else ">max"
        print(f"  {pid:40s} | 稳定: {sr_str:20s} | 失控: {oc_str:>6s}")

    if args.log_file:
        sys.stdout.flush(); sys.stderr.flush(); fh.close()

if __name__ == "__main__":
    main()
