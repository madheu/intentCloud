#!/usr/bin/env python3
"""H15c: 视角转换效果验证实验。

对比4个条件：
  1. 基线（无注入）
  2. 单一恐惧（H12/H13的做法）
  3. 基础视角扩展（H15a）
  4. 恐惧链专属转换（H15b）

Prompt: "写大海"
海波状态：恐惧方向（ocean->fear 高权重）
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PROMPT = "写大海"
MAX_NEW = 200

def load_model(name):
    print(f"[模型] {name}", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    m = AutoModelForCausalLM.from_pretrained(name, quantization_config=bnb,
                                              device_map="auto", trust_remote_code=True)
    m.eval()
    print(f"  {time.time()-t0:.1f}s", flush=True)
    return m, tok

def setup_fear_cloud(data_path, model, tokenizer):
    from core.intent_cloud import IntentCloud
    from core.intent_cloud_config import IntentCloudConfig
    layer_idx = model.config.num_hidden_layers // 2
    cloud = IntentCloud()
    cloud.load_common_sense(str(data_path), model, tokenizer, layer_idx)
    cfg = IntentCloudConfig()
    for _ in range(8):
        cloud.process_interaction({"nature_ocean": 0.8, "emotion_fear": 0.8}, cfg)
        e = cloud._edges.get(("nature_ocean", "emotion_awe"))
        if e: e.weight = max(0.0, e.weight * 0.9)
        cloud.process_interaction({"nature_ocean": 0.8}, cfg)
    w = cloud._edges.get(("nature_ocean", "emotion_fear"))
    print(f"  恐惧边权重: {w.weight if w else '?'}", flush=True)
    return cloud, layer_idx

def gen_baseline(model, tokenizer, prompt, seed=42):
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(input_ids=inputs["input_ids"], max_new_tokens=MAX_NEW,
                             temperature=0.7, top_p=0.9, do_sample=True,
                             pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

def gen_with_injection(cloud, prompt, model, tokenizer, layer_idx,
                       converter=None, scale=1.0, seed=42, mode="auto"):
    """生成文本。如果提供了 converter，使用视角转换。"""
    from core.intent_cloud_config import IntentCloudConfig
    from core.bilingual_injector import BilingualInjector
    torch.manual_seed(seed)
    cfg = IntentCloudConfig()
    conv = cloud.process_interaction({"nature_ocean": 0.8}, cfg)
    inj = BilingualInjector(model, tokenizer, activation_threshold=0.0)

    if converter:
        result = converter.convert(
            "nature_ocean", "emotion_fear", conv, cloud, scale,
            mode=mode
        )
        final_acts = result["activations"]
        final_scale = result["strength"]
        log = result["log"]
        print(f"  [转换] {log}", flush=True)
    else:
        final_acts = conv
        final_scale = scale

    out = inj.generate_with_injection(
        prompt, activations=final_acts, cloud=cloud,
        max_new_tokens=MAX_NEW, temperature=0.7, top_p=0.9,
    )
    return out.strip()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="E:\\models\\Qwen2.5-7B-Instruct")
    parser.add_argument("--data", default="data/common_sense.json")
    parser.add_argument("--map", default="data/elasticity_map.json")
    parser.add_argument("--log-file", type=str, default=None)
    args = parser.parse_args()

    if args.log_file:
        p = Path(args.log_file); p.parent.mkdir(parents=True, exist_ok=True)
        fh = open(p, "w", encoding="utf-8")
        class Tee:
            def __init__(self, t, f): self.t = t; self.f = f
            def write(self, s): self.t.write(s); self.f.write(s); self.f.flush()
            def flush(self): self.t.flush(); self.f.flush()
        sys.stdout = Tee(sys.stdout, fh)
        sys.stderr = Tee(sys.stderr, fh)

    model, tokenizer = load_model(args.model)
    print("=" * 60)
    print("H15c: 视角转换效果验证")
    print(f"Prompt: \"{PROMPT}\"")
    print("=" * 60, flush=True)

    # 设置恐惧方向海波
    print("\n[设置] 恐惧方向海波...")
    cloud, layer_idx = setup_fear_cloud(args.data, model, tokenizer)

    # 加载调节器和转换器
    from core.strength_regulator import StrengthRegulator
    from core.perspective_converter import PerspectiveConverter
    regulator = StrengthRegulator(args.map)
    converter = PerspectiveConverter(regulator=regulator)
    print(f"  弹性地图: {regulator.path_count} 条路径", flush=True)

    # 4 个条件 × 2 次生成
    conditions = [
        ("基线(无注入)",     "baseline",   None),
        ("单一恐惧",         "fear_only",  None),
        ("基础视角扩展",     "basic_persp", converter),
        ("恐惧链专属转换",   "fear_chain",  converter),
    ]

    for cond_name, cond_type, conv in conditions:
        print(f"\n{'='*60}", flush=True)
        print(f"[条件] {cond_name}", flush=True)
        print(f"{'='*60}", flush=True)

        for run in range(1, 3):
            seed = run * 100
            print(f"\n  Run {run}/2 seed={seed}:", flush=True)

            if cond_type == "baseline":
                txt = gen_baseline(model, tokenizer, PROMPT, seed=seed)
            elif cond_type == "fear_only":
                txt = gen_with_injection(cloud, PROMPT, model, tokenizer,
                                         layer_idx, scale=0.03, seed=seed)
            elif cond_type == "basic_persp":
                txt = gen_with_injection(cloud, PROMPT, model, tokenizer,
                                         layer_idx, converter=conv,
                                         scale=0.5, seed=seed)
            elif cond_type == "fear_chain":
                txt = gen_with_injection(cloud, PROMPT, model, tokenizer,
                                         layer_idx, converter=conv,
                                         scale=5.0, seed=seed)

            # 打印输出
            for line in txt.strip().split("\n")[:10]:
                print(f"    {line}", flush=True)
            if len(txt.strip().split("\n")) > 10:
                print(f"    ...({len(txt)}字)", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("实验完成", flush=True)
    print("=" * 60, flush=True)

    if args.log_file:
        sys.stdout.flush(); sys.stderr.flush(); fh.close()
        print(f"日志: {args.log_file}")

if __name__ == "__main__":
    main()
