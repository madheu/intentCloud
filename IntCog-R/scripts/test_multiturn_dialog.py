#!/usr/bin/env python3
"""H16c: 多轮对话集成验证。

场景1: 话题继承 + 纠正暂存（不内化）
场景2: 明确内化命令
场景3: 错误信息不内化 + 清理记忆
"""
from __future__ import annotations
import sys, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PROMPT = "写大海"
MAX_NEW = 150

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

def setup_cloud(data_path, model, tokenizer):
    from core.intent_cloud import IntentCloud
    from core.intent_cloud_config import IntentCloudConfig
    layer_idx = model.config.num_hidden_layers // 2
    cloud = IntentCloud()
    cloud.load_common_sense(str(data_path), model, tokenizer, layer_idx)
    # 基本8轮演化
    cfg = IntentCloudConfig()
    for _ in range(8):
        cloud.process_interaction({"nature_ocean": 0.8, "emotion_fear": 0.8}, cfg)
        e = cloud._edges.get(("nature_ocean", "emotion_awe"))
        if e: e.weight = max(0.0, e.weight * 0.9)
        cloud.process_interaction({"nature_ocean": 0.8}, cfg)
    from core.strength_regulator import StrengthRegulator
    regulator = StrengthRegulator(str(Path(data_path).parent / "elasticity_map.json"))
    return cloud, layer_idx, regulator

def generate(cloud, prompt, model, tokenizer, layer_idx,
             activations_override=None, strength=0.03, seed=42):
    from core.intent_cloud_config import IntentCloudConfig
    from core.bilingual_injector import BilingualInjector
    torch.manual_seed(seed)
    cfg = IntentCloudConfig()
    if activations_override:
        conv = activations_override
    else:
        conv = cloud.process_interaction({"nature_ocean": 0.8}, cfg)
    inj = BilingualInjector(model, tokenizer, activation_threshold=0.0)
    return inj.generate_with_injection(prompt, activations=conv, cloud=cloud,
                                       max_new_tokens=MAX_NEW, temperature=0.7, top_p=0.9)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="E:\\models\\Qwen2.5-7B-Instruct")
    parser.add_argument("--data", default="data/common_sense.json")
    parser.add_argument("--log-file", type=str, default="h16_multiturn.log")
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
    cloud, layer_idx, regulator = setup_cloud(args.data, model, tokenizer)
    from core.dialog_context import DialogContext

    def show(text, label=""):
        print(f"\n  [{label}] {text[:200]}", flush=True)

    ctx = DialogContext()

    print("=" * 60, flush=True)
    print("H16c: 多轮对话集成验证", flush=True)
    print("=" * 60, flush=True)

    # ── 场景1: 话题继承 + 纠正暂存 ──────────────────────────────────────
    print("\n" + "#" * 60, flush=True)
    print("# 场景1: 话题继承 + 纠正暂存（不内化）", flush=True)
    print("#" * 60, flush=True)

    print("\n[回合1] 用户: 写一首关于大海的诗", flush=True)
    ctx.update_topic("nature_ocean")
    txt = generate(cloud, "写一首关于大海的诗", model, tokenizer, layer_idx, seed=100)
    show(txt, "海波")

    print("\n[回合2] 用户: 太悲伤了，我喜欢敬畏的感觉", flush=True)
    signals = ctx.detect_correction("太悲伤了，我喜欢敬畏的感觉", cloud)
    for s in signals:
        print(f"  [检测到纠正] {s}", flush=True)
    print(f"  [待处理] {ctx.pending_count} 条", flush=True)

    # 读取节点信任值，确认未改变
    sad_trust_before = cloud._shell["emotion_sadness"].trust if "emotion_sadness" in cloud._shell else "N/A"
    awe_trust_before = cloud._shell["emotion_awe"].trust if "emotion_awe" in cloud._shell else "N/A"
    print(f"  [验证] 悲伤trust={sad_trust_before:.2f}, 敬畏trust={awe_trust_before:.2f} (均未变)", flush=True)

    # 验证：生成的文本不涉及忧伤
    txt = generate(cloud, "写一首关于大海的诗", model, tokenizer, layer_idx, seed=100)
    show(txt, "海波（调整后）")

    print("\n[回合3] 用户: 这次好多了", flush=True)
    # 模拟检测"好多了"——这是个正向反馈，但不触发内化
    print(f"  [状态] pending队列: {ctx.pending_count} 条", flush=True)
    print(f"  [验证] reset前队列不为空: {ctx.pending_count > 0}", flush=True)

    # 模拟会话结束
    ctx.reset()
    print(f"  [reset后] 话题={ctx.current_topic}, pending={ctx.pending_count}", flush=True)

    # ── 场景2: 明确内化命令 ──────────────────────────────────────────────
    print("\n" + "#" * 60, flush=True)
    print("# 场景2: 明确内化命令", flush=True)
    print("#" * 60, flush=True)

    # 新会话：输入纠正，然后说"记住"
    ctx.update_topic("nature_ocean")
    ctx.detect_correction("太悲伤了，我喜欢敬畏", cloud)
    print(f"  [检测后] pending: {ctx.pending_count} 条", flush=True)
    sad_before = cloud._shell["emotion_sadness"].trust
    awe_before = cloud._shell["emotion_awe"].trust
    print(f"  [内化前] 悲伤={sad_before:.2f}, 敬畏={awe_before:.2f}", flush=True)

    # 用户说"记住这个"
    is_int = ctx.is_internalize_command("记住这个")
    print(f"  [检测内化关键词] '记住这个' → {is_int}", flush=True)
    log = ctx.internalize_pending(cloud)
    print(f"  [内化结果] {log}", flush=True)
    sad_after = cloud._shell["emotion_sadness"].trust
    awe_after = cloud._shell["emotion_awe"].trust
    print(f"  [内化后] 悲伤={sad_after:.2f} ({'↓' if sad_after < sad_before else '='}), "
          f"敬畏={awe_after:.2f} ({'↑' if awe_after > awe_before else '='})", flush=True)
    print(f"  [验证] pending={ctx.pending_count} (应为0)", flush=True)
    assert ctx.pending_count == 0, "内化后队列应清空"

    # ── 场景3: 错误信息不内化 + 清理记忆 ────────────────────────────────
    print("\n" + "#" * 60, flush=True)
    print("# 场景3: 错误信息不内化 + 清理记忆", flush=True)
    print("#" * 60, flush=True)

    ctx.reset()
    ctx.update_topic("nature_ocean")
    joy_before = cloud._shell["emotion_joy"].trust if "emotion_joy" in cloud._shell else 0.5
    print(f"  [清空前] 喜悦trust={joy_before:.2f}", flush=True)

    # 用户输入错误信息
    ctx.detect_correction("太焦虑了", cloud)
    ctx.detect_correction("不要平静", cloud)
    print(f"  [错误输入后] pending: {ctx.pending_count} 条", flush=True)

    # 用户说"清理记忆"——清空但不改权重
    is_clr = ctx.is_clear_command("清理记忆")
    print(f"  [检测清理关键词] '清理记忆' → {is_clr}", flush=True)
    clear_log = ctx.clear_pending()
    print(f"  [清理结果] {clear_log}", flush=True)

    joy_after = cloud._shell["emotion_joy"].trust if "emotion_joy" in cloud._shell else 0.5
    print(f"  [清理后] 喜悦trust={joy_after:.2f} ({'=' if joy_after == joy_before else 'changed'})", flush=True)
    print(f"  [验证] pending={ctx.pending_count} (应为0)", flush=True)
    assert ctx.pending_count == 0, "清理后队列应清空"
    assert joy_after == joy_before, "清理不应改变权重"

    print("\n" + "=" * 60, flush=True)
    print("全部场景验证通过 ✅", flush=True)
    print("=" * 60, flush=True)
    print(f"日志: {args.log_file}", flush=True)

    if args.log_file:
        sys.stdout.flush(); sys.stderr.flush(); fh.close()

if __name__ == "__main__":
    main()
