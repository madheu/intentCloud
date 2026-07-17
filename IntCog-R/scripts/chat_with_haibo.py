#!/usr/bin/env python3
"""Haibo interactive chat — Qwen2.5-0.5B + 中文双语者注入."""
from __future__ import annotations
import sys, time, argparse, torch, re
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = r"E:\intentCloud\models\qwen2.5-1.5b"


def clean_output(text: str, aggressive: bool = False) -> str:
    """提取有效中文回应。

    优先策略：找第一个中文段落。
    如果段落太短或找不到中文，返回原始输出的截断版本。
    """
    if not text or not text.strip():
        return "..."

    original = text.strip()

    # 策略 1：提取第一个有意义的中文段落（≥8 字）
    # 按段落分割，找第一个含足够多中文的段落
    paragraphs = original.split('\n')
    for para in paragraphs:
        # 统计中文字符数
        chinese_count = len(re.findall(r'[\u4e00-\u9fff]', para))
        if chinese_count >= 8:
            return para.strip()[:200]

    # 策略 2：直接清理前导杂音
    # 找到第一个中文字符的位置，从那里开始
    match = re.search(r'[\u4e00-\u9fff]', original)
    if match:
        start = match.start()
        # 再往前找一句话的合理起始（冒号、引号后，或句首）
        ctx_before = original[max(0, start-3):start]
        return original[max(0, start):min(start+200, len(original))].strip()

    # 策略 3：实在找不到中文，返回前 150 字
    return original[:150].strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL_PATH)
    p.add_argument("--data", default=str(PROJECT / "data/common_sense.json"))
    args = p.parse_args()

    print("[海波] 加载模型...", end=" ", flush=True)
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    m.to(device)
    m.eval()
    layer_idx = 14  # middle of 28 Qwen layers
    print(f"{time.time()-t0:.1f}s ({device})")

    print("[海波] 加载常识图...", end=" ", flush=True)
    from core.intent_cloud import IntentCloud
    from core.intent_cloud_config import IntentCloudConfig
    from core.bilingual_injector import BilingualInjector
    from core.graph_interpreter import GraphInterpreter

    cloud = IntentCloud()
    cloud.load_common_sense(str(args.data), m, tok, layer_idx)
    cfg = IntentCloudConfig()

    for _ in range(5):
        cloud.process_interaction({"nature_ocean": 0.6}, cfg)

    inj = BilingualInjector(m, tok, activation_threshold=0.1)
    gi = GraphInterpreter()

    print("OK")
    print(); print("=" * 50)
    print("  海波（Hypergraph）交互式聊天")
    print("  输入 q 退出")
    print("=" * 50)

    # ═══ 中文话题检测 ═══
    topic_map = {
        ("海", "大海", "海洋", "浪", "水", "鱼", "船"): "nature_ocean",
        ("天空", "天", "云", "星", "月"): "nature_sky",
        ("悲伤", "恐惧", "害怕", "焦虑", "担心", "怕", "紧张", "不安"): "emotion_fear",
        ("快乐", "开心", "喜悦", "爱", "敬畏", "美丽", "美好"): "emotion_joy",
        ("平静", "安宁", "宁静", "平和"): "emotion_calm",
        ("诗", "诗歌", "写诗", "押韵", "韵"): "concept_poetry",
        ("孤独", "孤单", "寂寞"): "concept_loneliness",
        ("生命", "人生", "意义", "活着", "死亡", "死"): "nature_life",
        ("壮丽", "宏大", "宇宙", "星空"): "nature_wide",
        ("愤怒", "生气", "怒"): "emotion_anger",
    }
    tone_map = {
        "neutral": "用自然的语气说话。",
        "warm": "用温暖的语气说话。",
        "calm": "用平静的语气说话。",
        "gentle": "用温柔的语气说话。",
    }

    while True:
        try:
            user = input("\n[你] ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not user:
            continue
        if user.lower() in ("q", "quit", "exit"):
            break

        # 话题检测 → 意图云激活
        acts = {}
        for kws, node in topic_map.items():
            if any(kw in user for kw in kws):
                acts[node] = min(1.0, acts.get(node, 0) + 0.5)

        # 话题检测 → 意图云扩散
        # 接住扩散结果：原始 acts 只触发 2-3 个节点，扩散后 40+ 个节点被激活
        spread = cloud.process_interaction(acts, cfg) if acts else cloud.process_interaction({}, cfg)

        # GraphInterpreter → Decision
        d = gi.interpret(user, acts)

        # 构建中文系统提示（极短！）
        sp = "你是海波。"
        if d.topic:
            sp += f" 当前话题：{d.topic}。"
        sp += " " + tone_map.get(d.tone, "用自然的语气说话。")

        # Qwen 聊天模板
        messages = [
            {"role": "system", "content": sp},
            {"role": "user", "content": user},
        ]
        full_prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # 生成 — 把扩散后的全图激活传给注入器
        try:
            torch.manual_seed(int(time.time() * 1000) % 10000)
            txt = inj.generate_with_injection(
                full_prompt, activations=spread, cloud=cloud,
                max_new_tokens=80, temperature=0.7, top_p=0.9,
                repetition_penalty=1.15, no_repeat_ngram_size=3,
            )
            # 截断到第一轮对话结束
            # Qwen 生成时会继续编造后续对话，遇到 im_end / double newline / 问句 就停
            raw = txt.strip()
            # 切掉 <|im_end|> 及之后的所有内容
            if "<|im_end|>" in raw:
                raw = raw.split("<|im_end|>")[0]
            # 如果出现第二个对话轮次（以 \n\n 用户话语开头），截断
            for marker in ["\n\n你好", "\n\n好", "\n\n我", "\n\n那", "\n\n谢谢", "\n\n嗯"]:
                if marker in raw:
                    raw = raw.split(marker)[0]
                    break
            out = clean_output(raw)
            if out == "..." and raw.strip():
                # clean_output 没找到中文，直接用原始截断
                out = raw.strip()[:150]
            print(f"[海波] {out}")
        except Exception as e:
            print(f"[海波] 出错了：{e}")
            import traceback
            traceback.print_exc()

    print("\n[海波] 再见！")


if __name__ == "__main__":
    main()
