#!/usr/bin/env python3
"""Talk to Haibo — 图状态可见 + 纯 prompt 通信（无注入）
====================================================
旧版本 chat_with_haibo.py 用 embedding 注入试图控制 LLM 生成方向。
Pilot 实验证明这条路走不通（互信息 ≈ 0）。

新版本：海波通过自然语言 prompt 与 LLM 通信。
海波负责语义组织（图扩散 → Decision → Prompt），LLM 只负责语言表达。
"""

from __future__ import annotations
import sys, time, argparse, torch, re
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = r"E:\intentCloud\models\qwen2.5-1.5b"

# ── 节点中文名映射 ──
NODE_NAMES = {
    "nature_ocean":"大海","nature_sky":"天空","nature_life":"生命","nature_wide":"壮丽",
    "nature_sea":"海洋","nature_change":"变化","nature_world":"世界",
    "emotion_fear":"恐惧","emotion_joy":"喜悦","emotion_sadness":"悲伤",
    "emotion_anger":"愤怒","emotion_calm":"宁静","emotion_awe":"敬畏",
    "emotion_nervous":"紧张","emotion_excited":"兴奋",
    "concept_poetry":"诗歌","concept_loneliness":"孤独",
    "concept_beginning":"开端","concept_end":"终结","concept_boundary":"边界",
    "time_past":"过去","time_future":"未来","time_now":"当下","time_eternity":"永恒",
    "space_here":"近处","space_there":"远处","space_far":"远方",
    "logic_cause":"因果","logic_contrast":"对比","logic_progression":"递进",
    "social_cooperation":"合作","social_conflict":"冲突",
    "social_trust":"信任","social_doubt":"怀疑",
    "style_formal":"正式","style_casual":"随意",
}

TOPIC_MAP = {
    ("海","大海","海洋","浪","水","鱼","船"):"nature_ocean",
    ("天空","天","云","星","月"):"nature_sky",
    ("悲伤","恐惧","害怕","焦虑","担心","怕","紧张","不安"):"emotion_fear",
    ("快乐","开心","喜悦","爱","敬畏","美丽","美好"):"emotion_joy",
    ("平静","安宁","宁静","平和"):"emotion_calm",
    ("诗","诗歌","写诗","押韵","韵"):"concept_poetry",
    ("孤独","孤单","寂寞"):"concept_loneliness",
    ("生命","人生","意义","活着","死亡","死"):"nature_life",
    ("壮丽","宏大","宇宙","星空"):"nature_wide",
    ("愤怒","生气","怒"):"emotion_anger",
}

# ── 系统节点（不展示给用户）──
SYSTEM_NODES = {"self", "user", "task", "goal", "neo_self", "neo_intent",
                "anchor_user", "anchor_dialog_self", "anchor_observe", "anchor_ground"}


def clean_output(text):
    """清洗 LLM 输出：去重前缀、截断多轮幻觉"""
    if not text or not text.strip():
        return "..."
    original = text.strip()
    if "<|im_end|>" in original:
        original = original.split("<|im_end|>")[0]
    for marker in ["\n\n你好", "\n\n好", "\n\n我", "\n\n那", "\n\n谢谢"]:
        if marker in original:
            original = original.split(marker)[0]
            break
    for para in original.split('\n'):
        if len(re.findall(r'[\u4e00-\u9fff]', para)) >= 8:
            return para.strip()[:200]
    match = re.search(r'[\u4e00-\u9fff]', original)
    if match:
        start = match.start()
        return original[max(0, start):min(start + 200, len(original))].strip()
    return original[:150].strip()


def build_graph_state_display(acts, spread):
    """构建图状态的终端可视化"""
    lines = []
    # 触发节点
    if acts:
        trig = ", ".join(f"{NODE_NAMES.get(k, k)}({v:.1f})" for k, v in acts.items())
        lines.append(f"触发: {trig}")

    # 扩散后的语义节点
    semantic = [(k, v) for k, v in spread.items()
                if v > 0.05 and k not in acts and k in NODE_NAMES and k not in SYSTEM_NODES]
    semantic.sort(key=lambda x: x[1], reverse=True)
    if semantic:
        parts = [f"{NODE_NAMES.get(k, k)}↑{v:.2f}" for k, v in semantic[:8]]
        lines.append(f"扩散: {', '.join(parts)}")

    # 统计
    n_active = sum(1 for v in spread.values() if v > 0.05)
    lines.append(f"活跃: {n_active} 节点")
    return lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL_PATH)
    p.add_argument("--data", default=str(PROJECT / "data/common_sense.json"))
    args = p.parse_args()

    # ── 加载模型 ──
    print("加载模型...", end=" ", flush=True)
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    m.eval()
    print(f"{time.time() - t0:.1f}s ({device})")

    # ── 初始化海波 ──
    from core.intent_cloud import IntentCloud
    from core.intent_cloud_config import IntentCloudConfig
    from core.graph_interpreter import GraphInterpreter
    from core.prompt_builder import PromptBuilder

    cloud = IntentCloud()
    cloud.load_common_sense(str(args.data), m, tok, 14)
    cfg = IntentCloudConfig()
    # 预热
    for _ in range(3):
        cloud.process_interaction({"nature_ocean": 0.6}, cfg)

    gi = GraphInterpreter()
    pb = PromptBuilder()

    print("=" * 60)
    print("  海波 Talk — 图状态 + Prompt 通信")
    print("  输入 q 退出")
    print("=" * 60)

    while True:
        try:
            user = input("\n[你] ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in ("q", "quit", "exit"):
            break

        # 1. 话题检测
        acts = {}
        for kws, node in TOPIC_MAP.items():
            if any(kw in user for kw in kws):
                acts[node] = min(1.0, acts.get(node, 0) + 0.5)

        # 2. 图扩散
        spread = cloud.process_interaction(acts, cfg) if acts else cloud.process_interaction({}, cfg)

        # 3. 图解释 → Decision
        d = gi.interpret(user, acts)
        system_prompt = pb.build(d)

        # 4. 构建 Qwen chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # 5. LLM 纯 prompt 生成（无注入）
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = m.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=120,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
                pad_token_id=tok.eos_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        raw = tok.decode(new_tokens, skip_special_tokens=True)

        # 6. 清洗 + 显示
        text = clean_output(raw)

        # ── 显示 ──
        print()
        print("━" * 60)
        for line in build_graph_state_display(acts, spread):
            print(line)
        print(f"System: {system_prompt}")
        print(f"输出: {text}")
        print("━" * 60)

    print("\n再见！")


if __name__ == "__main__":
    main()
