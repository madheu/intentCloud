"""
Pilot: 1 condition × 10 seed — 校验情绪分类器可用性
=====================================================
目的：在完整实验前确认：
  (a) 注入 + 生成管线可运行
  (b) 分类器输出有区分度（不全部坍缩到 neutral）
  (c) 连续 embedding 能捕获差异

如果 pilot 显示关键词分类器 → 100% neutral → 放弃关键词，仅用连续 embedding。
"""

import json, sys, os, random, time
from pathlib import Path

# Fix Diviner shadow issue
sys.path = [p for p in sys.path if 'Diviner' not in p]

import numpy as np
import torch
from sklearn.metrics import mutual_info_score
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── 项目路径 ──
PROJECT = Path(r"E:\intentCloud\IntCog-R")
MODEL_PATH = r"E:\intentCloud\models\qwen2.5-1.5b"
DATA_PATH = str(PROJECT / "data" / "common_sense.json")
OUTPUT_DIR = Path(r"E:\intentCloud\Haibo-Research\06_Experiment")
OUTPUT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(PROJECT))
from core.intent_cloud import IntentCloud
from core.bilingual_injector import BilingualInjector
from core.intent_cloud_config import IntentCloudConfig

# ── 配置 ──
PROMPT = "请描述一张桌子"
N_SEEDS = 10
INJECTION_STATES = {
    "FEAR":    {"emotion_fear": 0.8, "nature_ocean": 0.5},
    # Pilot 只跑 FEAR + NONE — 足够校验分类器
    "NONE":    {"nature_ocean": 0.5},  # 已修复：共享话题锚
}

# 关键词分类器（辅助指标）
EMOTION_KEYWORDS = {
    "fear":     ["害怕", "恐惧", "焦虑", "担心", "紧张", "不安", "危险", "威胁", "可怕", "令人畏惧"],
    "awe":      ["敬畏", "壮丽", "宏大", "震撼", "奇迹", "赞叹", "崇高", "磅礴", "浩瀚"],
    "joy":      ["快乐", "喜悦", "开心", "美好", "幸福", "温暖", "欢快", "舒畅", "心旷神怡"],
    "sadness":  ["悲伤", "失落", "孤独", "忧郁", "沉重", "哀伤", "泪水", "凄凉", "寂寥"],
    "neutral":  [],
    "mixed":    [],
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def classify_by_keywords(text):
    """关键词分类器：返回 (label, {class: count})"""
    hits = {}
    for label, words in EMOTION_KEYWORDS.items():
        if not words:
            continue
        n = sum(text.count(w) for w in words)
        if n > 0:
            hits[label] = n
    if not hits:
        return "neutral", {}
    sorted_labels = sorted(hits.items(), key=lambda x: -x[1])
    top, second = sorted_labels[0], sorted_labels[1] if len(sorted_labels) > 1 else ("", 0)
    if top[1] >= second[1] * 2:
        return top[0], hits
    return "mixed", hits

def get_qwen_embedding(model, tokenizer, text, device):
    """用 Qwen 的 last hidden state mean-pooling 作为文本 embedding"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # 取最后一层 hidden states，mean pool over tokens
        last_hidden = outputs.hidden_states[-1]  # (1, seq_len, dim)
        embedding = last_hidden.mean(dim=1).squeeze().cpu().numpy()
    return embedding

def clean_output(text):
    """清洗生成输出：去掉重复前缀"""
    import re
    text = re.sub(r'^(?:[^\u4e00-\u9fff\uff0c\u3002\uff01\uff1f\u3001\u3000]+\s*)+', '', text)
    # 如果清洗后为空，返回原文本做兜底
    if not text.strip():
        return text[:80] if len(text) > 80 else text
    return text[:200]

# ── 主流程 ──
def main():
    print("=" * 60)
    print("Pilot: 1 condition × 10 seed — 分类器校验")
    print(f"Prompt: {PROMPT}")
    print(f"CUDA: {torch.cuda.is_available()}")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    # 1. 加载模型
    print("\n[1/4] 加载模型...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16,
    ).to(device)
    model.eval()
    print(f"  ✓ 模型加载完成 ({time.time()-t0:.1f}s)")

    # 2. 初始化海波
    print("\n[2/4] 初始化 Intent Cloud...")
    cloud = IntentCloud()
    cloud.load_common_sense(DATA_PATH, model, tok, 14)
    # 预热：3 轮 neutral 扩散
    for _ in range(3):
        try:
            cloud.process_interaction({"nature_ocean": 0.5}, IntentCloudConfig())
        except Exception:
            pass  # 预热可能失败，继续
    injector = BilingualInjector(model, tok, activation_threshold=0.1)
    config = IntentCloudConfig()
    print(f"  ✓ 海波初始化完成")

    # 3. 运行实验
    print("\n[3/4] 运行生成...")
    results = []
    for state_label, acts in INJECTION_STATES.items():
        print(f"\n  --- {state_label} ---")
        for seed in range(N_SEEDS):
            set_seed(seed)
            spread = cloud.process_interaction(acts, config)

            raw = injector.generate_with_injection(
                PROMPT, activations=spread, cloud=cloud,
                max_new_tokens=80, temperature=0.7, top_p=0.9,
            )
            text = clean_output(raw)
            kw_label, kw_hits = classify_by_keywords(text)
            kw_hits_str = json.dumps(kw_hits, ensure_ascii=False)

            print(f"    seed={seed:2d} | kw={kw_label:8s} | {text[:60]}")

            results.append({
                "state": state_label,
                "seed": seed,
                "text": text,
                "kw_label": kw_label,
                "kw_hits": kw_hits,
            })

    # 4. 分析
    print("\n[4/4] 分析结果...")
    total = len(results)

    # 4a. 关键词分类器分布
    print(f"\n  ── 关键词分类器分布 (N={total}) ──")
    from collections import Counter
    kw_dist = Counter(r["kw_label"] for r in results)
    for label, cnt in sorted(kw_dist.items()):
        print(f"    {label:10s}: {cnt:2d} ({cnt/total*100:5.1f}%)")

    # 按状态分别统计
    for state in INJECTION_STATES:
        per_state = [r for r in results if r["state"] == state]
        dist = Counter(r["kw_label"] for r in per_state)
        print(f"    [{state}]")
        for label, cnt in sorted(dist.items()):
            print(f"        {label:10s}: {cnt:2d} / {N_SEEDS}")

    # 4b. 连续 embedding
    print(f"\n  ── 连续 embedding 分析 ──")
    embeddings = {}
    for state_label in INJECTION_STATES:
        texts = [r["text"] for r in results if r["state"] == state_label]
        emb_list = [get_qwen_embedding(model, tok, t, device) for t in texts]
        embeddings[state_label] = np.array(emb_list)

    # 类内/类间距离
    for s1 in INJECTION_STATES:
        for s2 in INJECTION_STATES:
            if s1 < s2:
                inner_s1 = np.mean([
                    np.linalg.norm(embeddings[s1][i] - embeddings[s1][j])
                    for i in range(len(embeddings[s1]))
                    for j in range(i+1, len(embeddings[s1]))
                ]) if len(embeddings[s1]) > 1 else 0.0
                inner_s2 = np.mean([
                    np.linalg.norm(embeddings[s2][i] - embeddings[s2][j])
                    for i in range(len(embeddings[s2]))
                    for j in range(i+1, len(embeddings[s2]))
                ]) if len(embeddings[s2]) > 1 else 0.0
                cross = np.mean([
                    np.linalg.norm(embeddings[s1][i] - embeddings[s2][j])
                    for i in range(len(embeddings[s1]))
                    for j in range(len(embeddings[s2]))
                ])
                print(f"    {s1:6s}↔{s2:6s}: cross={cross:.3f}  inner({s1})={inner_s1:.3f}  inner({s2})={inner_s2:.3f}")

    # 4c. 离散 MI（关键词分类器）
    print(f"\n  ── 离散 MI（关键词分类器）──")
    X = [r["state"] for r in results]
    Y = [r["kw_label"] for r in results]
    mi = mutual_info_score(X, Y)
    n_states = len(set(X))
    n_labels = len(set(Y))
    n = len(X)
    mi_bias = (n_states - 1) * (n_labels - 1) / (2 * n)
    mi_corrected = max(0, mi - mi_bias)
    print(f"    MI(raw) = {mi:.4f}, MI(corrected) = {mi_corrected:.4f}")
    print(f"    警告：如果所有输出标为 neutral → MI=0 → 无法判断是注入无效还是分类器太差")

    # 4d. 保存
    output_file = OUTPUT_DIR / "pilot_mi_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "prompt": PROMPT,
                "n_seeds": N_SEEDS,
                "states": list(INJECTION_STATES.keys()),
            },
            "results": [{
                "state": r["state"], "seed": r["seed"],
                "text": r["text"], "kw_label": r["kw_label"],
                "kw_hits": r["kw_hits"],
            } for r in results],
            "kw_distribution": dict(kw_dist),
        }, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    print(f"\n  ✓ 结果保存到 {output_file}")
    print(f"  ⏱ 总耗时: {elapsed:.1f}s ({elapsed/N_SEEDS/len(INJECTION_STATES):.1f}s/生成)")
    print("=" * 60)

    # 5. Pilot 结论
    print("\nPILOT 结论:")
    neutral_ratio = kw_dist.get("neutral", 0) / total * 100
    if neutral_ratio > 80:
        print(f"  ❌ 关键词分类器 {neutral_ratio:.0f}% 坍缩到 neutral → 不可用")
        print("  → 正式实验必须使用连续 embedding + kNN MI 作为主要指标")
    elif neutral_ratio > 50:
        print(f"  ⚠️ 关键词分类器 {neutral_ratio:.0f}% neutral → 边界可用")
        print("  → 以连续 embedding MI 为主，关键词分类保留为交叉验证")
    else:
        print(f"  ✅ 关键词分类器 neutral={neutral_ratio:.0f}% → 可用")
        print("  → 关键词分类可作为辅助指标")

    # 类间/类内距离比
    if len(INJECTION_STATES) > 1:
        s1, s2 = list(INJECTION_STATES.keys())[:2]
        cross = np.mean([
            np.linalg.norm(embeddings[s1][i] - embeddings[s2][j])
            for i in range(len(embeddings[s1]))
            for j in range(len(embeddings[s2]))
        ])
        inner = np.mean([
            np.linalg.norm(embeddings[s1][i] - embeddings[s1][j])
            for i in range(len(embeddings[s1]))
            for j in range(i+1, len(embeddings[s1]))
        ]) if len(embeddings[s1]) > 1 else 0.0
        ratio = cross / (inner + 1e-8)
        if ratio > 1.5:
            print(f"  ✅ 类间/类内距离比 = {ratio:.2f} → 连续 embedding 能区分注入状态")
        else:
            print(f"  ⚠️ 类间/类内距离比 = {ratio:.2f} → 连续 embedding 区分度有限")
            print(f"     可能意味着注入信号太弱，或 N=10 不够")

if __name__ == "__main__":
    main()
