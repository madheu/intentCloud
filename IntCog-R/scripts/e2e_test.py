#!/usr/bin/env python3
"""H11: 端到端验证脚本。

在本地加载 Qwen2.5-7B，跑通完整链路：
  常识加载 → 激活扩散 → 双语者注入 → 生成

用法：
    python scripts/e2e_test.py --model E:\\models\\Qwen2.5-7B-Instruct --quantize 4bit
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── 测试用例 ────────────────────────────────────────────────────────────────

TEST_CASES: list[tuple[str, dict[str, float]]] = [
    (
        "讲讲大海",
        {"nature_ocean": 0.8, "nature_sea": 0.5, "nature_wide": 0.5},
    ),
    (
        "我很开心但也有些紧张",
        {"emotion_joy": 0.8, "emotion_excited": 0.6, "emotion_nervous": 0.6, "emotion_fear": 0.3},
    ),
    (
        "过去和未来有什么区别",
        {"time_past": 0.8, "time_future": 0.8, "time_now": 0.3, "logic_contrast": 0.6},
    ),
    (
        "合作和冲突哪个更重要",
        {"social_cooperation": 0.8, "social_conflict": 0.8, "social_trust": 0.4, "social_doubt": 0.3},
    ),
    (
        "天空为什么是蓝色的",
        {"nature_sky": 0.8, "nature_world": 0.3, "logic_cause": 0.6},
    ),
]

# ── 日志辅助 ────────────────────────────────────────────────────────────────

class Tee:
    """同时输出到终端和文件。"""
    def __init__(self, terminal, file_handle):
        self.terminal = terminal
        self.file = file_handle

    def write(self, text):
        self.terminal.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self):
        self.terminal.flush()
        self.file.flush()


# ── 模型加载 ────────────────────────────────────────────────────────────────

def load_model(model_name: str, quantize: bool = True):
    """加载模型和 tokenizer。"""
    print(f"[加载模型] {model_name}")
    print(f"          量化: {'4-bit' if quantize else 'fp16'}")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    model.eval()
    elapsed = time.time() - t0
    print(f"          用时 {elapsed:.1f}s")
    print(f"          设备: {next(model.parameters()).device}")
    print(f"          参数量: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")
    print(f"          层数: {model.config.num_hidden_layers}")
    print(f"          hidden_dim: {model.config.hidden_size}")
    return model, tokenizer


# ── 意图提取：关键词匹配 ─────────────────────────────────────────────────────

def extract_activations(input_text: str, cloud) -> dict[str, float]:
    """从输入文本中提取激活信号（简单关键词匹配）。

    遍历 cloud._shell 中的所有节点，如果节点的 text 出现在输入中，
    就给该节点一个激活值 0.8。
    """
    activations: dict[str, float] = {}
    for node_id, node in cloud._shell.items():
        if node.text and node.text in input_text:
            activations[node_id] = 0.8
    return activations


# ── 基线生成（无注入）───────────────────────────────────────────────────────

def generate_baseline(model, tokenizer, prompt: str, max_new_tokens: int = 200) -> str:
    """无注入的普通 Qwen 生成。"""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


# ── 主流程 ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="H11: 端到端验证脚本")
    parser.add_argument(
        "--model",
        default="E:\\models\\Qwen2.5-7B-Instruct",
        help="模型路径或名称 (默认: E:\\models\\Qwen2.5-7B-Instruct)",
    )
    parser.add_argument("--no-quantize", action="store_true", help="不使用 4-bit 量化")
    parser.add_argument(
        "--data",
        default="data/common_sense.json",
        help="常识数据路径 (默认: data/common_sense.json)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="最大生成 token 数 (默认: 200)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="日志输出文件路径 (默认: 输出到终端)",
    )
    args = parser.parse_args()

    # 如果指定了日志文件，重定向 stdout 和 stderr
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(log_path, "w", encoding="utf-8")
        import copy
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = Tee(sys.stdout, log_fh)
        sys.stderr = Tee(sys.stderr, log_fh)

    def _print_divider():
        print("=" * 60)

    quantize = not args.no_quantize

    # ── 步骤 1：加载模型 ─────────────────────────────────────────────────────
    model, tokenizer = load_model(args.model, quantize=quantize)

    layer_idx = model.config.num_hidden_layers // 2  # Qwen2.5-7B: 14
    print(f"[准备] layer_idx={layer_idx}, data={args.data}, max_new_tokens={args.max_new_tokens}")
    print()

    # ── 步骤 2：创建 IntentCloud 并加载常识 ──────────────────────────────────
    print("[步骤 1/6] 创建 IntentCloud ...")
    from core.intent_cloud import IntentCloud
    cloud = IntentCloud()
    print(f"          shell 节点数: {len(cloud._shell)}")
    print(f"          边数: {len(cloud._edges)}")

    print(f"[步骤 2/6] 加载常识数据 ({args.data}) ...")
    data_path = PROJECT_ROOT / args.data
    t0 = time.time()
    n_nodes, n_edges = cloud.load_common_sense(str(data_path), model, tokenizer, layer_idx)
    elapsed = time.time() - t0
    print(f"          加载了 {n_nodes} 个节点, {n_edges} 条边 (用时 {elapsed:.1f}s)")

    # 统计有多少节点有 embedding
    emb_count = sum(1 for n in cloud._shell.values() if n.llm_embedding is not None)
    print(f"          带 embedding 的节点: {emb_count}/{n_nodes}")
    print()

    # ── 步骤 3：测试用例循环 ────────────────────────────────────────────────
    from core.intent_cloud_config import IntentCloudConfig
    from core.bilingual_injector import BilingualInjector

    injector = BilingualInjector(model, tokenizer, activation_threshold=0.1)

    for i, (input_text, expected_activations) in enumerate(TEST_CASES):
        print(f"{'='*60}")
        print(f"[测试 {i+1}/{len(TEST_CASES)}] 输入: {input_text}")
        print(f"{'='*60}")

        # a. 意图提取（关键词匹配）
        matched = extract_activations(input_text, cloud)
        print(f"  [提取] 关键词匹配节点: {matched}")

        # 合并期望激活值
        input_signals = {}
        for nid in matched:
            input_signals[nid] = matched[nid]
        for nid, val in expected_activations.items():
            if nid not in input_signals:
                input_signals[nid] = val

        if not input_signals:
            print("  [跳过] 无匹配节点")
            continue

        print(f"  [信号] 输入激活信号: {len(input_signals)} 个节点")

        # b. 激活扩散
        t0 = time.time()
        converged = cloud.process_interaction(input_signals)
        elapsed = time.time() - t0

        # 找出激活值 > 0.05 的节点
        active_nodes = {nid: val for nid, val in converged.items() if val > 0.05}
        sorted_active = sorted(active_nodes.items(), key=lambda x: x[1], reverse=True)
        print(f"  [扩散] 收敛后激活节点 (>{0.05}): {len(active_nodes)}/{len(converged)} 个 (用时 {elapsed:.3f}s)")
        for nid, val in sorted_active[:10]:  # 只打印前 10 个
            node = cloud._shell.get(nid)
            text = node.text if node else nid
            print(f"          {nid} ({text}): {val:.4f}")

        # c. 双语者注入 + 生成
        prompt = input_text

        # 无注入（基线）
        t0 = time.time()
        baseline = generate_baseline(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens)
        baseline_time = time.time() - t0
        print(f"  [基线] 无注入生成 ({baseline_time:.1f}s):")
        print(f"         {baseline[:150]}")

        # 有注入
        t0 = time.time()
        try:
            injection_output = injector.generate_with_injection(
                prompt,
                activations=converged,
                cloud=cloud,
                max_new_tokens=args.max_new_tokens,
                temperature=0.7,
                top_p=0.9,
            )
            injection_time = time.time() - t0
            print(f"  [注入] 双语者注入生成 ({injection_time:.1f}s):")
            print(f"         {injection_output[:150]}")
        except Exception as e:
            injection_time = time.time() - t0
            print(f"  [注入] 失败 ({injection_time:.1f}s): {e}")
            import traceback
            traceback.print_exc()

        print()

    # ── 步骤 4：总结 ────────────────────────────────────────────────────────
    print(f"{'='*60}")
    print("端到端验证完成！")
    print(f"{'='*60}")
    print(f"  模型: {args.model}")
    print(f"  量化: {'4-bit' if quantize else 'fp16'}")
    print(f"  常识节点: {n_nodes}")
    print(f"  常识边: {n_edges}")
    print(f"  测试用例: {len(TEST_CASES)}")

    # 关闭日志文件
    if args.log_file:
        sys.stdout.flush()
        sys.stderr.flush()
        log_fh.close()
        print(f"\n日志已保存到: {args.log_file}", file=old_stdout)


if __name__ == "__main__":
    main()
