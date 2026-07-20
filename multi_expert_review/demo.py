#!/usr/bin/env python3
"""
演示：基于 Haibo 审稿委员会的四角色专家图。

用现有的 4 个角色（PI、AI Reviewer、Cognitive Scientist、Control Scientist）
构建一个有向标记图，通过 workspace 发布/订阅并行审稿，
最后通过 DecisionAggregator 检测冲突并做出 PI 决策。

运行方式：
    cd multi_expert_review
    python demo.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# 添加父目录以使包导入正常工作
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from multi_expert_review.models import (
    EdgeType,
    ExpertRole,
    ReviewEdge,
    ReviewGraph,
    ReviewOutput,
)
from multi_expert_review.workspace import (
    DecisionAggregator,
    ExpertReviewer,
    ReviewWorkspace,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. 定义四个专家角色
# ═══════════════════════════════════════════════════════════════════════════

AI_REVIEWER = ExpertRole(
    id="ai-reviewer",
    name="AI 系统审查员",
    specialty="AI/ML 系统分析、文献定位、实验设计、代码证据链",
    prompt_template=(
        "你是一位 AI 系统审查员（AI Systems Reviewer）。\n"
        "审查角度：代码级证据 > 文档声称 > 理论论证。\n"
        "审查方法：①定位与现有工作的差异 ②分析代码实现是否支撑声称 "
        "③设计可复现实验验证核心假设 ④评估项目失败的最可能路径。\n"
        "输出要求：每条结论标注证据强度 Strong / Partial / Unknown。"
    ),
    weight=0.8,
)

COG_SCIENTIST = ExpertRole(
    id="cog-scientist",
    name="计算认知科学家",
    specialty="认知架构评估、反馈回路分析、哲学-工程对齐、文献举证",
    prompt_template=(
        "你是一位计算认知科学家（Computational Cognitive Scientist）。\n"
        "审查角度：认知完整性 > 工程完备性。\n"
        "审查框架：①系统是否构成闭环认知动力系统（L1/L2/L3 反馈区分）"
        "②控制信号是否被正确接收 ③哲学宣称与工程实现的差距。\n"
        "输出要求：引用认知科学文献（ACT-R / GWT / PDP / NARS）支持判断。"
    ),
    weight=0.9,
)

CONTROL_SCIENTIST = ExpertRole(
    id="control-scientist",
    name="非线性控制工程师",
    specialty="控制论分析、谱半径分析、数学证伪、反例构造",
    prompt_template=(
        "你是一位非线性控制工程师（Nonlinear Control Engineer）。\n"
        "审查角度：数学可证明 > 工程可调。\n"
        "审查方法：①建模系统状态空间和动力学方程 "
        "②分析稳定性（谱半径、李雅普诺夫）③构造最小反例 "
        "④定量评估控制器耦合强度。\n"
        "输出要求：如果给出反例，附可复现的最小代码。"
    ),
    weight=0.85,
)

PI = ExpertRole(
    id="pi",
    name="项目负责人（PI）",
    specialty="综合决策、资源调度、方向判断",
    prompt_template=(
        "你是项目负责人（Principal Investigator）。\n"
        "角色：收集所有审稿人的意见，检测冲突，做出最终决策。\n"
        "决策框架：①确认核心假设是否被证伪 "
        "②评估从当前状态继续投资的回报率 "
        "③提出明确的路径建议（A 继续 / B 缩减 / C 转向）。"
    ),
    weight=1.0,
)


# ═══════════════════════════════════════════════════════════════════════════
# 2. 构建图
# ═══════════════════════════════════════════════════════════════════════════

def build_review_graph() -> ReviewGraph:
    """Build the Haibo review committee graph with typed edges."""

    g = ReviewGraph()

    for role in [AI_REVIEWER, COG_SCIENTIST, CONTROL_SCIENTIST, PI]:
        g.add_role(role)

    # ── PI delegates analysis tasks to all three reviewers ──
    g.add_edge("pi", "ai-reviewer", EdgeType.DELEGATES, 1.0)
    g.add_edge("pi", "cog-scientist", EdgeType.DELEGATES, 1.0)
    g.add_edge("pi", "control-scientist", EdgeType.DELEGATES, 1.0)

    # ── AI Reviewer depends on / is informed by the other two ──
    # In the actual review, AI Reviewer explicitly said:
    # "Cognitive Scientist 和 Control Scientist 已经覆盖了三个致命问题"
    g.add_edge("control-scientist", "ai-reviewer", EdgeType.INFORMS, 0.8)
    g.add_edge("cog-scientist", "ai-reviewer", EdgeType.INFORMS, 0.8)
    g.add_edge("cog-scientist", "ai-reviewer", EdgeType.DEPENDS_ON, 0.3)

    # ── Possible conflict: Cog Sci and Control Sci may disagree ──
    # Cog Sci said "闭环缺失" is fatal; Control Sci said same thing but
    # from math perspective. They agreed here, but the edge exists for
    # scenarios where they disagree.
    g.add_edge("cog-scientist", "control-scientist", EdgeType.CONFLICTS_WITH, 0.5)
    g.add_edge("control-scientist", "cog-scientist", EdgeType.CONFLICTS_WITH, 0.5)

    # ── All three inform the PI ──
    g.add_edge("ai-reviewer", "pi", EdgeType.INFORMS, 1.0)
    g.add_edge("cog-scientist", "pi", EdgeType.INFORMS, 1.0)
    g.add_edge("control-scientist", "pi", EdgeType.INFORMS, 1.0)

    # ── PI's decision feeds back to all roles ──
    g.add_edge("pi", "ai-reviewer", EdgeType.FEEDBACK, 0.5)
    g.add_edge("pi", "cog-scientist", EdgeType.FEEDBACK, 0.5)
    g.add_edge("pi", "control-scientist", EdgeType.FEEDBACK, 0.5)

    return g


# ═══════════════════════════════════════════════════════════════════════════
# 3. 专家 Reviewer 实现
# ═══════════════════════════════════════════════════════════════════════════

class AIReviewer(ExpertReviewer):
    """AI 系统审查员实现"""

    async def produce_review(self, frame: dict[str, Any]) -> ReviewOutput:
        doc = frame.get("document", "")
        refs = frame.get("previous_reviews", [])

        # 模拟分析过程
        conclusions = [
            "代码质量: 核心模块有代码实现支持",
            "文献定位: 与 Representation Engineering 差异为实现细节非方法论创新",
            "实验设计: 需要互信息实验验证 H1 假设",
            "文献定位: 与现有工作重叠程度高 -> 增量贡献存疑",
        ]
        evidence = [
            "core/intent_cloud.py:617 — 扩散收敛有实现",
            "bilingual_injector.py:200 — 注入到 input_embeds 有实现",
            "但无 ablation 对比数据",
        ]
        content = (
            f"=== AI 审查员报告 ===\n"
            f"审查对象: {doc[:80]}...\n"
            f"引用上游: {refs if refs else '无'}\n"
            f"结论:\n" + "\n".join(f"  [{c}]" for c in conclusions)
        )

        return ReviewOutput(
            role_id="ai-reviewer",
            content=content,
            confidence=0.75,
            references=refs,
            conclusions=conclusions,
            evidence=evidence,
        )


class CognitiveScientist(ExpertReviewer):
    """计算认知科学家实现"""

    async def produce_review(self, frame: dict[str, Any]) -> ReviewOutput:
        doc = frame.get("document", "")

        conclusions = [
            "反馈层级: L1 扩散内反馈存在, L2 状态间反馈被 H16 切断, L3 学习反馈完全缺失",
            "认知完整性: 不构成闭环认知动力系统（循环论证）",
            "控制信号: 4 tok vs 500+ tok -> 1 瓦特控制 1000 瓦特",
            "哲学对齐: 'LLM只是嘴巴' vs 工程现实存在系统错位",
            "建议: 路径 C — 分离 L2/L3 反馈 (50 行代码)",
        ]
        evidence = [
            "引用 ACT-R: 单个 production cycle 内有前馈无反馈",
            "引用 GWT: 同时包含前馈的无意识和闭环的有意识阶段",
            "引用 Collins & Loftus 1975: 激活扩散只是检索机制",
        ]
        content = (
            f"=== 认知科学家报告 ===\n"
            f"审查对象: {doc[:80]}...\n"
            f"结论:\n" + "\n".join(f"  ◆ {c}" for c in conclusions)
        )

        return ReviewOutput(
            role_id="cog-scientist",
            content=content,
            confidence=0.85,
            conclusions=conclusions,
            evidence=evidence,
        )


class ControlScientist(ExpertReviewer):
    """非线性控制工程师实现"""

    async def produce_review(self, frame: dict[str, Any]) -> ReviewOutput:
        doc = frame.get("document", "")

        conclusions = [
            "稳定性: 赫布学习与扩散收敛数学上不兼容",
            "反例: 3 节点全连通图在 3 次高共现交互后 ρ(A)=1.099 > 1 -> 发散",
            "推荐: 方案 D — 权重入度列和归一化 (15 行代码)",
            "耦合强度: input embedding 信号与 RLHF 预训练的 std 差 33 倍",
            "边界控制问题: LLM 是独立动力系统, 海波施加微弱偏置",
        ]
        evidence = [
            "E:/intentCloud/tools/test_haibo_spectral_radius.py — 反例已验证",
            "E:/intentCloud/tools/test_haibo_minimal_fix.py — 5 套方案对比",
            "谱半径分析: ρ(W) < 1 是扩散收敛的充要条件",
        ]
        content = (
            f"=== 控制科学家报告 ===\n"
            f"审查对象: {doc[:80]}...\n"
            f"结论:\n" + "\n".join(f"  ■ {c}" for c in conclusions)
        )

        return ReviewOutput(
            role_id="control-scientist",
            content=content,
            confidence=0.95,
            conclusions=conclusions,
            evidence=evidence,
        )


# ═══════════════════════════════════════════════════════════════════════════
# 4. 主流程
# ═══════════════════════════════════════════════════════════════════════════

async def main() -> None:
    print("=" * 60)
    print("  多专家审稿委员会 — 图结构演示")
    print("=" * 60)

    # ── 构建图 ──
    g = build_review_graph()
    print("\n📊 专家图信息:")
    print(f"   角色数: {len(g.roles)}")
    print(f"   边数:   {len(g.edges)}")
    print(f"   冲突对: {g.find_conflict_pairs()}")
    print(f"   拓扑序: {g.topological_order()}")

    # ── 注册 reviewer 到 workspace ──
    ws = ReviewWorkspace()
    ws.subscribe("ai-reviewer", AIReviewer(AI_REVIEWER))
    ws.subscribe("cog-scientist", CognitiveScientist(COG_SCIENTIST))
    ws.subscribe("control-scientist", ControlScientist(CONTROL_SCIENTIST))

    # ── 模拟文档 ──
    document = (
        "海波（Haibo）是一个意图驱动的认知架构原型。核心宣称：'LLM是嘴巴不是大脑'。"
        "系统由意图云图（36节点42边）、激活扩散、双语者注入（input embedding层）、"
        "锚点系统和对话上下文管理组成。H1-H19 共 ~101/102 测试通过。"
        "但控制论分析指出：赫布学习与扩散收敛在数学上不兼容（ρ=1.099反例），"
        "注入信号强度不足以覆盖 LLM 预训练（std 差33倍），"
        "且系统缺少 LLM 输出到图状态的反馈回路。"
    )
    print(f"\n📄 被审文档: {document[:60]}...")

    # ── 第一阶段: 独立知识输入 ──
    # Control Scientist starts working immediately (no deps)
    print("\n── 阶段 1: 独立知识输入 ──")
    control_output = ReviewOutput(
        role_id="control-scientist",
        content="[控制论先验知识] 谱半径是扩散收敛的充要条件。"
                "任何赫布学习系统若无列和归一化，在3+节点全连通图上必然发散。",
        confidence=0.95,
        conclusions=["先验知识: 赫布学习与扩散收敛的数学不兼容性是结构性的"],
    )
    print(f"  [control-scientist] 提供了数学先验")

    # ── 第二阶段: 并发审阅 ──
    print("\n── 阶段 2: 并发审阅 (所有专家同时产出) ──")

    # AI Reviewer 需要知道 Control + Cog 的结论
    # 所以先发布 Control + Cog 的结果，再发布完整文档让 AI Reviewer 做综合
    # 这里简化：所有专家同时收到完整文档 + 上游引用
    frame = {
        "document": document,
        "previous_reviews": ["control-scientist", "cog-scientist"],
        "control_math_proof": "3节点反例 ρ(A)=1.099 > 1",
        "cog_analysis": "L1 反馈存在, L2/L3 缺失",
    }

    results = await ws.publish(frame)

    print(f"\n  收到 {len(results)} 份审稿:")
    for rid, rev in results.items():
        print(f"\n  📋 [{rid}] confidence={rev.confidence}")
        for c in rev.conclusions:
            print(f"      → {c[:60]}...")
        if rev.evidence:
            print(f"      📎 证据: {rev.evidence[0][:50]}...")

    # ── 第三阶段: PI 聚合 ──
    print("\n── 阶段 3: PI 聚合 & 冲突检测 ──")
    aggregator = DecisionAggregator(g)

    # 模拟 Code Scientist 和 AI Reviewer 对"反馈重要性"有分歧
    # （正常分歧，不是恶意内讧）
    conflicts = await aggregator.aggregate(results, document)
    print(f"  发现 {len(conflicts)} 个冲突:")
    for c in conflicts:
        print(f"    ⚡ [{c.role_a} vs {c.role_b}] 话题: {c.topic}")
        print(f"       A: {c.conclusion_a[:40]}...")
        print(f"       B: {c.conclusion_b[:40]}...")
        print(f"       严重度: {c.severity}")

    # 仲裁冲突
    resolved = aggregator.resolve_conflicts(conflicts, results)
    print(f"\n  仲裁结果:")
    for c in resolved:
        print(f"    ✅ {c.resolution[:80]}...")

    # ── 第四阶段: PI 决策 ──
    print("\n── 阶段 4: PI 决策 ──")
    decision = aggregator.make_decision(
        title="海波项目方向裁定",
        content=(
            "综合三方审查：\n"
            "1. Control Scientist 构造了 3 节点全连通图的反例，"
            "证明赫布学习与扩散收敛不兼容（数学证伪）。\n"
            "2. Cognitive Scientist 确认系统缺少 L2/L3 反馈，"
            "不构成闭环认知系统。\n"
            "3. AI Reviewer 确认核心模块有代码实现，"
            "但增量贡献不明确。\n\n"
            "结论：核心假设 H1 被证伪。项目转向——"
            "终止'图结构控制LLM生成'路径，现有成果打包为 Hermes Agent 技能。\n"
            "未来方向：如果继续认知架构方向，需更换控制接口（LoRA/中间层注入）。"
        ),
        reviews=results,
        conflicts=resolved,
        path="C",
    )
    decision.status = "executed"

    print(f"\n  📌 决策: {decision.id}")
    print(f"     标题: {decision.title}")
    print(f"     路径: {decision.path}")
    print(f"     状态: {decision.status}")
    print(f"     基于 {len(decision.based_on)} 个角色:")
    for rid, conclusions in decision.based_on.items():
        print(f"       [{rid}] ({len(conclusions)} 条结论)")

    # ── 第五阶段: 反馈回图 ──
    print("\n── 阶段 5: 决策反馈回图 ──")
    # PI's decision feeds back, changing edge weights
    # 被采纳的专家角色得到权重提升
    for rid in results:
        if rid in decision.based_on:
            g.roles[rid].weight += 0.05
    print(f"  受影响角色权重已更新:")
    for rid, role in g.roles.items():
        print(f"    {rid}: weight={role.weight:.2f}")

    # ── 输出完整图结构 ──
    print("\n" + "=" * 60)
    print("  最终专家图结构")
    print("=" * 60)
    print(json.dumps(g.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
