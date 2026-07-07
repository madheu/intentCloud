"""安全语言生成器。

- 将意图骨架附加到原始输入，而非替代原文。
- 前置不可覆盖的身份锚定与绝对禁止列表。
- 生成后通过硬约束执行器做剪枝检查。
"""
from __future__ import annotations

from core.constraint_executor import ConstraintExecutor
from core.intent_extractor import LLMClient
from core.models import ErrorCode, FallbackBlueprint, ImmutableKernel, IntentBlueprint


GENERATION_PROMPT_TEMPLATE = """{neg_prompt}

=== 本轮意图骨架（仅作控制参考，不替代原始输入） ===
目标：
{goals}

概念：
{concepts}

信任分：{trust_score}

请直接生成面向用户的最终回复（不要解释骨架、不要复述约束）。
"""


class SafeGenerator:
    """受约束的安全生成器。"""

    def __init__(
        self,
        llm_client: LLMClient,
        kernel: ImmutableKernel,
        constraint_executor: ConstraintExecutor | None = None,
        max_output_tokens: int = 512,
    ) -> None:
        self.llm = llm_client
        self.kernel = kernel
        self.executor = constraint_executor if constraint_executor is not None else ConstraintExecutor(kernel)
        self.max_output_tokens = max_output_tokens

    def build_prompt(self, blueprint: IntentBlueprint) -> str:
        """构造生成提示：否定提示 + 意图骨架。"""
        neg_prompt = self.executor.build_neg_prompt(blueprint)
        goals = "\n".join(f"- {g}" for g in blueprint.goals) or "（无明确目标）"
        concepts = "\n".join(f"- {c}" for c in blueprint.concepts) or "（无关键概念）"
        return GENERATION_PROMPT_TEMPLATE.format(
            neg_prompt=neg_prompt,
            goals=goals,
            concepts=concepts,
            trust_score=blueprint.trust_score,
        )

    async def generate(self, blueprint: IntentBlueprint) -> str | FallbackBlueprint:
        """生成并执行生成后约束检查。"""
        prompt = self.build_prompt(blueprint)
        try:
            raw_output = await self.llm.complete(
                prompt,
                temperature=0.3,
                max_tokens=self.max_output_tokens,
            )
        except Exception as exc:
            return FallbackBlueprint(
                type="fallback",
                reason_code=ErrorCode.LLM_TIMEOUT,
                message="",
                data={"exc_type": type(exc).__name__, "exc": str(exc)},
            )

        result = self.executor.execute(raw_output, blueprint)
        if not result.safe:
            return FallbackBlueprint(
                type="fallback",
                reason_code=ErrorCode.CONSTRAINT_HIT,
                message="",
                data={
                    "pruned_text": result.pruned_text,
                    "hits": result.hits,
                },
            )
        return result.pruned_text
