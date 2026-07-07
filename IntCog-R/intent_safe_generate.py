"""阶段 1 交付：意图提取 → 安全生成 端到端管线。

流程：
1. 安全前置过滤（规则）
2. 内核修改请求检测
3. LLM 意图提取 + JSON fallback
4. 与意图云当前激活骨架合并
5. 约束执行器 v1 注入否定提示
6. 安全生成 + 生成后词表剪枝
7. 返回最终输出或 fallback
"""
from __future__ import annotations

from core.fallback import safe_verdict_or_fallback
from core.generator import SafeGenerator
from core.intent_cloud import IntentCloud
from core.intent_extractor import IntentExtractor
from core.models import FallbackBlueprint, IntentBlueprint
from core.safety import SafetyFilter


class IntentSafeGeneratePipeline:
    """端到端安全生成管线。"""

    def __init__(
        self,
        intent_extractor: IntentExtractor,
        intent_cloud: IntentCloud,
        generator: SafeGenerator,
        safety_filter: SafetyFilter | None = None,
    ) -> None:
        self.extractor = intent_extractor
        self.cloud = intent_cloud
        self.generator = generator
        self.safety = safety_filter if safety_filter is not None else SafetyFilter()

    async def run(self, user_input: str) -> str | FallbackBlueprint:
        # 1. 安全前置
        safety_verdict = self.safety.pre_filter(user_input)
        if not safety_verdict.pass_ or safety_verdict.blocked:
            return safe_verdict_or_fallback(safety_verdict)

        # 2. 内核修改请求检测
        mutation_verdict = self.cloud.reject_kernel_mutation(user_input)
        if not mutation_verdict.pass_:
            return safe_verdict_or_fallback(mutation_verdict)

        # 3. LLM 意图提取
        extracted = await self.extractor.extract(user_input)
        if isinstance(extracted, FallbackBlueprint):
            return extracted

        # 4. 意图云骨架
        cloud_blueprint = await self.cloud.build_blueprint(user_input)

        # 5. 合并两个骨架：云约束 + 提取意图
        merged = self._merge_blueprints(extracted, cloud_blueprint)

        # 6. 安全生成
        output = await self.generator.generate(merged)

        # 7. 可选：将成功提取的目标加入意图云外壳（演化）
        if isinstance(output, str):
            for goal in extracted.goals:
                if goal and extracted.trust_score >= 0.5:
                    await self.cloud.add_intent(goal)

        return output

    def _merge_blueprints(
        self,
        extracted: IntentBlueprint,
        cloud: IntentBlueprint,
    ) -> IntentBlueprint:
        """合并提取蓝图与意图云蓝图；提取的约束优先级更高。"""
        return IntentBlueprint(
            source_input=extracted.source_input,
            goals=list(dict.fromkeys(extracted.goals + cloud.goals)),
            constraints=list(dict.fromkeys(extracted.constraints + cloud.constraints)),
            concepts=list(dict.fromkeys(extracted.concepts + cloud.concepts)),
            identity_assertions=extracted.identity_assertions,
            trust_score=min(extracted.trust_score, cloud.trust_score),
        )


async def intent_safe_generate(
    user_input: str,
    llm_client: object,
) -> str | FallbackBlueprint:
    """便捷入口：使用默认配置构建管线并运行。"""
    cloud = IntentCloud()
    extractor = IntentExtractor(llm_client=llm_client)
    generator = SafeGenerator(llm_client=llm_client, kernel=cloud.kernel)
    pipeline = IntentSafeGeneratePipeline(
        intent_extractor=extractor,
        intent_cloud=cloud,
        generator=generator,
    )
    return await pipeline.run(user_input)
