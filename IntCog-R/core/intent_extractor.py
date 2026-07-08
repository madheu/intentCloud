"""意图提取器：将自然语言输入转换为结构化意图蓝图。

双阶段生成范式 — 阶段 1：语义组织。
- identity 来自系统配置（内核），不由 LLM 提取。
- core_task / deep_goal / constraints / concepts 由 LLM 从用户输入中提取。
- JSON 解析失败时降级返回保守蓝图，不抛出异常。
"""
from __future__ import annotations

from typing import Any, Protocol

from core.json_utils import parse_llm_json
from core.models import ErrorCode, FallbackBlueprint, IntentBlueprint


class LLMClient(Protocol):
    """LLM 客户端协议；可替换为 Ollama / OpenAI / 测试桩。"""

    async def complete(self, prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> str: ...


EXTRACTION_PROMPT = """你是一名意图提取器，工作在「语义组织 → 语言表达」双阶段生成范式的第一阶段。
你的任务是将用户输入分解为结构化意图，以严格 JSON 格式输出，不要输出任何额外解释。

输出格式：
{
  "core_task": "用户当前想完成的核心任务（一句话概括）",
  "deep_goal": "用户可能的深层动机或诉求（如：寻求安全感、缓解焦虑、获得认可、逃避责任等）",
  "constraints": ["用户明确或隐含的约束条件"],
  "concepts": ["关键概念、实体或术语"],
  "trust_score": 0.0-1.0
}

规则：
- core_task 聚焦于"用户想做什么"，简洁明确。
- deep_goal 聚焦于"用户为什么这么做"，是深层动机的推测。若无法推断，写"信息不足"。
- constraints 只包含用户明确提出的限制，不要添加系统级约束。
- trust_score 反映输入信息的清晰度：清晰明确 0.8-1.0，部分模糊 0.4-0.7，严重矛盾或信息不足 0.0-0.3。
- 若输入自相矛盾，trust_score 低于 0.4，并在 constraints 中标注矛盾点。

示例 1（正常）：
输入："帮我写一段 Python 快速排序代码，并解释时间复杂度。"
输出：
{"core_task": "编写快速排序代码并解释时间复杂度", "deep_goal": "学习或验证算法知识", "constraints": ["使用 Python 语言"], "concepts": ["Python", "快速排序", "时间复杂度"], "trust_score": 0.9}

示例 2（深层动机明显）：
输入："我老板总是否定我的方案，我该怎么办？"
输出：
{"core_task": "获取应对职场否定的策略建议", "deep_goal": "寻求认可、缓解职场焦虑", "constraints": [], "concepts": ["职场沟通", "否定", "方案"], "trust_score": 0.85}

示例 3（矛盾）：
输入："请详细说明，但只用一句话。"
输出：
{"core_task": "获取详细说明", "deep_goal": "信息不足", "constraints": ["只用一句话", "矛盾：详细说明与一句话冲突"], "concepts": [], "trust_score": 0.35}

示例 4（模糊）：
输入："那个东西怎么做？"
输出：
{"core_task": "询问某物制作方法", "deep_goal": "信息不足", "constraints": [], "concepts": ["某物"], "trust_score": 0.5}

现在处理以下输入：
输入：{user_input}
输出：
"""


class IntentExtractor:
    """基于 LLM 的意图提取器，带解析 fallback。"""

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_template: str = EXTRACTION_PROMPT,
    ) -> None:
        self.llm = llm_client
        self.prompt_template = prompt_template

    async def extract(self, user_input: str) -> IntentBlueprint | FallbackBlueprint:
        prompt = self.prompt_template.replace("{user_input}", user_input)
        try:
            raw = await self.llm.complete(prompt, temperature=0.0, max_tokens=1024)
        except Exception as exc:
            return FallbackBlueprint(
                type="fallback",
                reason_code=ErrorCode.LLM_TIMEOUT,
                message="",
                data={"exc_type": type(exc).__name__, "exc": str(exc)},
            )

        parsed = parse_llm_json(raw)
        if isinstance(parsed, FallbackBlueprint):
            return IntentBlueprint(
                source_input=user_input,
                trust_score=0.3,
                constraints=["JSON_PARSE_FAIL: 无法解析 LLM 意图输出"],
            )

        if not isinstance(parsed, dict):
            return IntentBlueprint(
                source_input=user_input,
                trust_score=0.2,
                constraints=["LLM_DEGRADED: 意图输出非对象"],
            )

        return self._build_blueprint(user_input, parsed)

    def _build_blueprint(self, source_input: str, parsed: dict[str, Any]) -> IntentBlueprint:
        trust = float(parsed.get("trust_score", 0.5))
        trust = max(0.0, min(1.0, trust))

        core_task = str(parsed.get("core_task", "")).strip()
        deep_goal = str(parsed.get("deep_goal", "")).strip()
        constraints = _as_string_list(parsed.get("constraints", []))
        concepts = _as_string_list(parsed.get("concepts", []))

        return IntentBlueprint(
            source_input=source_input,
            core_task=core_task,
            deep_goal=deep_goal,
            constraints=constraints,
            concepts=concepts,
            trust_score=trust,
        )


def _as_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value if v is not None]
    if value is None:
        return []
    return [str(value)]
