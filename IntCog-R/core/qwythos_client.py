# [NEW] Qwythos GGUF 直推客户端 (llama-cpp-python) - 2026-07-08
#
"""Qwythos LLM 客户端 — 基于 llama-cpp-python 直接加载 GGUF。

实现 core.intent_extractor.LLMClient 协议（async def complete）。
支持 logits_processor 实现 Steering 效果。
"""

from __future__ import annotations

import asyncio
from functools import partial
from typing import Any, Callable

from llama_cpp import Llama, LogitsProcessorList, LogitsProcessor


class QwythosLLMClient:
    """基于 llama-cpp-python 的 LLM 客户端，直接加载 GGUF 模型。

    实现 LLMClient 协议（async def complete）。

    Args:
        model_path: GGUF 模型文件路径
        n_ctx: 上下文窗口大小
        n_threads: CPU 线程数
        verbose: 是否显示加载详情
    """

    def __init__(
        self,
        model_path: str = r"E:\.Ollama\models\Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf",
        n_ctx: int = 2048,
        n_threads: int = 4,
        verbose: bool = False,
    ) -> None:
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self._llm: Llama | None = None
        self._verbose = verbose

    @property
    def llm(self) -> Llama:
        if self._llm is None:
            if self._verbose:
                print(f"[QwythosLLM] 加载模型: {self.model_path}")
            self._llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=0,
                verbose=False,
            )
            if self._verbose:
                print(f"[QwythosLLM] 加载完成 (ctx={self.n_ctx})")
        return self._llm

    async def complete(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        stop: list[str] | None = None,
        logits_processors: LogitsProcessorList | None = None,
    ) -> str:
        """调用 llama.cpp 推理。

        Args:
            prompt: 输入提示
            temperature: 采样温度
            max_tokens: 最大生成长度
            stop: 停止词列表
            logits_processors: LogitsProcessor 列表（用于 Steering）

        Returns:
            模型生成的文本
        """
        # 在子线程中同步调用，避免阻塞 asyncio 事件循环
        loop = asyncio.get_event_loop()
        output = await loop.run_in_executor(
            None,
            partial(
                self.llm,
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop or [],
                echo=False,
                logits_processor=logits_processors,
            ),
        )

        text = output["choices"][0]["text"].strip()
        return text

    def complete_sync(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        stop: list[str] | None = None,
        logits_processors: LogitsProcessorList | None = None,
    ) -> str:
        """同步版本，用于无需异步的场景。"""
        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop or [],
            echo=False,
            logits_processor=logits_processors,
        )
        return output["choices"][0]["text"].strip()

    def close(self) -> None:
        self._llm = None


# ── Steering LogitsProcessor ─────────────────────────────────────────────────

class SteeringLogitsProcessor(LogitsProcessor):
    """基于蓝图语义的 Steering LogitsProcessor。

    在每步生成时修改 logits，将生成方向拉向蓝图意图。
    虽然不是 hidden state 注入，但在 9B 模型上仍能产生可观测的引导效果。

    Args:
        boost_tokens: {token_id: bias} — 鼓励生成的 token 及其偏置
        suppress_tokens: {token_id: bias} — 抑制生成的 token 及其偏置
    """

    def __init__(
        self,
        boost_tokens: dict[int, float] | None = None,
        suppress_tokens: dict[int, float] | None = None,
    ) -> None:
        self.boost = boost_tokens or {}
        self.suppress = suppress_tokens or {}

    def __call__(self, input_ids: list[int], logits: list[float]) -> list[float]:
        """修改 logits 以体现引导方向。"""
        for token_id, bias in self.boost.items():
            if 0 <= token_id < len(logits):
                logits[token_id] += bias
        for token_id, bias in self.suppress.items():
            if 0 <= token_id < len(logits):
                logits[token_id] -= bias
        return logits


def logits_processors_from_blueprint(
    llm: Llama,
    blueprint_dict: dict,
    strength: float = 1.0,
) -> LogitsProcessorList:
    """从蓝图生成 LogitsProcessor 列表。

    根据 core_task 和 concepts 中的关键词，找到对应的 token ID，
    给予正偏置；根据 constraints 中的关键词，给予负偏置。

    这是简化版——真正的语义级 steering 需要 embedding 相似度搜索。
    """
    boost_tokens: dict[int, float] = {}
    suppress_tokens: dict[int, float] = {}

    # 从蓝图提取关键词
    core_task = blueprint_dict.get("core_task", "")
    concepts = blueprint_dict.get("concepts", [])
    constraints = blueprint_dict.get("constraints", [])

    # 将关键词分词并找到对应 token ID
    all_text = " ".join([core_task] + concepts + constraints)
    encoded = llm.tokenize(all_text.encode("utf-8"), add_bos=False)

    boost_keywords = [core_task] + concepts
    suppress_keywords = constraints

    # 根据关键词重要性分配偏置
    boost_bias = strength * 0.5
    suppress_bias = strength * 0.5

    # TODO: 更精确的 keyword→token_id 映射
    # 当前简化实现：将整段蓝图文本的所有 token 按比例偏置
    for i, tid in enumerate(encoded):
        # 前 1/3 token 偏正例（概念），后 1/3 偏负例（约束）
        pos = i / max(len(encoded), 1)
        if pos < 0.33 and boost_keywords:
            boost_tokens[tid] = boost_tokens.get(tid, 0) + boost_bias
        elif pos > 0.66 and suppress_keywords:
            suppress_tokens[tid] = suppress_tokens.get(tid, 0) + suppress_bias

    processors = LogitsProcessorList()
    if boost_tokens or suppress_tokens:
        processors.append(SteeringLogitsProcessor(boost_tokens, suppress_tokens))

    return processors
