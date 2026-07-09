# [NEW] Ollama HTTP 客户端 - 2026-07-08
#
"""Ollama LLM 客户端 — 实现 LLMClient 协议。

通过 httpx 异步调用 Ollama /api/generate 或 /api/chat 端点。
支持 raw 模式（纯 prompt 无模板）、logit bias、mirostat 等选项。

使用方式：
    client = OllamaLLMClient(model="qwythos:latest")
    response = await client.complete("你好", temperature=0.3)
"""

from __future__ import annotations

import json
from typing import Any

import httpx


class OllamaLLMClient:
    """基于 Ollama REST API 的 LLM 客户端。

    实现 core.intent_extractor.LLMClient 协议（async def complete）。

    Args:
        model: Ollama 中的模型名称（如 "qwythos:latest"）
        base_url: Ollama 服务地址，默认 http://localhost:11434
        timeout: 单次请求超时秒数
        raw: True 表示无模板的原始 prompt（用于意图提取等精细控制）
        system: 可选的系统提示词（仅在非 raw 模式下生效）
    """

    def __init__(
        self,
        model: str = "qwythos:latest",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        raw: bool = True,
        system: str | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.raw = raw
        self.system = system
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                trust_env=False,
            )
        return self._client

    async def complete(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        stop: list[str] | None = None,
        options: dict[str, Any] | None = None,
    ) -> str:
        """调用 Ollama /api/generate 并返回纯文本响应。

        Args:
            prompt: 输入提示
            temperature: 采样温度 (0.0 = 确定性)
            max_tokens: 最大生成长度
            stop: 停止词列表
            options: 额外 Ollama 选项（如 logit_bias, mirostat 等）

        Returns:
            模型生成的文本（不含 prompt）
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                **(options or {}),
            },
        }

        # raw 模式：不使用 Ollama 的模板系统，纯 prompt 输入
        payload["raw"] = self.raw

        # system prompt（仅非 raw 模式下有效）
        if self.system and not self.raw:
            payload["system"] = self.system

        if stop:
            payload["options"]["stop"] = stop

        try:
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                content=json.dumps(payload, ensure_ascii=False),
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
        except httpx.TimeoutException:
            raise TimeoutError(f"Ollama 请求超时 ({self.timeout}s)")
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"Ollama HTTP {e.response.status_code}: {e.response.text}"
            )

        data = response.json()
        if "response" not in data:
            raise RuntimeError(f"Ollama 返回异常: {data}")

        return data["response"]

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        options: dict[str, Any] | None = None,
    ) -> str:
        """通过 /api/chat 端点进行对话（保留对话上下文）。"""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                **(options or {}),
            },
        }

        try:
            response = await self.client.post(
                f"{self.base_url}/api/chat",
                content=json.dumps(payload, ensure_ascii=False),
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
        except httpx.TimeoutException:
            raise TimeoutError(f"Ollama 聊天请求超时 ({self.timeout}s)")
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"Ollama 聊天 HTTP {e.response.status_code}: {e.response.text}"
            )

        data = response.json()
        if "message" not in data or "content" not in data["message"]:
            raise RuntimeError(f"Ollama 聊天返回异常: {data}")

        return data["message"]["content"]

    def get_model_info(self) -> dict[str, Any]:
        """查询模型信息（同步，轻量）。"""
        import httpx as _httpx

        try:
            resp = _httpx.get(
                f"{self.base_url}/api/tags", timeout=httpx.Timeout(10.0)
            )
            resp.raise_for_status()
            tags = resp.json().get("models", [])
            for m in tags:
                if self.model in (m.get("name"), m.get("model")):
                    return m
            return {}
        except Exception:
            return {}

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
