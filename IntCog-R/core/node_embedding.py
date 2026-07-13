"""节点 embedding 查询。

双语者方案核心：每个概念节点在 LLM 中间层有自己的 hidden state 坐标。
激活时，这些坐标作为虚拟 token prepend 到 input_embeds，替代传统的 steering vector 暴力注入。

用法：
    embedding = get_node_embedding("大海", model, tokenizer, layer_idx)
    node.llm_embedding = embedding
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import ModuleList


def get_node_embedding(
    text: str,
    model: nn.Module,
    tokenizer,
    layer_idx: int,
) -> torch.Tensor:
    """查询一个概念节点在 LLM 中间层的 embedding 坐标。

    流程：
      1. 把概念文本 tokenize
      2. 前向传播到 layer_idx 层
      3. 取该层最后一个 token 的 hidden state
      4. 返回 shape [hidden_dim] 的向量

    Args:
        text: 概念文本（如 "大海"、"星空"、"诗歌"）
        model: HuggingFace transformers 模型
        tokenizer: 对应的 tokenizer
        layer_idx: 目标层索引（建议用 model.config.num_hidden_layers // 2）

    Returns:
        shape [hidden_dim] 的 hidden state 向量，位于模型所在设备

    Raises:
        ValueError: 如果 text 为空或 layer_idx 越界
        RuntimeError: 如果模型前向传播失败
    """
    if not text or not text.strip():
        raise ValueError("text must not be empty")

    # 确定 hidden_dim 和 layer 范围
    hidden_dim = _get_hidden_dim(model)
    num_layers = _get_num_layers(model)
    if layer_idx < 0 or layer_idx >= num_layers:
        raise ValueError(
            f"layer_idx {layer_idx} out of range [0, {num_layers - 1}]"
        )

    device = next(model.parameters()).device

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 获取目标层
    layer = _get_layer(model, layer_idx)

    # 用 hook 捕获该层的 hidden state
    captured: dict[str, torch.Tensor] = {}

    def _hook(module: nn.Module, _input, output) -> None:
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        captured["hidden"] = hidden.detach().clone()

    handle = layer.register_forward_hook(_hook)

    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        handle.remove()

    hidden = captured["hidden"]
    # 取最后一个非填充 token 的 hidden state
    if "attention_mask" in inputs:
        last_idx = inputs["attention_mask"].sum(dim=1) - 1
        return hidden[0, last_idx[0], :].clone()
    return hidden[0, -1, :].clone()


def _get_hidden_dim(model: nn.Module) -> int:
    """获取模型的 hidden dimension。"""
    if hasattr(model.config, "hidden_size"):
        return model.config.hidden_size
    if hasattr(model.config, "d_model"):
        return model.config.d_model
    if hasattr(model.config, "n_embd"):
        return model.config.n_embd
    raise ValueError("Cannot determine hidden_dim from model config")


def _get_num_layers(model: nn.Module) -> int:
    """获取模型的层数。"""
    if hasattr(model.config, "num_hidden_layers"):
        return model.config.num_hidden_layers
    if hasattr(model.config, "n_layer"):
        return model.config.n_layer
    raise ValueError("Cannot determine num_layers from model config")


def _get_layer(model: nn.Module, layer_idx: int) -> nn.Module:
    """获取指定索引的 Transformer 层。支持主流模型结构。"""
    # LLaMA / Mistral / Qwen 系列
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    # GPT-2 系列
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]
    # OPT 系列
    if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "layers"):
        return model.model.decoder.layers[layer_idx]
    # TinyTransformer（自包含模型）
    if hasattr(model, "layers") and isinstance(model.layers, nn.ModuleList):
        return model.layers[layer_idx]
    raise ValueError(
        f"Cannot locate layer {layer_idx} in model of type {type(model).__name__}"
    )