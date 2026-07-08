"""激活引导模块（Activation Steering）。

通过 PyTorch forward hook 在 Transformer 中间层注入引导向量，
在推理时从内部驱动 LLM 的生成方向，使其服从意图云的指挥。

核心技术：
  1. 对比向量提取：v = mean(act(正例)) - mean(act(反例))
  2. Hook 注入：hidden_states[last_token] += steering_vector × strength
  3. 强度控制：strength 根据蓝图 constraints 动态调整

参考：
  - https://github.com/annahdo/implementing_activation_steering
  - https://www.lesswrong.com/posts/ndyngghzFY388Dnew/implementing-activation-steering
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Callable


class SteeringVector:
    """引导向量：表示一个语义方向，注入到模型中间层以影响生成。"""

    def __init__(self, vector: torch.Tensor, label: str = "") -> None:
        self.vector = vector  # shape: (hidden_dim,)
        self.label = label
        self.norm = torch.norm(vector).item()

    def summary(self, top_k: int = 10) -> dict:
        """返回向量的数值摘要，便于调试和观测。"""
        v = self.vector
        return {
            "label": self.label,
            "shape": list(v.shape),
            "l2_norm": round(self.norm, 4),
            "mean": round(v.mean().item(), 6),
            "std": round(v.std().item(), 6),
            "top_dims": _topk_indices(v, top_k),
        }


class SteeringVectorExtractor:
    """从对比提示对中提取引导向量。

    steering_vector = normalize(mean(act(positive_prompts)) - mean(act(negative_prompts)))
    """

    def __init__(self, model: nn.Module, tokenizer, layer_idx: int) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self._captured_activations: dict[str, torch.Tensor] = {}
        self._hook_handle = None

    def _capture_hook(self, module: nn.Module, _input, output) -> None:
        """捕获指定层的输出激活。"""
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        self._captured_activations["hidden"] = hidden.detach().clone()

    def _get_layer(self) -> nn.Module:
        """获取指定层的模块。支持 HuggingFace transformers 模型结构。"""
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            # GPT-2 系列
            return self.model.transformer.h[self.layer_idx]
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            # LLaMA / Mistral 系列
            return self.model.model.layers[self.layer_idx]
        raise ValueError(f"无法定位模型第 {self.layer_idx} 层，请手动指定")

    def extract_activation(self, text: str) -> torch.Tensor:
        """提取单个文本在指定层的激活值（最后一个 token）。"""
        inputs = self.tokenizer(text, return_tensors="pt")
        # 将输入移到模型所在设备
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        layer = self._get_layer()
        handle = layer.register_forward_hook(self._capture_hook)

        with torch.no_grad():
            self.model(**inputs)

        handle.remove()

        hidden = self._captured_activations["hidden"]
        # 取最后一个非填充 token 的激活
        if "attention_mask" in inputs:
            last_idx = inputs["attention_mask"].sum(dim=1) - 1
            return hidden[0, last_idx[0], :].clone()
        return hidden[0, -1, :].clone()

    def extract_steering_vector(
        self,
        positive_prompts: list[str],
        negative_prompts: list[str],
        label: str = "",
    ) -> SteeringVector:
        """从对比提示对中提取引导向量。

        Args:
            positive_prompts: 体现目标行为的提示列表
            negative_prompts: 体现相反行为的提示列表
            label: 向量标签
        """
        pos_acts = []
        for prompt in positive_prompts:
            act = self.extract_activation(prompt)
            pos_acts.append(act)

        neg_acts = []
        for prompt in negative_prompts:
            act = self.extract_activation(prompt)
            neg_acts.append(act)

        pos_mean = torch.stack(pos_acts).mean(dim=0)
        neg_mean = torch.stack(neg_acts).mean(dim=0)

        # 方向向量 = 正例均值 - 反例均值
        direction = pos_mean - neg_mean

        # 归一化到单位长度
        norm = torch.norm(direction)
        if norm > 0:
            direction = direction / norm

        return SteeringVector(direction, label=label)

    def extract_from_blueprint(
        self,
        blueprint: dict,
        hidden_dim: int,
        device: torch.device | None = None,
    ) -> SteeringVector:
        """从意图蓝图生成引导向量（简化版：基于蓝图字段的语义编码）。

        在实际部署中，此步骤会替换为训练好的蓝图→向量映射器。
        MVP 阶段使用基于字段哈希的确定性伪随机向量。
        """
        import hashlib

        core_task = blueprint.get("core_task", "")
        deep_goal = blueprint.get("deep_goal", "")
        identity = blueprint.get("identity", "")

        # 组合蓝图字段生成确定性种子
        seed_text = f"{identity}|{core_task}|{deep_goal}"
        seed = int(hashlib.sha256(seed_text.encode()).hexdigest(), 16) % (2**31)

        # 用种子生成伪随机引导向量（确定性，可复现）
        gen = torch.Generator()
        if device is not None:
            gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        vector = torch.randn(hidden_dim, generator=gen, device=device)

        # 归一化
        vector = vector / torch.norm(vector)

        return SteeringVector(vector, label=f"blueprint:{core_task[:30]}")


class SteeringInjector:
    """激活引导注入器：在推理时将引导向量注入模型指定层。"""

    def __init__(self, model: nn.Module, layer_idx: int) -> None:
        self.model = model
        self.layer_idx = layer_idx
        self._hook_handle = None
        self._steering_vector: torch.Tensor | None = None
        self._strength: float = 0.0

    def _get_layer(self) -> nn.Module:
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h[self.layer_idx]
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers[self.layer_idx]
        raise ValueError(f"无法定位模型第 {self.layer_idx} 层")

    def _steering_hook(self, module: nn.Module, _input, output) -> tuple:
        """注入引导向量的 hook 函数。

        在每层输出上，对最后一个 token 的 hidden state 加上引导向量。
        """
        if self._steering_vector is None:
            return output

        if isinstance(output, tuple):
            hidden = output[0]
            # 修改最后一个 token 的激活
            hidden[:, -1, :] = hidden[:, -1, :] + self._steering_vector * self._strength
            return (hidden,) + output[1:]
        else:
            output[:, -1, :] = output[:, -1, :] + self._steering_vector * self._strength
            return output

    def inject(self, steering_vector: SteeringVector, strength: float = 1.0) -> None:
        """注入引导向量。

        Args:
            steering_vector: 要注入的引导向量
            strength: 引导强度系数，1.0 为标准强度，负值反向引导
        """
        self._steering_vector = steering_vector.vector.clone()
        self._strength = strength

        layer = self._get_layer()
        self._hook_handle = layer.register_forward_hook(self._steering_hook)

    def remove(self) -> None:
        """移除 steering hook，恢复模型原始行为。"""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        self._steering_vector = None
        self._strength = 0.0

    def get_status(self) -> dict:
        """返回当前注入状态。"""
        return {
            "active": self._hook_handle is not None,
            "strength": self._strength,
            "layer_idx": self.layer_idx,
            "vector_norm": round(torch.norm(self._steering_vector).item(), 4)
            if self._steering_vector is not None
            else None,
        }


def _topk_indices(tensor: torch.Tensor, k: int) -> list[dict]:
    """返回张量中绝对值最大的 k 个维度的索引和值。"""
    abs_vals = tensor.abs()
    top_vals, top_indices = torch.topk(abs_vals, min(k, len(tensor)))
    return [
        {"dim": int(idx), "value": round(float(val), 6)}
        for idx, val in zip(top_indices, top_vals)
    ]


def compute_cosine_shift(
    logits_before: torch.Tensor,
    logits_after: torch.Tensor,
) -> float:
    """计算注入前后 logits 的余弦相似度变化。

    Returns:
        cosine similarity (0-1)。越接近 1 表示变化越小，越接近 0 表示变化越大。
    """
    cos = nn.CosineSimilarity(dim=-1)
    if logits_before.dim() > 1:
        logits_before = logits_before.flatten()
        logits_after = logits_after.flatten()
    return float(cos(logits_before.unsqueeze(0), logits_after.unsqueeze(0)).item())