"""双语者注入器（Bilingual Injector）。

替代传统的 steering vector 暴力注入方案。每个激活节点自带 LLM 中间层
embedding 坐标，激活时作为虚拟 token prepend 到 input_embeds。

核心思想：
  - 传统方案：选一条边 → 生成 steering_vector → hook 注入中间层
  - 双语者方案：每个激活节点 = 独立虚拟 token → prepend 到 input_embeds
  - 节点 embedding × 激活值作为加权，激活值高的排前面

用法：
    injector = BilingualInjector(model, tokenizer)
    output = injector.generate_with_injection(prompt, cloud, max_new_tokens=200)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BilingualInjector:
    """双语者注入器：将意图云中激活节点的 embedding 作为虚拟 token prepend。"""

    # 默认激活阈值：激活值低于此的节点不参与注入
    DEFAULT_ACTIVATION_THRESHOLD: float = 0.1

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        activation_threshold: float | None = None,
    ) -> None:
        """初始化注入器。

        Args:
            model: HuggingFace transformers 因果语言模型
            tokenizer: 对应的 tokenizer
            activation_threshold: 激活阈值，低于此值的节点不参与注入。
                默认 0.1，设为 0 表示所有非零激活节点都参与。
        """
        self.model = model
        self.tokenizer = tokenizer
        self.activation_threshold = (
            activation_threshold
            if activation_threshold is not None
            else self.DEFAULT_ACTIVATION_THRESHOLD
        )

    def inject(
        self,
        activations: dict[str, float],
        cloud,
    ) -> torch.Tensor | None:
        """把激活节点的 embedding 组装为虚拟 token 序列。

        流程：
          1. 从 cloud 中取出所有激活值 > 阈值的节点
          2. 取出每个节点的 llm_embedding
          3. 每个 embedding 乘以激活值作为权重
          4. 按激活值从高到低排序
          5. 返回加权后的 embedding 序列

        Args:
            activations: 节点 ID → 激活值的映射（来自 process_interaction 的输出）
            cloud: IntentCloud 实例（用于获取节点和 llm_embedding）

        Returns:
            shape [num_activated, hidden_dim] 的虚拟 token embedding 序列，
            如果无激活节点则返回 None
        """
        # 收集激活节点的 (激活值, embedding) 对
        activated: list[tuple[float, torch.Tensor]] = []

        for node_id, activation in activations.items():
            # 过滤低于阈值的节点
            if activation < self.activation_threshold:
                continue

            # 获取节点
            node = cloud._shell.get(node_id)
            if node is None:
                continue

            # 获取 llm_embedding
            embedding = node.llm_embedding
            if embedding is None:
                continue

            # 转换为 tensor（兼容 numpy 和 list 输入）
            if not isinstance(embedding, torch.Tensor):
                embedding = torch.tensor(embedding)

            activated.append((activation, embedding))

        if not activated:
            return None

        # 按激活值降序排序
        activated.sort(key=lambda x: x[0], reverse=True)

        device = self._get_device()

        # 计算输入 embedding 的典型尺度，用于缩放虚拟 token
        embed_layer = self.model.get_input_embeddings()
        # 取一批随机 token 的 embedding 估算 std
        dummy_ids = torch.randint(0, embed_layer.num_embeddings, (100,), device=device)
        ref_embeds = embed_layer(dummy_ids)  # [100, hidden_dim]
        ref_std = ref_embeds.std().item()

        # 组装加权 embedding 序列
        virtual_embeds = []
        model_dtype = embed_layer.weight.dtype
        VIRTUAL_REPEAT = 1  # 每个节点重复次数
        for act, emb in activated:
            emb = emb.to(device=device, dtype=model_dtype)
            # 支持单向量 [hidden_dim] 和 多 token [num_tokens, hidden_dim] 两种格式
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)  # → [1, hidden_dim]
            # 缩放虚拟 embedding 到输入 embedding 的尺度
            num_tokens = emb.shape[0]
            emb_std = emb.std().item()
            if emb_std > 0:
                emb = emb * (ref_std / emb_std)
            weighted = emb * act
            # 重复虚拟 token 以增强信号，对抗 RLHF 身份固化
            for _ in range(VIRTUAL_REPEAT):
                virtual_embeds.append(weighted)

        # shape: [total_tokens, hidden_dim]
        return torch.cat(virtual_embeds, dim=0)

    def generate_with_injection(
        self,
        prompt: str,
        activations: dict[str, float],
        cloud,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **generate_kwargs,
    ) -> str:
        """带双语者注入的文本生成。

        流程：
          1. tokenizer prompt → input_ids
          2. 获取 input_embeds = model.get_input_embeddings()(input_ids)
          3. inject() → virtual_embeds [num_activated, hidden_dim]
          4. 拼接: combined = cat([virtual_embeds, input_embeds], dim=1)
          5. 构造 attention_mask
          6. model.generate(inputs_embeds=combined, ...)
          7. decode 输出（跳过虚拟 token 位置）

        Args:
            prompt: 用户输入文本
            activations: 节点 ID → 激活值的映射
            cloud: IntentCloud 实例
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_p: nucleus sampling 阈值
            **generate_kwargs: 传递给 model.generate() 的其他参数

        Returns:
            生成的文本（不含虚拟 token 部分）
        """
        device = self._get_device()

        # 步骤 1: tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        seq_len = input_ids.shape[1]

        # 步骤 2: 获取 input embeddings
        embed_layer = self.model.get_input_embeddings()
        input_embeds = embed_layer(input_ids)  # [1, seq_len, hidden_dim]

        # 步骤 3: 获取虚拟 token embeddings
        virtual_embeds = self.inject(activations, cloud)

        if virtual_embeds is None:
            # 无激活节点，直接生成（回退路径：跳过 input tokens）
            outputs = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                **generate_kwargs,
            )
            # 跳过 input 部分，只返回新生成的内容
            new_tokens = outputs[0][seq_len:]
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        num_virtual = virtual_embeds.shape[0]

        # 步骤 4: 拼接虚拟 token 和真实输入
        # virtual_embeds: [num_virtual, hidden_dim] → [1, num_virtual, hidden_dim]
        virtual_embeds = virtual_embeds.unsqueeze(0)
        combined_embeds = torch.cat([virtual_embeds, input_embeds], dim=1)
        # combined_embeds: [1, num_virtual + seq_len, hidden_dim]

        # 步骤 5: 构造 attention mask
        total_len = num_virtual + seq_len
        attention_mask = torch.ones((1, total_len), dtype=torch.long, device=device)

        # 步骤 6: 生成
        outputs = self.model.generate(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            **generate_kwargs,
        )

        # 步骤 7: decode 输出
        # 虚拟 token 不对应真实词，需要跳过前 num_virtual 个位置
        # 但 generate 返回的是完整的 [1, num_virtual + seq_len + new_tokens]
        full_output = outputs[0]
        # 跳过虚拟 token 部分，只取真实输入 + 生成部分
        real_output = full_output[num_virtual + seq_len:]
        if real_output.numel() == 0:
            # 如果没有生成新 token，返回空
            return ""

        return self.tokenizer.decode(real_output, skip_special_tokens=True)

    def _get_device(self) -> torch.device:
        """安全获取模型所在设备，兼容空参数列表的 mock 模型。"""
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def get_status(self) -> dict:
        """返回当前注入器状态。"""
        return {
            "activation_threshold": self.activation_threshold,
            "model_type": type(self.model).__name__,
        }